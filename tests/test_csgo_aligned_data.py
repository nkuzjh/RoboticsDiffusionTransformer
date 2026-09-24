"""Focused CPU checks for aligned Seen-10 data identity and augmentation."""

from __future__ import annotations

import itertools
import json
import pickle
import random
import unittest
from pathlib import Path

import numpy as np
import torch
import imgaug as ia
from PIL import Image
from torch.utils.data import DataLoader

from data.csgo_augmentation import augment_image, validate_augmentation
from data.csgo_seen10 import SEEN_MAPS, Seen10Dataset, collate_seen10
from data.csgo_update_sampler import AlignedUpdateSampler, SampleOccurrence


_CONFIG = {
    "policy": "rdt_native_image_v1",
    "train_only": True,
    "views": ["fpv", "radar"],
    "per_view_probability": 0.5,
    "independent_views": True,
    "geometry": "none",
    "auto_adjust_image_brightness": False,
    "state_noise_snr": None,
    "condition_dropout_probability": 0.0,
    "rng_mode": "sample_occurrence",
    "seed": 42,
}


class _TinyProcessor:
    image_mean = (0.5, 0.5, 0.5)

    def preprocess(self, image, return_tensors):
        pixels = np.asarray(image.resize((16, 16)), dtype=np.uint8).copy()
        return {"pixel_values": torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)}


class AlignedAugmentationTest(unittest.TestCase):
    def test_local_rng_is_replayable_and_view_specific(self):
        pixels = np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3)
        image = Image.fromarray(pixels)
        for occurrence in range(20):
            args = dict(config=_CONFIG, epoch=2, global_occurrence=occurrence, sample_id="de_nuke/file_num1_frame_1")
            first, decision = augment_image(image, view="fpv", **args)
            replay, replay_decision = augment_image(image, view="fpv", **args)
            self.assertEqual(decision, replay_decision)
            np.testing.assert_array_equal(np.asarray(first), np.asarray(replay))
        different_view = any(
            augment_image(image, config=_CONFIG, epoch=2, global_occurrence=i, sample_id="sample", view="fpv")[1]
            != augment_image(image, config=_CONFIG, epoch=2, global_occurrence=i, sample_id="sample", view="radar")[1]
            for i in range(20)
        )
        self.assertTrue(different_view)

    def test_geometry_and_unapproved_conditions_are_rejected(self):
        for override in ({"geometry": "flip"}, {"state_noise_snr": 40}, {"condition_dropout_probability": 0.1}):
            with self.assertRaises(ValueError):
                validate_augmentation({**_CONFIG, **override})

    def test_augmentation_does_not_advance_process_rngs(self):
        image = Image.fromarray(np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3))
        kwargs = dict(config=_CONFIG, epoch=1, sample_id="de_nuke/file_num1_frame_1", view="fpv")
        occurrence = next(
            i for i in range(100)
            if augment_image(image, global_occurrence=i, **kwargs)[1].branch == "both"
        )
        python_state = random.getstate()
        numpy_state = pickle.dumps(np.random.get_state())
        torch_state = torch.random.get_rng_state().clone()
        imgaug_state = pickle.dumps(ia.random.get_global_rng().state)
        augment_image(image, global_occurrence=occurrence, **kwargs)
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(pickle.dumps(np.random.get_state()), numpy_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        self.assertEqual(pickle.dumps(ia.random.get_global_rng().state), imgaug_state)


class AlignedSamplerTest(unittest.TestCase):
    def test_exact_19500_step_budget_and_resume(self):
        sampler = AlignedUpdateSampler(50000, 42, 1, 0, 4, 32)
        self.assertEqual(sampler.updates_per_epoch, 390)
        self.assertEqual(sampler.samples_per_epoch, 49920)
        self.assertEqual(len(sampler), 19500 * 32)
        first = list(itertools.islice(iter(sampler), 32))
        self.assertEqual(len({item.index for batch in first for item in batch}), 128)
        self.assertEqual([item.global_occurrence for batch in first for item in batch], list(range(128)))
        sampler.set_start_update(390)
        resumed = next(iter(sampler))
        self.assertEqual(resumed[0].epoch, 1)
        self.assertEqual(resumed[0].global_occurrence, 49920)
        independent = AlignedUpdateSampler(50000, 42, 1, 0, 4, 32, start_update=390)
        self.assertEqual(resumed, next(iter(independent)))
        sampler.set_start_update(19500)
        self.assertEqual(len(sampler), 0)
        self.assertEqual(list(sampler), [])

    def test_rank_shards_cover_one_global_batch(self):
        args = dict(dataset_size=50000, seed=7, world_size=2, microbatch_size=4, gradient_accumulation_steps=16)
        left = list(itertools.islice(iter(AlignedUpdateSampler(rank=0, **args)), 16))
        right = list(itertools.islice(iter(AlignedUpdateSampler(rank=1, **args)), 16))
        all_items = [item for batch in left + right for item in batch]
        self.assertEqual(len(all_items), 128)
        self.assertEqual(len({item.index for item in all_items}), 128)
        self.assertEqual({item.global_occurrence for item in all_items}, set(range(128)))


class AlignedDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("/home/jiahao/task/UniLIP/data/csgo_benchmark_v2")
        if not (cls.root / "benchmark_manifest.json").is_file():
            raise unittest.SkipTest("Seen-10 fixture is unavailable")
        cls.language = {name: torch.ones((2, 4)) for name in SEEN_MAPS}

    def test_external_target_and_validation_boundary(self):
        train = Seen10Dataset(
            self.root, "seen_train", _TinyProcessor(), self.language,
            limit_per_map=1, augmentation=_CONFIG, external_action_dim=5,
        )
        item = train[SampleOccurrence(0, 0, 0)]
        replay = train[SampleOccurrence(0, 0, 0)]
        self.assertEqual(tuple(item["actions"].shape), (1, 5))
        self.assertTrue(torch.equal(item["actions"], replay["actions"]))
        self.assertTrue(torch.equal(item["images"][0], replay["images"][0]))
        self.assertEqual(int(item["state_elem_mask"].sum()), 0)
        self.assertFalse(any(key in item["metadata"] for key in ("pose", "x", "y", "z")))
        batch = collate_seen10([item])
        self.assertEqual(tuple(batch["actions"].shape), (1, 1, 5))
        self.assertNotIn("global_occurrence", batch)

        validation = Seen10Dataset(
            self.root, "seen_validation", _TinyProcessor(), self.language,
            limit_per_map=1, augmentation=_CONFIG, external_action_dim=5,
        )
        a = validation[SampleOccurrence(0, 0, 0)]
        b = validation[SampleOccurrence(0, 3, 123456)]
        self.assertTrue(torch.equal(a["images"][0], b["images"][0]))
        self.assertTrue(torch.equal(a["images"][1], b["images"][1]))

    def test_worker_count_and_resume_replay_exact_views(self):
        train = Seen10Dataset(
            self.root, "seen_train", _TinyProcessor(), self.language,
            limit_per_map=1, augmentation=_CONFIG, external_action_dim=5,
        )
        occurrences = [SampleOccurrence(i, 2, 49920 + i) for i in range(4)]

        def read(indices, workers):
            loader = DataLoader(
                train, batch_size=2, sampler=indices, num_workers=workers,
                collate_fn=collate_seen10,
            )
            return torch.cat([batch["images"] for batch in loader])

        baseline = read(occurrences, 0)
        workers = read(occurrences, 2)
        resumed = read(occurrences[2:], 2)
        self.assertTrue(torch.equal(baseline, workers))
        self.assertTrue(torch.equal(baseline[2:], resumed))

    def test_normalization_roundtrip_and_unlabeled_eval(self):
        for split in ("seen_validation", "seen_discrete_test"):
            dataset = Seen10Dataset(
                self.root, split, _TinyProcessor(), self.language,
                limit_per_map=1, augmentation=_CONFIG, external_action_dim=5,
                labels=False,
            )
            row = dataset.rows[0]
            normalized = dataset.pose(row)
            physical = dataset.physical_pose(normalized, row["map_name"])
            self.assertAlmostEqual(physical[0], row["x"], places=4)
            self.assertAlmostEqual(physical[1], row["y"], places=4)
            self.assertAlmostEqual(physical[2], row["z"], places=4)
            self.assertAlmostEqual(physical[3], row["angle_v"] * 180 / np.pi, places=4)
            self.assertAlmostEqual(physical[4], row["angle_h"] * 180 / np.pi, places=4)
            a = dataset[SampleOccurrence(0, 0, 0)]
            b = dataset[SampleOccurrence(0, 40, 1996800)]
            self.assertNotIn("actions", a)
            self.assertNotIn("labels", a)
            self.assertEqual(set(a["metadata"]).isdisjoint({"pose", "x", "y", "z", "angle_v", "angle_h"}), True)
            self.assertTrue(torch.equal(a["images"][0], b["images"][0]))
            self.assertTrue(torch.equal(a["images"][1], b["images"][1]))
            batch = collate_seen10([a])
            self.assertNotIn("actions", batch)
            self.assertNotIn("labels", batch)

    def test_all_published_split_identities_and_counts(self):
        manifest = json.loads((self.root / "benchmark_manifest.json").read_text())
        identities = {}
        for split, expected in (("train", 50000), ("validation", 5000), ("discrete_test", 20000)):
            ids = []
            for map_name in SEEN_MAPS:
                rows = json.loads((self.root / "splits" / "seen" / map_name / f"{split}.json").read_text())
                self.assertEqual(len(rows), manifest["counts"]["seen"][map_name][split])
                ids.extend((map_name, Path(row["file_frame"]).stem) for row in rows)
            self.assertEqual(len(ids), expected)
            self.assertEqual(len(set(ids)), expected)
            identities[split] = set(ids)
        self.assertFalse(identities["train"] & identities["validation"])
        self.assertFalse(identities["train"] & identities["discrete_test"])
        self.assertFalse(identities["validation"] & identities["discrete_test"])


if __name__ == "__main__":
    unittest.main()
