"""Contract tests for the aligned update and checkpoint schedule."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from accelerate import Accelerator

from data.csgo_update_sampler import AlignedUpdateSampler
from train.csgo_aligned import _checkpoint_steps, _lr_factor, _resolve_resume


class AlignedTrainingContractTest(unittest.TestCase):
    def test_formal_and_smoke_save_steps(self):
        config = {"training": {"checkpoint_steps": [4000, 8000, 12000, 16000, 19500]}}
        self.assertEqual(_checkpoint_steps(config, 19500, False), (4000, 8000, 12000, 16000, 19500))
        self.assertEqual(_checkpoint_steps({"training": {"checkpoint_interval_updates": 4000}}, 19500, False),
                         (4000, 8000, 12000, 16000, 19500))
        self.assertEqual(_checkpoint_steps(config, 5, True), (1, 2, 3, 4, 5))
        with self.assertRaises(ValueError):
            _checkpoint_steps({"training": {"checkpoint_steps": [3900, 7800, 11700, 15600, 19500]}}, 19500, False)

    def test_scheduler_counts_successful_updates(self):
        factor = lambda completed: _lr_factor(
            completed, total_updates=19500, warmup_updates=59, min_ratio=0.1,
        )
        self.assertAlmostEqual(factor(0), 1 / 59)
        self.assertAlmostEqual(factor(58), 1.0)
        self.assertAlmostEqual(factor(59), 1.0)
        self.assertAlmostEqual(factor(19500), 0.1)
        self.assertGreater(factor(12000), factor(16000))

    def test_resume_sampler_restarts_at_next_complete_update(self):
        settings = dict(dataset_size=50000, seed=42, world_size=1, rank=0,
                        microbatch_size=4, gradient_accumulation_steps=32,
                        total_updates=19500)
        first = AlignedUpdateSampler(**settings, start_update=4000)
        resumed = AlignedUpdateSampler(**settings, start_update=4000)
        previous = AlignedUpdateSampler(**settings, start_update=3999)
        first_batch = next(iter(first))
        self.assertEqual(first_batch, next(iter(resumed)))
        prior_iterator = iter(previous)
        for _ in range(32):
            next(prior_iterator)
        self.assertEqual(first_batch, next(prior_iterator))

    def test_accelerate_checkpoint_has_single_model_file_and_scheduler_state(self):
        with tempfile.TemporaryDirectory() as directory:
            accelerator = Accelerator(cpu=True)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            model = accelerator.prepare_model(model)
            optimizer = accelerator.prepare_optimizer(optimizer)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 if step else 1.0)
            accelerator.register_for_checkpointing(scheduler)
            prediction = model(torch.ones(1, 2))
            prediction.sum().backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-5)
            checkpoint = Path(directory) / "checkpoint-1"
            accelerator.save_state(str(checkpoint))
            files = {path.name for path in checkpoint.iterdir()}
            self.assertTrue({"model.safetensors", "pytorch_model.bin"} & files, files)
            self.assertTrue(any(name.startswith("custom_checkpoint") for name in files), files)
            (checkpoint / "training_state.json").write_text("{}", encoding="utf-8")
            self.assertEqual(_resolve_resume(Path(directory), "latest"), checkpoint)
            scheduler.step()
            accelerator.load_state(str(checkpoint))
            self.assertEqual(scheduler.last_epoch, 1)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-5)


if __name__ == "__main__":
    unittest.main()
