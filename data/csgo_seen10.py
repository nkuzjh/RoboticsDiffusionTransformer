"""Manifest-driven CSGO Benchmark v2 Seen-10 localization data.

The adapter deliberately has no dependency on the repository's robot data
pipeline.  It reads the published report, manifest, calibration and split
rows, and only opens an image when a sample is requested.  The returned
``metadata`` record is kept separate from model tensors so callers can use it
for prediction identity and qualitative visualizations without leaking GT
pose into a model input.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.csgo_update_sampler import SampleOccurrence


SEEN_MAPS = (
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
)
MAP_TO_INDEX = {name: index for index, name in enumerate(SEEN_MAPS)}
_TAU = 2.0 * math.pi
_FILE_FRAME_RE = re.compile(r"^file_num(?P<file_num>\d+)_frame_(?P<frame_id>\d+)$")
_SPLIT_ALIASES = {
    "train": "train",
    "seen_train": "train",
    "validation": "validation",
    "seen_validation": "validation",
    "discrete_test": "discrete_test",
    "seen_discrete_test": "discrete_test",
}
_COUNT_NAMES = {"train": "train", "validation": "validation", "discrete_test": "discrete_test"}

# Kept as a template and exposed for callers that build their own tokenizer
# inputs.  The map name is a condition, never a pose or target label.
INSTRUCTION = (
    "Localize the player in {map_name} using the first-person image and radar map. "
    "Predict the absolute normalized x, y, z, pitch, and yaw."
)


class BenchmarkDataError(ValueError):
    """Raised when the published Seen-10 bundle violates its data contract."""


def instruction_for_map(map_name: str) -> str:
    """Return the fixed localization instruction for one Seen-10 map."""

    if map_name not in MAP_TO_INDEX:
        raise ValueError(f"Unknown Seen-10 map: {map_name!r}")
    return INSTRUCTION.format(map_name=map_name)


# A few integrations use a function named ``get_instruction``; keeping this
# tiny alias costs nothing and makes the adapter convenient without importing
# any model code.
get_instruction = instruction_for_map


class Seen10Dataset(Dataset):
    """Read one published Seen-10 split and produce RDT-compatible samples.

    Args:
        data_root: Root of the read-only ``csgo_benchmark_v2`` bundle.
        split: ``train``, ``validation`` or ``discrete_test``; ``seen_*``
            aliases are accepted.
        image_processor: Native SigLIP image processor (or a compatible
            callable/transform).  Both FPV and map radar are aspect-padded
            before processing.
        language_embeddings: Optional per-map precomputed ``.pt`` files or a
            mapping from map name to tensor/path.  A tensor can also be shared
            for every map.
        tokenizer: Optional tokenizer used when no map embedding is supplied.
        state_dim: RDT state/action width.  The adapter always places the five
            labels at the beginning of the action vector.
        limit_per_map: Optional prefix limit applied independently per map;
            useful for smoke tests while preserving map ordering.
        labels: If false, omit ``actions`` and ``labels`` from each item.
            Metadata still carries only sample identity/media paths; GT pose is
            not included in model inputs or metadata.
    """

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        split: str,
        image_processor: Any,
        language_embeddings: Any = None,
        tokenizer: Any = None,
        state_dim: int = 128,
        limit_per_map: int | None = None,
        labels: bool = True,
        augmentation: Mapping[str, Any] | None = None,
        external_action_dim: int | None = None,
    ) -> None:
        super().__init__()
        if image_processor is None:
            raise ValueError("image_processor is required so FPV and radar use native preprocessing")
        if not isinstance(state_dim, int) or state_dim < 5:
            raise ValueError(f"state_dim must be an integer >= 5, got {state_dim!r}")
        if limit_per_map is not None and (not isinstance(limit_per_map, int) or limit_per_map < 0):
            raise ValueError(f"limit_per_map must be a non-negative integer or None, got {limit_per_map!r}")

        canonical_split = _SPLIT_ALIASES.get(str(split))
        if canonical_split is None:
            raise ValueError(
                f"Unsupported Seen-10 split {split!r}; choose from {sorted(_SPLIT_ALIASES)}"
            )
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = canonical_split
        self.split_name = str(split)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.state_dim = state_dim
        self.labels = bool(labels)
        if external_action_dim not in (None, 5):
            raise ValueError("external_action_dim must be 5 for aligned localization or None for legacy")
        self.external_action_dim = external_action_dim
        self.augmentation = dict(augmentation) if augmentation is not None else None
        if self.augmentation is not None:
            from data.csgo_augmentation import validate_augmentation

            validate_augmentation(self.augmentation)
        self.language_embeddings = language_embeddings
        self.maps = SEEN_MAPS
        self.map_to_index = dict(MAP_TO_INDEX)

        manifest_path = self.data_root / "benchmark_manifest.json"
        report_path = self.data_root / "minimal_dataset_report.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")
        if not report_path.is_file():
            raise FileNotFoundError(f"Minimal dataset report not found: {report_path}")
        self.manifest = _read_json(manifest_path)
        self.report = _read_json(report_path)
        self._validate_bundle_headers()
        self.z_ranges = self._load_z_ranges()
        self._image_template, self._image_root, self._radar_root, self._radar_targets = self._load_media_mappings()
        self.counts = {
            map_name: int(self.manifest.get("counts", {}).get("seen", {}).get(map_name, {}).get(_COUNT_NAMES[self.split], -1))
            for map_name in self.maps
        }
        if any(value < 0 for value in self.counts.values()):
            missing = [name for name, value in self.counts.items() if value < 0]
            raise BenchmarkDataError(f"Manifest is missing {self.split} counts for {missing}")

        self.rows = self._load_rows(limit_per_map)
        # A stable alias is useful to code written against other Seen-10
        # adapters, while ``rows`` remains the canonical public attribute.
        self.records = self.rows

        # Resolve/cache per-map language embeddings once.  Loading them here
        # avoids repeatedly deserializing a .pt for each frame, while images
        # stay lazy and are never scanned.
        self._language_by_map = {
            map_name: self._load_language_embedding(map_name)
            for map_name in self.maps
        }

    def _validate_bundle_headers(self) -> None:
        if self.manifest.get("benchmark_id") != "csgo_benchmark_v2":
            raise BenchmarkDataError("benchmark_manifest.json is not csgo_benchmark_v2")
        if self.report.get("benchmark_id") != "csgo_benchmark_v2":
            raise BenchmarkDataError("minimal_dataset_report.json is not csgo_benchmark_v2")
        if self.report.get("status") != "verified":
            raise BenchmarkDataError(
                f"minimal dataset report status must be verified, got {self.report.get('status')!r}"
            )
        protocol = self.manifest.get("protocol", {})
        manifest_maps = tuple(protocol.get("seen_maps", ()))
        if manifest_maps != SEEN_MAPS:
            raise BenchmarkDataError(
                f"Seen-10 map order differs from published contract: {manifest_maps!r}"
            )

    def _load_z_ranges(self) -> dict[str, dict[str, float]]:
        calibration_meta = self.manifest.get("calibration", {})
        calibration_rel = str(calibration_meta.get("file", "calibration/z_calibration.json"))
        calibration_path = _safe_under_root(self.data_root, calibration_rel, "calibration")
        calibration = _read_json(calibration_path)
        fingerprint = calibration_meta.get("fingerprint")
        if fingerprint and calibration.get("calibration_sha256") != fingerprint:
            raise BenchmarkDataError("Manifest and z_calibration.json fingerprints differ")

        manifest_ranges = calibration_meta.get("z_ranges", {})
        calibration_ranges = calibration.get("z_ranges", {})
        ranges: dict[str, dict[str, float]] = {}
        for map_name in self.maps:
            values = manifest_ranges.get(map_name) or calibration_ranges.get(map_name)
            if not isinstance(values, Mapping):
                raise BenchmarkDataError(f"Published z calibration is missing map={map_name}")
            try:
                z_min, z_max = float(values["z_min"]), float(values["z_max"])
            except (KeyError, TypeError, ValueError) as exc:
                raise BenchmarkDataError(f"Invalid z calibration for {map_name}: {values!r}") from exc
            if not math.isfinite(z_min) or not math.isfinite(z_max) or not z_max > z_min:
                raise BenchmarkDataError(f"Invalid z range for {map_name}: {z_min}, {z_max}")
            other = calibration_ranges.get(map_name)
            if isinstance(other, Mapping) and (
                z_min != float(other["z_min"]) or z_max != float(other["z_max"])
            ):
                raise BenchmarkDataError(f"Manifest and calibration z range differ for {map_name}")
            ranges[map_name] = {"z_min": z_min, "z_max": z_max}
        return ranges

    def _load_media_mappings(self):
        images = self.report.get("images", {})
        image_template = images.get("target_template")
        image_root = str(images.get("root", "images"))
        if not isinstance(image_template, str) or "{map}" not in image_template or "{file_frame}" not in image_template:
            raise BenchmarkDataError("minimal report has no usable images.target_template")

        radars = self.report.get("radars", {})
        radar_root = str(radars.get("root", "radars"))
        radar_targets: dict[str, str] = {}
        entries = radars.get("entries", [])
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                map_name, target = entry.get("map"), entry.get("target")
                if map_name in radar_targets:
                    raise BenchmarkDataError(f"Duplicate radar mapping for map={map_name!r}")
                if map_name and target:
                    radar_targets[str(map_name)] = str(target)
        missing = [name for name in self.maps if name not in radar_targets]
        if missing:
            raise BenchmarkDataError(f"Minimal report is missing Seen-10 radar mappings: {missing}")
        return image_template, image_root, radar_root, radar_targets

    def _load_rows(self, limit_per_map: int | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        split_file = f"{self.split}.json"
        for map_name in self.maps:
            path = _safe_under_root(
                self.data_root,
                Path("splits") / "seen" / map_name / split_file,
                f"{map_name}/{split_file}",
            )
            payload = _read_json(path)
            if not isinstance(payload, list):
                raise BenchmarkDataError(f"Expected JSON row list in {path}")
            expected = self.counts[map_name]
            if len(payload) != expected:
                raise BenchmarkDataError(
                    f"Manifest count mismatch for {map_name}/{self.split}: expected {expected}, got {len(payload)}"
                )
            selected = payload if limit_per_map is None else payload[:limit_per_map]
            for raw in selected:
                rows.append(self._normalize_row(map_name, raw))
        return rows

    def _normalize_row(self, expected_map: str, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise BenchmarkDataError(f"Split row must be an object, got {type(raw).__name__}")
        map_name = str(raw.get("map", expected_map))
        if map_name != expected_map:
            raise BenchmarkDataError(f"Row map mismatch: expected {expected_map}, got {map_name}")
        file_frame_value = raw.get("file_frame")
        if not isinstance(file_frame_value, str) or not file_frame_value or Path(file_frame_value).name != file_frame_value:
            raise BenchmarkDataError(f"Invalid file_frame {file_frame_value!r}")
        file_frame = Path(file_frame_value).stem
        parsed = _FILE_FRAME_RE.fullmatch(file_frame)
        if parsed is None:
            raise BenchmarkDataError(f"Invalid file_frame identity {file_frame_value!r}")
        try:
            x, y, z = (float(raw[key]) for key in ("x", "y", "z"))
            angle_v, angle_h = float(raw["angle_v"]), float(raw["angle_h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkDataError(f"Invalid pose fields for {map_name}/{file_frame}") from exc
        if not all(math.isfinite(value) for value in (x, y, z, angle_v, angle_h)):
            raise BenchmarkDataError(f"Non-finite pose for {map_name}/{file_frame}")

        z_range = self.z_ranges[map_name]
        pose = [
            x / 1024.0,
            y / 1024.0,
            (z - z_range["z_min"]) / (z_range["z_max"] - z_range["z_min"]),
            angle_v / _TAU,
            angle_h / _TAU,
        ]
        image_rel = self._image_template.format(map=map_name, file_frame=file_frame_value)
        image_path = _safe_under_root(self.data_root, image_rel, f"image {map_name}/{file_frame}")
        radar_rel = Path(self._radar_root) / self._radar_targets[map_name]
        radar_path = _safe_under_root(self.data_root, radar_rel, f"radar {map_name}")
        # Resolve media paths now so metadata is deterministic.  This is a
        # direct check of manifest-referenced files, never a directory scan.
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image for {map_name}/{file_frame}: {image_path}")
        if not radar_path.is_file():
            raise FileNotFoundError(f"Missing radar for {map_name}: {radar_path}")

        raw_pose = {
            "x": x,
            "y": y,
            "z": z,
            "pitch": angle_v,
            "yaw": angle_h,
            "angle_v_rad": angle_v,
            "angle_h_rad": angle_h,
        }
        return {
            # Identity follows the evaluator's accepted map/file_frame form.
            "sample_id": f"{map_name}/{file_frame}",
            "map_name": map_name,
            "map": map_name,
            "file_frame": file_frame,
            "file_num": int(parsed.group("file_num")),
            "frame_id": int(parsed.group("frame_id")),
            "image_path": str(image_path),
            "radar_path": str(radar_path),
            "pose": pose,
            "pose_raw": raw_pose,
            "z_calibration": dict(z_range),
            # Retain source row fields for callers that need exact metadata.
            "x": x,
            "y": y,
            "z": z,
            "angle_v": angle_v,
            "angle_h": angle_h,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def pose(self, row: Mapping[str, Any]) -> list[float]:
        """Return normalized ``[x,y,z,pitch,yaw]`` for a metadata row."""

        if "pose" in row:
            values = row["pose"]
            if len(values) != 5:
                raise ValueError(f"Expected a 5DoF pose, got {values!r}")
            return [float(value) for value in values]
        map_name = str(row.get("map_name", row.get("map", "")))
        if map_name not in self.z_ranges:
            raise ValueError(f"Unknown map in row: {map_name!r}")
        return self._normalize_pose_from_raw(row, map_name)

    def _normalize_pose_from_raw(self, row: Mapping[str, Any], map_name: str) -> list[float]:
        z_range = self.z_ranges[map_name]
        return [
            float(row["x"]) / 1024.0,
            float(row["y"]) / 1024.0,
            (float(row["z"]) - z_range["z_min"]) / (z_range["z_max"] - z_range["z_min"]),
            float(row["angle_v"]) / _TAU,
            float(row["angle_h"]) / _TAU,
        ]

    def physical_pose(self, pose: Sequence[float], map_name: str) -> list[float]:
        """Convert normalized pose to physical ``[x,y,z,pitch,yaw]``.

        Angles are returned in degrees, matching the shared evaluator and
        visualization contract.  Values are intentionally not clipped.
        """

        if map_name not in self.z_ranges:
            raise ValueError(f"Unknown Seen-10 map: {map_name!r}")
        if len(pose) != 5:
            raise ValueError(f"Expected normalized 5DoF pose, got {pose!r}")
        z_range = self.z_ranges[map_name]
        return [
            float(pose[0]) * 1024.0,
            float(pose[1]) * 1024.0,
            float(pose[2]) * (z_range["z_max"] - z_range["z_min"]) + z_range["z_min"],
            float(pose[3]) * 360.0,
            float(pose[4]) * 360.0,
        ]

    def __getitem__(self, index: int | SampleOccurrence) -> dict[str, Any]:
        if isinstance(index, SampleOccurrence):
            occurrence = index
            index = occurrence.index
        else:
            occurrence = SampleOccurrence(index=int(index), epoch=0, global_occurrence=int(index))
        row = self.rows[index]
        pose = self.pose(row)
        fpv = self._process_path(row["image_path"], row=row, occurrence=occurrence, view="fpv")
        radar = self._process_path(row["radar_path"], row=row, occurrence=occurrence, view="radar")

        # The native RDT collator expects a list of view tensors and stacks it
        # into (B, 2, C, H, W).  State is always a zero placeholder; GT pose
        # lives only in actions/metadata and is never copied to state.
        state = torch.zeros((1, self.state_dim), dtype=torch.float32)
        state_mask = torch.zeros((self.state_dim,), dtype=torch.float32)
        if self.external_action_dim is None:
            state_mask[:5] = 1.0
        item: dict[str, Any] = {
            "states": state,
            "state_elem_mask": state_mask,
            "state_norm": torch.ones((self.state_dim,), dtype=torch.float32),
            "images": [fpv, radar],
            "ctrl_freq": 1,
            "ctrl_freqs": 1,
            "data_idx": self.map_to_index[row["map_name"]],
            "dataset_name": row["map_name"],
            "instruction": instruction_for_map(row["map_name"]),
            "language_instruction": instruction_for_map(row["map_name"]),
            # Metadata is consumed by output/visualization code only.  The
            # collator below intentionally excludes this field.
            "metadata": _model_metadata(row),
        }
        lang_embed = self._language_by_map[row["map_name"]]
        if lang_embed is not None:
            item["lang_embed"] = lang_embed.clone() if isinstance(lang_embed, torch.Tensor) else lang_embed
        elif self.tokenizer is not None:
            tokenized = self._tokenize(item["instruction"])
            item.update(tokenized)
        else:
            raise ValueError(
                "Seen-10 localization requires a precomputed language embedding "
                f"for map {row['map_name']!r} or an explicit tokenizer"
            )

        if self.labels:
            action = torch.zeros((1, self.external_action_dim or self.state_dim), dtype=torch.float32)
            action[0, :5] = torch.as_tensor(pose, dtype=torch.float32)
            item["actions"] = action
            # ``labels`` is a convenient explicit alias for non-RDT heads;
            # native train.py consumes ``actions``.
            item["labels"] = torch.as_tensor(pose, dtype=torch.float32)
        return item

    def _tokenize(self, instruction: str) -> dict[str, torch.Tensor]:
        result = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        if isinstance(result, Mapping):
            input_ids = result.get("input_ids")
            attention = result.get("attention_mask")
        else:
            input_ids = getattr(result, "input_ids", None)
            attention = getattr(result, "attention_mask", None)
        if input_ids is None:
            raise ValueError("tokenizer output does not contain input_ids")
        input_ids = torch.as_tensor(input_ids)
        if input_ids.ndim == 2:
            input_ids = input_ids[0]
        answer = {"input_ids": input_ids.long()}
        if attention is not None:
            attention = torch.as_tensor(attention)
            if attention.ndim == 2:
                attention = attention[0]
            answer["lang_attn_mask"] = attention.bool()
        return answer

    def _load_language_embedding(self, map_name: str) -> torch.Tensor | None:
        source = _embedding_source(self.language_embeddings, map_name)
        if source is None:
            return None
        value = source
        if isinstance(source, (str, os.PathLike)):
            path = Path(source).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Language embedding not found for {map_name}: {path}")
            value = torch.load(path, map_location="cpu")
        if isinstance(value, Mapping):
            if map_name in value:
                value = value[map_name]
        if isinstance(value, Mapping):
            for key in ("embeddings", "lang_embed", "embedding", "tensor"):
                if key in value:
                    value = value[key]
                    if isinstance(value, Mapping) and map_name in value:
                        value = value[map_name]
                    break
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2 or value.shape[0] == 0:
            raise ValueError(f"Language embedding for {map_name} must have shape (tokens, dim), got {tuple(value.shape)}")
        return value.detach().cpu()

    def _process_path(
        self,
        path: str | os.PathLike[str],
        *,
        row: Mapping[str, Any] | None = None,
        occurrence: SampleOccurrence | None = None,
        view: str | None = None,
    ) -> torch.Tensor:
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.augmentation is not None and self.split == "train":
            if row is None or occurrence is None or view is None:
                raise ValueError("Augmentation requires row, occurrence and view identity")
            from data.csgo_augmentation import augment_image

            image, _ = augment_image(
                image,
                config=self.augmentation,
                epoch=occurrence.epoch,
                global_occurrence=occurrence.global_occurrence,
                sample_id=row["sample_id"],
                view=view,
            )
        image = _expand_to_square(image, _processor_mean(self.image_processor))
        return _process_image(self.image_processor, image)


def collate_seen10(instances: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate Seen-10 items without importing ``train.dataset``.

    ``metadata`` and all GT aliases are deliberately kept as a Python list and
    are not sent to the model.  This mirrors the tensor names used by RDT's
    native collator while also supporting ``labels=False`` inference samples.
    """

    if not instances:
        raise ValueError("collate_seen10 requires at least one instance")
    batch: dict[str, Any] = {
        "states": torch.stack([_tensor(instance["states"]) for instance in instances]),
        "state_elem_mask": torch.stack([_tensor(instance["state_elem_mask"]) for instance in instances]),
        "state_norm": torch.stack([_tensor(instance["state_norm"]) for instance in instances]),
        "images": torch.stack([torch.stack([_tensor(view) for view in instance["images"]]) for instance in instances]),
        "data_indices": torch.tensor([int(instance.get("data_idx", instance.get("data_idx", 0))) for instance in instances], dtype=torch.long),
        "ctrl_freqs": torch.tensor([int(instance.get("ctrl_freq", instance.get("ctrl_freqs", 1))) for instance in instances], dtype=torch.long),
        "metadata": [copy.deepcopy(instance.get("metadata", {})) for instance in instances],
    }
    if all("actions" in instance for instance in instances):
        batch["actions"] = torch.stack([_tensor(instance["actions"]) for instance in instances])
    if all("labels" in instance for instance in instances):
        batch["labels"] = torch.stack([_tensor(instance["labels"]) for instance in instances])

    if all("lang_embed" in instance for instance in instances):
        embeds = [_tensor(instance["lang_embed"]) for instance in instances]
        lengths = [int(embed.shape[0]) for embed in embeds]
        batch["lang_embeds"] = torch.nn.utils.rnn.pad_sequence(embeds, batch_first=True, padding_value=0)
        mask = torch.zeros(batch["lang_embeds"].shape[:2], dtype=torch.bool)
        for index, length in enumerate(lengths):
            mask[index, :length] = True
        batch["lang_attn_mask"] = mask
    elif all("input_ids" in instance for instance in instances):
        input_ids = [instance["input_ids"] for instance in instances]
        pad_id = 0
        tokenizer = instances[0].get("tokenizer")
        if tokenizer is not None:
            pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        batch["lang_attn_mask"] = batch["input_ids"].ne(pad_id)
    else:
        raise ValueError("Seen-10 instances need either lang_embed or input_ids")
    return batch


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _safe_under_root(root: Path, relative: str | os.PathLike[str], description: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    if root != candidate and root not in candidate.parents:
        raise BenchmarkDataError(f"{description} path escapes data root: {relative}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing {description}: {candidate}")
    return candidate


def _embedding_source(sources: Any, map_name: str) -> Any:
    if sources is None:
        return None
    if isinstance(sources, Mapping):
        if map_name in sources:
            return sources[map_name]
        # A common wrapper stores paths/tensors under this key.
        for key in ("embeddings", "lang_embeddings", "by_map"):
            nested = sources.get(key)
            if isinstance(nested, Mapping) and map_name in nested:
                return nested[map_name]
        return None
    if isinstance(sources, (str, os.PathLike)):
        path = Path(sources).expanduser()
        if path.is_dir():
            for filename in (f"{map_name}.pt", f"lang_embed_{map_name}.pt", f"{map_name}_lang_embed.pt"):
                candidate = path / filename
                if candidate.is_file():
                    return candidate
            return None
        return path
    return sources


def _model_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return non-label metadata safe to carry alongside model tensors."""

    fields = ("sample_id", "map_name", "map", "file_frame", "image_path", "radar_path")
    return {field: row[field] for field in fields if field in row}


def _processor_mean(processor: Any) -> tuple[int, int, int]:
    mean = getattr(processor, "image_mean", None)
    if mean is None and isinstance(processor, Mapping):
        mean = processor.get("image_mean")
    if mean is None:
        mean = (0.5, 0.5, 0.5)
    values = list(mean)
    if len(values) == 1:
        values *= 3
    if len(values) < 3:
        values = (values + [0.5, 0.5, 0.5])[:3]
    return tuple(max(0, min(255, int(round(float(value) * 255.0)))) for value in values[:3])


def _expand_to_square(image: Image.Image, background: Sequence[int]) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    canvas = Image.new("RGB", (size, size), tuple(int(value) for value in background))
    canvas.paste(image, ((size - width) // 2, (size - height) // 2))
    return canvas


def _process_image(processor: Any, image: Image.Image) -> torch.Tensor:
    if hasattr(processor, "preprocess"):
        output = processor.preprocess(image, return_tensors="pt")
    elif callable(processor):
        output = processor(image)
    else:
        raise TypeError("image_processor must expose preprocess() or be callable")
    if isinstance(output, Mapping):
        output = output.get("pixel_values", output.get("images", output))
    elif hasattr(output, "pixel_values"):
        output = output.pixel_values
    tensor = torch.as_tensor(output)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3) and tensor.shape[0] not in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    if tensor.ndim != 3:
        raise ValueError(f"image_processor must return one CHW tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _tensor(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


__all__ = [
    "BenchmarkDataError",
    "INSTRUCTION",
    "MAP_TO_INDEX",
    "SEEN_MAPS",
    "Seen10Dataset",
    "collate_seen10",
    "get_instruction",
    "instruction_for_map",
]
