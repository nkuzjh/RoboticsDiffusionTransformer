#!/usr/bin/env python3
"""Run RDT localization on the manifest-driven Seen-10 discrete test split."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Subset

from data.csgo_seen10 import Seen10Dataset, collate_seen10
from models.csgo_rdt import CSGORDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from train.csgo_hooks import prepare_csgo_batch
from train.csgo_visualize import render_localization
from scripts.csgo_paths import data_root as resolve_data_root
from scripts.csgo_paths import run_directories


PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTION_FIELDS = ("pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw")


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return value


def _path(value: Any, base: Path) -> Path:
    candidate = Path(os.fspath(value)).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _asset(value: Any, config_path: Path) -> Any:
    """Resolve a project-local model directory while preserving Hub IDs."""

    if value is None:
        return None
    candidate = Path(os.fspath(value)).expanduser()
    # Configs may live under ``.cache/...`` for smoke runs.  Resolve the
    # repository-local asset paths from this entrypoint's project root rather
    # than guessing from the config's nesting depth.
    project_root = PROJECT_ROOT
    if not candidate.is_absolute() and (project_root / candidate).exists():
        return str((project_root / candidate).resolve())
    return value


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _resolve_batch_size(config: Mapping[str, Any], override: int | None = None) -> int:
    """Inference batches are independent of the training update budget."""
    value = override if override is not None else config.get("inference", {}).get(
        "batch_size", config.get("training", {}).get("eval_batch_size", 1),
    )
    try:
        return _positive_int(str(value))
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise ValueError(f"inference batch size must be a positive integer, got {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("infer", "smoke"), nargs="?", default="infer")
    parser.add_argument("--config", "--config-path", dest="config_path", type=Path, default=PROJECT_ROOT / "configs/csgo_seen10.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Complete seed artifact directory")
    parser.add_argument("--language-embeddings", default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--limit-per-map", type=_positive_int, default=None)
    parser.add_argument("--visualization-seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


def _run_paths(config: Mapping[str, Any], config_path: Path, cli: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_root = PROJECT_ROOT
    seed = int(cli.seed if cli.seed is not None else config.get("seed", 0))
    output_dir, checkpoint_dir = run_directories(
        config, seed=seed, smoke=cli.mode == "smoke", output=cli.output_dir,
        checkpoint=cli.checkpoint, root=project_root,
    )
    data_root = resolve_data_root(config, cli.data_root, root=project_root)
    return data_root, output_dir, checkpoint_dir


def _checkpoint_path(root: Path, *, rule: str | None = None) -> Path:
    if root.is_file():
        return root
    for name in ((rule,) if rule else ("best", "late")):
        candidate = root / name
        if candidate.exists():
            return candidate
    if (root / "config.json").is_file():
        return root
    checkpoints = [p for p in root.glob("checkpoint-*") if p.is_dir()]
    if checkpoints:
        def step(path: Path) -> int:
            try:
                return int(path.name.split("-", 1)[1])
            except (IndexError, ValueError):
                return -1

        return sorted(checkpoints, key=lambda p: (step(p), p.name))[-1]
    raise FileNotFoundError(f"No loadable checkpoint found under {root}")


def _reject_smoke_checkpoint(checkpoint: Path, *, formal: bool) -> None:
    if not formal:
        return
    marker = checkpoint / "smoke_only.json"
    if not marker.is_file():
        return
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint smoke marker: {marker}") from exc
    if bool(payload.get("smoke_only")):
        raise ValueError(f"Refusing smoke-only checkpoint for formal inference: {checkpoint}")


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    if not path.is_file():
        return rows, identities
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Prediction at {path}:{line_no} is not an object")
            map_name = row.get("map_name")
            sample_id = row.get("sample_id")
            if not isinstance(map_name, str) or not isinstance(sample_id, str):
                raise ValueError(f"Prediction at {path}:{line_no} lacks map_name/sample_id")
            for field in PREDICTION_FIELDS:
                value = row.get(field)
                if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
                    raise ValueError(f"Prediction at {path}:{line_no} has invalid {field}")
            identity = (map_name, sample_id)
            if identity in identities:
                raise ValueError(f"Duplicate prediction identity in {path}: {identity}")
            identities.add(identity)
            rows.append(row)
    return rows, identities


def _provenance(path: Path, *, checkpoint: Path, seed: int, data_root: Path,
                extra: Mapping[str, Any] | None = None) -> None:
    expected = {
        "model_name": "RDT",
        "task": "localization",
        "split": "seen_discrete_test",
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "data_root": str(data_root.resolve()),
        "prediction_pose_space": "normalized",
        "model_inputs_contain_ground_truth": False,
    }
    if extra:
        expected.update(extra)
    def portable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, Mapping):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [portable(item) for item in value]
        return value
    expected = portable(expected)
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(f"Inference provenance mismatch for {key}: {existing.get(key)!r} != {value!r}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _language_embeddings(
    config: Mapping[str, Any],
    cli: argparse.Namespace,
    config_path: Path,
    checkpoint: Path,
    output_dir: Path,
):
    csgo = config.get("csgo", {})
    if not isinstance(csgo, dict):
        csgo = {}
    language = cli.language_embeddings or config.get("language_embeddings") or csgo.get("language_embeddings")
    metadata_path = output_dir / "training_metadata.json"
    if language is None and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid training metadata: {metadata_path}") from exc
        language = metadata.get("language_embeddings")
        if language is None:
            args_metadata = metadata.get("args")
            if isinstance(args_metadata, Mapping):
                language = args_metadata.get("csgo_language_embeddings")
    if language:
        candidate = Path(os.fspath(language)).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Training language embedding cache is missing: {candidate}")
        return candidate
    run_cache = checkpoint.parent / "language_embeddings.pt"
    if run_cache.is_file():
        return run_cache
    max_length = int(config.get("dataset", {}).get("tokenizer_max_length", 1024))
    cache_path = checkpoint.parent / "language_embeddings.pt"
    text_path = _asset(config.get("pretrained_text_encoder_name_or_path"), config_path)
    if not text_path:
        raise ValueError(f"Missing language cache {cache_path}; run training once or provide --language-embeddings")
    # Match the native hook's one-time, ten-map cache.  Inference never runs
    # T5 in the batch loop and therefore cannot accidentally encode a label.
    from train.csgo_hooks import _precompute_map_language_embeddings

    _precompute_map_language_embeddings(
        model_path=text_path,
        max_length=max_length,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cache_path=cache_path,
    )
    if not cache_path.is_file():
        raise FileNotFoundError(f"Language embedding cache was not created: {cache_path}")
    return cache_path


def _predict(
    model: CSGORDTRunner,
    dataset: Seen10Dataset,
    pending: Sequence[int],
    *,
    vision: SiglipVisionTower,
    text_encoder: Any,
    accelerator: Accelerator,
    dtype: torch.dtype,
    batch_size: int,
    output_path: Path,
) -> None:
    loader = DataLoader(
        Subset(dataset, list(pending)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_seen10,
        pin_memory=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream, torch.no_grad():
        for batch in loader:
            prepared = prepare_csgo_batch(
                batch,
                vision_encoder=vision,
                text_encoder=text_encoder,
                accelerator=accelerator,
                dtype=dtype,
                training=False,
            )
            action_mask = torch.zeros(
                (prepared["img_tokens"].shape[0], 1, 128),
                device=accelerator.device,
                dtype=dtype,
            )
            action_mask[..., :5] = 1
            prepared["action_mask"] = action_mask
            context = (torch.autocast(device_type=accelerator.device.type, dtype=dtype)
                       if getattr(model, "diffusion_channel_policy", "legacy_valid5") == "native_full"
                       and dtype == torch.bfloat16 else contextlib.nullcontext())
            with context:
                prediction = model.predict_action(**prepared).detach().float().cpu()
            metadata = batch["metadata"]
            for index, item in enumerate(metadata):
                values = prediction[index, 0, :5].tolist()
                if not all(torch.isfinite(torch.tensor(value)) for value in values):
                    raise ValueError(f"Model produced non-finite prediction for {item}")
                row = {
                    "sample_id": str(item["sample_id"]),
                    "map_name": str(item["map_name"]),
                    "pred_x": float(values[0]),
                    "pred_y": float(values[1]),
                    "pred_z": float(values[2]),
                    "pred_pitch": float(values[3]),
                    "pred_yaw": float(values[4]),
                }
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    config_path = cli.config_path.expanduser().resolve()
    config = _yaml(config_path)
    aligned = config.get("profile") == "aligned_native_aug_v1"
    data_root, output_dir, checkpoint_root = _run_paths(config, config_path, cli)
    seed = int(cli.seed if cli.seed is not None else config.get("seed", 0))
    if aligned and seed != 42:
        raise ValueError("The approved aligned inference seed is 42")
    if cli.mode == "infer" and cli.limit_per_map is not None:
        raise ValueError("--limit-per-map is only allowed for smoke")
    limit_per_map = cli.limit_per_map if cli.limit_per_map is not None else (10 if cli.mode == "smoke" else None)
    batch_size = _resolve_batch_size(config, cli.batch_size)
    if aligned and int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("aligned inference requires a single process")
    checkpoint = _checkpoint_path(checkpoint_root, rule="late" if aligned else None).resolve()
    _reject_smoke_checkpoint(checkpoint, formal=cli.mode == "infer")
    predictions_path = output_dir / "localization" / "predictions.jsonl"
    provenance_path = output_dir / "localization" / "inference_provenance.json"
    if predictions_path.is_file() and not provenance_path.is_file():
        raise ValueError(
            f"Existing predictions have no provenance file: {predictions_path}; "
            "refusing to guess their checkpoint or seed"
        )
    if not aligned:
        _provenance(provenance_path, checkpoint=checkpoint, seed=seed, data_root=data_root)
    existing, identities = _load_existing(predictions_path)

    if cli.cpu or not torch.cuda.is_available():
        torch.set_num_threads(2)
        accelerator = Accelerator(cpu=True)
        dtype = torch.float32
    else:
        accelerator = Accelerator()
        dtype = torch.bfloat16 if config.get("training", {}).get("mixed_precision") == "bf16" else torch.float32
    device = accelerator.device
    if aligned and accelerator.num_processes != 1:
        raise ValueError("aligned inference requires a single process")
    set_seed(seed)

    vision_path = _asset(config.get("pretrained_vision_encoder_name_or_path"), config_path)
    if not vision_path:
        raise ValueError("pretrained_vision_encoder_name_or_path is required")
    vision = SiglipVisionTower(vision_tower=vision_path, args=None)
    vision.vision_tower.to(device=device, dtype=dtype).eval()
    language = _language_embeddings(config, cli, config_path, checkpoint, output_dir)
    # Native training stores the single shared cache as a map->tensor mapping.
    # Seen10Dataset accepts that mapping (and also accepts a shared tensor),
    # but it intentionally does not interpret a cache file itself.  Load it
    # once here so inference never deserializes or encodes language per item.
    if isinstance(language, (str, os.PathLike)):
        language_path = Path(os.fspath(language)).expanduser()
        if language_path.is_file():
            language = torch.load(language_path, map_location="cpu", weights_only=False)
    tokenizer = None
    text_encoder = None
    dataset = Seen10Dataset(
        data_root,
        "seen_discrete_test",
        image_processor=vision.image_processor,
        language_embeddings=language,
        tokenizer=tokenizer,
        state_dim=128,
        limit_per_map=limit_per_map,
        labels=False,
    )
    expected = {(str(row["map_name"]), str(row["sample_id"])) for row in dataset.rows}
    unexpected = identities - expected
    if unexpected:
        raise ValueError(f"Existing predictions contain identities outside this split: {sorted(unexpected)[:5]}")
    if aligned and existing and len(identities) != len(expected):
        raise ValueError("Partial aligned predictions cannot be resumed by skipping IDs because this changes diffusion noise allocation. Use a new output directory and rerun from the first sample.")
    pending = [
        index
        for index, row in enumerate(dataset.rows)
        if (str(row["map_name"]), str(row["sample_id"])) not in identities
    ]
    if pending or aligned:
        model = CSGORDTRunner.from_pretrained(checkpoint, dtype=dtype)
        if aligned:
            if getattr(model, "diffusion_channel_policy", None) != "native_full":
                raise ValueError("aligned inference requires a native_full checkpoint")
            if model.num_inference_timesteps != 5:
                raise ValueError("aligned inference requires the approved five solver steps")
            model.to(device=device).eval()
            import diffusers
            manifest = data_root / "benchmark_manifest.json"
            order = [(str(row["map_name"]), str(row["sample_id"])) for row in dataset.rows]
            _provenance(provenance_path, checkpoint=checkpoint, seed=seed, data_root=data_root, extra={
                "profile": config["profile"], "batch_size": batch_size,
                "num_processes": accelerator.num_processes, "augmentation": "none",
                "solver": type(model.noise_scheduler_sample).__name__,
                "solver_config": {key: value for key, value in model.noise_scheduler_sample.config.items()
                                  if not key.startswith("_")},
                "num_inference_timesteps": model.num_inference_timesteps,
                "diffusers_version": diffusers.__version__, "torch_version": torch.__version__,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "sample_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                "checkpoint_config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
                "partial_output_policy": "reject",
            })
        else:
            model.to(device=device, dtype=dtype).eval()
    if pending:
        _predict(
            model,
            dataset,
            pending,
            vision=vision,
            text_encoder=text_encoder,
            accelerator=accelerator,
            dtype=dtype,
            batch_size=batch_size,
            output_path=predictions_path,
        )

    final_rows, final_identities = _load_existing(predictions_path)
    missing = expected - final_identities
    unexpected = final_identities - expected
    if missing or unexpected or len(final_rows) != len(expected):
        raise ValueError(
            f"Prediction coverage is incomplete: missing={len(missing)}, "
            f"unexpected={len(unexpected)}, rows={len(final_rows)}, expected={len(expected)}"
        )

    visualization = config.get("visualization", {})
    if not isinstance(visualization, dict):
        visualization = {}
    viz_seed = int(cli.visualization_seed if cli.visualization_seed is not None else visualization.get("seed", 0))
    if visualization.get("enabled", True):
        render_localization(
            dataset,
            predictions_path,
            output_dir / "localization" / "visualization",
            seed=viz_seed,
            per_map=int(visualization.get("samples_per_map", 10)),
        )
    print(json.dumps({
        "seed": seed,
        "checkpoint": str(checkpoint),
        "predictions": str(predictions_path),
        "written": len(pending),
        "total": len(dataset),
        "remaining": len(expected - final_identities),
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"infer_seen10: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
