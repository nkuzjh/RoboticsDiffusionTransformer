#!/usr/bin/env python3
"""Fixed Seen-10 training entry point for the native RDT trainer.

The benchmark adapter owns only argument translation.  Model construction,
data loading, validation, checkpointing and resume are implemented by the
native ``train.train`` path.  Keeping this file as a small wrapper also makes
the exact native invocation visible in run logs and avoids a second training
loop in the benchmark project.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "csgo_seen10.yaml"


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


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _optional_nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "smoke"), nargs="?", default="train")
    parser.add_argument("--config", "--config-path", dest="config_path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Benchmark artifact directory")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Native checkpoint directory")
    parser.add_argument("--pretrained-model", dest="pretrained_model", default=None)
    parser.add_argument("--pretrained-text", dest="pretrained_text", default=None)
    parser.add_argument("--pretrained-vision", dest="pretrained_vision", default=None)
    parser.add_argument("--language-embeddings", default=None)
    parser.add_argument("--max-train-steps", type=_positive_int, default=None)
    parser.add_argument("--train-limit-per-map", type=_optional_nonnegative_int, default=None)
    parser.add_argument("--eval-limit-per-map", type=_optional_nonnegative_int, default=None)
    parser.add_argument("--visualization-seed", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--cpu", action="store_true", help="Run the native adapter on CPU")
    return parser


def _resolved_paths(config: dict[str, Any], config_path: Path, cli: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_root = PROJECT_ROOT
    data_root = _path(cli.data_root if cli.data_root is not None else config["data_root"], project_root)
    output_root = config.get("output_root", "outputs/csgo_benchmark_v2_seen10")
    checkpoint_root = config.get("checkpoint_root", "checkpoints/csgo_benchmark_v2_seen10")
    model_name = str(config.get("model_name", "RDT"))
    seed = int(cli.seed if cli.seed is not None else config.get("seed", 0))
    run_name = f"seed_{seed}"
    if cli.output_dir is None:
        output_dir = _path(output_root, project_root) / model_name / run_name
    else:
        output_dir = _path(cli.output_dir, project_root)
    if cli.checkpoint_dir is None:
        checkpoint_dir = _path(checkpoint_root, project_root) / model_name / run_name
    else:
        checkpoint_dir = _path(cli.checkpoint_dir, project_root)
    return data_root, output_dir, checkpoint_dir


def _occupied(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _is_main_process() -> bool:
    """Return whether this wrapper process owns pre-launch filesystem checks.

    ``train.train`` creates the Accelerate process group after this wrapper has
    translated arguments.  Under ``torchrun``/``accelerate launch`` every rank
    therefore enters this function before Accelerate can coordinate them.  A
    non-zero rank must not reject a run merely because rank zero has already
    created its metadata directory.
    """

    rank = os.environ.get("RANK")
    if rank is None:
        rank = os.environ.get("LOCAL_RANK")
    return rank in (None, "", "0")


def _native_argv(
    *,
    config_path: Path,
    checkpoint_dir: Path,
    seed: int,
    max_steps: int,
    interval: int,
    config: dict[str, Any],
    cli: argparse.Namespace,
    extra: Sequence[str],
) -> list[str]:
    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("config.training must be a mapping")
    project_root = PROJECT_ROOT

    def asset_path(value: Any) -> Any:
        if value is None:
            return None
        candidate = Path(os.fspath(value)).expanduser()
        if not candidate.is_absolute() and (project_root / candidate).exists():
            return str((project_root / candidate).resolve())
        return value

    model_path = asset_path(cli.pretrained_model or config.get("pretrained_model_name_or_path"))
    text_path = asset_path(cli.pretrained_text or config.get("pretrained_text_encoder_name_or_path"))
    vision_path = asset_path(cli.pretrained_vision or config.get("pretrained_vision_encoder_name_or_path"))
    argv = [
        "--config_path",
        str(config_path),
        "--output_dir",
        str(checkpoint_dir),
        "--seed",
        str(seed),
        "--max_train_steps",
        str(max_steps),
        "--checkpointing_period",
        str(interval),
        "--train_batch_size",
        str(int(training.get("train_batch_size", 1))),
        "--sample_batch_size",
        str(int(training.get("eval_batch_size", 1))),
        "--gradient_accumulation_steps",
        str(int(training.get("gradient_accumulation_steps", 1))),
        "--dataloader_num_workers",
        str(int(training.get("dataloader_num_workers", 0))),
        "--learning_rate",
        str(float(training.get("learning_rate", 5e-6))),
        "--adam_weight_decay",
        str(float(training.get("weight_decay", 1e-2))),
        "--lr_warmup_steps",
        str(int(training.get("warmup_steps", 0))),
        "--mixed_precision",
        str(training.get("mixed_precision", "no")),
        "--precomp_lang_embed",
    ]
    if model_path:
        argv += ["--pretrained_model_name_or_path", str(model_path)]
    if text_path:
        argv += ["--pretrained_text_encoder_name_or_path", str(text_path)]
    if vision_path:
        argv += ["--pretrained_vision_encoder_name_or_path", str(vision_path)]
    if cli.resume_from_checkpoint:
        argv += ["--resume_from_checkpoint", str(cli.resume_from_checkpoint)]
    report_to = training.get("report_to")
    if report_to:
        argv += ["--report_to", str(report_to)]
    argv.extend(extra)
    return argv


def main(argv: Sequence[str] | None = None) -> int:
    cli, extra = build_parser().parse_known_args(argv)
    config_path = cli.config_path.expanduser().resolve()
    config = _yaml(config_path)
    data_root, artifact_dir, checkpoint_dir = _resolved_paths(config, config_path, cli)
    seed = int(cli.seed if cli.seed is not None else config.get("seed", 0))
    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("config.training must be a mapping")
    max_steps = int(cli.max_train_steps if cli.max_train_steps is not None else training.get("max_train_steps", 5))
    if cli.mode == "smoke":
        if cli.output_dir is None:
            artifact_dir = artifact_dir.parent / "smoke" / artifact_dir.name
        if cli.checkpoint_dir is None:
            checkpoint_dir = checkpoint_dir.parent / "smoke" / checkpoint_dir.name
        max_steps = 5
    elif cli.train_limit_per_map is not None or cli.eval_limit_per_map is not None:
        raise ValueError("per-map limits are only allowed for smoke")
    if max_steps < 5 or max_steps % 5:
        raise ValueError("max_train_steps must be a positive multiple of 5")
    interval = max_steps // 5
    resume = cli.resume_from_checkpoint is not None
    if not resume and _is_main_process() and (_occupied(artifact_dir) or _occupied(checkpoint_dir)):
        raise FileExistsError(
            f"Refusing to overwrite existing run; use a new --seed or --resume-from-checkpoint: {artifact_dir}"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    native_argv = _native_argv(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        max_steps=max_steps,
        interval=interval,
        config=config,
        cli=cli,
        extra=extra,
    )

    # This is the one and only training call.  Native ``train.train`` handles
    # the CSGO hook, optimizer, validation, checkpoint aliases and resume.
    import main as native_main
    import train.train as native_train
    from accelerate.logging import get_logger

    native_args = native_main.parse_args(native_argv)
    # Native extra arguments are accepted by design, but the benchmark
    # contract still requires five evenly spaced validation/save points.
    native_steps = int(native_args.max_train_steps)
    native_period = int(native_args.checkpointing_period)
    if native_steps < 5 or native_steps % 5 or native_period != native_steps // 5:
        raise ValueError(
            "native max_train_steps must be a multiple of 5 and "
            "checkpointing_period must equal max_train_steps / 5"
        )
    if cli.mode == "smoke" and (native_steps != 5 or native_period != 1):
        raise ValueError("smoke requires exactly five train steps and one step cadence")
    native_args.csgo_data_root = str(data_root)
    csgo = config.get("csgo", {})
    if not isinstance(csgo, dict):
        csgo = {}
    native_args.csgo_language_embeddings = cli.language_embeddings or config.get("language_embeddings") or csgo.get("language_embeddings")
    native_args.csgo_smoke = cli.mode == "smoke"
    native_args.csgo_train_limit_per_map = (
        cli.train_limit_per_map if cli.train_limit_per_map is not None else (1 if cli.mode == "smoke" else None)
    )
    native_args.csgo_eval_limit_per_map = (
        cli.eval_limit_per_map if cli.eval_limit_per_map is not None else (10 if cli.mode == "smoke" else None)
    )
    visualization = config.get("visualization", {})
    if not isinstance(visualization, dict):
        visualization = {}
    native_args.csgo_visualization_seed = int(
        cli.visualization_seed if cli.visualization_seed is not None else visualization.get("seed", 0)
    )
    native_args.csgo_output_dir = str(artifact_dir)
    native_args.csgo_cpu = bool(cli.cpu)
    native_args.project_root = str(PROJECT_ROOT)
    if native_args.report_to in ("none", "null", ""):
        native_args.report_to = None
    native_train.train(native_args, get_logger("train_seen10"))
    if cli.mode == "smoke":
        # Keep the provenance on each native checkpoint so a later formal
        # inference cannot accidentally consume a five-step smoke model.
        import json

        marker = json.dumps({"smoke_only": True, "seed": seed}, indent=2) + "\n"
        for checkpoint in checkpoint_dir.glob("checkpoint-*"):
            if checkpoint.is_dir():
                (checkpoint / "smoke_only.json").write_text(marker, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError, KeyError, OSError) as exc:
        print(f"train_seen10: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
