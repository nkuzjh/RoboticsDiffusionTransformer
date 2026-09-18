#!/usr/bin/env python3
"""Build a deterministic local CSGO smoke fixture from real model classes.

The fixture contains a small T5 encoder/tokenizer, SigLIP vision encoder and
CSGO RDT checkpoint.  It is intended for CPU smoke runs and never downloads
weights.  The output directory must be new; existing model assets are never
updated in place.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))
DEFAULT_OUTPUT = PROJECT_ROOT / ".cache" / "csgo_seen10" / "tiny_fixture"
DEFAULT_DATA_ROOT = Path(
    os.environ.get("CSGO_DATA_ROOT", "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2")
)
DEFAULT_SHARED_EVAL = Path(
    os.environ.get("SHARED_EVAL_DIR", "/home/jiahao/task/csgo_benchmark_v2_eval_general")
)
MAPS = (
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New fixture directory (refuses any existing path)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--shared-eval-dir", type=Path, default=DEFAULT_SHARED_EVAL)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _project_path(path: Path) -> str:
    """Use a portable project relative path when the output is in this repo."""

    try:
        return os.fspath(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return os.fspath(path)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_tokenizer(t5_dir: Path, seed: int) -> int:
    import sentencepiece as spm
    from transformers import T5Tokenizer

    corpus = t5_dir / "corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "Localize the player using the first person image and radar map.",
                "Predict absolute normalized x y z pitch and yaw.",
                *(f"Localize the player in {name}." for name in MAPS),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prefix = t5_dir / "spiece"
    spm.SentencePieceTrainer.Train(
        input=os.fspath(corpus),
        model_prefix=os.fspath(prefix),
        vocab_size=64,
        model_type="unigram",
        character_coverage=1.0,
        bos_id=-1,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        hard_vocab_limit=False,
        minloglevel=2,
        seed_sentencepiece_size=1000,
    )
    corpus.unlink()
    vocab = prefix.with_suffix(".vocab")
    if vocab.exists():
        vocab.unlink()
    tokenizer = T5Tokenizer(os.fspath(prefix.with_suffix(".model")), extra_ids=0)
    tokenizer.save_pretrained(t5_dir)
    return int(tokenizer.vocab_size)


def _build_t5(t5_dir: Path, seed: int) -> None:
    from transformers import T5Config, T5EncoderModel

    vocab_size = _build_tokenizer(t5_dir, seed)
    torch.manual_seed(seed)
    config = T5Config(
        vocab_size=vocab_size,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        dropout_rate=0.1,
        feed_forward_proj="relu",
        relative_attention_num_buckets=8,
        use_cache=False,
    )
    T5EncoderModel(config).save_pretrained(t5_dir, safe_serialization=True)


def _build_siglip(siglip_dir: Path, seed: int) -> None:
    from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

    torch.manual_seed(seed + 1)
    config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=8,
        patch_size=4,
        num_channels=3,
    )
    SiglipVisionModel(config).save_pretrained(siglip_dir, safe_serialization=True)
    processor = SiglipImageProcessor(
        do_resize=True,
        size={"height": 8, "width": 8},
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    processor.save_pretrained(siglip_dir)


def _tiny_config(
    *,
    output: Path,
    data_root: Path,
    shared_eval_dir: Path,
) -> dict[str, Any]:
    rdt = output / "rdt"
    t5 = output / "t5"
    siglip = output / "siglip"
    return {
        "schema_version": 1,
        "model_name": "RDT",
        "model_type": "VLA",
        "task": "localization",
        "seed": 0,
        "data_root": os.fspath(data_root.expanduser().resolve()),
        "shared_eval_dir": os.fspath(shared_eval_dir.expanduser().resolve()),
        "unilip_python": "/home/jiahao/miniconda3/envs/UniLIP/bin/python",
        "pretrained_model_name_or_path": _project_path(rdt),
        "pretrained_text_encoder_name_or_path": _project_path(t5),
        "pretrained_vision_encoder_name_or_path": _project_path(siglip),
        "output_root": "outputs/csgo_benchmark_v2_seen10",
        "checkpoint_root": "checkpoints/csgo_benchmark_v2_seen10",
        "common": {
            "img_history_size": 1,
            "action_chunk_size": 1,
            "num_cameras": 2,
            "state_dim": 128,
        },
        "dataset": {
            "buf_path": "/path/to/buffer",
            "buf_num_chunks": 512,
            "buf_chunk_size": 512,
            "epsd_len_thresh_low": 32,
            "epsd_len_thresh_high": 2048,
            "image_aspect_ratio": "pad",
            # Keep the whole fixed instruction, including the map name.  A
            # four-token smoke setting can truncate every map to the same
            # prefix and erase the only language distinction.
            "tokenizer_max_length": 128,
        },
        "model": {
            "lang_adaptor": "mlp2x_gelu",
            "img_adaptor": "mlp2x_gelu",
            "state_adaptor": "mlp3x_gelu",
            "lang_token_dim": 16,
            "img_token_dim": 16,
            "state_token_dim": 128,
            "rdt": {"hidden_size": 64, "depth": 2, "num_heads": 4, "cond_pos_embed_type": "multimodal"},
            "noise_scheduler": {
                "type": "ddpm",
                "num_train_timesteps": 16,
                "num_inference_timesteps": 2,
                "beta_schedule": "squaredcos_cap_v2",
                "prediction_type": "sample",
                "clip_sample": False,
            },
            "ema": {
                "update_after_step": 0,
                "inv_gamma": 1.0,
                "power": 0.75,
                "min_value": 0.0,
                "max_value": 0.9999,
            },
        },
        "csgo": {
            "maps": list(MAPS),
            "train_split": "seen_train",
            "validation_split": "seen_validation",
            "inference_split": "seen_discrete_test",
            "horizon": 1,
            "num_cameras": 2,
            "state_dim": 128,
            "action_dim": 128,
            "active_action_dim": 5,
            "action_order": ["x", "y", "z", "pitch", "yaw"],
            "prediction_pose_space": "normalized",
            "state_is_zero": True,
            "state_is_masked": True,
            "language_embeddings": None,
        },
        "training": {
            "max_train_steps": 5,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "warmup_steps": 0,
            "dataloader_num_workers": 0,
            "mixed_precision": "no",
            "validation_passes": 5,
            "save_passes": 5,
            "map_sampling": "equal_map",
            "report_to": "none",
        },
        "visualization": {"enabled": True, "samples_per_map": 10, "seed": 0},
        "outputs": {
            "localization_predictions": "localization/predictions.jsonl",
            "visualization": "localization/visualization",
            "loss_curve": "loss_curve.svg",
            "evaluation": "evaluation/localization",
        },
    }


def _build_fixture(output: Path, data_root: Path, shared_eval_dir: Path, seed: int) -> None:
    from models.csgo_rdt import build_csgo_rdt

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing fixture directory: {output}")
    temporary_name = tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        rdt_dir = temporary / "rdt"
        t5_dir = temporary / "t5"
        siglip_dir = temporary / "siglip"
        rdt_dir.mkdir()
        t5_dir.mkdir()
        siglip_dir.mkdir()
        _build_t5(t5_dir, seed)
        _build_siglip(siglip_dir, seed)

        native_config = _tiny_config(output=temporary, data_root=data_root, shared_eval_dir=shared_eval_dir)
        torch.manual_seed(seed + 2)
        rdt = build_csgo_rdt(native_config, img_cond_len=8, dtype=torch.float32)
        rdt.save_pretrained(rdt_dir)

        # The config is generated after all model directories exist so its
        # relative paths point at the final output location.
        final_config = _tiny_config(output=output, data_root=data_root, shared_eval_dir=shared_eval_dir)
        import yaml

        (temporary / "config.yaml").write_text(
            yaml.safe_dump(final_config, sort_keys=False), encoding="utf-8"
        )
        _write_json(
            temporary / "fixture_manifest.json",
            {
                "kind": "csgo_seen10_tiny_fixture",
                "seed": int(seed),
                "architecture": {"rdt": "CSGORDTRunner", "vision": "SiglipVisionModel", "text": "T5EncoderModel"},
                "files": ["config.yaml", "rdt", "siglip", "t5"],
            },
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    cli = _parser().parse_args(argv)
    output = cli.output.expanduser().resolve()
    _build_fixture(
        output,
        cli.data_root.expanduser().resolve(),
        cli.shared_eval_dir.expanduser().resolve(),
        int(cli.seed),
    )
    print(json.dumps({"fixture": os.fspath(output), "config": os.fspath(output / "config.yaml")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"make_csgo_tiny_fixture: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
