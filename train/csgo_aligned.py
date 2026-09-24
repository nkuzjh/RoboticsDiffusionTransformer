"""Isolated training path for the approved Seen-10 aligned RDT experiment.

An update always consumes one complete global batch.  Its sampler key is the
completed optimizer-update count, which is also the checkpoint resume key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, broadcast_object_list, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.csgo_seen10 import SEEN_MAPS, Seen10Dataset, collate_seen10
from data.csgo_update_sampler import AlignedUpdateSampler
from models.csgo_adaptation import build_role_adaptation
from train import csgo_hooks


FORMAL_STEPS = (4_000, 8_000, 12_000, 16_000, 19_500)
FORMAL_UPDATES = 19_500
FORMAL_BATCH = 128
FORMAL_TRAIN_ROWS = 50_000
FORMAL_VALIDATION_ROWS = 5_000
ASSET_MANIFEST = Path(__file__).resolve().parents[1] / ".cache/csgo_seen10/asset_manifest.json"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _checkpoint_steps(config: Mapping[str, Any], total_updates: int, smoke: bool) -> tuple[int, ...]:
    training = _mapping(config.get("training"))
    configured = training.get("checkpoint_steps")
    if smoke:
        steps = tuple(range(1, total_updates + 1))
    elif configured is not None:
        steps = tuple(int(value) for value in configured)
    else:
        interval = int(training.get("checkpoint_interval_updates", 4_000))
        steps = tuple(range(interval, total_updates, interval)) + (total_updates,)
    if not steps or sorted(set(steps)) != list(steps) or steps[-1] != total_updates:
        raise ValueError("checkpoint_steps must be increasing, unique, and end at max_train_steps")
    if not smoke and steps != FORMAL_STEPS:
        raise ValueError(f"Aligned checkpoint steps must be {FORMAL_STEPS}, got {steps}")
    return steps


def _lr_factor(completed_updates: int, *, total_updates: int, warmup_updates: int, min_ratio: float) -> float:
    """Factor used *after* the stated count of successful optimizer updates."""

    if completed_updates < warmup_updates:
        return float(completed_updates + 1) / warmup_updates
    progress = min(1.0, (completed_updates - warmup_updates) / (total_updates - warmup_updates))
    return min_ratio + (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_identity(args: Any, *, smoke: bool) -> dict[str, Any]:
    """Record exact encoder/base origins; verify the official RDT base bytes."""

    if not ASSET_MANIFEST.is_file():
        if smoke:
            return {"asset_manifest": None}
        raise FileNotFoundError(f"Official asset manifest is missing: {ASSET_MANIFEST}")
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in manifest["assets"]}
    arguments = (
        ("rdt-1b", "pretrained_model_name_or_path"),
        ("siglip-so400m-patch14-384", "pretrained_vision_encoder_name_or_path"),
        ("t5-v1_1-xxl", "pretrained_text_encoder_name_or_path"),
    )
    result: dict[str, Any] = {"asset_manifest_sha256": _sha256_file(ASSET_MANIFEST)}
    for name, argument in arguments:
        configured = getattr(args, argument, None)
        if not configured:
            raise ValueError(f"Aligned run requires {argument}")
        source = Path(os.fspath(configured)).expanduser().resolve()
        asset = by_name[name]
        if smoke:
            weight_files = sorted(source.glob("*.safetensors")) + sorted(source.glob("*.bin"))
            result[name] = {
                "path": str(source), "kind": "synthetic_acceptance_fixture",
                "files": [{"name": path.name, "size": path.stat().st_size,
                           "sha256": _sha256_file(path)} for path in weight_files],
            }
            continue
        official = Path(asset["local_dir"]).resolve()
        if not smoke and source != official:
            raise ValueError(f"Aligned {name} must use the audited official asset: {official}")
        weight_entry = next((entry for entry in asset["files"] if entry["path"] in
                             ("pytorch_model.bin", "model.safetensors")), None)
        if weight_entry is None:
            raise ValueError(f"No weight file recorded for {name}")
        weight = source / weight_entry["path"]
        if not smoke and (not weight.is_file() or weight.stat().st_size != weight_entry["size"]):
            raise ValueError(f"Official {name} weight file is absent or has changed size: {weight}")
        result[name] = {
            "path": str(source), "revision": asset["revision"],
            "weight_file": weight_entry["path"], "expected_sha256": weight_entry.get("sha256"),
            "weight_size": weight_entry["size"],
        }
        if name == "rdt-1b" and not smoke:
            actual = _sha256_file(weight)
            if actual != weight_entry["sha256"]:
                raise ValueError(f"Official RDT base SHA256 differs: {actual}")
            result[name]["actual_sha256"] = actual
    return result


def _run_contract(args: Any, config: Mapping[str, Any], train_rows: int, validation_rows: int,
                  world_size: int, batch_size: int, steps: tuple[int, ...],
                  assets: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(os.fspath(args.csgo_data_root)).expanduser().resolve()
    manifest = root / "benchmark_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Seen-10 manifest is missing: {manifest}")
    csgo = _mapping(config.get("csgo"))
    training = _mapping(config.get("training"))
    # The full YAML fingerprint catches changes to LoRA, data, evaluation,
    # optimizer and preprocessing settings.  Runtime overrides are recorded
    # separately because the YAML alone does not contain the effective run.
    return {
        "profile": config.get("profile"),
        "config_sha256": _contract_digest(config),
        "assets": assets,
        "language_embeddings_sha256": _sha256_file(Path(args.csgo_language_embeddings)),
        "data_root": str(root),
        "seed": int(args.seed),
        "manifest_sha256": _sha256_file(manifest),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "world_size": world_size,
        "microbatch_size": int(args.train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "global_batch_size": batch_size,
        "max_train_steps": int(args.max_train_steps),
        "checkpoint_steps": list(steps),
        "csgo": csgo,
        "training": training,
        "effective_optimizer": {
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.adam_weight_decay),
            "betas": [0.9, 0.999], "eps": 1e-8,
            "max_grad_norm": float(args.max_grad_norm),
            "mixed_precision": str(args.mixed_precision),
        },
    }


def _contract_digest(contract: Mapping[str, Any]) -> str:
    value = json.dumps(contract, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _resolve_resume(root: Path, requested: str | os.PathLike[str] | None) -> Path | None:
    if not requested:
        return None
    if os.fspath(requested) == "latest":
        candidates = [
            path for path in root.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
            and (path / "training_state.json").is_file()
        ]
        if not candidates:
            raise FileNotFoundError(f"No complete aligned checkpoint found under {root}")
        return max(candidates, key=lambda path: int(path.name.rsplit("-", 1)[1])).resolve()
    candidate = Path(requested).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_dir() or not (candidate / "training_state.json").is_file():
        raise FileNotFoundError(f"Aligned resume requires a complete checkpoint: {candidate}")
    return candidate


def _actual_optimizer_audit(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    seen: set[int] = set()
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        members = list(group["params"])
        ids = [id(parameter) for parameter in members]
        if any(identifier in seen for identifier in ids):
            raise RuntimeError("Optimizer contains duplicate trainable parameters")
        seen.update(ids)
        if any(not parameter.requires_grad or parameter.dtype != torch.float32 for parameter in members):
            raise RuntimeError("Aligned optimizer members must be trainable FP32 parameters")
        groups.append({
            "name": group.get("name"), "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "num_parameters": sum(parameter.numel() for parameter in members),
            "parameter_names": [names[id(parameter)] for parameter in members],
            "dtypes": sorted({str(parameter.dtype) for parameter in members}),
        })
    required = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != required:
        raise RuntimeError("Optimizer does not cover exactly the aligned trainable parameters")
    return {"groups": groups, "trainable_parameters": sum(group["num_parameters"] for group in groups)}


def _validate_formal(args: Any, config: Mapping[str, Any], *, smoke: bool, global_batch: int) -> None:
    training = _mapping(config.get("training"))
    csgo = _mapping(config.get("csgo"))
    if smoke:
        return
    if (getattr(args, "csgo_language_embeddings", None) is not None
            or config.get("language_embeddings") is not None
            or csgo.get("language_embeddings") is not None):
        raise ValueError("Aligned formal training builds its map-text cache from the audited T5; external embedding overrides are not accepted")
    expected = {
        "seed": (int(args.seed), 42),
        "max_train_steps": (int(args.max_train_steps), FORMAL_UPDATES),
        "global_batch_size": (global_batch, FORMAL_BATCH),
        "learning_rate": (float(args.learning_rate), 1e-4),
        "weight_decay": (float(args.adam_weight_decay), 0.0),
        "mixed_precision": (str(args.mixed_precision), "bf16"),
        "warmup_updates": (int(training.get("warmup_updates", -1)), 59),
        "min_lr": (float(training.get("min_lr", -1)), 1e-5),
        "adaptation_mode": (training.get("adaptation_mode"), "role_lora"),
        "scheduler": (training.get("scheduler"), "cosine_with_min_lr"),
        "effective_localization_batch_size": (int(training.get("effective_localization_batch_size", -1)), FORMAL_BATCH),
        "trainable_parameter_dtype": (training.get("trainable_parameter_dtype"), "float32"),
        "adam_beta1": (float(args.adam_beta1), 0.9),
        "adam_beta2": (float(args.adam_beta2), 0.999),
        "adam_epsilon": (float(args.adam_epsilon), 1e-8),
        "diffusion_channel_policy": (csgo.get("diffusion_channel_policy"), "native_full"),
        "external_action_dim": (int(csgo.get("external_action_dim", -1)), 5),
        "declared_effective_batch": (int(training.get("effective_localization_batch_size", -1)), 128),
        "checkpoint_interval": (int(training.get("checkpoint_interval_updates", -1)), 4000),
        "validation_interval": (int(training.get("validation_interval_updates", -1)), 4000),
        "runtime_checkpoint_interval": (int(args.checkpointing_period), 4000),
        "runtime_warmup": (int(args.lr_warmup_steps), 59),
    }
    wrong = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
    if wrong:
        raise ValueError(f"Aligned formal run settings differ from approved protocol: {wrong}")
    if getattr(args, "scale_lr", False) or getattr(args, "use_8bit_adam", False) or getattr(args, "gradient_checkpointing", False):
        raise ValueError("Aligned formal run requires unscaled LR and FP32 AdamW moments")
    if getattr(args, "csgo_train_limit_per_map", None) is not None or getattr(args, "csgo_eval_limit_per_map", None) is not None:
        raise ValueError("Aligned formal run does not permit per-map sample limits")
    if getattr(args, "deepspeed", None) is not None:
        raise ValueError("Aligned DeepSpeed optimizer semantics have not been validated")
    if float(getattr(args, "max_grad_norm", 0)) != 1.0:
        raise ValueError("Aligned formal run requires max_grad_norm=1.0")
    if (csgo.get("train_split"), csgo.get("validation_split"), csgo.get("inference_split")) != (
        "seen_train", "seen_validation", "seen_discrete_test",
    ):
        raise ValueError("Aligned run requires the published Seen-10 splits")
    if tuple(csgo.get("maps", ())) != SEEN_MAPS:
        raise ValueError("Aligned run requires exactly the published Seen-10 map order")
    if csgo.get("action_order") != ["x", "y", "z", "pitch", "yaw"]:
        raise ValueError("Aligned external pose order must be x,y,z,pitch,yaw")
    if training.get("lora") != {"r": 32, "alpha": 64, "dropout": 0.05, "bias": "none"}:
        raise ValueError("Aligned LoRA settings must be r32/alpha64/dropout0.05/biasnone")
    common = _mapping(config.get("common"))
    if any(common.get(key) != value for key, value in {
        "num_cameras": 2, "img_history_size": 1, "action_chunk_size": 1, "state_dim": 128,
    }.items()):
        raise ValueError("Aligned inputs require two current views, horizon one and state width128")
    if csgo.get("normalization") != "benchmark_evaluator_v2" or float(csgo.get("z_denominator_epsilon", -1)) != 0:
        raise ValueError("Aligned run requires benchmark evaluator pose normalization")
    if int(csgo.get("internal_action_dim", -1)) != 128 or int(csgo.get("horizon", -1)) != 1:
        raise ValueError("Aligned run requires 128D internal action and one-step horizon")
    if csgo.get("state_mode") != "zero_slot" or not csgo.get("state_is_zero"):
        raise ValueError("Aligned run requires the zero state slot")
    augmentation = _mapping(csgo.get("augmentation"))
    expected_aug = {
        "policy": "rdt_native_image_v1", "train_only": True,
        "views": ["fpv", "radar"], "per_view_probability": 0.5,
        "independent_views": True, "geometry": "none",
        "auto_adjust_image_brightness": False, "state_noise_snr": None,
        "condition_dropout_probability": 0.0, "rng_mode": "sample_occurrence",
    }
    if any(augmentation.get(key) != value for key, value in expected_aug.items()):
        raise ValueError("Aligned image augmentation differs from approved native RDT policy")
    model = _mapping(config.get("model"))
    noise = _mapping(model.get("noise_scheduler"))
    if (noise.get("prediction_type"), noise.get("clip_sample"), noise.get("num_inference_timesteps")) != (
        "sample", False, 5,
    ):
        raise ValueError("Aligned diffusion scheduler differs from approved native settings")


def train_aligned(args: Any, config: Mapping[str, Any], logger: logging.Logger) -> None:
    """Run the aligned experiment and save all five native checkpoint states."""

    smoke = bool(getattr(args, "csgo_smoke", False))
    train_options = _mapping(config.get("training"))
    csgo_options = _mapping(config.get("csgo"))
    if not csgo_options:
        raise ValueError("Aligned profile requires csgo settings")
    if not getattr(args, "csgo_data_root", None):
        setattr(args, "csgo_data_root", config.get("data_root") or csgo_options.get("data_root"))
    if not args.csgo_data_root:
        raise ValueError("Aligned Seen-10 data root is required")
    if int(args.max_train_steps) <= 0:
        raise ValueError("max_train_steps must be positive")
    checkpoint_steps = _checkpoint_steps(config, int(args.max_train_steps), smoke)
    output_root = Path(args.output_dir).expanduser().resolve()
    artifact_root = Path(getattr(args, "csgo_output_dir", None) or output_root).expanduser().resolve()
    accelerator = Accelerator(
        cpu=bool(getattr(args, "csgo_cpu", False)),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        mixed_precision=str(args.mixed_precision),
        # Other ranks wait while rank zero runs all 5,000 validation rows.
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=4))],
        project_config=ProjectConfiguration(project_dir=str(output_root), total_limit=None),
    )
    if args.seed is None:
        raise ValueError("Aligned experiment requires an explicit seed")
    set_seed(int(args.seed), device_specific=False)
    global_batch = int(args.train_batch_size) * int(args.gradient_accumulation_steps) * accelerator.num_processes
    _validate_formal(args, config, smoke=smoke, global_batch=global_batch)
    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    assets = _asset_identity(args, smoke=smoke)

    # Official base loading precedes adaptation.  The frozen encoders remain
    # outside the optimizer and are used only under no_grad in the hook.
    components = csgo_hooks.build_csgo_components(args, config, accelerator, logger)
    model = components["rdt"]
    if getattr(model, "diffusion_channel_policy", None) != "native_full":
        raise RuntimeError("Aligned runner was not constructed in native_full mode")
    adaptation = build_role_adaptation(model, config)
    if not smoke and adaptation["audit"]["trainable_parameters"] != 73_164_928:
        raise RuntimeError("Official aligned trainable parameter count changed")
    optimizer = torch.optim.AdamW(
        adaptation["optimizer_groups"],
        lr=float(args.learning_rate),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    optimizer_audit = _actual_optimizer_audit(model, optimizer)
    warmup = int(train_options.get("warmup_updates", 59))
    min_lr = float(train_options.get("min_lr", 1e-5))
    if warmup <= 0 or warmup >= int(args.max_train_steps) or min_lr <= 0:
        if not smoke:
            raise ValueError("Invalid aligned warmup or minimum LR")
        warmup = min(1, int(args.max_train_steps) - 1)
    lr_ratio = min_lr / float(args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_factor(
            step, total_updates=int(args.max_train_steps), warmup_updates=warmup, min_ratio=lr_ratio,
        ),
    )

    language_embeddings = components["language_embeddings"]
    data_kwargs = dict(
        image_processor=components["image_processor"],
        language_embeddings=language_embeddings,
        tokenizer=components["tokenizer"],
        state_dim=128,
    )
    augmentation = _mapping(csgo_options.get("augmentation"))
    augmentation["seed"] = int(args.seed)
    train_dataset = Seen10Dataset(
        args.csgo_data_root, "seen_train", limit_per_map=getattr(args, "csgo_train_limit_per_map", None),
        augmentation=augmentation, external_action_dim=5, **data_kwargs,
    )
    validation_dataset = Seen10Dataset(
        args.csgo_data_root, "seen_validation", limit_per_map=getattr(args, "csgo_eval_limit_per_map", None),
        augmentation=None, external_action_dim=5, **data_kwargs,
    )
    if not smoke and (len(train_dataset), len(validation_dataset)) != (FORMAL_TRAIN_ROWS, FORMAL_VALIDATION_ROWS):
        raise ValueError(f"Aligned split sizes changed: train={len(train_dataset)}, validation={len(validation_dataset)}")
    sampler = AlignedUpdateSampler(
        len(train_dataset), seed=int(args.seed), world_size=accelerator.num_processes,
        rank=accelerator.process_index, microbatch_size=int(args.train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        total_updates=int(args.max_train_steps), required_global_batch_size=None if smoke else FORMAL_BATCH,
    )
    loader_rng = torch.Generator(device="cpu").manual_seed(int(args.seed) + 11)
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, collate_fn=collate_seen10,
        num_workers=int(args.dataloader_num_workers),
        persistent_workers=False, pin_memory=True, generator=loader_rng,
    )
    validation_loader = None
    if accelerator.is_main_process:
        validation_rng = torch.Generator(device="cpu").manual_seed(int(args.seed) + 12)
        validation_loader = DataLoader(
            validation_dataset, batch_size=int(args.sample_batch_size), shuffle=False,
            collate_fn=collate_seen10, num_workers=int(args.dataloader_num_workers),
            persistent_workers=False, pin_memory=True, generator=validation_rng,
        )

    contract = _run_contract(
        args, config, len(train_dataset), len(validation_dataset), accelerator.num_processes,
        global_batch, checkpoint_steps, assets,
    )
    contract_hash = _contract_digest(contract)

    # Accelerator owns the model and optimizer state files.  The small config
    # is sufficient for CSGORDTRunner.from_pretrained to load that same model
    # file without a second multi-GB copy per checkpoint.
    def save_model_config(_models, _weights, directory):
        if accelerator.is_main_process:
            csgo_hooks.write_run_metadata(
                Path(directory) / "config.json", accelerator.unwrap_model(model)._hub_mixin_config,
            )

    accelerator.register_save_state_pre_hook(save_model_config)
    model = accelerator.prepare_model(model)
    optimizer = accelerator.prepare_optimizer(optimizer)
    accelerator.register_for_checkpointing(scheduler)

    resume = _resolve_resume(output_root, getattr(args, "resume_from_checkpoint", None))
    global_step = 0
    if resume is not None:
        saved = json.loads((resume / "training_state.json").read_text(encoding="utf-8"))
        global_step = int(saved["global_step"])
        if saved["contract_sha256"] != contract_hash or int(saved["sampler_start_update"]) != global_step:
            raise ValueError("Resume contract or sampler position differs from saved aligned experiment")
        if int(resume.name.rsplit("-", 1)[1]) != global_step or global_step not in checkpoint_steps:
            raise ValueError("Checkpoint name, saved update and approved save schedule disagree")
        later = [step for step in checkpoint_steps if step > global_step and
                 (output_root / f"checkpoint-{step}" / "training_state.json").is_file()]
        if later:
            raise ValueError(f"Resume from latest complete checkpoint; later checkpoints exist: {later}")
        accelerator.load_state(str(resume))
    sampler.set_start_update(global_step)
    if accelerator.is_main_process:
        metadata_path = artifact_root / "training_metadata.json"
        if resume is None:
            if metadata_path.exists():
                raise FileExistsError(f"Aligned run metadata already exists: {metadata_path}")
            csgo_hooks.write_run_metadata(metadata_path, {
                "task": "csgo_seen10_localization", "profile": config["profile"],
                "status": "running", "smoke_only": smoke, "contract": contract,
                "contract_sha256": contract_hash, "adaptation": adaptation["audit"],
                "optimizer": optimizer_audit, "raw_config": config,
                "raw_cli": getattr(args, "csgo_invocation", None), "resolved_args": vars(args),
                "validation_metric": "normalized_valid5_mse_5step_inference",
                "validation_rows": len(validation_dataset),
            })
        else:
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Run metadata is missing during resume: {metadata_path}")
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("contract_sha256") != contract_hash:
                raise ValueError("Resume run metadata contract differs from checkpoint")
    best_record = artifact_root / "best_validation.json"
    best_mse = math.inf
    best_path: Path | None = None
    if best_record.is_file():
        record = json.loads(best_record.read_text(encoding="utf-8"))
        best_mse = float(record["mse"])
        best_path = Path(record["checkpoint"])

    def validate_checkpoint(step: int, checkpoint: Path) -> None:
        nonlocal best_mse, best_path
        if not accelerator.is_main_process:
            return
        assert validation_loader is not None
        evaluation_dir = artifact_root / "validation" / f"step-{step}"
        metric_file = evaluation_dir / "metrics.json"
        if metric_file.is_file():
            metrics = json.loads(metric_file.read_text(encoding="utf-8"))
        else:
            metrics = csgo_hooks.evaluate_localization(
                accelerator.unwrap_model(model), validation_loader,
                vision_encoder=components["vision_encoder"], text_encoder=components["text_encoder"],
                accelerator=accelerator, dtype=components["weight_dtype"],
                output_dir=evaluation_dir, dataset=validation_dataset, step=step,
                seed=int(args.seed), per_map=int(_mapping(config.get("visualization")).get("samples_per_map", 10)),
                render=bool(_mapping(config.get("visualization")).get("enabled", True)),
            )
            metrics.update({"step": step, "checkpoint": str(checkpoint),
                            "metric": "normalized_valid5_mse_5step_inference"})
            if not smoke and int(metrics["count"]) != FORMAL_VALIDATION_ROWS:
                raise RuntimeError(f"Incomplete aligned validation at step {step}: {metrics['count']}")
            csgo_hooks.write_run_metadata(metric_file, metrics)
        if float(metrics["mse"]) < best_mse:
            best_mse = float(metrics["mse"])
            best_path = checkpoint
            csgo_hooks.write_run_metadata(best_record, {
                "mse": best_mse, "checkpoint": str(best_path), "step": step,
                "metric": "normalized_valid5_mse_5step_inference",
            })
        csgo_hooks.update_checkpoint_alias(output_root, "late", checkpoint)
        if best_path is None or not best_path.is_dir():
            raise RuntimeError("Aligned best checkpoint is missing")
        csgo_hooks.update_checkpoint_alias(output_root, "best", best_path)

    if resume is not None:
        accelerator.wait_for_everyone()
        validate_checkpoint(global_step, resume)
        accelerator.wait_for_everyone()

    logger.info(
        "Aligned Seen-10: %d rows, %d validation rows, global batch %d, %d updates, save steps %s, resume %d",
        len(train_dataset), len(validation_dataset), global_batch, args.max_train_steps,
        checkpoint_steps, global_step,
    )
    loss_path = artifact_root / "train_loss.jsonl"
    bar = tqdm(total=int(args.max_train_steps), initial=global_step,
               disable=not accelerator.is_local_main_process, desc="Aligned updates")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    microbatches_since_update = 0
    update_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    for batch in train_loader:
        with accelerator.accumulate(model):
            prepared = csgo_hooks.prepare_csgo_batch(
                batch, vision_encoder=components["vision_encoder"], text_encoder=components["text_encoder"],
                accelerator=accelerator, dtype=components["weight_dtype"], training=True, aligned=True,
            )
            loss = model(**prepared)
            if not bool(torch.isfinite(loss.detach()).all().item()):
                raise FloatingPointError(f"Non-finite aligned loss before update {global_step + 1}")
            update_loss_sum += loss.detach().float()
            accelerator.backward(loss)
            microbatches_since_update += 1
            if accelerator.sync_gradients:
                if microbatches_since_update != int(args.gradient_accumulation_steps):
                    raise RuntimeError("Aligned update did not consume exactly the configured microbatch count")
                accelerator.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=1.0,
                )
                optimizer.step()
                if accelerator.optimizer_step_was_skipped:
                    raise FloatingPointError("Aligned optimizer skipped an update; sample budget is no longer exact")
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                microbatches_since_update = 0
                bar.update(1)
                mean_update_loss = accelerator.reduce(update_loss_sum, reduction="sum")
                mean_update_loss /= accelerator.num_processes * int(args.gradient_accumulation_steps)
                update_loss_sum.zero_()
                if accelerator.is_main_process:
                    csgo_hooks.append_loss_jsonl(
                        loss_path, step=global_step, loss=float(mean_update_loss.item()),
                        lr=float(scheduler.get_last_lr()[0]),
                    )
        if global_step in checkpoint_steps and microbatches_since_update == 0:
            checkpoint = output_root / f"checkpoint-{global_step}"
            if checkpoint.exists():
                raise FileExistsError(f"Refusing to overwrite aligned checkpoint: {checkpoint}")
            staging_names = [f".checkpoint-{global_step}.{uuid.uuid4().hex}.saving"
                             if accelerator.is_main_process else None]
            broadcast_object_list(staging_names)
            staging = output_root / staging_names[0]
            accelerator.wait_for_everyone()
            accelerator.save_state(str(staging))
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if smoke:
                    csgo_hooks.write_run_metadata(staging / "smoke_only.json", {
                        "smoke_only": True, "seed": int(args.seed),
                    })
                csgo_hooks.write_run_metadata(staging / "training_state.json", {
                    "global_step": global_step, "sampler_start_update": global_step,
                    "global_samples_exposed": global_step * global_batch,
                    "contract_sha256": contract_hash,
                    "scheduler_last_epoch": scheduler.last_epoch,
                    "optimizer_step": global_step,
                })
                # Readers and `latest` only see a checkpoint after all rank
                # state files and completion metadata have been written.
                os.replace(staging, checkpoint)
            accelerator.wait_for_everyone()
            validate_checkpoint(global_step, checkpoint)
            accelerator.wait_for_everyone()
            model.train()
        if global_step >= int(args.max_train_steps):
            break

    bar.close()
    if global_step != int(args.max_train_steps) or microbatches_since_update:
        raise RuntimeError(f"Aligned run ended at update {global_step} with {microbatches_since_update} pending microbatches")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metadata_path = artifact_root / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = "complete"
        metadata["completed_updates"] = global_step
        metadata["localization_sample_exposures"] = global_step * global_batch
        metadata["best_checkpoint"] = str(best_path) if best_path is not None else None
        metadata["late_checkpoint"] = str(output_root / f"checkpoint-{global_step}")
        csgo_hooks.write_run_metadata(metadata_path, metadata)
        csgo_hooks.plot_loss_curve(loss_path, artifact_root / "loss_curve.svg")
    accelerator.end_training()


__all__ = ["train_aligned", "_checkpoint_steps", "_lr_factor", "_resolve_resume"]
