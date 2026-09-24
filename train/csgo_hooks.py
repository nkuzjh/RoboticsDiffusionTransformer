"""Native Accelerator hooks for CSGO Seen-10 localization.

This module is deliberately small at the integration boundary: ``train.py``
keeps the native optimizer, scheduler, accelerator state, and training loop
when the CSGO opt-in flag is present.  These helpers own only benchmark data
preparation, masked RDT calls, validation, and run bookkeeping.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from models.csgo_rdt import CSGORDTRunner, build_csgo_rdt
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from train.csgo_visualize import render_localization


LOGGER = logging.getLogger(__name__)
ACTIVE_ACTION_DIM = 5
ACTION_DIM = 128
CSGO_NUM_CAMERAS = 2
CSGO_HISTORY = 1


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _value(args: Any, config: Mapping[str, Any], name: str, default: Any = None) -> Any:
    value = getattr(args, name, None) if args is not None else None
    if value is not None:
        return value
    if name in config:
        return config[name]
    training = config.get("training")
    if isinstance(training, Mapping) and name in training:
        return training[name]
    return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    import yaml

    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return _mapping(value)


def _native_config(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    """Return the base RDT YAML when a benchmark-only config was supplied."""

    if isinstance(config.get("model"), Mapping) and isinstance(config.get("common"), Mapping):
        return dict(config)
    candidates: list[Path] = []
    config_path = _value(args, config, "config_path")
    if config_path:
        path = Path(config_path).expanduser()
        candidates.append(path)
        candidates.append(path.parent / "base.yaml")
        candidates.append(path.parent.parent / "configs" / "base.yaml")
    candidates.append(Path("configs/base.yaml"))
    for candidate in candidates:
        if candidate.is_file():
            loaded = _load_yaml(candidate)
            if isinstance(loaded.get("model"), Mapping) and isinstance(loaded.get("common"), Mapping):
                # Benchmark overrides can still carry a nested model/common.
                merged = dict(loaded)
                for key in ("model", "common", "dataset"):
                    if key in config and isinstance(config[key], Mapping):
                        merged[key] = dict(config[key])
                return merged
    raise FileNotFoundError("Could not locate an RDT config with common/model/dataset sections")


def _weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _build_text_encoder(
    model_path: str | os.PathLike[str] | None,
    *,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Build tokenizer/T5, including local tiny T5 models used by smoke tests."""

    if not model_path:
        raise ValueError("pretrained_text_encoder_name_or_path is required without CSGO language embeddings")
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=max_length)
    try:
        text_encoder = T5EncoderModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
    except TypeError:
        text_encoder = T5EncoderModel.from_pretrained(model_path)
    text_encoder.eval().to(device=device, dtype=dtype)
    return tokenizer, text_encoder


def _precompute_map_language_embeddings(
    *,
    model_path: str | os.PathLike[str] | None,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
    cache_path: str | os.PathLike[str] | None,
) -> dict[str, torch.Tensor]:
    """Encode the fixed ten map instructions once for ``--precomp_lang_embed``."""

    from data.csgo_seen10 import SEEN_MAPS, instruction_for_map

    cache_path = Path(cache_path).expanduser() if cache_path is not None else None
    if cache_path is not None:
        if cache_path.is_file():
            value = torch.load(cache_path, map_location="cpu")
            if isinstance(value, Mapping) and all(name in value for name in SEEN_MAPS):
                return {name: torch.as_tensor(value[name]).detach().cpu() for name in SEEN_MAPS}
    tokenizer, text_encoder = _build_text_encoder(
        model_path,
        max_length=max_length,
        device=device,
        dtype=dtype,
    )
    encoded = tokenizer(
        [instruction_for_map(name) for name in SEEN_MAPS],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device=device)
    attention = encoded.get("attention_mask")
    if attention is not None:
        attention = attention.to(device=device)
    with torch.no_grad():
        output = text_encoder(input_ids=input_ids, attention_mask=attention)
    hidden = output["last_hidden_state"] if isinstance(output, Mapping) else output.last_hidden_state
    result = {}
    for index, name in enumerate(SEEN_MAPS):
        length = int(attention[index].sum().item()) if attention is not None else hidden.shape[1]
        result[name] = hidden[index, :length].detach().cpu()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.tmp")
        torch.save(result, temporary)
        os.replace(temporary, cache_path)
    # Explicitly release the large encoder after this one-time operation.
    del text_encoder
    return result


def build_csgo_components(
    args: Any,
    config: Mapping[str, Any],
    accelerator: Accelerator,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Construct the CSGO RDT and frozen native encoders.

    The returned mapping is intentionally public so a standalone inference
    script can use exactly the same constructor and checkpoint adaptation as
    the training hook.
    """

    logger = logger or LOGGER
    native_config = _native_config(config, args)
    model_config = _mapping(native_config.get("model"))
    common = _mapping(native_config.get("common"))
    dtype = _weight_dtype(accelerator)
    vision_path = _value(args, config, "pretrained_vision_encoder_name_or_path")
    if vision_path is None:
        vision_path = config.get("pretrained_vision_encoder_name_or_path")
    if vision_path is None:
        raise ValueError("pretrained_vision_encoder_name_or_path is required for CSGO")
    vision_encoder = SiglipVisionTower(vision_tower=vision_path, args=None)
    image_processor = vision_encoder.image_processor
    img_cond_len = CSGO_NUM_CAMERAS * vision_encoder.num_patches

    model_path = _value(args, config, "pretrained_model_name_or_path")
    model_kwargs = dict(
        config=model_config,
        lang_token_dim=int(model_config.get("lang_token_dim", 4096)),
        img_token_dim=int(model_config.get("img_token_dim", vision_encoder.hidden_size)),
        state_token_dim=int(model_config.get("state_token_dim", ACTION_DIM)),
        max_lang_cond_len=int(_mapping(native_config.get("dataset")).get("tokenizer_max_length", 1024)),
        img_cond_len=img_cond_len,
        dtype=dtype,
        active_action_dim=ACTIVE_ACTION_DIM,
    )
    csgo_options = _mapping(config.get("csgo"))
    if csgo_options.get("diffusion_channel_policy") is not None:
        model_kwargs["diffusion_channel_policy"] = str(csgo_options["diffusion_channel_policy"])
    if model_path:
        logger.info("Loading CSGO RDT weights from %s", model_path)
        rdt = CSGORDTRunner.from_pretrained(model_path, **model_kwargs)
    else:
        rdt = build_csgo_rdt(native_config, img_cond_len=img_cond_len, dtype=dtype)

    language_embeddings = _value(args, config, "csgo_language_embeddings")
    if language_embeddings is None:
        language_embeddings = config.get("language_embeddings")
    precomp = bool(_value(args, config, "precomp_lang_embed", False))
    text_encoder = None
    tokenizer = None
    if language_embeddings is None and precomp:
        text_path = _value(args, config, "pretrained_text_encoder_name_or_path")
        project_root = _value(args, config, "project_root", ".")
        # Keep the ten-map cache beside this run's checkpoint.  A global
        # max-length-only cache can silently mix models with different T5
        # weights or tokenization settings.
        run_cache = (
            Path(_value(args, config, "output_dir", Path(project_root) / "checkpoints"))
            .expanduser()
            .resolve()
            / "language_embeddings.pt"
        )
        if not run_cache.is_file() and accelerator.is_main_process:
            _precompute_map_language_embeddings(
                model_path=text_path,
                max_length=int(_mapping(native_config.get("dataset")).get("tokenizer_max_length", 1024)),
                device=accelerator.device,
                dtype=dtype,
                cache_path=run_cache,
            )
        # Only rank zero writes the cache.  All ranks consume the same fully
        # written file before constructing their datasets.
        accelerator.wait_for_everyone()
        if not run_cache.is_file():
            raise FileNotFoundError(f"Language embedding cache was not created: {run_cache}")
        # Dataset accepts a per-map tensor dictionary directly.  Keep the
        # path only as provenance on args; passing the loaded mapping avoids
        # asking the dataset to reinterpret a map dictionary as one tensor.
        language_embeddings = torch.load(run_cache, map_location="cpu")
        if not isinstance(language_embeddings, Mapping):
            raise ValueError(f"Language embedding cache must contain a map dictionary: {run_cache}")
        # Persist the resolved path in native run metadata for infer_seen10.
        try:
            setattr(args, "csgo_language_embeddings", str(run_cache))
        except Exception:
            pass
    elif language_embeddings is None and not precomp:
        text_path = _value(args, config, "pretrained_text_encoder_name_or_path")
        tokenizer, text_encoder = _build_text_encoder(
            text_path,
            max_length=int(_mapping(native_config.get("dataset")).get("tokenizer_max_length", 1024)),
            device=accelerator.device,
            dtype=dtype,
        )

    # The encoders are inference-only.  Their parameters are deliberately not
    # passed to Accelerator.prepare, preserving native RDT optimizer state.
    vision_encoder.vision_tower.requires_grad_(False)
    vision_encoder.vision_tower.to(accelerator.device, dtype=dtype).eval()
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
    return {
        "config": native_config,
        "rdt": rdt,
        "vision_encoder": vision_encoder,
        "image_processor": image_processor,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "language_embeddings": language_embeddings,
        "weight_dtype": dtype,
        "img_cond_len": img_cond_len,
        "common": common,
    }


def _move_tensor(value: Any, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(device=device, dtype=dtype) if dtype is not None else tensor.to(device=device)


def _image_tokens(
    images: torch.Tensor,
    vision_encoder: SiglipVisionTower,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    images = _move_tensor(images, device=device, dtype=dtype)
    if images.ndim == 4:
        images = images.unsqueeze(1)
    if images.ndim != 5:
        raise ValueError(f"CSGO images must have shape (B,2,C,H,W), got {tuple(images.shape)}")
    batch_size, views, channels, height, width = images.shape
    if views != CSGO_NUM_CAMERAS:
        raise ValueError(f"CSGO expects FPV+radar ({CSGO_NUM_CAMERAS} views), got {views}")
    embeds = vision_encoder(images.reshape(batch_size * views, channels, height, width)).detach()
    return embeds.reshape(batch_size, views * embeds.shape[1], embeds.shape[2])


def _language_tokens(
    batch: Mapping[str, Any],
    text_encoder: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    mask = batch.get("lang_attn_mask")
    if "lang_embeds" in batch:
        tokens = _move_tensor(batch["lang_embeds"], device=device, dtype=dtype)
        return tokens, _move_tensor(mask, device=device).bool() if mask is not None else None
    if "input_ids" not in batch or text_encoder is None:
        raise ValueError("CSGO batch needs lang_embeds or input_ids plus a text encoder")
    input_ids = _move_tensor(batch["input_ids"], device=device).long()
    attention = _move_tensor(mask, device=device).bool() if mask is not None else None
    with torch.no_grad():
        output = text_encoder(input_ids=input_ids, attention_mask=attention)
    tokens = output["last_hidden_state"] if isinstance(output, Mapping) else output.last_hidden_state
    return tokens.detach().to(dtype=dtype), attention


def prepare_csgo_batch(
    batch: Mapping[str, Any],
    *,
    vision_encoder: SiglipVisionTower,
    text_encoder: Any,
    accelerator: Accelerator,
    dtype: torch.dtype,
    training: bool = True,
    aligned: bool = False,
) -> dict[str, Any]:
    """Turn a Seen-10 collated batch into native CSGO RDT tensors."""

    if "images" not in batch:
        raise KeyError("CSGO batch is missing images")
    device = accelerator.device
    images = _image_tokens(batch["images"], vision_encoder, device=device, dtype=dtype)
    lang_tokens, lang_mask = _language_tokens(batch, text_encoder, device=device, dtype=dtype)
    states = _move_tensor(batch.get("states", torch.zeros((images.shape[0], 1, ACTION_DIM))), device=device, dtype=dtype)
    if states.ndim == 2:
        states = states.unsqueeze(1)
    states = torch.zeros((images.shape[0], 1, ACTION_DIM), device=device, dtype=dtype)
    ctrl = _move_tensor(batch.get("ctrl_freqs", batch.get("ctrl_freq", 1)), device=device, dtype=dtype).reshape(-1)
    if ctrl.numel() == 1 and images.shape[0] > 1:
        ctrl = ctrl.expand(images.shape[0])
    result = {
        "img_tokens": images,
        "lang_tokens": lang_tokens,
        "lang_attn_mask": lang_mask,
        "state_tokens": states,
        "ctrl_freqs": ctrl,
    }
    if training:
        if "actions" not in batch:
            raise KeyError("Training CSGO batch is missing actions")
        action = _move_tensor(batch["actions"], device=device, dtype=dtype)
        if aligned:
            if action.ndim != 3 or action.shape[1:] != (1, ACTIVE_ACTION_DIM):
                raise ValueError(f"Aligned CSGO requires external action (B,1,5), got {tuple(action.shape)}")
            padded = torch.zeros((images.shape[0], 1, ACTION_DIM), device=device, dtype=dtype)
            padded[..., :ACTIVE_ACTION_DIM] = action
            action = padded
        result["action_gt"] = action
        # The dataset may expose a 128-D mask, but the CSGO objective always
        # owns the first five dimensions and never trusts a robot state mask.
        mask = torch.zeros((images.shape[0], 1, ACTION_DIM), device=device, dtype=dtype)
        mask[..., :ACTIVE_ACTION_DIM] = 1
        result["action_mask"] = mask
    return result


def _metadata_for_index(dataset: Any, index: int) -> dict[str, Any]:
    rows = getattr(dataset, "rows", getattr(dataset, "records", ()))
    if rows and index < len(rows) and isinstance(rows[index], Mapping):
        return dict(rows[index])
    return {"sample_id": f"sample-{index}", "map_name": "unknown"}


def _prediction_row(metadata: Mapping[str, Any], values: Sequence[float]) -> dict[str, Any]:
    map_name = str(metadata.get("map_name", metadata.get("map", "unknown")))
    sample_id = metadata.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        file_frame = metadata.get("file_frame")
        sample_id = f"{map_name}/{Path(str(file_frame)).stem}" if file_frame else f"{map_name}/sample"
    fields = ("pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw")
    row = {"sample_id": sample_id, "map_name": map_name}
    row.update({field: float(value) for field, value in zip(fields, values)})
    return row


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


@torch.no_grad()
def evaluate_localization(
    model: CSGORDTRunner,
    dataloader: Any,
    *,
    vision_encoder: SiglipVisionTower,
    text_encoder: Any = None,
    accelerator: Accelerator,
    dtype: torch.dtype,
    output_dir: str | os.PathLike[str] | None = None,
    dataset: Any = None,
    step: int | None = None,
    seed: int = 0,
    per_map: int = 10,
    render: bool = True,
    distributed: bool = False,
) -> dict[str, Any]:
    """Run complete normalized 5D validation and optionally render panels."""

    was_training = model.training
    model.eval()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    predictions: list[dict[str, Any]] = []
    squared_error = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    count = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    row_index = 0
    for batch in dataloader:
        prepared = prepare_csgo_batch(
            batch,
            vision_encoder=vision_encoder,
            text_encoder=text_encoder,
            accelerator=accelerator,
            dtype=dtype,
            training=True,
        )
        predicted = model.predict_action(
            lang_tokens=prepared["lang_tokens"],
            lang_attn_mask=prepared["lang_attn_mask"],
            img_tokens=prepared["img_tokens"],
            state_tokens=prepared["state_tokens"],
            action_mask=prepared["action_mask"],
            ctrl_freqs=prepared["ctrl_freqs"],
        )[:, 0, :ACTIVE_ACTION_DIM]
        target = prepared["action_gt"][:, 0, :ACTIVE_ACTION_DIM]
        squared_error += (predicted.float() - target.float()).pow(2).sum().to(torch.float64)
        count += float(predicted.shape[0] * ACTIVE_ACTION_DIM)

        metadata = batch.get("metadata")
        if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
            metadata = [_metadata_for_index(dataset, row_index + offset) for offset in range(predicted.shape[0])]
        for offset, values in enumerate(predicted.float().cpu().tolist()):
            item = metadata[offset] if offset < len(metadata) and isinstance(metadata[offset], Mapping) else {}
            predictions.append(_prediction_row(item, values))
        row_index += predicted.shape[0]

    # Aggregate the scalar metric across processes.  Prediction identity is
    # written by the main process; ordinary Seen-10 runs use one process, while
    # this scalar remains correct for native distributed launches as well.
    stats = torch.stack([squared_error, count]).reshape(1, 2)
    if distributed and accelerator.num_processes > 1:
        stats = accelerator.gather_for_metrics(stats).sum(dim=0)
    else:
        stats = stats[0]
    mse = float((stats[0] / stats[1].clamp_min(1.0)).item())
    result: dict[str, Any] = {"mse": mse, "count": int(stats[1].item() // ACTIVE_ACTION_DIM), "num_values": int(stats[1].item())}
    if output_dir is not None and accelerator.is_main_process:
        output_root = Path(output_dir).expanduser()
        prediction_path = output_root / "predictions.jsonl"
        _write_jsonl(prediction_path, predictions)
        result["predictions_path"] = os.fspath(prediction_path)
        if render and dataset is not None:
            visualization_dir = output_root / "visualization"
            render_localization(dataset, predictions, visualization_dir, seed=seed, per_map=per_map)
            result["visualization_dir"] = os.fspath(visualization_dir)
    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)
    if was_training:
        model.train()
    return result


def append_loss_jsonl(
    path: str | os.PathLike[str],
    *,
    step: int,
    loss: float,
    lr: float | None = None,
) -> None:
    """Append one native training loss record for restart-safe plotting."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"step": int(step), "loss": float(loss)}
    if lr is not None:
        payload["lr"] = float(lr)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def plot_loss_curve(loss_path: str | os.PathLike[str], output_path: str | os.PathLike[str]) -> str | None:
    """Render the JSONL training loss with matplotlib's non-interactive backend."""

    values: list[tuple[float, float]] = []
    try:
        lines = Path(loss_path).expanduser().read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for index, line in enumerate(lines):
        try:
            payload = json.loads(line)
            step, loss = float(payload.get("step", index)), float(payload["loss"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if math.isfinite(step) and math.isfinite(loss):
            values.append((step, loss))
    if not values:
        return None
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Import lazily so importing the training hooks does not initialize a GUI
    # backend on headless workers before a run actually needs the plot.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.plot([step for step, _ in values], [loss for _, loss in values], color="#1565c0", linewidth=1.8)
    axis.set_title("RDT CSGO training loss")
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.grid(True, alpha=0.25)
    figure.savefig(output)
    plt.close(figure)
    return os.fspath(output)


def update_checkpoint_alias(checkpoint_root: str | os.PathLike[str], alias: str, target: str | os.PathLike[str]) -> str:
    """Atomically update a late/best relative symlink without touching checkpoints."""

    root = Path(checkpoint_root).expanduser()
    destination = Path(target).expanduser().resolve()
    if not destination.is_dir():
        raise FileNotFoundError(f"Checkpoint target is not a directory: {destination}")
    link = root / alias
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{alias}.tmp"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(os.path.relpath(destination, root), target_is_directory=True)
    os.replace(temporary, link)
    return os.fspath(destination)


def write_run_metadata(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> str:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return os.fspath(target)


__all__ = [
    "ACTIVE_ACTION_DIM",
    "ACTION_DIM",
    "build_csgo_components",
    "prepare_csgo_batch",
    "evaluate_localization",
    "append_loss_jsonl",
    "plot_loss_curve",
    "update_checkpoint_alias",
    "write_run_metadata",
]
