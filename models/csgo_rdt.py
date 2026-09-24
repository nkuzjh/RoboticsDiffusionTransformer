"""RDT adapter for the CSGO Benchmark v2 localization task.

The benchmark uses the native 128 dimensional RDT action representation so a
checkpoint can retain its original action head.  The legacy path masks the
unused channels during diffusion.  The aligned path keeps all 128 channels in
the diffusion process and masks only the final predicted action.

``CSGORDTRunner`` intentionally keeps the public call signature of
``RDTRunner``.  This lets the normal RDT training loop call the adapter without
knowing about the benchmark-specific masking.  The custom ``from_pretrained``
loader is needed because an official RDT checkpoint has a 64-action/6-view
positional layout while CSGO has one action and two current views.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from models.rdt_runner import RDTRunner


LOGGER = logging.getLogger(__name__)
DEFAULT_ACTION_DIM = 128
DEFAULT_ACTIVE_ACTION_DIM = 5
DEFAULT_STATE_TOKEN_DIM = 128
DEFAULT_LANG_TOKEN_DIM = 4096
DEFAULT_IMG_TOKEN_DIM = 1152
DEFAULT_LANG_COND_LEN = 1024
DEFAULT_SOURCE_HISTORY = 2
DEFAULT_SOURCE_CAMERAS = 3
DEFAULT_TARGET_HISTORY = 1
DEFAULT_TARGET_CAMERAS = 2
DEFAULT_SOURCE_PATCHES = 729


def _as_plain_dict(value: Any) -> dict[str, Any]:
    """Return a shallow plain mapping, accepting the config variants in use."""

    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _model_config(config: Any, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize base YAML, official flat Hub, and saved CSGO configs.

    The official ``rdt-1b`` config is flat (``rdt``, ``noise_scheduler`` and
    adaptor fields are top-level).  The repository training YAML nests those
    fields below ``model``.  The runner itself only needs the latter mapping.
    """

    source = _as_plain_dict(config)
    if isinstance(source.get("config"), Mapping):
        source = dict(source["config"])
    if isinstance(source.get("model"), Mapping):
        source = dict(source["model"])
    if extra:
        for key in (
            "lang_adaptor",
            "img_adaptor",
            "state_adaptor",
            "lang_token_dim",
            "img_token_dim",
            "state_token_dim",
            "rdt",
            "noise_scheduler",
        ):
            if key not in source and key in extra:
                source[key] = extra[key]

    # These defaults are only for a tiny/random smoke model.  Official
    # checkpoints always provide the complete values.
    source.setdefault("lang_adaptor", "mlp2x_gelu")
    source.setdefault("img_adaptor", "mlp2x_gelu")
    source.setdefault("state_adaptor", "mlp3x_gelu")
    source.setdefault("lang_token_dim", DEFAULT_LANG_TOKEN_DIM)
    source.setdefault("img_token_dim", DEFAULT_IMG_TOKEN_DIM)
    source.setdefault("state_token_dim", DEFAULT_STATE_TOKEN_DIM)
    source.setdefault(
        "rdt",
        {"hidden_size": 2048, "depth": 28, "num_heads": 32, "cond_pos_embed_type": "multimodal"},
    )
    source.setdefault(
        "noise_scheduler",
        {
            "num_train_timesteps": 1000,
            "num_inference_timesteps": 5,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "sample",
            "clip_sample": False,
        },
    )
    return source


def _layout_from_pos_config(config: Any, default_history: int, default_cameras: int, default_patches: int):
    """Extract ``(history, cameras, patches)`` from an image pos config."""

    if isinstance(config, Mapping):
        config = config.get("img_pos_embed_config")
    if isinstance(config, (list, tuple)):
        for item in config:
            if not isinstance(item, (list, tuple)) or len(item) != 2 or item[0] != "image":
                continue
            value = item[1]
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    history, cameras, patches = int(value[0]), int(value[1]), abs(int(value[-1]))
                except (TypeError, ValueError):
                    continue
                if history > 0 and cameras > 0 and patches > 0:
                    return history, cameras, patches
    return default_history, default_cameras, default_patches


def _strip_module_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _unwrap_state(value: Any) -> Mapping[str, torch.Tensor]:
    """Unwrap Hub and native Accelerate/DeepSpeed checkpoint containers."""

    if isinstance(value, Mapping):
        for key in ("state_dict", "module", "model"):
            nested = value.get(key)
            if isinstance(nested, Mapping) and nested is not value:
                try:
                    return _unwrap_state(nested)
                except ValueError:
                    pass
        tensors = {str(key): item for key, item in value.items() if isinstance(item, torch.Tensor)}
        if tensors:
            return tensors
    raise ValueError("checkpoint does not contain a tensor state dict")


def _load_state_file(path: Path, map_location: str | torch.device) -> Mapping[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device=str(map_location))
    try:
        value = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # torch < 2.0
        value = torch.load(path, map_location=map_location)
    return _unwrap_state(value)


class CSGORDTRunner(RDTRunner):
    """Native RDT runner with a masked 5DoF CSGO localization objective."""

    def __init__(
        self,
        *,
        action_dim: int = DEFAULT_ACTION_DIM,
        pred_horizon: int = 1,
        config: Mapping[str, Any] | None = None,
        lang_token_dim: int = DEFAULT_LANG_TOKEN_DIM,
        img_token_dim: int = DEFAULT_IMG_TOKEN_DIM,
        state_token_dim: int = DEFAULT_STATE_TOKEN_DIM,
        max_lang_cond_len: int = DEFAULT_LANG_COND_LEN,
        img_cond_len: int | None = None,
        lang_pos_embed_config: Any = None,
        img_pos_embed_config: Any = None,
        dtype: torch.dtype = torch.bfloat16,
        active_action_dim: int = DEFAULT_ACTIVE_ACTION_DIM,
        mask_invalid_action_dims: bool = True,
        diffusion_channel_policy: str = "legacy_valid5",
        role_adaptation: Mapping[str, Any] | None = None,
        csgo_img_history_size: int = DEFAULT_TARGET_HISTORY,
        csgo_num_cameras: int = DEFAULT_TARGET_CAMERAS,
        **extra_config: Any,
    ) -> None:
        if int(action_dim) != DEFAULT_ACTION_DIM:
            raise ValueError(f"CSGO keeps the native {DEFAULT_ACTION_DIM}-D action head, got {action_dim}")
        if int(pred_horizon) != 1:
            raise ValueError(f"CSGO localization requires pred_horizon=1, got {pred_horizon}")
        if not 1 <= int(active_action_dim) <= int(action_dim):
            raise ValueError(f"active_action_dim must be in [1, {action_dim}], got {active_action_dim}")
        if int(csgo_img_history_size) != 1 or int(csgo_num_cameras) != 2:
            raise ValueError("CSGO expects one current history and exactly FPV+radar (two cameras)")
        if diffusion_channel_policy not in ("legacy_valid5", "native_full"):
            raise ValueError(f"Unsupported diffusion_channel_policy: {diffusion_channel_policy}")

        normalized_config = _model_config(config, extra_config)
        if diffusion_channel_policy == "native_full":
            if int(active_action_dim) != DEFAULT_ACTIVE_ACTION_DIM:
                raise ValueError("Aligned CSGO requires five external pose dimensions")
            scheduler = _as_plain_dict(normalized_config.get("noise_scheduler"))
            if scheduler.get("prediction_type") != "sample" or scheduler.get("clip_sample") is not False:
                raise ValueError("Aligned CSGO requires clean-action prediction and clip_sample=False")
        # Explicit constructor arguments win over the nested model config.
        lang_token_dim = int(
            extra_config.get("lang_token_dim", normalized_config.get("lang_token_dim", lang_token_dim))
        )
        img_token_dim = int(
            extra_config.get("img_token_dim", normalized_config.get("img_token_dim", img_token_dim))
        )
        state_token_dim = int(
            extra_config.get("state_token_dim", normalized_config.get("state_token_dim", state_token_dim))
        )
        if diffusion_channel_policy == "native_full" and state_token_dim != DEFAULT_STATE_TOKEN_DIM:
            raise ValueError("Aligned CSGO requires a 128-D zero state slot")
        max_lang_cond_len = int(
            extra_config.get("max_lang_cond_len", normalized_config.get("max_lang_cond_len", max_lang_cond_len))
        )
        source_layout = _layout_from_pos_config(
            normalized_config,
            DEFAULT_SOURCE_HISTORY,
            DEFAULT_SOURCE_CAMERAS,
            DEFAULT_SOURCE_PATCHES,
        )
        target_patches = source_layout[2]
        if img_cond_len is None:
            # For an official flat config this changes 2*3*729 to 1*2*729.
            img_cond_len = int(csgo_num_cameras) * target_patches
        img_cond_len = int(img_cond_len)
        if img_cond_len <= 0 or img_cond_len % int(csgo_num_cameras) != 0:
            raise ValueError(f"img_cond_len must be divisible by two cameras, got {img_cond_len}")
        target_patches = img_cond_len // int(csgo_num_cameras)
        if img_pos_embed_config is None:
            img_pos_embed_config = [("image", (1, int(csgo_num_cameras), -target_patches))]
        if lang_pos_embed_config is None:
            lang_pos_embed_config = [("lang", -max_lang_cond_len)]

        super().__init__(
            action_dim=DEFAULT_ACTION_DIM,
            pred_horizon=1,
            config=normalized_config,
            lang_token_dim=lang_token_dim,
            img_token_dim=img_token_dim,
            state_token_dim=state_token_dim,
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )
        # RDTRunner moves only its internal RDT to ``dtype``.  CSGO uses the
        # same precision for the adaptors as well, otherwise an un-autocast
        # inference call fails with a bf16/float32 matmul mismatch.
        self.to(dtype=dtype)
        if hasattr(self.model, "t_embedder"):
            self.model.t_embedder.dtype = dtype
        if hasattr(self.model, "freq_embedder"):
            self.model.freq_embedder.dtype = dtype
        self.active_action_dim = int(active_action_dim)
        self.mask_invalid_action_dims = bool(mask_invalid_action_dims)
        self.diffusion_channel_policy = diffusion_channel_policy
        self.role_adaptation = dict(role_adaptation) if role_adaptation else None
        self.state_token_dim = int(state_token_dim)
        self.csgo_img_history_size = int(csgo_img_history_size)
        self.csgo_num_cameras = int(csgo_num_cameras)
        self.source_img_layout = source_layout
        self.target_img_layout = (self.csgo_img_history_size, self.csgo_num_cameras, target_patches)
        self._csgo_load_report: dict[str, Any] = {}

        # ModelHubMixin uses this attribute when save_pretrained is called.  A
        # flat config makes a saved CSGO checkpoint reloadable without the
        # original training YAML or encoder paths.
        serializable_model_config = {
            key: normalized_config[key]
            for key in (
                "lang_adaptor",
                "img_adaptor",
                "state_adaptor",
                "rdt",
                "noise_scheduler",
            )
            if key in normalized_config
        }
        self._hub_mixin_config = {
            **serializable_model_config,
            # Target constructor values intentionally come last: the source
            # official config may still contain horizon=64 and img_len=4374.
            "action_dim": self.action_dim,
            "pred_horizon": self.pred_horizon,
            "lang_token_dim": lang_token_dim,
            "img_token_dim": img_token_dim,
            "state_token_dim": state_token_dim,
            "max_lang_cond_len": max_lang_cond_len,
            "img_cond_len": img_cond_len,
            "lang_pos_embed_config": lang_pos_embed_config,
            "img_pos_embed_config": img_pos_embed_config,
            "active_action_dim": self.active_action_dim,
            "mask_invalid_action_dims": self.mask_invalid_action_dims,
            "diffusion_channel_policy": self.diffusion_channel_policy,
            "role_adaptation": self.role_adaptation,
            "csgo_img_history_size": self.csgo_img_history_size,
            "csgo_num_cameras": self.csgo_num_cameras,
        }

    @property
    def parameter_dtype(self) -> torch.dtype:
        return self.model.x_pos_embed.dtype

    def _full_action_mask(self, action_mask: torch.Tensor | None, batch_size: int, device, dtype) -> torch.Tensor:
        """Pad a 5-D mask and force all inactive native action dimensions off."""

        if action_mask is None:
            mask = torch.zeros((batch_size, 1, self.action_dim), device=device, dtype=dtype)
            mask[..., : self.active_action_dim] = 1
            return mask
        mask = torch.as_tensor(action_mask, device=device, dtype=dtype)
        if mask.ndim == 1:
            mask = mask.view(1, 1, -1)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(1)
        if mask.ndim != 3 or mask.shape[0] not in (1, batch_size):
            raise ValueError(f"action_mask must be (B,1,D), got {tuple(mask.shape)}")
        if mask.shape[0] == 1 and batch_size != 1:
            mask = mask.expand(batch_size, -1, -1)
        if mask.shape[-1] == self.active_action_dim:
            padded = torch.zeros((batch_size, mask.shape[1], self.action_dim), device=device, dtype=dtype)
            padded[..., : self.active_action_dim] = mask
            mask = padded
        elif mask.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_mask width must be {self.active_action_dim} or {self.action_dim}, got {mask.shape[-1]}"
            )
        mask = mask.clone()
        mask[..., self.active_action_dim :] = 0
        return mask

    def _full_action(self, action_gt: torch.Tensor, batch_size: int, device, dtype) -> torch.Tensor:
        action = torch.as_tensor(action_gt, device=device, dtype=dtype)
        if action.ndim == 2:
            action = action.unsqueeze(1)
        if action.ndim != 3 or action.shape[0] != batch_size:
            raise ValueError(f"action_gt must be (B,1,D), got {tuple(action.shape)}")
        if action.shape[1] != self.pred_horizon:
            raise ValueError(f"CSGO action horizon must be one, got {action.shape[1]}")
        if action.shape[-1] == self.active_action_dim:
            padded = torch.zeros((batch_size, action.shape[1], self.action_dim), device=device, dtype=dtype)
            padded[..., : self.active_action_dim] = action
            action = padded
        elif action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_gt width must be {self.active_action_dim} or {self.action_dim}, got {action.shape[-1]}"
            )
        return action

    def _state_tokens(self, state_tokens: torch.Tensor, batch_size: int, device, dtype) -> torch.Tensor:
        state = torch.as_tensor(state_tokens, device=device, dtype=dtype)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3 or state.shape[0] != batch_size:
            raise ValueError(f"state_tokens must be (B,T,D), got {tuple(state.shape)}")
        if state.shape[1] != 1:
            state = state[:, -1:, :]
        if state.shape[-1] < self.state_adaptor[0].in_features // 2:
            width = self.state_adaptor[0].in_features // 2
            padded = torch.zeros((batch_size, 1, width), device=device, dtype=dtype)
            padded[..., : state.shape[-1]] = state
            state = padded
        elif state.shape[-1] != self.state_adaptor[0].in_features // 2:
            raise ValueError(f"state_tokens width must be 128, got {state.shape[-1]}")
        return state

    def _validate_native_conditions(self, state: torch.Tensor, mask: torch.Tensor, ctrl_freqs: torch.Tensor) -> None:
        if torch.any(state != 0):
            raise ValueError("Aligned CSGO requires the all-zero state slot")
        if torch.any(mask[..., : self.active_action_dim] != 1):
            raise ValueError("Aligned CSGO requires all five pose action indicators")
        if torch.any(ctrl_freqs != 1):
            raise ValueError("Aligned CSGO requires control frequency one")

    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        # The parent adapters are dtype-sensitive under bf16.  Casting here
        # also makes lightweight CPU smoke models accept ordinary float32
        # tensors while preserving the native public API.
        dtype = self.parameter_dtype
        lang_tokens = lang_tokens.to(dtype=dtype)
        img_tokens = img_tokens.to(dtype=dtype)
        state_tokens = state_tokens.to(dtype=dtype)
        return super().adapt_conditions(lang_tokens, img_tokens, state_tokens)

    def compute_loss(
        self,
        lang_tokens,
        lang_attn_mask,
        img_tokens,
        state_tokens,
        action_gt,
        action_mask,
        ctrl_freqs,
    ) -> torch.Tensor:
        """Train the selected diffusion channel policy."""

        if self.diffusion_channel_policy == "native_full":
            return self._compute_native_full_loss(
                lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                action_gt, action_mask, ctrl_freqs,
            )

        if not self.mask_invalid_action_dims:
            return super().compute_loss(
                lang_tokens,
                lang_attn_mask,
                img_tokens,
                state_tokens,
                action_gt,
                action_mask,
                ctrl_freqs,
            )
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        dtype = self.parameter_dtype
        state = self._state_tokens(state_tokens, batch_size, device, dtype)
        actions = self._full_action(action_gt, batch_size, device, dtype)
        mask = self._full_action_mask(action_mask, batch_size, device, dtype)
        actions = actions * mask
        noise = torch.randn_like(actions) * mask
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(actions, noise, timesteps) * mask

        state_mask = torch.zeros_like(state)
        action_mask_expanded = mask.expand(-1, self.pred_horizon, -1)
        state_action = torch.cat([state, noisy_action], dim=1)
        state_action_mask = torch.cat([state_mask, action_mask_expanded], dim=1)
        state_action = torch.cat([state_action, state_action_mask], dim=2)
        lang_cond, img_cond, state_action = self.adapt_conditions(lang_tokens, img_tokens, state_action)
        ctrl_freqs = torch.as_tensor(ctrl_freqs, device=device, dtype=dtype).reshape(-1)
        pred = self.model(
            state_action,
            ctrl_freqs,
            timesteps,
            lang_cond,
            img_cond,
            lang_mask=lang_attn_mask,
        )
        target = noise if self.prediction_type == "epsilon" else actions
        if self.prediction_type not in ("epsilon", "sample"):
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")
        error = (pred - target).pow(2) * action_mask_expanded
        return error.sum() / action_mask_expanded.sum().clamp_min(1.0)

    def _compute_native_full_loss(
        self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
        action_gt, action_mask, ctrl_freqs,
    ) -> torch.Tensor:
        """Original RDT full-width noising and MSE with a CSGO state slot."""

        import torch.nn.functional as F

        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        dtype = self.parameter_dtype
        state = self._state_tokens(state_tokens, batch_size, device, dtype)
        action = self._full_action(action_gt, batch_size, device, dtype)
        mask = self._full_action_mask(action_mask, batch_size, device, dtype)
        if torch.any(action[..., self.active_action_dim:] != 0):
            raise ValueError("The padded CSGO target channels must be zero")
        noise = torch.randn_like(action)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
        state_action = torch.cat([state, noisy_action], dim=1)
        indicators = torch.cat([torch.zeros_like(state), mask.expand(-1, self.pred_horizon, -1)], dim=1)
        state_action = torch.cat([state_action, indicators], dim=2)
        ctrl_freqs = torch.as_tensor(ctrl_freqs, device=device, dtype=dtype).reshape(-1)
        self._validate_native_conditions(state, mask, ctrl_freqs)
        with self._role_autocast():
            lang_cond, img_cond, state_action = self.adapt_conditions(lang_tokens, img_tokens, state_action)
            pred = self.model(
                state_action, ctrl_freqs, timesteps, lang_cond, img_cond,
                lang_mask=lang_attn_mask,
            )
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = action
        else:
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")
        return F.mse_loss(pred.float(), target.float())

    def _role_autocast(self):
        """Allow FP32 trainable LoRA/adaptors beside frozen BF16 base weights."""

        from contextlib import nullcontext

        if self.role_adaptation and self.parameter_dtype in (torch.bfloat16, torch.float16):
            device_type = self.model.x_pos_embed.device.type
            if device_type in ("cuda", "cpu") and not (device_type == "cpu" and self.parameter_dtype == torch.float16):
                return torch.autocast(device_type=device_type, dtype=self.parameter_dtype)
        return nullcontext()

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs):
        """Sample with the selected diffusion channel policy."""

        if self.diffusion_channel_policy == "native_full":
            return self._conditional_sample_native_full(
                lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs,
            )

        device = state_traj.device
        dtype = state_traj.dtype
        batch_size = state_traj.shape[0]
        mask = self._full_action_mask(action_mask, batch_size, device, dtype)
        noisy_action = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim), device=device, dtype=dtype
        ) * mask.expand(-1, self.pred_horizon, -1)
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        ctrl_freqs = torch.as_tensor(ctrl_freqs, device=device, dtype=dtype).reshape(-1)
        for timestep in self.noise_scheduler_sample.timesteps:
            action_mask_expanded = mask.expand(-1, self.pred_horizon, -1)
            action_traj = torch.cat([noisy_action, action_mask_expanded], dim=2)
            action_traj = self.state_adaptor(action_traj)
            full_traj = torch.cat([state_traj, action_traj], dim=1)
            model_output = self.model(
                full_traj,
                ctrl_freqs,
                timestep.unsqueeze(-1).to(device),
                lang_cond,
                img_cond,
                lang_mask=lang_attn_mask,
            )
            model_output = model_output * action_mask_expanded
            noisy_action = self.noise_scheduler_sample.step(
                model_output, timestep, noisy_action
            ).prev_sample
            noisy_action = noisy_action.to(dtype=dtype) * action_mask_expanded
        return noisy_action * mask.expand(-1, self.pred_horizon, -1)

    def _conditional_sample_native_full(
        self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs,
    ):
        """Keep the complete noisy latent until the original final action mask."""

        device = state_traj.device
        dtype = state_traj.dtype
        batch_size = state_traj.shape[0]
        mask = self._full_action_mask(action_mask, batch_size, device, dtype)
        mask = mask.expand(-1, self.pred_horizon, -1)
        noisy_action = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim), device=device, dtype=dtype,
        )
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        ctrl_freqs = torch.as_tensor(ctrl_freqs, device=device, dtype=dtype).reshape(-1)
        if torch.any(ctrl_freqs != 1):
            raise ValueError("Aligned CSGO requires control frequency one")
        with self._role_autocast():
            for timestep in self.noise_scheduler_sample.timesteps:
                action_traj = self.state_adaptor(torch.cat([noisy_action, mask], dim=2))
                full_traj = torch.cat([state_traj, action_traj], dim=1)
                model_output = self.model(
                    full_traj, ctrl_freqs, timestep.unsqueeze(-1).to(device),
                    lang_cond, img_cond, lang_mask=lang_attn_mask,
                )
                noisy_action = self.noise_scheduler_sample.step(
                    model_output, timestep, noisy_action,
                ).prev_sample.to(dtype=dtype)
        return noisy_action * mask

    def predict_action(
        self,
        lang_tokens,
        lang_attn_mask,
        img_tokens,
        state_tokens,
        action_mask,
        ctrl_freqs,
    ):
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        dtype = self.parameter_dtype
        state = self._state_tokens(state_tokens, batch_size, device, dtype)
        mask = self._full_action_mask(action_mask, batch_size, device, dtype)
        if self.diffusion_channel_policy == "native_full":
            self._validate_native_conditions(
                state, mask, torch.as_tensor(ctrl_freqs, device=device, dtype=dtype).reshape(-1),
            )
        # CSGO has no proprioception.  Its state value and indicator are both
        # zero; the active mask belongs to action tokens only.
        state_with_mask = torch.cat([state, torch.zeros_like(state)], dim=2)
        with self._role_autocast():
            lang_cond, img_cond, state_traj = self.adapt_conditions(lang_tokens, img_tokens, state_with_mask)
        sample = self.conditional_sample(lang_cond, lang_attn_mask, img_cond, state_traj, mask, ctrl_freqs)
        if self.diffusion_channel_policy == "native_full":
            return sample[..., : self.active_action_dim]
        return sample

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load an official/local RDT checkpoint with explicit positional crops.

        The implementation intentionally does not call the repository's old
        Hub ``_from_pretrained`` path: recent ``huggingface_hub`` versions no
        longer pass its legacy ``proxies``/``resume_download`` arguments.
        """

        if args:
            raise TypeError("CSGORDTRunner.from_pretrained accepts keyword arguments only after the model path")
        model_id = os.fspath(pretrained_model_name_or_path)
        revision = kwargs.get("revision")
        source_config: dict[str, Any] = {}
        source_file: Path | None = None
        if os.path.isdir(model_id):
            model_dir = Path(model_id)
            source_config = _read_json(model_dir / "config.json") or {}
            for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt"):
                candidate = model_dir / name
                if candidate.is_file():
                    source_file = candidate
                    break
            if source_file is None:
                # Accept an Accelerate/DeepSpeed model file supplied as a
                # directory, while keeping discovery explicit.
                candidates = sorted(model_dir.glob("pytorch_model/**/*.pt"))
                if candidates:
                    source_file = candidates[0]
        elif os.path.isfile(model_id):
            source_file = Path(model_id)
            # A bare model file normally sits beside its config.  Keep the
            # older cache layout fallback for checkpoints whose file is
            # nested one directory below the repository snapshot.
            parent_config = source_file.parent / "config.json"
            fallback_config = source_file.parent.parent / "config.json"
            source_config = _read_json(parent_config) or _read_json(fallback_config) or {}
        else:
            try:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename="config.json",
                    revision=revision,
                    cache_dir=kwargs.get("cache_dir"),
                    token=kwargs.get("token"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                source_config = _read_json(Path(config_path)) or {}
            except Exception as exc:
                LOGGER.debug("No remote RDT config available for %s: %s", model_id, exc)
            try:
                from huggingface_hub import hf_hub_download

                common_download = {
                    "repo_id": model_id,
                    "revision": revision,
                    "cache_dir": kwargs.get("cache_dir"),
                    "token": kwargs.get("token"),
                    "local_files_only": kwargs.get("local_files_only", False),
                }
                try:
                    source_file = Path(hf_hub_download(filename="model.safetensors", **common_download))
                except Exception:
                    source_file = Path(hf_hub_download(filename="pytorch_model.bin", **common_download))
            except Exception as exc:
                raise FileNotFoundError(f"Could not download RDT weights for {model_id}") from exc

        # Constructor config is optional: official source config is used for
        # architecture, while the CSGO constructor derives target image layout.
        target_config = kwargs.pop("config", None)
        if target_config is None:
            target_config = source_config
        source_config_map = _as_plain_dict(source_config)

        def source_value(name: str, default: Any = None):
            if name in kwargs:
                return kwargs[name]
            if name in source_config_map:
                return source_config_map[name]
            nested = source_config_map.get("config")
            if isinstance(nested, Mapping) and name in nested:
                return nested[name]
            return default

        # A saved CSGO config has the target image length and explicit adapter
        # dimensions.  The official flat config has the source 2x3/history-2
        # layout, so its image length must remain a source-only metadata value
        # and the constructor derives the target 1x2 length.
        source_is_csgo = (
            "csgo_num_cameras" in source_config_map
            or "csgo_img_history_size" in source_config_map
            or "active_action_dim" in source_config_map
        )
        if source_is_csgo and "img_cond_len" not in kwargs:
            saved_img_len = source_value("img_cond_len")
            if saved_img_len is not None:
                kwargs["img_cond_len"] = int(saved_img_len)
        for name in ("lang_token_dim", "img_token_dim", "state_token_dim", "max_lang_cond_len"):
            value = source_value(name)
            if value is not None:
                kwargs[name] = int(value)
        for name in ("active_action_dim", "mask_invalid_action_dims", "diffusion_channel_policy", "role_adaptation", "csgo_img_history_size", "csgo_num_cameras"):
            value = source_value(name)
            if value is not None:
                kwargs[name] = value
        kwargs.pop("revision", None)
        kwargs.pop("cache_dir", None)
        kwargs.pop("token", None)
        kwargs.pop("local_files_only", None)
        kwargs.pop("map_location", None)
        strict = bool(kwargs.pop("strict", False))
        if "csgo_img_cond_len" in kwargs and "img_cond_len" not in kwargs:
            kwargs["img_cond_len"] = kwargs.pop("csgo_img_cond_len")
        # Official config values describe the source six-view/64-step layout;
        # never forward those two fields as target constructor overrides.
        kwargs.pop("img_pos_embed_config", None)
        kwargs.pop("pred_horizon", None)
        kwargs.pop("action_dim", None)
        if source_file is None:
            raise FileNotFoundError(f"No model weights found at {model_id}")

        model = cls(config=target_config, **kwargs)
        if model.role_adaptation:
            from models.csgo_adaptation import build_role_adaptation

            build_role_adaptation(model, model.role_adaptation)
        source_state = _load_state_file(source_file, "cpu")
        model._load_adapted_state_dict(source_state, source_config, strict=strict)
        model.eval()
        return model

    def _load_adapted_state_dict(
        self,
        source_state: Mapping[str, torch.Tensor],
        source_config: Mapping[str, Any] | None = None,
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Load exact keys plus the two documented positional crops."""

        target_state = self.state_dict()
        source_cfg = _model_config(source_config or {})
        src_layout = _layout_from_pos_config(
            source_cfg,
            DEFAULT_SOURCE_HISTORY,
            DEFAULT_SOURCE_CAMERAS,
            self.target_img_layout[2],
        )
        adapted: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        cropped: list[str] = []
        for raw_key, value in source_state.items():
            key = _strip_module_prefix(str(raw_key))
            if not isinstance(value, torch.Tensor) or key not in target_state:
                skipped.append(key)
                continue
            target = target_state[key]
            candidate = value
            if key == "model.x_pos_embed" and value.ndim == 3 and target.ndim == 3:
                # RDT's first three positions are timestep/frequency/state;
                # retain those plus the first action token.
                if value.shape[0] == target.shape[0] and value.shape[2] == target.shape[2] and value.shape[1] >= 4:
                    candidate = value[:, :4]
                    cropped.append(key)
            elif key == "model.img_cond_pos_embed" and value.ndim == 3 and target.ndim == 3:
                src_history, src_cameras, src_patches = src_layout
                expected_source_len = src_history * src_cameras * src_patches
                target_len = target.shape[1]
                if (
                    value.shape[0] == target.shape[0]
                    and value.shape[2] == target.shape[2]
                    and value.shape[1] == expected_source_len
                    and target_len == self.csgo_num_cameras * self.target_img_layout[2]
                    and src_patches == self.target_img_layout[2]
                ):
                    start = (src_history - 1) * src_cameras * src_patches
                    candidate = value[:, start : start + target_len]
                    cropped.append(key)
                elif tuple(value.shape) == tuple(target.shape):
                    # A CSGO checkpoint has already been cropped.  Its config
                    # may be unavailable when loading a bare state file, so
                    # exact shape equality is sufficient in this one case.
                    candidate = value
            if tuple(candidate.shape) != tuple(target.shape):
                skipped.append(key)
                continue
            adapted[key] = candidate.to(dtype=target.dtype)

        missing, unexpected = self.load_state_dict(adapted, strict=False)
        # ``adapted`` is already restricted to exact target keys; unexpected
        # can only arise from a future PyTorch/module wrapper change.
        report = {
            "source_keys": len(source_state),
            "loaded_keys": len(adapted),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "cropped_keys": cropped,
            "skipped_keys": sorted(set(skipped)),
        }
        self._csgo_load_report = report
        # The only permitted structural differences are the documented x/img
        # positional crops.  Silently accepting a partial model would leave
        # random adaptor/transformer weights in an ostensibly pretrained run.
        if missing or unexpected or skipped:
            raise RuntimeError(f"CSGO RDT load failed structural validation: {report}")
        LOGGER.info(
            "Loaded CSGO RDT checkpoint: %d/%d keys (%d cropped, %d skipped)",
            len(adapted),
            len(source_state),
            len(cropped),
            len(set(skipped)),
        )
        return report


def build_csgo_rdt(
    config: Mapping[str, Any],
    *,
    img_cond_len: int,
    dtype: torch.dtype = torch.bfloat16,
    active_action_dim: int = DEFAULT_ACTIVE_ACTION_DIM,
) -> CSGORDTRunner:
    """Construct a random CSGO runner from a base/model YAML mapping."""

    model_config = _model_config(config)
    return CSGORDTRunner(
        action_dim=DEFAULT_ACTION_DIM,
        pred_horizon=1,
        config=model_config,
        lang_token_dim=int(model_config.get("lang_token_dim", DEFAULT_LANG_TOKEN_DIM)),
        img_token_dim=int(model_config.get("img_token_dim", DEFAULT_IMG_TOKEN_DIM)),
        state_token_dim=int(model_config.get("state_token_dim", DEFAULT_STATE_TOKEN_DIM)),
        max_lang_cond_len=int(_as_plain_dict(config).get("dataset", {}).get("tokenizer_max_length", DEFAULT_LANG_COND_LEN)),
        img_cond_len=int(img_cond_len),
        dtype=dtype,
        active_action_dim=active_action_dim,
    )


__all__ = ["CSGORDTRunner", "build_csgo_rdt"]
