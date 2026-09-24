"""Role-mapped LoRA adaptation for the aligned CSGO RDT experiment.

Inject this only after loading the official RDT base checkpoint.  The seven
returned optimizer groups are exhaustive and disjoint.  Saving the complete
runner state retains both frozen base weights and LoRA weights; the runner's
serialized ``role_adaptation`` recreates these modules before a reload.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


LORA_TARGETS = (
    "attn.qkv", "attn.proj", "cross_attn.q", "cross_attn.kv",
    "cross_attn.proj", "ffn.fc1", "ffn.fc2",
)
TRAINABLE_MODULES = (
    ("img_adaptor", "img_adaptor"),
    ("lang_adaptor", "lang_adaptor"),
    ("state_adaptor", "state_adaptor"),
    ("final_layer", "model.final_layer"),
    ("t_embedder", "model.t_embedder.mlp"),
    ("freq_embedder", "model.freq_embedder.mlp"),
)


class RoleLoRALinear(nn.Module):
    """A frozen linear layer with trainable FP32 low-rank residual matrices."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        update = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + update


def _get_module(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _canonical_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(config or {})
    if isinstance(source.get("role_adaptation"), Mapping):
        source = dict(source["role_adaptation"])
    elif isinstance(source.get("adaptation"), Mapping):
        source = dict(source["adaptation"])
    elif isinstance(source.get("training"), Mapping):
        source = dict(source["training"])
    mode = source.get("mode", source.get("adaptation_mode", "role_lora"))
    if mode != "role_lora":
        raise ValueError(f"Role adaptation requires mode='role_lora', got {mode!r}")
    lora = dict(source.get("lora", {})) if isinstance(source.get("lora"), Mapping) else {}
    rank = int(source.get("r", source.get("rank", lora.get("r", 32))))
    alpha = int(source.get("alpha", lora.get("alpha", 64)))
    dropout = float(source.get("dropout", lora.get("dropout", 0.05)))
    lr = float(source.get("learning_rate", source.get("lr", 1e-4)))
    weight_decay = float(source.get("weight_decay", 0.0))
    if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1 or lr <= 0 or weight_decay != 0:
        raise ValueError("Invalid aligned LoRA rank/alpha/dropout/LR or weight_decay (must be zero)")
    if rank != 32 or alpha != 64 or dropout != 0.05:
        raise ValueError("The aligned experiment fixes LoRA r=32, alpha=64, dropout=0.05")
    if lora.get("bias", source.get("bias", "none")) != "none":
        raise ValueError("The aligned experiment fixes LoRA bias=none")
    if "targets" in source and list(source["targets"]) != list(LORA_TARGETS):
        raise ValueError("Saved aligned LoRA targets differ from the approved RDT block targets")
    return {
        "mode": "role_lora", "r": rank, "alpha": alpha, "dropout": dropout,
        "learning_rate": lr, "weight_decay": weight_decay,
        "targets": list(LORA_TARGETS),
    }


def build_role_adaptation(rdt: nn.Module, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Freeze the RDT base, inject LoRA, and return optimizer groups and audit.

    ``config`` may be the full experiment mapping, its ``training`` section,
    or a serialized ``role_adaptation`` mapping.  The caller creates AdamW
    with ``result['optimizer_groups']`` after calling this function.
    """

    spec = _canonical_config(config)
    if getattr(rdt, "diffusion_channel_policy", None) != "native_full":
        raise ValueError("Role LoRA is only defined for native_full CSGO diffusion")
    if getattr(rdt, "role_adaptation", None) is not None:
        if _canonical_config(rdt.role_adaptation) != spec:
            raise ValueError("Existing role adaptation differs from requested configuration")
        if all(isinstance(block.attn.qkv, RoleLoRALinear) for block in rdt.model.blocks):
            return _build_result(rdt, spec)

    rdt.requires_grad_(False)
    for block in rdt.model.blocks:
        for path in LORA_TARGETS:
            parent_path, name = path.rsplit(".", 1)
            parent = _get_module(block, parent_path)
            base = getattr(parent, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Expected nn.Linear at RDT block {path}, got {type(base).__name__}")
            setattr(parent, name, RoleLoRALinear(base, spec["r"], spec["alpha"], spec["dropout"]))

    for _, path in TRAINABLE_MODULES:
        module = _get_module(rdt, path)
        module.to(dtype=torch.float32)
        module.requires_grad_(True)

    # RDT's timestep embedder creates its sinusoid in this dtype.  The FP32
    # trainable MLP accepts it even outside CUDA autocast.
    rdt.model.t_embedder.dtype = torch.float32
    rdt.model.freq_embedder.dtype = torch.float32
    rdt.role_adaptation = spec
    rdt._hub_mixin_config["role_adaptation"] = spec
    return _build_result(rdt, spec)


def _build_result(rdt: nn.Module, spec: Mapping[str, Any]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    lora = [
        parameter for name, parameter in rdt.named_parameters()
        if parameter.requires_grad and (name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"))
    ]
    groups.append({"name": "rdt_blocks_lora", "params": lora, "lr": spec["learning_rate"], "weight_decay": 0.0})
    for group_name, path in TRAINABLE_MODULES:
        module = _get_module(rdt, path)
        params = [parameter for parameter in module.parameters() if parameter.requires_grad]
        groups.append({"name": group_name, "params": params, "lr": spec["learning_rate"], "weight_decay": 0.0})

    members = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in rdt.parameters() if parameter.requires_grad]
    if not members or len({id(p) for p in members}) != len(members):
        raise RuntimeError("Role optimizer groups contain missing or duplicated parameters")
    if {id(p) for p in members} != {id(p) for p in trainable}:
        raise RuntimeError("Role optimizer groups do not cover exactly the trainable parameters")
    if any(parameter.dtype != torch.float32 for parameter in members):
        raise RuntimeError("Every aligned trainable parameter must be FP32")
    if any(parameter.requires_grad for name, parameter in rdt.named_parameters() if "pos_embed" in name):
        raise RuntimeError("Adapted RDT positional embeddings must be frozen")
    audit = {
        "trainable_parameters": sum(p.numel() for p in members),
        "total_parameters": sum(p.numel() for p in rdt.parameters()),
        "groups": {group["name"]: sum(p.numel() for p in group["params"]) for group in groups},
        "lora_layers": sum(isinstance(module, RoleLoRALinear) for module in rdt.modules()),
    }
    if len(rdt.model.blocks) == 28 and rdt.model.hidden_size == 2048:
        if audit["trainable_parameters"] != 73_164_928 or audit["lora_layers"] != 196:
            raise RuntimeError(f"Official aligned trainable parameter count differs: {audit}")
    return {"optimizer_groups": groups, "audit": audit, "config": dict(spec)}


__all__ = ["RoleLoRALinear", "build_role_adaptation", "LORA_TARGETS"]
