"""Original RDT image augmentation with occurrence-local random sources.

Only the photometric and corruption operations from ``train/dataset.py`` and
``train/image_corrupt.py`` are supported. They do not change Seen-10 pose labels.
"""

from __future__ import annotations

import hashlib
import copy
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

# imgaug 0.4.0 still uses np.bool on the repository's NumPy version. Its
# original RDT module applies the same compatibility alias.
np.bool = np.bool_
import imgaug.augmenters as iaa  # noqa: E402
import imgaug as ia  # noqa: E402


_BRANCHES = ("corrupt_only", "color_only", "both")
_ALLOWED_KEYS = {
    "policy", "train_only", "views", "per_view_probability", "independent_views",
    "geometry", "auto_adjust_image_brightness", "state_noise_snr",
    "condition_dropout_probability", "rng_mode", "seed",
}


@dataclass(frozen=True)
class AugmentationDecision:
    applied: bool
    branch: str | None


def validate_augmentation(config: Mapping[str, object]) -> None:
    unknown = set(config) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unknown Seen-10 augmentation options: {sorted(unknown)}")
    if config.get("policy") != "rdt_native_image_v1":
        raise ValueError("Aligned image augmentation requires policy=rdt_native_image_v1")
    if config.get("train_only", True) is not True:
        raise ValueError("Seen-10 augmentation must be train-only")
    if tuple(config.get("views", ("fpv", "radar"))) != ("fpv", "radar"):
        raise ValueError("Original image augmentation must cover FPV and radar")
    if float(config.get("per_view_probability", 0.5)) != 0.5:
        raise ValueError("Original RDT image augmentation uses per-view probability 0.5")
    if config.get("independent_views", True) is not True:
        raise ValueError("FPV and radar must use independent random draws")
    if config.get("geometry", "none") != "none":
        raise ValueError("Geometric augmentation has no matching Seen-10 pose transform")
    if config.get("auto_adjust_image_brightness", False) is not False:
        raise ValueError("Auto brightness is outside the approved augmentation")
    if config.get("state_noise_snr") is not None:
        raise ValueError("Zero state placeholders cannot receive state noise")
    if float(config.get("condition_dropout_probability", 0.0)) != 0.0:
        raise ValueError("Condition dropout is outside the approved augmentation")
    if config.get("rng_mode", "sample_occurrence") != "sample_occurrence":
        raise ValueError("Augmentation requires rng_mode=sample_occurrence")
    int(config.get("seed", 0))


def _seed(*parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        value = str(part).encode("utf-8")
        digest.update(len(value).to_bytes(4, "little"))
        digest.update(value)
    return int.from_bytes(digest.digest(), "little")


def _color_jitter(image: Image.Image, seed: int) -> Image.Image:
    """Match torchvision ColorJitter's ranges and randomized operation order."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63))
    order = torch.randperm(4, generator=generator).tolist()
    ranges = ((0.7, 1.3), (0.6, 1.4), (0.5, 1.5), (-0.03, 0.03))
    factors = [lo + (hi - lo) * torch.rand((), generator=generator).item() for lo, hi in ranges]
    operations = (F.adjust_brightness, F.adjust_contrast, F.adjust_saturation, F.adjust_hue)
    for index in order:
        image = operations[index](image, factors[index])
    return image


def _corruption(seed: int) -> iaa.Sequential:
    """Rebuild RDT's noise and optional blur graph with seeded child nodes."""

    generator = np.random.default_rng(seed)

    def node_seed() -> int:
        return int(generator.integers(0, 2**31 - 1))

    return iaa.Sequential(
        [
            iaa.OneOf(
                [
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5, seed=node_seed()),
                    iaa.AdditiveLaplaceNoise(scale=(0.0, 0.05 * 255), per_channel=0.5, seed=node_seed()),
                    iaa.AdditivePoissonNoise(lam=(0.0, 0.05 * 255), per_channel=0.5, seed=node_seed()),
                ],
                seed=node_seed(),
            ),
            iaa.SomeOf(
                (0, 1),
                [
                    iaa.OneOf(
                        [
                            iaa.GaussianBlur((0, 3.0), seed=node_seed()),
                            iaa.AverageBlur(k=(2, 7), seed=node_seed()),
                            iaa.MedianBlur(k=(3, 11), seed=node_seed()),
                        ],
                        seed=node_seed(),
                    ),
                    iaa.MotionBlur(k=(3, 36), seed=node_seed()),
                ],
                seed=node_seed(),
            ),
        ],
        random_order=True,
        seed=node_seed(),
    )


def augment_image(
    image: Image.Image,
    *,
    config: Mapping[str, object],
    epoch: int,
    global_occurrence: int,
    sample_id: str,
    view: str,
) -> tuple[Image.Image, AugmentationDecision]:
    """Apply an independently seeded RDT augmentation to one RGB view."""

    if view not in ("fpv", "radar"):
        raise ValueError(f"Unknown Seen-10 view {view!r}")
    base = _seed(int(config.get("seed", 0)), epoch, global_occurrence, sample_id, view)
    rng = np.random.default_rng(_seed(base, "branch"))
    if float(rng.random()) <= 0.5:
        return image, AugmentationDecision(False, None)
    branch = _BRANCHES[int(rng.integers(0, len(_BRANCHES)))]
    if branch != "corrupt_only":
        image = _color_jitter(image, _seed(base, "color"))
    if branch != "color_only":
        array = np.asarray(image, dtype=np.uint8)[None, ...]
        # imgaug's constructors consume NumPy's legacy process RNG even when
        # every augmenter is explicitly seeded. Restore that bookkeeping so
        # augmentation cannot shift the diffusion-noise RNG of a 0-worker run.
        numpy_state = np.random.get_state()
        imgaug_rng = ia.random.get_global_rng()
        imgaug_state = copy.deepcopy(imgaug_rng.state)
        try:
            result = _corruption(_seed(base, "corruption"))(images=array)[0]
        finally:
            np.random.set_state(numpy_state)
            imgaug_rng.set_state_(imgaug_state)
        image = Image.fromarray(result)
    return image, AugmentationDecision(True, branch)


__all__ = ["AugmentationDecision", "augment_image", "validate_augmentation"]
