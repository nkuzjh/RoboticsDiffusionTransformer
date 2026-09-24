"""Deterministic global-update sampling for the aligned Seen-10 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class SampleOccurrence:
    """A dataset row together with its position in the training exposure stream."""

    index: int
    epoch: int
    global_occurrence: int


class AlignedUpdateSampler(Sampler[list[SampleOccurrence]]):
    """Yield rank-local microbatches from complete, shuffled global updates.

    ``start_update`` is the number of *completed* optimizer updates. A resumed
    iterator starts at the next update, independently of DataLoader workers or
    prefetch. The same update gets the same sample indices and occurrence IDs.
    """

    def __init__(
        self,
        dataset_size: int,
        seed: int,
        world_size: int,
        rank: int,
        microbatch_size: int,
        gradient_accumulation_steps: int,
        total_updates: int = 19500,
        start_update: int = 0,
        required_global_batch_size: int | None = 128,
    ) -> None:
        for name, value in (
            ("dataset_size", dataset_size),
            ("world_size", world_size),
            ("microbatch_size", microbatch_size),
            ("gradient_accumulation_steps", gradient_accumulation_steps),
            ("total_updates", total_updates),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank!r}")
        self.dataset_size = dataset_size
        self.seed = int(seed)
        self.world_size = world_size
        self.rank = rank
        self.microbatch_size = microbatch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_batch_size = world_size * microbatch_size * gradient_accumulation_steps
        if required_global_batch_size is not None and self.global_batch_size != required_global_batch_size:
            raise ValueError(
                f"Aligned Seen-10 requires global batch {required_global_batch_size}, "
                f"got {self.global_batch_size}"
            )
        self.updates_per_epoch = dataset_size // self.global_batch_size
        if self.updates_per_epoch == 0:
            raise ValueError("Dataset does not contain one complete global update")
        self.samples_per_epoch = self.updates_per_epoch * self.global_batch_size
        self.total_updates = total_updates
        self.set_start_update(start_update)

    def set_start_update(self, start_update: int) -> None:
        if not isinstance(start_update, int) or not 0 <= start_update <= self.total_updates:
            raise ValueError(f"start_update must be in [0, {self.total_updates}], got {start_update!r}")
        self.start_update = start_update

    def state_dict(self) -> dict[str, int]:
        return {"start_update": self.start_update, "seed": self.seed}

    def __len__(self) -> int:
        return (self.total_updates - self.start_update) * self.gradient_accumulation_steps

    def __iter__(self) -> Iterator[list[SampleOccurrence]]:
        cached_epoch = -1
        permutation: list[int] = []
        rank_span = self.microbatch_size * self.gradient_accumulation_steps
        for update in range(self.start_update, self.total_updates):
            epoch, update_in_epoch = divmod(update, self.updates_per_epoch)
            if epoch != cached_epoch:
                generator = torch.Generator(device="cpu")
                generator.manual_seed((self.seed + epoch) % (2**63))
                permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
                cached_epoch = epoch
            update_start = update_in_epoch * self.global_batch_size
            rank_start = update_start + self.rank * rank_span
            for accumulation in range(self.gradient_accumulation_steps):
                local_start = rank_start + accumulation * self.microbatch_size
                yield [
                    SampleOccurrence(
                        index=permutation[position],
                        epoch=epoch,
                        global_occurrence=epoch * self.samples_per_epoch + position,
                    )
                    for position in range(local_start, local_start + self.microbatch_size)
                ]


__all__ = ["AlignedUpdateSampler", "SampleOccurrence"]
