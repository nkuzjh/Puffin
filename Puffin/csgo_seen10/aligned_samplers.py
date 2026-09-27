"""Logical update sampling for exact, prefetch-independent Seen-10 resume."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class AlignedTrainBatchSampler(Sampler[list[int]]):
    """Shuffled 50k rows; consume exactly 49,920 per 390 global updates.

    ``start_update`` is the completed optimizer-update cursor within an epoch,
    not the DataLoader iterator position. Prefetch cannot move this cursor.
    """

    def __init__(self, size: int, *, rank: int, world_size: int, micro_batch: int,
                 accumulation: int, seed: int, epoch: int, start_update: int = 0):
        self.size = int(size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.micro_batch = int(micro_batch)
        self.accumulation = int(accumulation)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_update = int(start_update)
        self.global_batch = self.world_size * self.micro_batch * self.accumulation
        if self.global_batch != 128:
            raise ValueError(f"Aligned global batch must be 128, got {self.global_batch}")
        if self.size != 50_000:
            raise ValueError(f"Aligned training split must have exactly 50,000 rows, got {self.size}")
        if not 0 <= self.rank < self.world_size or not 0 <= self.start_update <= 390:
            raise ValueError("Invalid rank or logical update cursor")

    def __len__(self) -> int:
        return (390 - self.start_update) * self.accumulation

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator)[:49_920]
        for update in range(self.start_update, 390):
            update_start = update * 128
            for micro in range(self.accumulation):
                rank_start = update_start + micro * self.world_size * self.micro_batch + self.rank * self.micro_batch
                yield order[rank_start:rank_start + self.micro_batch].tolist()


class AlignedValidationSampler(Sampler[int]):
    """No padding, repetition, or dropped rows across DDP validation ranks."""

    def __init__(self, size: int, *, rank: int, world_size: int):
        self.size = int(size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid validation rank")

    def __len__(self) -> int:
        return len(range(self.rank, self.size, self.world_size))

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.size, self.world_size))
