"""Real two-process CPU/DDP check of the aligned accumulation semantics.

This is not a claim of multi-GPU Puffin validation. It checks the sampler,
DDP averaging and loss/accumulation against one explicit global batch of 128.
"""
from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from csgo_seen10.aligned_samplers import AlignedTrainBatchSampler


def _worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(directory) / "rendezvous"), rank=rank, world_size=2)
    try:
        torch.manual_seed(23)
        model = torch.nn.Linear(3, 2, dtype=torch.float64)
        ddp = DDP(model)
        data = torch.randn(50000, 3, generator=torch.Generator().manual_seed(97), dtype=torch.float64)
        sampler = iter(AlignedTrainBatchSampler(50000, rank=rank, world_size=2, micro_batch=4,
                                               accumulation=16, seed=42, epoch=0))
        for micro in range(16):
            indexes = next(sampler)
            x = data[indexes]
            target = x[:, :2] * 0.4
            with contextlib.nullcontext() if micro == 15 else ddp.no_sync():
                loss = ((ddp(x) - target) ** 2).mean() / 16
                loss.backward()
        if rank == 0:
            torch.save({name: p.grad for name, p in model.named_parameters()}, Path(directory) / "gradients.pt")
    finally:
        dist.destroy_process_group()


class DistributedAccumulationTests(unittest.TestCase):
    def test_two_process_accumulation_matches_global_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(directory,), nprocs=2, join=True)
            actual = torch.load(Path(directory) / "gradients.pt", weights_only=True)
        torch.manual_seed(23)
        reference = torch.nn.Linear(3, 2, dtype=torch.float64)
        data = torch.randn(50000, 3, generator=torch.Generator().manual_seed(97), dtype=torch.float64)
        order = torch.randperm(50000, generator=torch.Generator().manual_seed(42))[:128]
        x = data[order]
        ((reference(x) - x[:, :2] * 0.4) ** 2).mean().backward()
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(actual[name], parameter.grad, atol=1e-12, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
