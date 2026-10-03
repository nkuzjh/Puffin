"""Real two-process CPU/DDP check of the aligned accumulation semantics.

This is not a claim of multi-GPU Puffin validation. It checks the sampler,
DDP averaging and loss/accumulation against one explicit global batch of 128.
"""
from __future__ import annotations

import contextlib
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from csgo_seen10.aligned_samplers import AlignedTrainBatchSampler
from csgo_seen10.aligned_train import _restore_rng, _rng, _save


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


def _checkpoint_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(directory) / "checkpoint_rendezvous"),
                            rank=rank, world_size=2)
    object_group = dist.new_group(backend="gloo")
    try:
        random.seed(100 + rank)
        np.random.seed(100 + rank)
        torch.manual_seed(100 + rank)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        checkpoints = Path(directory) / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        arguments = dict(model=model, optimizer=optimizer, scheduler=scheduler, rank=rank, world=2,
                         seed=42, topology={"world_size": 2}, identities={}, best=0.5,
                         validation_loss=0.5, is_best=True, smoke_only=True, max_updates=2,
                         resolved_configs={}, object_group=object_group)

        with patch.object(torch.cuda, "is_available", return_value=False):
            before = _rng()
            expected = (random.random(), float(np.random.rand()), torch.rand(()).item())
            _restore_rng(before)

            original_gather = dist.all_gather_object
            original_broadcast = dist.broadcast_object_list

            def checked_gather(*args, **kwargs):
                assert kwargs.get("group") is object_group
                return original_gather(*args, **kwargs)

            def checked_broadcast(*args, **kwargs):
                assert kwargs.get("group") is object_group
                return original_broadcast(*args, **kwargs)

            with patch.object(dist, "all_gather_object", side_effect=checked_gather), \
                    patch.object(dist, "broadcast_object_list", side_effect=checked_broadcast):
                saved = _save(checkpoints, step=1, **arguments)
                payload = torch.load(saved, map_location="cpu", weights_only=False)
                assert len(payload["rng_by_rank"]) == 2
                restored = payload["rng_by_rank"][rank]
                assert restored["python"] == before["python"]
                assert np.array_equal(restored["numpy"][1], before["numpy"][1])
                assert torch.equal(restored["torch"], before["torch"])
                _restore_rng(restored)
                actual = (random.random(), float(np.random.rand()), torch.rand(()).item())
                assert actual == expected

                collision = checkpoints / "step_00000002.pth"
                if rank == 0:
                    collision.write_bytes(b"existing checkpoint")
                dist.barrier()
                try:
                    _save(checkpoints, step=2, **arguments)
                except RuntimeError as exc:
                    error = str(exc)
                else:
                    raise AssertionError("Existing checkpoint was overwritten")
                assert error == ("Aligned checkpoint save failed at step 2: FileExistsError: "
                                 f"Refusing to overwrite completed milestone: {collision}")
                assert collision.read_bytes() == b"existing checkpoint"
    finally:
        dist.destroy_process_group(object_group)
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

    def test_checkpoint_uses_object_group_preserves_rng_and_broadcasts_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_checkpoint_worker, args=(directory,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
