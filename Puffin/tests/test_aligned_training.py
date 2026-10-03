"""CPU checks for aligned optimizer semantics and restartable sampling."""

from __future__ import annotations

import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from csgo_seen10.aligned import DETERMINISM, aligned_semantics, apply_aligned_peft, config_fingerprint
from csgo_seen10.aligned_samplers import AlignedTrainBatchSampler, AlignedValidationSampler
from csgo_seen10.aligned_train import _configure_determinism, _restore_rng, _rng, _save, make_optimizer, make_scheduler


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj"):
            setattr(self, name, nn.Linear(8, 8))


class _QwenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self.mlp, name, nn.Linear(8, 8))


class _SD3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"):
            setattr(self.attn, name, nn.Linear(8, 8))
        self.attn.to_out = nn.Sequential(nn.Linear(8, 8))
        for name in ("ff", "ff_context"):
            block = nn.Module()
            block.net = nn.Sequential(nn.Module(), nn.Identity(), nn.Linear(8, 8))
            block.net[0].proj = nn.Linear(8, 8)
            setattr(self, name, block)


class _ConnectorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(8, 8)
        self.mlp.fc2 = nn.Linear(8, 8)


class _TinyPuffin(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = nn.Module()
        self.llm.model = nn.Module()
        self.llm.model.layers = nn.ModuleList([_QwenBlock()])
        self.transformer = nn.Module()
        self.transformer.transformer_blocks = nn.ModuleList([_SD3Block()])
        self.connector_1 = nn.Module()
        self.connector_1.layers = nn.ModuleList([_ConnectorBlock()])
        self.connector_2 = nn.Module()
        self.connector_2.layers = nn.ModuleList([_ConnectorBlock()])
        self.llm2connector_1 = nn.Linear(8, 8)
        self.llm2connector_2 = nn.Linear(8, 8)
        self.meta_queries = nn.Parameter(torch.randn(2, 8))
        self.projector_1 = nn.Linear(8, 8)
        self.projector_2 = nn.Linear(8, 8)
        self.visual_encoder = nn.Linear(8, 8)
        self.projector = nn.Linear(8, 8)
        self.vae = nn.Linear(8, 8)


class AlignedTrainingTests(unittest.TestCase):
    def test_single_rank_checkpoint_restores_rng_without_object_group(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda, "is_available", return_value=False):
            random.seed(417)
            np.random.seed(417)
            torch.manual_seed(417)
            model = nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            before = _rng()
            expected = (random.random(), float(np.random.rand()), torch.rand(()).item())
            _restore_rng(before)
            arguments = dict(step=1, model=model, optimizer=optimizer, scheduler=scheduler,
                             rank=0, world=1, seed=42, topology={"world_size": 1}, identities={},
                             best=0.5, validation_loss=0.5, is_best=True, smoke_only=True,
                             max_updates=2, resolved_configs={}, object_group=None)
            path = _save(Path(directory), **arguments)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(len(payload["rng_by_rank"]), 1)
            self.assertEqual(payload["rng_by_rank"][0]["python"], before["python"])
            _restore_rng(payload["rng_by_rank"][0])
            self.assertEqual((random.random(), float(np.random.rand()), torch.rand(()).item()), expected)
            with self.assertRaisesRegex(ValueError, "requires the Gloo object group"):
                _save(Path(directory), **{**arguments, "world": 2})

    def test_deterministic_runtime_is_explicit_and_conflicts_fail(self):
        original_algorithms = torch.are_deterministic_algorithms_enabled()
        original_cudnn = torch.backends.cudnn.deterministic
        original_benchmark = torch.backends.cudnn.benchmark
        try:
            with patch.dict(os.environ, {}, clear=True):
                _configure_determinism()
                self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertTrue(torch.backends.cudnn.deterministic)
                self.assertFalse(torch.backends.cudnn.benchmark)
            with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
                with self.assertRaisesRegex(ValueError, "CUBLAS_WORKSPACE_CONFIG"):
                    _configure_determinism()
            self.assertEqual(aligned_semantics()["determinism"], DETERMINISM)
        finally:
            torch.use_deterministic_algorithms(original_algorithms)
            torch.backends.cudnn.deterministic = original_cudnn
            torch.backends.cudnn.benchmark = original_benchmark

    def test_sampler_covers_exact_truncated_epoch_and_resumes_by_update(self):
        first = AlignedTrainBatchSampler(50_000, rank=0, world_size=2, micro_batch=4,
                                         accumulation=16, seed=42, epoch=0)
        second = AlignedTrainBatchSampler(50_000, rank=1, world_size=2, micro_batch=4,
                                          accumulation=16, seed=42, epoch=0)
        left, right = list(first), list(second)
        self.assertEqual(len(left), 390 * 16)
        self.assertEqual(len(right), 390 * 16)
        flattened = [index for batch in left + right for index in batch]
        self.assertEqual(len(flattened), 49_920)
        self.assertEqual(len(set(flattened)), 49_920)
        resumed = AlignedTrainBatchSampler(50_000, rank=0, world_size=2, micro_batch=4,
                                           accumulation=16, seed=42, epoch=0, start_update=123)
        self.assertEqual(list(resumed), left[123 * 16:])

    def test_validation_shards_have_no_padding(self):
        indices = [index for rank in range(3) for index in AlignedValidationSampler(5000, rank=rank, world_size=3)]
        self.assertEqual(sorted(indices), list(range(5000)))

    def test_lora_coverage_and_optimizer_groups(self):
        model = _TinyPuffin().to(dtype=torch.bfloat16)
        audit = apply_aligned_peft(model, strict=False)
        self.assertEqual(audit["counts"]["unexpected"], 0)
        self.assertGreater(audit["counts"]["llm"], 0)
        self.assertGreater(audit["counts"]["transformer"], 0)
        self.assertGreater(audit["counts"]["connectors"], 0)
        self.assertGreater(audit["counts"]["full"], 0)
        self.assertTrue(any(name.startswith("llm2connector_1.lora_A") for name in audit["names"]["connectors"]))
        optimizer, names = make_optimizer(model)
        self.assertEqual(sum(len(group["params"]) for group in optimizer.param_groups),
                         sum(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual({group["lr"] for group in optimizer.param_groups}, {1e-4, 5e-6})
        self.assertTrue(names["full_no_decay"])
        self.assertTrue(names["lora_decay"])
        self.assertFalse(model.visual_encoder.weight.requires_grad)

    def test_scheduler_tracks_optimizer_updates_without_lr_multiplication(self):
        parameter = nn.Parameter(torch.zeros(2, 2))
        optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1e-4}])
        scheduler = make_scheduler(optimizer, max_updates=20)
        initial = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(initial, 1e-9)
        for _ in range(20):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0, places=12)
        one = torch.optim.AdamW([{"params": [nn.Parameter(torch.zeros(1))], "lr": 1e-4}])
        one_scheduler = make_scheduler(one, max_updates=1)
        self.assertAlmostEqual(one.param_groups[0]["lr"], 1e-9)
        one.step()
        one_scheduler.step()
        self.assertEqual(one.param_groups[0]["lr"], 0.0)
        self.assertEqual(len(config_fingerprint()), 64)


if __name__ == "__main__":
    unittest.main()
