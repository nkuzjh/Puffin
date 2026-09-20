from __future__ import annotations

import os
import pickle
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from mmengine.logging import HistoryBuffer

from csgo_seen10.checkpoint_utils import (
    configure_native_resume_load,
    load_trusted_state_dict,
)
from csgo_seen10 import dataset as dataset_module
from csgo_seen10.dataset import CollateSeen10, CsgoSeen10Dataset, collate_seen10
from csgo_seen10.inference_utils import (
    atomic_write_json,
    is_valid_rgb_jpeg,
    load_matching_manifest,
    run_signature,
    write_rgb_jpeg,
)
from src.models.puffin.model import Qwen2p5RadioStableDiffusion3HFDynamic


class _FakeBenchmarkData:
    def __init__(self, row):
        self.row = row

    def rows(self, split, maps=None, max_samples=None):
        return [self.row]


class Seen10DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        radar = self.root / "radar.png"
        target = self.root / "target.jpg"
        Image.new("RGB", (12, 9), color=(40, 80, 120)).save(radar)
        Image.new("RGB", (10, 8), color=(120, 80, 40)).save(target)
        self.row = {
            "sample_id": "cs_agency/file_num3_frame_0042",
            "map_name": "cs_agency",
            "file_frame": "file_num3_frame_0042",
            "image_path": str(target),
            "radar_path": str(radar),
            "pose": [0.25, 0.5, 0.75, 0.125, 0.875],
            "clip_id": None,
            "frame_index": None,
        }
        self.protocol = SimpleNamespace(
            BenchmarkData=lambda _data_root: _FakeBenchmarkData(self.row)
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_training_batch_reads_target_and_numeric_pose(self):
        with patch.object(dataset_module, "_load_protocol", return_value=self.protocol):
            dataset = CsgoSeen10Dataset(data_root=str(self.root), split="seen_train")
            batch = CollateSeen10()([dataset[0]])["data"]["cam2image"]

        self.assertEqual(tuple(batch["pixel_values"][0].shape), (3, 448, 448))
        self.assertEqual(tuple(batch["cam_values"][0][0].shape), (3, 448, 448))
        self.assertEqual(tuple(batch["pose_values"].shape), (1, 5))
        self.assertEqual(batch["metadata"][0]["sample_id"], self.row["sample_id"])
        self.assertEqual(batch["texts"], ["Generate a first-person view from the provided radar map. Map name: cs_agency."])

    def test_target_free_inference_never_opens_target(self):
        self.row["image_path"] = str(self.root / "does-not-exist.jpg")
        with patch.object(dataset_module, "_load_protocol", return_value=self.protocol):
            dataset = CsgoSeen10Dataset(
                data_root=str(self.root), split="seen_continuous", include_target=False
            )
            sample = dataset[0]
        self.assertNotIn("pixel_values", sample)
        self.assertEqual(sample["pose_values"].tolist(), self.row["pose"])

    def test_collator_rejects_mixed_target_contracts(self):
        with patch.object(dataset_module, "_load_protocol", return_value=self.protocol):
            with_target = CsgoSeen10Dataset(data_root=str(self.root), split="seen_train")[0]
            without_target = CsgoSeen10Dataset(
                data_root=str(self.root), split="seen_discrete_test", include_target=False
            )[0]
        with self.assertRaises(ValueError):
            collate_seen10([with_target, without_target])


class Seen10InferenceArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_trusted_loader_accepts_mmengine_history_metadata(self):
        checkpoint_path = self.root / "iter_1.pth"
        expected = {"weight": torch.tensor([1.0, 2.0])}
        torch.save(
            {
                "state_dict": expected,
                "meta": {"loss_history": HistoryBuffer([1.0], [1])},
            },
            checkpoint_path,
        )

        with self.assertRaises(pickle.UnpicklingError):
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        loaded = load_trusted_state_dict(checkpoint_path)
        self.assertEqual(set(loaded), {"weight"})
        torch.testing.assert_close(loaded["weight"], expected["weight"])

    def test_trusted_loader_accepts_plain_state_dict(self):
        checkpoint_path = self.root / "weights_only.pth"
        expected = {"weight": torch.tensor([3.0])}
        torch.save(expected, checkpoint_path)
        loaded = load_trusted_state_dict(checkpoint_path)
        torch.testing.assert_close(loaded["weight"], expected["weight"])

    def test_native_resume_configures_full_checkpoint_load(self):
        child_env = {}
        configure_native_resume_load(child_env)
        self.assertEqual(child_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"], "1")
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            configure_native_resume_load({"TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"})

        checkpoint_path = self.root / "resume.pth"
        torch.save(
            {"meta": {"seed": 23, "history": HistoryBuffer([2.0], [1])}},
            checkpoint_path,
        )
        # The same process-level override inherited by the native runner makes
        # XTuner's seed read and MMEngine's checkpoint loader accept this metadata.
        with patch.dict(
            os.environ,
            {
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "0",
            },
        ):
            loaded = torch.load(checkpoint_path, map_location="cpu")
            from mmengine.runner.checkpoint import CheckpointLoader
            from xtuner.tools.utils import get_seed_from_checkpoint

            self.assertEqual(get_seed_from_checkpoint(str(checkpoint_path)), 23)
            runner_checkpoint = CheckpointLoader.load_checkpoint(
                str(checkpoint_path), map_location="cpu"
            )
        self.assertIsInstance(loaded["meta"]["history"], HistoryBuffer)
        self.assertIsInstance(
            runner_checkpoint["meta"]["history"], HistoryBuffer
        )

    def test_generated_output_is_448_rgb_jpeg_and_valid_file_is_skipped(self):
        output_path = self.root / "cs_agency" / "file_num3_frame_0042.jpg"
        tensor = torch.zeros((3, 448, 448), dtype=torch.float32)
        tensor[0].fill_(-1.0)
        tensor[1].fill_(0.0)
        tensor[2].fill_(1.0)

        self.assertTrue(write_rgb_jpeg(tensor, output_path))
        self.assertTrue(is_valid_rgb_jpeg(output_path))
        with Image.open(output_path) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (448, 448))
        self.assertFalse(write_rgb_jpeg(tensor, output_path))

    def test_invalid_existing_output_is_never_overwritten(self):
        output_path = self.root / "bad.jpg"
        original = b"preserve this invalid user file"
        output_path.write_bytes(original)

        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            write_rgb_jpeg(torch.zeros((3, 448, 448)), output_path)
        self.assertEqual(output_path.read_bytes(), original)

    def test_manifest_signature_accepts_resume_only_for_same_run(self):
        manifest_path = self.root / "inference_manifest.json"
        first_signature = run_signature({"checkpoint_sha256": "base-a", "seed": 7})
        atomic_write_json({"run_signature": first_signature}, manifest_path)

        self.assertEqual(
            load_matching_manifest(manifest_path, first_signature)["run_signature"],
            first_signature,
        )
        with self.assertRaisesRegex(ValueError, "different checkpoint/seed/settings"):
            load_matching_manifest(
                manifest_path,
                run_signature({"checkpoint_sha256": "base-b", "seed": 7}),
            )

    def test_pose_mlp_learns_through_frozen_language_path(self):
        pose_mlp = torch.nn.Sequential(
            torch.nn.Linear(5, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8)
        )
        torch.nn.init.zeros_(pose_mlp[-1].weight)
        torch.nn.init.zeros_(pose_mlp[-1].bias)
        meta_queries = torch.nn.Parameter(torch.randn(1, 8))
        frozen_language_layer = torch.nn.Linear(8, 8)
        frozen_language_layer.requires_grad_(False)

        pose = torch.tensor([[0.2, 0.4, 0.6, 0.8, 0.1]])
        conditioned_query = meta_queries[None] + pose_mlp(pose)[:, None]
        # Match prepare_forward_input's indexed copy from the query source.
        language_inputs = torch.zeros((1, 3, 8))
        query_position = torch.tensor([[False, True, False]])
        language_inputs[query_position] = conditioned_query.reshape(-1, 8)
        loss = frozen_language_layer(language_inputs).square().mean()
        loss.backward()

        self.assertTrue(all(parameter.requires_grad for parameter in pose_mlp.parameters()))
        self.assertIsNotNone(pose_mlp[-1].weight.grad)
        self.assertGreater(float(pose_mlp[-1].weight.grad.abs().sum()), 0.0)
        self.assertIsNone(frozen_language_layer.weight.grad)


class Seen10ConfigTests(unittest.TestCase):
    def test_bfloat16_amp_disables_loss_scaling(self):
        from mmengine.config import Config

        config_path = Path(__file__).resolve().parents[1] / "configs/pipelines/csgo_seen10.py"
        with patch.dict(os.environ, {"PUFFIN_SEEN10_SMOKE": "1"}):
            config = Config.fromfile(str(config_path))

        self.assertEqual(config.optim_wrapper.dtype, "bfloat16")
        self.assertEqual(config.optim_wrapper.loss_scale, {"enabled": False})
        self.assertEqual(config.model.val_autocast_dtype, "bfloat16")

        # Exercise the same disabled GradScaler path used by MMEngine; with
        # scaling disabled, BF16 gradients pass through unscale_/step directly.
        scaler = torch.amp.GradScaler("cuda", **config.optim_wrapper.loss_scale)
        parameter = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.bfloat16))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler.scale(parameter.square()).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        self.assertLess(float(parameter), 2.0)


class PuffinForwardInputTests(unittest.TestCase):
    def test_predict_loss_uses_cuda_bfloat16_autocast_only_when_configured(self):
        amp_calls = []

        @contextmanager
        def record_autocast(**kwargs):
            amp_calls.append(kwargs)
            yield

        model = SimpleNamespace(
            device=torch.device("cuda"),
            val_autocast_dtype=torch.bfloat16,
            compute_loss=lambda data_dict: {"loss_cam2image": torch.tensor(1.0)},
        )
        data = {"cam2image": {"texts": ["smoke"]}}
        with patch("torch.autocast", side_effect=record_autocast):
            Qwen2p5RadioStableDiffusion3HFDynamic.forward(model, data, mode="predict")
        self.assertEqual(
            amp_calls,
            [{"device_type": "cuda", "dtype": torch.bfloat16}],
        )

        # CPU validation and the training loss entry point retain their prior
        # no-extra-autocast behavior; generation does not use mode='predict'.
        model.device = torch.device("cpu")
        with patch("torch.autocast", side_effect=record_autocast) as autocast_mock:
            Qwen2p5RadioStableDiffusion3HFDynamic.forward(model, data, mode="predict")
            Qwen2p5RadioStableDiffusion3HFDynamic.forward(model, data, mode="loss")
        autocast_mock.assert_not_called()

    def test_forward_inputs_align_query_and_image_embeddings(self):
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        token_embeddings = torch.nn.Embedding(8, 4, device=device, dtype=torch.bfloat16)
        llm = SimpleNamespace(
            config=SimpleNamespace(hidden_size=4),
            get_input_embeddings=lambda: token_embeddings,
        )
        model = SimpleNamespace(
            llm=llm,
            num_queries=1,
            image_token_id=2,
            device=device,
            dtype=torch.bfloat16,
        )
        query_embeds = torch.randn(1, 1, 4, dtype=torch.float32, requires_grad=True)
        image_embeds = torch.randn(1, 1, 4, dtype=torch.float32, requires_grad=True)

        inputs = Qwen2p5RadioStableDiffusion3HFDynamic.prepare_forward_input(
            model,
            query_embeds=query_embeds,
            input_ids=torch.tensor([[2, 3]]),
            image_embeds=image_embeds,
            attention_mask=torch.ones((1, 2), dtype=torch.long),
        )
        inputs_embeds = inputs["inputs_embeds"]

        self.assertEqual(inputs_embeds.device, device)
        self.assertEqual(inputs_embeds.dtype, torch.bfloat16)
        torch.testing.assert_close(
            inputs_embeds[0, 0], image_embeds[0, 0].to(device=device, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            inputs_embeds[0, 2], query_embeds[0, 0].to(device=device, dtype=torch.bfloat16)
        )

        inputs_embeds.float().sum().backward()
        self.assertIsNotNone(query_embeds.grad)
        self.assertIsNotNone(image_embeds.grad)

        cached_inputs = Qwen2p5RadioStableDiffusion3HFDynamic.prepare_forward_input(
            model,
            query_embeds=query_embeds.detach(),
            input_ids=None,
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            past_key_values=object(),
            append_queries=False,
        )
        self.assertEqual(cached_inputs["inputs_embeds"].device, device)
        self.assertEqual(cached_inputs["inputs_embeds"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
