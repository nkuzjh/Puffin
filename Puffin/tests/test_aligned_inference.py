from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch

from csgo_seen10.inference_utils import (
    atomic_write_json,
    inspect_prediction_root,
    pad_inference_batch,
    run_signature,
    sha256_file,
    write_rgb_jpeg,
)
from src.models.stable_diffusion3.transformer_sd3_dynamic import SD3Transformer2DModel


class AlignedInferenceTests(unittest.TestCase):
    def test_checkpoint_requires_snapshot_and_training_source_identity(self):
        import infer_seen10
        from csgo_seen10.aligned import OFFICIAL_BASE_SHA256, OFFICIAL_VAE_SHA256
        from csgo_seen10.prompts import PROMPT_SHA256

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scheduler = root / "sd3" / "scheduler" / "scheduler_config.json"
            vae = root / "sd3" / "vae" / "config.json"
            qwen = root / "qwen" / "config.json"
            radio = root / "radio" / "config.json"
            for path in (scheduler, vae, qwen, radio):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"file": path.name}) + "\n")
            source_names = (
                "csgo_seen10/aligned.py", "csgo_seen10/aligned_train.py",
                "csgo_seen10/aligned_samplers.py", "csgo_seen10/dataset.py",
                "csgo_seen10/prompts.py", "csgo_seen10/contracts.py",
                "csgo_seen10/paths.py", "csgo_seen10/model_builders.py",
                "src/models/puffin/model.py", "src/models/connector/modeling_connector.py",
                "src/models/connector/configuration_connector.py",
                "src/models/stable_diffusion3/transformer_sd3_dynamic.py",
                "configs/pipelines/csgo_seen10_exp32gen_aligned.py",
                "configs/pipelines/csgo_seen10.py",
                "configs/models/qwen2_5_1_5b_radio_sd3_dynamic_puffin.py",
                "requirements_seen10.txt",
            )
            benchmark = {"sha256": "published-bundle"}
            identities = {
                "benchmark": benchmark,
                "prompt_sha256": PROMPT_SHA256,
                "assets": {"puffin_base_sha256": OFFICIAL_BASE_SHA256,
                           "puffin_vae_sha256": OFFICIAL_VAE_SHA256},
                "metadata": {
                    "sd3/scheduler/scheduler_config.json": sha256_file(scheduler),
                    "sd3/vae/config.json": sha256_file(vae),
                    "qwen/config.json": sha256_file(qwen),
                    "radio/config.json": sha256_file(radio),
                },
                "source": {name: sha256_file(infer_seen10.PROJECT_ROOT / name)
                           for name in source_names},
            }
            with patch.dict("os.environ", {"PUFFIN_QWEN_PATH": str(qwen.parent),
                                        "PUFFIN_RADIO_PATH": str(radio.parent)}):
                payload = {"identities": identities, "resolved_configs": {
                    "scheduler": json.loads(scheduler.read_text()),
                    "vae": json.loads(vae.read_text()),
                }}
                infer_seen10._verify_aligned_checkpoint_data(
                    payload, benchmark, scheduler
                )
                identities["source"][source_names[0]] = "stale-source"
                with self.assertRaisesRegex(ValueError, "training source fingerprint"):
                    infer_seen10._verify_aligned_checkpoint_data(
                        payload, benchmark, scheduler
                    )
                identities["source"][source_names[0]] = sha256_file(
                    infer_seen10.PROJECT_ROOT / source_names[0]
                )
                vae.write_text('{"changed": true}\n')
                with self.assertRaisesRegex(ValueError, "model metadata differs"):
                    infer_seen10._verify_aligned_checkpoint_data(
                        payload, benchmark, scheduler
                    )

    def test_scheduler_location_does_not_change_prediction_signature(self):
        from infer_seen10 import _aligned_signature_fields

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.py").write_text("# fixed protocol\n")
            first = root / "first" / "scheduler" / "scheduler_config.json"
            second = root / "second" / "scheduler" / "scheduler_config.json"
            for path in (first, second):
                path.parent.mkdir(parents=True)
                path.write_text('{"shift": 3.0}\n')
                vae_config = path.parents[1] / "vae" / "config.json"
                vae_config.parent.mkdir()
                vae_config.write_text('{"scaling_factor": 1.5305}\n')
            args = SimpleNamespace(
                selection="late", prediction_tag="native50_compiled_b16",
                seed=42, steps=50, cfg_scale=4.5, maps=None,
                max_samples=None, shared_eval_dir=str(root),
                benchmark_identity_sha256="fixed-bundle", inference_engine="compiled",
                batch_size=16, decoder_chunk_size=1,
                training_source_fingerprint="fixed-source",
            )
            dataset = SimpleNamespace(
                rows=[{"sample_id": "m/frame1"}],
                benchmark=SimpleNamespace(maps=["m"]),
                __len__=lambda: 1,
            )
            # SimpleNamespace does not implement len; use a minimal class.
            class Dataset:
                rows = dataset.rows
                benchmark = dataset.benchmark

                def __len__(self):
                    return len(self.rows)

            left = _aligned_signature_fields(
                Dataset(), split="seen_discrete_test", task_name="discrete",
                args=args, checkpoint_sha256="checkpoint", scheduler_file=first,
            )
            right = _aligned_signature_fields(
                Dataset(), split="seen_discrete_test", task_name="discrete",
                args=args, checkpoint_sha256="checkpoint", scheduler_file=second,
            )
            self.assertEqual(run_signature(left), run_signature(right))

    def test_unequal_dense_radar_matches_native_list_tokens(self):
        torch.manual_seed(31)
        model = SD3Transformer2DModel(
            sample_size=56,
            patch_size=2,
            in_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=1,
            joint_attention_dim=8,
            caption_projection_dim=8,
            pooled_projection_dim=8,
            out_channels=4,
            pos_embed_max_size=56,
        ).eval()
        target = torch.randn(2, 4, 56, 56)
        radar = torch.randn(2, 1, 4, 28, 28)
        context = torch.randn(2, 5, 8)
        pooled = torch.randn(2, 8)
        timestep = torch.ones(2)
        common = dict(
            hidden_states=target,
            encoder_hidden_states=context,
            pooled_projections=pooled,
            timestep=timestep,
            return_dict=False,
        )
        with torch.inference_mode():
            native = model(cond_hidden_states=[[radar[0, 0]], [radar[1, 0]]], **common)[0]
            dense = model(cond_hidden_states=radar, **common)[0]
        self.assertEqual(tuple(dense.shape), (2, 4, 56, 56))
        torch.testing.assert_close(dense, native, rtol=2e-5, atol=2e-5)

    def test_text_pose_padding_has_no_numeric_pose(self):
        batch = {"data": {"cam2image": {
            "cam_values": [[torch.zeros(3, 224, 224)]],
            "texts": ["pose in text"],
            "metadata": [{"sample_id": "a", "map_name": "m", "file_frame": "f"}],
        }}}
        padded = pad_inference_batch(batch, 16)["data"]["cam2image"]
        self.assertEqual(len(padded["texts"]), 16)
        self.assertNotIn("pose_values", padded)
        self.assertNotIn("pixel_values", padded)

    def test_resume_rejects_mixed_and_corrupt_prediction_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "discrete"
            image = root / "gen_imgs" / "m" / "f.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"old output")
            signature = run_signature({"experiment": "aligned", "seed": 42})
            with self.assertRaisesRegex(ValueError, "nonempty prediction root"):
                inspect_prediction_root(root, signature, {"m/f.jpg"})
            atomic_write_json({"run_signature": signature}, root / "inference_manifest.json")
            with self.assertRaisesRegex(ValueError, "Invalid existing prediction"):
                inspect_prediction_root(root, signature, {"m/f.jpg"})
            image.unlink()
            write_rgb_jpeg(torch.zeros(3, 448, 448), image)
            self.assertIsNotNone(inspect_prediction_root(root, signature, {"m/f.jpg"}))
            with self.assertRaisesRegex(ValueError, "different checkpoint/seed/settings"):
                inspect_prediction_root(root, run_signature({"seed": 7}), {"m/f.jpg"})


if __name__ == "__main__":
    unittest.main()
