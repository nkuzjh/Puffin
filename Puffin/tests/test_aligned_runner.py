from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from csgo_seen10.inference_utils import run_signature, sha256_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_csgo_aligned.py"
spec = importlib.util.spec_from_file_location("puffin_aligned_runner_tests", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def args(self, *more):
        args = runner.parser().parse_args(["train", "--experiment", runner.EXPERIMENT, *more])
        args.output_root = Path("/tmp/puffin-command-only")
        args.data_root = Path("/tmp/data-command-only")
        args.shared_eval_dir = Path("/tmp/eval-command-only")
        args.prediction_tag = "smoke_native50_eager"
        return args

    def test_multi_gpu_only_product_is_fixed(self):
        for world, micro, accumulate in ((1, 4, 32), (2, 8, 8), (4, 4, 8), (8, 2, 8)):
            with self.subTest(world=world), patch.dict(os.environ, {"NPROC_PER_NODE": str(world), "NNODES": "1"}):
                command = runner.training_command(self.args("--micro-batch-size", str(micro)))
                self.assertEqual(command[command.index("--gradient-accumulation-steps") + 1], str(accumulate))
                self.assertEqual("torch.distributed.run" in command, world > 1)

    def test_reject_wrong_effective_batch(self):
        with patch.dict(os.environ, {"NPROC_PER_NODE": "4", "NNODES": "1"}):
            with self.assertRaisesRegex(ValueError, "Effective batch"):
                runner.training_command(self.args("--gradient-accumulation-steps", "4"))

    def test_smoke_infer_is_explicitly_diagnostic(self):
        command = runner.inference_command(self.args(), smoke=True)
        self.assertIn("--allow-smoke-checkpoint", command)
        self.assertIn("--max-samples", command)
        self.assertEqual(command[command.index("--steps") + 1], "50")
        self.assertEqual(command[command.index("--checkpoint") + 1], "/tmp/puffin-command-only/seed_42/checkpoints/latest.pth")

    def test_dry_run_does_not_start_process(self):
        with patch.object(runner.subprocess, "run") as process:
            runner.launch(["python", "never-run.py"], dry_run=True)
            process.assert_not_called()


class FormalPredictionGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = runner.parser().parse_args([
            "eval", "--experiment", runner.EXPERIMENT,
            "--selection", "late", "--prediction-tag", "native50_compiled_b16",
        ])
        self.args.output_root = self.root
        self.args.data_root = self.root / "data"
        self.args.shared_eval_dir = self.root / "eval"
        checkpoint = self.root / "seed_42" / "checkpoints" / "late.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"full aligned checkpoint")
        self.checkpoint_hash = sha256_file(checkpoint)
        self.audit = {"maps": ["cs_agency", "de_dust2"],
                      "identity": {"sha256": "published-data-identity"}}
        for task, count in (("discrete", 20000), ("continuous", 12800)):
            fields = {
                "experiment": runner.EXPERIMENT,
                "task": task,
                "selection": "late",
                "prediction_tag": "native50_compiled_b16",
                "checkpoint_sha256": self.checkpoint_hash,
                "data_identity_sha256": self.audit["identity"]["sha256"],
                "maps": self.audit["maps"],
                "seed": 42,
                "steps": 50,
                "cfg_scale": 4.5,
                "pose_mode": "text",
                "max_samples": None,
                "samples_selected": count,
            }
            manifest = {
                **fields,
                "signature_fields": dict(fields),
                "run_signature": run_signature(fields),
                "complete": True,
                "smoke_only": False,
                "target_read": False,
                "present_samples": count,
            }
            self._write(task, manifest)

    def _path(self, task):
        return (self.root / "seed_42" / "predictions" / "late" /
                "native50_compiled_b16" / task / "inference_manifest.json")

    def _read(self, task):
        return json.loads(self._path(task).read_text(encoding="utf-8"))

    def _write(self, task, manifest):
        path = self._path(task)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def _check(self):
        with patch("csgo_seen10.contracts.audit_benchmark", return_value=self.audit):
            return runner.require_formal_predictions(self.args, ("discrete", "continuous"))

    def test_complete_two_tracks_with_same_checkpoint_pass(self):
        self.assertEqual(self._check(), self.checkpoint_hash)

    def test_smoke_manifest_is_rejected(self):
        manifest = self._read("discrete")
        manifest["smoke_only"] = True
        self._write("discrete", manifest)
        with self.assertRaisesRegex(ValueError, "Not complete native-protocol"):
            self._check()

    def test_signed_partial_sample_count_is_rejected(self):
        manifest = self._read("continuous")
        manifest["samples_selected"] = 12799
        manifest["signature_fields"]["samples_selected"] = 12799
        manifest["run_signature"] = run_signature(manifest["signature_fields"])
        manifest["present_samples"] = 12799
        self._write("continuous", manifest)
        with self.assertRaisesRegex(ValueError, "Not complete native-protocol"):
            self._check()

    def test_signature_field_and_top_level_tampering_are_rejected(self):
        manifest = self._read("discrete")
        manifest["signature_fields"]["steps"] = 28
        self._write("discrete", manifest)
        with self.assertRaisesRegex(ValueError, "Invalid aligned inference signature"):
            self._check()

        manifest = self._read("discrete")
        manifest["signature_fields"]["steps"] = 50
        manifest["steps"] = 28
        self._write("discrete", manifest)
        with self.assertRaisesRegex(ValueError, "Manifest differs from its signed fields"):
            self._check()

    def test_other_checkpoint_in_one_track_is_rejected(self):
        manifest = self._read("continuous")
        manifest["checkpoint_sha256"] = "different-checkpoint-sha256"
        manifest["signature_fields"]["checkpoint_sha256"] = "different-checkpoint-sha256"
        manifest["run_signature"] = run_signature(manifest["signature_fields"])
        self._write("continuous", manifest)
        with self.assertRaisesRegex(ValueError, "Not complete native-protocol"):
            self._check()


if __name__ == "__main__":
    unittest.main()
