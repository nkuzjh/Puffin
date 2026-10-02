"""File and process fakes only: no CUDA work and no training launch."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/watch_csgo_aligned_start.py"
spec = importlib.util.spec_from_file_location("watch_csgo_aligned_start", SCRIPT)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def inspect(self, *, process=None, active=(), complete=(True, "complete"), state=None):
        return watch.inspect(
            known={5: 123}, process_reader=lambda pid: process,
            process_scanner=lambda: list(active), evidence=lambda: complete,
            formal_root=self.root / "formal", control_state=state)

    def test_waits_for_original_process_and_other_same_experiment(self):
        process = {"pid": 5, "start_ticks": 123,
                   "cmdline": "python train_csgo.py --csgo_config exp32_1.yaml"}
        self.assertEqual(self.inspect(process=process)[0], "waiting_unilip")
        self.assertEqual(self.inspect(active=[{"pid": 8}])[0], "waiting_active_run")

    def test_pid_reuse_is_not_original_but_incomplete_training_blocks(self):
        reused = {"pid": 5, "start_ticks": 999, "cmdline": "unrelated"}
        status, _ = self.inspect(process=reused, complete=(False, "no final save"))
        self.assertEqual(status, "unilip_incomplete_needs_review")

    def test_existing_formal_run_or_claim_blocks_duplicate(self):
        formal = self.root / "formal"
        formal.mkdir()
        (formal / "seed_42").mkdir()
        self.assertEqual(self.inspect()[0], "existing_formal_run")
        self.assertEqual(self.inspect(state={"launch_claimed": True})[0], "already_launched")

    def test_success_evidence_requires_final_summary_and_complete_weights(self):
        output = self.root / "unilip"
        output.mkdir()
        log = self.root / "output.log"
        log.write_text("Training completed\n")
        (output / "trainer_state.json").write_text(json.dumps({
            "global_step": 19550, "max_steps": 19550,
            "log_history": [{"train_runtime": 100.0, "train_loss": 0.5}],
        }))
        for name in ("config.json", "mm_projector.bin", "gen_projector.bin"):
            (output / name).write_bytes(b"data")
        header = json.dumps({"tensor": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}).encode()
        (output / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"abcd")
        adapter = output / "lora_adapters/language_model/adapter_model.safetensors"
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(b"adapter")
        self.assertTrue(watch.completion_evidence(output, log)[0])
        (output / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"ab")
        self.assertFalse(watch.completion_evidence(output, log)[0])

    def test_missing_prerequisites_and_exact_launch_recipe(self):
        with patch.object(watch, "WORKSPACE", self.root), patch.object(watch, "ROOT", self.root), \
                patch.object(watch, "DATA_ROOT", self.root / "published_bundle"):
            self.assertTrue(any("calibration/z_calibration.json" in p for p in watch.simple_prerequisites()))
        self.assertEqual(watch.training_command(), [
            "bash", "scripts/run_csgo_seen10.sh", "train", "--experiment",
            "csgo_seen10_exp32gen_aligned", "--seed", "42", "--micro-batch-size", "64",
            "--gradient-accumulation-steps", "1", "--output-root", str(watch.FORMAL_ROOT),
        ])
        self.assertEqual(watch.TRAIN_LOG.name, "puffin_aligned.nohup.out")
        self.assertIn("export CUDA_VISIBLE_DEVICES=\n", watch.prep_command())
        shell = watch.launch_shell_command()
        self.assertLess(shell.index("source scripts/activate_csgo_seen10.sh"),
                        shell.index("exec bash scripts/run_csgo_seen10.sh train"))

    def test_launch_uses_fixed_log_and_detached_training_shell(self):
        class FakeChild:
            pid = 456

        control = self.root / "control"
        fixed_log = self.root / "puffin_aligned.nohup.out"
        fixed_log.write_text("old run\n")
        with patch.object(watch, "CONTROL", control), patch.object(watch, "TRAIN_LOG", fixed_log), \
                patch.object(watch, "inspect", return_value=("ready_for_prerequisites", "complete")), \
                patch.object(watch, "simple_prerequisites", return_value=[]), \
                patch.object(watch, "proc_info", return_value={"start_ticks": 222}), \
                patch.object(watch.subprocess, "Popen", return_value=FakeChild()) as popen:
            state = {}
            watch.launch(state)
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["bash", "-c", watch.launch_shell_command()])
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(kwargs["env"]["NPROC_PER_NODE"], "2")
        self.assertEqual(kwargs["env"]["CSGO_DATA_ROOT"], str(watch.DATA_ROOT))
        self.assertEqual(kwargs["env"]["NCCL_SHM_DISABLE"], "1")
        self.assertEqual(kwargs["stdout"].name, str(fixed_log))
        self.assertTrue(state["launch_claimed"])
        self.assertEqual(state["pid"], 456)
        self.assertEqual(fixed_log.read_text(), "")
        self.assertEqual(Path(state["previous_log_archive"]).read_text(), "old run\n")

    def test_launch_refuses_missing_prerequisite_before_popen(self):
        with patch.object(watch, "inspect", return_value=("ready_for_prerequisites", "complete")), \
                patch.object(watch, "simple_prerequisites", return_value=["missing calibration"]), \
                patch.object(watch.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "missing calibration"):
                watch.launch({})
        popen.assert_not_called()

    def test_preflight_uses_same_data_bundle_and_hides_cuda(self):
        class Result:
            returncode = 0

        bundle = self.root / "published_bundle"
        with patch.object(watch, "DATA_ROOT", bundle), \
                patch.object(watch.subprocess, "run", return_value=Result()) as run:
            self.assertEqual(watch.run_prerequisites(self.root / "preflight.log"), 0)
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(run.call_args.kwargs["env"]["CSGO_DATA_ROOT"], str(bundle))


if __name__ == "__main__":
    unittest.main()
