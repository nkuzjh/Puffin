"""Scheduler tests never call the real Codex CLI."""

import importlib.util
import subprocess
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/schedule_csgo_aligned_check.py"
spec = importlib.util.spec_from_file_location("schedule_csgo_aligned_check", SCRIPT)
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = self.root / "state.json"
        self.lock = self.root / "state.lock"
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text("Check Puffin\nRead docs if start failed.", encoding="utf-8")
        self.codex = self.root / "codex"
        self.codex.write_text("stub", encoding="utf-8")
        self.thread = str(uuid.uuid4())
        self.due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()

    def configure(self, event_id=None):
        return schedule.configure(self.thread, self.prompt, self.codex, self.due,
                                  event_id, self.state, self.lock)

    def submit(self, runner):
        return schedule.submit_due(self.state, self.lock, runner=runner)

    def test_delivers_once_with_literal_prompt_and_records_message_id(self):
        configured = self.configure()
        message_id = str(uuid.uuid4())
        runner = Mock(return_value=SimpleNamespace(
            returncode=0,
            stdout=f"Queued message {message_id} for thread {self.thread}.\n",
            stderr=""))
        result = self.submit(runner)
        self.assertEqual(result["phase"], "delivered")
        self.assertFalse(result["enabled"])
        self.assertEqual(result["history"][0]["message_id"], message_id)
        self.assertEqual(result["history"][0]["event_id"], configured["event_id"])
        self.assertEqual(runner.call_args.args[0], [str(self.codex), "queue", "--thread",
                                                    self.thread, "--message", self.prompt.read_text()])
        self.assertTrue(runner.call_args.kwargs["capture_output"])
        self.submit(runner)
        runner.assert_called_once()

    def test_failure_is_terminal_and_rearm_preserves_delivery_history(self):
        first = self.configure()
        runner = Mock(return_value=SimpleNamespace(returncode=7, stdout="", stderr="offline"))
        failed = self.submit(runner)
        self.assertEqual(failed["phase"], "failed")
        self.assertEqual(failed["history"][0]["returncode"], 7)
        self.submit(runner)
        runner.assert_called_once()
        with self.assertRaisesRegex(ValueError, "already used"):
            self.configure(first["event_id"])
        second = self.configure()
        self.assertNotEqual(second["event_id"], first["event_id"])
        self.assertEqual(second["history"], failed["history"])
        self.assertEqual(second["phase"], "armed")

    def test_cancel_prevents_queue_and_missing_prompt_refuses_arming(self):
        self.configure()
        cancelled = schedule.cancel(self.state, self.lock)
        self.assertEqual(cancelled["phase"], "cancelled")
        runner = Mock()
        self.submit(runner)
        runner.assert_not_called()
        self.prompt.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "prompt missing or empty"):
            self.configure()

    def test_early_check_does_not_queue(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        schedule.configure(self.thread, self.prompt, self.codex, future,
                           state_path=self.state, lock_path=self.lock)
        runner = Mock()
        result = self.submit(runner)
        self.assertEqual(result["phase"], "armed")
        runner.assert_not_called()

    def test_preclaimed_event_is_never_retried_after_crash(self):
        state = self.configure()
        state["phase"] = "sending"
        schedule.write_state(state, self.state)
        runner = Mock()
        self.submit(runner)
        runner.assert_not_called()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.configure()

    def test_success_without_matching_thread_ack_is_uncertain(self):
        self.configure()
        runner = Mock(return_value=SimpleNamespace(
            returncode=0, stdout=f"Queued message {uuid.uuid4()} for thread {uuid.uuid4()}.\n", stderr=""))
        result = self.submit(runner)
        self.assertEqual(result["phase"], "uncertain")
        self.assertFalse(result["enabled"])
        self.submit(runner)
        runner.assert_called_once()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.configure()

    def test_timeout_preserves_ambiguous_delivery_without_retry(self):
        self.configure()
        runner = Mock(side_effect=subprocess.TimeoutExpired(["codex", "queue"], 120))
        result = self.submit(runner)
        self.assertEqual(result["phase"], "uncertain")
        self.submit(runner)
        runner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
