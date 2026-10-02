#!/usr/bin/env python3
"""Queue one follow-up in the existing Codex thread at a chosen time.

This scheduler does not inspect training or start a new Codex thread. A failed
submission stays failed until an operator configures a new event explicitly.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


CONTROL = Path(__file__).resolve().parents[1] / "outputs/launch_control"
STATE = CONTROL / "puffin_scheduled_check.json"
LOCK = CONTROL / "puffin_scheduled_check.lock"
PROMPT = CONTROL / "puffin_scheduled_check_prompt.txt"
CODEX = Path("/home/jiahao/.vscode-server/extensions/openai.chatgpt-26.5928.31416/bin/linux-x86_64/codex")
DEFAULT_DUE = "2026-10-02T07:50:00+08:00"
QUEUE_RESULT = re.compile(r"Queued message ([0-9a-fA-F-]{36}) for thread ([0-9a-fA-F-]{36})\.")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def due_time(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid ISO time: {value}") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("due time must include a UTC offset")
    return result


@contextmanager
def locked(lock_path: Path = LOCK):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_state(path: Path = STATE) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def write_state(state: dict, path: Path = STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def configure(thread: str, prompt: Path = PROMPT, codex: Path = CODEX,
              due: str = DEFAULT_DUE, event_id: str | None = None,
              state_path: Path = STATE, lock_path: Path = LOCK) -> dict:
    thread = str(uuid.UUID(thread))
    due_time(due)
    if not prompt.is_file() or not prompt.read_text(encoding="utf-8").strip():
        raise ValueError(f"prompt missing or empty: {prompt}")
    if not codex.is_file():
        raise ValueError(f"Codex binary missing: {codex}")
    event_id = str(uuid.UUID(event_id)) if event_id else str(uuid.uuid4())
    with locked(lock_path):
        state = read_state(state_path)
        if state.get("phase") in ("sending", "uncertain"):
            raise ValueError("previous submission is in flight or ambiguous; inspect before rearming")
        used = {row.get("event_id") for row in state.get("history", [])}
        used.add(state.get("event_id"))
        if event_id in used:
            raise ValueError(f"event ID already used: {event_id}")
        state.update({
            "event_id": event_id, "thread": thread, "prompt": str(prompt.resolve()),
            "codex": str(codex.resolve()), "due": due, "enabled": True,
            "phase": "armed", "configured_at": timestamp(),
        })
        write_state(state, state_path)
    return state


def cancel(state_path: Path = STATE, lock_path: Path = LOCK) -> dict:
    with locked(lock_path):
        state = read_state(state_path)
        if state:
            state["enabled"] = False
            if state.get("phase") == "armed":
                state["phase"] = "cancelled"
            state["cancelled_at"] = timestamp()
            write_state(state, state_path)
    return state


def submit_due(state_path: Path = STATE, lock_path: Path = LOCK,
               current_time: datetime | None = None, runner=subprocess.run) -> dict:
    """Atomically claim and submit one due event; no retries after uncertainty."""
    with locked(lock_path):
        state = read_state(state_path)
        if not state or not state.get("enabled") or state.get("phase") != "armed":
            return state
        if (current_time or datetime.now(timezone.utc)) < due_time(state["due"]):
            return state
        # Record the claim before invoking the external CLI. A crash thereafter
        # leaves an ambiguous state, so a second scheduler cannot double-send.
        state["phase"] = "sending"
        state["claimed_at"] = timestamp()
        write_state(state, state_path)
        try:
            prompt = Path(state["prompt"]).read_text(encoding="utf-8")
            if not prompt.strip():
                raise ValueError("prompt is empty")
            result = runner([state["codex"], "queue", "--thread", state["thread"],
                             "--message", prompt], text=True, capture_output=True,
                            check=False, timeout=120)
            output = (result.stdout or "")[-4000:]
            error = (result.stderr or "")[-4000:]
            match = QUEUE_RESULT.search(output)
            message_id = match.group(1) if match and match.group(2) == state["thread"] else None
            delivery = {
                "event_id": state["event_id"], "thread": state["thread"],
                "attempted_at": state["claimed_at"], "finished_at": timestamp(),
                "returncode": result.returncode, "message_id": message_id,
                "stdout": output, "stderr": error,
                "phase": ("delivered" if message_id and result.returncode == 0 else
                          "uncertain" if result.returncode == 0 else "failed"),
            }
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            delivery = {
                "event_id": state["event_id"], "thread": state["thread"],
                "attempted_at": state["claimed_at"], "finished_at": timestamp(),
                "returncode": None, "message_id": None, "error": str(exc),
                "phase": "uncertain" if isinstance(exc, subprocess.TimeoutExpired) else "failed",
            }
        state.setdefault("history", []).append(delivery)
        state.update({"phase": delivery["phase"], "enabled": False,
                      "finished_at": delivery["finished_at"],
                      "last_delivery": delivery})
        write_state(state, state_path)
        return state


def run(state_path: Path = STATE, lock_path: Path = LOCK) -> int:
    while True:
        state = submit_due(state_path, lock_path)
        if not state or not state.get("enabled") or state.get("phase") != "armed":
            return 1 if state.get("phase") in ("failed", "uncertain", "sending") else 0
        remaining = (due_time(state["due"]) - datetime.now(timezone.utc)).total_seconds()
        time.sleep(min(60, max(0.1, remaining)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("configure", help="arm one future queue event")
    setup.add_argument("--thread", required=True, help="existing Codex thread UUID")
    setup.add_argument("--prompt", type=Path, default=PROMPT)
    setup.add_argument("--codex", type=Path, default=CODEX)
    setup.add_argument("--at", default=DEFAULT_DUE, help="ISO time with UTC offset")
    setup.add_argument("--event-id", help="optional unique UUID for this event")
    commands.add_parser("status", help="print current JSON state")
    commands.add_parser("cancel", help="disable the currently armed event")
    commands.add_parser("run", help="wait until due, then queue once")
    args = parser.parse_args()
    try:
        if args.command == "configure":
            result = configure(args.thread, args.prompt, args.codex, args.at, args.event_id)
        elif args.command == "cancel":
            result = cancel()
        elif args.command == "status":
            with locked():
                result = read_state()
        else:
            return run()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
