#!/usr/bin/env python3
"""Start the authorized aligned run once UniLIP and all local gates are complete.

Run detached with the system Python. ``--check-once`` is read-only and never
starts training. The controller keeps its state outside the formal seed folder.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import selectors
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
# The task/UniLIP copy links images and radars outside its directory. The
# shared protocol requires resolved paths inside one complete published bundle.
DATA_ROOT = Path("/data/jiahao/data/csgo_benchmark_v2")
UNILIP = WORKSPACE / "UniLIP"
UNILIP_OUTPUT = UNILIP / "outputs/csgo_1b/exp32_1"
UNILIP_LOG = UNILIP / "wandb/run-20260920_183420-rqtfijcu/files/output.log"
UNILIP_STEP_LOG = UNILIP / "logs/csgo_1b/exp32_1/train_20260920_183417/train.log"
FORMAL_ROOT = ROOT / "outputs/csgo_seen10_exp32gen_aligned/Puffin"
CONTROL = ROOT / "outputs/launch_control"
TRAIN_LOG = ROOT / "puffin_aligned.nohup.out"
EXPECTED = {1224680: 95215179, 1224778: 95215316, 1224779: 95215316}
INTERVAL = 1800


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def proc_info(pid):
    """Return command and Linux start ticks, or None; reject PID reuse."""
    try:
        base = Path("/proc") / str(pid)
        stat = (base / "stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        if fields[0] == "Z":
            return None
        ticks = int(fields[19])
        cmd = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        return {"pid": pid, "start_ticks": ticks, "cmdline": cmd}
    except (FileNotFoundError, ProcessLookupError):
        return None


def relevant_processes():
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            info = proc_info(int(entry.name))
        except (PermissionError, OSError, ValueError):
            continue
        if info is None:
            continue
        cmd = info["cmdline"]
        if ("train_csgo.py" in cmd and ("exp32_1.yaml" in cmd or "outputs/csgo_1b/exp32_1" in cmd)) or (
            "run_csgo_aligned.py" in cmd and "csgo_seen10_exp32gen_aligned" in cmd and "train" in cmd
        ) or ("train_seen10.py" in cmd and "csgo_seen10_exp32gen_aligned" in cmd) or (
            "run_csgo_seen10.sh" in cmd and "csgo_seen10_exp32gen_aligned" in cmd and "train" in cmd
        ):
            result.append(info)
    return result


def completion_evidence(output=UNILIP_OUTPUT, log=UNILIP_LOG):
    state_path = output / "trainer_state.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        return False, "UniLIP root trainer_state.json unavailable or invalid"
    if state.get("global_step") != 19550 or state.get("max_steps") != 19550:
        return False, f"UniLIP root steps are {state.get('global_step')}/{state.get('max_steps')}, expected 19550/19550"
    if not any(isinstance(row, dict) and row.get("train_runtime") and row.get("train_loss") is not None
               for row in state.get("log_history", [])):
        return False, "UniLIP root trainer state lacks final train summary"
    for name in ("config.json", "mm_projector.bin", "gen_projector.bin"):
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            return False, f"UniLIP final model file missing or empty: {path}"
    index = output / "model.safetensors.index.json"
    try:
        weights = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
        if not weights or any(Path(name).name != name for name in weights):
            return False, "UniLIP final model index has invalid shards"
        for name in weights:
            path = output / name
            with path.open("rb") as stream:
                header_size = int.from_bytes(stream.read(8), "little")
                if not 0 < header_size < 100_000_000:
                    return False, f"invalid safetensors header: {path}"
                header = json.loads(stream.read(header_size))
                end = max((item["data_offsets"][1] for item in header.values()
                           if isinstance(item, dict) and "data_offsets" in item), default=0)
                if end == 0 or 8 + header_size + end > path.stat().st_size:
                    return False, f"incomplete safetensors shard: {path}"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"UniLIP final weights invalid: {exc}"
    adapters = output / "lora_adapters"
    if not any(p.is_file() and p.stat().st_size > 0 for p in adapters.rglob("adapter_model.*")):
        return False, "UniLIP final LoRA adapter files absent"
    # The final TrainerState train summary is written by trainer.train() on
    # normal return; logger wording differs across Transformers versions.
    return True, "19550 steps, final train summary and root model save verified"


def inspect(known=EXPECTED, process_reader=proc_info, process_scanner=relevant_processes,
            evidence=completion_evidence, formal_root=FORMAL_ROOT, control_state=None):
    """Pure gate apart from reads; dependencies can be replaced in tests."""
    if control_state and control_state.get("launch_claimed"):
        return "already_launched", "launch was already claimed; controller will never start a duplicate"
    for pid, ticks in known.items():
        info = process_reader(pid)
        if info and info["start_ticks"] == ticks:
            cmd = info["cmdline"]
            if "exp32_1.yaml" not in cmd or ("train_csgo.py" not in cmd and "torchrun" not in cmd):
                return "needs_review", f"known UniLIP PID {pid} has unexpected command"
            return "waiting_unilip", f"known UniLIP process {pid} remains active"
    active = process_scanner()
    if active:
        return "waiting_active_run", f"relevant training process remains active: {[p['pid'] for p in active]}"
    complete, detail = evidence()
    if not complete:
        return "unilip_incomplete_needs_review", detail
    if formal_root.is_symlink() or (formal_root.exists() and (
            not formal_root.is_dir() or any(formal_root.iterdir()))):
        return "existing_formal_run", f"formal output is non-empty or a symlink: {formal_root}"
    return "ready_for_prerequisites", detail


def simple_prerequisites(root=ROOT):
    paths = [
        WORKSPACE / "csgo_benchmark_v2_eval_general/protocol.py",
        WORKSPACE / "csgo_benchmark_v2_eval_general/run_eval.py",
        DATA_ROOT / "benchmark_manifest.json",
        DATA_ROOT / "calibration/z_calibration.json",
        root / ".venv/bin/python",
        root / "checkpoints/Puffin-Base.pth",
        root / "checkpoints/vae.pth",
    ]
    return [str(p) for p in paths if not p.is_file() or p.stat().st_size == 0]


def prep_command():
    return ("set -euo pipefail\n"
            "export CUDA_VISIBLE_DEVICES=\n"
            "bash scripts/setup_csgo_seen10.sh --env-only --check\n"
            "source scripts/activate_csgo_seen10.sh\n"
            "bash scripts/run_csgo_seen10.sh check --experiment csgo_seen10_exp32gen_aligned --seed 42\n")


def run_prerequisites(log=CONTROL / "prerequisites.log"):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(runtime_environment())
    env["CUDA_VISIBLE_DEVICES"] = ""
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{now()}] CPU-only setup, activation, asset and data checks\n")
        stream.flush()
        result = subprocess.run(["bash", "-c", prep_command()], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, timeout=3600, check=False)
    return result.returncode


def training_command():
    return ["bash", "scripts/run_csgo_seen10.sh", "train", "--experiment",
            "csgo_seen10_exp32gen_aligned", "--seed", "42", "--micro-batch-size", "64",
            "--gradient-accumulation-steps", "1", "--output-root", str(FORMAL_ROOT)]


def runtime_environment():
    return {"CUDA_VISIBLE_DEVICES": "0,1", "NPROC_PER_NODE": "2", "CSGO_DATA_ROOT": str(DATA_ROOT),
            "NCCL_SHM_DISABLE": "1"}


def launch_shell_command():
    return ("set -euo pipefail\n"
            "source scripts/activate_csgo_seen10.sh\n"
            "exec " + shlex.join(training_command()) + "\n")


def write_state(value, path=None):
    path = path or CONTROL / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_state(path=None):
    path = path or CONTROL / "state.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def record(status, detail, state):
    state.update({"status": status, "detail": detail, "updated_at": now()})
    if status.startswith("waiting_"):
        state["gates"] = quick_snapshot()
    write_state(state)
    with (CONTROL / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": state["updated_at"], "status": status, "detail": detail}) + "\n")


def launch(state):
    # The lock remains held through this transaction; claim before Popen so a
    # controller crash cannot silently duplicate an already started run.
    status, detail = inspect(control_state=state)
    if status != "ready_for_prerequisites" or simple_prerequisites():
        raise RuntimeError(f"launch gates changed: {status}: {detail}; missing={simple_prerequisites()}")
    state["launch_claimed"] = True
    state["command"] = training_command()
    state["environment"] = runtime_environment()
    state["log"] = str(TRAIN_LOG)
    record("launch_claimed", "all gates passed; launch transaction claimed", state)
    if TRAIN_LOG.exists():
        archive = TRAIN_LOG.with_name(TRAIN_LOG.name + "." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".archive")
        if archive.exists():
            raise FileExistsError(archive)
        TRAIN_LOG.rename(archive)
        state["previous_log_archive"] = str(archive)
    env = dict(os.environ)
    env.update(runtime_environment())
    with TRAIN_LOG.open("x") as stream:
        child = subprocess.Popen(["bash", "-c", launch_shell_command()], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    state["pid"] = child.pid
    info = proc_info(child.pid)
    state["pid_start_ticks"] = info["start_ticks"] if info else None
    state["started_at"] = now()
    record("launched", f"PID {child.pid}; {shlex.join(training_command())}", state)
    return child


def healthy_updates(path=FORMAL_ROOT / "seed_42/train_loss.jsonl"):
    if not path.is_file():
        return 0
    seen = set()
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
                step, loss = row["step"], float(row["loss"])
                if isinstance(step, int) and step > 0 and math.isfinite(loss):
                    seen.add(step)
            except (ValueError, KeyError, TypeError):
                continue
    return len(seen)


def error_tail(path=TRAIN_LOG):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 100_000))
            lines = stream.read().decode(errors="replace").splitlines()
        traceback = [line for line in lines if "Traceback" in line or "Error" in line or "Exception" in line]
        return "\n".join(traceback[-12:])[-5000:]
    except OSError:
        return "training log unavailable"


def quick_snapshot():
    """Independent read-only gates, including blockers hidden by a running job."""
    state = read_state()
    status, detail = inspect(control_state=state)
    missing = simple_prerequisites()
    if status == "ready_for_prerequisites" and missing:
        status, detail = "waiting_prerequisites", f"missing: {missing}"
    latest_step = latest_unilip_step()
    downloader = proc_info(2958876)
    return {"status": status, "detail": detail,
            "unilip": {"latest_logged_step": latest_step, "target_steps": 19550,
                       "active_pids": [p["pid"] for p in relevant_processes()
                                       if "exp32_1" in p["cmdline"]]},
            "prerequisites": {"missing_paths": missing,
                              "downloader_pid": 2958876 if downloader and "download_csgo_seen10_assets.py" in downloader["cmdline"] else None,
                              "cpu_checks": "pending until initial files are present" if missing else "required before launch"},
            "formal_output": "nonempty" if FORMAL_ROOT.is_symlink() or (
                FORMAL_ROOT.exists() and (not FORMAL_ROOT.is_dir() or any(FORMAL_ROOT.iterdir()))) else "empty_or_absent",
            "command": training_command(), "environment": runtime_environment(),
            "log": str(TRAIN_LOG)}


def latest_unilip_step():
    try:
        with UNILIP_STEP_LOG.open("rb") as stream:
            stream.seek(max(0, UNILIP_STEP_LOG.stat().st_size - 200_000))
            steps = re.findall(rb"\[ Step (\d+) \]", stream.read())
        return int(steps[-1]) if steps else None
    except OSError:
        return None


def wait_until_next_progress():
    """Estimate next 20 percentage points (or the terminal step) at 83 s/update."""
    step = latest_unilip_step()
    if step is None:
        return INTERVAL
    next_boundary = min(19550, (step // 3910 + 1) * 3910)
    return max(INTERVAL, (next_boundary - step) * 83)


def wait_interval(pids=(), seconds=INTERVAL):
    """Sleep until a tracked PID exits or the 30-minute status interval ends."""
    with selectors.DefaultSelector() as selector:
        for pid in pids:
            try:
                fd = os.pidfd_open(pid)
                selector.register(fd, selectors.EVENT_READ)
            except (AttributeError, OSError):
                continue
        selector.select(seconds)
        for key in selector.get_map().values():
            os.close(key.fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-once", action="store_true", help="Read-only gate inspection; never launch")
    args = parser.parse_args(argv)
    if args.check_once:
        print(json.dumps(quick_snapshot(), indent=2))
        return 0

    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("aligned launch controller already running", file=sys.stderr)
            return 2
        state = read_state()
        child = None
        while True:
            if child is not None:
                count = healthy_updates()
                code = child.poll()
                if code is not None:
                    state["returncode"] = code
                    if code != 0 or count < 10:
                        record("failed_needs_repair", f"exit={code}; finite_updates={count}; {error_tail()}", state)
                    else:
                        record("training_exited", f"exit=0; finite_updates={count}", state)
                    return 0 if code == 0 and count >= 10 else 1
                if count >= 10:
                    record("started_healthy", f"PID {child.pid} running with {count} finite optimizer updates", state)
                    return 0
                wait_interval((child.pid,))
                continue
            status, detail = inspect(control_state=state)
            if status == "already_launched":
                record(status, detail, state)
                return 0
            if status in ("unilip_incomplete_needs_review", "existing_formal_run", "needs_review"):
                record(status, detail, state)
                return 1
            if status == "ready_for_prerequisites":
                missing = simple_prerequisites()
                if missing:
                    status, detail = "waiting_prerequisites", f"missing: {missing}"
                else:
                    try:
                        code = run_prerequisites()
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        code, detail = 1, repr(exc)
                    if code:
                        status, detail = "waiting_prerequisites", f"CPU-only checks failed (exit={code}): {detail}; see prerequisites.log"
                    else:
                        # Recheck all gates after possibly lengthy setup/audit.
                        child = launch(state)
                        continue
            gates = quick_snapshot()
            if state.get("status") != status or state.get("detail") != detail or state.get("gates", {}).get("prerequisites") != gates["prerequisites"]:
                record(status, detail, state)
            downloader = gates["prerequisites"]["downloader_pid"]
            watched = list(EXPECTED) if status == "waiting_unilip" else []
            if downloader:
                watched.append(downloader)
            wait_interval(watched, wait_until_next_progress() if status == "waiting_unilip" else INTERVAL)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if "--check-once" not in sys.argv:
            state = read_state()
            record("failed_needs_repair", f"controller exception: {type(exc).__name__}: {exc}", state)
        raise
