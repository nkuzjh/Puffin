#!/usr/bin/env python3
"""Explicit aligned orchestration; the legacy shell entry remains unchanged."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from csgo_seen10.paths import (EXPERIMENT, aligned_output_root, project_path,
                               resolve_data_root, resolve_eval_python, resolve_eval_root, resolve_model_python)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("check", "smoke", "train", "infer", "eval", "all"))
    result.add_argument("--experiment", choices=(EXPERIMENT,), required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--data-root")
    result.add_argument("--shared-eval-dir", "--eval-root", dest="shared_eval_dir")
    result.add_argument("--eval-python", "--unilip-python", dest="eval_python")
    result.add_argument("--output-root")
    result.add_argument("--micro-batch-size", type=int, default=4)
    result.add_argument("--gradient-accumulation-steps", type=int)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--resume")
    result.add_argument("--init-checkpoint")
    result.add_argument("--vae-checkpoint")
    result.add_argument("--checkpoint")
    result.add_argument("--selection", choices=("late", "best"), default="late")
    result.add_argument("--prediction-tag")
    result.add_argument("--mode", choices=("discrete", "continuous", "both"), default="both")
    result.add_argument("--task", choices=("discrete", "continuous", "both"), default="both")
    result.add_argument("--inference-engine", choices=("native-eager", "eager", "compiled"), default="compiled")
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--decoder-chunk-size", type=int, default=1)
    result.add_argument("--steps", type=int, default=50)
    result.add_argument("--cfg-scale", type=float, default=4.5)
    result.add_argument("--max-samples", type=int)
    result.add_argument("--maps", nargs="+")
    result.add_argument("--smoke-updates", "--smoke-steps", type=int, default=2)
    result.add_argument("--dry-run", action="store_true", help="Print commands without creating outputs, loading models, or starting jobs")
    return result


def launch(command, dry_run=False):
    import shlex
    print(shlex.join([str(item) for item in command]), flush=True)
    if not dry_run:
        subprocess.run([str(item) for item in command], cwd=PROJECT_ROOT, check=True)


def training_command(args, smoke=False):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_count = len(visible.split(",")) if visible and visible != "-1" else 1
    nproc = int(os.environ.get("NPROC_PER_NODE", str(visible_count)))
    nnodes = int(os.environ.get("NNODES", "1"))
    world = nproc * nnodes
    if min(nproc, nnodes, args.micro_batch_size) <= 0:
        raise ValueError("GPU count, node count and micro batch must be positive")
    divisor = world * args.micro_batch_size
    accumulation = args.gradient_accumulation_steps
    if accumulation is None:
        if 128 % divisor:
            raise ValueError(f"128 is not divisible by world*micro={divisor}")
        accumulation = 128 // divisor
    if accumulation <= 0 or divisor * accumulation != 128:
        raise ValueError(f"Effective batch must be 128, got {world}*{args.micro_batch_size}*{accumulation}")
    command = [str(resolve_model_python())]
    if world > 1:
        command += ["-m", "torch.distributed.run", f"--nproc_per_node={nproc}"]
        if nnodes == 1:
            command += ["--standalone"]
        else:
            if not os.environ.get("MASTER_ADDR") or not os.environ.get("NODE_RANK"):
                raise ValueError("Multi-node training requires MASTER_ADDR and NODE_RANK")
            command += [f"--nnodes={nnodes}", f"--node_rank={os.environ['NODE_RANK']}",
                        f"--master_addr={os.environ['MASTER_ADDR']}", f"--master_port={os.environ.get('MASTER_PORT', '29500')}"]
    command += [str(PROJECT_ROOT / "train_seen10.py"), "--experiment", EXPERIMENT,
                "--seed", str(args.seed), "--output-root", str(args.output_root),
                "--micro-batch-size", str(args.micro_batch_size), "--gradient-accumulation-steps", str(accumulation),
                "--data-root", str(args.data_root), "--shared-eval-dir", str(args.shared_eval_dir),
                "--num-workers", str(args.num_workers)]
    for name in ("resume", "init_checkpoint", "vae_checkpoint"):
        if getattr(args, name):
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if smoke:
        command += ["--smoke", "--smoke-updates", str(args.smoke_updates)]
    return command


def inference_command(args, smoke=False):
    command = [str(resolve_model_python()), str(PROJECT_ROOT / "infer_seen10.py"), "--experiment", EXPERIMENT,
               "--seed", str(args.seed), "--output-root", str(args.output_root), "--selection", args.selection,
               "--prediction-tag", args.prediction_tag, "--mode", args.mode,
               "--data-root", str(args.data_root), "--shared-eval-dir", str(args.shared_eval_dir),
               "--steps", str(args.steps), "--cfg-scale", str(args.cfg_scale),
               "--inference-engine", args.inference_engine, "--batch-size", str(args.batch_size),
               "--decoder-chunk-size", str(args.decoder_chunk_size)]
    checkpoint = args.checkpoint
    if smoke:
        checkpoint = args.output_root / f"seed_{args.seed}" / "checkpoints/latest.pth"
        command += ["--allow-smoke-checkpoint"]
    if checkpoint:
        command += ["--checkpoint", str(checkpoint)]
    limit = args.max_samples if args.max_samples is not None else (1 if smoke else None)
    if limit is not None:
        command += ["--max-samples", str(limit)]
    if args.maps:
        command += ["--maps", *args.maps]
    return command


def require_formal_predictions(args, tasks):
    """Fail closed before calling shared metrics on smoke, mixed or partial data."""
    from csgo_seen10.contracts import audit_benchmark
    from csgo_seen10.inference_utils import run_signature, sha256_file
    audit = audit_benchmark(args.data_root, args.shared_eval_dir)
    common_checkpoint = None
    seed_root = args.output_root / f"seed_{args.seed}"
    selected_checkpoint = seed_root / "checkpoints" / f"{args.selection}.pth"
    selected_hash = sha256_file(selected_checkpoint)
    for task in tasks:
        task_root = seed_root / "predictions" / args.selection / args.prediction_tag / task
        manifest = json.loads((task_root / "inference_manifest.json").read_text())
        expected = 20000 if task == "discrete" else 12800
        fields = manifest.get("signature_fields")
        if not isinstance(fields, dict) or run_signature(fields) != manifest.get("run_signature"):
            raise ValueError(f"Invalid aligned inference signature: {task_root}")
        if any(manifest.get(key) != value for key, value in fields.items()):
            raise ValueError(f"Manifest differs from its signed fields: {task_root}")
        if (manifest.get("experiment") != EXPERIMENT or not manifest.get("complete")
                or manifest.get("smoke_only", False) or manifest.get("max_samples") is not None
                or manifest.get("present_samples") != expected or manifest.get("samples_selected") != expected
                or manifest.get("target_read") is not False or manifest.get("pose_mode") != "text"
                or manifest.get("steps") != 50 or manifest.get("cfg_scale") != 4.5 or manifest.get("seed") != 42
                or manifest.get("selection") != args.selection or manifest.get("maps") != audit["maps"]
                or manifest.get("checkpoint_sha256") != selected_hash):
            raise ValueError(f"Not complete native-protocol aligned predictions: {task_root}")
        data_hash = manifest.get("data_identity_sha256")
        if data_hash != audit["identity"]["sha256"]:
            raise ValueError(f"Prediction data identity does not match selected bundle: {task_root}")
        if common_checkpoint is not None and manifest.get("checkpoint_sha256") != common_checkpoint:
            raise ValueError("Discrete and continuous predictions use different checkpoints")
        common_checkpoint = manifest["checkpoint_sha256"]
        # The shared evaluator remains the authority for exact JPEG identity and coverage.
    return common_checkpoint


def evaluate(args, smoke=False):
    tasks = ("discrete", "continuous") if args.task == "both" else (args.task,)
    interpreter = resolve_eval_python(args.eval_python, args.shared_eval_dir)
    if not smoke and not args.dry_run:
        require_formal_predictions(args, tasks)
    for task in tasks:
        seed_root = args.output_root / f"seed_{args.seed}"
        pred = seed_root / "predictions" / args.selection / args.prediction_tag / task / "gen_imgs"
        command = [str(interpreter), str(args.shared_eval_dir / "run_eval.py")]
        command += ["smoke", task] if smoke else [task]
        command += ["--config", str(args.shared_eval_dir / "benchmark_v2.yaml"), "--pred-root", str(pred),
                    "--data-root", str(args.data_root)]
        if smoke:
            command += ["--limit", "1"] if task == "discrete" else ["--max-clips", "1", "--frame-only"]
        else:
            out = seed_root / "evaluation_shared" / args.selection / args.prediction_tag / task
            command += ["--output", str(out)]
        launch(command, args.dry_run)


def check(args):
    launch([str(resolve_model_python()), str(PROJECT_ROOT / "scripts/download_csgo_seen10_assets.py"),
            "--check", "--profile", "aligned", "--json"], args.dry_run)
    from csgo_seen10.contracts import audit_benchmark, check_target_isolation
    from csgo_seen10.aligned import aligned_semantics
    from mmengine.config import Config
    Config.fromfile(str(PROJECT_ROOT / "configs/pipelines/csgo_seen10_exp32gen_aligned.py"))
    audit = audit_benchmark(args.data_root, args.shared_eval_dir)
    print(json.dumps({"recipe": aligned_semantics(), "data_audit": audit,
                      "target_isolation": check_target_isolation(args.data_root, args.shared_eval_dir)}, indent=2))
    launch([str(resolve_model_python()), "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], args.dry_run)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    interpreter = resolve_model_python()
    if Path(sys.executable).resolve() != interpreter.resolve():
        return subprocess.call([str(interpreter), str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])])
    if args.action not in ("infer", "smoke") and (args.max_samples is not None or args.maps):
        p.error("Partial maps/samples are diagnostic inference only")
    if args.resume and args.action not in ("train", "smoke"):
        p.error("--resume is training-only")
    if args.checkpoint and args.action != "infer":
        p.error("--checkpoint is inference-only")
    if args.action == "smoke" and (args.mode != "both" or args.task != "both"):
        p.error("Smoke covers both test tracks")
    args.data_root = resolve_data_root(args.data_root)
    args.shared_eval_dir = resolve_eval_root(args.shared_eval_dir)
    if args.action == "smoke":
        default = PROJECT_ROOT / "outputs" / f"smoke_{EXPERIMENT}" / "Puffin"
        args.output_root = project_path(args.output_root or default)
        if args.output_root == aligned_output_root():
            p.error("Smoke cannot use the default formal output root")
    else:
        args.output_root = aligned_output_root(args.output_root)
    args.prediction_tag = args.prediction_tag or ("smoke_native50_" if args.action == "smoke" else "native50_") + args.inference_engine + f"_b{args.batch_size}"
    if not args.prediction_tag or Path(args.prediction_tag).name != args.prediction_tag or args.prediction_tag in (".", ".."):
        p.error("--prediction-tag must be a single directory name")
    if args.action == "check":
        check(args)
    elif args.action in ("train", "all", "smoke"):
        launch(training_command(args, smoke=args.action == "smoke"), args.dry_run)
        if args.action in ("all", "smoke"):
            launch(inference_command(args, smoke=args.action == "smoke"), args.dry_run)
            evaluate(args, smoke=args.action == "smoke")
    elif args.action == "infer":
        launch(inference_command(args), args.dry_run)
    elif args.action == "eval":
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
