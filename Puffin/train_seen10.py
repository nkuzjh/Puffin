#!/usr/bin/env python3
"""Small wrapper around Puffin's native scripts/train.py entry point."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from torch.utils.data import DataLoader

from csgo_seen10.checkpoint_utils import configure_native_resume_load
from csgo_seen10.dataset import DEFAULT_DATA_ROOT, CsgoSeen10Dataset, collate_seen10


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = PROJECT_ROOT / "configs/pipelines/csgo_seen10.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/csgo_benchmark_v2_seen10/Puffin"


def _data_check(data_root: str, shared_eval_dir: str, limit: int) -> int:
    dataset = CsgoSeen10Dataset(
        data_root=data_root,
        shared_eval_dir=shared_eval_dir,
        split="seen_train",
        include_target=True,
        max_samples=max(limit, 1),
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=min(max(limit, 1), 2),
                num_workers=0,
                collate_fn=collate_seen10,
            )
        )
    )
    samples = batch["data"]["cam2image"]
    target = samples["pixel_values"]
    radar = samples["cam_values"]
    pose = samples["pose_values"]
    if pose.shape[-1] != 5 or len(target) != len(radar):
        raise RuntimeError("Seen-10 dataset batch violated Puffin's cam2image contract")
    print(
        "Seen-10 train batch OK:",
        f"samples={len(target)}",
        f"target={tuple(target[0].shape)}",
        f"radar={tuple(radar[0][0].shape)}",
        f"pose={tuple(pose.shape)}",
        f"first_id={samples['metadata'][0]['sample_id']}",
    )

    infer_dataset = CsgoSeen10Dataset(
        data_root=data_root,
        shared_eval_dir=shared_eval_dir,
        split="seen_discrete_test",
        include_target=False,
        max_samples=1,
    )
    infer_sample = infer_dataset[0]
    if "pixel_values" in infer_sample:
        raise RuntimeError("Target-free Seen-10 inference unexpectedly opened a target image")
    print("Seen-10 target-free inference sample OK:", infer_sample["metadata"]["sample_id"])
    return 0


def main() -> int:
    # Explicit aligned selection routes to the standalone optimizer-update
    # runner. The native MMEngine path below remains the default.
    experiment_parser = argparse.ArgumentParser(add_help=False)
    experiment_parser.add_argument("--experiment")
    selected, _ = experiment_parser.parse_known_args()
    if selected.experiment == "csgo_seen10_exp32gen_aligned":
        from csgo_seen10.aligned_train import main as aligned_main

        return aligned_main(sys.argv[1:])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument(
        "--init-checkpoint",
        default=os.environ.get("PUFFIN_INIT_CHECKPOINT", str(PROJECT_ROOT / "checkpoints/Puffin-Base.pth")),
        help="local Puffin checkpoint used to initialize a fresh run",
    )
    parser.add_argument(
        "--vae-checkpoint",
        default=os.environ.get("PUFFIN_VAE_CHECKPOINT", str(PROJECT_ROOT / "checkpoints/vae.pth")),
        help="local official Puffin VAE checkpoint used for a fresh run",
    )
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--shared-eval-dir",
        default=os.environ.get("SHARED_EVAL_DIR", "/home/jiahao/task/csgo_benchmark_v2_eval_general"),
    )
    parser.add_argument("--dataset-only", action="store_true", help="Read a real train batch without loading models")
    parser.add_argument("--smoke", action="store_true", help="Run one native train/validation/checkpoint iteration")
    parser.add_argument("--limit", type=int, default=2, help="Batch size for --dataset-only")
    args, passthrough = parser.parse_known_args()

    os.environ["DATA_ROOT"] = str(Path(args.data_root).expanduser())
    os.environ["SHARED_EVAL_DIR"] = str(Path(args.shared_eval_dir).expanduser())
    if args.dataset_only:
        return _data_check(args.data_root, args.shared_eval_dir, args.limit)

    output_root = Path(os.environ.get("OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))).expanduser()
    output_seed_dir = output_root / f"seed_{args.seed}"
    if args.smoke and args.resume:
        parser.error("--smoke cannot be combined with --resume")
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            parser.error(f"Resume checkpoint does not exist: {resume_path}")
        os.environ.pop("PUFFIN_INIT_CHECKPOINT", None)
        os.environ.pop("PUFFIN_VAE_CHECKPOINT", None)
    else:
        init_checkpoint = Path(args.init_checkpoint).expanduser().resolve()
        if not init_checkpoint.is_file():
            parser.error(
                f"Fresh Seen-10 training requires a local Puffin initialization checkpoint: {init_checkpoint}"
            )
        os.environ["PUFFIN_INIT_CHECKPOINT"] = str(init_checkpoint)
        vae_checkpoint = Path(args.vae_checkpoint).expanduser().resolve()
        if not vae_checkpoint.is_file():
            parser.error(
                f"Fresh Seen-10 training requires the separate official Puffin VAE checkpoint: {vae_checkpoint}"
            )
        os.environ["PUFFIN_VAE_CHECKPOINT"] = str(vae_checkpoint)
    if args.smoke:
        os.environ["PUFFIN_SEEN10_SMOKE"] = "1"
    else:
        os.environ.pop("PUFFIN_SEEN10_SMOKE", None)

    default_work_dir = output_seed_dir / ("smoke/checkpoints" if args.smoke else "checkpoints")
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else default_work_dir.resolve()
    if args.smoke and args.work_dir is None and work_dir.exists():
        parser.error(
            f"Refusing to reuse the fixed smoke directory: {work_dir}; "
            "choose a new --work-dir"
        )
    if not args.resume and work_dir.exists():
        if not work_dir.is_dir():
            parser.error(
                f"Training work directory is an existing file: {work_dir}"
            )
        try:
            next(work_dir.iterdir())
        except StopIteration:
            pass
        else:
            parser.error(
                f"Refusing to overwrite a non-empty training directory: {work_dir}; "
                "choose a new --work-dir or resume from a checkpoint"
            )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        str(CONFIG),
        "--work-dir",
        str(work_dir),
        "--seed",
        str(args.seed),
    ]
    if args.resume:
        command.extend(["--resume", args.resume])
    command.extend(passthrough)
    child_env = os.environ.copy()
    if args.resume:
        try:
            configure_native_resume_load(child_env)
        except RuntimeError as exc:
            parser.error(str(exc))
    python_paths = child_env.get("PYTHONPATH", "").split(os.pathsep)
    if str(PROJECT_ROOT) not in python_paths:
        child_env["PYTHONPATH"] = os.pathsep.join(
            [str(PROJECT_ROOT), *(path for path in python_paths if path)]
        )
    print("Running native Puffin training:", " ".join(command), flush=True)
    return subprocess.call(command, cwd=PROJECT_ROOT, env=child_env)


if __name__ == "__main__":
    raise SystemExit(main())
