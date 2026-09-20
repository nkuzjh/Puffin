#!/usr/bin/env python3
"""Generate Seen-10 discrete/continuous outputs with one Puffin checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import torch
from mmengine.config import Config
from torch.utils.data import DataLoader
from xtuner.registry import BUILDER

from csgo_seen10.checkpoint_utils import load_trusted_state_dict
from csgo_seen10.dataset import CsgoSeen10Dataset, collate_seen10
from csgo_seen10.inference_utils import (
    _IMAGE_SUFFIXES,
    atomic_write_json,
    is_valid_rgb_jpeg,
    load_matching_manifest,
    run_signature,
    sample_seed,
    sha256_file,
    write_rgb_jpeg,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = PROJECT_ROOT / "configs/pipelines/csgo_seen10.py"
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/csgo_benchmark_v2_seen10/Puffin")
)


def _load_model(checkpoint: str, device: torch.device):
    config = Config.fromfile(str(CONFIG))
    model_config = config.model
    model_config.pretrained_pth = None
    model_config.pretrained_vae_pth = None
    if device.type == "cuda":
        model_config.device_map = {"": 0}
    else:
        model_config.device_map = None
    model = BUILDER.build(model_config).to(device=device, dtype=torch.bfloat16).eval()
    state_dict = load_trusted_state_dict(checkpoint)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def _generate_split(
    model,
    *,
    data_root: str,
    shared_eval_dir: str,
    split: str,
    task_name: str,
    output_root: Path,
    seed: int,
    steps: int,
    cfg_scale: float,
    maps: list[str] | None,
    max_samples: int | None,
    checkpoint_sha256: str,
) -> dict[str, int]:
    dataset = CsgoSeen10Dataset(
        data_root=data_root,
        shared_eval_dir=shared_eval_dir,
        split=split,
        include_target=False,
        image_size=448,
        maps=maps,
        max_samples=max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_seen10)
    gen_root = output_root / task_name / "gen_imgs"
    task_root = output_root / task_name
    manifest_path = task_root / "inference_manifest.json"

    expected_paths: dict[str, Path] = {}
    sample_id_digest = hashlib.sha256()
    for row in dataset.rows:
        relative = f"{row['map_name']}/{row['file_frame']}.jpg"
        if relative in expected_paths:
            raise ValueError(f"Duplicate Seen-10 output identity: {relative}")
        expected_paths[relative] = gen_root / relative
        sample_id_digest.update(row["sample_id"].encode("utf-8"))
        sample_id_digest.update(b"\n")

    signature_fields = {
        "task": task_name,
        "split": split,
        "seed": int(seed),
        "checkpoint_sha256": checkpoint_sha256,
        "steps": int(steps),
        "cfg_scale": float(cfg_scale),
        "maps": [row for row in dataset.benchmark.maps if maps is None or row in maps],
        "max_samples": max_samples,
        "samples_selected": len(dataset),
        "sample_ids_sha256": sample_id_digest.hexdigest(),
        "output_size": [448, 448],
        "output_format": "RGB JPEG",
    }
    signature = run_signature(signature_fields)
    existing_manifest = load_matching_manifest(manifest_path, signature)
    if existing_manifest is None and gen_root.exists():
        existing_images = [
            path for path in gen_root.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ]
        if existing_images:
            raise ValueError(
                f"Refusing to mix predictions without a matching inference manifest: {existing_images[0]}"
            )

    expected_relatives = set(expected_paths)
    if gen_root.exists():
        extra_images = [
            path for path in gen_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _IMAGE_SUFFIXES
            and path.relative_to(gen_root).as_posix() not in expected_relatives
        ]
        if extra_images:
            raise ValueError(f"Unexpected prediction image outside selected split: {extra_images[0]}")

    task_root.mkdir(parents=True, exist_ok=True)
    initial_manifest = {
        **signature_fields,
        "run_signature": signature,
        "checkpoint": str(model._csgo_checkpoint),
        "target_read": False,
        "deterministic_seed": "sha256(seed:sample_id) first 8 bytes little-endian",
        "complete": False,
    }
    atomic_write_json(initial_manifest, manifest_path)

    counts = {"generated": 0, "skipped_existing": 0}
    device = model.device
    for batch in dataloader:
        sample = batch["data"]["cam2image"]
        metadata = sample["metadata"][0]
        sample_id = metadata["sample_id"]
        map_name = metadata["map_name"]
        file_frame = metadata["file_frame"]
        output_path = gen_root / map_name / f"{file_frame}.jpg"
        if output_path.exists():
            if not is_valid_rgb_jpeg(output_path):
                raise ValueError(f"Refusing to overwrite an invalid existing prediction: {output_path}")
            counts["skipped_existing"] += 1
            continue

        radar = sample["cam_values"][0][0].to(device=device, dtype=model.dtype)
        pose = sample["pose_values"].to(device=device, dtype=torch.float32)
        prompt = sample["texts"]
        item_seed = sample_seed(seed, sample_id)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(item_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(item_seed)
            generator = torch.Generator(device=device).manual_seed(item_seed)
            with torch.inference_mode():
                generated, _ = model.generate(
                    prompt=prompt,
                    cfg_prompt=[""],
                    cam_values=[[radar]],
                    pose_values=pose,
                    cfg_scale=cfg_scale,
                    num_steps=steps,
                    generator=generator,
                    height=448,
                    width=448,
                    progress_bar=False,
                )
        if write_rgb_jpeg(generated[0], output_path):
            counts["generated"] += 1
            if counts["generated"] > 0 and counts["generated"] % 100 == 0:
                print(f"{task_name}: generated={counts['generated']} skipped={counts['skipped_existing']}", flush=True)
        else:
            counts["skipped_existing"] += 1

    manifest = {
        **signature_fields,
        "run_signature": signature,
        "checkpoint": str(model._csgo_checkpoint),
        "counts": counts,
        "target_read": False,
    }
    manifest["present_samples"] = sum(
        (gen_root / row["map_name"] / f"{row['file_frame']}.jpg").is_file()
        for row in dataset.rows
    )
    manifest["complete"] = manifest["present_samples"] == len(dataset)
    atomic_write_json(manifest, manifest_path)
    print(f"{task_name} complete: {counts}; output={gen_root}", flush=True)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("discrete", "continuous", "both"), default="both")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2"))
    parser.add_argument("--shared-eval-dir", default=os.environ.get("SHARED_EVAL_DIR", "/home/jiahao/task/csgo_benchmark_v2_eval_general"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--maps", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Diagnostic prefix only; omit for full evaluation coverage")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Puffin checkpoint not found: {checkpoint}")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not math.isfinite(args.cfg_scale):
        parser.error("--cfg-scale must be finite")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1 or (torch.distributed.is_available() and torch.distributed.is_initialized()):
        parser.error("Seen-10 inference is a single-process entry point; do not launch it with torchrun/accelerate")
    os.environ["DATA_ROOT"] = str(Path(args.data_root).expanduser())
    os.environ["SHARED_EVAL_DIR"] = str(Path(args.shared_eval_dir).expanduser())
    if not torch.cuda.is_available():
        parser.error("Puffin Seen-10 inference requires one CUDA GPU")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank != 0:
        parser.error("Single-GPU inference uses cuda:0; unset LOCAL_RANK or run with LOCAL_RANK=0")
    device = torch.device("cuda:0")
    model = _load_model(str(checkpoint), device)
    model._csgo_checkpoint = str(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)

    tasks = []
    if args.mode in ("discrete", "both"):
        tasks.append(("seen_discrete_test", "discrete"))
    if args.mode in ("continuous", "both"):
        tasks.append(("seen_continuous", "continuous"))
    for split, task_name in tasks:
        _generate_split(
            model,
            data_root=args.data_root,
            shared_eval_dir=args.shared_eval_dir,
            split=split,
            task_name=task_name,
            output_root=args.output_root / f"seed_{args.seed}",
            seed=args.seed,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            maps=args.maps,
            max_samples=args.max_samples,
            checkpoint_sha256=checkpoint_sha256,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
