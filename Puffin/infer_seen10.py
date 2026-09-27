#!/usr/bin/env python3
"""Generate Seen-10 discrete/continuous outputs with one Puffin checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
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
    inspect_prediction_root,
    load_matching_manifest,
    run_signature,
    sample_diagonal_gaussian,
    sample_seed,
    sha256_file,
    pad_inference_batch,
    write_rgb_jpeg,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = PROJECT_ROOT / "configs/pipelines/csgo_seen10.py"
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/csgo_benchmark_v2_seen10/Puffin")
)
_COMPILE_SETTINGS = {
    "target": "model.transformer fixed dense 448px single-radar path",
    "mode": "reduce-overhead",
    "dynamic": False,
    "fullgraph": True,
}
_ALIGNED_COMPILE_SETTINGS = {
    "target": "model.transformer dense 448px target / 224px single radar",
    "mode": "reduce-overhead",
    "dynamic": False,
    "fullgraph": True,
}
_ALIGNED_CONFIG = PROJECT_ROOT / "configs/pipelines/csgo_seen10_exp32gen_aligned.py"


def _verify_aligned_checkpoint_data(payload: dict, benchmark_identity: dict, scheduler_file: Path) -> None:
    """Confirm a full checkpoint belongs to the audited data and local model metadata."""
    from csgo_seen10.prompts import PROMPT_SHA256
    from csgo_seen10.aligned import OFFICIAL_BASE_SHA256, OFFICIAL_VAE_SHA256

    identities = payload.get("identities")
    if not isinstance(identities, dict) or identities.get("benchmark") != benchmark_identity:
        raise ValueError("Aligned checkpoint has no training data/scheduler identities")
    if identities.get("prompt_sha256") != PROMPT_SHA256:
        raise ValueError("Aligned checkpoint prompt identity differs")
    if identities.get("assets") != {
        "puffin_base_sha256": OFFICIAL_BASE_SHA256,
        "puffin_vae_sha256": OFFICIAL_VAE_SHA256,
    }:
        raise ValueError("Aligned checkpoint official Base/VAE provenance differs")
    qwen_root = Path(os.environ.get("PUFFIN_QWEN_PATH", ""))
    radio_root = Path(os.environ.get("PUFFIN_RADIO_PATH", ""))
    paths = {
        "sd3/scheduler/scheduler_config.json": scheduler_file,
        "sd3/vae/config.json": scheduler_file.parents[1] / "vae" / "config.json",
        "qwen/config.json": qwen_root / "config.json",
        "radio/config.json": radio_root / "config.json",
    }
    for path in sorted(qwen_root.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".txt", ".model") and path.stat().st_size < 50_000_000:
            paths[f"qwen/{path.relative_to(qwen_root)}"] = path
    expected_metadata = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    if identities.get("metadata") != expected_metadata or set(expected_metadata) != set(paths):
        raise ValueError("Aligned checkpoint model metadata differs from selected snapshot")
    if payload.get("resolved_configs") != {
        "scheduler": json.loads(scheduler_file.read_text(encoding="utf-8")),
        "vae": json.loads(paths["sd3/vae/config.json"].read_text(encoding="utf-8")),
    }:
        raise ValueError("Aligned checkpoint resolved scheduler/VAE config differs from selected snapshot")

    # A resume after training code changes could silently assign a familiar
    # filename to a different recipe. The producer commits these exact source
    # bytes into each checkpoint; inference enforces the same source inventory.
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
    expected_sources = {name: sha256_file(PROJECT_ROOT / name) for name in source_names}
    if identities.get("source") != expected_sources:
        raise ValueError("Aligned checkpoint training source fingerprint differs from current recipe")


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


def _configure_transformer_inference(model, inference_engine: str) -> None:
    if inference_engine == "compiled":
        model.transformer = torch.compile(
            model.transformer,
            mode=_COMPILE_SETTINGS["mode"],
            dynamic=_COMPILE_SETTINGS["dynamic"],
            fullgraph=_COMPILE_SETTINGS["fullgraph"],
        )
    elif inference_engine != "eager":
        raise ValueError(f"Unsupported inference engine: {inference_engine!r}")


def _configure_aligned_transformer(model, inference_engine: str) -> None:
    if inference_engine == "compiled":
        model.transformer = torch.compile(
            model.transformer,
            mode=_ALIGNED_COMPILE_SETTINGS["mode"],
            dynamic=_ALIGNED_COMPILE_SETTINGS["dynamic"],
            fullgraph=_ALIGNED_COMPILE_SETTINGS["fullgraph"],
        )
    elif inference_engine not in ("eager", "native-eager"):
        raise ValueError(f"Unsupported aligned inference engine: {inference_engine!r}")


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
    inference_engine: str = "eager",
    batch_size: int = 1,
    decoder_chunk_size: int | None = None,
    radar_posterior_cache: dict | None = None,
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
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_seen10
    )
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

    legacy_signature_fields = {
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
    accelerated = inference_engine == "compiled" or batch_size > 1
    seed_strategy = "per-sample separate posterior/diffusion generators, both seeded by sha256(seed:sample_id)"
    signature_fields = dict(legacy_signature_fields)
    if accelerated:
        signature_fields.update(
            inference_engine=inference_engine,
            batch_size=int(batch_size),
            inference_optimization_version=1,
            seed_strategy=seed_strategy,
            decoder_chunk_size=decoder_chunk_size,
        )
        if inference_engine == "compiled":
            signature_fields["compile_settings"] = dict(_COMPILE_SETTINGS)
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
    initial_compile_status = "pending_first_inference"
    if inference_engine == "compiled" and existing_manifest is not None:
        initial_compile_status = existing_manifest.get(
            "compile_status", initial_compile_status
        )
    initial_manifest = {
        **signature_fields,
        "run_signature": signature,
        "checkpoint": str(model._csgo_checkpoint),
        "target_read": False,
        "deterministic_seed": "sha256(seed:sample_id) first 8 bytes little-endian",
        "complete": False,
    }
    if accelerated and inference_engine == "compiled":
        initial_manifest["compile_status"] = initial_compile_status
    atomic_write_json(initial_manifest, manifest_path)

    counts = {"generated": 0, "skipped_existing": 0}
    device = model.device
    if radar_posterior_cache is None:
        radar_posterior_cache = {}
    compile_status = "not_requested"
    if inference_engine == "compiled":
        compile_status = initial_compile_status
    for batch in dataloader:
        sample = batch["data"]["cam2image"]
        real_count = len(sample["metadata"])
        if inference_engine == "compiled" and real_count < batch_size:
            pad_inference_batch(batch, batch_size)
            sample = batch["data"]["cam2image"]

        metadata = sample["metadata"]
        real_metadata = metadata[:real_count]
        output_paths = [
            gen_root / row["map_name"] / f"{row['file_frame']}.jpg" for row in real_metadata
        ]
        missing_indices = []
        for index, output_path in enumerate(output_paths):
            if output_path.exists():
                if not is_valid_rgb_jpeg(output_path):
                    raise ValueError(f"Refusing to overwrite an invalid existing prediction: {output_path}")
                counts["skipped_existing"] += 1
            else:
                missing_indices.append(index)

        # Manifest rows are stable contiguous blocks. If any output in one is
        # absent, rerun the whole block so all batch inputs and RNG streams
        # retain their original positions, then write only the absent images.
        if not missing_indices:
            continue

        prompts = sample["texts"]
        poses = sample["pose_values"].to(device=device, dtype=torch.float32)
        if not accelerated:
            metadata_row = metadata[0]
            radar = sample["cam_values"][0][0].to(device=device, dtype=model.dtype)
            item_seed = sample_seed(seed, metadata_row["sample_id"])
            cuda_devices = (
                [device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(item_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(item_seed)
                generator = torch.Generator(device=device).manual_seed(item_seed)
                with torch.inference_mode():
                    generated, _ = model.generate(
                        prompt=prompts,
                        cfg_prompt=[""],
                        cam_values=[[radar]],
                        pose_values=poses,
                        cfg_scale=cfg_scale,
                        num_steps=steps,
                        generator=generator,
                        height=448,
                        width=448,
                        progress_bar=False,
                        decoder_chunk_size=decoder_chunk_size,
                    )
        else:
            radar_latents = []
            diffusion_generators = []
            for index, metadata_row in enumerate(metadata):
                map_name = metadata_row["map_name"]
                if map_name not in radar_posterior_cache:
                    radar = sample["cam_values"][index][0]
                    radar_posterior_cache[map_name] = model.encode_radar_posterior(radar)
                posterior = radar_posterior_cache[map_name]
                item_seed = sample_seed(seed, metadata_row["sample_id"])
                posterior_generator = torch.Generator(device=device).manual_seed(item_seed)
                posterior_sample = sample_diagonal_gaussian(*posterior, posterior_generator)
                radar_latent = (
                    (posterior_sample - model.vae.config.shift_factor)
                    * model.vae.config.scaling_factor
                )
                radar_latents.append(radar_latent[0])
                # A fresh generator with the same per-sample seed preserves the
                # legacy diffusion stream while separating it from VAE sampling.
                diffusion_generators.append(torch.Generator(device=device).manual_seed(item_seed))

            with torch.inference_mode():
                generated, _ = model.generate(
                    prompt=prompts,
                    cfg_prompt=[""] * len(prompts),
                    cam_values=sample["cam_values"],
                    radar_latents=radar_latents,
                    pose_values=poses,
                    cfg_scale=cfg_scale,
                    num_steps=steps,
                    generator=diffusion_generators,
                    height=448,
                    width=448,
                    progress_bar=False,
                    decoder_chunk_size=decoder_chunk_size,
                )

        if inference_engine == "compiled" and compile_status == "pending_first_inference":
            compile_status = "first_inference_succeeded"
            initial_manifest["compile_status"] = compile_status
            atomic_write_json(initial_manifest, manifest_path)

        for index in missing_indices:
            if write_rgb_jpeg(generated[index], output_paths[index]):
                counts["generated"] += 1
                if counts["generated"] % 100 == 0:
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
    if accelerated and inference_engine == "compiled":
        manifest["compile_status"] = compile_status
    manifest["present_samples"] = sum(
        is_valid_rgb_jpeg(gen_root / row["map_name"] / f"{row['file_frame']}.jpg")
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
    parser.add_argument(
        "--inference-engine", choices=("eager", "compiled"), default="eager",
        help="Compile only the fixed-size SD3 transformer path when set to compiled",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Stable contiguous manifest block size")
    parser.add_argument(
        "--decoder-chunk-size", type=int, default=None,
        help="VAE decode chunk size; batched inference defaults to one image per decode chunk",
    )
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
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.decoder_chunk_size is not None and args.decoder_chunk_size <= 0:
        parser.error("--decoder-chunk-size must be positive")
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
    _configure_transformer_inference(model, args.inference_engine)
    decoder_chunk_size = args.decoder_chunk_size
    if decoder_chunk_size is None and args.batch_size > 1:
        decoder_chunk_size = 1
    radar_posterior_cache = {}

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
            inference_engine=args.inference_engine,
            batch_size=args.batch_size,
            decoder_chunk_size=decoder_chunk_size,
            radar_posterior_cache=radar_posterior_cache,
        )
    return 0


def _aligned_signature_fields(
    dataset, *, split, task_name, args, checkpoint_sha256, scheduler_file
) -> dict:
    from csgo_seen10 import dataset as dataset_module, prompts as prompts_module
    from csgo_seen10.prompts import PROMPT_SHA256, PROMPT_VERSION

    digest = hashlib.sha256()
    for row in dataset.rows:
        digest.update(row["sample_id"].encode("utf-8"))
        digest.update(b"\n")
    fields = {
        "experiment": "csgo_seen10_exp32gen_aligned",
        "task": task_name,
        "split": split,
        "selection": args.selection,
        "prediction_tag": args.prediction_tag,
        "seed": args.seed,
        "checkpoint_sha256": checkpoint_sha256,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "maps": [name for name in dataset.benchmark.maps if args.maps is None or name in args.maps],
        "max_samples": args.max_samples,
        "samples_selected": len(dataset),
        "sample_ids_sha256": digest.hexdigest(),
        "output_size": [448, 448],
        "radar_size": [224, 224],
        "pose_mode": "text",
        "pose_conditioning": False,
        "negative_prompt": "",
        "reasoning": False,
        "output_format": "RGB JPEG",
        "prompt_template_sha256": PROMPT_SHA256,
        "prompt_version": PROMPT_VERSION,
        "prompt_implementation_sha256": sha256_file(Path(prompts_module.__file__)),
        "dataset_implementation_sha256": sha256_file(Path(dataset_module.__file__)),
        "data_protocol_sha256": sha256_file(Path(args.shared_eval_dir) / "protocol.py"),
        "data_identity_sha256": args.benchmark_identity_sha256,
        "aligned_config_sha256": sha256_file(_ALIGNED_CONFIG),
        "scheduler_config_sha256": sha256_file(scheduler_file),
        "vae_config_sha256": sha256_file(scheduler_file.parents[1] / "vae" / "config.json"),
        "checkpoint_training_source_fingerprint": args.training_source_fingerprint,
        "dtype": "bfloat16",
        "inference_engine": args.inference_engine,
        "batch_size": args.batch_size,
        "decoder_chunk_size": args.decoder_chunk_size,
        "seed_strategy": "per-sample separate posterior/diffusion generators, both seeded by sha256(seed:sample_id)",
        "inference_optimization_version": 2,
        "native_reference": args.inference_engine == "native-eager",
    }
    if args.inference_engine == "compiled":
        fields["compile_settings"] = dict(_ALIGNED_COMPILE_SETTINGS)
    return fields


def _generate_aligned_split(
    model, *, args, split, task_name, checkpoint_sha256, scheduler_file,
    radar_posterior_cache,
) -> dict[str, int]:
    dataset = CsgoSeen10Dataset(
        data_root=args.data_root,
        shared_eval_dir=args.shared_eval_dir,
        split=split,
        include_target=False,
        radar_size=224,
        target_size=448,
        pose_mode="text",
        maps=args.maps,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_seen10,
    )
    task_root = (
        args.output_root / f"seed_{args.seed}" / "predictions" /
        args.selection / args.prediction_tag / task_name
    )
    gen_root = task_root / "gen_imgs"
    manifest_path = task_root / "inference_manifest.json"
    expected = {}
    for row in dataset.rows:
        relative = f"{row['map_name']}/{row['file_frame']}.jpg"
        if relative in expected:
            raise ValueError(f"Duplicate Seen-10 output identity: {relative}")
        expected[relative] = gen_root / relative
    fields = _aligned_signature_fields(
        dataset, split=split, task_name=task_name, args=args,
        checkpoint_sha256=checkpoint_sha256, scheduler_file=scheduler_file,
    )
    signature = run_signature(fields)
    existing = inspect_prediction_root(task_root, signature, set(expected))
    if existing is not None and (
        existing.get("signature_fields") != fields
        or any(existing.get(key) != value for key, value in fields.items())
    ):
        raise ValueError(f"Existing aligned inference manifest has corrupt fields: {manifest_path}")
    manifest = {
        **fields,
        "signature_fields": dict(fields),
        "run_signature": signature,
        "checkpoint": str(model._csgo_checkpoint),
        "scheduler_source": str(scheduler_file.resolve()),
        "smoke_only": args.diagnostic_only,
        "diagnostic_only": args.diagnostic_only,
        "target_read": False,
        "deterministic_seed": "sha256(seed:sample_id) first 8 bytes little-endian",
        "complete": False,
    }
    if args.inference_engine == "compiled":
        manifest["compile_status"] = (
            existing.get("compile_status", "pending_first_inference")
            if existing else "pending_first_inference"
        )
    atomic_write_json(manifest, manifest_path)

    counts = {"generated": 0, "skipped_existing": 0}
    device = model.device
    for batch in dataloader:
        sample = batch["data"]["cam2image"]
        real_count = len(sample["metadata"])
        if args.inference_engine == "compiled" and real_count < args.batch_size:
            pad_inference_batch(batch, args.batch_size)
        metadata = sample["metadata"]
        output_paths = [
            gen_root / row["map_name"] / f"{row['file_frame']}.jpg"
            for row in metadata[:real_count]
        ]
        missing = []
        for index, path in enumerate(output_paths):
            if path.exists():
                if not is_valid_rgb_jpeg(path):
                    raise ValueError(f"Invalid existing prediction: {path}")
                counts["skipped_existing"] += 1
            else:
                missing.append(index)
        if not missing:
            continue

        if "pose_values" in sample:
            raise ValueError("Aligned text-pose inference must not supply pose_values")
        if args.inference_engine == "native-eager":
            # Full original one-sample VAE and transformer list path. This is
            # an independent numerical reference for dense/compiled outputs.
            generated = []
            for index in range(real_count):
                row = metadata[index]
                item_seed = sample_seed(args.seed, row["sample_id"])
                cuda_devices = (
                    [device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else []
                )
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(item_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(item_seed)
                    generator = torch.Generator(device=device).manual_seed(item_seed)
                    with torch.inference_mode():
                        images, _ = model.generate(
                            prompt=[sample["texts"][index]], cfg_prompt=[""],
                            cam_values=[sample["cam_values"][index]],
                            cfg_scale=args.cfg_scale, num_steps=args.steps,
                            generator=generator, height=448, width=448,
                            progress_bar=False, decoder_chunk_size=args.decoder_chunk_size,
                            radar_dense=False, reasoning=False,
                        )
                generated.append(images[0])
            generated = torch.stack(generated)
        else:
            radar_latents = []
            diffusion_generators = []
            for index, row in enumerate(metadata):
                map_name = row["map_name"]
                if map_name not in radar_posterior_cache:
                    radar_posterior_cache[map_name] = model.encode_radar_posterior(
                        sample["cam_values"][index][0]
                    )
                item_seed = sample_seed(args.seed, row["sample_id"])
                posterior_generator = torch.Generator(device=device).manual_seed(item_seed)
                posterior_sample = sample_diagonal_gaussian(
                    *radar_posterior_cache[map_name], posterior_generator
                )
                radar_latents.append(
                    ((posterior_sample - model.vae.config.shift_factor)
                     * model.vae.config.scaling_factor)[0]
                )
                diffusion_generators.append(
                    torch.Generator(device=device).manual_seed(item_seed)
                )
            with torch.inference_mode():
                generated, _ = model.generate(
                    prompt=sample["texts"], cfg_prompt=[""] * len(metadata),
                    cam_values=sample["cam_values"], radar_latents=radar_latents,
                    cfg_scale=args.cfg_scale, num_steps=args.steps,
                    generator=diffusion_generators, height=448, width=448,
                    progress_bar=False, decoder_chunk_size=args.decoder_chunk_size,
                    reasoning=False,
                )

        if args.inference_engine == "compiled" and manifest["compile_status"] == "pending_first_inference":
            manifest["compile_status"] = "first_inference_succeeded"
            atomic_write_json(manifest, manifest_path)
        for index in missing:
            if write_rgb_jpeg(generated[index], output_paths[index]):
                counts["generated"] += 1
            else:
                counts["skipped_existing"] += 1
        if counts["generated"] and counts["generated"] % 100 == 0:
            print(f"{task_name}: generated={counts['generated']} skipped={counts['skipped_existing']}", flush=True)

    manifest["counts"] = counts
    manifest["present_samples"] = sum(is_valid_rgb_jpeg(path) for path in expected.values())
    manifest["complete"] = manifest["present_samples"] == len(dataset)
    atomic_write_json(manifest, manifest_path)
    print(f"{task_name} complete: {counts}; output={gen_root}", flush=True)
    return counts


def aligned_main() -> int:
    from csgo_seen10.paths import aligned_output_root, resolve_data_root, resolve_eval_root
    from csgo_seen10.aligned import build_aligned_model, load_aligned_checkpoint

    parser = argparse.ArgumentParser(description="Aligned exp32 generation inference")
    parser.add_argument("--experiment", choices=("csgo_seen10_exp32gen_aligned",), required=True)
    parser.add_argument("--mode", choices=("discrete", "continuous", "both"), default="both")
    parser.add_argument("--selection", choices=("late", "best"), required=True)
    parser.add_argument("--prediction-tag", default="native50_compiled_b16")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root")
    parser.add_argument("--shared-eval-dir")
    parser.add_argument("--output-root", type=Path, default=aligned_output_root())
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--inference-engine", choices=("native-eager", "eager", "compiled"), default="compiled")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decoder-chunk-size", type=int, default=1)
    parser.add_argument("--maps", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    args = parser.parse_args()
    for name in ("steps", "batch_size", "decoder_chunk_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if not math.isfinite(args.cfg_scale):
        parser.error("--cfg-scale must be finite")
    if not args.prediction_tag or args.prediction_tag in (".", "..") or "/" in args.prediction_tag:
        parser.error("--prediction-tag must be one nonempty path component")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ):
        parser.error("Aligned inference is single process")
    if not torch.cuda.is_available() or int(os.environ.get("LOCAL_RANK", "0")) != 0:
        parser.error("Aligned inference requires cuda:0 and LOCAL_RANK=0")
    args.data_root = str(resolve_data_root(args.data_root))
    args.shared_eval_dir = str(resolve_eval_root(args.shared_eval_dir))
    args.output_root = aligned_output_root(args.output_root)
    checkpoint = (args.checkpoint or args.output_root / f"seed_{args.seed}" /
                  "checkpoints" / f"{args.selection}.pth").expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Aligned checkpoint not found: {checkpoint}")
    sd3_path = os.environ.get("PUFFIN_SD3_PATH")
    if not sd3_path:
        raise ValueError("PUFFIN_SD3_PATH is required for the official SD3 scheduler")
    scheduler_file = Path(sd3_path).expanduser().resolve() / "scheduler" / "scheduler_config.json"
    if not scheduler_file.is_file():
        raise FileNotFoundError(f"Official SD3 scheduler config not found: {scheduler_file}")
    from csgo_seen10.contracts import audit_benchmark
    benchmark_identity = audit_benchmark(args.data_root, args.shared_eval_dir)["identity"]
    args.benchmark_identity_sha256 = benchmark_identity["sha256"]
    model = build_aligned_model(load_base=False, enable_lora=True, device=torch.device("cuda:0"))
    checkpoint_payload = load_aligned_checkpoint(model, checkpoint)
    _verify_aligned_checkpoint_data(checkpoint_payload, benchmark_identity, scheduler_file)
    args.training_source_fingerprint = run_signature(checkpoint_payload["identities"]["source"])
    args.diagnostic_only = bool(checkpoint_payload.get("smoke_only", False))
    del checkpoint_payload
    if args.diagnostic_only and (not args.allow_smoke_checkpoint or args.max_samples is None):
        parser.error("Smoke checkpoint requires --allow-smoke-checkpoint and --max-samples for diagnostic inference")
    if args.allow_smoke_checkpoint and not args.diagnostic_only:
        parser.error("--allow-smoke-checkpoint is only valid for a smoke checkpoint")
    model._csgo_checkpoint = str(checkpoint)
    # The training checkpoint retains FP32 adapter and bridge masters. Native
    # generation passes BF16 language features into those layers without a
    # surrounding autocast, so cast the complete inference graph after the
    # strict FP32 parameter audit in load_aligned_checkpoint.
    model.to(dtype=torch.bfloat16).eval()
    _configure_aligned_transformer(model, args.inference_engine)
    checkpoint_sha256 = sha256_file(checkpoint)
    radar_posterior_cache = {}
    tasks = []
    if args.mode in ("discrete", "both"):
        tasks.append(("seen_discrete_test", "discrete"))
    if args.mode in ("continuous", "both"):
        tasks.append(("seen_continuous", "continuous"))
    for split, task_name in tasks:
        _generate_aligned_split(
            model, args=args, split=split, task_name=task_name,
            checkpoint_sha256=checkpoint_sha256, scheduler_file=scheduler_file,
            radar_posterior_cache=radar_posterior_cache,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        aligned_main() if any(
            arg == "--experiment" or arg.startswith("--experiment=")
            for arg in sys.argv[1:]
        ) else main()
    )
