"""Standalone optimizer-update loop for the aligned Puffin Seen-10 recipe."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from csgo_seen10.aligned import DETERMINISM, FORMAT, OFFICIAL_VAE_SHA256, build_aligned_model, config_fingerprint, load_aligned_checkpoint, parameter_audit
from csgo_seen10.aligned_samplers import AlignedTrainBatchSampler, AlignedValidationSampler
from csgo_seen10.contracts import audit_benchmark, file_sha256
from csgo_seen10.dataset import CsgoSeen10Dataset, collate_seen10
from csgo_seen10.paths import aligned_output_root, resolve_data_root, resolve_eval_root


MILESTONES = (4_000, 8_000, 12_000, 16_000, 19_500)


def _configure_determinism() -> None:
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    expected = DETERMINISM["cublas_workspace_config"]
    if configured is not None and configured != expected:
        raise ValueError(
            f"Aligned training requires CUBLAS_WORKSPACE_CONFIG={expected}; got {configured!r}"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _identity(data_root: Path, shared_eval_dir: Path, metadata_root: Path) -> dict[str, Any]:
    benchmark = audit_benchmark(data_root, shared_eval_dir)
    metadata = {
        "sd3/scheduler/scheduler_config.json": file_sha256(metadata_root / "scheduler/scheduler_config.json"),
        "sd3/vae/config.json": file_sha256(metadata_root / "vae/config.json"),
        "qwen/config.json": file_sha256(Path(os.environ["PUFFIN_QWEN_PATH"]) / "config.json"),
        "radio/config.json": file_sha256(Path(os.environ["PUFFIN_RADIO_PATH"]) / "config.json"),
    }
    qwen_dir = Path(os.environ["PUFFIN_QWEN_PATH"])
    for path in sorted(qwen_dir.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".txt", ".model") and path.stat().st_size < 50_000_000:
            metadata[f"qwen/{path.relative_to(qwen_dir)}"] = file_sha256(path)
    source_root = Path(__file__).resolve().parents[1]
    sources = ["csgo_seen10/aligned.py", "csgo_seen10/aligned_train.py",
               "csgo_seen10/aligned_samplers.py", "csgo_seen10/dataset.py",
               "csgo_seen10/prompts.py", "csgo_seen10/contracts.py",
               "csgo_seen10/paths.py", "csgo_seen10/model_builders.py",
               "src/models/puffin/model.py", "src/models/connector/modeling_connector.py",
               "src/models/connector/configuration_connector.py",
               "src/models/stable_diffusion3/transformer_sd3_dynamic.py",
               "configs/pipelines/csgo_seen10_exp32gen_aligned.py",
               "configs/pipelines/csgo_seen10.py",
               "configs/models/qwen2_5_1_5b_radio_sd3_dynamic_puffin.py",
               "requirements_seen10.txt"]
    source = {name: file_sha256(source_root / name) for name in sources}
    return {"benchmark": benchmark["identity"], "prompt_sha256": benchmark["prompt_sha256"],
            "metadata": metadata, "source": source,
            "assets": {"puffin_base_sha256": "4045661c81b29adc8aa1cc22079ddbbf86d353d2e0f35c0ffec310f193504257",
                       "puffin_vae_sha256": OFFICIAL_VAE_SHA256}}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _link(directory: Path, name: str, target: Path) -> None:
    path = directory / name
    if path.exists() and not path.is_symlink():
        raise FileExistsError(f"Cannot replace non-symlink checkpoint artifact: {path}")
    temporary = directory / f".{name}.{os.getpid()}.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target.name)
    os.replace(temporary, path)


def make_optimizer(model: torch.nn.Module) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    """Four disjoint groups: LoRA/full x decayed/non-decayed."""

    groups: dict[tuple[str, bool], list[torch.nn.Parameter]] = {
        (family, decay): [] for family in ("lora", "full") for decay in (True, False)
    }
    names: dict[tuple[str, bool], list[str]] = {key: [] for key in groups}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError(f"Duplicate trainable optimizer parameter: {name}")
        seen.add(id(parameter))
        family = "lora" if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name else "full"
        key = (family, parameter.ndim >= 2)
        groups[key].append(parameter)
        names[key].append(name)
    audited = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != audited:
        raise RuntimeError("Optimizer groups omit a trainable parameter")
    for parameter in model.parameters():
        if not parameter.requires_grad and id(parameter) in seen:
            raise RuntimeError("Frozen parameter entered optimizer")
    optimizer = torch.optim.AdamW([
        {"params": groups[key], "lr": 1e-4 if key[0] == "lora" else 5e-6,
         "weight_decay": 0.05 if key[1] else 0.0,
         "name": f"{key[0]}_{'decay' if key[1] else 'no_decay'}"}
        for key in groups if groups[key]
    ], betas=(0.9, 0.95), eps=1e-8)
    return optimizer, {f"{family}_{'decay' if decay else 'no_decay'}": names[(family, decay)]
                       for family, decay in groups}


def make_scheduler(optimizer: torch.optim.Optimizer, max_updates: int = 19_500):
    if max_updates == 1:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda update: 1e-5 if update == 0 else 0.0)
    if max_updates < 1:
        raise ValueError("max_updates must be positive")
    warmup = 585 if max_updates == 19_500 else max(1, math.ceil(max_updates * 0.03))
    warmup = min(warmup, max_updates - 1)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_updates - warmup, eta_min=0.0),
        ],
        milestones=[warmup],
    )


def _loader(dataset, *, batch_size: int | None = None, batch_sampler=None, sampler=None, workers: int = 0, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(dataset=dataset, num_workers=workers, pin_memory=True, collate_fn=collate_seen10,
                  generator=generator, persistent_workers=workers > 0)
    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs.update(batch_size=batch_size, sampler=sampler, drop_last=False)
    return DataLoader(**kwargs)


def _loss(model, batch: dict) -> torch.Tensor:
    result = model(batch["data"], mode="loss")
    if set(result) != {"loss_cam2image"}:
        raise RuntimeError(f"Aligned generation batch returned unexpected losses: {list(result)}")
    loss = result["loss_cam2image"]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite native Puffin flow loss: {loss}")
    return loss


def _validate(model, dataset, *, rank: int, world: int, micro_batch: int, workers: int,
              device: torch.device, seed: int) -> float:
    saved = _rng()
    _seed(seed + 20260827)
    was_training = model.training
    unconditional = model.unconditional
    unconditional_cross_view = model.unconditional_cross_view
    model.unconditional = 0.0
    model.unconditional_cross_view = 0.0
    model.eval()
    sampler = AlignedValidationSampler(len(dataset), rank=rank, world_size=world)
    loader = _loader(dataset, batch_size=1, sampler=sampler, workers=workers, seed=seed + rank + 2)
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch in loader:
                sample_id = batch["data"]["cam2image"]["metadata"][0]["sample_id"]
                sample_seed = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big")
                _seed((seed + 20260827 + sample_seed) % (2**63))
                count = len(batch["data"]["cam2image"]["texts"])
                totals[0] += _loss(model, batch).double() * count
                totals[1] += count
        if dist.is_initialized():
            dist.all_reduce(totals)
        if int(totals[1].item()) != len(dataset):
            raise RuntimeError(f"Validation covered {int(totals[1])} rows, expected {len(dataset)}")
        return float((totals[0] / totals[1]).item())
    finally:
        model.unconditional = unconditional
        model.unconditional_cross_view = unconditional_cross_view
        model.train(was_training)
        _restore_rng(saved)


def _save(checkpoint_dir: Path, *, step: int, model, optimizer, scheduler, rank: int, world: int,
          seed: int, topology: dict, identities: dict, best: float, validation_loss: float,
          is_best: bool, smoke_only: bool, max_updates: int, resolved_configs: dict,
          object_group: dist.ProcessGroup | None) -> Path:
    if world > 1 and object_group is None:
        raise ValueError("Distributed checkpoint save requires the Gloo object group")
    states: list[Any] = [None] * world
    local_rng = _rng()
    if world > 1:
        dist.all_gather_object(states, local_rng, group=object_group)
    else:
        states[0] = local_rng
    path = checkpoint_dir / f"step_{step:08d}.pth"
    save_error = None
    if rank == 0:
        try:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite completed milestone: {path}")
            payload = {
                "format": FORMAT,
                "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "config_fingerprint": config_fingerprint(), "identities": identities,
                "resolved_configs": resolved_configs, "scaler": {"enabled": False},
                "determinism": dict(DETERMINISM),
                "step": step, "sampler": {"epoch": step // 390, "logical_update": step % 390},
                "accumulation_boundary": True, "seed": seed, "topology": topology,
                "max_updates": max_updates,
                "rng_by_rank": states, "best_validation_loss": best,
                "validation_loss": validation_loss,
                "smoke_only": bool(smoke_only),
            }
            _atomic_save(payload, path)
            if is_best:
                _link(checkpoint_dir, "best.pth", path)
            if step == 19_500:
                _link(checkpoint_dir, "late.pth", path)
            _link(checkpoint_dir, "latest.pth", path)
        except Exception as exc:
            save_error = f"Aligned checkpoint save failed at step {step}: {type(exc).__name__}: {exc}"
    if world > 1:
        result = [save_error]
        dist.broadcast_object_list(result, src=0, group=object_group)
        save_error = result[0]
    if save_error is not None:
        raise RuntimeError(save_error)
    return path


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _reconcile_log(path: Path, completed_step: int) -> None:
    """Archive records after a checkpoint; they belong to abandoned work."""

    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    retained = [line for line in lines if int(json.loads(line)["step"]) <= completed_step]
    if len(retained) == len(lines):
        return
    archive = path.with_name(f"{path.name}.abandoned_after_{completed_step}")
    if archive.exists():
        raise FileExistsError(f"Cannot archive stale aligned log; archive already exists: {archive}")
    os.replace(path, archive)
    with path.open("x", encoding="utf-8") as stream:
        for line in retained:
            stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=["csgo_seen10_exp32gen_aligned"])
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-checkpoint", default=os.environ.get("PUFFIN_INIT_CHECKPOINT"))
    parser.add_argument("--vae-checkpoint", default=os.environ.get("PUFFIN_VAE_CHECKPOINT"))
    parser.add_argument("--data-root")
    parser.add_argument("--shared-eval-dir")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", "--smoke-updates", type=int, default=2)
    parser.add_argument("--stop-after", type=int, help="Stop an isolated smoke after this completed update")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    _configure_determinism()
    rank, world, local_rank = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("LOCAL_RANK", "0"))
    if args.gradient_accumulation_steps is None:
        quotient, remainder = divmod(128, world * args.micro_batch_size)
        if remainder:
            raise ValueError("128 must be divisible by WORLD_SIZE * micro_batch_size")
        args.gradient_accumulation_steps = quotient
    if not torch.cuda.is_available():
        raise RuntimeError("Aligned Puffin training requires CUDA BF16")
    if world * args.micro_batch_size * args.gradient_accumulation_steps != 128:
        raise ValueError("WORLD_SIZE * micro_batch_size * accumulation must equal 128")
    if args.num_workers < 0 or args.micro_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Microbatch, accumulation, and worker counts must be valid")
    object_group = None
    try:
        if world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if world > 1:
            # Control-plane Python objects remain on CPU; DDP and numeric
            # collectives continue using the default NCCL group.
            object_group = dist.new_group(backend="gloo")
        return _run(args, rank=rank, world=world, local_rank=local_rank,
                    device=device, object_group=object_group)
    finally:
        try:
            if object_group is not None:
                dist.destroy_process_group(object_group)
        finally:
            if world > 1 and dist.is_initialized():
                dist.destroy_process_group()


def _run(args: argparse.Namespace, *, rank: int, world: int, local_rank: int,
         device: torch.device, object_group: dist.ProcessGroup | None) -> int:
    if world > 1 and object_group is None:
        raise ValueError("Distributed aligned training requires the Gloo object group")
    _seed(args.seed + rank)
    max_updates = args.smoke_steps if args.smoke else 19_500
    if args.smoke and not 1 <= max_updates <= 10:
        raise ValueError("Smoke updates must be in [1, 10]")
    if args.stop_after is not None and (not args.smoke or not 1 <= args.stop_after <= max_updates):
        raise ValueError("--stop-after is only supported for isolated smoke updates")
    stop_after = args.stop_after or max_updates
    output_root = aligned_output_root(args.output_root)
    run_dir = output_root / f"seed_{args.seed}"
    checkpoint_dir = run_dir / "checkpoints"
    resume = checkpoint_dir / "latest.pth" if args.resume == "auto" else Path(args.resume).expanduser().resolve() if args.resume else None
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(f"Aligned resume checkpoint missing: {resume}")
    latest = checkpoint_dir / "latest.pth"
    if resume is not None and latest.is_file() and resume.resolve() != latest.resolve():
        raise ValueError("Aligned resume must use the last complete checkpoint in this output directory")
    if run_dir.exists() and any(run_dir.iterdir()) and resume is None:
        raise FileExistsError(f"Aligned run output is non-empty: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()) and resume is not None and not latest.is_file():
        raise ValueError("Cannot resume another run into a non-empty aligned output directory")

    data_root = resolve_data_root(args.data_root)
    shared_eval_dir = resolve_eval_root(args.shared_eval_dir)
    metadata_root = Path(os.environ.get("PUFFIN_SD3_PATH", "")).expanduser().resolve()
    identities = _identity(data_root, shared_eval_dir, metadata_root)
    resolved_configs = {
        "scheduler": json.loads((metadata_root / "scheduler/scheduler_config.json").read_text(encoding="utf-8")),
        "vae": json.loads((metadata_root / "vae/config.json").read_text(encoding="utf-8")),
    }
    train = CsgoSeen10Dataset(data_root=str(data_root), shared_eval_dir=str(shared_eval_dir),
                              split="seen_train", include_target=True, radar_size=224, target_size=448, pose_mode="text")
    valid = CsgoSeen10Dataset(data_root=str(data_root), shared_eval_dir=str(shared_eval_dir),
                              split="seen_validation", include_target=True, radar_size=224, target_size=448,
                              pose_mode="text", max_samples=1 if args.smoke else None)
    if len(train) != 50_000 or (not args.smoke and len(valid) != 5_000):
        raise RuntimeError(f"Published Seen-10 counts mismatch: train={len(train)}, validation={len(valid)}")
    model = build_aligned_model(args.init_checkpoint, args.vae_checkpoint, load_base=resume is None, device=device)
    audit = parameter_audit(model)
    optimizer, group_names = make_optimizer(model)
    scheduler = make_scheduler(optimizer, max_updates=max_updates)
    topology = {"world_size": world, "micro_batch_size": args.micro_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps}
    step = 0
    best = float("inf")
    if resume is not None:
        payload = load_aligned_checkpoint(model, resume)
        if bool(payload.get("smoke_only")) != bool(args.smoke):
            raise ValueError("Smoke and formal aligned checkpoints cannot be mixed")
        if payload.get("identities") != identities:
            raise ValueError("Aligned resume data/scheduler content identities differ")
        if payload.get("resolved_configs") != resolved_configs:
            raise ValueError("Aligned resume scheduler/VAE config contents differ")
        if payload.get("scaler") != {"enabled": False}:
            raise ValueError("Aligned resume AMP scaler state is incompatible")
        if payload.get("determinism") != DETERMINISM:
            raise ValueError("Aligned resume deterministic runtime policy differs")
        if int(payload.get("seed", -1)) != args.seed:
            raise ValueError("Aligned resume seed differs")
        if int(payload.get("max_updates", -1)) != max_updates:
            raise ValueError("Aligned resume target optimizer update count differs")
        if not payload.get("accumulation_boundary"):
            raise ValueError("Aligned checkpoint is not at an optimizer update boundary")
        step = int(payload["step"])
        if payload.get("sampler") != {"epoch": step // 390, "logical_update": step % 390}:
            raise ValueError("Aligned checkpoint sampler cursor differs from completed updates")
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        best = float(payload["best_validation_loss"])
        if payload.get("topology") == topology and len(payload.get("rng_by_rank", [])) == world:
            _restore_rng(payload["rng_by_rank"][rank])
        else:
            warnings.warn("Validation topology changed; resume is semantic but not bitwise equivalent", stacklevel=1)
            _seed(args.seed + rank + step * 1009)
        if rank == 0:
            print(f"resumed aligned completed update={step} from {resume}", flush=True)
    if step > max_updates:
        raise ValueError(f"Checkpoint step {step} exceeds requested updates {max_updates}")
    setup_error = None
    if rank == 0:
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if resume is None:
                audit["optimizer_groups"] = group_names
                audit_path = run_dir / "parameter_audit.json"
                if audit_path.exists():
                    raise FileExistsError(f"Refusing to overwrite aligned parameter audit: {audit_path}")
                with audit_path.open("x", encoding="utf-8") as stream:
                    json.dump(audit, stream, sort_keys=True)
            else:
                _reconcile_log(run_dir / "train_loss.jsonl", step)
                _reconcile_log(run_dir / "seen_validation.jsonl", step)
        except Exception as exc:
            setup_error = f"Aligned output preparation failed: {type(exc).__name__}: {exc}"
    if world > 1:
        result = [setup_error]
        dist.broadcast_object_list(result, src=0, group=object_group)
        setup_error = result[0]
    if setup_error is not None:
        raise RuntimeError(setup_error)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    bare = model.module if isinstance(model, DistributedDataParallel) else model
    model.train()
    train_iter = None
    current_epoch = -1
    for update in range(step, stop_after):
        epoch = update // 390
        if epoch != current_epoch:
            current_epoch = epoch
            sampler = AlignedTrainBatchSampler(len(train), rank=rank, world_size=world,
                                               micro_batch=args.micro_batch_size,
                                               accumulation=args.gradient_accumulation_steps,
                                               seed=args.seed, epoch=epoch, start_update=update % 390)
            loader = _loader(train, batch_sampler=sampler, workers=args.num_workers, seed=args.seed + rank + epoch)
            train_iter = iter(loader)
        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), dtype=torch.float64, device=device)
        for micro in range(args.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration as exc:
                raise RuntimeError("Aligned logical train sampler ended before optimizer boundary") from exc
            sync = micro == args.gradient_accumulation_steps - 1
            context = contextlib.nullcontext() if sync or world == 1 else model.no_sync()
            with context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = _loss(model, batch)
                    scaled = loss / args.gradient_accumulation_steps
                scaled.backward()
            loss_total += loss.detach().double() / args.gradient_accumulation_steps
        torch.nn.utils.clip_grad_norm_([p for p in bare.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step = update + 1
        if world > 1:
            dist.all_reduce(loss_total)
            loss_total /= world
        if rank == 0:
            _append(run_dir / "train_loss.jsonl", {"step": step, "loss": float(loss_total.item()),
                                                    "lora_lr": optimizer.param_groups[0]["lr"]})
            if step % 10 == 0 or args.smoke:
                print(f"aligned update={step}/{max_updates} loss={float(loss_total):.7g}", flush=True)
        milestone = step in MILESTONES or args.smoke
        if milestone:
            validation_loss = _validate(bare, valid, rank=rank, world=world, micro_batch=args.micro_batch_size,
                                        workers=args.num_workers, device=device, seed=args.seed)
            is_best = validation_loss < best
            best = min(best, validation_loss)
            if rank == 0:
                _append(run_dir / "seen_validation.jsonl", {"step": step, "loss": validation_loss,
                                                              "best": best, "count": len(valid)})
                print(f"aligned validation update={step} loss={validation_loss:.7g} best={best:.7g}", flush=True)
            _save(checkpoint_dir, step=step, model=bare, optimizer=optimizer, scheduler=scheduler,
                  rank=rank, world=world, seed=args.seed, topology=topology,
                  identities=identities, best=best, validation_loss=validation_loss,
                  is_best=is_best, smoke_only=args.smoke,
                  max_updates=max_updates, resolved_configs=resolved_configs,
                  object_group=object_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
