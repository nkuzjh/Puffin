"""Seen-10 milestone links and loss artifacts for the native MMEngine runner."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping

import torch
from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _as_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _train_loss(outputs: Any) -> float | None:
    if not isinstance(outputs, Mapping):
        return _as_float(outputs)
    for key in ("loss", "loss_cam2image"):
        if key in outputs:
            value = _as_float(outputs[key])
            if value is not None:
                return value
    log_vars = outputs.get("log_vars")
    if isinstance(log_vars, Mapping):
        return _train_loss(log_vars)
    losses = [_as_float(v) for k, v in outputs.items() if str(k).startswith("loss_")]
    losses = [value for value in losses if value is not None]
    return sum(losses) if losses else None


def _atomic_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() and not link_path.is_symlink():
        raise FileExistsError(f"Refusing to replace non-symlink checkpoint link: {link_path}")
    temporary = link_path.with_name(f".{link_path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    relative_target = os.path.relpath(target_path.resolve(), link_path.parent.resolve())
    os.symlink(relative_target, temporary)
    os.replace(temporary, link_path)


@HOOKS.register_module()
class Seen10TrainingArtifactHook(Hook):
    """Record the main loss and link exactly the five native milestone files.

    ``CheckpointHook`` runs first (its priority is ``VERY_LOW``); this hook is
    ``LOWEST`` and consumes the saved path after each ValLoop milestone.
    """

    priority = "LOWEST"

    def __init__(
        self,
        max_iters: int,
        milestone_count: int = 5,
        validation_seed: int = 20260827,
    ):
        if max_iters <= 0 or milestone_count <= 0 or max_iters % milestone_count:
            raise ValueError(
                f"max_iters must be divisible by milestone_count; got "
                f"max_iters={max_iters}, milestone_count={milestone_count}"
            )
        self.max_iters = int(max_iters)
        self.milestone_count = int(milestone_count)
        self.interval = self.max_iters // self.milestone_count
        self.validation_seed = int(validation_seed)
        self.rank = 0
        self.work_dir: Path | None = None
        self._cpu_rng_state = None
        self._cuda_rng_state = None
        self._python_rng_state = None
        self._best_loss = float("inf")
        self._best_checkpoint: str | None = None

    def before_train(self, runner) -> None:
        self.rank = get_rank()
        self.work_dir = Path(runner.work_dir).expanduser().resolve()
        if self.rank != 0:
            return
        self.work_dir.mkdir(parents=True, exist_ok=True)
        sidecar = self.work_dir / "seen10_best.json"
        if sidecar.is_file():
            state = json.loads(sidecar.read_text(encoding="utf-8"))
            self._best_loss = float(state["val_loss"])
            self._best_checkpoint = str(state["checkpoint"])

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        if self.rank != 0 or self.work_dir is None:
            return
        loss = _train_loss(outputs)
        if loss is None:
            return
        step = int(runner.iter) + 1
        record = json.dumps({"iter": step, "loss": loss}, sort_keys=True)
        with (self.work_dir / "train_loss.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(record + "\n")

    def before_val_epoch(self, runner) -> None:
        # Make each validation pass comparable without perturbing the training RNG.
        self._python_rng_state = random.getstate()
        self._cpu_rng_state = torch.get_rng_state()
        self._cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        random.seed(self.validation_seed)
        torch.manual_seed(self.validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.validation_seed)

    def after_val_epoch(self, runner, metrics=None) -> None:
        if self._python_rng_state is not None:
            random.setstate(self._python_rng_state)
            self._python_rng_state = None
        if self._cpu_rng_state is not None:
            torch.set_rng_state(self._cpu_rng_state)
            self._cpu_rng_state = None
        if self._cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(self._cuda_rng_state)
            self._cuda_rng_state = None

        step = int(runner.iter)
        if step <= 0 or step % self.interval:
            return
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"Seen-10 ValLoop returned no metrics at iter {step}: {metrics!r}")
        val_loss = None
        for key in ("val_loss", "val/loss", "loss"):
            if key in metrics:
                val_loss = _as_float(metrics[key])
                if val_loss is not None:
                    break
        if val_loss is None:
            raise RuntimeError(f"Seen-10 ValLoop did not return a finite val_loss: {metrics!r}")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            backend = torch.distributed.get_backend()
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if backend == "nccl"
                else torch.device("cpu")
            )
            aggregate = torch.tensor(val_loss, dtype=torch.float64, device=device)
            torch.distributed.all_reduce(aggregate, op=torch.distributed.ReduceOp.SUM)
            val_loss = float((aggregate / torch.distributed.get_world_size()).cpu().item())

        if self.rank != 0 or self.work_dir is None:
            return

        last_checkpoint = runner.message_hub.get_info("last_ckpt")
        if last_checkpoint:
            checkpoint = Path(last_checkpoint).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = (self.work_dir / checkpoint).resolve()
        else:
            checkpoint = self.work_dir / f"iter_{step}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Expected native milestone checkpoint before validation artifacts: {checkpoint}"
            )

        _atomic_symlink(self.work_dir / "late.pth", checkpoint)
        _atomic_symlink(self.work_dir / "latest.pth", checkpoint)
        is_best = val_loss < self._best_loss
        if is_best:
            self._best_loss = val_loss
            self._best_checkpoint = str(checkpoint.resolve())
            _atomic_symlink(self.work_dir / "best.pth", checkpoint)
            (self.work_dir / "seen10_best.json").write_text(
                json.dumps(
                    {"iter": step, "val_loss": val_loss, "checkpoint": self._best_checkpoint},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        with (self.work_dir / "seen_validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "iter": step,
                        "val_loss": val_loss,
                        "checkpoint": str(checkpoint.resolve()),
                        "is_best": is_best,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def after_train(self, runner) -> None:
        if self.rank != 0 or self.work_dir is None:
            return
        loss_path = self.work_dir / "train_loss.jsonl"
        if not loss_path.is_file():
            return
        points = []
        for line in loss_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            points.append((int(record["iter"]), float(record["loss"])))
        if not points:
            return
        _write_loss_svg(points, self.work_dir / "loss_curve.svg")


def _write_loss_svg(points: list[tuple[int, float]], output_path: Path) -> None:
    width, height = 900, 440
    left, right, top, bottom = 76, 24, 24, 58
    plot_width, plot_height = width - left - right, height - top - bottom
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 1e-6)
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.04
        y_min -= padding
        y_max += padding
    coordinates = [
        (
            left + (x - x_min) / (x_max - x_min) * plot_width,
            top + (1.0 - (y - y_min) / (y_max - y_min)) * plot_height,
        )
        for x, y in points
    ]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.0f}" y="18" text-anchor="middle" font-family="sans-serif" font-size="16">Puffin CSGO Seen-10 training loss</text>
<path d="M {left} {top} V {height - bottom} H {width - right}" fill="none" stroke="#333" stroke-width="1"/>
<polyline points="{polyline}" fill="none" stroke="#1769aa" stroke-width="1.5"/>
<text x="{width / 2:.0f}" y="{height - 12}" text-anchor="middle" font-family="sans-serif" font-size="13">Iteration</text>
<text x="18" y="{height / 2:.0f}" transform="rotate(-90 18 {height / 2:.0f})" text-anchor="middle" font-family="sans-serif" font-size="13">Main loss</text>
<text x="{left}" y="{height - bottom + 20}" font-family="sans-serif" font-size="11">{x_min}</text>
<text x="{width - right}" y="{height - bottom + 20}" text-anchor="end" font-family="sans-serif" font-size="11">{x_max}</text>
<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="sans-serif" font-size="11">{y_max:.5g}</text>
<text x="{left - 8}" y="{height - bottom}" text-anchor="end" font-family="sans-serif" font-size="11">{y_min:.5g}</text>
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")
