"""Manifest-driven Seen-10 image/radar/pose dataset for Puffin generation.

The shared evaluator's protocol reader is the source of truth for identities,
calibration, mapped radar paths, and continuous clip order. This module only
adapts those rows to Puffin's existing ``cam2image`` input contract.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from csgo_seen10.prompts import build_pose_prompt


DEFAULT_DATA_ROOT = "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2"
DEFAULT_SHARED_EVAL_DIR = "/home/jiahao/task/csgo_benchmark_v2_eval_general"
TASK_TEXT = "Generate a first-person view from the provided radar map. Map name: {map_name}."


def _load_protocol(shared_eval_dir: str | os.PathLike[str] | None):
    eval_dir = Path(
        shared_eval_dir
        or os.environ.get("SHARED_EVAL_DIR", DEFAULT_SHARED_EVAL_DIR)
    ).expanduser().resolve()
    protocol_path = eval_dir / "protocol.py"
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Shared CSGO evaluator protocol not found: {protocol_path}. "
            "Set SHARED_EVAL_DIR to the read-only evaluator directory."
        )

    module_name = f"_puffin_csgo_protocol_{abs(hash(protocol_path))}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, protocol_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load shared protocol module: {protocol_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def _image_tensor(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BICUBIC
        )
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


class CsgoSeen10Dataset(Dataset):
    """Read one published Seen-10 split and adapt it to Puffin cam2image.

    ``include_target=False`` is the inference contract: the target image path
    remains metadata only and is never opened. Legacy uses normalized numeric
    pose; the explicit aligned text mode adds no numerical adapter input.
    """

    def __init__(
        self,
        data_root: str = DEFAULT_DATA_ROOT,
        split: str = "seen_train",
        include_target: bool = True,
        image_size: int = 448,
        maps: Sequence[str] | None = None,
        max_samples: int | None = None,
        shared_eval_dir: str | None = None,
        radar_size: int | None = None,
        target_size: int | None = None,
        pose_mode: str = "numeric",
    ):
        if image_size != 448:
            raise ValueError(f"CSGO Benchmark v2 generation requires 448px, got {image_size}")
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.include_target = bool(include_target)
        self.image_size = int(image_size)
        self.radar_size = int(radar_size if radar_size is not None else image_size)
        self.target_size = int(target_size if target_size is not None else image_size)
        if self.target_size != 448 or self.radar_size not in (224, 448):
            raise ValueError("Seen-10 requires FPV448 and radar224 or legacy radar448")
        if pose_mode not in ("numeric", "text"):
            raise ValueError(f"Unsupported pose mode: {pose_mode}")
        self.pose_mode = pose_mode
        self._radar_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        protocol = _load_protocol(shared_eval_dir)
        self.benchmark = protocol.BenchmarkData(self.data_root)
        if split in ("seen_continuous", "continuous"):
            self.rows = self.benchmark.rows(split, maps=maps, max_samples=max_samples)
        else:
            self.rows = self.benchmark.rows(split, maps=maps, max_samples=max_samples)
        if not self.rows:
            raise ValueError(f"Seen-10 split {split!r} contains no rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        radar_path = row["radar_path"]
        radar = self._radar_cache.get(radar_path)
        if radar is None:
            radar = _image_tensor(radar_path, self.radar_size)
            self._radar_cache[radar_path] = radar
            if len(self._radar_cache) > 10:
                self._radar_cache.popitem(last=False)
        else:
            self._radar_cache.move_to_end(radar_path)
        item: dict[str, Any] = {
            "cam_values": radar,
            "text": build_pose_prompt(row) if self.pose_mode == "text" else TASK_TEXT.format(map_name=row["map_name"]),
            "metadata": {
                "sample_id": row["sample_id"],
                "map_name": row["map_name"],
                "file_frame": row["file_frame"],
                "clip_id": row.get("clip_id"),
                "frame_index": row.get("frame_index"),
                "target_path": row["image_path"],
            },
        }
        if self.pose_mode == "numeric":
            item["pose_values"] = torch.tensor(row["pose"], dtype=torch.float32)
        if self.include_target:
            item["pixel_values"] = _image_tensor(row["image_path"], self.target_size)
        return item


def collate_seen10(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Batch Seen-10 samples using the existing Puffin cam2image field names."""

    if not samples:
        raise ValueError("Cannot collate an empty Seen-10 batch")
    include_target = ["pixel_values" in sample for sample in samples]
    if any(include_target) and not all(include_target):
        raise ValueError("A batch cannot mix target-free and target-bearing samples")
    include_pose = ["pose_values" in sample for sample in samples]
    if any(include_pose) and not all(include_pose):
        raise ValueError("A batch cannot mix numeric and text-only pose contracts")

    data: dict[str, Any] = {
        "cam2image": {
            "cam_values": [[sample["cam_values"]] for sample in samples],
            "texts": [sample["text"] for sample in samples],
            "metadata": [dict(sample["metadata"]) for sample in samples],
        }
    }
    if all(include_pose):
        data["cam2image"]["pose_values"] = torch.stack([sample["pose_values"] for sample in samples])
    if all(include_target):
        data["cam2image"]["pixel_values"] = [sample["pixel_values"] for sample in samples]
    return {"data": data, "data_samples": None}


class CollateSeen10:
    """Registry-builder-friendly callable wrapper for MMEngine configs."""

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return collate_seen10(samples)
