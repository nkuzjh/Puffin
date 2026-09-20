"""Small deterministic output helpers shared by inference and unit tests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from PIL import Image


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_signature(fields: dict) -> str:
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_matching_manifest(path: Path, signature: str) -> dict | None:
    if path.is_symlink():
        raise ValueError(f"Refusing to follow an inference manifest symlink: {path}")
    if not path.exists():
        return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot trust existing inference manifest: {path}") from exc
    if not isinstance(existing, dict) or existing.get("run_signature") != signature:
        raise ValueError(
            f"Existing output manifest belongs to a different checkpoint/seed/settings: {path}; "
            "choose a new output root"
        )
    return existing


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def is_valid_rgb_jpeg(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            if image.format != "JPEG" or image.mode != "RGB" or image.size != (448, 448):
                return False
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def write_rgb_jpeg(image_tensor: torch.Tensor, output_path: Path) -> bool:
    """Write exact benchmark encoding; return False only for a valid existing file."""
    if image_tensor.ndim != 3 or tuple(image_tensor.shape) != (3, 448, 448):
        raise ValueError(f"Expected generated RGB tensor (3, 448, 448), got {tuple(image_tensor.shape)}")
    image = image_tensor.detach().float().cpu()
    # Match UniLIP's current numpy_to_pil conversion: [-1, 1] -> [0, 1],
    # multiply by 255, round to the nearest integer, then encode as RGB JPEG.
    image = torch.clamp((image + 1.0) * 0.5, 0, 1).mul_(255).round_().to(torch.uint8)
    array = image.permute(1, 2, 0).numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".jpg", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        Image.fromarray(array).save(temporary_path, format="JPEG")
        try:
            # Hard-link creation is atomic and fails instead of replacing an existing image.
            os.link(temporary_path, output_path)
            return True
        except FileExistsError:
            if is_valid_rgb_jpeg(output_path):
                return False
            raise ValueError(f"Refusing to overwrite an invalid existing prediction: {output_path}")
    finally:
        temporary_path.unlink(missing_ok=True)
