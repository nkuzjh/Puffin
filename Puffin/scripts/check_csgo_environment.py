#!/usr/bin/env python3
"""Read-only Puffin Seen-10 environment check; CUDA runs only with --cuda-only."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import os
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "torch", "torchvision", "numpy", "PIL", "einops", "huggingface_hub",
    "mmengine", "diffusers", "transformers", "peft", "safetensors",
    "timm", "xtuner", "deepspeed",
)


def check_identity(expected: Path) -> None:
    actual = Path(sys.prefix).resolve()
    if actual != expected.resolve():
        raise SystemExit(f"Python belongs to {actual}, expected {expected}")
    if not (expected / "pyvenv.cfg").is_file() and not (expected / "conda-meta").is_dir():
        raise SystemExit(f"{expected} is not a virtual or Conda environment")


def check_imports() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(f"Python 3.10 required, found {sys.version.split()[0]}")
    failures = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit("Core import failures:\n  " + "\n  ".join(failures))
    import torch

    if torch.cuda.is_initialized():
        raise SystemExit("CUDA was initialized during the CPU-only check")
    torch_version = metadata.version("torch")
    vision_version = metadata.version("torchvision")
    if torch_version.split("+", 1)[0] != "2.7.0" or vision_version.split("+", 1)[0] != "0.22.0":
        raise SystemExit(f"Expected torch 2.7.0 and torchvision 0.22.0, found {torch_version} and {vision_version}")
    print(f"Puffin CPU environment ready: Python {sys.version.split()[0]}, torch {torch_version}, torchvision {vision_version}; CUDA uninitialized")


def check_fresh_pins(requirements: Path) -> None:
    mismatches = []
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, expected = line.split("==", 1)
        name = name.split("[", 1)[0]
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "missing"
        if actual.split("+", 1)[0] != expected:
            mismatches.append(f"{name}: expected {expected}, found {actual}")
    if mismatches:
        raise SystemExit("Fresh dependency pins differ:\n  " + "\n  ".join(mismatches))


def check_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable to PyTorch")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("GPU does not support BF16")
    device = torch.device("cuda:0")
    matrix = torch.eye(8, dtype=torch.bfloat16, device=device)
    result = matrix @ matrix
    torch.cuda.synchronize(device)
    if not torch.allclose(result, matrix):
        raise SystemExit("CUDA BF16 matrix check failed")
    capability = torch.cuda.get_device_capability(device)
    print(f"CUDA BF16 check passed: {torch.cuda.get_device_name(device)}, sm_{capability[0]}{capability[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-prefix", type=Path, required=True)
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--cuda-only", action="store_true")
    parser.add_argument("--fresh-pins", type=Path)
    args = parser.parse_args()
    check_identity(args.expected_prefix)
    if args.cuda_only:
        check_cuda()
    elif not args.identity_only:
        check_imports()
        if args.fresh_pins:
            check_fresh_pins(args.fresh_pins)


if __name__ == "__main__":
    main()
