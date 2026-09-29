#!/usr/bin/env python3
"""Read-only Puffin Seen-10 environment check; CUDA runs only with --cuda-only."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata as metadata
import os
import site
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "torch", "torchvision", "numpy", "PIL", "einops", "huggingface_hub",
    "mmengine", "diffusers", "transformers", "peft", "safetensors",
    "timm", "xtuner", "deepspeed", "cv2",
)


def check_identity(expected: Path) -> None:
    actual = Path(sys.prefix).resolve()
    if actual != expected.resolve():
        raise SystemExit(f"Python belongs to {actual}, expected {expected}")
    if not (expected / "pyvenv.cfg").is_file() and not (expected / "conda-meta").is_dir():
        raise SystemExit(f"{expected} is not a virtual or Conda environment")


def check_isolation(expected: Path) -> None:
    """Fail before imports if Python can obtain packages from another environment."""
    prefix = expected.resolve()
    if site.ENABLE_USER_SITE:
        raise SystemExit("User site-packages is enabled; use the Puffin environment wrapper")
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        raise SystemExit("External PYTHONPATH/PYTHONHOME is active; use the Puffin environment wrapper")
    for entry in sys.path:
        path = Path(entry).resolve()
        if any(part in ("site-packages", "dist-packages") for part in path.parts):
            if not path.is_relative_to(prefix):
                raise SystemExit(f"External package search path: {path}")


def check_module_origin(module, expected: Path) -> None:
    origin = getattr(module, "__file__", None)
    if not origin or not Path(origin).resolve().is_relative_to(expected.resolve()):
        raise RuntimeError(f"{module.__name__} was loaded outside the project environment: {origin}")


def check_native_libraries(expected: Path) -> None:
    """Verify that GUI OpenCV's runtime dependencies are private, not host fixes."""
    libraries = ("libGL.so.1", "libglib-2.0.so.0", "libgthread-2.0.so.0")
    prefix = expected.resolve()
    if sys.platform != "linux":
        raise SystemExit("The self-contained CUDA environment currently requires Linux")
    handles = []
    for name in libraries:
        if not (prefix / "lib" / name).is_file():
            raise SystemExit(f"Missing project runtime library {name}; rerun setup with --repair")
        try:
            handles.append(ctypes.CDLL(name))
        except OSError as exc:
            raise SystemExit(f"Cannot load project runtime library {name}: {exc}") from exc
    # Checking filenames alone would miss LD_LIBRARY_PATH/RPATH loading a host copy.
    mappings = Path("/proc/self/maps").read_text().splitlines()
    for name in libraries:
        loaded = []
        for line in mappings:
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            path = Path(fields[5])
            if path.name == name or path.name.startswith(name + "."):
                loaded.append(path.resolve())
        if not loaded or any(not path.is_relative_to(prefix) for path in loaded):
            raise SystemExit(f"{name} did not resolve exclusively inside {prefix}: {loaded}")
    print("Puffin native libraries ready: libGL / GLib / GThread from project environment")


def check_imports(expected: Path | None = None) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(f"Python 3.10 required, found {sys.version.split()[0]}")
    failures = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            if expected is not None:
                check_module_origin(module, expected)
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
    parser.add_argument("--isolated", action="store_true", help="Reject packages from user/base/other environments")
    parser.add_argument("--native-libs", action="store_true", help="Require project-local libGL and GLib on Linux")
    args = parser.parse_args()
    check_identity(args.expected_prefix)
    if args.isolated:
        check_isolation(args.expected_prefix)
    if args.native_libs:
        check_native_libraries(args.expected_prefix)
    if args.cuda_only:
        check_cuda()
    elif not args.identity_only:
        check_imports(args.expected_prefix if args.isolated else None)
        if args.fresh_pins:
            check_fresh_pins(args.fresh_pins)


if __name__ == "__main__":
    main()
