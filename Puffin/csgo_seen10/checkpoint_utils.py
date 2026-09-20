"""Checkpoint loading helpers for trusted local Seen-10 training artifacts."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path

import torch


_TORCH_TRUE_VALUES = {"1", "y", "yes", "true"}


def load_trusted_state_dict(checkpoint_path: str | Path) -> Mapping:
    """Load weights from a trusted local checkpoint, including MMEngine metadata.

    ``weights_only=False`` is required for MMEngine training checkpoints, whose
    metadata may contain objects such as ``HistoryBuffer``. This unpickles the
    file, so callers must only pass checkpoints they trust.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected a state-dict mapping in checkpoint {checkpoint_path}, "
            f"got {type(checkpoint).__name__}"
        )

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Expected checkpoint state_dict to be a mapping, got {type(state_dict).__name__}"
        )
    if any(not isinstance(key, str) for key in state_dict):
        raise TypeError("Checkpoint state_dict keys must be strings")
    return state_dict


def configure_native_resume_load(env: MutableMapping[str, str]) -> None:
    """Allow PyTorch's default ``torch.load`` in trusted MMEngine resume paths.

    XTuner reads the seed from a resume checkpoint before constructing the
    runner, and MMEngine loads it again to restore the training state. Both
    call ``torch.load`` without ``weights_only``. PyTorch 2.7 defaults to the
    restricted loader, so set its documented process-level override for the
    native resume subprocess only.
    """
    if env.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0") in _TORCH_TRUE_VALUES:
        raise RuntimeError(
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD conflicts with trusted MMEngine resume; "
            "unset it for this resume subprocess"
        )

    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if env.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "0") not in _TORCH_TRUE_VALUES:
        raise RuntimeError(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is disabled; MMEngine resume cannot "
            "load the trusted full training checkpoint"
        )
