"""Config-only model factories for loading Seen-10 weights from Puffin files."""

from __future__ import annotations

import copy

import torch


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported model dtype: {dtype!r}")
    return dtype


class EmptyQwenFromConfig:
    """Build Qwen from its small HF config without fetching model weights."""

    def __new__(
        cls,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        from transformers import AutoConfig, AutoModelForCausalLM

        source_kwargs = {}
        if revision is not None:
            source_kwargs["revision"] = revision
        if local_files_only:
            source_kwargs["local_files_only"] = True
        config = AutoConfig.from_pretrained(model_name_or_path, **source_kwargs)
        config._attn_implementation = attn_implementation
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=_resolve_dtype(dtype),
            attn_implementation=attn_implementation,
        )
        requested_dtype = _resolve_dtype(dtype)
        if model.dtype != requested_dtype:
            model.to(dtype=requested_dtype)
        return model


class EmptyRadioFromConfig:
    """Build RADIO's bf16 architecture without downloading its weights."""

    def __new__(
        cls,
        model_name_or_path: str = "nvidia/C-RADIOv3-H",
        dtype: str = "bfloat16",
        amp_dtype: str = "bfloat16",
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        from src.models.radiov3.hf_model import RADIOConfig, RADIOModel

        source_kwargs = {}
        if revision is not None:
            source_kwargs["revision"] = revision
        if local_files_only:
            source_kwargs["local_files_only"] = True
        config = copy.deepcopy(RADIOConfig.from_pretrained(model_name_or_path, **source_kwargs))
        config.args = dict(config.args or {})
        config.args.update(
            dtype=dtype,
            amp_dtype=amp_dtype,
            pretrained=False,
            initial_checkpoint=None,
        )
        model_kwargs = dict(config.args.get("model_kwargs") or {})
        model_kwargs["weight_init"] = "skip"
        config.args["model_kwargs"] = model_kwargs
        return RADIOModel(config)
