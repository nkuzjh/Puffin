"""Aligned Puffin model construction, exact PEFT coverage, and checkpoint ABI."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from mmengine.config import Config


FORMAT = "puffin_seen10_exp32gen_aligned_v1"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/pipelines/csgo_seen10_exp32gen_aligned.py"
OFFICIAL_BASE_SHA256 = "4045661c81b29adc8aa1cc22079ddbbf86d353d2e0f35c0ffec310f193504257"
OFFICIAL_VAE_SHA256 = "9c9bdf9aabbd3efce5450eb628cf8d965c37f3fa4210a7fef151a94a939db211"
EXPECTED_LORA = {"llm": 36_929_536, "transformer": 41_877_504, "connectors": 3_620_864}
EXPECTED_FULL = 6_395_904
EXPECTED_TRAINABLE = sum(EXPECTED_LORA.values()) + EXPECTED_FULL
DETERMINISM = {
    "torch_deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cublas_workspace_config": ":4096:8",
}

_PATTERNS = {
    "llm": re.compile(r"^model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"),
    "transformer": re.compile(
        r"^transformer_blocks\.\d+\."
        r"(?:attn\.(?:to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out)"
        r"|ff(?:_context)?\.net\.(?:0\.proj|2))$"
    ),
    "connectors": re.compile(r"^layers\.\d+\.(?:self_attn\.(?:q|k|v|out)_proj|mlp\.(?:fc1|fc2))$"),
}


def aligned_semantics() -> dict[str, Any]:
    """Portable recipe identity; paths and worker count are intentionally absent."""

    return {
        "format": FORMAT,
        "experiment": "csgo_seen10_exp32gen_aligned",
        "conditioning": {"radar_size": 224, "target_size": 448, "pose_mode": "text", "train_dropout": 0.1, "validation_dropout": 0.0},
        "flow": "native_puffin_sd3_uniform_density_none_weight",
        "global_batch": 128,
        "train_examples": 50_000,
        "examples_per_epoch": 49_920,
        "updates_per_epoch": 390,
        "max_updates": 19_500,
        "warmup_updates": 585,
        "milestones": [4_000, 8_000, 12_000, 16_000, 19_500],
        "validation_examples": 5_000,
        "validation_seed": 20260827,
        "lora": {"llm": [32, 64, 0.05], "transformer": [32, 64, 0.05], "connectors": [16, 64, 0.05]},
        "full_modules": ["meta_queries", "projector_1", "projector_2"],
        "optimizer": {"type": "AdamW", "lora_lr": 1e-4, "full_lr": 5e-6, "betas": [0.9, 0.95], "decay": 0.05, "clip": 1.0},
        "determinism": dict(DETERMINISM),
    }


def config_fingerprint() -> str:
    encoded = json.dumps(aligned_semantics(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _matching_linears(module: torch.nn.Module, pattern: re.Pattern[str]) -> list[str]:
    names = [name for name, child in module.named_modules() if pattern.fullmatch(name) and isinstance(child, torch.nn.Linear)]
    if not names:
        raise RuntimeError(f"No aligned LoRA targets matched {pattern.pattern}")
    return names


def _inject(module: torch.nn.Module, pattern: re.Pattern[str], rank: int, alpha: int) -> list[str]:
    from peft import LoraConfig, inject_adapter_in_model

    targets = _matching_linears(module, pattern)
    inject_adapter_in_model(
        LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none", target_modules=targets),
        module,
    )
    # PEFT target matching is suffix based. A shape and count audit below catches
    # accidental extra matches after a dependency or architecture change.
    return targets


def _unique_named_parameters(model: torch.nn.Module):
    seen: set[int] = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) not in seen:
            seen.add(id(parameter))
            yield name, parameter


def parameter_audit(model: torch.nn.Module, *, strict: bool = True) -> dict[str, Any]:
    groups: dict[str, list[str]] = {"llm": [], "transformer": [], "connectors": [], "full": [], "unexpected": []}
    counts = {key: 0 for key in groups}
    records: list[dict[str, Any]] = []
    total = 0
    frozen = 0
    lora_modules: set[str] = set()
    for name, parameter in _unique_named_parameters(model):
        size = parameter.numel()
        total += size
        if not parameter.requires_grad:
            frozen += size
            records.append({"name": name, "shape": list(parameter.shape), "numel": size,
                            "dtype": str(parameter.dtype), "requires_grad": False,
                            "group": "frozen", "lr": None, "weight_decay": None})
            continue
        if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name:
            prefix = name.split(".", 1)[0]
            group = "connectors" if prefix in ("connector_1", "connector_2", "llm2connector_1", "llm2connector_2") else prefix
            lora_modules.add(name.rsplit(".lora_", 1)[0])
        elif name == "meta_queries" or name.startswith(("projector_1.", "projector_2.")):
            group = "full"
        else:
            group = "unexpected"
        groups[group].append(name)
        counts[group] += size
        is_lora = group in ("llm", "transformer", "connectors")
        records.append({"name": name, "shape": list(parameter.shape), "numel": size,
                        "dtype": str(parameter.dtype), "requires_grad": True,
                        "group": ("lora" if is_lora else "full") + ("_decay" if parameter.ndim >= 2 else "_no_decay"),
                        "lr": 1e-4 if is_lora else 5e-6,
                        "weight_decay": 0.05 if parameter.ndim >= 2 else 0.0})
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"Aligned trainable parameter is not FP32: {name}: {parameter.dtype}")
    if any(p.requires_grad for module in (model.visual_encoder, model.projector, model.vae) for p in module.parameters()):
        raise RuntimeError("RADIO, visual projector, and VAE must be frozen")
    if strict:
        for group, expected in {**EXPECTED_LORA, "full": EXPECTED_FULL, "unexpected": 0}.items():
            if counts[group] != expected:
                raise RuntimeError(f"Aligned {group} trainable count {counts[group]:,} != {expected:,}")
        if sum(counts.values()) != EXPECTED_TRAINABLE:
            raise RuntimeError("Aligned total trainable parameter count differs from approved coverage")
        invalid_targets = [name for name in lora_modules if not (
            (name.startswith("llm.") and _PATTERNS["llm"].fullmatch(name[4:]))
            or (name.startswith("transformer.") and _PATTERNS["transformer"].fullmatch(name[12:]))
            or (name.startswith(("connector_1.", "connector_2.")) and _PATTERNS["connectors"].fullmatch(name.split(".", 1)[1]))
            or name in ("llm2connector_1", "llm2connector_2")
        )]
        if invalid_targets:
            raise RuntimeError(f"Aligned LoRA includes unapproved modules: {invalid_targets[:12]}")
    return {"counts": counts, "trainable": sum(counts.values()), "total_unique": total,
            "trainable_fraction": sum(counts.values()) / total, "frozen": frozen,
            "names": groups, "lora_modules": sorted(lora_modules), "parameters": records}


def apply_aligned_peft(model: torch.nn.Module, *, strict: bool = True) -> dict[str, Any]:
    """Freeze the official Base, then inject the three approved LoRA families."""

    model.requires_grad_(False)
    expected_targets = {f"llm.{name}" for name in _inject(model.llm, _PATTERNS["llm"], 32, 64)}
    expected_targets.update(f"transformer.{name}" for name in _inject(model.transformer, _PATTERNS["transformer"], 32, 64))
    for prefix, connector in (("connector_1", model.connector_1), ("connector_2", model.connector_2)):
        expected_targets.update(f"{prefix}.{name}" for name in _inject(connector, _PATTERNS["connectors"], 16, 64))
    from peft import LoraConfig, inject_adapter_in_model

    inject_adapter_in_model(
        LoraConfig(r=16, lora_alpha=64, lora_dropout=0.05, bias="none",
                   target_modules=["llm2connector_1", "llm2connector_2"]),
        model,
    )
    expected_targets.update(("llm2connector_1", "llm2connector_2"))
    for name, parameter in _unique_named_parameters(model):
        if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name:
            parameter.requires_grad_(True)
        elif name == "meta_queries" or name.startswith(("projector_1.", "projector_2.")):
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.visual_encoder.requires_grad_(False)
    model.projector.requires_grad_(False)
    model.vae.requires_grad_(False)
    audit = parameter_audit(model, strict=strict)
    if set(audit["lora_modules"]) != expected_targets:
        missing = sorted(expected_targets - set(audit["lora_modules"]))
        extra = sorted(set(audit["lora_modules"]) - expected_targets)
        raise RuntimeError(f"Aligned LoRA target coverage mismatch: missing={missing[:12]}, extra={extra[:12]}")
    return audit


def build_aligned_model(
    init_checkpoint: str | Path | None = None,
    vae_checkpoint: str | Path | None = None,
    *,
    load_base: bool = True,
    device: torch.device | str | None = None,
    config_path: str | Path | None = None,
    enable_lora: bool = True,
) -> torch.nn.Module:
    """Build the aligned architecture; optionally load official Base and VAE first.

    Full aligned checkpoint loading uses ``load_base=False`` and then a strict
    ``load_aligned_checkpoint`` call. Fresh training uses both official files.
    """

    from xtuner.registry import BUILDER

    config = Config.fromfile(str(config_path or CONFIG_PATH))
    model_config = config.model.copy()
    if load_base:
        if not init_checkpoint or not vae_checkpoint:
            raise ValueError("Fresh aligned model construction requires official Base and Puffin VAE checkpoints")
        for path, expected in ((init_checkpoint, OFFICIAL_BASE_SHA256), (vae_checkpoint, OFFICIAL_VAE_SHA256)):
            path = Path(path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Official Puffin asset SHA256 mismatch: {path}")
        model_config.pretrained_pth = str(init_checkpoint)
        model_config.pretrained_vae_pth = str(vae_checkpoint)
    else:
        model_config.pretrained_pth = None
        model_config.pretrained_vae_pth = None
    model = BUILDER.build(model_config)
    if device is not None:
        model.to(device)
    if enable_lora:
        audit = apply_aligned_peft(model)
        print("aligned parameter audit:", json.dumps({key: value for key, value in audit.items() if key not in ("names", "parameters")}, sort_keys=True), flush=True)
    return model


def load_aligned_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Strictly load a trusted local full-state aligned training checkpoint."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"Not a {FORMAT} checkpoint: {checkpoint_path}")
    fingerprint = expected_fingerprint or config_fingerprint()
    if payload.get("config_fingerprint") != fingerprint:
        raise ValueError("Aligned checkpoint recipe fingerprint mismatch")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Aligned checkpoint has no full model state_dict")
    model.load_state_dict(payload["state_dict"], strict=True)
    parameter_audit(model)
    return payload


__all__ = [
    "FORMAT", "CONFIG_PATH", "EXPECTED_LORA", "EXPECTED_FULL", "EXPECTED_TRAINABLE", "DETERMINISM",
    "aligned_semantics", "config_fingerprint", "parameter_audit", "apply_aligned_peft",
    "build_aligned_model", "load_aligned_checkpoint",
]
