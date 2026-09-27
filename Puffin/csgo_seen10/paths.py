"""Portable runtime paths; explicit choices never silently fall back."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
EXPERIMENT = "csgo_seen10_exp32gen_aligned"


def project_path(value: str | os.PathLike) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _directory(cli, env_names, default, required_file):
    value = cli
    if value is None:
        value = next((os.environ[name] for name in env_names if os.environ.get(name)), None)
    path = project_path(value if value is not None else default)
    if not (path / required_file).is_file():
        raise FileNotFoundError(f"Required {required_file} is missing under {path}; set {env_names[0]} or the CLI path")
    return path


def resolve_data_root(cli=None) -> Path:
    return _directory(cli, ("CSGO_DATA_ROOT", "DATA_ROOT"), WORKSPACE_ROOT / "UniLIP/data/csgo_benchmark_v2", "benchmark_manifest.json")


def resolve_eval_root(cli=None) -> Path:
    return _directory(cli, ("SHARED_EVAL_DIR",), WORKSPACE_ROOT / "csgo_benchmark_v2_eval_general", "protocol.py")


def resolve_eval_python(cli=None, eval_root=None) -> Path:
    root = resolve_eval_root(eval_root)
    value = cli or os.environ.get("EVAL_PYTHON") or os.environ.get("UNILIP_PYTHON")
    path = project_path(value) if value else root / ".venv/bin/python"
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"Evaluator Python unavailable: {path}. Run {root / 'setup_env.sh'} explicitly.")
    return path


def resolve_model_python() -> Path:
    value = os.environ.get("PUFFIN_PYTHON")
    if value:
        found = shutil.which(value) if os.sep not in value else None
        path = project_path(found or value)
    else:
        path = PROJECT_ROOT / ".venv/bin/python"
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"Puffin Python unavailable: {path}; run scripts/setup_csgo_seen10.sh --env-only")
    return path


def aligned_output_root(cli=None) -> Path:
    # OUTPUT_ROOT may be a legacy shell variable; do not inherit it implicitly.
    return project_path(cli or PROJECT_ROOT / "outputs" / EXPERIMENT / "Puffin")
