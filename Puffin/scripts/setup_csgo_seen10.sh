#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS="$PROJECT_ROOT/requirements_seen10.txt"
CHECKER="$PROJECT_ROOT/scripts/check_csgo_environment.py"
ASSETS="$PROJECT_ROOT/scripts/download_csgo_seen10_assets.py"
MODE=setup
ENV_ONLY=0
PROFILE=aligned

usage() {
    cat <<'HELP'
Usage: bash scripts/setup_csgo_seen10.sh [--env-only] [--check | --check-cuda] [--profile aligned|legacy|all]

Default: use Puffin/.venv/bin/python. An existing environment is checked and
preserved, including its installed package versions. A missing environment is
created with Python 3.10, torch 2.7.0 / torchvision 0.22.0 (CUDA 12.8), and
requirements_seen10.txt. Asset download uses fixed official revisions.

--env-only    Prepare/check Python only; skip assets.
--check       Read-only CPU import and asset integrity checks; no network/CUDA.
--check-cuda  Explicit CPU check followed by a small CUDA BF16 matrix check.
--profile     Asset profile: aligned (default), legacy, or all.

PUFFIN_PYTHON selects another environment bin/python. PUFFIN_BOOTSTRAP_PYTHON
selects a Python 3.10 interpreter for a new venv. PUFFIN_CONDA_EXE selects
Conda if no venv-capable Python 3.10 exists. PUFFIN_TORCH_INDEX_URL selects
the wheel index for a fresh environment; default is the CUDA 12.8 index.
HELP
}

die() { printf 'setup_csgo_seen10: %s\n' "$*" >&2; exit 2; }

while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --env-only) ENV_ONLY=1; shift ;;
        --check) [[ "$MODE" == setup || "$MODE" == check ]] || die "--check conflicts with --check-cuda"; MODE=check; shift ;;
        --check-cuda) [[ "$MODE" == setup || "$MODE" == check-cuda ]] || die "--check-cuda conflicts with --check"; MODE=check-cuda; shift ;;
        --profile) (( $# >= 2 )) || die "--profile needs aligned, legacy, or all"; PROFILE="$2"; shift 2 ;;
        --profile=*) PROFILE="${1#*=}"; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
case "$PROFILE" in aligned|legacy|all) ;; *) die "invalid profile: $PROFILE" ;; esac
[[ -f "$REQUIREMENTS" && -f "$CHECKER" && -f "$ASSETS" ]] || die "missing setup files"

PUFFIN_ENV_PYTHON="${PUFFIN_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
[[ "$PUFFIN_ENV_PYTHON" == /* ]] || PUFFIN_ENV_PYTHON="$PROJECT_ROOT/$PUFFIN_ENV_PYTHON"
[[ "$(basename -- "$(dirname -- "$PUFFIN_ENV_PYTHON")")" == bin ]] || die "PUFFIN_PYTHON must name an environment bin/python"
PUFFIN_RAW_ENV="$(dirname -- "$(dirname -- "$PUFFIN_ENV_PYTHON")")"
[[ ! -L "$PUFFIN_RAW_ENV" ]] || die "environment directory is a symlink; preserved unchanged"
PUFFIN_ENV_DIR="$(realpath -m -- "$PUFFIN_RAW_ENV")"
[[ "$PUFFIN_ENV_DIR" != / && "$PUFFIN_ENV_DIR" != "$PROJECT_ROOT" ]] || die "unsafe environment path"

check_environment() {
    [[ -x "$PUFFIN_ENV_PYTHON" ]] || die "missing Python: $PUFFIN_ENV_PYTHON"
    CUDA_VISIBLE_DEVICES= "$PUFFIN_ENV_PYTHON" "$CHECKER" --expected-prefix "$PUFFIN_ENV_DIR" "$@"
}

if [[ "$MODE" == check ]]; then
    check_environment
    if (( ! ENV_ONLY )); then "$PUFFIN_ENV_PYTHON" "$ASSETS" --check --profile "$PROFILE"; fi
    exit 0
fi
if [[ "$MODE" == check-cuda ]]; then
    check_environment
    "$PUFFIN_ENV_PYTHON" "$CHECKER" --expected-prefix "$PUFFIN_ENV_DIR" --cuda-only
    exit 0
fi

if [[ -e "$PUFFIN_ENV_DIR" || -L "$PUFFIN_ENV_DIR" ]]; then
    [[ -d "$PUFFIN_ENV_DIR" && -x "$PUFFIN_ENV_PYTHON" ]] || die "existing path is not a usable environment; preserved unchanged"
    check_environment --identity-only
    check_environment || die "existing environment is incomplete; preserved unchanged. Set PUFFIN_PYTHON to a separate new environment bin/python"
else
    PUFFIN_BOOTSTRAP="${PUFFIN_BOOTSTRAP_PYTHON:-}"
    python_usable() {
        "$1" -c 'import ensurepip, sys, venv; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' >/dev/null 2>&1
    }
    if [[ -n "$PUFFIN_BOOTSTRAP" ]]; then
        command -v "$PUFFIN_BOOTSTRAP" >/dev/null 2>&1 || die "bootstrap Python unavailable"
        python_usable "$PUFFIN_BOOTSTRAP" || die "bootstrap Python must be 3.10 with venv and ensurepip"
    else
        for candidate in python3.10 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && python_usable "$candidate"; then
                PUFFIN_BOOTSTRAP="$candidate"
                break
            fi
        done
    fi
    if [[ -n "$PUFFIN_BOOTSTRAP" ]]; then
        "$PUFFIN_BOOTSTRAP" -m venv "$PUFFIN_ENV_DIR" || die "venv creation failed; directory preserved for inspection"
    else
        PUFFIN_CONDA="${PUFFIN_CONDA_EXE:-$(command -v conda || true)}"
        [[ -n "$PUFFIN_CONDA" ]] || die "Python 3.10 with venv or Conda is required"
        "$PUFFIN_CONDA" create --prefix "$PUFFIN_ENV_DIR" python=3.10 pip -y || die "Conda creation failed; directory preserved for inspection"
    fi
    check_environment --identity-only
    "$PUFFIN_ENV_PYTHON" -m pip --version >/dev/null || die "pip missing from new environment"
    PUFFIN_TORCH_INDEX="${PUFFIN_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    "$PUFFIN_ENV_PYTHON" -m pip install --disable-pip-version-check --no-input --index-url "$PUFFIN_TORCH_INDEX" 'torch==2.7.0' 'torchvision==0.22.0'
    DS_BUILD_OPS=0 "$PUFFIN_ENV_PYTHON" -m pip install --disable-pip-version-check --no-input -r "$REQUIREMENTS"
    "$PUFFIN_ENV_PYTHON" -m pip check
    check_environment --fresh-pins "$REQUIREMENTS"
fi

if (( ! ENV_ONLY )); then "$PUFFIN_ENV_PYTHON" "$ASSETS" --profile "$PROFILE"; fi
printf 'Puffin Seen-10 environment ready: %s\n' "$PUFFIN_ENV_DIR"
