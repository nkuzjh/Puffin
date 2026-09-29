#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
REQUIREMENTS="$PROJECT_ROOT/requirements_seen10.txt"
CHECKER="$PROJECT_ROOT/scripts/check_csgo_environment.py"
ASSETS="$PROJECT_ROOT/scripts/download_csgo_seen10_assets.py"
RUNTIME_MARKER=.puffin-conda-runtime
MODE=setup
ENV_ONLY=0
PROFILE=aligned
CREATED=0

usage() {
    cat <<'HELP'
Usage: bash scripts/setup_csgo_seen10.sh [--env-only] [--check | --check-cuda | --repair] [--profile aligned|legacy|all]

Default: use Puffin/.venv/bin/python. A missing prefix is created with Conda
from conda-forge: Python 3.10, pip, libgl, and libglib. Torch 2.7.0 and
torchvision 0.22.0 use the CUDA 12.8 wheel index, followed by the pinned
requirements_seen10.txt. An existing environment is only checked by default.

--env-only    Prepare/check Python only; skip assets.
--check       Read-only CPU import and asset integrity checks; no network/CUDA.
--check-cuda  Explicit CPU check followed by a small CUDA BF16 matrix check.
--repair      Explicitly repair an existing Conda prefix and check it.
--profile     Asset profile: aligned (default), legacy, or all.

PUFFIN_PYTHON selects another environment bin/python. PUFFIN_CONDA_EXE
selects Conda; CONDA_EXE and the conda executable on PATH are fallbacks.
PUFFIN_TORCH_INDEX_URL selects the PyTorch wheel index, default CUDA 12.8.
PUFFIN_DRIVER_LIBRARY_PATH can add an explicit GPU driver library path.
HELP
}

die() { printf 'setup_csgo_seen10: %s\n' "$*" >&2; exit 2; }

while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --env-only) ENV_ONLY=1; shift ;;
        --check) [[ "$MODE" == setup || "$MODE" == check ]] || die "--check conflicts with another mode"; MODE=check; shift ;;
        --check-cuda) [[ "$MODE" == setup || "$MODE" == check-cuda ]] || die "--check-cuda conflicts with another mode"; MODE=check-cuda; shift ;;
        --repair) [[ "$MODE" == setup || "$MODE" == repair ]] || die "--repair conflicts with another mode"; MODE=repair; shift ;;
        --profile) (( $# >= 2 )) || die "--profile needs aligned, legacy, or all"; PROFILE="$2"; shift 2 ;;
        --profile=*) PROFILE="${1#*=}"; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
case "$PROFILE" in aligned|legacy|all) ;; *) die "invalid profile: $PROFILE" ;; esac
[[ -f "$REQUIREMENTS" && -f "$CHECKER" && -f "$ASSETS" ]] || die "missing setup files"

# Resolve Conda while the caller's PATH still contains its installation.
CONDA_CANDIDATE="${PUFFIN_CONDA_EXE:-${CONDA_EXE:-$(type -P conda || true)}}"
CONDA_PATH=""
if [[ -n "$CONDA_CANDIDATE" ]]; then
    if [[ "$CONDA_CANDIDATE" == */* ]]; then
        CONDA_PATH="$(realpath -e -- "$CONDA_CANDIDATE" 2>/dev/null || true)"
    else
        CONDA_PATH="$(type -P "$CONDA_CANDIDATE" || true)"
        [[ -z "$CONDA_PATH" ]] || CONDA_PATH="$(realpath -e -- "$CONDA_PATH" 2>/dev/null || true)"
    fi
    [[ -n "$CONDA_PATH" && -x "$CONDA_PATH" ]] || CONDA_PATH=""
fi
ACTIVE_CONDA_PREFIX="${CONDA_PREFIX:-}"
ACTIVE_VIRTUAL_ENV="${VIRTUAL_ENV:-}"

source "$PROJECT_ROOT/scripts/puffin_environment.sh"
puffin_use_environment || die "invalid Puffin environment path"
MARKER="$PUFFIN_ENV_PREFIX/$RUNTIME_MARKER"

conda_command() {
    [[ -n "$CONDA_PATH" ]] || die "Conda is required; set PUFFIN_CONDA_EXE to a working executable${CONDA_CANDIDATE:+ (unavailable: $CONDA_CANDIDATE)}"
    env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE \
        -u VIRTUAL_ENV -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL \
        PYTHONNOUSERSITE=1 PATH="$(dirname -- "$CONDA_PATH"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        "$CONDA_PATH" "$@"
}

check_base_safety() {
    [[ -n "$CONDA_PATH" ]] || return 0
    local base
    base="$(conda_command info --base)" || die "cannot inspect Conda base"
    [[ "$PUFFIN_ENV_PREFIX" != "$(realpath -m -- "$base")" ]] || die "Conda base cannot be a Puffin environment"
}

active_prefix_matches() {
    local active
    for active in "$ACTIVE_CONDA_PREFIX" "$ACTIVE_VIRTUAL_ENV"; do
        [[ -n "$active" ]] || continue
        [[ "$(realpath -m -- "$active")" != "$PUFFIN_ENV_PREFIX" ]] || return 0
    done
    return 1
}

check_running_prefix_users() {
    [[ -d /proc ]] || die "cannot inspect running processes before --repair: /proc is unavailable"
    "$PUFFIN_PYTHON" -s - "$PUFFIN_ENV_PREFIX" <<'PY'
import os
import sys
from pathlib import Path

prefix = Path(sys.argv[1]).resolve()
own_pid = os.getpid()
busy = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdecimal() or int(entry.name) == own_pid:
        continue
    try:
        executable = Path(os.readlink(entry / 'exe').removesuffix(' (deleted)')).resolve()
    except (OSError, PermissionError):
        continue
    if executable.is_relative_to(prefix):
        busy.append((int(entry.name), executable))
if busy:
    for pid, executable in sorted(busy):
        print(f'Puffin prefix is in use by PID {pid}: {executable}', file=sys.stderr)
    raise SystemExit(1)
PY
}

check_environment() {
    [[ -x "$PUFFIN_PYTHON" ]] || die "missing Python: $PUFFIN_PYTHON"
    local checker_args=(--expected-prefix "$PUFFIN_ENV_PREFIX" --isolated)
    [[ ! -f "$MARKER" ]] || checker_args+=(--native-libs)
    CUDA_VISIBLE_DEVICES= "$PUFFIN_PYTHON" -s "$CHECKER" "${checker_args[@]}" "$@"
}

check_base_safety
if [[ "$MODE" == check ]]; then
    check_environment
    if (( ! ENV_ONLY )); then "$PUFFIN_PYTHON" -s "$ASSETS" --check --profile "$PROFILE"; fi
    exit 0
fi
if [[ "$MODE" == check-cuda ]]; then
    check_environment
    "$PUFFIN_PYTHON" -s "$CHECKER" --expected-prefix "$PUFFIN_ENV_PREFIX" --isolated --cuda-only
    exit 0
fi

if [[ "$MODE" == repair ]]; then
    [[ -n "$CONDA_PATH" ]] || die "Conda is required for --repair; set PUFFIN_CONDA_EXE to a working executable"
    [[ -d "$PUFFIN_ENV_PREFIX/conda-meta" && -x "$PUFFIN_PYTHON" ]] ||
        die "--repair requires an existing Conda prefix; choose a new PUFFIN_PYTHON bin/python for a fresh prefix"
    if active_prefix_matches; then die "deactivate the selected environment before --repair"; fi
    [[ -x "$PUFFIN_PYTHON" ]] || die "missing Python: $PUFFIN_PYTHON"
    CUDA_VISIBLE_DEVICES= "$PUFFIN_PYTHON" -s "$CHECKER" --expected-prefix "$PUFFIN_ENV_PREFIX" --isolated --identity-only
    check_running_prefix_users || die "--repair refused because the selected prefix is in use; stop those jobs before retrying"
elif [[ -e "$PUFFIN_ENV_PREFIX" || -L "$PUFFIN_ENV_PREFIX" ]]; then
    [[ -d "$PUFFIN_ENV_PREFIX" && -x "$PUFFIN_PYTHON" ]] || die "existing path is not a usable environment; preserved unchanged"
    check_environment --identity-only
    check_environment || die "existing environment is incomplete; preserved unchanged. Use --repair for a Conda prefix or set PUFFIN_PYTHON to a new prefix/bin/python"
else
    [[ -n "$CONDA_PATH" ]] || die "Conda is required to create an isolated Puffin environment; set PUFFIN_CONDA_EXE"
    if active_prefix_matches; then die "deactivate the selected environment before creating it"; fi
    conda_command create --prefix "$PUFFIN_ENV_PREFIX" --channel conda-forge --override-channels --strict-channel-priority --no-default-packages -y \
        python=3.10 pip libgl libglib || die "Conda creation failed; prefix preserved for inspection"
    CREATED=1
    check_environment --identity-only
fi

if [[ "$MODE" == repair || "$CREATED" == 1 ]]; then
    # This branch only reaches a fresh Conda prefix or explicit --repair.
    if [[ "$MODE" == repair ]]; then
        conda_command install --prefix "$PUFFIN_ENV_PREFIX" --channel conda-forge --override-channels --strict-channel-priority -y \
            python=3.10 pip libgl libglib || die "Conda native-library repair failed; prefix preserved for inspection"
    fi
    [[ -d "$PUFFIN_ENV_PREFIX/conda-meta" ]] || die "expected a Conda prefix after creation"
    for pip_name in ${!PIP_@}; do unset "$pip_name"; done
    export PIP_CONFIG_FILE=/dev/null PIP_USER=0
    "$PUFFIN_PYTHON" -s -m pip --version >/dev/null || die "pip missing from Conda prefix"
    TORCH_INDEX="${PUFFIN_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    "$PUFFIN_PYTHON" -s -m pip install --disable-pip-version-check --no-input --index-url "$TORCH_INDEX" \
        'torch==2.7.0' 'torchvision==0.22.0'
    DS_BUILD_OPS=0 "$PUFFIN_PYTHON" -s -m pip install --disable-pip-version-check --no-input \
        'torch==2.7.0' 'torchvision==0.22.0' -r "$REQUIREMENTS"
    "$PUFFIN_PYTHON" -s -m pip check
    check_environment --native-libs --fresh-pins "$REQUIREMENTS"
    : > "$MARKER"
fi

if (( ! ENV_ONLY )); then "$PUFFIN_PYTHON" -s "$ASSETS" --profile "$PROFILE"; fi
printf 'Puffin Seen-10 environment ready: %s\n' "$PUFFIN_ENV_PREFIX"
