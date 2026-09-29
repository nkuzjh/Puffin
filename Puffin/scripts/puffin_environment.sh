#!/usr/bin/env bash
# Source this file and call puffin_use_environment [bin/python]. No writes.

puffin_use_environment() {
    local script_dir project_root requested raw_prefix prefix home_prefix
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || return 2
    project_root="$(cd -- "$script_dir/.." && pwd -P)" || return 2
    requested="${1:-${PUFFIN_PYTHON:-$project_root/.venv/bin/python}}"
    [[ "$requested" == /* ]] || requested="$project_root/$requested"
    if [[ "$(basename -- "$requested")" != python || "$(basename -- "$(dirname -- "$requested")")" != bin ]]; then
        printf 'puffin_environment: PUFFIN_PYTHON must name an environment bin/python\n' >&2
        return 2
    fi
    raw_prefix="$(dirname -- "$(dirname -- "$requested")")"
    if [[ -L "$raw_prefix" ]]; then
        printf 'puffin_environment: environment directory is a symlink: %s\n' "$raw_prefix" >&2
        return 2
    fi
    prefix="$(realpath -m -- "$raw_prefix")" || return 2
    home_prefix="$(realpath -m -- "${HOME:-/nonexistent}")" || return 2
    if [[ "$prefix" == / || "$prefix" == "$project_root" || "$project_root" == "$prefix/"* || "$prefix" == "$home_prefix" ]]; then
        printf 'puffin_environment: unsafe environment prefix: %s\n' "$prefix" >&2
        return 2
    fi

    export PUFFIN_PROJECT_ROOT="$project_root"
    export PUFFIN_ENV_PREFIX="$prefix"
    export PUFFIN_PYTHON="$prefix/bin/python"
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
    unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
    export PATH="$prefix/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    export LD_LIBRARY_PATH="$prefix/lib${PUFFIN_DRIVER_LIBRARY_PATH:+:$PUFFIN_DRIVER_LIBRARY_PATH}"
}
