#!/usr/bin/env bash
# Source this file to select Puffin Python, native libraries, and asset paths.

puffin_activate_csgo_seen10() {
    local script_dir profile asset_exports
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || return 2
    profile=aligned
    while (( $# )); do
        case "$1" in
            --profile) [[ $# -ge 2 ]] || { printf 'activate_csgo_seen10: --profile needs a value\n' >&2; return 2; }; profile="$2"; shift 2 ;;
            --profile=*) profile="${1#*=}"; shift ;;
            *) printf 'activate_csgo_seen10: unknown argument: %s\n' "$1" >&2; return 2 ;;
        esac
    done
    case "$profile" in aligned|legacy|all) ;; *) printf 'activate_csgo_seen10: invalid profile: %s\n' "$profile" >&2; return 2 ;; esac
    source "$script_dir/puffin_environment.sh"
    puffin_use_environment || return 2
    [[ -x "$PUFFIN_PYTHON" ]] || { printf 'activate_csgo_seen10: missing Python: %s\n' "$PUFFIN_PYTHON" >&2; return 2; }
    asset_exports="$("$PUFFIN_PYTHON" -s "$script_dir/download_csgo_seen10_assets.py" --profile "$profile" --print-env)" || return
    eval "$asset_exports"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'activate_csgo_seen10: source this script in your shell\n' >&2
    exit 2
fi
puffin_activate_csgo_seen10 "$@"
