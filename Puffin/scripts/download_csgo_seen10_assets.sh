#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/puffin_environment.sh"
puffin_use_environment
[[ -x "$PUFFIN_PYTHON" ]] || { printf 'download_csgo_seen10_assets: missing Python: %s\n' "$PUFFIN_PYTHON" >&2; exit 2; }
exec "$PUFFIN_PYTHON" -s "$SCRIPT_DIR/download_csgo_seen10_assets.py" "$@"
