#!/usr/bin/env python3
"""Pin Puffin Seen-10 assets; --check is offline, CPU-only, and read-only.

The aligned profile uses the official Puffin demo recipe: KangLiao Base,
wusize/Puffin VAE, and the author's SD3 scheduler/VAE configuration files.
The config blobs match the upstream SD3 repository byte for byte. This does
not assert that the wusize VAE equals upstream native SD3 weights.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("csgo_seen10_assets.json")


def anchored(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def cache_root(explicit: str | None = None, environ: dict | None = None) -> Path:
    env = os.environ if environ is None else environ
    if explicit:
        return anchored(explicit)
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(key):
            return anchored(env[key])
    if env.get("HF_HOME"):
        return anchored(env["HF_HOME"]) / "hub"
    return anchored(env.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface/hub"


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Puffin asset manifest schema")
    assets = payload["assets"]
    if len({asset["id"] for asset in assets}) != len(assets):
        raise ValueError("Duplicate Puffin asset identifiers")
    return assets


def selected_assets(assets: list[dict], profile: str) -> list[dict]:
    return [asset for asset in assets if not asset.get("reference_only") and
            (profile == "all" or profile in asset["profiles"])]


def snapshot_dir(root: Path, repo_id: str, revision: str) -> Path:
    return root / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision


def asset_path(asset: dict, root: Path, environ: dict | None = None) -> Path:
    env = os.environ if environ is None else environ
    if asset["id"] == "base":
        return anchored(env.get("PUFFIN_INIT_CHECKPOINT") or asset["local_path"])
    if asset["id"] == "puffin_vae":
        return anchored(env.get("PUFFIN_VAE_CHECKPOINT") or asset["local_path"])
    if asset["id"].startswith("qwen_") and env.get("PUFFIN_QWEN_PATH"):
        return anchored(env["PUFFIN_QWEN_PATH"]) / asset["filename"]
    if asset["id"].startswith("radio_") and env.get("PUFFIN_RADIO_PATH"):
        return anchored(env["PUFFIN_RADIO_PATH"]) / asset["filename"]
    if asset["id"] in ("sd3_scheduler", "sd3_vae_config") and env.get("PUFFIN_SD3_PATH"):
        return anchored(env["PUFFIN_SD3_PATH"]) / Path(*Path(asset["local_path"]).parts[-2:])
    if asset.get("local_path"):
        return anchored(asset["local_path"])
    return snapshot_dir(root, asset["repo_id"], asset["revision"]) / asset["filename"]


def check_file(path: Path, asset: dict) -> str:
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size != asset["size"]:
        return "size_mismatch"
    if asset["algorithm"] == "git-blob-sha1":
        digest = hashlib.sha1()
        digest.update(f"blob {size}\0".encode("ascii"))
    elif asset["algorithm"] == "sha256":
        digest = hashlib.sha256()
    else:
        raise ValueError(f"Unknown digest algorithm for {asset['id']}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "ok" if digest.hexdigest() == asset["digest"] else "hash_mismatch"


def audit_assets(assets: list[dict], root: Path, profile: str, environ: dict | None = None) -> list[dict]:
    results = []
    for asset in selected_assets(assets, profile):
        path = asset_path(asset, root, environ)
        results.append({"id": asset["id"], "path": str(path), "status": check_file(path, asset)})
    return results


def download_missing(assets: list[dict], root: Path, profile: str, results: list[dict]) -> None:
    by_id = {result["id"]: result for result in results}
    corrupt = [result for result in results if result["status"] not in ("ok", "missing")]
    if corrupt:
        raise RuntimeError("Existing assets failed integrity checks; preserved unchanged: " +
                           ", ".join(result["id"] for result in corrupt))
    if not any(result["status"] == "missing" for result in results):
        return
    from huggingface_hub import hf_hub_download

    for asset in selected_assets(assets, profile):
        if by_id[asset["id"]]["status"] != "missing":
            continue
        path = asset_path(asset, root)
        print(f"Fetching {asset['repo_id']}/{asset['filename']} at {asset['revision']}", file=sys.stderr, flush=True)
        kwargs = dict(repo_id=asset["repo_id"], revision=asset["revision"],
                      filename=asset["filename"], cache_dir=str(root))
        if asset.get("repo_type"):
            kwargs["repo_type"] = asset["repo_type"]
        try:
            downloaded = Path(hf_hub_download(**kwargs))
        except Exception as exc:
            raise RuntimeError(
                f"Fetching {asset['id']} failed ({type(exc).__name__}); "
                "check network or repository access. Existing files were preserved."
            ) from None
        source_state = check_file(downloaded, asset)
        if source_state != "ok":
            raise RuntimeError(f"Cached source for {asset['id']} failed integrity verification ({source_state})")
        default_snapshot = snapshot_dir(root, asset["repo_id"], asset["revision"]) / asset["filename"]
        if asset.get("local_path") or path != default_snapshot:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(downloaded.resolve(), path)
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    raise RuntimeError(f"Target appeared during download; preserved unchanged: {path}") from None
                if exc.errno != errno.EXDEV:
                    raise
                # A different filesystem needs a copy. Link the completed
                # temporary file into place without replacing a later writer.
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".puffin-asset-", delete=False) as handle:
                    temporary = Path(handle.name)
                try:
                    shutil.copyfile(downloaded, temporary)
                    os.link(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        elif downloaded.resolve() != path.resolve():
            raise RuntimeError(f"Unexpected downloaded path for {asset['id']}: {downloaded}")
        state = check_file(path, asset)
        if state != "ok":
            raise RuntimeError(f"Downloaded {asset['id']} failed integrity verification ({state}); preserved for inspection")
        by_id[asset["id"]]["status"] = "ok"


def environment_paths(assets: list[dict], root: Path, profile: str, environ: dict | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    by_id = {asset["id"]: asset for asset in assets}
    qwen = by_id["qwen_config"]
    radio = by_id["radio_config"]
    sd3_root = anchored(env.get("PUFFIN_SD3_PATH") or by_id["sd3_scheduler"]["local_path"])
    if not env.get("PUFFIN_SD3_PATH"):
        sd3_root = sd3_root.parents[1]
    return {
        "PUFFIN_INIT_CHECKPOINT": str(asset_path(by_id["base"], root, environ)),
        "PUFFIN_VAE_CHECKPOINT": str(asset_path(by_id["puffin_vae"], root, environ)),
        "PUFFIN_QWEN_PATH": str(anchored(env["PUFFIN_QWEN_PATH"])) if env.get("PUFFIN_QWEN_PATH") else str(snapshot_dir(root, qwen["repo_id"], qwen["revision"])),
        "PUFFIN_RADIO_PATH": str(anchored(env["PUFFIN_RADIO_PATH"])) if env.get("PUFFIN_RADIO_PATH") else str(snapshot_dir(root, radio["repo_id"], radio["revision"])),
        "PUFFIN_SD3_PATH": "" if profile == "legacy" else str(sd3_root),
    }


def print_environment(assets: list[dict], root: Path, profile: str, environ: dict | None = None) -> None:
    for name, value in environment_paths(assets, root, profile, environ).items():
        print(f"export {name}={shlex.quote(value)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Offline full hash verification; no writes or CUDA")
    mode.add_argument("--print-env", action="store_true", help="Print fixed paths for shell sourcing; does not verify files")
    parser.add_argument("--profile", choices=("aligned", "legacy", "all"), default="aligned")
    parser.add_argument("--cache-dir", help="Hub cache root; relative paths are anchored at Puffin checkout")
    parser.add_argument("--json", action="store_true", help="Machine-readable audit result")
    args = parser.parse_args(argv)
    assets = load_manifest()
    root = cache_root(args.cache_dir)
    if args.print_env:
        print_environment(assets, root, args.profile)
        return 0
    results = audit_assets(assets, root, args.profile)
    if not args.check:
        download_missing(assets, root, args.profile, results)
    ready = all(result["status"] == "ok" for result in results)
    if args.json:
        print(json.dumps({"profile": args.profile, "ready": ready, "files": results}, indent=2))
    else:
        for result in results:
            print(f"{result['status'].upper()}: {result['id']} {result['path']}")
        print(f"Puffin {args.profile} assets ready={ready}")
        if ready:
            print_environment(assets, root, args.profile)
    return 0 if ready else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
