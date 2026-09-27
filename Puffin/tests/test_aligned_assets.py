"""CPU-only tests for fixed-revision asset verification and safe reuse."""

from __future__ import annotations

import hashlib
import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import download_csgo_seen10_assets as assets


def _asset(name: str, payload: bytes, path: str, profiles: list[str]) -> dict:
    digest = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    return {
        "id": name,
        "repo_id": "test/asset",
        "revision": "a" * 40,
        "filename": path,
        "local_path": path,
        "size": len(payload),
        "algorithm": "git-blob-sha1",
        "digest": digest,
        "profiles": profiles,
    }


class AlignedAssetTests(unittest.TestCase):
    def test_check_file_detects_missing_size_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = _asset("test", b"expected", "item.bin", ["aligned"])
            path = Path(directory) / "item.bin"
            self.assertEqual(assets.check_file(path, asset), "missing")
            path.write_bytes(b"short")
            self.assertEqual(assets.check_file(path, asset), "size_mismatch")
            path.write_bytes(b"altered!")
            self.assertEqual(assets.check_file(path, asset), "hash_mismatch")
            path.write_bytes(b"expected")
            self.assertEqual(assets.check_file(path, asset), "ok")

    def test_profiles_and_audit_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_root = assets.PROJECT_ROOT
            assets.PROJECT_ROOT = root
            try:
                aligned = _asset("aligned", b"a", "aligned.bin", ["aligned"])
                legacy = _asset("legacy", b"l", "legacy.bin", ["legacy"])
                (root / "aligned.bin").write_bytes(b"a")
                selection = [aligned, legacy]
                before = sorted(path.name for path in root.iterdir())
                results = assets.audit_assets(selection, root / "cache", "aligned", environ={})
                self.assertEqual(results, [{"id": "aligned", "path": str(root / "aligned.bin"), "status": "ok"}])
                self.assertEqual([item["id"] for item in assets.selected_assets(selection, "all")], ["aligned", "legacy"])
                self.assertEqual(sorted(path.name for path in root.iterdir()), before)
            finally:
                assets.PROJECT_ROOT = original_root

    def test_corrupt_existing_file_is_preserved_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_root = assets.PROJECT_ROOT
            assets.PROJECT_ROOT = root
            try:
                asset = _asset("test", b"good", "item.bin", ["aligned"])
                path = root / "item.bin"
                path.write_bytes(b"bad!")
                results = assets.audit_assets([asset], root / "cache", "aligned", environ={})
                self.assertEqual(results[0]["status"], "hash_mismatch")
                with self.assertRaisesRegex(RuntimeError, "preserved unchanged"):
                    assets.download_missing([asset], root / "cache", "aligned", results)
                self.assertEqual(path.read_bytes(), b"bad!")
            finally:
                assets.PROJECT_ROOT = original_root

    def test_missing_file_is_linked_once_then_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_root = assets.PROJECT_ROOT
            assets.PROJECT_ROOT = root
            try:
                asset = _asset("test", b"verified", "item.bin", ["aligned"])
                source = root / "cached-source.bin"
                source.write_bytes(b"verified")
                results = assets.audit_assets([asset], root / "cache", "aligned", environ={})
                with patch("huggingface_hub.hf_hub_download", return_value=str(source)) as fetch:
                    assets.download_missing([asset], root / "cache", "aligned", results)
                    fetch.assert_called_once()
                self.assertEqual((root / "item.bin").read_bytes(), b"verified")
                reused = assets.audit_assets([asset], root / "cache", "aligned", environ={})
                with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("redownloaded")):
                    assets.download_missing([asset], root / "cache", "aligned", reused)
            finally:
                assets.PROJECT_ROOT = original_root

    def test_manifest_revisions_and_official_space_config_hashes(self):
        indexed = {item["id"]: item for item in assets.load_manifest()}
        self.assertEqual(indexed["base"]["revision"], "b56171f03e35a0be50b8d8b09dc4ccd09aa679ec")
        self.assertEqual(indexed["qwen_generation_config"]["revision"], "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
        self.assertEqual(indexed["qwen_generation_config"]["size"], 242)
        self.assertEqual(indexed["qwen_generation_config"]["digest"], "dfc11073787daf1b0f9c0f1499487ab5f4c93738")
        self.assertEqual(indexed["sd3_scheduler"]["digest"], "eca5b2eed3727b3ad375eb20e467e4264120fd5d")
        self.assertEqual(indexed["sd3_vae_config"]["digest"], "5cc5fb203426bfac5ecee2710115bb1697737211")

    def test_aligned_manifest_includes_qwen_resume_identity_files(self):
        selected = assets.selected_assets(assets.load_manifest(), "aligned")
        actual = {item["filename"] for item in selected if item["id"].startswith("qwen_")}
        self.assertEqual(actual, {
            "config.json", "generation_config.json", "tokenizer_config.json",
            "tokenizer.json", "vocab.json", "merges.txt",
        })

    def test_print_env_matches_override_paths(self):
        manifest = assets.load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overrides = {
                "PUFFIN_INIT_CHECKPOINT": str(root / "Base custom.pth"),
                "PUFFIN_VAE_CHECKPOINT": str(root / "VAE custom.pth"),
                "PUFFIN_QWEN_PATH": str(root / "qwen custom"),
                "PUFFIN_RADIO_PATH": str(root / "radio custom"),
                "PUFFIN_SD3_PATH": str(root / "sd3 custom"),
            }
            values = assets.environment_paths(manifest, root / "hub", "aligned", overrides)
            self.assertEqual(values["PUFFIN_INIT_CHECKPOINT"], overrides["PUFFIN_INIT_CHECKPOINT"])
            self.assertEqual(values["PUFFIN_VAE_CHECKPOINT"], overrides["PUFFIN_VAE_CHECKPOINT"])
            self.assertEqual(values["PUFFIN_QWEN_PATH"], overrides["PUFFIN_QWEN_PATH"])
            self.assertEqual(values["PUFFIN_RADIO_PATH"], overrides["PUFFIN_RADIO_PATH"])
            self.assertEqual(values["PUFFIN_SD3_PATH"], overrides["PUFFIN_SD3_PATH"])
            by_id = {item["id"]: item for item in manifest}
            self.assertEqual(str(assets.asset_path(by_id["base"], root / "hub", overrides)), values["PUFFIN_INIT_CHECKPOINT"])
            self.assertEqual(str(assets.asset_path(by_id["puffin_vae"], root / "hub", overrides)), values["PUFFIN_VAE_CHECKPOINT"])
            self.assertEqual(str(assets.asset_path(by_id["qwen_config"], root / "hub", overrides)), str(root / "qwen custom/config.json"))
            self.assertEqual(str(assets.asset_path(by_id["radio_config"], root / "hub", overrides)), str(root / "radio custom/config.json"))
            self.assertEqual(str(assets.asset_path(by_id["sd3_scheduler"], root / "hub", overrides)), str(root / "sd3 custom/scheduler/scheduler_config.json"))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                assets.print_environment(manifest, root / "hub", "aligned", overrides)
            self.assertIn("PUFFIN_INIT_CHECKPOINT=", stream.getvalue())
            self.assertIn("PUFFIN_VAE_CHECKPOINT=", stream.getvalue())

    def test_check_with_missing_fixture_does_not_fetch_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _asset("test", b"expected", "item.bin", ["aligned"])
            original_root = assets.PROJECT_ROOT
            assets.PROJECT_ROOT = root
            try:
                with patch.object(assets, "load_manifest", return_value=[fixture]), \
                     patch.object(assets, "cache_root", return_value=root / "hub"), \
                     patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network called")):
                    stream = io.StringIO()
                    with contextlib.redirect_stdout(stream):
                        code = assets.main(["--check", "--profile", "aligned", "--json"])
                self.assertEqual(code, 1)
                self.assertIn('"status": "missing"', stream.getvalue())
                self.assertEqual(list(root.iterdir()), [])
            finally:
                assets.PROJECT_ROOT = original_root

    def test_setup_check_missing_python_does_not_create_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-env" / "bin" / "python"
            result = subprocess.run(
                ["bash", str(assets.PROJECT_ROOT / "scripts/setup_csgo_seen10.sh"), "--check", "--env-only"],
                env={**os.environ, "PUFFIN_PYTHON": str(path)},
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("missing Python", result.stderr)
            self.assertFalse(path.parent.parent.exists())


if __name__ == "__main__":
    unittest.main()
