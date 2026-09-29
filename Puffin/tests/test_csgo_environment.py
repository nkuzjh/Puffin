"""No-GPU checks for project-only Python packages and native libraries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import check_csgo_environment as checker


PROJECT = Path(__file__).resolve().parents[1]


class EnvironmentIsolationTests(unittest.TestCase):
    def test_user_site_and_injected_python_path_are_rejected(self):
        with patch.object(checker.site, "ENABLE_USER_SITE", True):
            with self.assertRaisesRegex(SystemExit, "User site"):
                checker.check_isolation(Path(sys.prefix))
        with patch.object(checker.site, "ENABLE_USER_SITE", False), \
                patch.dict(os.environ, {"PYTHONPATH": "/another/environment"}):
            with self.assertRaisesRegex(SystemExit, "PYTHONPATH"):
                checker.check_isolation(Path(sys.prefix))

    def test_foreign_site_packages_are_rejected(self):
        prefix = Path("/tmp/puffin-test-prefix")
        with patch.object(checker.site, "ENABLE_USER_SITE", False), \
                patch.dict(os.environ, {}, clear=True), \
                patch.object(sys, "path", [str(prefix / "lib/python3.10/site-packages"),
                                          "/another/env/lib/python3.10/site-packages"]):
            with self.assertRaisesRegex(SystemExit, "External package search path"):
                checker.check_isolation(prefix)

    def test_project_code_and_local_packages_are_allowed(self):
        prefix = Path("/tmp/puffin-test-prefix")
        with patch.object(checker.site, "ENABLE_USER_SITE", False), \
                patch.dict(os.environ, {}, clear=True), \
                patch.object(sys, "path", [str(PROJECT), str(prefix / "lib/python3.10/site-packages")]):
            checker.check_isolation(prefix)

    def test_required_module_must_be_installed_in_selected_prefix(self):
        prefix = Path("/tmp/puffin-test-prefix")
        local = SimpleNamespace(__name__="cv2", __file__=str(prefix / "lib/python3.10/site-packages/cv2/__init__.py"))
        checker.check_module_origin(local, prefix)
        external = SimpleNamespace(__name__="cv2", __file__="/another/env/lib/python3.10/site-packages/cv2/__init__.py")
        with self.assertRaisesRegex(RuntimeError, "outside"):
            checker.check_module_origin(external, prefix)

    def test_native_libraries_cannot_be_satisfied_by_system_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "lib").mkdir()
            names = ("libGL.so.1", "libglib-2.0.so.0", "libgthread-2.0.so.0")
            for name in names:
                (prefix / "lib" / name).touch()
            with patch.object(checker.ctypes, "CDLL"), \
                    patch.object(Path, "read_text", return_value="\n".join(
                        f"000-fff r-xp 0000 00:00 0 /usr/lib/{name}" for name in names)):
                with self.assertRaisesRegex(SystemExit, "exclusively inside"):
                    checker.check_native_libraries(prefix)

    def test_private_native_libraries_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "lib").mkdir()
            names = ("libGL.so.1", "libglib-2.0.so.0", "libgthread-2.0.so.0")
            for name in names:
                (prefix / "lib" / name).touch()
            with patch.object(checker.ctypes, "CDLL") as loader, \
                    patch.object(Path, "read_text", return_value="\n".join(
                        f"000-fff r-xp 0000 00:00 0 {prefix}/lib/{name}" for name in names)):
                checker.check_native_libraries(prefix)
                self.assertEqual(loader.call_count, 3)

    def test_missing_native_library_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(checker.ctypes, "CDLL") as loader:
            with self.assertRaisesRegex(SystemExit, "Missing project runtime library"):
                checker.check_native_libraries(Path(directory))
            loader.assert_not_called()

    def test_runtime_wrapper_cleans_python_and_foreign_libraries(self):
        code = """
source scripts/puffin_environment.sh
puffin_use_environment "$TEST_PUFFIN_PYTHON"
"$PUFFIN_PYTHON" -c 'import json, os, site; print(json.dumps({"user_site": site.ENABLE_USER_SITE, "pythonpath": os.environ.get("PYTHONPATH"), "pythonhome": os.environ.get("PYTHONHOME"), "library_path": os.environ.get("LD_LIBRARY_PATH"), "path": os.environ["PATH"]}))'
"""
        environment = dict(os.environ)
        environment.update({"TEST_PUFFIN_PYTHON": str(PROJECT / ".venv/bin/python"),
                            "PYTHONPATH": "/foreign/env/lib/python3.10/site-packages",
                            "PYTHONHOME": "/foreign/env", "LD_LIBRARY_PATH": "/foreign/env/lib"})
        environment.pop("PUFFIN_DRIVER_LIBRARY_PATH", None)
        result = subprocess.run(["bash", "-c", code], cwd=PROJECT, env=environment,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertFalse(record["user_site"])
        self.assertFalse(record["pythonpath"])
        self.assertFalse(record["pythonhome"])
        self.assertEqual(record["library_path"], str(PROJECT / ".venv/lib"))
        self.assertNotIn("/foreign/env", record["path"])


if __name__ == "__main__":
    unittest.main()
