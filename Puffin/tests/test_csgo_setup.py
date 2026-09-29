"""Exercise installer control flow without package installation or network."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


class SetupContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.scripts = self.project / "scripts"
        self.scripts.mkdir(parents=True)
        for name in ("setup_csgo_seen10.sh", "puffin_environment.sh"):
            shutil.copyfile(PROJECT / "scripts" / name, self.scripts / name)
        for name in ("check_csgo_environment.py", "download_csgo_seen10_assets.py"):
            (self.scripts / name).touch()
        (self.project / "requirements_seen10.txt").write_text("numpy==2.2.5\n")
        self.prefix = self.project / ".venv"
        self.log = self.root / "calls.jsonl"
        self.python_template = self.root / "python-template"
        self.python_template.write_text(f"#!{sys.executable} -s\n" + """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
record = {"tool": "python", "args": args, "env": dict(os.environ)}
with Path(os.environ["TEST_SETUP_LOG"]).open("a") as handle:
    handle.write(json.dumps(record) + "\\n")
if os.environ.get("TEST_CHECK_FAIL") == "1" and any(a.endswith("check_csgo_environment.py") for a in args) and "--identity-only" not in args:
    raise SystemExit(1)
""")
        self.python_template.chmod(0o755)
        self.conda = self.root / "conda"
        self.conda.write_text(f"#!{sys.executable} -s\n" + """
import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ["TEST_SETUP_LOG"]).open("a") as handle:
    handle.write(json.dumps({"tool": "conda", "args": args, "env": dict(os.environ)}) + "\\n")
if args == ["info", "--base"]:
    print(os.environ["TEST_CONDA_BASE"])
elif args[0] == "create":
    prefix = Path(args[args.index("--prefix") + 1])
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "bin").mkdir()
    shutil.copyfile(os.environ["TEST_PYTHON_TEMPLATE"], prefix / "bin/python")
    (prefix / "bin/python").chmod(0o755)
elif args[0] != "install":
    raise SystemExit("Unexpected Conda action")
""")
        self.conda.chmod(0o755)
        self.environment = {
            **os.environ,
            "PUFFIN_CONDA_EXE": str(self.conda),
            "PUFFIN_PYTHON": str(self.prefix / "bin/python"),
            "TEST_SETUP_LOG": str(self.log),
            "TEST_CONDA_BASE": str(self.root / "base"),
            "TEST_PYTHON_TEMPLATE": str(self.python_template),
            "PIP_TARGET": str(self.root / "foreign-install"),
            "PIP_USER": "1",
            "PIP_PREFIX": str(self.root / "foreign-prefix"),
            "PYTHONPATH": str(self.root / "foreign-packages"),
            "PYTHONHOME": str(self.root / "foreign-python"),
            "LD_LIBRARY_PATH": str(self.root / "foreign-libraries"),
        }

    def existing(self, conda=True):
        (self.prefix / "bin").mkdir(parents=True)
        shutil.copyfile(self.python_template, self.prefix / "bin/python")
        (self.prefix / "bin/python").chmod(0o755)
        if conda:
            (self.prefix / "conda-meta").mkdir()
        else:
            (self.prefix / "pyvenv.cfg").write_text("include-system-site-packages = false\n")

    def run_setup(self, *arguments):
        return subprocess.run(["bash", str(self.scripts / "setup_csgo_seen10.sh"),
                               "--env-only", *arguments],
                              env=self.environment, text=True, capture_output=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def mutations(self):
        return [call for call in self.calls() if (
            call["tool"] == "conda" and call["args"][0] in ("create", "install")
            or call["tool"] == "python" and "install" in call["args"])]

    def test_fresh_prefix_installs_only_into_selected_environment(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        create = next(call for call in calls if call["tool"] == "conda" and call["args"][0] == "create")
        for argument in ("libgl", "libglib", "--override-channels", "--no-default-packages"):
            self.assertIn(argument, create["args"])
        self.assertNotIn("LD_LIBRARY_PATH", create["env"])
        self.assertEqual(create["env"]["PYTHONNOUSERSITE"], "1")
        installs = [call for call in calls if call["tool"] == "python" and "install" in call["args"]]
        self.assertEqual(len(installs), 2)
        for call in installs:
            self.assertEqual(call["env"]["PYTHONNOUSERSITE"], "1")
            self.assertEqual(call["env"]["PIP_USER"], "0")
            self.assertEqual(call["env"]["PIP_CONFIG_FILE"], "/dev/null")
            for name in ("PIP_TARGET", "PIP_PREFIX", "PYTHONPATH", "PYTHONHOME"):
                self.assertNotIn(name, call["env"])
        self.assertTrue((self.prefix / ".puffin-conda-runtime").is_file())
        self.assertFalse((self.root / "foreign-install").exists())

    def test_existing_unmarked_conda_is_not_modified(self):
        self.existing()
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.mutations(), [])
        self.assertFalse((self.prefix / ".puffin-conda-runtime").exists())

    def test_existing_venv_is_not_modified(self):
        self.existing(conda=False)
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.mutations(), [])

    def test_broken_existing_prefix_is_preserved_without_repair(self):
        self.existing()
        self.environment["TEST_CHECK_FAIL"] = "1"
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.mutations(), [])
        self.assertIn("--repair", result.stderr)

    def test_explicit_repair_installs_native_and_python_dependencies(self):
        self.existing()
        result = self.run_setup("--repair")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.mutations()[0]["args"][0], "install")
        self.assertTrue((self.prefix / ".puffin-conda-runtime").is_file())

    def test_repair_venv_is_rejected_without_writes(self):
        self.existing(conda=False)
        result = self.run_setup("--repair")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.mutations(), [])
        self.assertIn("Conda prefix", result.stderr)

    def test_conda_base_is_never_a_target(self):
        self.environment["TEST_CONDA_BASE"] = str(self.prefix)
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Conda base", result.stderr)
        self.assertEqual(self.mutations(), [])
        self.assertFalse(self.prefix.exists())

    def test_check_missing_prefix_does_not_create_it(self):
        result = self.run_setup("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.mutations(), [])
        self.assertFalse(self.prefix.exists())

    def test_active_conda_prefix_cannot_be_repaired(self):
        self.existing()
        self.environment["CONDA_PREFIX"] = str(self.prefix)
        result = self.run_setup("--repair")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("deactivate", result.stderr)
        self.assertEqual(self.mutations(), [])

    def test_failed_install_does_not_mark_prefix_ready(self):
        self.environment["TEST_CHECK_FAIL"] = "1"
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.prefix.exists())
        self.assertFalse((self.prefix / ".puffin-conda-runtime").exists())


if __name__ == "__main__":
    unittest.main()
