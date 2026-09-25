"""Exercise OpenCV wheel ownership during CSGO setup without pip or downloads."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class OpenCVSetupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "rdt"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / ".venv/bin").mkdir(parents=True)
        shutil.copy2(SOURCE_ROOT / "scripts/setup_csgo_seen10.sh", self.root / "scripts/setup_csgo_seen10.sh")
        shutil.copy2(SOURCE_ROOT / "requirements_csgo.txt", self.root / "requirements_csgo.txt")
        self.state = self.root / "opencv-state"
        self.log = self.root / "pip.log"
        fake_python = self.root / ".venv/bin/python"
        fake_python.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ $1 == -c ]]; then exit 0; fi
if [[ $1 == -m && $2 == pip ]]; then
  printf '%s\\n' "$*" >> "$TEST_PIP_LOG"
  if [[ "${TEST_PIP_FAIL:-}" == "$3" ]]; then exit 17; fi
  if [[ $3 == uninstall ]]; then printf 'none\\n' > "$TEST_OPENCV_STATE"; fi
  if [[ $3 == install && " $* " == *' --no-deps '* ]]; then
    printf 'headless\\n' > "$TEST_OPENCV_STATE"
  fi
  exit 0
fi
if [[ $1 == - ]]; then
  code=$(cat)
  if [[ $code == *'requirements_csgo.txt must pin exactly one'* ]]; then
    printf 'opencv-python-headless==4.11.0.86\\n'
  elif [[ $code == *'for name in ("opencv-python", "opencv-python-headless"'* ]]; then
    case $(cat "$TEST_OPENCV_STATE") in
      both) printf 'opencv-python\\nopencv-python-headless\\n' ;;
      gui) printf 'opencv-python\\n' ;;
      headless|broken) printf 'opencv-python-headless\\n' ;;
    esac
  elif [[ $code == *'if metadata.version("opencv-python-headless") != sys.argv[1]'* ]]; then
    [[ $(cat "$TEST_OPENCV_STATE") == headless ]]
  fi
  exit 0
fi
exit 8
""",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

    def run_setup(self, *arguments: str, pip_fail: str = "") -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment.update(TEST_OPENCV_STATE=str(self.state), TEST_PIP_LOG=str(self.log),
                           TEST_PIP_FAIL=pip_fail)
        for key in ("RDT_CLONE_SOURCE", "RDT_SETUP_PYTHON", "RDT_TORCH_BACKEND", "RDT_TORCH_INDEX_URL"):
            environment.pop(key, None)
        return subprocess.run(
            ["bash", str(self.root / "scripts/setup_csgo_seen10.sh"), *arguments],
            cwd=self.root, env=environment, capture_output=True, text=True, check=False,
        )

    def test_check_reports_gui_collision_without_changing_environment(self) -> None:
        self.state.write_text("both\n", encoding="utf-8")
        result = self.run_setup("--check")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("OpenCV is not a healthy, sole", result.stderr)
        self.assertIn("opencv-python opencv-python-headless", result.stderr)
        self.assertEqual(self.state.read_text(encoding="utf-8"), "both\n")
        self.assertFalse(self.log.exists())

    def test_setup_repairs_collision_and_subsequent_runs_are_idempotent(self) -> None:
        self.state.write_text("both\n", encoding="utf-8")
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 2)
        self.assertIn("uninstall", calls[0])
        self.assertIn("opencv-python opencv-python-headless", calls[0])
        self.assertIn("install", calls[1])
        self.assertIn("--no-deps --force-reinstall opencv-python-headless==4.11.0.86", calls[1])
        self.assertEqual(self.state.read_text(encoding="utf-8"), "headless\n")
        self.assertEqual(self.run_setup("--check").returncode, 0)
        self.assertEqual(self.run_setup().returncode, 0)
        self.assertEqual(self.log.read_text(encoding="utf-8").splitlines(), calls)

    def test_setup_repairs_broken_headless_import(self) -> None:
        self.state.write_text("broken\n", encoding="utf-8")
        checked = self.run_setup("--check")
        self.assertEqual(checked.returncode, 2)
        self.assertIn("opencv-python-headless", checked.stderr)
        self.assertFalse(self.log.exists())
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.log.read_text(encoding="utf-8").splitlines()), 2)
        self.assertEqual(self.run_setup("--check").returncode, 0)

    def test_pip_failure_never_reports_ready(self) -> None:
        for failed_action in ("uninstall", "install"):
            with self.subTest(failed_action=failed_action):
                self.state.write_text("both\n", encoding="utf-8")
                self.log.unlink(missing_ok=True)
                result = self.run_setup(pip_fail=failed_action)
                self.assertEqual(result.returncode, 17, result.stderr)
                self.assertNotIn("setup_csgo_seen10: ready", result.stderr)
                self.assertNotEqual(self.run_setup("--check").returncode, 0)


if __name__ == "__main__":
    unittest.main()
