"""Exercise environment setup decisions without creating environments or installing wheels."""

from __future__ import annotations

import os
import contextlib
import io
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _staged_setup(tmp_path: Path) -> Path:
    root = tmp_path / "rdt"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(SOURCE_ROOT / "scripts/setup_csgo_seen10.sh", root / "scripts/setup_csgo_seen10.sh")
    shutil.copy2(SOURCE_ROOT / "requirements_csgo.txt", root / "requirements_csgo.txt")
    return root


def _run(root: Path, *args: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    for key in ("RDT_SETUP_PYTHON", "RDT_CLONE_SOURCE", "RDT_TORCH_BACKEND", "RDT_TORCH_INDEX_URL"):
        environment.pop(key, None)
    environment.update(overrides)
    return subprocess.run(
        ["bash", str(root / "scripts/setup_csgo_seen10.sh"), *args],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


class SetupScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = _staged_setup(Path(self.tmp.name))

    def _missing_dependencies(self, finder) -> str:
        source = (self.root / 'scripts/setup_csgo_seen10.sh').read_text()
        probe = re.search(r"mapfile -t MISSING < <.*?<<'PY'\n(.*?)\nPY", source, re.S).group(1)
        output = io.StringIO()
        with patch('sys.argv', ['-', str(self.root / 'requirements_csgo.txt')]), \
                patch('importlib.util.find_spec', side_effect=finder), contextlib.redirect_stdout(output):
            exec(compile(probe, 'setup-missing-dependency-probe', 'exec'), {})
        return output.getvalue()

    def test_missing_google_namespace_installs_protobuf(self) -> None:
        def finder(module):
            if module == 'google.protobuf':
                raise ModuleNotFoundError("No module named 'google'")
            return object()
        self.assertEqual(self._missing_dependencies(finder), 'protobuf==6.33.4\n')

    def test_missing_protobuf_under_existing_google_namespace(self) -> None:
        self.assertEqual(self._missing_dependencies(
            lambda module: None if module == 'google.protobuf' else object()), 'protobuf==6.33.4\n')

    def test_existing_google_protobuf_is_not_reinstalled(self) -> None:
        self.assertEqual(self._missing_dependencies(lambda module: object()), '')

    def test_existing_environment_ignores_absent_clone_and_backend(self) -> None:
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(SOURCE_ROOT / ".venv/bin/python")
        result = _run(
            self.root,
            "--dry-run",
            RDT_CLONE_SOURCE="/no/such/ControlAR/.venv",
            RDT_TORCH_BACKEND="invalid",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reusing environment=", result.stderr)
        self.assertNotIn("would clone", result.stderr)

    def test_fresh_venv_and_backend_pair_are_reported(self) -> None:
        result = _run(
            self.root,
            "--dry-run",
            RDT_SETUP_PYTHON=str(SOURCE_ROOT / ".venv/bin/python"),
            RDT_TORCH_BACKEND="cu118",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=venv torch=2.6.0/0.21.0 backend=cu118", result.stderr)
        self.assertIn("would run ", result.stderr)
        self.assertIn("/whl/cu118", result.stderr)
        self.assertFalse((self.root / ".venv").exists())

    def test_cuda_13_driver_selects_supported_cuda_12_8_wheels(self) -> None:
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        nvidia_smi = fake_bin / "nvidia-smi"
        nvidia_smi.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' 'NVIDIA-SMI 580.125.09    Driver Version: 580.125.09    CUDA Version: 13.0'\n",
            encoding="utf-8",
        )
        nvidia_smi.chmod(0o755)
        result = _run(
            self.root,
            "--dry-run",
            RDT_SETUP_PYTHON=str(SOURCE_ROOT / ".venv/bin/python"),
            PATH=f"{fake_bin}:{os.environ['PATH']}",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("backend=cu128", result.stderr)
        self.assertIn("torch=2.8.0/0.23.0", result.stderr)

    def test_missing_explicit_clone_is_actionable(self) -> None:
        result = _run(self.root, "--dry-run", RDT_CLONE_SOURCE="/no/such/ControlAR/.venv")
        self.assertEqual(result.returncode, 2)
        self.assertIn("RDT_CLONE_SOURCE has no executable bin/python", result.stderr)

    def test_missing_python311_falls_back_to_conda_without_clone(self) -> None:
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        for name, status in (("python3.11", 1), ("python3", 1), ("conda", 0)):
            executable = fake_bin / name
            executable.write_text(f"#!/usr/bin/env bash\nexit {status}\n")
            executable.chmod(0o755)
        result = _run(self.root, "--dry-run", RDT_TORCH_BACKEND="cpu",
                      PATH=f"{fake_bin}:{os.environ['PATH']}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=conda", result.stderr)
        self.assertIn("would create conda Python 3.11", result.stderr)
        self.assertNotIn("would clone", result.stderr)
        self.assertFalse((self.root / ".venv").exists())

    def test_check_does_not_create_an_environment(self) -> None:
        result = _run(self.root, "--check")
        self.assertEqual(result.returncode, 2)
        self.assertIn("environment is missing", result.stderr)
        self.assertFalse((self.root / ".venv").exists())

    def test_fresh_setup_installs_all_pins_after_torch(self) -> None:
        # The fake interpreter records installation requests. It never invokes
        # pip, so this catches the fresh NumPy/imgaug ordering without downloads.
        log = self.root / "pip.log"
        fake_python = self.root / "fake-python3.11"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "set -e\n"
            "if [[ $1 == -c ]]; then exit 0; fi\n"
            "if [[ $1 == -m && $2 == venv ]]; then\n"
            "  mkdir -p \"$3/bin\"\n"
            "  cp \"$0\" \"$3/bin/python\"\n"
            "  exit 0\n"
            "fi\n"
            "if [[ $1 == -m && $2 == pip ]]; then\n"
            "  printf '%s\\n' \"$*\" >> \"$TEST_SETUP_LOG\"\n"
            "  exit 0\n"
            "fi\n"
            "if [[ $1 == - ]]; then\n"
            "  code=$(cat)\n"
            "  if [[ $code == *'for name in (\"torch\", \"torchvision\")'* ]]; then\n"
            "    printf 'torch\\ntorchvision\\n'\n"
            "  fi\n"
            "  exit 0\n"
            "fi\n"
            "exit 8\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        result = _run(
            self.root,
            RDT_SETUP_PYTHON=str(fake_python),
            RDT_TORCH_BACKEND="cpu",
            TEST_SETUP_LOG=str(log),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        installs = log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(installs), 2)
        self.assertIn("torch==2.8.0 torchvision==0.23.0", installs[0])
        self.assertIn(f"-r {self.root / 'requirements_csgo.txt'}", installs[1])
        self.assertIn("numpy==1.26.4", (self.root / "requirements_csgo.txt").read_text(encoding="utf-8"))
        marker = self.root / ".venv/.csgo_bootstrap_pending"
        self.assertFalse(marker.exists())

        # A process killed after environment creation leaves this marker. The
        # next invocation must apply all pins again, not a missing-only probe.
        marker.touch()
        retried = _run(
            self.root,
            RDT_TORCH_BACKEND="cpu",
            TEST_SETUP_LOG=str(log),
        )
        self.assertEqual(retried.returncode, 0, retried.stderr)
        self.assertIn("mode=recovery", retried.stderr)
        self.assertIn(f"-r {self.root / 'requirements_csgo.txt'}", log.read_text(encoding="utf-8").splitlines()[-1])
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
