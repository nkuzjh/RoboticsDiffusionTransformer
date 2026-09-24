"""Machine-local paths for Seen-10 without rewriting experiment YAML files.

CLI > environment > YAML. Only the original machine-specific defaults are
relocated when unavailable; arbitrary user-configured paths are never replaced.
Relative paths are rooted at the RDT checkout, independent of the caller's cwd.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DATA = "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2"
LEGACY_EVAL = "/home/jiahao/task/csgo_benchmark_v2_eval_general"
LEGACY_PYTHON = "/home/jiahao/miniconda3/envs/UniLIP/bin/python"


def _path(value: object, root: Path) -> Path:
    candidate = Path(os.fspath(value)).expanduser()
    # Do not resolve Python symlinks: a venv interpreter must keep its venv path.
    return candidate if candidate.is_absolute() else root / candidate


def _resolve(config: Mapping, key: str, explicit: object, env_names: tuple[str, ...],
             legacy: str, fallback: Path, root: Path, env: Mapping[str, str]) -> Path:
    if explicit is not None:
        return _path(explicit, root)
    for name in env_names:
        if env.get(name):
            return _path(env[name], root)
    configured = config.get(key)
    if configured:
        candidate = _path(configured, root)
        if str(configured) != legacy or candidate.exists():
            return candidate
    return fallback


def data_root(config: Mapping, explicit: object = None, *, root: Path = PROJECT_ROOT,
              env: Mapping[str, str] | None = None) -> Path:
    return _resolve(config, "data_root", explicit, ("DATA_ROOT", "CSGO_DATA_ROOT"),
                    LEGACY_DATA, root.parent / "UniLIP/data/csgo_benchmark_v2", root,
                    os.environ if env is None else env)


def evaluator_root(config: Mapping, explicit: object = None, *, root: Path = PROJECT_ROOT,
                   env: Mapping[str, str] | None = None) -> Path:
    return _resolve(config, "shared_eval_dir", explicit, ("SHARED_EVAL_DIR", "CSGO_EVAL_ROOT"),
                    LEGACY_EVAL, root.parent / "csgo_benchmark_v2_eval_general", root,
                    os.environ if env is None else env)


def evaluator_python(config: Mapping, explicit: object = None, *, root: Path = PROJECT_ROOT,
                     env: Mapping[str, str] | None = None, python: object = None) -> Path:
    return _resolve(config, "unilip_python", explicit, ("UNILIP_PYTHON",),
                    LEGACY_PYTHON, _path(python, root) if python else root / ".venv/bin/python",
                    root, os.environ if env is None else env)
