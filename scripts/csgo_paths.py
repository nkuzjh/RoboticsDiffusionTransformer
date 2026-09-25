"""Machine-local paths for Seen-10 without rewriting experiment YAML files.

CLI > environment > YAML. Only original data/evaluator directory defaults are
relocated when unavailable; explicit Python paths are never replaced.
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


def run_directories(config: Mapping, *, seed: int, smoke: bool = False,
                    output: object = None, checkpoint: object = None,
                    root: Path = PROJECT_ROOT) -> tuple[Path, Path]:
    """Resolve both entrypoints and the shell wrapper with the same layout.

    Explicit null checkpoint_root opts into <output>/checkpoints. Old YAML
    without this key keeps its original default, including for strict resume.
    """
    parts = [str(config.get("model_name", "RDT"))]
    if smoke:
        parts.append("smoke")
    parts.append(f"seed_{seed}")
    artifacts = (_path(output, root) if output is not None else
                 _path(config.get("output_root", "outputs/csgo_benchmark_v2_seen10"), root).joinpath(*parts))
    checkpoint_root = config.get("checkpoint_root", "checkpoints/csgo_benchmark_v2_seen10")
    if checkpoint is not None:
        checkpoints = _path(checkpoint, root)
    elif checkpoint_root is None:
        checkpoints = artifacts / "checkpoints"
    else:
        checkpoints = _path(checkpoint_root, root).joinpath(*parts)
    return artifacts, checkpoints


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
                     env: Mapping[str, str] | None = None, eval_root: object = None,
                     python: object = None) -> Path:
    """Select the shared evaluator's interpreter without probing or repairing it.

    ``python`` is a deprecated compatibility argument; the RDT interpreter is
    never an implicit evaluator fallback. Keep virtualenv symlinks intact.
    """
    environment = os.environ if env is None else env
    if explicit is not None:
        return _path(explicit, root)
    for name in ("CSGO_EVAL_PYTHON", "UNILIP_PYTHON"):
        if environment.get(name):
            return _path(environment[name], root)
    if config.get("unilip_python"):
        return _path(config["unilip_python"], root)
    selected_eval_root = (_path(eval_root, root) if eval_root is not None else
                          evaluator_root(config, root=root, env=environment))
    return selected_eval_root / ".venv/bin/python"
