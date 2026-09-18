#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${PROJECT_ROOT}/.venv"
SOURCE_ENV="${RDT_CLONE_SOURCE:-/home/jiahao/task/ControlAR/.venv}"
REQUIREMENTS="${PROJECT_ROOT}/requirements_csgo.txt"

if ! command -v conda >/dev/null 2>&1; then
    echo "setup_csgo_seen10: conda is required to clone ${SOURCE_ENV}" >&2
    exit 2
fi
if [[ ! -x "${SOURCE_ENV}/bin/python" ]]; then
    echo "setup_csgo_seen10: clone source is unavailable: ${SOURCE_ENV}" >&2
    exit 2
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "setup_csgo_seen10: cloning ${SOURCE_ENV} -> ${ENV_DIR}" >&2
    conda create --prefix "${ENV_DIR}" --clone "${SOURCE_ENV}" -y
fi

PYTHON="${ENV_DIR}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "setup_csgo_seen10: environment Python was not created: ${PYTHON}" >&2
    exit 2
fi

# Keep the requirements file auditable while avoiding reinstalling packages
# already present in the cloned environment.
mapfile -t MISSING < <("${PYTHON}" - "${REQUIREMENTS}" <<'PY'
import importlib.util
import pathlib
import sys

requirements = pathlib.Path(sys.argv[1])
import_for_distribution = {
    "sentencepiece": "sentencepiece",
    "h5py": "h5py",
    "imgaug": "imgaug",
}
missing = []
for raw in requirements.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    distribution = line.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0]
    module = import_for_distribution.get(distribution.lower(), distribution.replace("-", "_"))
    if importlib.util.find_spec(module) is None:
        missing.append(line)
print("\n".join(missing))
PY
)

if (( ${#MISSING[@]} )); then
    echo "setup_csgo_seen10: installing missing packages: ${MISSING[*]}" >&2
    "${PYTHON}" -m pip install --disable-pip-version-check --no-input "${MISSING[@]}"
else
    echo "setup_csgo_seen10: all extra CSGO dependencies are already installed" >&2
fi

"${PYTHON}" - <<'PY'
import importlib.metadata as metadata
import importlib.util
import sys

required = ("torch", "transformers", "diffusers", "huggingface_hub", "accelerate", "timm", "numpy")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing core imports: {missing}")
print("python", sys.version.replace("\n", " "))
for distribution in ("torch", "transformers", "diffusers", "huggingface-hub", "accelerate", "timm", "numpy", "sentencepiece", "h5py", "imgaug"):
    try:
        print(f"{distribution} {metadata.version(distribution)}")
    except metadata.PackageNotFoundError:
        print(f"{distribution} MISSING")
import torch
print("torch.cuda.is_available", torch.cuda.is_available())
print("torch.cuda.device_count", torch.cuda.device_count())
PY

echo "setup_csgo_seen10: environment=${PYTHON}" >&2
