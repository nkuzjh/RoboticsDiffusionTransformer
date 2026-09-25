#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${PROJECT_ROOT}/.venv"
REQUIREMENTS="${PROJECT_ROOT}/requirements_csgo.txt"
BOOTSTRAP_MARKER="${ENV_DIR}/.csgo_bootstrap_pending"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_csgo_seen10.sh [--dry-run | --check]

Reuse the project's .venv if present; otherwise create a Python 3.11
environment and install CSGO training/inference dependencies. This script
does not download model weights or benchmark data.

Options:
  --dry-run           Show the planned environment and PyTorch backend.
  --check             Verify an existing .venv without changing it.

Optional environment variables:
  RDT_SETUP_PYTHON     Python 3.11 executable for creating a new venv.
  RDT_CLONE_SOURCE     Existing conda environment to clone when .venv is absent.
  RDT_TORCH_BACKEND    auto (default), cpu, cu118, cu124, cu126, or cu128.
  RDT_TORCH_INDEX_URL  Mirror of the matching official PyTorch wheel index.

Auto selects wheels from the CUDA version reported by nvidia-smi. If no
CUDA version is visible, it selects CPU wheels. On GPU login nodes without
nvidia-smi, set RDT_TORCH_BACKEND explicitly before setup.
EOF
}

die() { echo "setup_csgo_seen10: $*" >&2; exit 2; }

MODE=setup
if (( $# > 1 )); then usage >&2; exit 2; fi
if (( $# == 1 )); then
    case "$1" in
        --dry-run) MODE=dry-run ;;
        --check|--verify-only) MODE=check ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
fi
[[ -f "${REQUIREMENTS}" ]] || die "requirements file is missing: ${REQUIREMENTS}"

select_python() {
    local candidate
    if [[ -n "${RDT_SETUP_PYTHON:-}" ]]; then
        command -v "${RDT_SETUP_PYTHON}" >/dev/null 2>&1 \
            || die "RDT_SETUP_PYTHON is unavailable: ${RDT_SETUP_PYTHON}"
        candidate="${RDT_SETUP_PYTHON}"
        "${candidate}" -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>/dev/null \
            || die "RDT_SETUP_PYTHON must be Python 3.11: ${candidate}"
        "${candidate}" -c 'import ensurepip, venv' 2>/dev/null \
            || die "RDT_SETUP_PYTHON lacks venv/ensurepip; install python3.11-venv or use conda"
        printf '%s\n' "${candidate}"
        return
    fi
    for candidate in python3.11 python3; do
        if command -v "${candidate}" >/dev/null 2>&1 && \
            "${candidate}" -c 'import sys, ensurepip, venv; assert sys.version_info[:2] == (3, 11)' 2>/dev/null; then
            printf '%s\n' "${candidate}"
            return
        fi
    done
    if command -v conda >/dev/null 2>&1; then printf '%s\n' conda; return; fi
    die "Python 3.11 is unavailable; install it or conda, or set RDT_SETUP_PYTHON"
}

select_torch_backend() {
    local detected major minor
    TORCH_BACKEND="${RDT_TORCH_BACKEND:-auto}"
    if [[ "${TORCH_BACKEND}" == auto ]]; then
        detected=""
        if command -v nvidia-smi >/dev/null 2>&1; then
            detected="$(nvidia-smi 2>/dev/null || true)"
        fi
        if [[ "${detected}" =~ CUDA[[:space:]]+Version:[[:space:]]*([0-9]+)\.([0-9]+) ]]; then
            major="${BASH_REMATCH[1]}"; minor="${BASH_REMATCH[2]}"
            if (( major > 12 || (major == 12 && minor >= 8) )); then
                TORCH_BACKEND=cu128
            elif (( major == 12 && minor >= 6 )); then
                TORCH_BACKEND=cu126
            elif (( major == 12 && minor >= 4 )); then
                TORCH_BACKEND=cu124
            elif (( major > 11 || (major == 11 && minor >= 8) )); then
                TORCH_BACKEND=cu118
            else
                die "nvidia-smi reports CUDA ${major}.${minor}; use a prepared .venv or choose a supported RDT_TORCH_BACKEND"
            fi
        else
            TORCH_BACKEND=cpu
            echo "setup_csgo_seen10: nvidia-smi did not report CUDA; selecting CPU wheels" >&2
        fi
    fi
    case "${TORCH_BACKEND}" in
        cpu|cu128|cu126) TORCH_VERSION=2.8.0; TORCHVISION_VERSION=0.23.0 ;;
        cu124) TORCH_VERSION=2.6.0; TORCHVISION_VERSION=0.21.0 ;;
        cu118) TORCH_VERSION=2.6.0; TORCHVISION_VERSION=0.21.0 ;;
        *) die "unsupported RDT_TORCH_BACKEND=${TORCH_BACKEND}; expected auto, cpu, cu118, cu124, cu126, or cu128" ;;
    esac
    TORCH_INDEX_URL="${RDT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_BACKEND}}"
}

ENV_KIND=existing
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    [[ "${MODE}" != check ]] || die "environment is missing: ${ENV_DIR}; run setup first"
    [[ ! -e "${ENV_DIR}" ]] || die "${ENV_DIR} exists without bin/python; repair or remove that incomplete environment manually"
    if [[ -n "${RDT_CLONE_SOURCE:-}" ]]; then
        [[ -x "${RDT_CLONE_SOURCE}/bin/python" ]] \
            || die "RDT_CLONE_SOURCE has no executable bin/python: ${RDT_CLONE_SOURCE}"
        command -v conda >/dev/null 2>&1 \
            || die "conda is required to clone RDT_CLONE_SOURCE=${RDT_CLONE_SOURCE}"
        ENV_KIND=clone
    else
        ENV_PYTHON="$(select_python)"
        if [[ "${ENV_PYTHON}" == conda ]]; then ENV_KIND=conda; else ENV_KIND=venv; fi
    fi
elif [[ -f "${BOOTSTRAP_MARKER}" ]]; then
    ENV_KIND=recovery
fi

if [[ "${ENV_KIND}" != existing && "${MODE}" != check ]]; then
    select_torch_backend
    echo "setup_csgo_seen10: environment=${ENV_DIR} mode=${ENV_KIND} torch=${TORCH_VERSION}/${TORCHVISION_VERSION} backend=${TORCH_BACKEND}" >&2
else
    echo "setup_csgo_seen10: reusing environment=${ENV_DIR}" >&2
fi
if [[ "${MODE}" == dry-run ]]; then
    case "${ENV_KIND}" in
        clone) echo "setup_csgo_seen10: would clone ${RDT_CLONE_SOURCE} into ${ENV_DIR}" >&2 ;;
        conda) echo "setup_csgo_seen10: would create conda Python 3.11 environment at ${ENV_DIR}" >&2 ;;
        venv) echo "setup_csgo_seen10: would run ${ENV_PYTHON} -m venv ${ENV_DIR}" >&2 ;;
        recovery) echo "setup_csgo_seen10: would resume a previously interrupted clean bootstrap" >&2 ;;
    esac
    if [[ "${ENV_KIND}" != existing ]]; then
        echo "setup_csgo_seen10: would use PyTorch index ${TORCH_INDEX_URL} and install packages from ${REQUIREMENTS}" >&2
    fi
    exit 0
fi

case "${ENV_KIND}" in
    clone) conda create --prefix "${ENV_DIR}" --clone "${RDT_CLONE_SOURCE}" -y ;;
    conda) conda create --prefix "${ENV_DIR}" python=3.11 pip -y ;;
    venv) "${ENV_PYTHON}" -m venv "${ENV_DIR}" \
        || die "venv creation failed; install python3.11-venv or use conda" ;;
esac

PYTHON="${ENV_DIR}/bin/python"
[[ -x "${PYTHON}" ]] || die "environment Python was not created: ${PYTHON}"
if [[ "${ENV_KIND}" == venv || "${ENV_KIND}" == conda ]]; then
    touch "${BOOTSTRAP_MARKER}"
fi
"${PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>/dev/null \
    || die "${PYTHON} must be Python 3.11 for this pinned CSGO setup"

if [[ "${MODE}" == setup ]]; then
    mapfile -t MISSING_TORCH < <("${PYTHON}" - <<'PY'
import importlib.util
for name in ("torch", "torchvision"):
    if importlib.util.find_spec(name) is None:
        print(name)
PY
)
    if (( ${#MISSING_TORCH[@]} == 2 )); then
        if [[ "${ENV_KIND}" == existing ]]; then select_torch_backend; fi
        echo "setup_csgo_seen10: installing PyTorch ${TORCH_VERSION}/${TORCHVISION_VERSION} from ${TORCH_INDEX_URL}" >&2
        "${PYTHON}" -m pip install --disable-pip-version-check --no-input \
            --index-url "${TORCH_INDEX_URL}" \
            "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
    elif (( ${#MISSING_TORCH[@]} == 1 )); then
        die "only ${MISSING_TORCH[0]} is missing; install a matching torch/torchvision pair in ${ENV_DIR}"
    fi

    if [[ "${ENV_KIND}" == venv || "${ENV_KIND}" == conda || "${ENV_KIND}" == recovery ]]; then
        # Torch's wheel dependencies can initially resolve NumPy 2.x. Apply
        # every CSGO pin afterwards so imgaug sees the required NumPy 1.26.
        "${PYTHON}" -m pip install --disable-pip-version-check --no-input \
            -r "${REQUIREMENTS}"
    else
        # Probe imports here to avoid reinstalling unrelated packages. OpenCV
        # ownership is checked separately after all pip installs: imgaug's
        # metadata can pull in the GUI wheel even when headless is present.
        mapfile -t MISSING < <("${PYTHON}" - "${REQUIREMENTS}" <<'PY'
import importlib.util
from pathlib import Path
import sys

modules = {"pillow": "PIL", "pyyaml": "yaml", "opencv-python-headless": "cv2", "huggingface-hub": "huggingface_hub", "protobuf": "google.protobuf"}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    distribution = line.split("==", 1)[0]
    module = modules.get(distribution.lower(), distribution.replace("-", "_"))
    try:
        available = importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        # A missing namespace parent (e.g. google) also means the dependency
        # is missing; do not abort the probe before printing its requirement.
        available = False
    if not available:
        print(line)
PY
)
        if (( ${#MISSING[@]} )); then
            # Pin every installed distribution during resolution so a missing
            # package cannot silently replace a preexisting training stack.
            CONSTRAINTS="$(mktemp)"
            trap 'rm -f "${CONSTRAINTS}"' EXIT
            "${PYTHON}" - "${CONSTRAINTS}" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
pins = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata.get("Name")})
Path(sys.argv[1]).write_text("\n".join(pins) + "\n", encoding="utf-8")
PY
            echo "setup_csgo_seen10: installing missing CSGO packages: ${MISSING[*]}" >&2
            "${PYTHON}" -m pip install --disable-pip-version-check --no-input \
                --upgrade-strategy only-if-needed --constraint "${CONSTRAINTS}" "${MISSING[@]}"
        else
            echo "setup_csgo_seen10: CSGO packages are already present" >&2
        fi
    fi
fi

# The four OpenCV wheels install files into the same cv2 package. In
# particular, imgaug declares opencv-python even though its augmentation
# functions also work with the headless wheel. A fresh pip -r can therefore
# leave both variants installed; import cv2 then fails on servers without
# libGL. Normalize only after every resolver-driven install has finished.
OPENCV_HEADLESS_PIN="$("${PYTHON}" - "${REQUIREMENTS}" <<'PY'
from pathlib import Path
import sys

pins = [line.strip() for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
        if line.strip().lower().startswith("opencv-python-headless==")]
if len(pins) != 1:
    raise SystemExit("requirements_csgo.txt must pin exactly one opencv-python-headless version")
print(pins[0])
PY
)"

opencv_packages() {
    "${PYTHON}" - <<'PY'
from importlib import metadata

for name in ("opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"):
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        continue
    print(name)
PY
}

opencv_headless_healthy() {
    "${PYTHON}" - "${OPENCV_HEADLESS_PIN#*==}" <<'PY'
from importlib import metadata
import re
import sys

if metadata.version("opencv-python-headless") != sys.argv[1]:
    raise SystemExit(1)
import cv2
if not cv2.__file__:
    raise SystemExit(1)
if not re.search(r"(?m)^\s*GUI:\s*NONE(?:\s|$)", cv2.getBuildInformation()):
    raise SystemExit("cv2 was built with GUI support")
PY
}

OPENCV_PACKAGES_OUTPUT="$(opencv_packages)" \
    || die "cannot inspect installed OpenCV distributions in ${ENV_DIR}"
OPENCV_DISTRIBUTIONS=()
if [[ -n "${OPENCV_PACKAGES_OUTPUT}" ]]; then
    mapfile -t OPENCV_DISTRIBUTIONS <<< "${OPENCV_PACKAGES_OUTPUT}"
fi
if [[ ${#OPENCV_DISTRIBUTIONS[@]} != 1 || "${OPENCV_DISTRIBUTIONS[0]:-}" != opencv-python-headless ]] \
    || ! opencv_headless_healthy >/dev/null 2>&1; then
    if [[ "${MODE}" == check ]]; then
        die "OpenCV is not a healthy, sole ${OPENCV_HEADLESS_PIN} install (found: ${OPENCV_DISTRIBUTIONS[*]:-none}); run bash scripts/setup_csgo_seen10.sh to repair it"
    fi
    echo "setup_csgo_seen10: repairing OpenCV; found: ${OPENCV_DISTRIBUTIONS[*]:-none}" >&2
    if (( ${#OPENCV_DISTRIBUTIONS[@]} )); then
        "${PYTHON}" -m pip uninstall --disable-pip-version-check --yes "${OPENCV_DISTRIBUTIONS[@]}"
    fi
    "${PYTHON}" -m pip install --disable-pip-version-check --no-input \
        --no-deps --force-reinstall "${OPENCV_HEADLESS_PIN}"
    OPENCV_PACKAGES_OUTPUT="$(opencv_packages)" \
        || die "cannot inspect OpenCV distributions after repair"
    OPENCV_DISTRIBUTIONS=()
    if [[ -n "${OPENCV_PACKAGES_OUTPUT}" ]]; then
        mapfile -t OPENCV_DISTRIBUTIONS <<< "${OPENCV_PACKAGES_OUTPUT}"
    fi
    [[ ${#OPENCV_DISTRIBUTIONS[@]} == 1 && "${OPENCV_DISTRIBUTIONS[0]}" == opencv-python-headless ]] \
        && opencv_headless_healthy \
        || die "OpenCV repair did not produce a working ${OPENCV_HEADLESS_PIN}; inspect the pip output above"
fi

"${PYTHON}" - "${PROJECT_ROOT}" <<'PY'
import importlib
import importlib.metadata as metadata
import sys
from packaging.version import Version

required = (
    "torch", "torchvision", "transformers", "diffusers", "huggingface_hub",
    "accelerate", "timm", "numpy", "PIL", "yaml", "h5py", "imgaug",
    "sentencepiece", "google.protobuf", "google.protobuf.message", "cv2", "einops", "safetensors", "tqdm",
)
failed = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(f"{name}: {type(exc).__name__}: {exc}")
if failed:
    raise SystemExit("CSGO dependency verification failed:\n  " + "\n  ".join(failed))

try:
    from transformers import AutoTokenizer, T5EncoderModel, SiglipVisionModel, SiglipImageProcessor
    from transformers.convert_slow_tokenizer import import_protobuf
    # Exercise the SentencePiece protobuf schema used by slow-to-fast T5
    # conversion even before official model assets have been downloaded.
    import_protobuf().ModelProto().SerializeToString()
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
    from accelerate import Accelerator
    sys.path.insert(0, sys.argv[1])
    import train.csgo_aligned
    import infer_seen10
except Exception as exc:
    raise SystemExit(f"CSGO code import verification failed: {type(exc).__name__}: {exc}")

from pathlib import Path
tokenizer_dir = Path(sys.argv[1]) / ".cache/csgo_seen10/models/t5-v1_1-xxl"
if tokenizer_dir.is_dir():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, model_max_length=1024, local_files_only=True,
        )
        encoded = tokenizer("Locate the player on de_dust2.", return_tensors="pt")
        if encoded["input_ids"].numel() == 0:
            raise ValueError("tokenizer produced empty input_ids")
        print("T5 tokenizer local load/encode: OK", type(tokenizer).__name__)
    except Exception as exc:
        raise SystemExit(f"CSGO local T5 tokenizer check failed at {tokenizer_dir}: {type(exc).__name__}: {exc}")
else:
    print("T5 tokenizer local load/encode: SKIPPED (assets absent; rerun --check after prepare_csgo_assets.py)")

import torch
if Version(torch.__version__.split("+", 1)[0]) < Version("2.6.0"):
    raise SystemExit("CSGO requires torch >= 2.6 to safely load the official T5 pytorch_model.bin; use a matching supported torch/torchvision pair")
print("python", sys.version.replace("\n", " "))
for distribution in (
    "torch", "torchvision", "transformers", "diffusers", "huggingface-hub",
    "accelerate", "timm", "numpy", "sentencepiece", "protobuf", "h5py", "imgaug",
):
    print(f"{distribution} {metadata.version(distribution)}")
print("torch.cuda.is_available", torch.cuda.is_available())
print("torch.cuda.device_count", torch.cuda.device_count())
PY

if [[ "${MODE}" == setup && -f "${BOOTSTRAP_MARKER}" ]]; then
    rm -f "${BOOTSTRAP_MARKER}"
elif [[ "${MODE}" == check && -f "${BOOTSTRAP_MARKER}" ]]; then
    echo "setup_csgo_seen10: clean bootstrap is marked incomplete; rerun setup to apply all pins" >&2
fi
echo "setup_csgo_seen10: ready: ${PYTHON}" >&2
