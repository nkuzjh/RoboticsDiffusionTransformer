#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
UNILIP_PYTHON="${UNILIP_PYTHON:-/home/jiahao/miniconda3/envs/UniLIP/bin/python}"
EVAL_ROOT="${SHARED_EVAL_DIR:-${CSGO_EVAL_ROOT:-/home/jiahao/task/csgo_benchmark_v2_eval_general}}"
DATA_ROOT="${DATA_ROOT:-${CSGO_DATA_ROOT:-}}"
CONFIG="$PROJECT_ROOT/configs/csgo_seen10.yaml"
CONFIG_EXPLICIT=0
SEED=""
MODE=""
OUTPUT_DIR=""
CHECKPOINT_DIR=""
NATIVE_EXTRA=()
CPU=0

usage() {
  cat <<'EOF'
Usage: scripts/run_csgo_seen10.sh {train|infer|eval|smoke} [options] [-- mode-specific extra args]

Options:
  --seed N                 Seed directory (default: YAML seed)
  --config PATH            Seen-10 YAML config
  --data-root PATH         Released benchmark bundle
  --output-dir PATH        Complete artifact directory for this seed
  --checkpoint-dir PATH    Complete native checkpoint directory for this seed
  --python PATH            Project Python interpreter
  --unilip-python PATH     Shared evaluator Python interpreter
  --cpu                    Run model training/inference on CPU
  --                       Forward remaining arguments to train_seen10.py for train/smoke,
                           or infer_seen10.py for infer
EOF
}

if (($# == 0)); then
  usage >&2
  exit 2
fi
MODE="$1"
shift

while (($# > 0)); do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      CONFIG_EXPLICIT=1
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --unilip-python)
      UNILIP_PYTHON="$2"
      shift 2
      ;;
    --cpu)
      CPU=1
      shift
      ;;
    --)
      shift
      NATIVE_EXTRA+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      NATIVE_EXTRA+=("$1")
      shift
      ;;
  esac
done

# The environment preparation leaves a local tiny model bundle for the
# five-step smoke.  Formal train/infer always keep using the released config.
if [[ "$MODE" == "smoke" && "$CONFIG_EXPLICIT" == 0 && -f "$PROJECT_ROOT/.cache/csgo_seen10/smoke/config.yaml" ]]; then
  CONFIG="$PROJECT_ROOT/.cache/csgo_seen10/smoke/config.yaml"
fi

cd "$PROJECT_ROOT"

# Resolve defaults from the chosen profile; never replace aligned paths with
# legacy seed_0 defaults. Explicit CLI/environment paths still win.
resolved="$("$PYTHON" - "$CONFIG" "$SEED" "$MODE" <<'PY'
import sys
from pathlib import Path
import yaml
c = yaml.safe_load(Path(sys.argv[1]).read_text())
seed = int(sys.argv[2]) if sys.argv[2] else int(c.get('seed', 0))
parts = [str(c.get('model_name', 'RDT'))]
if sys.argv[3] == 'smoke':
    parts.append('smoke')
parts.append(f'seed_{seed}')
print(seed)
print(c['data_root'])
print(Path(c.get('output_root', 'outputs/csgo_benchmark_v2_seen10')).joinpath(*parts))
print(Path(c.get('checkpoint_root', 'checkpoints/csgo_benchmark_v2_seen10')).joinpath(*parts))
PY
)"
mapfile -t profile_paths <<< "$resolved"
SEED="${profile_paths[0]}"
DATA_ROOT="${DATA_ROOT:-${profile_paths[1]}}"
OUTPUT_DIR="${OUTPUT_DIR:-${profile_paths[2]}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${profile_paths[3]}}"

cpu_args=()
if ((CPU)); then
  cpu_args+=(--cpu)
fi
train_cmd=("$PYTHON" "$PROJECT_ROOT/train_seen10.py" train --config "$CONFIG" --data-root "$DATA_ROOT" --seed "$SEED" --output-dir "$OUTPUT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" "${cpu_args[@]}")
infer_cmd=("$PYTHON" "$PROJECT_ROOT/infer_seen10.py" infer --config "$CONFIG" --data-root "$DATA_ROOT" --seed "$SEED" --output-dir "$OUTPUT_DIR" --checkpoint "$CHECKPOINT_DIR" "${cpu_args[@]}")

case "$MODE" in
  train)
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" "${train_cmd[@]}" "${NATIVE_EXTRA[@]}"
    ;;
  infer)
    "${infer_cmd[@]}" "${NATIVE_EXTRA[@]}"
    ;;
  eval)
    if ((${#NATIVE_EXTRA[@]})); then
      echo "run_csgo_seen10.sh: native extra args are supported only for train/smoke" >&2
      exit 2
    fi
    mkdir -p "$OUTPUT_DIR/evaluation"
    "$UNILIP_PYTHON" "$EVAL_ROOT/run_eval.py" localization \
      --pred-root "$OUTPUT_DIR/localization" \
      --data-root "$DATA_ROOT" \
      --pose-space normalized \
      --output "$OUTPUT_DIR/evaluation/localization"
    ;;
  smoke)
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" \
      "$PYTHON" "$PROJECT_ROOT/train_seen10.py" smoke --config "$CONFIG" --data-root "$DATA_ROOT" \
      --seed "$SEED" --output-dir "$OUTPUT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" "${cpu_args[@]}" "${NATIVE_EXTRA[@]}"
    "$PYTHON" "$PROJECT_ROOT/infer_seen10.py" smoke --config "$CONFIG" --data-root "$DATA_ROOT" \
      --seed "$SEED" --output-dir "$OUTPUT_DIR" --checkpoint "$CHECKPOINT_DIR" "${cpu_args[@]}"
    "$UNILIP_PYTHON" "$EVAL_ROOT/run_eval.py" smoke localization \
      --pred-root "$OUTPUT_DIR/localization" \
      --data-root "$DATA_ROOT" \
      --limit 1
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
