#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
EVAL_PYTHON_EXPLICIT=""
EVAL_ROOT="${SHARED_EVAL_DIR:-${CSGO_EVAL_ROOT:-}}"
DATA_ROOT="${DATA_ROOT:-${CSGO_DATA_ROOT:-}}"
CONFIG="$PROJECT_ROOT/configs/csgo_seen10.yaml"
CONFIG_EXPLICIT=0
SEED=""
MODE=""
OUTPUT_DIR=""
CHECKPOINT_DIR=""
NATIVE_EXTRA=()
CPU=0
PRINT_PATHS=0

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
  --eval-python PATH       Shared evaluator Python interpreter
  --unilip-python PATH     Alias for --eval-python
  --eval-root PATH         Shared evaluator checkout (containing run_eval.py)
  --print-paths            Print resolved paths only; do not run any task
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
    --eval-python|--unilip-python)
      EVAL_PYTHON_EXPLICIT="$2"
      shift 2
      ;;
    --eval-root)
      EVAL_ROOT="$2"
      shift 2
      ;;
    --print-paths)
      PRINT_PATHS=1
      shift
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
if [[ ! -x "$PYTHON" ]]; then
  echo "run_csgo_seen10: Python unavailable: $PYTHON; run bash scripts/setup_csgo_seen10.sh first" >&2
  exit 2
fi

# Resolve defaults from the chosen profile; never replace aligned paths with
# legacy seed_0 defaults. Explicit CLI/environment paths still win.
resolved="$("$PYTHON" - "$CONFIG" "$SEED" "$MODE" "$DATA_ROOT" "$EVAL_ROOT" "$EVAL_PYTHON_EXPLICIT" "$OUTPUT_DIR" "$CHECKPOINT_DIR" <<'PY'
import sys
from pathlib import Path
import yaml
from scripts.csgo_paths import data_root, evaluator_root, evaluator_python, run_directories
c = yaml.safe_load(Path(sys.argv[1]).read_text())
seed = int(sys.argv[2]) if sys.argv[2] else int(c.get('seed', 0))
artifacts, checkpoints = run_directories(c, seed=seed, smoke=sys.argv[3] == 'smoke',
                                        output=sys.argv[7] or None, checkpoint=sys.argv[8] or None)
print(seed)
print(data_root(c, sys.argv[4] or None))
print(artifacts)
print(checkpoints)
selected_eval_root = evaluator_root(c, sys.argv[5] or None)
print(selected_eval_root)
print(evaluator_python(c, sys.argv[6] or None, eval_root=selected_eval_root))
PY
)"
mapfile -t profile_paths <<< "$resolved"
SEED="${profile_paths[0]}"
DATA_ROOT="${profile_paths[1]}"
OUTPUT_DIR="${profile_paths[2]}"
CHECKPOINT_DIR="${profile_paths[3]}"
EVAL_ROOT="${profile_paths[4]}"
UNILIP_PYTHON="${profile_paths[5]}"

if ((PRINT_PATHS)); then
  "$PYTHON" - "$CONFIG" "$SEED" "$DATA_ROOT" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$EVAL_ROOT" "$UNILIP_PYTHON" <<'PY'
import json, sys
paths = dict(zip(("config", "seed", "data_root", "output_dir", "checkpoint_dir", "shared_eval_dir", "unilip_python"), sys.argv[1:]))
paths["evaluator_python"] = paths["unilip_python"]
print(json.dumps(paths, indent=2))
PY
  exit 0
fi

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
