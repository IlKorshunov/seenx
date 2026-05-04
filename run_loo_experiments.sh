#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EXPERIMENTS_ROOT="$ROOT_DIR/experiments/loo"
LOG_ROOT="$ROOT_DIR/experiments/_logs"
mkdir -p "$EXPERIMENTS_ROOT" "$LOG_ROOT"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_ROOT/loo_${RUN_TS}"
mkdir -p "$RUN_LOG_DIR"
MASTER_LOG="$RUN_LOG_DIR/all.log"

clean_stream() {
  perl -pe 's/\r/\n/g; s/\x1b\[[0-9;?]*[ -\/]*[@-~]//g; s/\x1b\][^\x07]*(?:\x07|\x1b\\)//g; s/\x1b[@-_]//g' \
    | grep -aEv '^[[:space:]]*$'
}

exec > >(tee >(clean_stream >> "$MASTER_LOG")) 2>&1

[[ -f "$ROOT_DIR/venv/bin/activate" ]] && source "$ROOT_DIR/venv/bin/activate"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-data}"
CURVE_POINTS="${CURVE_POINTS:-20}"
RANDOM_SEED="${RANDOM_SEED:-42}"
ITERATIONS="${ITERATIONS:-700}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
DEPTH="${DEPTH:-6}"
LIMIT_VIDEOS="${LIMIT_VIDEOS:-83}"
TRAIN_VIDEOS="${TRAIN_VIDEOS:-82}"

echo "[loo] log=${RUN_LOG_DIR#$ROOT_DIR/} curve_points=$CURVE_POINTS seed=$RANDOM_SEED iterations=$ITERATIONS"

DEVICE="${DEVICE:-cuda}"
TOTAL_MODELS=15
CURRENT_MODEL=0
FAILED=()

run_loo() {
  local name="$1"
  local script="$2"
  shift 2
  local include_device_args=1
  local extra_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-device-args)
        include_device_args=0
        ;;
      *)
        extra_args+=("$1")
        ;;
    esac
    shift
  done

  CURRENT_MODEL=$((CURRENT_MODEL + 1))
  local out_dir="$EXPERIMENTS_ROOT/$name"
  local log_file="$RUN_LOG_DIR/${name}.log"

  echo "[run] ${CURRENT_MODEL}/${TOTAL_MODELS} $name out=${out_dir#$ROOT_DIR/} log=${log_file#$ROOT_DIR/}"

  mkdir -p "$out_dir"

  local device_args=()
  [[ "$include_device_args" -eq 1 && "$DEVICE" == "cuda" ]] && device_args=(--task-type GPU --gpu-ram-part "${GPU_RAM_PART:-0.6}")
  [[ "$include_device_args" -eq 1 && "$DEVICE" != "cuda" ]] && device_args=(--task-type CPU)

  local t_start=$SECONDS
  set +e
  "$PYTHON_BIN" "train/$script" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --curve-points "$CURVE_POINTS" \
    --random-seed "$RANDOM_SEED" \
    --limit-videos "$LIMIT_VIDEOS" \
    --train-videos "$TRAIN_VIDEOS" \
    --output-dir "$out_dir" \
    "${device_args[@]}" \
    "${extra_args[@]}" \
    2>&1 | tee >(clean_stream > "$log_file")
  local rc=${PIPESTATUS[0]}
  set -e
  local elapsed=$(( SECONDS - t_start ))

  [[ $rc -ne 0 ]] && { echo "[fail] $name exit=$rc ${elapsed}s"; FAILED+=("$name"); } || echo "[ok] $name ${elapsed}s"
  echo
}

run_loo "regressor" "train_retention_regressor_loo.py" --iterations "$ITERATIONS" --learning-rate "$LEARNING_RATE" --depth "$DEPTH"
run_loo "shape_only" "train_retention_shape_only_loo.py"
run_loo "local_knn" "train_retention_local_knn_loo.py"
run_loo "ranker" "train_retention_ranker_loo.py" --iterations "$ITERATIONS" --learning-rate "$LEARNING_RATE" --depth "$DEPTH"
run_loo "hybrid" "train_retention_hybrid_loo.py"
run_loo "stacked" "train_retention_stacked_loo.py" --iterations "$ITERATIONS" --learning-rate "$LEARNING_RATE" --depth "$DEPTH"
run_loo "conservative_catboost" "train_retention_conservative_catboost_loo.py"
run_loo "ad_peak_weighted" "train_retention_ad_peak_weighted_loo.py"
run_loo "peak_weighted" "train_retention_peak_weighted_loo.py"
run_loo "blended_quantile" "train_retention_blended_quantile_loo.py"
run_loo "residual_huber" "train_retention_residual_huber_loo.py"
run_loo "kernel_baseline" "train_retention_kernel_baseline_loo.py"
run_loo "xgb_flat" "train_retention_xgb_flat_loo.py" --no-device-args
run_loo "meta_ensemble" "train_retention_meta_ensemble_loo.py"
run_loo "integration_penalty" "train_retention_integration_penalty_loo.py"

echo "[report] loo"
"$PYTHON_BIN" train/loo_report.py \
  --loo-root "$EXPERIMENTS_ROOT" \
  --snapshot-dir "$SNAPSHOT_DIR" \
  --curve-points "$CURVE_POINTS"

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[done] failed=${#FAILED[@]}/$TOTAL_MODELS"
  printf '  - %s\n' "${FAILED[@]}"
  echo "log=${MASTER_LOG#$ROOT_DIR/}"
  exit 1
fi
echo "[done] all=$TOTAL_MODELS log=${MASTER_LOG#$ROOT_DIR/} leaderboard=${EXPERIMENTS_ROOT#$ROOT_DIR/}/leaderboard.png"
