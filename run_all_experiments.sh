#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PATIENCE="${PATIENCE:-50}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --patience)
      PATIENCE="${2:-50}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

EXPERIMENTS_ROOT="$ROOT_DIR/experiments"
LOG_ROOT="$ROOT_DIR/experiments/_logs"
mkdir -p "$EXPERIMENTS_ROOT" "$LOG_ROOT"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_ROOT/run_${RUN_TS}"
mkdir -p "$RUN_LOG_DIR"
MASTER_LOG="$RUN_LOG_DIR/all.log"

clean_stream() {
  perl -pe 's/\r/\n/g; s/\x1b\[[0-9;?]*[ -\/]*[@-~]//g; s/\x1b\][^\x07]*(?:\x07|\x1b\\)//g; s/\x1b[@-_]//g' \
    | grep -aEv '^(Training:|Epoch [0-9]+ \[(train|val)\]:|Ep [0-9]+ \[(train|val)\]:)' \
    | grep -aEv '(^[[:space:]]*[0-9]+%?\|)|(\|[[:space:]]*[0-9]+/[0-9]+[[:space:]]*\[)|(\?[[:space:]]*[A-Za-z]+/s\]$)' \
    | grep -aEv '^[[:space:]]*$'
}

exec > >(
  tee >(
    clean_stream >> "$MASTER_LOG"
  )
) 2>&1

[[ -f "$ROOT_DIR/seenx-ml-venv/bin/activate" ]] && source "$ROOT_DIR/seenx-ml-venv/bin/activate"
[[ -f "$ROOT_DIR/venv/bin/activate" ]] && source "$ROOT_DIR/venv/bin/activate"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
VAL_N="${VAL_N:-10}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CURVE_POINTS="${CURVE_POINTS:-0}"
TIME_FEATURES="${TIME_FEATURES:-none}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-540}"
MAX_DURATION_SEC="${MAX_DURATION_SEC:-1620}"
SEQ_EXTRA_ARGS=()
[[ "$CURVE_POINTS" != "0" ]] && SEQ_EXTRA_ARGS+=(--curve-points "$CURVE_POINTS")
[[ "$TIME_FEATURES" != "none" ]] && SEQ_EXTRA_ARGS+=(--time-features "$TIME_FEATURES")
[[ "$MIN_DURATION_SEC" != "0" ]] && SEQ_EXTRA_ARGS+=(--min-duration-sec "$MIN_DURATION_SEC")
[[ "$MAX_DURATION_SEC" != "0" ]] && SEQ_EXTRA_ARGS+=(--max-duration-sec "$MAX_DURATION_SEC")

HEAVY_BATCH="${HEAVY_BATCH:-16}"
HEAVY_D_MODEL="${HEAVY_D_MODEL:-512}"
HEAVY_N_LAYERS_LSTM="${HEAVY_N_LAYERS_LSTM:-4}"
HEAVY_N_LAYERS_TRF="${HEAVY_N_LAYERS_TRF:-6}"
HEAVY_N_HEADS="${HEAVY_N_HEADS:-8}"
HEAVY_D_FF="${HEAVY_D_FF:-1024}"

TUNE_LSTM_JSON="${TUNE_LSTM_JSON:-$ROOT_DIR/tune_hp/results/tune_multimodal_lstm_best.json}"
TUNE_TRF_JSON="${TUNE_TRF_JSON:-$ROOT_DIR/tune_hp/results/tune_multimodal_transformer_best.json}"
TUNE_TABULAR_JSON="${TUNE_TABULAR_JSON:-$ROOT_DIR/configs/tuned_tabular_transformer.json}"

echo "[all] log=${RUN_LOG_DIR#$ROOT_DIR/} device=$DEVICE val=$VAL_N epochs=$EPOCHS patience=$PATIENCE"

run_exp() {
  local name="$1"
  local out_dir="$2"
  shift 2
  local cmd=("$@")

  local abs_out="$ROOT_DIR/$out_dir"
  local log_file="$RUN_LOG_DIR/${name}.log"

  echo "[run] $name out=$out_dir log=${log_file#$ROOT_DIR/}"

  if [[ -d "$abs_out" ]]; then
    echo "[clean] $out_dir"
    rm -rf "$abs_out"
  fi
  mkdir -p "$abs_out"

  set +e
  "${cmd[@]}" 2>&1 \
    | tee >(
        clean_stream > "$log_file"
      )
  local rc=${PIPESTATUS[0]}
  set -e

  [[ $rc -ne 0 ]] && { echo "[fail] $name exit=$rc"; FAILED_EXPERIMENTS+=("$name"); } || echo "[ok] $name"
  echo
}

tuned_args() {
  local json="$1"
  [[ -f "$json" ]] && printf '%s\n%s\n' --tuned-params-json "$json"
}

FAILED_EXPERIMENTS=()

if [[ "${RUN_OPTUNA:-0}" == "1" ]]; then
  N_OPTUNA="${N_OPTUNA_TRIALS:-5}"
  O_EPOCHS="${OPTUNA_EPOCHS_PER_TRIAL:-150}"
  echo "[pre] optuna trials=$N_OPTUNA epochs=$O_EPOCHS"
  mkdir -p "$ROOT_DIR/tune_hp/results"
  for arch in multimodal_transformer multimodal_lstm transformer lstm; do
    set +e
    "$PYTHON_BIN" "$ROOT_DIR/tune_hp/tune.py" \
      --arch "$arch" \
      --n-trials "$N_OPTUNA" \
      --epochs-per-trial "$O_EPOCHS" \
      --device "$DEVICE" \
      --output-dir "$ROOT_DIR/tune_hp/results" \
      --val-first-n-output "$VAL_N"
    tune_rc=$?
    set -e
    [[ $tune_rc -ne 0 ]] && echo "[warn] tune arch=$arch exit=$tune_rc"
    study_json="tune_${arch}_best.json"
    if [[ -f "$ROOT_DIR/tune_hp/results/$study_json" ]]; then
      cp -f "$ROOT_DIR/tune_hp/results/$study_json" "$ROOT_DIR/$study_json"
      echo "[ok] copied $study_json"
    fi
  done
  echo
fi

if [[ "${SKIP_CLUSTER_JSON:-0}" != "1" ]]; then
  echo "[pre] duration_video_clusters"
  "$PYTHON_BIN" -m train.clustering.cluster_specialists_multimodal \
    --repo-root "$ROOT_DIR" \
    --features-root "$ROOT_DIR/data" \
    --clusters-json "$ROOT_DIR/configs/video_clusters.json" \
    --lists-json "$ROOT_DIR/configs/video_cluster_train_lists.json" || true
  echo
fi

if [[ "${SKIP_NEW_CLUSTERING:-0}" != "1" ]]; then
  echo "[pre] video_clustering"
  "$PYTHON_BIN" analysis/video_clustering.py --min-k 6 --max-k 8 --strategy retention || true
  echo "[pre] cluster_curve_features"
  "$PYTHON_BIN" src/analysis/cluster_and_curve_features.py --clusters-file analysis/video_clustering/retention/clusters.json || true
  echo
fi
run_exp \
  "lstm_v1_baseline" \
  "experiments/lstm_exp/v1_baseline" \
  "$PYTHON_BIN" train/train_lstm_seq_v1.py \
  --output-dir "experiments/lstm_exp/v1_baseline" \
  --output-dir-features output \
  --snapshot-dir data \
  --val-first-n-output "$VAL_N" \
  --hidden-size 256 \
  --n-layers 3 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --patience "$PATIENCE" \
  --min-duration-sec "$MIN_DURATION_SEC" \
  --max-duration-sec "$MAX_DURATION_SEC" \
  --device "$DEVICE"

run_exp \
  "lstm_v2_tabular_pca" \
  "experiments/lstm_exp/v2_tabular_pca" \
  "$PYTHON_BIN" train/train_lstm_seq.py \
  --output-dir "experiments/lstm_exp/v2_tabular_pca" \
  --output-dir-features output \
  --snapshot-dir data \
  --val-first-n-output "$VAL_N" \
  --hidden-size 256 \
  --n-layers 3 \
  --emb-pca-components 12 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --feature-mask-prob 0 \
  --noise-std 0 \
  --swa-start-epoch 100000 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" \
  --device "$DEVICE"

run_exp \
  "lstm_v3_multimodal" \
  "experiments/lstm_exp/v3_multimodal" \
  "$PYTHON_BIN" train/train_multimodal_seq.py \
  --arch lstm \
  --output-dir "experiments/lstm_exp/v3_multimodal" \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" \
  --n-layers "$HEAVY_N_LAYERS_LSTM" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" \
  --device "$DEVICE"

mapfile -t TUNE_LSTM_ARGS < <(tuned_args "$TUNE_LSTM_JSON")
[[ ${#TUNE_LSTM_ARGS[@]} -eq 0 ]] && echo "[warn] missing $TUNE_LSTM_JSON"
run_exp \
  "lstm_v4_tuned_multimodal" \
  "experiments/lstm_exp/v4_tuned_multimodal" \
  "$PYTHON_BIN" train/train_multimodal_seq.py \
  --arch lstm \
  --output-dir "experiments/lstm_exp/v4_tuned_multimodal" \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" \
  --n-layers "$HEAVY_N_LAYERS_LSTM" \
  "${TUNE_LSTM_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" \
  --device "$DEVICE"

run_exp \
  "transformer_v3_multimodal" \
  "experiments/transformer_exp/v3_multimodal" \
  "$PYTHON_BIN" train/train_multimodal_seq.py \
  --arch transformer \
  --output-dir "experiments/transformer_exp/v3_multimodal" \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" \
  --n-heads "$HEAVY_N_HEADS" \
  --n-layers "$HEAVY_N_LAYERS_TRF" \
  --d-ff "$HEAVY_D_FF" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" \
  --device "$DEVICE"

mapfile -t TUNE_TAB_ARGS < <(tuned_args "$TUNE_TABULAR_JSON")
run_exp \
  "transformer_v4_tuned" \
  "experiments/transformer_exp/v4_tuned" \
  "$PYTHON_BIN" train/train_transformer_seq.py \
  --output-dir "experiments/transformer_exp/v4_tuned" \
  --output-dir-features output \
  --snapshot-dir data \
  --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" \
  --n-heads "$HEAVY_N_HEADS" \
  --n-layers "$HEAVY_N_LAYERS_TRF" \
  --d-ff "$HEAVY_D_FF" \
  --emb-pca-components 36 \
  "${TUNE_TAB_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" \
  --device "$DEVICE"

mapfile -t TUNE_TRF_ARGS < <(tuned_args "$TUNE_TRF_JSON")
[[ ${#TUNE_TRF_ARGS[@]} -eq 0 ]] && echo "[warn] missing $TUNE_TRF_JSON"
run_exp \
  "transformer_v4_tuned_multimodal" \
  "experiments/transformer_exp/v4_tuned_multimodal" \
  "$PYTHON_BIN" train/train_multimodal_seq.py \
  --arch transformer \
  --output-dir "experiments/transformer_exp/v4_tuned_multimodal" \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" \
  --n-heads "$HEAVY_N_HEADS" \
  --n-layers "$HEAVY_N_LAYERS_TRF" \
  --d-ff "$HEAVY_D_FF" \
  "${TUNE_TRF_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" \
  --device "$DEVICE"

run_exp \
  "videomae_v1_extract" \
  "experiments/videomae_exp/v1_extract" \
  "$PYTHON_BIN" train/train_videomae_seq.py \
  --mode extract \
  --backbone videomae-base \
  --output-dir "experiments/videomae_exp/v1_extract" \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" \
  --n-heads "$HEAVY_N_HEADS" \
  --n-layers "$HEAVY_N_LAYERS_TRF" \
  --d-ff "$HEAVY_D_FF" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" \
  --device "$DEVICE"

run_exp \
  "videomae_v1_hybrid" \
  "experiments/videomae_exp/v1_hybrid" \
  "$PYTHON_BIN" train/train_videomae_seq.py \
  --mode hybrid \
  --backbone videomae-base \
  --output-dir "experiments/videomae_exp/v1_hybrid" \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" \
  --n-heads "$HEAVY_N_HEADS" \
  --n-layers "$HEAVY_N_LAYERS_TRF" \
  --d-ff "$HEAVY_D_FF" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" \
  --device "$DEVICE"

run_exp \
  "bert_v1_extract" \
  "experiments/bert_exp/v1_extract" \
  "$PYTHON_BIN" train/train_bert_seq.py \
  --mode extract \
  --backbone deberta-base \
  --output-dir "experiments/bert_exp/v1_extract" \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" \
  --n-heads "$HEAVY_N_HEADS" \
  --n-layers "$HEAVY_N_LAYERS_TRF" \
  --d-ff "$HEAVY_D_FF" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" \
  --device "$DEVICE"

run_exp \
  "bert_v1_e2e" \
  "experiments/bert_exp/v1_e2e" \
  "$PYTHON_BIN" train/train_bert_seq.py \
  --mode e2e \
  --backbone deberta-base \
  --lora-rank 8 \
  --lora-alpha 16 \
  --output-dir "experiments/bert_exp/v1_e2e" \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output "$VAL_N" \
  --n-layers 2 \
  --d-ff "$HEAVY_D_FF" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" \
  --device "$DEVICE"

run_exp \
  "bert_v1_hybrid" \
  "experiments/bert_exp/v1_hybrid" \
  "$PYTHON_BIN" train/train_bert_seq.py \
  --mode hybrid \
  --backbone deberta-base \
  --output-dir "experiments/bert_exp/v1_hybrid" \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" \
  --n-heads "$HEAVY_N_HEADS" \
  --n-layers "$HEAVY_N_LAYERS_TRF" \
  --d-ff "$HEAVY_D_FF" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" \
  --device "$DEVICE"

run_exp \
  "metamodel_nnls_ensemble" \
  "experiments/metamodel_exp" \
  "$PYTHON_BIN" train/metamodel/train_metamodel.py \
  --output-dir "experiments/metamodel_exp" \
  --lstm-exp "experiments/lstm_exp/v3_multimodal" \
  --tf-exp "experiments/transformer_exp/v4_tuned_multimodal" \
  --vmae-exp "experiments/videomae_exp/v1_hybrid" \
  --device "$DEVICE"

run_exp \
  "lstm_content_cluster_specialists" \
  "experiments/lstm_exp/content_cluster_specialists" \
  "$PYTHON_BIN" train/content_cluster_specialists.py \
  --arch lstm \
  --repo-root "$ROOT_DIR" \
  --run-clustering-first \
  --data-dir "$ROOT_DIR/data" \
  --embeddings-dir "$ROOT_DIR/embeddings" \
  --features-output-dir "$ROOT_DIR/output" \
  --cluster-out-root "$ROOT_DIR/analysis/video_clustering" \
  --clusters-json "$ROOT_DIR/analysis/video_clustering/retention/clusters.json" \
  --cluster-min-k 6 \
  --cluster-max-k 8 \
  --clustering-strategy retention \
  --output-base "experiments/lstm_exp/content_cluster_specialists" \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" \
  --n-layers "$HEAVY_N_LAYERS_LSTM" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  --device "$DEVICE" \
  "${TUNE_LSTM_ARGS[@]}"

run_exp \
  "transformer_content_cluster_specialists" \
  "experiments/transformer_exp/content_cluster_specialists" \
  "$PYTHON_BIN" train/content_cluster_specialists.py \
  --arch transformer \
  --repo-root "$ROOT_DIR" \
  --clusters-json "$ROOT_DIR/analysis/video_clustering/retention/clusters.json" \
  --output-base "experiments/transformer_exp/content_cluster_specialists" \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" \
  --n-layers "$HEAVY_N_LAYERS_TRF" \
  --n-heads "$HEAVY_N_HEADS" \
  --d-ff "$HEAVY_D_FF" \
  --epochs "$EPOCHS" \
  --batch-size "$HEAVY_BATCH" \
  --device "$DEVICE" \
  "${TUNE_TRF_ARGS[@]}"

echo "[summary]"
"$PYTHON_BIN" "$ROOT_DIR/train/summarize_experiments.py" --root "$ROOT_DIR"

echo
if [[ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]]; then
  echo "[done] failed=${#FAILED_EXPERIMENTS[@]}"
  printf '  - %s\n' "${FAILED_EXPERIMENTS[@]}"
  echo "log=${MASTER_LOG#$ROOT_DIR/} runs=${RUN_LOG_DIR#$ROOT_DIR/}"
  exit 1
fi

echo "[done] log=${MASTER_LOG#$ROOT_DIR/} runs=${RUN_LOG_DIR#$ROOT_DIR/}"
