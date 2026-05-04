set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/venv/bin/activate" ]] && source "$SCRIPT_DIR/venv/bin/activate"

CONFIG="configs/local.json"
DATA_DIR="data"
OUTPUT_DIR="output"
WEIGHTS_DIR="static/weights"
mkdir -p "$WEIGHTS_DIR"

download_if_missing() {
  local path="$1" url="$2"
  [[ -f "$path" ]] && return
  echo "[download] $(basename "$path")"
  wget -q --show-progress -O "$path" "$url"
}

download_if_missing "$WEIGHTS_DIR/transnetv2-pytorch-weights.pth" \
  "https://huggingface.co/Sn4kehead/TransNetV2/resolve/main/transnetv2-pytorch-weights.pth?download=true"
download_if_missing "$WEIGHTS_DIR/arcface_weights.h5" \
  "https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5"
download_if_missing "$WEIGHTS_DIR/yolov12l-face.pt" \
  "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt"
download_if_missing "$WEIGHTS_DIR/raft-small.pth" \
  "https://huggingface.co/ddrfan/RAFT/resolve/main/raft-small.pth?download=true"

if [ ! -d "RAFT" ]; then
  echo "[download] RAFT"
  git clone --depth 1 https://github.com/princeton-vl/RAFT.git RAFT
fi

python3 -c "import demucs" 2>/dev/null || \
  pip install --quiet --no-deps "demucs @ git+https://github.com/facebookresearch/demucs"

mkdir -p "$OUTPUT_DIR"

echo "[run] feature extraction"
for video_dir in "$DATA_DIR"/*/; do
  video_id="$(basename "$video_dir")"
  video_path="${video_dir}video.mp4"
  retention_path="${video_dir}retention.csv"
  output_csv="${OUTPUT_DIR}/${video_id}_features.csv"
  [[ ! -f "$video_path" || -f "$output_csv" ]] && continue
  args=(main.py aggregate -v "$video_path" -o "$output_csv" -c "$CONFIG")
  [[ -f "$retention_path" ]] && args+=(-r "$retention_path")
  echo "[aggregate] $video_id"
  python3 "${args[@]}"
done

echo "[run] summary"
python3 -m src.summarize --features_dir "$OUTPUT_DIR" --data_dir "$DATA_DIR" --output_dir my_metrics

echo "[run] catboost"
python3 main.py train \
  --features_dir "$OUTPUT_DIR" \
  --data_dir "$DATA_DIR" \
  --output_dir my_metrics \
  --save_path static/weights/model.cbm

echo "[run] error_analysis"
python3 main.py error_analysis --features_dir "$OUTPUT_DIR" --output_dir my_metrics

echo "[run] feature_importance"
python3 -m analysis.feature_importance.run_all \
  --output_dir "$OUTPUT_DIR" \
  --results_dir analysis/feature_importance/results \
  --top_n 30

if [[ "${RUN_OPTUNA:-0}" == "1" ]]; then
  echo "[run] optuna"
  python3 train/optimize_hyperparams.py --features_dir "$OUTPUT_DIR" --n_trials 30
fi

TRAIN_TARGET="${TRAIN_TARGET:-}"
if [[ "$TRAIN_TARGET" == "transformer" || "$TRAIN_TARGET" == "ensemble" ]]; then
  echo "[run] train target=$TRAIN_TARGET"
  python3 main.py train \
    --features_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir my_metrics \
    --save_path static/weights/model.cbm \
    --target "$TRAIN_TARGET"
fi

echo "[done] my_metrics"
