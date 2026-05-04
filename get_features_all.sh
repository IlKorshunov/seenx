source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com

python3 main.py aggregate --batch \
  -o output \
  -c configs/local.json \
  -d data
