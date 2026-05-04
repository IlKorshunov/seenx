# Experimental models in `new/`

This directory contains isolated experiments that do **not** modify the main
training pipeline under `train/` and `src/`.

## SARIMAX + multimodal LSTM

Files:

- `model_sarima_multimodal_lstm.py`  
  Residual multimodal LSTM that predicts corrections over a dynamic SARIMAX prior.

- `sarima_trend.py`  
  Fits one SARIMAX trend per content cluster (fallback: global train mean curve).

- `train_sarima_multimodal_lstm.py`  
  End-to-end training script for the hybrid model.

## Design

The main motivation is to add a stronger **trend prior** than the single global
baseline currently used by `MultimodalRetentionLSTM`.

1. Train videos are grouped by `video_cluster` when available.
2. For each cluster, a mean retention curve is built.
3. A small SARIMAX grid search selects the best `(p,d,q)` order by AIC.
4. Exogenous signals such as `hook_score`, `topic_change_rate`,
   `question_density`, `is_ad`, `edit_pace`, `duration_sec`, and
   `video_cluster` are used as trend controls.
5. For each video, the corresponding cluster SARIMAX prior is used as a baseline.
6. The multimodal LSTM predicts the residual around that baseline.

This avoids the circular dependency of extracting curve parameters from the
unknown future retention of a new video.

## Dependency

This experiment requires `statsmodels` in addition to the existing project stack.

## Example run

```bash
python new/train_sarima_multimodal_lstm.py \
  --output-dir new/experiments/sarimax_multimodal_lstm \
  --output-dir-features output \
  --snapshot-dir data \
  --embeddings-root embeddings \
  --val-first-n-output 10 \
  --device cuda
```

## Mixture-of-experts (MoE) multimodal LSTM

Files:

- `model_moe_multimodal_lstm.py` — shared multimodal LSTM backbone with ``K``
  lightweight expert deviation heads; gate mixes experts using **soft** logits,
  **hard** routing from ``video_cluster % K``, or a **hybrid** convex combination.
- `train_moe_multimodal_lstm.py` — same data path as the SARIMAX experiment;
  optional SARIMAX baseline (default on) or ``--no-sarimax-baseline`` for a
  pure neural curve. Optional ``--load-balance-weight`` penalises uneven
  average gate weights.

Example:

```bash
python new/train_moe_multimodal_lstm.py \
  --output-dir new/experiments/moe_multimodal_lstm \
  --routing-mode soft \
  --n-experts 6 \
  --cluster-embed-buckets 32 \
  --device cuda
```

## A/B comparison of metrics

`ab_testing.py` aligns ``per_video`` entries from two ``metrics.json`` files,
builds bootstrap CIs for per-metric means, and runs paired **t-test**, **Wilcoxon**,
and a **paired permutation** test on the mean difference (default report:
``mean(B−A)``).

```bash
python new/ab_testing.py \
  new/experiments/sarimax_multimodal_lstm/metrics.json \
  new/experiments/moe_multimodal_lstm/metrics.json \
  --metrics mae rmse pearson \
  --split val \
  --json-out new/experiments/ab_report.json
```
