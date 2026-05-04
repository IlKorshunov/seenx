"""Transformer-only helpers: tuned feature filtering, permutation importance."""

from __future__ import annotations

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from train.common.seq_data_utils import predict_video
from train.lstm.lstm_seq_base import COLOR_ACTUAL, GRID_ALPHA, save_figure


logger = logging.getLogger(__name__)


def build_tuned_feature_filter_kwargs(args: Any) -> dict[str, Any]:
    """Merge CLI top_k with optional tuned_* thresholds from Optuna JSON."""
    filter_kw: dict[str, Any] = {"top_k": args.top_k_features or None}
    if hasattr(args, "tuned_corr_threshold"):
        filter_kw["redundant_corr_threshold"] = args.tuned_corr_threshold
    if hasattr(args, "tuned_nan_pct"):
        filter_kw["max_nan_pct"] = args.tuned_nan_pct
    if hasattr(args, "tuned_nonzero_pct"):
        filter_kw["min_nonzero_pct"] = args.tuned_nonzero_pct
    if hasattr(args, "tuned_top_k"):
        tk = args.tuned_top_k
        filter_kw["top_k"] = tk if tk and tk > 0 else None
    return filter_kw


def column_is_constant_across_val_videos(feature_name: str, video_dfs: dict, val_ids: list[str]) -> bool:
    for vid in val_ids:
        if feature_name not in video_dfs[vid].columns:
            continue
        values = video_dfs[vid][feature_name].dropna().values
        if len(values) > 1 and np.std(values) > 1e-8:
            return False
    return True


def compute_permutation_feature_importance(
    model: torch.nn.Module,
    feature_cols: list[str],
    video_dfs: dict,
    val_ids: list[str],
    normalizer,
    device: torch.device,
    out_dir: str,
    window_size: int,
    n_repeats: int = 5,
    time_feature_mode: str = "none",
    ref_time_sec_max: float = 1.0,
) -> None:
    model.eval()
    baseline_mae_by_video = {}
    for vid in val_ids:
        y_true, y_pred = predict_video(model, video_dfs[vid], feature_cols, normalizer, device, window_size, time_feature_mode=time_feature_mode, ref_time_sec_max=ref_time_sec_max)
        baseline_mae_by_video[vid] = float(np.mean(np.abs(y_pred - y_true)))
    baseline_mae = np.mean(list(baseline_mae_by_video.values()))
    rng = np.random.RandomState(42)
    importance = np.zeros(len(feature_cols))

    for feat_idx, col_name in enumerate(tqdm(feature_cols, desc="Permutation importance")):
        constant = column_is_constant_across_val_videos(col_name, video_dfs, val_ids)
        deltas = []
        for _ in range(n_repeats):
            if constant:
                per_video_first = {}
                for vid in val_ids:
                    if col_name in video_dfs[vid].columns:
                        vals = video_dfs[vid][col_name].dropna().values
                        per_video_first[vid] = float(vals[0]) if len(vals) else 0.0
                    else:
                        per_video_first[vid] = 0.0
                shuffled_vals = list(per_video_first.values())
                rng.shuffle(shuffled_vals)
                remap = dict(zip(per_video_first.keys(), shuffled_vals, strict=True))
                mae_list = []
                for vid in val_ids:
                    sdf = video_dfs[vid].copy()
                    if col_name in sdf.columns:
                        sdf[col_name] = remap[vid]
                    _, pred = predict_video(model, sdf, feature_cols, normalizer, device, window_size, time_feature_mode=time_feature_mode, ref_time_sec_max=ref_time_sec_max)
                    true_y = video_dfs[vid]["retention"].values
                    mae_list.append(float(np.mean(np.abs(pred - true_y))))
            else:
                mae_list = []
                for vid in val_ids:
                    sdf = video_dfs[vid].copy()
                    if col_name not in sdf.columns:
                        continue
                    sdf[col_name] = rng.permutation(sdf[col_name].values)
                    _, pred = predict_video(model, sdf, feature_cols, normalizer, device, window_size, time_feature_mode=time_feature_mode, ref_time_sec_max=ref_time_sec_max)
                    true_y = video_dfs[vid]["retention"].values
                    mae_list.append(float(np.mean(np.abs(pred - true_y))))
            deltas.append(np.mean(mae_list) - baseline_mae)
        importance[feat_idx] = np.mean(deltas)

    ranking = sorted(zip(feature_cols, importance, strict=True), key=lambda item: -item[1])
    pd.DataFrame(ranking, columns=["feature", "importance_mae_increase"]).to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
    top_n = min(30, len(ranking))
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    names, vals = zip(*ranking[:top_n], strict=True)
    ax.barh(names[::-1], vals[::-1], color=COLOR_ACTUAL)
    ax.set(xlabel="MAE increase when shuffled", title=f"Permutation Feature Importance (top {top_n})")
    ax.grid(True, alpha=GRID_ALPHA, axis="x")
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "feature_importance.png"))
    logger.info("Feature importance: %d features, baseline MAE=%.4f", len(ranking), baseline_mae)
