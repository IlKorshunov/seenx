"""
Shared utilities for LOO retention training scripts.

Consolidates duplicated helpers from:
  train_retention_regressor_loo, train_retention_ranker_loo,
  train_retention_local_knn_loo, train_retention_shape_only_loo,
  train_retention_stacked_loo, train_retention_hybrid_loo,
  compare_fixed_inference_models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def resample_series(values: list[float], points: int) -> list[float]:
    if points <= 0:
        return []
    if not values:
        return [0.0] * points
    if len(values) == 1:
        return [float(values[0])] * points
    out: list[float] = []
    n = len(values)
    for i in range(points):
        pos = (i * (n - 1)) / max(1, points - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        val = values[lo] * (1.0 - frac) + values[hi] * frac
        out.append(float(max(0.0, min(1.0, val))))
    return out


def point_col(idx: int) -> str:
    return f"target__retention_point__{idx:03d}"


def clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr.astype(float), 0.0, 1.0)


def get_target_matrix(df: pd.DataFrame, curve_points: int) -> np.ndarray:
    mat = np.zeros((len(df), curve_points), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        for p in range(curve_points):
            mat[i, p] = safe_float(row[point_col(p)], 0.0)
    return clamp01(mat)


def make_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    exclude_cols = {"video_folder", "transcript_path", "drive_file_id"}
    feature_cols = [c for c in df.columns if c not in exclude_cols and not str(c).startswith("target__")]
    return df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def curve_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    y_pred = clamp01(y_pred)
    y_true = clamp01(y_true)
    abs_err = np.abs(y_pred - y_true)
    d_pred = np.diff(y_pred)
    d_true = np.diff(y_true)
    dd_pred = np.diff(y_pred, n=2)
    dd_true = np.diff(y_true, n=2)
    spike_rmse = float(np.sqrt(np.mean((d_pred - d_true) ** 2))) if d_pred.size else 0.0
    curvature_rmse = float(np.sqrt(np.mean((dd_pred - dd_true) ** 2))) if dd_pred.size else 0.0
    return {
        "spearman": float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")),
        "pearson": float(pd.Series(y_pred).corr(pd.Series(y_true), method="pearson")),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(abs_err)),
        "spike_rmse": spike_rmse,
        "curvature_rmse": curvature_rmse,
    }


def kfold_indices(n: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    parts = np.array_split(idx, n_folds)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        val = parts[i]
        train = np.concatenate([parts[j] for j in range(n_folds) if j != i], axis=0)
        out.append((train, val))
    return out


def enforce_non_increasing(curve: np.ndarray) -> np.ndarray:
    out = curve.astype(float).copy()
    for i in range(1, len(out)):
        if out[i] > out[i - 1]:
            out[i] = out[i - 1]
    return out


def anchor_mean(curve: np.ndarray, k: int) -> float:
    if curve.size == 0:
        return 0.0
    kk = max(1, min(int(k), int(curve.size)))
    return float(np.mean(curve[:kk]))


def tail_mean(curve: np.ndarray, k: int) -> float:
    if curve.size == 0:
        return 0.0
    kk = max(1, min(int(k), int(curve.size)))
    return float(np.mean(curve[-kk:]))


def shape_from_curve(curve: np.ndarray, anchor_points: int, eps: float = 1e-6) -> np.ndarray:
    base = max(eps, anchor_mean(curve, anchor_points))
    shape = curve.astype(float) / base
    if shape.size > 0:
        shape[0] = 1.0
    return shape


def print_experiment_results(title: str, metrics: dict[str, Any], result_df: pd.DataFrame, curve_points: int) -> None:
    print(f"=== {title} ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\n=== Holdout Prediction vs True ({curve_points} points) ===")
    formatters = {}
    for col in ("pred_retention", "pred_retention_norm", "pred_score_raw", "true_retention", "abs_error"):
        if col in result_df.columns:
            formatters[col] = lambda x: f"{x:0.5f}"
    print(result_df.to_string(index=False, formatters=formatters))
