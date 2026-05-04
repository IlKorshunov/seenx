from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from train.common.train_utils import clamp01 as _clamp01
from train.common.train_utils import curve_metrics as _curve_metrics
from train.common.train_utils import get_target_matrix as _get_target_matrix
from train.common.train_utils import kfold_indices as _kfold_indices
from train.common.train_utils import make_feature_matrix
from train.loo.train_retention_regressor_loo import build_rows_with_targets_source, select_train_test
from train.tools.drive_feature_labeling_pipeline import DEFAULT_PARENT_FOLDER_ID


DEFAULT_CONFIG: dict[str, Any] = {
    "env_file": ".env",
    "snapshot_dir": "",
    "root_folder_id": DEFAULT_PARENT_FOLDER_ID,
    "limit_videos": 90,
    "train_videos": 89,
    "curve_points": 50,
    "eval_video_folder": "",
    "eval_drive_file_id": "",
    "output_dir": "local_knn_experiment",
    "neighbors_k": 12,
    "distance_temperature": 1.0,
    "residual_strength": 0.65,
    "spike_gain": 1.05,
    "smooth_window": 7,
    "auto_tune": False,
    "tune_folds": 5,
}


def default_args() -> SimpleNamespace:
    return SimpleNamespace(**DEFAULT_CONFIG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local kNN residual LOO retention experiment.")
    parser.add_argument("--env-file", default=DEFAULT_CONFIG["env_file"])
    parser.add_argument("--snapshot-dir", default=DEFAULT_CONFIG["snapshot_dir"])
    parser.add_argument("--root-folder-id", default=DEFAULT_CONFIG["root_folder_id"])
    parser.add_argument("--limit-videos", type=int, default=DEFAULT_CONFIG["limit_videos"])
    parser.add_argument("--train-videos", type=int, default=DEFAULT_CONFIG["train_videos"])
    parser.add_argument("--curve-points", type=int, default=DEFAULT_CONFIG["curve_points"])
    parser.add_argument("--eval-video-folder", default=DEFAULT_CONFIG["eval_video_folder"])
    parser.add_argument("--eval-drive-file-id", default=DEFAULT_CONFIG["eval_drive_file_id"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--neighbors-k", type=int, default=DEFAULT_CONFIG["neighbors_k"])
    parser.add_argument("--distance-temperature", type=float, default=DEFAULT_CONFIG["distance_temperature"])
    parser.add_argument("--residual-strength", type=float, default=DEFAULT_CONFIG["residual_strength"])
    parser.add_argument("--spike-gain", type=float, default=DEFAULT_CONFIG["spike_gain"])
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_CONFIG["smooth_window"])
    parser.add_argument("--auto-tune", action="store_true", default=DEFAULT_CONFIG["auto_tune"])
    parser.add_argument("--tune-folds", type=int, default=DEFAULT_CONFIG["tune_folds"])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    return parser.parse_args()


def _rolling_mean_1d(curve: np.ndarray, window: int) -> np.ndarray:
    w = int(max(3, window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    if curve.size == 0:
        return curve.astype(float)
    padded = np.pad(curve.astype(float), (pad, pad), mode="edge")
    kernel = np.ones((w,), dtype=float) / float(w)
    return np.convolve(padded, kernel, mode="valid")


def _robust_standardize(x_train: np.ndarray, x_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x_train, axis=0)
    q75 = np.nanpercentile(x_train, 75.0, axis=0)
    q25 = np.nanpercentile(x_train, 25.0, axis=0)
    iqr = q75 - q25
    scale = np.where(iqr > 1e-9, iqr, 1.0)
    x_train_z = (x_train - med) / scale
    x_pred_z = (x_pred - med) / scale
    x_train_z = np.nan_to_num(x_train_z, nan=0.0, posinf=0.0, neginf=0.0)
    x_pred_z = np.nan_to_num(x_pred_z, nan=0.0, posinf=0.0, neginf=0.0)
    return x_train_z, x_pred_z


def _pca_project(x_train: np.ndarray, x_pred: np.ndarray, max_components: int = 24) -> tuple[np.ndarray, np.ndarray]:
    n_train, n_features = x_train.shape
    if n_train == 0 or n_features == 0:
        return x_train, x_pred
    n_comp = int(max(2, min(max_components, n_train - 1, n_features)))
    if n_comp <= 0:
        return x_train, x_pred
    mean = np.mean(x_train, axis=0, keepdims=True)
    centered_train = x_train - mean
    centered_pred = x_pred - mean
    try:
        _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
    except np.linalg.LinAlgError:
        return x_train, x_pred
    basis = vt[:n_comp].T
    return centered_train @ basis, centered_pred @ basis


def _neighbor_weights(distances: np.ndarray, temperature: float) -> np.ndarray:
    if distances.size == 0:
        return distances
    temp = float(max(1e-6, temperature))
    d_nonzero = distances[distances > 1e-12]
    sigma = float(np.median(d_nonzero)) if d_nonzero.size else 1.0
    sigma = max(sigma, 1e-6)
    scaled = distances / sigma
    w = np.exp(-((scaled**2) / (2.0 * temp)))
    if float(np.sum(w)) <= 1e-12:
        w = np.ones_like(distances, dtype=float)
    return w / float(np.sum(w))


def _pointwise_volatility(curves: np.ndarray) -> np.ndarray:
    if curves.shape[1] < 3:
        return np.zeros((curves.shape[1],), dtype=float)
    # Локальная "неровность" по второй разности: помогает усилить реальный локальный shape.
    second_diff = np.abs(curves[:, 2:] - 2.0 * curves[:, 1:-1] + curves[:, :-2])
    vol_core = np.mean(second_diff, axis=0)
    vol = np.zeros((curves.shape[1],), dtype=float)
    vol[1:-1] = vol_core
    return vol


def _predict_one_local_knn_residual(
    x_train: np.ndarray, y_train: np.ndarray, x_test_vec: np.ndarray, neighbors_k: int, distance_temperature: float, residual_strength: float, spike_gain: float, smooth_window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = int(max(3, min(neighbors_k, len(x_train))))
    dists = np.sqrt(np.sum((x_train - x_test_vec) ** 2, axis=1))
    order = np.argsort(dists)[:k]
    d_top = dists[order]
    y_top = y_train[order]
    w = _neighbor_weights(d_top, temperature=distance_temperature)

    base_curve = np.sum(y_top * w[:, None], axis=0)
    smooth_top = np.vstack([_rolling_mean_1d(c, smooth_window) for c in y_top])
    residual_top = y_top - smooth_top
    residual_curve = np.sum(residual_top * w[:, None], axis=0)

    vol_top = _pointwise_volatility(y_top)
    if float(np.max(vol_top)) > 1e-9:
        vol_norm = vol_top / float(np.max(vol_top))
    else:
        vol_norm = np.zeros_like(vol_top)
    local_residual_scale = np.clip(0.25 + vol_norm, 0.25, 1.0)

    pred = base_curve + float(np.clip(residual_strength, 0.0, 1.5)) * local_residual_scale * residual_curve
    pred = _clamp01(pred)

    # Лёгкая коррекция дельт, чтобы не терять резкие локальные изменения формы.
    if pred.size >= 2:
        delta_base = np.diff(base_curve)
        delta_pred = np.diff(pred)
        blend = np.clip(0.25 + 0.5 * vol_norm[1:], 0.25, 0.8)
        boosted_delta = (1.0 - blend) * delta_pred + blend * float(max(0.2, spike_gain)) * delta_base
        out = np.zeros_like(pred)
        out[0] = pred[0]
        for i in range(1, pred.size):
            out[i] = out[i - 1] + boosted_delta[i - 1]
        pred = _clamp01(out)

    return _clamp01(pred), _clamp01(base_curve), residual_curve


def _predict_local_knn_residual(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, neighbors_k: int, distance_temperature: float, residual_strength: float, spike_gain: float, smooth_window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x_test) == 0:
        points = y_train.shape[1]
        return (np.zeros((0, points), dtype=float), np.zeros((0, points), dtype=float), np.zeros((0, points), dtype=float))
    preds: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for i in range(len(x_test)):
        pred, base, residual = _predict_one_local_knn_residual(
            x_train=x_train,
            y_train=y_train,
            x_test_vec=x_test[i],
            neighbors_k=neighbors_k,
            distance_temperature=distance_temperature,
            residual_strength=residual_strength,
            spike_gain=spike_gain,
            smooth_window=smooth_window,
        )
        preds.append(pred)
        bases.append(base)
        residuals.append(residual)
    return np.vstack(preds), np.vstack(bases), np.vstack(residuals)


def _build_tune_grid(train_size: int) -> list[dict[str, float | int]]:
    ks = sorted(set([max(4, min(train_size - 1, k)) for k in (8, 12, 16)]))
    temps = [0.8, 1.0, 1.3]
    residuals = [0.45, 0.70, 0.95]
    spikes = [1.0, 1.15, 1.30]
    windows = [5, 7, 9]
    grid: list[dict[str, float | int]] = []
    for k in ks:
        for t in temps:
            for r in residuals:
                for s in spikes:
                    for w in windows:
                        # Ограничиваем общее число конфигураций осмысленными сочетаниями.
                        if (r <= 0.5 and s >= 1.25) or (r >= 0.9 and s <= 1.0):
                            continue
                        grid.append({"neighbors_k": int(k), "distance_temperature": float(t), "residual_strength": float(r), "spike_gain": float(s), "smooth_window": int(w)})
    return grid


def _auto_tune_params(x_train: np.ndarray, y_train: np.ndarray, seed: int, folds: int) -> tuple[dict[str, float | int], list[dict[str, float | int | float]]]:
    n = len(x_train)
    folds = int(max(2, min(folds, n)))
    grid = _build_tune_grid(train_size=n)
    fold_idx = _kfold_indices(n, folds, seed)
    trials: list[dict[str, float | int | float]] = []
    best_cfg = grid[0]
    best_loss = float("inf")

    for cfg in grid:
        losses: list[float] = []
        rmse_vals: list[float] = []
        spike_vals: list[float] = []
        curve_vals: list[float] = []
        for tr_idx, va_idx in fold_idx:
            x_tr = x_train[tr_idx]
            y_tr = y_train[tr_idx]
            x_va = x_train[va_idx]
            y_va = y_train[va_idx]
            y_pred, _, _ = _predict_local_knn_residual(
                x_train=x_tr,
                y_train=y_tr,
                x_test=x_va,
                neighbors_k=int(cfg["neighbors_k"]),
                distance_temperature=float(cfg["distance_temperature"]),
                residual_strength=float(cfg["residual_strength"]),
                spike_gain=float(cfg["spike_gain"]),
                smooth_window=int(cfg["smooth_window"]),
            )
            for i in range(len(y_va)):
                m = _curve_metrics(y_pred[i], y_va[i])
                loss = float(m["rmse"] + 0.9 * m["spike_rmse"] + 0.4 * m["curvature_rmse"])
                losses.append(loss)
                rmse_vals.append(float(m["rmse"]))
                spike_vals.append(float(m["spike_rmse"]))
                curve_vals.append(float(m["curvature_rmse"]))
        mean_loss = float(np.mean(losses)) if losses else float("inf")
        rec: dict[str, float | int | float] = {
            **cfg,
            "cv_loss": mean_loss,
            "cv_rmse": float(np.mean(rmse_vals)) if rmse_vals else float("inf"),
            "cv_spike_rmse": float(np.mean(spike_vals)) if spike_vals else float("inf"),
            "cv_curvature_rmse": float(np.mean(curve_vals)) if curve_vals else float("inf"),
        }
        trials.append(rec)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_cfg = cfg

    trials.sort(key=lambda x: float(x["cv_loss"]))
    return best_cfg, trials


def run_experiment(args: Any) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    x_train_df = make_feature_matrix(train_df).reset_index(drop=True)
    x_test_df = make_feature_matrix(test_df).reset_index(drop=True)
    if x_train_df.empty:
        raise RuntimeError("Пустая матрица признаков для local_kNN")

    x_train = np.nan_to_num(x_train_df.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test_df.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_train = _get_target_matrix(train_df, args.curve_points)
    y_true = _get_target_matrix(test_df, args.curve_points)[0]

    x_train_z, x_test_z = _robust_standardize(x_train, x_test)
    x_train_p, x_test_p = _pca_project(x_train_z, x_test_z)

    if bool(args.auto_tune):
        best_cfg, trials = _auto_tune_params(x_train=x_train_p, y_train=y_train, seed=42, folds=int(args.tune_folds))
    else:
        best_cfg = {
            "neighbors_k": int(args.neighbors_k),
            "distance_temperature": float(args.distance_temperature),
            "residual_strength": float(args.residual_strength),
            "spike_gain": float(args.spike_gain),
            "smooth_window": int(args.smooth_window),
        }
        trials = []

    pred_mat, base_mat, resid_mat = _predict_local_knn_residual(
        x_train=x_train_p,
        y_train=y_train,
        x_test=x_test_p,
        neighbors_k=int(best_cfg["neighbors_k"]),
        distance_temperature=float(best_cfg["distance_temperature"]),
        residual_strength=float(best_cfg["residual_strength"]),
        spike_gain=float(best_cfg["spike_gain"]),
        smooth_window=int(best_cfg["smooth_window"]),
    )
    y_pred = pred_mat[0]
    y_base = base_mat[0]
    residual_component = resid_mat[0]

    abs_err = np.abs(y_pred - y_true)
    result_df = pd.DataFrame(
        {
            "point_idx": list(range(args.curve_points)),
            "point_frac": np.linspace(0.0, 1.0, args.curve_points),
            "pred_retention_base_knn": y_base,
            "pred_retention_residual_component": residual_component,
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "local_knn_dataset.csv"
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    metrics_path = out_dir / "metrics.json"
    tune_path = out_dir / "tuning_trials.json"
    all_df.to_csv(dataset_path, index=False)
    result_df.to_csv(pred_path, index=False)

    curve_metrics = _curve_metrics(y_pred, y_true)
    metrics = {
        "videos_total_with_target": len(rows),
        "videos_used": len(all_df),
        "train_videos": len(train_df),
        "curve_points": int(args.curve_points),
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(test_df.iloc[0]["drive_file_id"]),
        "auto_tune": bool(args.auto_tune),
        "neighbors_k": int(best_cfg["neighbors_k"]),
        "distance_temperature": float(best_cfg["distance_temperature"]),
        "residual_strength": float(best_cfg["residual_strength"]),
        "spike_gain": float(best_cfg["spike_gain"]),
        "smooth_window": int(best_cfg["smooth_window"]),
        "spearman": curve_metrics["spearman"],
        "pearson": curve_metrics["pearson"],
        "rmse": curve_metrics["rmse"],
        "mae": curve_metrics["mae"],
        "spike_rmse": curve_metrics["spike_rmse"],
        "curvature_rmse": curve_metrics["curvature_rmse"],
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if trials:
        tune_path.write_text(json.dumps(trials, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Local-kNN Residual LOO ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\n=== Holdout Prediction vs True ({args.curve_points} points) ===")
    print(result_df.to_string(index=False, formatters={"pred_retention": lambda x: f"{x:0.5f}", "true_retention": lambda x: f"{x:0.5f}", "abs_error": lambda x: f"{x:0.5f}"}))
    return metrics


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
