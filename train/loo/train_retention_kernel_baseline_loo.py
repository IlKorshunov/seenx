from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID, _point_col, _safe_float, build_rows_with_targets_source, make_feature_matrix, select_train_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LOO-модель: kernel baseline — взвешенное среднее train-кривых "
            "по feature-сходству. На типичных видео ≈ baseline, на уникальных — "
            "адаптируется к ближайшим соседям. Не может быть хуже baseline."
        )
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--snapshot-dir", default="")
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    parser.add_argument("--limit-videos", type=int, default=90)
    parser.add_argument("--train-videos", type=int, default=89)
    parser.add_argument("--curve-points", type=int, default=50)
    parser.add_argument("--eval-video-folder", default="")
    parser.add_argument("--eval-drive-file-id", default="")
    parser.add_argument("--output-dir", default="kernel_baseline_experiment")

    parser.add_argument("--temperature", type=float, default=1.0, help=("Температура ядра. Больше → ближе к uniform (baseline). Меньше → сильнее вес ближайших соседей."))
    parser.add_argument("--max-pca-components", type=int, default=20, help="Сколько PCA-компонент использовать для расстояний.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    return parser.parse_args()


def _clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr.astype(float), 0.0, 1.0)


def _get_target_matrix(df: pd.DataFrame, curve_points: int) -> np.ndarray:
    mat = np.zeros((len(df), curve_points), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        for p in range(curve_points):
            mat[i, p] = _safe_float(row[_point_col(p)], 0.0)
    return _clamp01(mat)


def _curve_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    y_pred = _clamp01(y_pred)
    y_true = _clamp01(y_true)
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


def _robust_standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x_train, axis=0)
    q75 = np.nanpercentile(x_train, 75.0, axis=0)
    q25 = np.nanpercentile(x_train, 25.0, axis=0)
    iqr = q75 - q25
    scale = np.where(iqr > 1e-9, iqr, 1.0)
    x_train_z = np.nan_to_num((x_train - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    x_test_z = np.nan_to_num((x_test - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    return x_train_z, x_test_z


def _pca_project(x_train: np.ndarray, x_test: np.ndarray, max_components: int) -> tuple[np.ndarray, np.ndarray]:
    n_train, n_features = x_train.shape
    if n_train == 0 or n_features == 0:
        return x_train, x_test
    n_comp = max(2, min(max_components, n_train - 1, n_features))
    mean = np.mean(x_train, axis=0, keepdims=True)
    centered_train = x_train - mean
    centered_test = x_test - mean
    try:
        _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
    except np.linalg.LinAlgError:
        return x_train, x_test
    basis = vt[:n_comp].T
    return centered_train @ basis, centered_test @ basis


def _kernel_weights(x_train: np.ndarray, x_test_vec: np.ndarray, temperature: float) -> np.ndarray:
    dists = np.sqrt(np.sum((x_train - x_test_vec) ** 2, axis=1))
    d_nonzero = dists[dists > 1e-12]
    sigma = float(np.median(d_nonzero)) if d_nonzero.size else 1.0
    sigma = max(sigma, 1e-6)
    temp = max(1e-6, temperature)
    scaled = dists / sigma
    w = np.exp(-(scaled**2) / (2.0 * temp))
    total = float(np.sum(w))
    if total <= 1e-12:
        return np.ones(len(x_train), dtype=float) / len(x_train)
    return w / total


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    x_train_df = make_feature_matrix(train_df).reset_index(drop=True)
    x_test_df = make_feature_matrix(test_df).reset_index(drop=True)
    if x_train_df.empty:
        raise RuntimeError("Пустая матрица признаков для kernel_baseline")

    x_train = np.nan_to_num(x_train_df.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test_df.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    y_train = _get_target_matrix(train_df, args.curve_points)
    y_true = _get_target_matrix(test_df, args.curve_points)[0]
    n_points = int(args.curve_points)

    x_train_z, x_test_z = _robust_standardize(x_train, x_test)
    x_train_p, x_test_p = _pca_project(x_train_z, x_test_z, args.max_pca_components)

    plain_baseline = _clamp01(np.mean(y_train, axis=0))

    print(f"[kernel_baseline] stage=predict temperature={args.temperature}")
    w = _kernel_weights(x_train_p, x_test_p[0], temperature=args.temperature)

    y_pred = _clamp01(np.sum(y_train * w[:, np.newaxis], axis=0))

    effective_n = float(1.0 / np.sum(w**2)) if float(np.sum(w**2)) > 1e-12 else len(w)
    max_weight = float(np.max(w))
    top5_weight = float(np.sum(np.sort(w)[-5:]))

    abs_err = np.abs(y_pred - y_true)
    cm = _curve_metrics(y_pred, y_true)
    cm_baseline = _curve_metrics(plain_baseline, y_true)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(n_points)),
            "point_frac": np.linspace(0.0, 1.0, n_points),
            "plain_baseline": plain_baseline,
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "kernel_baseline_dataset.csv"
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    metrics_path = out_dir / "metrics.json"

    all_df.to_csv(dataset_path, index=False)
    result_df.to_csv(pred_path, index=False)

    metrics: dict[str, Any] = {
        "videos_total_with_target": len(rows),
        "videos_used": len(all_df),
        "train_videos": len(train_df),
        "curve_points": n_points,
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(test_df.iloc[0]["drive_file_id"]),
        "temperature": float(args.temperature),
        "max_pca_components": int(args.max_pca_components),
        "effective_neighbors": round(effective_n, 1),
        "max_weight": round(max_weight, 4),
        "top5_weight_sum": round(top5_weight, 4),
        "baseline_rmse": cm_baseline["rmse"],
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
        **cm,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Kernel-Baseline LOO ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\n=== Holdout Prediction vs True ({n_points} points) ===")
    print(result_df.to_string(index=False, formatters={"pred_retention": lambda x: f"{x:0.5f}", "true_retention": lambda x: f"{x:0.5f}", "abs_error": lambda x: f"{x:0.5f}"}))
    return metrics


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
