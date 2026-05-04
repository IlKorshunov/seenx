from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID, _point_col, _safe_float, build_rows_with_targets_source, make_feature_matrix, select_train_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LOO-модель: XGBoost в flattened-формате. Вместо 50 отдельных моделей "
            "по 89 строк — одна модель на 89×50=4450 строк. "
            "Фичи = video_features + point_position. "
            "Модель обучается на ВСЕХ точках сразу и видит межточечные паттерны."
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
    parser.add_argument("--output-dir", default="xgb_flat_experiment")

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--reg-alpha", type=float, default=0.5)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--random-seed", type=int, default=42)
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


def _flatten_to_rows(x_video: pd.DataFrame, y_matrix: np.ndarray, curve_points: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Раскладываем (n_videos, curve_points) → (n_videos * curve_points,) rows."""
    n_videos = len(x_video)
    rows = []
    targets = []
    for vid_idx in range(n_videos):
        base = x_video.iloc[vid_idx].to_dict()
        for p in range(curve_points):
            row = dict(base)
            row["__point_idx__"] = float(p)
            row["__point_norm__"] = float(p / max(1, curve_points - 1))
            row["__point_norm_sq__"] = row["__point_norm__"] ** 2
            rows.append(row)
            targets.append(float(y_matrix[vid_idx, p]))
    return pd.DataFrame(rows).fillna(0.0), np.array(targets, dtype=float)


def _flatten_test(x_video: pd.DataFrame, curve_points: int) -> pd.DataFrame:
    base = x_video.iloc[0].to_dict()
    rows = []
    for p in range(curve_points):
        row = dict(base)
        row["__point_idx__"] = float(p)
        row["__point_norm__"] = float(p / max(1, curve_points - 1))
        row["__point_norm_sq__"] = row["__point_norm__"] ** 2
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    X_train_video = make_feature_matrix(train_df).reset_index(drop=True)
    X_test_video = make_feature_matrix(test_df).reset_index(drop=True)
    if X_train_video.empty:
        raise RuntimeError("Пустая матрица признаков для xgb_flat")

    y_train_mat = _get_target_matrix(train_df, args.curve_points)
    y_true = _get_target_matrix(test_df, args.curve_points)[0]
    n_points = int(args.curve_points)
    n_train = len(X_train_video)

    X_train_flat, y_train_flat = _flatten_to_rows(X_train_video, y_train_mat, n_points)
    X_test_flat = _flatten_test(X_test_video, n_points)

    print(f"[xgb_flat] train_videos={n_train} points={n_points} flat_rows={len(X_train_flat)} features={X_train_flat.shape[1]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        random_state=args.random_seed,
        verbosity=0,
    )

    print("[xgb_flat] stage=fit")
    model.fit(X_train_flat, y_train_flat)

    print("[xgb_flat] stage=predict")
    y_pred_raw = model.predict(X_test_flat)
    y_pred = _clamp01(np.array(y_pred_raw, dtype=float))

    model.save_model(str(out_dir / "xgb_flat_model.json"))

    abs_err = np.abs(y_pred - y_true)
    cm = _curve_metrics(y_pred, y_true)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(n_points)),
            "point_frac": np.linspace(0.0, 1.0, n_points),
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    dataset_path = out_dir / "xgb_flat_dataset.csv"
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    metrics_path = out_dir / "metrics.json"

    all_df.to_csv(dataset_path, index=False)
    result_df.to_csv(pred_path, index=False)

    metrics: dict[str, Any] = {
        "videos_total_with_target": len(rows),
        "videos_used": len(all_df),
        "train_videos": n_train,
        "curve_points": n_points,
        "flat_train_rows": len(X_train_flat),
        "n_features": int(X_train_flat.shape[1]),
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(test_df.iloc[0]["drive_file_id"]),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
        **cm,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention XGBoost-Flat LOO ===")
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
