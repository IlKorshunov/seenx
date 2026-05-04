from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID, _point_col, _safe_float, build_rows_with_targets_source, make_feature_matrix, select_train_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LOO-модель: CatBoost Quantile-median регрессор с консервативным "
            "blend-ом к baseline. Quantile(0.5) даёт робастный к выбросам прогноз, "
            "blend с baseline сохраняет форму кривой."
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
    parser.add_argument("--output-dir", default="blended_quantile_experiment")

    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--baseline-weight", type=float, default=0.35, help="Вес baseline в финальном прогнозе [0..1]. Остальное — модель.")
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


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    X_train = make_feature_matrix(train_df)
    X_test = make_feature_matrix(test_df)
    if X_train.empty:
        raise RuntimeError("Пустая матрица признаков для blended_quantile")

    y_train = _get_target_matrix(train_df, args.curve_points)
    y_true = _get_target_matrix(test_df, args.curve_points)[0]
    total_points = int(args.curve_points)

    baseline_curve = _clamp01(np.mean(y_train, axis=0))

    alpha = float(np.clip(args.baseline_weight, 0.0, 1.0))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "point_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_pred = np.zeros((total_points,), dtype=float)
    model_count = 0

    print(f"[blended_quantile] stage=quantile_points total={total_points} alpha={alpha:.2f}")
    for p in range(total_points):
        print(f"[blended_quantile][point] {p + 1}/{total_points}")
        target = y_train[:, p]

        if target.size == 0:
            model_pred[p] = baseline_curve[p]
        elif float(np.var(target)) < 1e-12:
            model_pred[p] = float(target[0])
        else:
            model = CatBoostRegressor(
                loss_function="Quantile:alpha=0.5",
                eval_metric="RMSE",
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
                random_seed=args.random_seed + p,
                task_type=getattr(args, "task_type", "GPU"),
                gpu_ram_part=getattr(args, "gpu_ram_part", 0.6),
                verbose=False,
            )
            model.fit(X_train, target)
            model_pred[p] = float(model.predict(X_test)[0])
            model.save_model(str(models_dir / f"quantile_point_{p:03d}.cbm"))
            model_count += 1

    model_pred = _clamp01(model_pred)

    y_pred = _clamp01(alpha * baseline_curve + (1.0 - alpha) * model_pred)

    abs_err = np.abs(y_pred - y_true)
    cm = _curve_metrics(y_pred, y_true)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(total_points)),
            "baseline_value": baseline_curve,
            "model_quantile_pred": model_pred,
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    dataset_path = out_dir / "blended_quantile_dataset.csv"
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    metrics_path = out_dir / "metrics.json"

    all_df.to_csv(dataset_path, index=False)
    result_df.to_csv(pred_path, index=False)

    metrics: dict[str, Any] = {
        "videos_total_with_target": len(rows),
        "videos_used": len(all_df),
        "train_videos": len(train_df),
        "curve_points": total_points,
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(test_df.iloc[0]["drive_file_id"]),
        "baseline_weight": alpha,
        "model_count": model_count,
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
        "models_dir": str(models_dir),
        **cm,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Blended-Quantile LOO ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\n=== Holdout Prediction vs True ({total_points} points) ===")
    print(result_df.to_string(index=False, formatters={"pred_retention": lambda x: f"{x:0.5f}", "true_retention": lambda x: f"{x:0.5f}", "abs_error": lambda x: f"{x:0.5f}"}))
    return metrics


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
