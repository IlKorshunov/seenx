from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID, _point_col, _safe_float, build_rows_with_targets_source, make_feature_matrix, select_train_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LOO-модель: conservative CatBoost. Baseline подаётся как фича, "
            "экстремальная регуляризация заставляет модель отклоняться от baseline "
            "только при сильном сигнале. На типичных видео ≈ baseline."
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
    parser.add_argument("--output-dir", default="conservative_catboost_experiment")

    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
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


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    X_train_base = make_feature_matrix(train_df).reset_index(drop=True)
    X_test_base = make_feature_matrix(test_df).reset_index(drop=True)
    if X_train_base.empty:
        raise RuntimeError("Пустая матрица признаков для conservative_catboost")

    y_train = _get_target_matrix(train_df, args.curve_points)
    y_true = _get_target_matrix(test_df, args.curve_points)[0]
    total_points = int(args.curve_points)

    baseline_curve = _clamp01(np.mean(y_train, axis=0))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "point_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    y_pred = np.zeros((total_points,), dtype=float)
    model_count = 0

    print(f"[conservative_cb] stage=train points={total_points} iter={args.iterations} depth={args.depth} l2={args.l2_leaf_reg}")
    for p in range(total_points):
        print(f"[conservative_cb][point] {p + 1}/{total_points}")
        target = y_train[:, p]
        bl_val = float(baseline_curve[p])
        p_norm = float(p / max(1, total_points - 1))

        X_train_aug = X_train_base.copy()
        X_train_aug["__baseline_val__"] = bl_val
        X_train_aug["__point_norm__"] = p_norm

        X_test_aug = X_test_base.copy()
        X_test_aug["__baseline_val__"] = bl_val
        X_test_aug["__point_norm__"] = p_norm

        if target.size == 0:
            y_pred[p] = bl_val
        elif float(np.var(target)) < 1e-12:
            y_pred[p] = float(target[0])
        else:
            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
                l2_leaf_reg=args.l2_leaf_reg,
                random_seed=args.random_seed + p,
                task_type=getattr(args, "task_type", "GPU"),
                gpu_ram_part=getattr(args, "gpu_ram_part", 0.6),
                verbose=False,
            )
            train_pool = Pool(data=X_train_aug, label=target, baseline=np.full(len(target), bl_val))
            model.fit(train_pool)

            test_pool = Pool(data=X_test_aug, baseline=np.full(1, bl_val))
            y_pred[p] = float(model.predict(test_pool)[0])
            model.save_model(str(models_dir / f"conservative_point_{p:03d}.cbm"))
            model_count += 1

    y_pred = _clamp01(y_pred)

    abs_err = np.abs(y_pred - y_true)
    cm = _curve_metrics(y_pred, y_true)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(total_points)),
            "baseline_value": baseline_curve,
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    dataset_path = out_dir / "conservative_catboost_dataset.csv"
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
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "model_count": model_count,
        "baseline_rmse": float(np.sqrt(np.mean((baseline_curve - y_true) ** 2))),
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
        "models_dir": str(models_dir),
        **cm,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Conservative-CatBoost LOO ===")
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
