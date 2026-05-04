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
            "LOO-эксперимент: residual-от-baseline регрессор с Huber loss. Модель предсказывает отклонение кривой от среднего по train, а не абсолютное значение retention."
        )
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--snapshot-dir", default="")
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    parser.add_argument("--limit-videos", type=int, default=45)
    parser.add_argument("--train-videos", type=int, default=44)
    parser.add_argument("--curve-points", type=int, default=50)
    parser.add_argument("--eval-video-folder", default="")
    parser.add_argument("--eval-drive-file-id", default="")
    parser.add_argument("--output-dir", default="residual_huber_experiment")

    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--delta-blend", type=float, default=0.65, help="Вес delta-residual пути в финальном blend [0..1].")
    parser.add_argument("--delta-max-step", type=float, default=0.15, help="Clamp для предсказанной дельты residual между соседними точками.")
    parser.add_argument("--huber-delta", type=float, default=0.03, help="Параметр delta для Huber loss основных point-wise моделей.")
    parser.add_argument("--huber-delta-delta", type=float, default=0.015, help="Параметр delta для Huber loss delta-моделей.")
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
        raise RuntimeError("Пустая матрица признаков для residual_huber")

    y_train_mat = _get_target_matrix(train_df, args.curve_points)
    y_true = _get_target_matrix(test_df, args.curve_points)[0]
    total_points = int(args.curve_points)

    baseline_curve = _clamp01(np.mean(y_train_mat, axis=0))

    residual_train = y_train_mat - baseline_curve[np.newaxis, :]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "point_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    huber_delta = float(max(1e-6, args.huber_delta))
    huber_delta_delta = float(max(1e-6, args.huber_delta_delta))
    delta_max_step = float(max(1e-6, args.delta_max_step))

    pred_residual_base = np.zeros((total_points,), dtype=float)
    point_model_count = 0

    print(f"[residual_huber] stage=residual_points total={total_points}")
    for p in range(total_points):
        print(f"[residual_huber][residual] {p + 1}/{total_points}")
        target = residual_train[:, p]

        if target.size == 0:
            pred_residual_base[p] = 0.0
        elif float(np.var(target)) < 1e-12:
            pred_residual_base[p] = float(target[0])
        else:
            model = CatBoostRegressor(
                loss_function=f"Huber:delta={huber_delta}",
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
            pred_residual_base[p] = float(model.predict(X_test)[0])
            model.save_model(str(models_dir / f"residual_point_{p:03d}.cbm"))
            point_model_count += 1

    print(f"[residual_huber] stage=delta_residual total={max(0, total_points - 1)}")
    delta_residual_curve = np.zeros((total_points,), dtype=float)
    delta_residual_curve[0] = pred_residual_base[0]
    delta_model_count = 0

    for p in range(1, total_points):
        print(f"[residual_huber][delta] {p}/{max(1, total_points - 1)}")
        target_delta = residual_train[:, p] - residual_train[:, p - 1]

        if target_delta.size == 0:
            pred_d = 0.0
        elif float(np.var(target_delta)) < 1e-12:
            pred_d = float(target_delta[0])
        else:
            delta_model = CatBoostRegressor(
                loss_function=f"Huber:delta={huber_delta_delta}",
                eval_metric="RMSE",
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
                random_seed=args.random_seed + 5000 + p,
                task_type=getattr(args, "task_type", "GPU"),
                gpu_ram_part=getattr(args, "gpu_ram_part", 0.6),
                verbose=False,
            )
            delta_model.fit(X_train, target_delta)
            pred_d = float(delta_model.predict(X_test)[0])
            delta_model.save_model(str(models_dir / f"residual_delta_{p:03d}.cbm"))
            delta_model_count += 1

        pred_d = float(np.clip(pred_d, -delta_max_step, delta_max_step))
        delta_residual_curve[p] = delta_residual_curve[p - 1] + pred_d

    blend = float(np.clip(args.delta_blend, 0.0, 1.0))
    final_residual = (1.0 - blend) * pred_residual_base + blend * delta_residual_curve
    y_pred = _clamp01(baseline_curve + final_residual)

    abs_err = np.abs(y_pred - y_true)
    cm = _curve_metrics(y_pred, y_true)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(total_points)),
            "baseline_value": baseline_curve,
            "pred_residual_base": pred_residual_base,
            "pred_residual_delta_curve": delta_residual_curve,
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    dataset_path = out_dir / "residual_huber_dataset.csv"
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
        "huber_delta": huber_delta,
        "huber_delta_delta": huber_delta_delta,
        "delta_blend": blend,
        "delta_max_step": delta_max_step,
        "point_models_count": point_model_count,
        "delta_models_count": delta_model_count,
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
        "models_dir": str(models_dir),
        **cm,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Residual-Huber LOO ===")
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
