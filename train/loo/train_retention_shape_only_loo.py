from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train.common.train_utils import clamp01 as _clamp01
from train.common.train_utils import curve_metrics as _curve_metrics
from train.common.train_utils import get_target_matrix as _get_target_matrix
from train.common.train_utils import make_feature_matrix
from train.common.train_utils import shape_from_curve as _shape_from_curve
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
    "output_dir": "shape_only_experiment",
    "iterations": 700,
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": 42,
    "shape_anchor_points": 1,
    "fixed_anchor_value": 1.0,
    "delta_max_step": 0.18,
    "curvature_max_step": 0.12,
    "curvature_blend": 0.55,
    "shape_max": 1.15,
    "spike_sensitivity": 1.35,
    "spike_threshold_quantile": 0.72,
    "spike_max_amplify": 1.7,
}


def default_args() -> SimpleNamespace:
    return SimpleNamespace(**DEFAULT_CONFIG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shape-only LOO retention experiment.")
    parser.add_argument("--env-file", default=DEFAULT_CONFIG["env_file"])
    parser.add_argument("--snapshot-dir", default=DEFAULT_CONFIG["snapshot_dir"])
    parser.add_argument("--root-folder-id", default=DEFAULT_CONFIG["root_folder_id"])
    parser.add_argument("--limit-videos", type=int, default=DEFAULT_CONFIG["limit_videos"])
    parser.add_argument("--train-videos", type=int, default=DEFAULT_CONFIG["train_videos"])
    parser.add_argument("--curve-points", type=int, default=DEFAULT_CONFIG["curve_points"])
    parser.add_argument("--eval-video-folder", default=DEFAULT_CONFIG["eval_video_folder"])
    parser.add_argument("--eval-drive-file-id", default=DEFAULT_CONFIG["eval_drive_file_id"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--iterations", type=int, default=DEFAULT_CONFIG["iterations"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--depth", type=int, default=DEFAULT_CONFIG["depth"])
    parser.add_argument("--random-seed", type=int, default=DEFAULT_CONFIG["random_seed"])
    parser.add_argument("--shape-anchor-points", type=int, default=DEFAULT_CONFIG["shape_anchor_points"])
    parser.add_argument("--fixed-anchor-value", type=float, default=DEFAULT_CONFIG["fixed_anchor_value"])
    parser.add_argument("--delta-max-step", type=float, default=DEFAULT_CONFIG["delta_max_step"])
    parser.add_argument("--curvature-max-step", type=float, default=DEFAULT_CONFIG["curvature_max_step"])
    parser.add_argument("--curvature-blend", type=float, default=DEFAULT_CONFIG["curvature_blend"])
    parser.add_argument("--shape-max", type=float, default=DEFAULT_CONFIG["shape_max"])
    parser.add_argument("--spike-sensitivity", type=float, default=DEFAULT_CONFIG["spike_sensitivity"])
    parser.add_argument("--spike-threshold-quantile", type=float, default=DEFAULT_CONFIG["spike_threshold_quantile"])
    parser.add_argument("--spike-max-amplify", type=float, default=DEFAULT_CONFIG["spike_max_amplify"])
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    return parser.parse_args()


def _fit_regressor(
    x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, iterations: int, learning_rate: float, depth: int, seed: int, task_type: str, gpu_ram_part: float
) -> np.ndarray:
    if y_train.size == 0:
        return np.zeros((len(x_test),), dtype=float)
    if float(np.var(y_train)) < 1e-12:
        return np.full((len(x_test),), float(y_train[0]), dtype=float)
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=seed,
        task_type=task_type,
        gpu_ram_part=gpu_ram_part,
        verbose=False,
    )
    model.fit(x_train, y_train)
    return model.predict(x_test).astype(float)


def _spike_profile_from_train_shapes(y_train_shape: np.ndarray) -> np.ndarray:
    n_points = y_train_shape.shape[1]
    if n_points <= 1:
        return np.ones((n_points,), dtype=float)
    d = np.diff(y_train_shape, axis=1)
    vol = np.std(d, axis=0)
    q = float(np.quantile(vol, 0.9)) if vol.size else 0.0
    if q <= 1e-9:
        s = np.ones_like(vol, dtype=float)
    else:
        s = np.clip(vol / q, 0.25, 2.5)
    out = np.ones((n_points,), dtype=float)
    out[1:] = s
    return out


def run_experiment(args) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    x_train = make_feature_matrix(train_df)
    x_test = make_feature_matrix(test_df)
    if x_train.empty:
        raise RuntimeError("Пустая матрица признаков для shape-only")

    y_train_abs = _get_target_matrix(train_df, args.curve_points)
    y_true = _get_target_matrix(test_df, args.curve_points)[0]
    n_points = int(args.curve_points)

    # Shape-only target: нормируем каждую кривую на её стартовый anchor.
    y_train_shape = np.zeros_like(y_train_abs, dtype=float)
    for i in range(len(y_train_abs)):
        y_train_shape[i] = _shape_from_curve(y_train_abs[i], anchor_points=args.shape_anchor_points)

    pred_delta = np.zeros((n_points,), dtype=float)
    pred_curv = np.zeros((n_points,), dtype=float)
    delta_max = float(max(1e-4, args.delta_max_step))
    curv_max = float(max(1e-4, args.curvature_max_step))
    spike_profile = _spike_profile_from_train_shapes(y_train_shape)
    spike_sensitivity = float(max(0.0, args.spike_sensitivity))
    spike_q = float(np.clip(args.spike_threshold_quantile, 0.0, 0.99))
    spike_max_amplify = float(max(1.0, args.spike_max_amplify))

    for p in range(1, n_points):
        y_delta_train = y_train_shape[:, p] - y_train_shape[:, p - 1]
        pred_d = float(
            _fit_regressor(
                x_train=x_train,
                y_train=y_delta_train,
                x_test=x_test,
                iterations=int(args.iterations),
                learning_rate=float(args.learning_rate),
                depth=int(args.depth),
                seed=int(args.random_seed) + 1000 + p,
                task_type=getattr(args, "task_type", "GPU"),
                gpu_ram_part=float(getattr(args, "gpu_ram_part", 0.6)),
            )[0]
        )
        pred_delta[p] = float(np.clip(pred_d, -delta_max, delta_max))

    for p in range(2, n_points):
        y_curv_train = y_train_shape[:, p] - 2.0 * y_train_shape[:, p - 1] + y_train_shape[:, p - 2]
        pred_c = float(
            _fit_regressor(
                x_train=x_train,
                y_train=y_curv_train,
                x_test=x_test,
                iterations=int(args.iterations),
                learning_rate=float(args.learning_rate),
                depth=max(4, int(args.depth)),
                seed=int(args.random_seed) + 2000 + p,
                task_type=getattr(args, "task_type", "GPU"),
                gpu_ram_part=float(getattr(args, "gpu_ram_part", 0.6)),
            )[0]
        )
        pred_curv[p] = float(np.clip(pred_c, -curv_max, curv_max))

    curv_blend = float(np.clip(args.curvature_blend, 0.0, 1.0))
    pred_shape = np.zeros((n_points,), dtype=float)
    pred_shape[0] = 1.0
    if n_points >= 2:
        boost_1 = 1.0 + spike_sensitivity * max(0.0, spike_profile[1] - spike_q) / max(1e-6, 1.0 - spike_q)
        boost_1 = float(np.clip(boost_1, 1.0, spike_max_amplify))
        d1 = float(np.clip(pred_delta[1] * boost_1, -delta_max * boost_1, delta_max * boost_1))
        pred_shape[1] = pred_shape[0] + d1
    for p in range(2, n_points):
        d_from_curv = (pred_shape[p - 1] - pred_shape[p - 2]) + pred_curv[p]
        d_mix = (1.0 - curv_blend) * pred_delta[p] + curv_blend * d_from_curv
        spike_boost = 1.0 + spike_sensitivity * max(0.0, spike_profile[p] - spike_q) / max(1e-6, 1.0 - spike_q)
        spike_boost = float(np.clip(spike_boost, 1.0, spike_max_amplify))
        d_mix = float(np.clip(d_mix * spike_boost, -delta_max * spike_boost, delta_max * spike_boost))
        pred_shape[p] = pred_shape[p - 1] + d_mix

    shape_max = float(max(1.0, args.shape_max))
    pred_shape = np.clip(pred_shape, 0.0, shape_max)

    # Абсолютный уровень не обучаем: фиксируем anchor (обычно 1.0 в начале ролика).
    anchor_value = float(np.clip(args.fixed_anchor_value, 0.0, 1.0))
    y_pred = _clamp01(pred_shape * anchor_value)
    abs_err = np.abs(y_pred - y_true)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(n_points)),
            "point_frac": np.linspace(0.0, 1.0, n_points),
            "pred_shape_only": pred_shape,
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "shape_only_dataset.csv"
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    metrics_path = out_dir / "metrics.json"
    all_df.to_csv(dataset_path, index=False)
    result_df.to_csv(pred_path, index=False)

    curve_metrics = _curve_metrics(y_pred, y_true)
    metrics = {
        "videos_total_with_target": len(rows),
        "videos_used": len(all_df),
        "train_videos": len(train_df),
        "curve_points": int(n_points),
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(test_df.iloc[0]["drive_file_id"]),
        "shape_anchor_points": int(args.shape_anchor_points),
        "fixed_anchor_value": anchor_value,
        "delta_max_step": delta_max,
        "curvature_max_step": curv_max,
        "curvature_blend": curv_blend,
        "shape_max": shape_max,
        "spike_sensitivity": spike_sensitivity,
        "spike_threshold_quantile": spike_q,
        "spike_max_amplify": spike_max_amplify,
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

    print("=== Retention Shape-Only LOO ===")
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
