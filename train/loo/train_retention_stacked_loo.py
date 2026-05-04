from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from tqdm import tqdm

from train.common.train_utils import anchor_mean as _anchor_mean
from train.common.train_utils import clamp01 as _clamp01
from train.common.train_utils import get_target_matrix as _get_target_matrix
from train.common.train_utils import kfold_indices as _kfold_indices
from train.common.train_utils import make_feature_matrix
from train.common.train_utils import shape_from_curve as _shape_from_curve
from train.common.train_utils import tail_mean as _tail_mean
from train.loo.train_retention_regressor_loo import build_rows_with_targets_source, select_train_test
from train.tools.drive_feature_labeling_pipeline import DEFAULT_PARENT_FOLDER_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Stacked retention architecture: 3 base experts (absolute, delta, shape+level) + ridge meta-learner."))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--snapshot-dir", default="")
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    parser.add_argument("--limit-videos", type=int, default=51)
    parser.add_argument("--train-videos", type=int, default=50)
    parser.add_argument("--curve-points", type=int, default=20)
    parser.add_argument("--eval-video-folder", default="")
    parser.add_argument("--eval-drive-file-id", default="")
    parser.add_argument("--output-dir", default="stacked_experiment")

    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--meta-l2", type=float, default=0.02)
    parser.add_argument("--shape-anchor-points", type=int, default=1)
    parser.add_argument("--level-anchor-points", type=int, default=2)
    parser.add_argument("--tail-anchor-points", type=int, default=2)
    return parser.parse_args()


def _fit_pointwise_predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_pred: pd.DataFrame,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
    task_type: str = "GPU",
    gpu_ram_part: float = 0.6,
) -> np.ndarray:
    n_points = y_train.shape[1]
    pred = np.zeros((len(x_pred), n_points), dtype=float)
    for p in range(n_points):
        target = y_train[:, p]
        if target.size == 0:
            pred[:, p] = 0.0
            continue
        if float(np.var(target)) < 1e-12:
            pred[:, p] = float(target[0])
            continue
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=seed + p,
            task_type=task_type,
            gpu_ram_part=gpu_ram_part,
            verbose=False,
        )
        model.fit(x_train, target)
        pred[:, p] = model.predict(x_pred)
    return _clamp01(pred)


def _fit_delta_predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_pred: pd.DataFrame,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
    task_type: str = "GPU",
    gpu_ram_part: float = 0.6,
) -> np.ndarray:
    base = _fit_pointwise_predict(
        x_train=x_train,
        y_train=y_train[:, :1],
        x_pred=x_pred,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        seed=seed + 5000,
        task_type=task_type,
        gpu_ram_part=gpu_ram_part,
    )
    out = np.zeros((len(x_pred), y_train.shape[1]), dtype=float)
    out[:, 0] = base[:, 0]

    for p in range(1, y_train.shape[1]):
        target_delta = y_train[:, p] - y_train[:, p - 1]
        if float(np.var(target_delta)) < 1e-12:
            pred_delta = np.full((len(x_pred),), float(target_delta[0]), dtype=float)
        else:
            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                random_seed=seed + 6000 + p,
                task_type=task_type,
                gpu_ram_part=gpu_ram_part,
                verbose=False,
            )
            model.fit(x_train, target_delta)
            pred_delta = model.predict(x_pred)
        pred_delta = np.clip(pred_delta, -0.30, 0.30)
        out[:, p] = np.clip(out[:, p - 1] + pred_delta, 0.0, 1.0)
    return out


def _fit_shape_level_predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_pred: pd.DataFrame,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
    shape_anchor_points: int,
    level_anchor_points: int,
    tail_anchor_points: int,
    task_type: str = "GPU",
    gpu_ram_part: float = 0.6,
) -> np.ndarray:
    n_train, n_points = y_train.shape
    y_shape = np.zeros_like(y_train, dtype=float)
    for i in range(n_train):
        y_shape[i] = _shape_from_curve(y_train[i], anchor_points=shape_anchor_points)

    pred_shape = _fit_pointwise_predict(
        x_train=x_train,
        y_train=y_shape,
        x_pred=x_pred,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=max(4, depth),
        seed=seed + 10000,
        task_type=task_type,
        gpu_ram_part=gpu_ram_part,
    )
    for i in range(len(pred_shape)):
        if n_points > 0:
            pred_shape[i, 0] = 1.0

    y_level = np.array([_anchor_mean(y_train[i], level_anchor_points) for i in range(n_train)], dtype=float)
    y_tail = np.array([_tail_mean(y_train[i], tail_anchor_points) for i in range(n_train)], dtype=float)

    if float(np.var(y_level)) < 1e-12:
        a = np.full((len(x_pred),), float(y_level[0]), dtype=float)
    else:
        level_model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=seed + 11000,
            task_type=task_type,
            gpu_ram_part=gpu_ram_part,
            verbose=False,
        )
        level_model.fit(x_train, y_level)
        a = level_model.predict(x_pred)
    if float(np.var(y_tail)) < 1e-12:
        b = np.full((len(x_pred),), float(y_tail[0]), dtype=float)
    else:
        tail_model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=seed + 12000,
            task_type=task_type,
            gpu_ram_part=gpu_ram_part,
            verbose=False,
        )
        tail_model.fit(x_train, y_tail)
        b = tail_model.predict(x_pred)
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)

    out = np.zeros_like(pred_shape, dtype=float)
    for i in range(len(out)):
        aa = float(a[i])
        bb = float(min(aa, b[i]))
        out[i] = np.clip(bb + (aa - bb) * pred_shape[i], 0.0, 1.0)
    return out


def _fit_ridge_pointwise(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    # Защита от NaN/inf и вырожденных конфигураций.
    x = np.nan_to_num(x.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    n_features = x.shape[1]
    x_aug = np.hstack([x, np.ones((x.shape[0], 1), dtype=float)])
    reg = np.eye(n_features + 1, dtype=float) * float(max(l2, 0.0))
    reg[-1, -1] = 0.0  # intercept not regularized
    lhs = x_aug.T @ x_aug + reg
    rhs = x_aug.T @ y
    # При почти линейно зависимых признаках solve может падать.
    # Fallback на pinv делает обучение устойчивым.
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        jitter = 1e-8
        lhs_jitter = lhs + np.eye(lhs.shape[0], dtype=float) * jitter
        try:
            beta = np.linalg.solve(lhs_jitter, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(lhs_jitter) @ rhs
    return beta


def _predict_ridge(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([x, np.ones((x.shape[0], 1), dtype=float)])
    return x_aug @ beta


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    x_train = make_feature_matrix(train_df).reset_index(drop=True)
    x_test = make_feature_matrix(test_df).reset_index(drop=True)
    y_train = _get_target_matrix(train_df, args.curve_points)
    y_true = _get_target_matrix(test_df, args.curve_points)[0]

    n_train = len(x_train)
    folds = max(2, min(int(args.oof_folds), n_train))
    oof_abs = np.zeros_like(y_train)
    oof_delta = np.zeros_like(y_train)
    oof_shape = np.zeros_like(y_train)

    for fold_idx, (tr_idx, va_idx) in enumerate(_kfold_indices(n_train, folds, args.random_seed), start=1):
        x_tr = x_train.iloc[tr_idx].reset_index(drop=True)
        x_va = x_train.iloc[va_idx].reset_index(drop=True)
        y_tr = y_train[tr_idx]

        pred_abs = _fit_pointwise_predict(
            x_train=x_tr,
            y_train=y_tr,
            x_pred=x_va,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            seed=args.random_seed + fold_idx * 100,
            task_type=args.task_type,
            gpu_ram_part=args.gpu_ram_part,
        )
        pred_delta = _fit_delta_predict(
            x_train=x_tr,
            y_train=y_tr,
            x_pred=x_va,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            seed=args.random_seed + fold_idx * 100,
            task_type=args.task_type,
            gpu_ram_part=args.gpu_ram_part,
        )
        pred_shape = _fit_shape_level_predict(
            x_train=x_tr,
            y_train=y_tr,
            x_pred=x_va,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            seed=args.random_seed + fold_idx * 100,
            shape_anchor_points=args.shape_anchor_points,
            level_anchor_points=args.level_anchor_points,
            tail_anchor_points=args.tail_anchor_points,
            task_type=args.task_type,
            gpu_ram_part=args.gpu_ram_part,
        )
        oof_abs[va_idx] = pred_abs
        oof_delta[va_idx] = pred_delta
        oof_shape[va_idx] = pred_shape

    # Meta training (ridge per point)
    meta_betas: list[np.ndarray] = []
    for p in tqdm(range(args.curve_points), desc="Training points"):
        p_norm = float(p / max(1, args.curve_points - 1))
        x_meta = np.column_stack([oof_abs[:, p], oof_delta[:, p], oof_shape[:, p], np.full((n_train,), p_norm, dtype=float)])
        beta = _fit_ridge_pointwise(x_meta, y_train[:, p], l2=args.meta_l2)
        meta_betas.append(beta)

    # Train full base experts and predict holdout
    test_abs = _fit_pointwise_predict(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_test,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        seed=args.random_seed + 900,
        task_type=args.task_type,
        gpu_ram_part=args.gpu_ram_part,
    )[0]
    test_delta = _fit_delta_predict(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_test,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        seed=args.random_seed + 900,
        task_type=args.task_type,
        gpu_ram_part=args.gpu_ram_part,
    )[0]
    test_shape = _fit_shape_level_predict(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_test,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        seed=args.random_seed + 900,
        shape_anchor_points=args.shape_anchor_points,
        level_anchor_points=args.level_anchor_points,
        tail_anchor_points=args.tail_anchor_points,
        task_type=args.task_type,
        gpu_ram_part=args.gpu_ram_part,
    )[0]

    y_pred = np.zeros((args.curve_points,), dtype=float)
    for p in tqdm(range(args.curve_points), desc="Training points"):
        p_norm = float(p / max(1, args.curve_points - 1))
        x_meta_test = np.array([[test_abs[p], test_delta[p], test_shape[p], p_norm]], dtype=float)
        y_pred[p] = float(_predict_ridge(x_meta_test, meta_betas[p])[0])
    y_pred = _clamp01(y_pred)

    abs_err = np.abs(y_pred - y_true)
    result_df = pd.DataFrame(
        {
            "point_idx": list(range(args.curve_points)),
            "point_frac": np.linspace(0.0, 1.0, args.curve_points),
            "pred_abs_expert": test_abs,
            "pred_delta_expert": test_delta,
            "pred_shape_expert": test_shape,
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "stacked_dataset.csv"
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    metrics_path = out_dir / "metrics.json"

    all_df.to_csv(dataset_path, index=False)
    result_df.to_csv(pred_path, index=False)

    metrics = {
        "videos_total_with_target": len(rows),
        "videos_used": len(all_df),
        "train_videos": len(train_df),
        "curve_points": int(args.curve_points),
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(test_df.iloc[0]["drive_file_id"]),
        "oof_folds": int(folds),
        "meta_l2": float(args.meta_l2),
        "spearman": float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")),
        "pearson": float(pd.Series(y_pred).corr(pd.Series(y_true), method="pearson")),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(abs_err)),
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Stacked LOO ===")
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
