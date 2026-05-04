from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID, _point_col, _safe_float, build_rows_with_targets_source, make_feature_matrix, select_train_test
from train.loo.train_retention_local_knn_loo import _pca_project, _predict_local_knn_residual, _robust_standardize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Meta-ensemble LOO: стекинг трёх архитектурно-разных базовых моделей (baseline, residual_huber, local_knn) через per-point ridge meta-learner.")
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--snapshot-dir", default="")
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    parser.add_argument("--limit-videos", type=int, default=90)
    parser.add_argument("--train-videos", type=int, default=89)
    parser.add_argument("--curve-points", type=int, default=50)
    parser.add_argument("--eval-video-folder", default="")
    parser.add_argument("--eval-drive-file-id", default="")
    parser.add_argument("--output-dir", default="meta_ensemble_experiment")

    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--meta-l2", type=float, default=0.02)

    parser.add_argument("--rh-iterations", type=int, default=500)
    parser.add_argument("--rh-learning-rate", type=float, default=0.05)
    parser.add_argument("--rh-depth", type=int, default=6)
    parser.add_argument("--rh-huber-delta", type=float, default=0.03)
    parser.add_argument("--rh-delta-blend", type=float, default=0.65)
    parser.add_argument("--rh-delta-max-step", type=float, default=0.15)
    parser.add_argument("--rh-huber-delta-delta", type=float, default=0.015)

    parser.add_argument("--knn-neighbors-k", type=int, default=12)
    parser.add_argument("--knn-distance-temperature", type=float, default=1.0)
    parser.add_argument("--knn-residual-strength", type=float, default=0.70)
    parser.add_argument("--knn-spike-gain", type=float, default=1.10)
    parser.add_argument("--knn-smooth-window", type=int, default=7)
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


def _predict_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    baseline = _clamp01(np.mean(y_train, axis=0))
    return np.tile(baseline, (n_test, 1))


def _predict_residual_huber(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_pred: pd.DataFrame,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
    huber_delta: float,
    huber_delta_delta: float,
    delta_blend: float,
    delta_max_step: float,
    task_type: str = "GPU",
    gpu_ram_part: float = 0.6,
) -> np.ndarray:
    n_pred = len(x_pred)
    n_points = y_train.shape[1]
    baseline = np.mean(y_train, axis=0)
    residual_train = y_train - baseline[np.newaxis, :]

    pred_res_base = np.zeros((n_pred, n_points), dtype=float)
    for p in range(n_points):
        target = residual_train[:, p]
        if target.size == 0:
            continue
        if float(np.var(target)) < 1e-12:
            pred_res_base[:, p] = float(target[0])
            continue
        model = CatBoostRegressor(
            loss_function=f"Huber:delta={huber_delta}",
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
        pred_res_base[:, p] = model.predict(x_pred)

    delta_res_curve = np.zeros((n_pred, n_points), dtype=float)
    delta_res_curve[:, 0] = pred_res_base[:, 0]
    for p in range(1, n_points):
        target_d = residual_train[:, p] - residual_train[:, p - 1]
        if target_d.size == 0:
            pred_d = np.zeros((n_pred,), dtype=float)
        elif float(np.var(target_d)) < 1e-12:
            pred_d = np.full((n_pred,), float(target_d[0]), dtype=float)
        else:
            dm = CatBoostRegressor(
                loss_function=f"Huber:delta={huber_delta_delta}",
                eval_metric="RMSE",
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                random_seed=seed + 5000 + p,
                task_type=task_type,
                gpu_ram_part=gpu_ram_part,
                verbose=False,
            )
            dm.fit(x_train, target_d)
            pred_d = dm.predict(x_pred)
        pred_d = np.clip(pred_d, -delta_max_step, delta_max_step)
        delta_res_curve[:, p] = delta_res_curve[:, p - 1] + pred_d

    blend = float(np.clip(delta_blend, 0.0, 1.0))
    final_res = (1.0 - blend) * pred_res_base + blend * delta_res_curve
    return _clamp01(baseline[np.newaxis, :] + final_res)


def _predict_knn(
    x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray, neighbors_k: int, distance_temperature: float, residual_strength: float, spike_gain: float, smooth_window: int
) -> np.ndarray:
    x_train_z, x_pred_z = _robust_standardize(x_train, x_pred)
    x_train_p, x_pred_p = _pca_project(x_train_z, x_pred_z)
    pred_mat, _, _ = _predict_local_knn_residual(
        x_train=x_train_p,
        y_train=y_train,
        x_test=x_pred_p,
        neighbors_k=neighbors_k,
        distance_temperature=distance_temperature,
        residual_strength=residual_strength,
        spike_gain=spike_gain,
        smooth_window=smooth_window,
    )
    return _clamp01(pred_mat)


def _kfold_indices(n: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != i], axis=0)
        out.append((train, val))
    return out


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    x = np.nan_to_num(x.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    n_features = x.shape[1]
    x_aug = np.hstack([x, np.ones((x.shape[0], 1), dtype=float)])
    reg = np.eye(n_features + 1, dtype=float) * float(max(l2, 0.0))
    reg[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + reg
    rhs = x_aug.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs + np.eye(lhs.shape[0]) * 1e-8) @ rhs
    return beta


def _predict_ridge(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([x, np.ones((x.shape[0], 1), dtype=float)])
    return x_aug @ beta


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    x_train_df = make_feature_matrix(train_df).reset_index(drop=True)
    x_test_df = make_feature_matrix(test_df).reset_index(drop=True)
    if x_train_df.empty:
        raise RuntimeError("Пустая матрица признаков для meta_ensemble")

    y_train = _get_target_matrix(train_df, args.curve_points)
    y_true = _get_target_matrix(test_df, args.curve_points)[0]
    n_train = len(x_train_df)
    n_points = int(args.curve_points)
    folds = max(2, min(int(args.oof_folds), n_train))

    x_train_np = np.nan_to_num(x_train_df.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    x_test_np = np.nan_to_num(x_test_df.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    oof_baseline = np.zeros_like(y_train)
    oof_rh = np.zeros_like(y_train)
    oof_knn = np.zeros_like(y_train)

    print(f"[meta_ensemble] stage=oof folds={folds} train={n_train} points={n_points}")
    for fold_idx, (tr_idx, va_idx) in enumerate(_kfold_indices(n_train, folds, args.random_seed), start=1):
        print(f"[meta_ensemble] fold {fold_idx}/{folds} train={len(tr_idx)} val={len(va_idx)}")
        x_tr_df = x_train_df.iloc[tr_idx].reset_index(drop=True)
        x_va_df = x_train_df.iloc[va_idx].reset_index(drop=True)
        y_tr = y_train[tr_idx]

        x_tr_np = x_train_np[tr_idx]
        x_va_np = x_train_np[va_idx]

        print(f"[meta_ensemble] fold {fold_idx}/{folds} base=baseline")
        oof_baseline[va_idx] = _predict_baseline(y_tr, len(va_idx))

        print(f"[meta_ensemble] fold {fold_idx}/{folds} base=residual_huber")
        oof_rh[va_idx] = _predict_residual_huber(
            x_train=x_tr_df,
            y_train=y_tr,
            x_pred=x_va_df,
            iterations=args.rh_iterations,
            learning_rate=args.rh_learning_rate,
            depth=args.rh_depth,
            seed=args.random_seed + fold_idx * 100,
            huber_delta=args.rh_huber_delta,
            huber_delta_delta=args.rh_huber_delta_delta,
            delta_blend=args.rh_delta_blend,
            delta_max_step=args.rh_delta_max_step,
            task_type=args.task_type,
            gpu_ram_part=args.gpu_ram_part,
        )

        print(f"[meta_ensemble] fold {fold_idx}/{folds} base=local_knn")
        oof_knn[va_idx] = _predict_knn(
            x_train=x_tr_np,
            y_train=y_tr,
            x_pred=x_va_np,
            neighbors_k=args.knn_neighbors_k,
            distance_temperature=args.knn_distance_temperature,
            residual_strength=args.knn_residual_strength,
            spike_gain=args.knn_spike_gain,
            smooth_window=args.knn_smooth_window,
        )
        print(f"[meta_ensemble] fold {fold_idx}/{folds} done")

    print(f"[meta_ensemble] stage=meta_train points={n_points}")
    meta_betas: list[np.ndarray] = []
    for p in range(n_points):
        p_norm = float(p / max(1, n_points - 1))
        spread = np.abs(
            np.max(np.column_stack([oof_baseline[:, p], oof_rh[:, p], oof_knn[:, p]]), axis=1) - np.min(np.column_stack([oof_baseline[:, p], oof_rh[:, p], oof_knn[:, p]]), axis=1)
        )
        x_meta = np.column_stack([oof_baseline[:, p], oof_rh[:, p], oof_knn[:, p], np.full((n_train,), p_norm, dtype=float), spread])
        beta = _fit_ridge(x_meta, y_train[:, p], l2=args.meta_l2)
        meta_betas.append(beta)

    print("[meta_ensemble] stage=holdout_base_models")
    print("[meta_ensemble] holdout base=baseline")
    test_baseline = _predict_baseline(y_train, 1)

    print("[meta_ensemble] holdout base=residual_huber")
    test_rh = _predict_residual_huber(
        x_train=x_train_df,
        y_train=y_train,
        x_pred=x_test_df,
        iterations=args.rh_iterations,
        learning_rate=args.rh_learning_rate,
        depth=args.rh_depth,
        seed=args.random_seed + 900,
        huber_delta=args.rh_huber_delta,
        huber_delta_delta=args.rh_huber_delta_delta,
        delta_blend=args.rh_delta_blend,
        delta_max_step=args.rh_delta_max_step,
        task_type=args.task_type,
        gpu_ram_part=args.gpu_ram_part,
    )

    print("[meta_ensemble] holdout base=local_knn")
    test_knn = _predict_knn(
        x_train=x_train_np,
        y_train=y_train,
        x_pred=x_test_np,
        neighbors_k=args.knn_neighbors_k,
        distance_temperature=args.knn_distance_temperature,
        residual_strength=args.knn_residual_strength,
        spike_gain=args.knn_spike_gain,
        smooth_window=args.knn_smooth_window,
    )

    print(f"[meta_ensemble] stage=meta_predict points={n_points}")
    y_pred = np.zeros((n_points,), dtype=float)
    for p in range(n_points):
        p_norm = float(p / max(1, n_points - 1))
        base_vals = np.array([float(test_baseline[0, p]), float(test_rh[0, p]), float(test_knn[0, p])])
        spread = float(base_vals.max() - base_vals.min())
        x_meta_test = np.array([[base_vals[0], base_vals[1], base_vals[2], p_norm, spread]], dtype=float)
        y_pred[p] = float(_predict_ridge(x_meta_test, meta_betas[p])[0])
    y_pred = _clamp01(y_pred)

    abs_err = np.abs(y_pred - y_true)
    cm = _curve_metrics(y_pred, y_true)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(n_points)),
            "point_frac": np.linspace(0.0, 1.0, n_points),
            "base_baseline": test_baseline[0],
            "base_residual_huber": test_rh[0],
            "base_local_knn": test_knn[0],
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "meta_ensemble_dataset.csv"
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
        "oof_folds": folds,
        "meta_l2": float(args.meta_l2),
        "base_models": ["baseline", "residual_huber", "local_knn"],
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
        **cm,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Meta-Ensemble LOO ===")
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
