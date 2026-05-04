from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, CatBoostRegressor

from train.common.train_utils import anchor_mean as _anchor_mean
from train.common.train_utils import clamp01 as _clamp01
from train.common.train_utils import make_feature_matrix
from train.common.train_utils import point_col as _point_col
from train.common.train_utils import safe_float as _safe_float
from train.common.train_utils import shape_from_curve as _shape_from_curve
from train.common.train_utils import tail_mean as _tail_mean
from train.loo.train_retention_regressor_loo import build_rows_with_targets_source, select_train_test
from train.tools.drive_feature_labeling_pipeline import DEFAULT_PARENT_FOLDER_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Гибридный LOO-эксперимент retention: shape-модель (ranker) + level-модель (regressor) для предсказания и формы, и абсолютных значений.")
    )
    parser.add_argument("--env-file", default=".env", help="Путь к .env.")
    parser.add_argument("--snapshot-dir", default="", help=("Локальный снапшот с данными (например drive_snapshot_90). Если задан и существует, загрузка с Drive не выполняется."))
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID, help="ID корневой папки с видео на Google Drive.")
    parser.add_argument("--limit-videos", type=int, default=45)
    parser.add_argument("--train-videos", type=int, default=44)
    parser.add_argument("--curve-points", type=int, default=20)
    parser.add_argument("--eval-video-folder", default="")
    parser.add_argument("--eval-drive-file-id", default="")
    parser.add_argument("--output-dir", default="hybrid_experiment")

    parser.add_argument("--ranker-iterations", type=int, default=600)
    parser.add_argument("--ranker-learning-rate", type=float, default=0.05)
    parser.add_argument("--ranker-depth", type=int, default=6)

    parser.add_argument("--level-iterations", type=int, default=600)
    parser.add_argument("--level-learning-rate", type=float, default=0.05)
    parser.add_argument("--level-depth", type=int, default=6)

    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--ensemble-seeds", default="", help=("Список seed через запятую для ансамбля, например '42,52,62'. Если пусто, используется --random-seed."))
    parser.add_argument("--shape-ranker-weight", type=float, default=0.7, help=("Вес ranker-shape в смешивании двух shape-моделей [0..1]. Остальной вес идет на shape-regressor."))
    parser.add_argument("--shape-anchor-points", type=int, default=1, help=("Сколько первых точек усреднять для нормализации shape-target. Обычно 1..3."))
    parser.add_argument("--level-anchor-points", type=int, default=2, help=("Сколько первых точек усреднять для абсолютного level-target. Обычно 1..3."))
    parser.add_argument("--tail-anchor-points", type=int, default=2, help=("Сколько последних точек усреднять для tail/floor-target (b). Обычно 1..3."))
    parser.add_argument("--disable-tail-floor", action="store_true", help="Отключить модель хвоста b и использовать формулу y=a*shape.")
    parser.add_argument("--enable-affine-calibration", action="store_true", help=("Включить affine-калибровку абсолютной шкалы y'=alpha*y+beta по train-видео (in-sample)."))
    return parser.parse_args()


def _shape_from_raw_scores(raw_scores: np.ndarray) -> np.ndarray:
    smin, smax = float(raw_scores.min()), float(raw_scores.max())
    if smax - smin < 1e-12:
        shape = np.zeros_like(raw_scores, dtype=float)
    else:
        shape = (raw_scores - smin) / (smax - smin)
    if shape.size > 0:
        first = max(1e-6, float(shape[0]))
        shape = shape / first
        shape = _clamp01(shape)
        shape[0] = 1.0
    return shape


def _build_absolute_curve(shape: np.ndarray, a: float, b: float, use_tail_floor: bool) -> np.ndarray:
    aa = float(np.clip(a, 0.0, 1.0))
    if not use_tail_floor:
        return _clamp01(aa * shape)
    bb = float(np.clip(b, 0.0, aa))
    y = bb + (aa - bb) * shape
    return _clamp01(y)


def _parse_seed_list(args: argparse.Namespace) -> list[int]:
    raw = str(args.ensemble_seeds or "").strip()
    if not raw:
        return [int(args.random_seed)]
    seeds: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            seeds.append(int(token))
        except Exception:
            continue
    if not seeds:
        seeds = [int(args.random_seed)]
    return sorted(list(dict.fromkeys(seeds)))


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    X_train_video = make_feature_matrix(train_df)
    X_test_video = make_feature_matrix(test_df)
    if X_train_video.empty:
        raise RuntimeError("Пустая матрица признаков для обучения")

    # ----- SHAPE: ranker по нормализованной форме кривой -----
    train_docs: list[dict[str, float]] = []
    train_y_shape: list[float] = []
    train_group_id: list[int] = []

    for group_idx, (_, row) in enumerate(train_df.iterrows()):
        base = X_train_video.iloc[group_idx].to_dict()
        true_curve = np.array([_safe_float(row[_point_col(i)], 0.0) for i in range(args.curve_points)], dtype=float)
        true_shape = _shape_from_curve(true_curve, anchor_points=args.shape_anchor_points)
        for bin_idx in range(args.curve_points):
            doc = dict(base)
            doc["bin_idx"] = float(bin_idx)
            doc["bin_pos_norm"] = float(bin_idx / max(1, args.curve_points - 1))
            train_docs.append(doc)
            train_y_shape.append(float(true_shape[bin_idx]))
            train_group_id.append(group_idx)

    X_shape_train = pd.DataFrame(train_docs).fillna(0.0)
    y_shape_train = np.array(train_y_shape, dtype=float)
    group_id = np.array(train_group_id, dtype=int)

    test_base = X_test_video.iloc[0].to_dict()
    test_docs: list[dict[str, float]] = []
    y_true_curve = np.array([_safe_float(test_df.iloc[0][_point_col(i)], 0.0) for i in range(args.curve_points)], dtype=float)
    y_true_curve = _clamp01(y_true_curve)
    y_true_shape = _shape_from_curve(y_true_curve, anchor_points=args.shape_anchor_points)

    for bin_idx in range(args.curve_points):
        doc = dict(test_base)
        doc["bin_idx"] = float(bin_idx)
        doc["bin_pos_norm"] = float(bin_idx / max(1, args.curve_points - 1))
        test_docs.append(doc)
    X_shape_test = pd.DataFrame(test_docs).fillna(0.0)

    # ----- SHAPE + LEVEL: ансамбль простых моделей -----
    seeds = _parse_seed_list(args)
    ranker_weight = float(np.clip(args.shape_ranker_weight, 0.0, 1.0))
    ensemble_parts: list[dict[str, Any]] = []

    # ----- LEVEL targets -----
    y_level_train: list[float] = []
    y_tail_train: list[float] = []
    for _, row in train_df.iterrows():
        curve = np.array([_safe_float(row[_point_col(i)], 0.0) for i in range(args.curve_points)], dtype=float)
        curve = _clamp01(curve)
        y_level_train.append(_anchor_mean(curve, args.level_anchor_points))
        y_tail_train.append(_tail_mean(curve, args.tail_anchor_points))
    y_level_train_arr = np.array(y_level_train, dtype=float)
    y_tail_train_arr = np.array(y_tail_train, dtype=float)

    level_true = _anchor_mean(y_true_curve, args.level_anchor_points)
    tail_true = _tail_mean(y_true_curve, args.tail_anchor_points)

    for seed in seeds:
        shape_ranker = CatBoostRanker(
            loss_function="YetiRank",
            eval_metric=f"NDCG:top={min(20, args.curve_points)}",
            iterations=args.ranker_iterations,
            learning_rate=args.ranker_learning_rate,
            depth=args.ranker_depth,
            random_seed=seed,
            task_type=getattr(args, "task_type", "GPU"),
            gpu_ram_part=getattr(args, "gpu_ram_part", 0.6),
            verbose=False,
        )
        shape_ranker.fit(X_shape_train, y_shape_train, group_id=group_id)
        shape_ranker_raw = np.array(shape_ranker.predict(X_shape_test), dtype=float)
        shape_ranker_pred = _shape_from_raw_scores(shape_ranker_raw)

        shape_regressor = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=args.ranker_iterations,
            learning_rate=args.ranker_learning_rate,
            depth=max(4, args.ranker_depth),
            random_seed=seed + 1000,
            task_type=getattr(args, "task_type", "GPU"),
            gpu_ram_part=getattr(args, "gpu_ram_part", 0.6),
            verbose=False,
        )
        shape_regressor.fit(X_shape_train, y_shape_train)
        shape_reg_raw = np.array(shape_regressor.predict(X_shape_test), dtype=float)
        shape_reg_pred = _clamp01(shape_reg_raw)
        if shape_reg_pred.size > 0:
            shape_reg_pred[0] = 1.0

        shape_pred_seed = _clamp01(ranker_weight * shape_ranker_pred + (1.0 - ranker_weight) * shape_reg_pred)
        if shape_pred_seed.size > 0:
            shape_pred_seed[0] = 1.0

        level_model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=args.level_iterations,
            learning_rate=args.level_learning_rate,
            depth=args.level_depth,
            random_seed=seed + 2000,
            task_type=getattr(args, "task_type", "GPU"),
            gpu_ram_part=getattr(args, "gpu_ram_part", 0.6),
            verbose=False,
        )
        level_model.fit(X_train_video, y_level_train_arr)
        level_pred_seed = float(level_model.predict(X_test_video)[0])
        level_pred_seed = float(np.clip(level_pred_seed, 0.0, 1.0))

        tail_model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=args.level_iterations,
            learning_rate=args.level_learning_rate,
            depth=args.level_depth,
            random_seed=seed + 3000,
            task_type=getattr(args, "task_type", "GPU"),
            gpu_ram_part=getattr(args, "gpu_ram_part", 0.6),
            verbose=False,
        )
        tail_model.fit(X_train_video, y_tail_train_arr)
        tail_pred_seed = float(tail_model.predict(X_test_video)[0])
        tail_pred_seed = float(np.clip(tail_pred_seed, 0.0, level_pred_seed))

        ensemble_parts.append(
            {
                "seed": seed,
                "shape_ranker": shape_ranker,
                "shape_regressor": shape_regressor,
                "level_model": level_model,
                "tail_model": tail_model,
                "shape_pred": shape_pred_seed,
                "level_pred": level_pred_seed,
                "tail_pred": tail_pred_seed,
            }
        )

    shape_pred = np.mean([x["shape_pred"] for x in ensemble_parts], axis=0)
    shape_pred = _clamp01(shape_pred)
    if shape_pred.size > 0:
        shape_pred[0] = 1.0
    level_pred = float(np.mean([x["level_pred"] for x in ensemble_parts]))
    level_pred = float(np.clip(level_pred, 0.0, 1.0))
    tail_pred = float(np.mean([x["tail_pred"] for x in ensemble_parts]))
    tail_pred = float(np.clip(tail_pred, 0.0, level_pred))

    # ----- Сборка финальной абсолютной кривой -----
    use_tail_floor = not args.disable_tail_floor
    y_pred_curve_uncal = _build_absolute_curve(shape=shape_pred, a=level_pred, b=tail_pred, use_tail_floor=use_tail_floor)

    # Опциональная affine-калибровка абсолютной шкалы по train-видео (in-sample).
    # Удобно для быстрой подстройки уровня без сторонних библиотек.
    calib_alpha = 1.0
    calib_beta = 0.0
    y_pred_curve = y_pred_curve_uncal.copy()
    if args.enable_affine_calibration:
        train_pred_points: list[float] = []
        train_true_points: list[float] = []
        for train_idx, (_, row) in enumerate(train_df.iterrows()):
            base = X_train_video.iloc[train_idx].to_dict()
            docs: list[dict[str, float]] = []
            for bin_idx in range(args.curve_points):
                doc = dict(base)
                doc["bin_idx"] = float(bin_idx)
                doc["bin_pos_norm"] = float(bin_idx / max(1, args.curve_points - 1))
                docs.append(doc)
            x_docs = pd.DataFrame(docs).fillna(0.0)

            shape_preds_part: list[np.ndarray] = []
            a_preds_part: list[float] = []
            b_preds_part: list[float] = []
            for part in ensemble_parts:
                ranker_raw = np.array(part["shape_ranker"].predict(x_docs), dtype=float)
                ranker_shape = _shape_from_raw_scores(ranker_raw)

                reg_raw = np.array(part["shape_regressor"].predict(x_docs), dtype=float)
                reg_shape = _clamp01(reg_raw)
                if reg_shape.size > 0:
                    reg_shape[0] = 1.0

                shape_mix = _clamp01(ranker_weight * ranker_shape + (1.0 - ranker_weight) * reg_shape)
                if shape_mix.size > 0:
                    shape_mix[0] = 1.0
                shape_preds_part.append(shape_mix)

                a_pred = float(part["level_model"].predict(X_train_video.iloc[[train_idx]])[0])
                a_pred = float(np.clip(a_pred, 0.0, 1.0))
                b_pred = float(part["tail_model"].predict(X_train_video.iloc[[train_idx]])[0])
                b_pred = float(np.clip(b_pred, 0.0, a_pred))
                a_preds_part.append(a_pred)
                b_preds_part.append(b_pred)

            shape_train_pred = _clamp01(np.mean(shape_preds_part, axis=0))
            if shape_train_pred.size > 0:
                shape_train_pred[0] = 1.0
            a_train_pred = float(np.clip(np.mean(a_preds_part), 0.0, 1.0))
            b_train_pred = float(np.clip(np.mean(b_preds_part), 0.0, a_train_pred))
            y_train_pred_curve = _build_absolute_curve(shape=shape_train_pred, a=a_train_pred, b=b_train_pred, use_tail_floor=use_tail_floor)
            y_train_true_curve = np.array([_safe_float(row[_point_col(i)], 0.0) for i in range(args.curve_points)], dtype=float)
            y_train_true_curve = _clamp01(y_train_true_curve)

            train_pred_points.extend(list(y_train_pred_curve))
            train_true_points.extend(list(y_train_true_curve))

        x = np.array(train_pred_points, dtype=float)
        y = np.array(train_true_points, dtype=float)
        if x.size >= 2 and float(np.var(x)) > 1e-12:
            # Решаем y ~= alpha*x + beta через least squares.
            A = np.vstack([x, np.ones_like(x)]).T
            alpha, beta = np.linalg.lstsq(A, y, rcond=None)[0]
            calib_alpha = float(alpha)
            calib_beta = float(beta)
            y_pred_curve = _clamp01(calib_alpha * y_pred_curve_uncal + calib_beta)

    abs_err = np.abs(y_pred_curve - y_true_curve)
    abs_err_uncal = np.abs(y_pred_curve_uncal - y_true_curve)
    shape_abs_err = np.abs(shape_pred - y_true_shape)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(args.curve_points)),
            "point_frac": np.linspace(0.0, 1.0, args.curve_points),
            "pred_shape": shape_pred,
            "true_shape": y_true_shape,
            "pred_retention_uncalibrated": y_pred_curve_uncal,
            "pred_retention": y_pred_curve,
            "pred_retention_norm": y_pred_curve,
            "pred_score_raw": y_pred_curve,
            "true_retention": y_true_curve,
            "abs_error": abs_err,
            "abs_error_uncalibrated": abs_err_uncal,
            "shape_abs_error": shape_abs_err,
        }
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "hybrid_dataset.csv"
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    metrics_path = out_dir / "metrics.json"
    ensemble_models_dir = out_dir / "ensemble_models"
    ensemble_models_dir.mkdir(parents=True, exist_ok=True)

    all_df.to_csv(dataset_path, index=False)
    result_df.to_csv(pred_path, index=False)
    for part in ensemble_parts:
        seed = int(part["seed"])
        part["shape_ranker"].save_model(str(ensemble_models_dir / f"shape_ranker_seed_{seed}.cbm"))
        part["shape_regressor"].save_model(str(ensemble_models_dir / f"shape_regressor_seed_{seed}.cbm"))
        part["level_model"].save_model(str(ensemble_models_dir / f"level_regressor_seed_{seed}.cbm"))
        part["tail_model"].save_model(str(ensemble_models_dir / f"tail_regressor_seed_{seed}.cbm"))

    metrics = {
        "videos_total_with_target": len(rows),
        "videos_used": len(all_df),
        "train_videos": len(train_df),
        "curve_points": int(args.curve_points),
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(test_df.iloc[0]["drive_file_id"]),
        "ensemble_seeds": seeds,
        "shape_ranker_weight": ranker_weight,
        "shape_anchor_points": int(args.shape_anchor_points),
        "level_anchor_points": int(args.level_anchor_points),
        "level_pred": level_pred,
        "level_true": level_true,
        "level_abs_error": float(abs(level_pred - level_true)),
        "tail_pred": tail_pred,
        "tail_true": tail_true,
        "tail_abs_error": float(abs(tail_pred - tail_true)),
        "use_tail_floor": int(use_tail_floor),
        "affine_calibration_enabled": int(args.enable_affine_calibration),
        "calib_alpha": float(calib_alpha),
        "calib_beta": float(calib_beta),
        "spearman_abs": float(pd.Series(y_pred_curve).corr(pd.Series(y_true_curve), method="spearman")),
        "pearson_abs": float(pd.Series(y_pred_curve).corr(pd.Series(y_true_curve), method="pearson")),
        "rmse_abs": float(np.sqrt(np.mean((y_pred_curve - y_true_curve) ** 2))),
        "mae_abs": float(np.mean(abs_err)),
        "rmse_abs_uncalibrated": float(np.sqrt(np.mean((y_pred_curve_uncal - y_true_curve) ** 2))),
        "mae_abs_uncalibrated": float(np.mean(abs_err_uncal)),
        "spearman_shape": float(pd.Series(shape_pred).corr(pd.Series(y_true_shape), method="spearman")),
        "pearson_shape": float(pd.Series(shape_pred).corr(pd.Series(y_true_shape), method="pearson")),
        "rmse_shape": float(np.sqrt(np.mean((shape_pred - y_true_shape) ** 2))),
        "mae_shape": float(np.mean(shape_abs_err)),
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
        "ensemble_models_dir": str(ensemble_models_dir),
        "ensemble_models_count": int(len(ensemble_parts) * 4),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Hybrid LOO ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\n=== Holdout Prediction vs True ({args.curve_points} points) ===")
    print(
        result_df.to_string(
            index=False,
            formatters={
                "pred_shape": lambda x: f"{x:0.5f}",
                "true_shape": lambda x: f"{x:0.5f}",
                "pred_retention": lambda x: f"{x:0.5f}",
                "true_retention": lambda x: f"{x:0.5f}",
                "abs_error": lambda x: f"{x:0.5f}",
                "shape_abs_error": lambda x: f"{x:0.5f}",
            },
        )
    )
    return metrics


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
