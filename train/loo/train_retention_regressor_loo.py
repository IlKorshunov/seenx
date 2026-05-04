from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from tqdm import tqdm

from train.tools.drive_feature_labeling_pipeline import DEFAULT_PARENT_FOLDER_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("LOO-эксперимент для retention regression: обучение на N-1 видео и предсказание retention-кривой для holdout-видео."))
    parser.add_argument("--env-file", default=".env", help="Путь к .env.")
    parser.add_argument("--snapshot-dir", default="", help=("Локальный снапшот с данными (например drive_snapshot_90). Если задан и существует, загрузка с Drive не выполняется."))
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID, help="ID корневой папки с видео на Google Drive.")
    parser.add_argument("--limit-videos", type=int, default=45, help="Сколько видео включить в эксперимент.")
    parser.add_argument("--train-videos", type=int, default=44, help="Сколько видео использовать для обучения.")
    parser.add_argument("--curve-points", type=int, default=20, help="Количество точек кривой retention (например 20 или 50).")
    parser.add_argument("--eval-video-folder", default="", help="Явно выбрать eval-видео по имени папки.")
    parser.add_argument("--eval-drive-file-id", default="", help="Явно выбрать eval-видео по drive_file_id транскрипта.")
    parser.add_argument("--output-dir", default="regressor_experiment", help="Куда сохранить результаты эксперимента.")
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6, help="Fraction of GPU RAM for CatBoost (0.0-1.0)")
    parser.add_argument(
        "--delta-blend", type=float, default=0.65, help=("Вес дельта-кривой в финальном прогнозе [0..1]. 0 = только point-wise baseline, 1 = только реконструкция по дельтам.")
    )
    parser.add_argument("--delta-max-step", type=float, default=0.25, help="Ограничение по модулю для предсказанной дельты между соседними точками.")
    return parser.parse_args()


from train.common.retention_data_layer import build_rows_with_targets_source  # noqa: E402  — unified local+drive loader
from train.common.train_utils import make_feature_matrix
from train.common.train_utils import point_col as _point_col
from train.common.train_utils import safe_float as _safe_float


def build_rows_with_targets(root_folder_id: str, env_file: Path, curve_points: int) -> list[dict[str, Any]]:
    return build_rows_with_targets_source(root_folder_id=root_folder_id, env_file=env_file, curve_points=curve_points, snapshot_dir=None)


def select_train_test(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(rows) < args.limit_videos:
        raise RuntimeError(f"Недостаточно видео с target retention: найдено {len(rows)}, нужно {args.limit_videos}")
    if args.train_videos >= args.limit_videos:
        raise RuntimeError("--train-videos должен быть меньше --limit-videos")

    eval_video_folder = str(args.eval_video_folder or "").strip()
    eval_drive_file_id = str(args.eval_drive_file_id or "").strip()

    if eval_video_folder or eval_drive_file_id:
        matched = []
        for row in rows:
            ok = True
            if eval_video_folder:
                ok = ok and (str(row.get("video_folder", "")) == eval_video_folder)
            if eval_drive_file_id:
                ok = ok and (str(row.get("drive_file_id", "")) == eval_drive_file_id)
            if ok:
                matched.append(row)
        if not matched:
            raise RuntimeError(f"Не найдено eval-видео по заданным параметрам (eval-video-folder='{eval_video_folder}', eval-drive-file-id='{eval_drive_file_id}')")
        eval_row = matched[0]
        remaining = [r for r in rows if str(r.get("drive_file_id", "")) != str(eval_row.get("drive_file_id", ""))]
        required_train_pool = max(args.limit_videos - 1, args.train_videos)
        if len(remaining) < required_train_pool:
            raise RuntimeError(f"Недостаточно данных после фиксации eval-видео: осталось {len(remaining)}, нужно минимум {required_train_pool}")
        train_pool = remaining[:required_train_pool]
        train_df = pd.DataFrame(train_pool[: args.train_videos])
        test_df = pd.DataFrame([eval_row])
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        return all_df, train_df, test_df

    rows = rows[: args.limit_videos]
    all_df = pd.DataFrame(rows)
    train_df = all_df.iloc[: args.train_videos].copy()
    test_df = all_df.iloc[args.train_videos : args.train_videos + 1].copy()
    if test_df.empty:
        raise RuntimeError("Не удалось выделить тестовое видео")
    return all_df, train_df, test_df


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    X_train = make_feature_matrix(train_df)
    X_test = make_feature_matrix(test_df)
    if X_train.empty:
        raise RuntimeError("Пустая матрица признаков для обучения")

    pred_values: list[float] = []
    pred_values_base: list[float] = []
    pred_values_delta_curve: list[float] = []
    true_values: list[float] = []
    point_model_paths: list[str] = []
    delta_model_paths: list[str] = []

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points_dir = out_dir / "point_models"
    points_dir.mkdir(parents=True, exist_ok=True)

    # 1) baseline: отдельная регрессия по каждой абсолютной точке.
    for point_idx in tqdm(range(args.curve_points), desc="Baseline points"):
        col = _point_col(point_idx)
        y_train = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_true = _safe_float(test_df.iloc[0][col], 0.0)

        cb_kwargs = dict(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            random_seed=args.random_seed,
            task_type=args.task_type,
            verbose=False,
        )
        if args.task_type == "GPU":
            cb_kwargs["gpu_ram_part"] = args.gpu_ram_part
        model = CatBoostRegressor(**cb_kwargs)
        model.fit(X_train, y_train)
        pred = float(model.predict(X_test)[0])
        pred = float(max(0.0, min(1.0, pred)))

        model_path = points_dir / f"catboost_regressor_point_{point_idx:03d}.cbm"
        model.save_model(str(model_path))

        pred_values_base.append(pred)
        true_values.append(y_true)
        point_model_paths.append(str(model_path))

    # 2) delta-path: отдельная регрессия по дельтам соседних точек.
    # Так модель лучше подхватывает локальные спады/всплески формы.
    delta_max_step = float(max(1e-6, args.delta_max_step))
    baseline_curve = np.array(pred_values_base, dtype=float)
    delta_curve = np.array(pred_values_base, dtype=float)
    for point_idx in tqdm(range(1, args.curve_points), desc="Delta points"):
        col_cur = _point_col(point_idx)
        col_prev = _point_col(point_idx - 1)
        y_train_cur = pd.to_numeric(train_df[col_cur], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_train_prev = pd.to_numeric(train_df[col_prev], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_train_delta = y_train_cur - y_train_prev

        cb_delta_kwargs = dict(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            random_seed=args.random_seed,
            task_type=args.task_type,
            verbose=False,
        )
        if args.task_type == "GPU":
            cb_delta_kwargs["gpu_ram_part"] = args.gpu_ram_part
        delta_model = CatBoostRegressor(**cb_delta_kwargs)
        delta_model.fit(X_train, y_train_delta)
        pred_delta = float(delta_model.predict(X_test)[0])
        pred_delta = float(np.clip(pred_delta, -delta_max_step, delta_max_step))

        reconstructed = float(delta_curve[point_idx - 1] + pred_delta)
        delta_curve[point_idx] = float(np.clip(reconstructed, 0.0, 1.0))

        delta_model_path = points_dir / f"catboost_regressor_delta_{point_idx:03d}.cbm"
        delta_model.save_model(str(delta_model_path))
        delta_model_paths.append(str(delta_model_path))

    # 3) Финальный прогноз: blend baseline + delta-curve.
    blend = float(np.clip(args.delta_blend, 0.0, 1.0))
    y_pred_blended = np.clip((1.0 - blend) * baseline_curve + blend * delta_curve, 0.0, 1.0)
    pred_values = list(y_pred_blended.astype(float))
    pred_values_delta_curve = list(delta_curve.astype(float))

    y_pred = np.array(pred_values, dtype=float)
    y_true = np.array(true_values, dtype=float)
    abs_err = np.abs(y_pred - y_true)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(args.curve_points)),
            "pred_retention_base": baseline_curve,
            "pred_retention_delta_curve": np.array(pred_values_delta_curve, dtype=float),
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    spearman = float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman"))
    pearson = float(pd.Series(y_pred).corr(pd.Series(y_true), method="pearson"))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(abs_err))

    dataset_path = out_dir / "regressor_dataset.csv"
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
        "spearman": spearman,
        "pearson": pearson,
        "rmse": rmse,
        "mae": mae,
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
        "point_models_dir": str(points_dir),
        "point_models_count": len(point_model_paths),
        "delta_models_count": len(delta_model_paths),
        "delta_blend": float(np.clip(args.delta_blend, 0.0, 1.0)),
        "delta_max_step": float(delta_max_step),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Regressor LOO ===")
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
