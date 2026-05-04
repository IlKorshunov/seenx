from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from train.common.train_utils import make_feature_matrix
from train.common.train_utils import point_col as _point_col
from train.common.train_utils import safe_float as _safe_float
from train.loo.train_retention_regressor_loo import build_rows_with_targets_source, select_train_test
from train.tools.drive_feature_labeling_pipeline import DEFAULT_PARENT_FOLDER_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("LOO-эксперимент: собрать 30 размеченных видео с retention target, обучить CatBoostRanker на 29 и предсказать кривую retention на 30-м.")
    )
    parser.add_argument("--env-file", default=".env", help="Путь к .env.")
    parser.add_argument("--snapshot-dir", default="", help=("Локальный снапшот с данными (например drive_snapshot_90). Если задан и существует, загрузка с Drive не выполняется."))
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID, help="ID корневой папки с видео на Google Drive.")
    parser.add_argument("--limit-videos", type=int, default=30, help="Сколько видео включить в эксперимент.")
    parser.add_argument("--train-videos", type=int, default=29, help="Сколько видео использовать для обучения ranker.")
    parser.add_argument("--output-dir", default="ranker_experiment", help="Куда сохранить датасет/предсказания/метрики.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--eval-video-folder", default="", help="Явно выбрать eval-видео по имени папки (например YouTube ID).")
    parser.add_argument("--eval-drive-file-id", default="", help="Явно выбрать eval-видео по drive_file_id транскрипта.")
    parser.add_argument("--curve-points", type=int, default=50, help="Количество точек кривой retention для обучения/предсказания.")
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    df, train_df, test_df = select_train_test(rows, args)

    X_train_video = make_feature_matrix(train_df)
    X_test_video = make_feature_matrix(test_df)

    train_docs: list[dict[str, float]] = []
    train_y: list[float] = []
    train_group_id: list[int] = []

    for group_idx, (_, row) in enumerate(train_df.iterrows()):
        base = X_train_video.iloc[group_idx].to_dict()
        for bin_idx in range(args.curve_points):
            doc = dict(base)
            doc["bin_idx"] = float(bin_idx)
            doc["bin_pos_norm"] = float(bin_idx / max(1, args.curve_points - 1))
            train_docs.append(doc)
            train_y.append(_safe_float(row[_point_col(bin_idx)], 0.0))
            train_group_id.append(group_idx)

    X_train = pd.DataFrame(train_docs).fillna(0.0)
    y_train = np.array(train_y, dtype=float)
    group_id = np.array(train_group_id, dtype=int)

    test_base = X_test_video.iloc[0].to_dict()
    test_docs: list[dict[str, float]] = []
    true_curve: list[float] = []
    for bin_idx in range(args.curve_points):
        doc = dict(test_base)
        doc["bin_idx"] = float(bin_idx)
        doc["bin_pos_norm"] = float(bin_idx / max(1, args.curve_points - 1))
        test_docs.append(doc)
        true_curve.append(_safe_float(test_df.iloc[0][_point_col(bin_idx)], 0.0))
    X_test = pd.DataFrame(test_docs).fillna(0.0)
    y_true = np.array(true_curve, dtype=float)

    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric=f"NDCG:top={min(20, args.curve_points)}",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        random_seed=args.random_seed,
        task_type=getattr(args, "task_type", "GPU"),
        gpu_ram_part=getattr(args, "gpu_ram_part", 0.6),
        verbose=False,
    )
    model.fit(X_train, y_train, group_id=group_id)
    pred_raw = model.predict(X_test)

    pmin, pmax = float(pred_raw.min()), float(pred_raw.max())
    if pmax - pmin < 1e-12:
        pred_norm = np.zeros_like(pred_raw)
    else:
        pred_norm = (pred_raw - pmin) / (pmax - pmin)

    result_df = pd.DataFrame({"point_idx": list(range(args.curve_points)), "pred_score_raw": pred_raw, "pred_retention_norm": pred_norm, "true_retention": y_true})

    spearman = float(pd.Series(pred_norm).corr(pd.Series(y_true), method="spearman"))
    pearson = float(pd.Series(pred_norm).corr(pd.Series(y_true), method="pearson"))
    rmse = float(np.sqrt(np.mean((pred_norm - y_true) ** 2)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "ranker_dataset.csv"
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    metrics_path = out_dir / "metrics.json"
    model_path = out_dir / "catboost_ranker.cbm"

    df.to_csv(dataset_path, index=False)
    result_df.to_csv(pred_path, index=False)
    model.save_model(str(model_path))

    metrics = {
        "videos_total_with_target": len(rows),
        "videos_used": int(args.limit_videos),
        "train_videos": int(args.train_videos),
        "curve_points": int(args.curve_points),
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "spearman": spearman,
        "pearson": pearson,
        "rmse": rmse,
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Ranker LOO ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\n=== Holdout Prediction vs True ({args.curve_points} points) ===")
    print(
        result_df.to_string(
            index=False, formatters={"pred_score_raw": lambda x: f"{x:0.5f}", "pred_retention_norm": lambda x: f"{x:0.5f}", "true_retention": lambda x: f"{x:0.5f}"}
        )
    )
    return metrics


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
