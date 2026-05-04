from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Визуализация результатов holdout-эксперимента retention ranker.")
    parser.add_argument("--experiment-dir", default="ranker_experiment", help="Папка с metrics.json и holdout_prediction_vs_true.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.experiment_dir)
    pred_path = exp_dir / "holdout_prediction_vs_true.csv"
    metrics_path = exp_dir / "metrics.json"

    if not pred_path.exists():
        raise FileNotFoundError(f"Не найден файл предсказаний: {pred_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Не найден файл метрик: {metrics_path}")

    df = pd.read_csv(pred_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    x_col = "point_idx" if "point_idx" in df.columns else "bin"
    required_cols = {x_col, "true_retention"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise RuntimeError(f"В {pred_path} не хватает колонок: {sorted(missing)}")
    pred_col = "pred_retention" if "pred_retention" in df.columns else "pred_retention_norm"
    if pred_col not in df.columns:
        raise RuntimeError(f"В {pred_path} не хватает колонки предсказания: ожидается pred_retention или pred_retention_norm")

    bins = df[x_col].to_numpy(dtype=float)
    # Нормализуем ось X в долю прогресса видео [0..1], чтобы график был
    # независим от количества точек кривой.
    if len(bins) <= 1:
        x = np.zeros_like(bins, dtype=float)
    else:
        x_max = float(np.max(bins))
        x = bins / max(1.0, x_max)
    pred = df[pred_col].to_numpy(dtype=float)
    true = df["true_retention"].to_numpy(dtype=float)
    abs_err = np.abs(pred - true)

    # 1) Кривые retention: pred vs true
    plt.figure(figsize=(10, 5))
    plt.plot(x, true, marker="o", linewidth=2, label="True retention")
    plt.plot(x, pred, marker="o", linewidth=2, label=f"Pred retention ({pred_col})")
    plt.title(f"Holdout retention curve: {metrics.get('test_video', 'unknown')}")
    plt.xlabel("Video progress (0..1)")
    plt.ylabel("Retention")
    plt.xlim(0.0, 1.0)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    curve_path = exp_dir / "holdout_curve_comparison.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()

    # 2) Абсолютная ошибка по бинам
    plt.figure(figsize=(10, 4))
    width = 0.8 / max(1, len(x))
    plt.bar(x, abs_err, width=width, color="#ff7f0e")
    plt.title("Absolute error by retention point")
    plt.xlabel("Video progress (0..1)")
    plt.ylabel("|pred - true|")
    plt.xlim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    err_path = exp_dir / "holdout_abs_error_by_bin.png"
    plt.savefig(err_path, dpi=150)
    plt.close()

    # 3) Краткий summary-постер с метриками
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    summary_lines = [
        "Retention Ranker Holdout Summary",
        f"Test video: {metrics.get('test_video', 'unknown')}",
        f"Videos used: {metrics.get('videos_used', 'n/a')} (train={metrics.get('train_videos', 'n/a')})",
        f"Spearman: {metrics.get('spearman', float('nan')):.4f}",
        f"Pearson:  {metrics.get('pearson', float('nan')):.4f}",
        f"RMSE:     {metrics.get('rmse', float('nan')):.4f}",
        f"Mean abs error: {float(abs_err.mean()):.4f}",
        f"Max abs error:  {float(abs_err.max()):.4f}",
    ]
    ax.text(0.02, 0.95, "\n".join(summary_lines), va="top", ha="left", fontsize=12, family="monospace")
    summary_path = exp_dir / "holdout_metrics_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close(fig)

    print("Saved visualizations:")
    print(f"- {curve_path}")
    print(f"- {err_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
