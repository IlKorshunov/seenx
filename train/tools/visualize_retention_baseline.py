from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Визуализация baseline retention из baseline_curve.csv.")
    parser.add_argument("--baseline-dir", default="retention_baseline_90", help="Папка с baseline_curve.csv и baseline_summary.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.baseline_dir)
    curve_path = base_dir / "baseline_curve.csv"
    summary_path = base_dir / "baseline_summary.json"

    if not curve_path.exists():
        raise FileNotFoundError(f"Не найден baseline curve: {curve_path}")

    df = pd.read_csv(curve_path)
    required = {"point_frac", "baseline_retention_abs_direct", "baseline_retention_norm_mean", "baseline_retention_abs_from_norm"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"В {curve_path} не хватает колонок: {sorted(missing)}")

    point_frac = df["point_frac"].to_numpy(dtype=float)
    abs_direct = df["baseline_retention_abs_direct"].to_numpy(dtype=float)
    norm_mean = df["baseline_retention_norm_mean"].to_numpy(dtype=float)
    abs_from_norm = df["baseline_retention_abs_from_norm"].to_numpy(dtype=float)

    # 1) Основной график
    plt.figure(figsize=(10, 5))
    plt.plot(point_frac, abs_direct, marker="o", linewidth=2, label="Abs direct mean")
    plt.plot(point_frac, abs_from_norm, marker="o", linewidth=2, label="Abs from norm mean")
    plt.plot(point_frac, norm_mean, marker="o", linewidth=2, linestyle="--", label="Norm mean shape")
    plt.title("Retention Baseline (90 videos)")
    plt.xlabel("Video progress (0..1)")
    plt.ylabel("Retention")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_curve = base_dir / "baseline_curve_plot.png"
    plt.savefig(out_curve, dpi=150)
    plt.close()

    # 2) Разница абсолютов
    plt.figure(figsize=(10, 4))
    diff = abs_direct - abs_from_norm
    plt.bar(point_frac, diff, width=0.8 / max(1, len(point_frac)), color="#ff7f0e")
    plt.title("Difference: abs_direct - abs_from_norm")
    plt.xlabel("Video progress (0..1)")
    plt.ylabel("Difference")
    plt.xlim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_diff = base_dir / "baseline_abs_difference.png"
    plt.savefig(out_diff, dpi=150)
    plt.close()

    # 3) Summary-постер
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    level_mean = float(summary.get("baseline_level_mean", abs_direct[0] if len(abs_direct) else 0.0))
    tail_mean = float(summary.get("baseline_tail_mean", abs_direct[-1] if len(abs_direct) else 0.0))
    curves_used = int(summary.get("curves_used", 0))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    text = [
        "Retention Baseline Summary",
        f"Curves used: {curves_used}",
        f"Points: {len(point_frac)}",
        f"Level mean: {level_mean:.4f}",
        f"Tail mean:  {tail_mean:.4f}",
        f"Abs direct @end: {float(abs_direct[-1]):.4f}" if len(abs_direct) else "Abs direct @end: n/a",
        f"Abs from norm @end: {float(abs_from_norm[-1]):.4f}" if len(abs_from_norm) else "Abs from norm @end: n/a",
    ]
    ax.text(0.02, 0.95, "\n".join(text), va="top", ha="left", fontsize=12, family="monospace")
    out_summary = base_dir / "baseline_metrics_summary.png"
    plt.tight_layout()
    plt.savefig(out_summary, dpi=150)
    plt.close(fig)

    print("Saved baseline visualizations:")
    print(f"- {out_curve}")
    print(f"- {out_diff}")
    print(f"- {out_summary}")


if __name__ == "__main__":
    main()
