"""
Post-processing for LOO experiments: per-video prediction plots + feature importance + leaderboard.

Usage:
    python train/loo_report.py --loo-root experiments/loo --snapshot-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--loo-root", type=Path, default=Path("experiments/loo"))
    p.add_argument("--snapshot-dir", type=Path, default=Path("data"))
    p.add_argument("--curve-points", type=int, default=20)
    return p.parse_args()


def _load_predictions(exp_dir: Path, curve_points: int) -> dict[str, np.ndarray] | None:
    csv = exp_dir / "holdout_prediction_vs_true.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    pred_col = next((c for c in ("pred_retention", "predicted", "pred") if c in df.columns), None)
    true_col = next((c for c in ("true_retention", "true", "actual") if c in df.columns), None)
    if pred_col is None or true_col is None:
        return None
    return {"pred": df[pred_col].values.astype(float), "true": df[true_col].values.astype(float)}


def _plot_prediction(pred: np.ndarray, true: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(true))
    ax.plot(x, true * 100, "b-", linewidth=2, label="True retention")
    ax.plot(x, pred * 100, "r--", linewidth=2, label="Predicted")
    ax.fill_between(x, true * 100, pred * 100, alpha=0.15, color="red")
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    ax.set_title(f"{title}\nRMSE={rmse:.4f}  MAE={mae:.4f}", fontsize=11)
    ax.set_xlabel("Curve point")
    ax.set_ylabel("Retention (%)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_feature_importance(importances: dict[str, float], title: str, out_path: Path, top_k: int = 30):
    if not importances:
        return
    sorted_feats = sorted(importances.items(), key=lambda x: -x[1])[:top_k]
    names = [f[0] for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.3)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color="#2196F3")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"{title} (top {top_k})", fontsize=11)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_leaderboard(results: list[dict[str, Any]], out_path: Path):
    if not results:
        return
    results = sorted(results, key=lambda x: x.get("rmse", 999))
    names = [r["name"] for r in results]
    rmses = [r.get("rmse", 0) for r in results]
    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.4)))
    colors = ["#4CAF50" if i == 0 else "#2196F3" for i in range(len(names))]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, rmses, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(rmses):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_title("LOO Experiment Leaderboard (RMSE, lower is better)", fontsize=12)
    ax.set_xlabel("RMSE")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _compute_catboost_fi(exp_dir: Path) -> dict[str, float]:
    """Extract feature importance from CatBoost .cbm models in the experiment."""
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        return {}
    model_dirs = [exp_dir / "point_models", exp_dir / "models", exp_dir]
    cbm_files = []
    for d in model_dirs:
        if d.exists():
            cbm_files.extend(d.glob("*.cbm"))
    if not cbm_files:
        return {}
    importances: dict[str, float] = {}
    count = 0
    for cbm_path in cbm_files:
        try:
            m = CatBoostRegressor()
            m.load_model(str(cbm_path))
            fi = m.get_feature_importance()
            fnames = m.feature_names_
            if fnames and len(fnames) == len(fi):
                for name, val in zip(fnames, fi, strict=True):
                    importances[name] = importances.get(name, 0.0) + float(val)
                count += 1
        except Exception:
            continue
    if count > 0:
        importances = {k: v / count for k, v in importances.items()}
    return importances


def main():
    args = parse_args()
    loo_root = args.loo_root
    if not loo_root.exists():
        print(f"LOO root not found: {loo_root}")
        return

    leaderboard = []

    for exp_dir in sorted(loo_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        print(f"[report] Processing {exp_name}...")

        metrics_path = exp_dir / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        preds = _load_predictions(exp_dir, args.curve_points)
        if preds is not None:
            video_name = metrics.get("test_video", exp_name)
            videos_dir = exp_dir / "videos" / video_name
            _plot_prediction(preds["pred"], preds["true"], f"{exp_name} — {video_name}", videos_dir / "prediction.png")
            _plot_prediction(preds["pred"], preds["true"], f"{exp_name} — {video_name}", exp_dir / "prediction.png")
            print(f"  -> prediction plot: {exp_dir / 'prediction.png'}")

        fi = _compute_catboost_fi(exp_dir)
        if fi:
            _plot_feature_importance(fi, f"Feature Importance — {exp_name}", exp_dir / "feature_importance.png")
            fi_df = pd.DataFrame(sorted(fi.items(), key=lambda x: -x[1]), columns=["feature", "importance"])
            fi_df.to_csv(exp_dir / "feature_importance.csv", index=False)
            print(f"  -> feature_importance: {exp_dir / 'feature_importance.png'}")

        rmse = metrics.get("rmse")
        mae = metrics.get("mae")
        if rmse is not None:
            leaderboard.append({"name": exp_name, "rmse": float(rmse), "mae": float(mae) if mae else None})

    if leaderboard:
        _plot_leaderboard(leaderboard, loo_root / "leaderboard.png")
        print(f"\n[report] Leaderboard: {loo_root / 'leaderboard.png'}")
        lb_df = pd.DataFrame(sorted(leaderboard, key=lambda x: x["rmse"]))
        lb_df.to_csv(loo_root / "leaderboard.csv", index=False)
        print(f"[report] Leaderboard CSV: {loo_root / 'leaderboard.csv'}")
        print("\n== LOO Leaderboard (RMSE) ==")
        for i, r in enumerate(sorted(leaderboard, key=lambda x: x["rmse"]), 1):
            mae_s = f"  MAE={r['mae']:.4f}" if r.get("mae") else ""
            print(f"  {i:2d}. {r['name']:<30} RMSE={r['rmse']:.4f}{mae_s}")


if __name__ == "__main__":
    main()
