"""Print experiment leaderboard and best-run diagnostics (used by run_all_experiments.sh)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."), help="Repository root (default: cwd)")
    args = p.parse_args()
    root = args.root.resolve()

    rows = []
    for pth in (root / "experiments").glob("**/metrics.json"):
        metrics = read_json(pth)
        if not isinstance(metrics, dict):
            continue

        if "per_video" in metrics:
            val_rmse = [float(info["rmse"]) for info in metrics["per_video"].values() if info.get("split") == "val"]
            score, score_name = float(sum(val_rmse) / len(val_rmse)), "mean_val_rmse"
        else:
            score, score_name = float(metrics["eval_rmse"]), "eval_rmse"

        rows.append({"dir": pth.parent.relative_to(root), "model": metrics.get("model", metrics.get("arch", "unknown")), "score": score, "score_name": score_name})

    if not rows:
        print("No metrics.json files found into experiments")
        return

    df_valid = pd.DataFrame(rows).sort_values("score", ascending=True).reset_index(drop=True)

    for i, r in df_valid.iterrows():
        print(f"{i + 1:2d}. {r['dir']} | model={r['model']} | {r['score_name']}={r['score']:.6f}")

    best = df_valid.iloc[0]
    best_dir = root / best["dir"]
    best_metrics = read_json(best_dir / "metrics.json") or {}

    print("best model")
    print(f"version: {best['dir']}")
    print(f"model: {best['model']}")
    print(f"metric: {best['score_name']}={best['score']:.6f}")

    per_video = best_metrics.get("per_video", {})
    worst = []
    if isinstance(per_video, dict):
        for vid, info in per_video.items():
            if not isinstance(info, dict):
                continue
            if info.get("split") != "val":
                continue
            worst.append((vid, float(info.get("rmse") or 0), float(info.get("mae") or 0)))
    worst.sort(key=lambda x: x[1], reverse=True)

    print("worst videos")
    if worst:
        for i, (vid, rmse, mae) in enumerate(worst[:3], 1):
            print(f"{i}. {vid}: rmse={rmse:.6f}, mae={mae:.6f}")
    else:
        print("No per_video validation found.")

    fi_path = best_dir / "feature_importance.csv"
    print("Feature importance for best model")
    if not fi_path.exists():
        print("feature_importance.csv not found")
    else:
        fi = pd.read_csv(fi_path)
        if "feature" not in fi.columns:
            print("feature_importance.csv has no need features column.")
        else:
            cand = []
            for c in fi.columns:
                if c == "feature":
                    continue
                s = pd.to_numeric(fi[c], errors="coerce")
                if s.notna().any():
                    cand.append(c)
            if not cand:
                print("No numeric importance column found.")
            else:
                imp_col = cand[0]
                tmp = fi[["feature", imp_col]].copy()
                tmp[imp_col] = pd.to_numeric(tmp[imp_col], errors="coerce")
                tmp = tmp.dropna().sort_values(imp_col, ascending=False)

                print(f"importance_column: {imp_col}")
                print("Top-5 most informative:")
                for i, (_, r) in enumerate(tmp.head(5).iterrows(), 1):
                    print(f"{i}. {r['feature']}: {float(r[imp_col]):.6f}")

                print("Bottom-5 least informative:")
                bot = tmp.sort_values(imp_col).head(5)
                for i, (_, row) in enumerate(bot.iterrows(), 1):
                    print(f"{i}. {row['feature']}: {float(row[imp_col]):.6f}")

if __name__ == "__main__":
    main()
