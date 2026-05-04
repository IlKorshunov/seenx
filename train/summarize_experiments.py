"""Print experiment leaderboard and best-run diagnostics (used by run_all_experiments.sh)."""

from __future__ import annotations

import argparse
import json
import sys
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
    exp_root = root / "experiments"

    rows = []
    for pth in exp_root.glob("**/metrics.json"):
        m = read_json(pth)
        if not isinstance(m, dict):
            continue

        rel_dir = pth.parent.relative_to(root).as_posix()
        model_name = m.get("model", m.get("arch", "unknown"))
        score = None
        score_name = None
        per_video = m.get("per_video", {})

        if isinstance(per_video, dict) and per_video:
            val_rmse = []
            for vid, info in per_video.items():
                if not isinstance(info, dict):
                    continue
                if info.get("split") == "val" and isinstance(info.get("rmse"), (int, float)):
                    val_rmse.append(float(info["rmse"]))
            if val_rmse:
                score = float(sum(val_rmse) / len(val_rmse))
                score_name = "mean_val_rmse"

        if score is None and isinstance(m.get("eval_rmse"), (int, float)):
            score = float(m["eval_rmse"])
            score_name = "eval_rmse"

        if score is None and isinstance(m.get("mean_eval_rmse"), (int, float)):
            score = float(m["mean_eval_rmse"])
            score_name = "mean_eval_rmse"

        if score is None and isinstance(m.get("best_val_loss"), (int, float)):
            score = float(m["best_val_loss"])
            score_name = "best_val_loss"

        rows.append({"dir": rel_dir, "model": model_name, "score": score, "score_name": score_name or "unknown"})

    if not rows:
        print("No metrics.json files found under experiments/", file=sys.stderr)
        raise SystemExit(0)

    df = pd.DataFrame(rows)
    df_valid = df[df["score"].notna()].sort_values("score", ascending=True).reset_index(drop=True)

    print("\n== Experiment leaderboard (lower is better) ==")
    if df_valid.empty:
        print("No experiments with numeric validation score.")
        raise SystemExit(0)

    for i, r in df_valid.iterrows():
        print(f"{i + 1:2d}. {r['dir']} | model={r['model']} | {r['score_name']}={r['score']:.6f}")

    best = df_valid.iloc[0]
    best_dir = root / best["dir"]
    best_metrics = read_json(best_dir / "metrics.json") or {}

    print("\n== Best model/version ==")
    print(f"version: {best['dir']}")
    print(f"model:   {best['model']}")
    print(f"metric:  {best['score_name']}={best['score']:.6f}")

    per_video = best_metrics.get("per_video", {})
    worst = []
    if isinstance(per_video, dict):
        for vid, info in per_video.items():
            if not isinstance(info, dict):
                continue
            if info.get("split") != "val":
                continue
            rmse = info.get("rmse")
            mae = info.get("mae")
            if isinstance(rmse, (int, float)):
                worst.append((vid, float(rmse), float(mae) if isinstance(mae, (int, float)) else None))
    worst.sort(key=lambda x: x[1], reverse=True)

    print("\n== Worst-3 validation videos (best model) ==")
    if worst:
        for i, (vid, rmse, mae) in enumerate(worst[:3], 1):
            mae_txt = f", mae={mae:.6f}" if mae is not None else ""
            print(f"{i}. {vid}: rmse={rmse:.6f}{mae_txt}")
    else:
        print("No per_video validation breakdown found.")

    fi_path = best_dir / "feature_importance.csv"
    print("\n== Feature importance (best model) ==")
    if not fi_path.exists():
        print("feature_importance.csv not found.")
    else:
        fi = pd.read_csv(fi_path)
        if "feature" not in fi.columns:
            print("feature_importance.csv has no 'feature' column.")
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
                print("\nTop-5 most informative:")
                for i, (_, r) in enumerate(tmp.head(5).iterrows(), 1):
                    print(f"{i}. {r['feature']}: {float(r[imp_col]):.6f}")

                print("\nTop-5 least informative:")
                for i, (_, r) in enumerate(tmp.tail(5).sort_values(imp_col, ascending=True).iterrows(), 1):
                    print(f"{i}. {r['feature']}: {float(r[imp_col]):.6f}")


if __name__ == "__main__":
    main()
