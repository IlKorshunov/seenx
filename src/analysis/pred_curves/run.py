from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .curves import CURVE_DEFS, MAX_PARAMS, N_POINTS, fit_curve, reconstruct, resample
from .features import build_feature_df
from .trainer import feature_importance, loo_predict, predict_params, train_models
from .visualize import plot_examples, plot_mae_comparison, plot_per_video_mae, plot_summary_table


def _load_retentions(data_dir: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for vid_dir in sorted(data_dir.iterdir()):
        if not vid_dir.is_dir():
            continue
        csv = vid_dir / "retention.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            if {"time_ratio", "audience_watch_ratio"} <= set(df.columns):
                vals = np.interp(np.linspace(0, 1, N_POINTS), df["time_ratio"].values, df["audience_watch_ratio"].values * 100.0)
                result[vid_dir.name] = vals
                continue
        for jname in ("retention.json", "retention_parsed.json"):
            jp = vid_dir / jname
            if not jp.exists():
                continue
            raw = json.loads(jp.read_text(encoding="utf-8"))
            if isinstance(raw, list) and raw:
                vals = np.array([float(r.get("audienceWatchRatio", r) if isinstance(r, dict) else r) for r in raw])
                if vals.max() <= 1.5:
                    vals *= 100.0
                result[vid_dir.name] = resample(vals, N_POINTS)
                break
    return result


def _discover_vids(data_dir: Path, emb_dir: Path, output_dir: Path) -> list[str]:
    ids: set[str] = set()
    for d in (data_dir, emb_dir):
        if d.is_dir():
            ids.update(c.name for c in d.iterdir() if c.is_dir())
    if output_dir.is_dir():
        ids.update(p.stem.removesuffix("_features") for p in output_dir.glob("*_features.csv"))
    return sorted(ids)


def main() -> None:
    p = argparse.ArgumentParser(description="Predict retention curve parameters")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--embeddings-dir", type=Path, default=Path("embeddings"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--out", type=Path, default=Path("src/analysis/pred_curves/results"))
    p.add_argument("--emb-pca-dim", type=int, default=16)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--use-loo", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = args.random_state

    vids = _discover_vids(args.data_dir, args.embeddings_dir, args.output_dir)
    retentions = _load_retentions(args.data_dir)
    vids_ret = sorted(v for v in vids if v in retentions)
    print(f"[data] {len(vids)} videos, {len(vids_ret)} with retention")
    assert vids_ret, "No videos with retention data"

    X = build_feature_df(vids, args.data_dir, args.embeddings_dir, args.output_dir, args.emb_pca_dim, rng)
    print(f"[features] {X.shape[1]} features")
    X_ret = X.loc[vids_ret]

    param_rows: list[dict] = []
    mae_rows: list[dict] = []
    examples: list[dict] = []

    for ctype, cdef in CURVE_DEFS.items():
        names = cdef["names"]
        n_p = cdef["n"]

        gt = np.full((len(vids_ret), n_p), np.nan)
        for i, vid in enumerate(vids_ret):
            params = fit_curve(ctype, retentions[vid])
            if params is not None and len(params) == n_p:
                gt[i] = params

        valid = np.all(np.isfinite(gt), axis=1)
        n_valid = int(valid.sum())
        print(f"[{ctype}] fitted {n_valid}/{len(vids_ret)}")

        if n_valid < 5:
            continue

        for i, vid in enumerate(vids_ret):
            row: dict = {"video_id": vid, "curve_type": ctype, "source": "fitted"}
            for j in range(MAX_PARAMS):
                row[f"p{j}"] = round(float(gt[i, j]), 6) if j < n_p and np.isfinite(gt[i, j]) else None
            if valid[i]:
                fitted_c = reconstruct(ctype, gt[i], N_POINTS)
                row["mae"] = round(float(np.mean(np.abs(retentions[vid] - fitted_c))), 4)
            param_rows.append(row)

        models = train_models(X_ret[valid], gt[valid], names, args.iterations, args.depth, args.lr, rng)

        fi = feature_importance(models, X.columns.tolist())
        if not fi.empty:
            fi.to_csv(args.out / f"importance_{ctype}.csv", index=False)

        pred_all = predict_params(models, X, names)
        for i, vid in enumerate(vids):
            row = {"video_id": vid, "curve_type": ctype, "source": "predicted"}
            for j in range(MAX_PARAMS):
                row[f"p{j}"] = round(float(pred_all[i, j]), 6) if j < n_p and np.isfinite(pred_all[i, j]) else None
            param_rows.append(row)

        if args.use_loo:
            eval_params = loo_predict(X_ret[valid], gt[valid], names, args.iterations, args.depth, args.lr, rng)
        else:
            eval_params = predict_params(models, X_ret[valid], names)

        eval_vids = [v for v, m in zip(vids_ret, valid, strict=True) if m]
        for i, vid in enumerate(eval_vids):
            fitted_c = reconstruct(ctype, gt[valid][i], N_POINTS)
            pp = eval_params[i]
            if np.any(np.isnan(pp)):
                continue
            pred_c = reconstruct(ctype, pp, N_POINTS)
            actual = retentions[vid]
            mf = round(float(np.mean(np.abs(actual - fitted_c))), 4)
            mp = round(float(np.mean(np.abs(actual - pred_c))), 4)
            mae_rows.append({"video_id": vid, "curve_type": ctype, "mae_fitted": mf, "mae_predicted": mp})
            if len([e for e in examples if e["curve_type"] == ctype]) < 3:
                examples.append({"video_id": vid, "curve_type": ctype, "actual": actual, "fitted": fitted_c, "predicted": pred_c})

        ct_mae = [r for r in mae_rows if r["curve_type"] == ctype]
        if ct_mae:
            fit_m = np.mean([r["mae_fitted"] for r in ct_mae])
            pred_m = np.mean([r["mae_predicted"] for r in ct_mae])
            print(f"  MAE fitted={fit_m:.3f}  predicted={pred_m:.3f}")

    pd.DataFrame(param_rows).to_csv(args.out / "curve_params.csv", index=False)

    mae_df = pd.DataFrame(mae_rows)
    if not mae_df.empty:
        mae_df.to_csv(args.out / "mae_results.csv", index=False)
        plot_mae_comparison(mae_df, args.out)
        plot_per_video_mae(mae_df, args.out)
        plot_examples(examples, args.out)
        plot_summary_table(mae_df, args.out)

    print(f"\nResults → {args.out}")


if __name__ == "__main__":
    main()
