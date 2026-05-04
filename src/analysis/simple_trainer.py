from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import catboost as cb
import matplotlib.pyplot as plt
import optuna
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..utils.logger import Logger


logger = Logger(show=True).get_logger()
EXCLUDE = frozenset({"retention", "frame", "time"})


def _load_llm_features(vid, snapshot_dir):
    _LLM_ID = {"video_folder", "transcript_path", "drive_file_id"}
    for candidate in [os.path.join(snapshot_dir, vid, "features_llm.json"), os.path.join(snapshot_dir, vid, "transcripts", "features_llm.json")]:
        if not os.path.exists(candidate):
            continue
        try:
            flat = json.load(open(candidate, encoding="utf-8")).get("video_features_flat", {})
            if not isinstance(flat, dict) or not flat:
                continue
            return {
                (f"llm_{k}" if not str(k).startswith("llm_") else str(k)): float(v)
                for k, v in flat.items()
                if k not in _LLM_ID and not str(k).startswith("target__") and isinstance(v, (int, float))
            }
        except Exception:
            continue
    return None


def _broadcast_llm(df, llm):
    return df if llm is None else pd.concat([df, pd.DataFrame({k: [v] * len(df) for k, v in llm.items()}, index=df.index)], axis=1)


def _load_video_frames(output_dir, snapshot_dir="data"):
    csvs = sorted(glob.glob(os.path.join(output_dir, "*_features.csv")))
    if not csvs:
        raise FileNotFoundError(f"No *_features.csv in {output_dir!r}")
    frames = {}
    for path in csvs:
        vid = os.path.basename(path).replace("_features.csv", "")
        df = pd.read_csv(path, index_col=0)
        if "retention" not in df.columns:
            continue
        frames[vid] = _broadcast_llm(df, _load_llm_features(vid, snapshot_dir)).dropna(subset=["retention"])
        logger.info("  %s: %d rows, %d cols", vid, *frames[vid].shape)
    return frames


def _add_positional_features(df):
    n = len(df)
    t = np.linspace(0, 1, n)
    df = df.copy()
    df["time_pct"] = t
    df["time_pct_sq"] = t**2
    df["log_time"] = np.log1p(np.arange(n, dtype=float))
    return df


def _metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_pred - y_true)))
    mse = float(np.mean((y_pred - y_true) ** 2))
    ss_res, ss_tot = float(np.sum((y_true - y_pred) ** 2)), float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {"mae": round(mae, 3), "mse": round(mse, 3), "r2": round(1.0 - ss_res / ss_tot, 3) if ss_tot >= 1e-6 else float("nan")}


def _select_features(all_df, top_features=None):
    cols = all_df[[c for c in all_df.columns if c not in EXCLUDE]].astype(float).std().pipe(lambda s: s[s > 0.01]).index.tolist()
    if top_features and top_features < len(cols):
        gm = cb.CatBoostRegressor(iterations=300, depth=5, verbose=0, random_seed=42)
        gm.fit(all_df[cols].astype(float).fillna(0), all_df["retention"].values.astype(float))
        cols = [cols[i] for i in np.argsort(gm.get_feature_importance())[::-1][:top_features]]
    logger.info("Feature count: %d", len(cols))
    return cols


def _safe_X(df, feature_cols):
    return df.reindex(columns=feature_cols, fill_value=0).astype(float).fillna(0).values


def _loo_ridge(video_frames, feature_cols):
    ids = sorted(video_frames.keys())
    all_mae, all_r2 = [], []
    for held_out in ids:
        train_df = pd.concat([video_frames[v] for v in ids if v != held_out], ignore_index=True)
        val_df = video_frames[held_out]
        if train_df.empty or val_df.empty:
            continue
        X_tr, y_tr = _safe_X(train_df, feature_cols), train_df["retention"].values
        X_vl, y_vl = _safe_X(val_df, feature_cols), val_df["retention"].values
        scaler = StandardScaler().fit(X_tr)
        m = _metrics(y_vl, Ridge(alpha=10.0).fit(scaler.transform(X_tr), y_tr).predict(scaler.transform(X_vl)))
        all_mae.append(m["mae"])
        if np.isfinite(m["r2"]):
            all_r2.append(m["r2"])
        logger.info("LOO Ridge held-out=%s MAE=%.3f R2=%.3f", held_out, m["mae"], m["r2"])
    return {
        "model": "Ridge",
        "loo_mae_mean": round(float(np.mean(all_mae)), 3),
        "loo_r2_mean": round(float(np.mean(all_r2)), 3) if all_r2 else float("nan"),
        "n_folds": len(all_mae),
    }


def _loo_catboost(video_frames, feature_cols, iterations=500, depth=5, lr=0.05, optuna_use=False, optuna_trials=0):
    ids = sorted(video_frames.keys())
    folds = []
    for held_out in ids:
        train_df = pd.concat([video_frames[v] for v in ids if v != held_out], ignore_index=True)
        val_df = video_frames[held_out]
        if train_df.empty or val_df.empty:
            continue
        folds.append(
            (_safe_X(train_df, feature_cols), train_df["retention"].values.astype(float), _safe_X(val_df, feature_cols), val_df["retention"].values.astype(float), held_out)
        )
    if not folds:
        return {"model": "CatBoost", "loo_mae_mean": float("nan"), "loo_r2_mean": float("nan"), "n_folds": 0}
    if optuna_use and optuna_trials > 0:

        def objective(trial):
            it, d = trial.suggest_int("iterations", 100, 800), trial.suggest_int("depth", 3, 7)
            l = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
            return float(
                np.mean(
                    [
                        float(np.mean(np.abs(y_vl - cb.CatBoostRegressor(iterations=it, depth=d, learning_rate=l, verbose=0, random_seed=42).fit(X_tr, y_tr).predict(X_vl))))
                        for X_tr, y_tr, X_vl, y_vl, _ in folds
                    ]
                )
            )

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        iterations, depth, lr = (study.best_params["iterations"], study.best_params["depth"], study.best_params["learning_rate"])
    all_mae, all_r2 = [], []
    for X_tr, y_tr, X_vl, y_vl, held_out in folds:
        m = _metrics(y_vl, cb.CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=lr, verbose=0, random_seed=42).fit(X_tr, y_tr).predict(X_vl))
        all_mae.append(m["mae"])
        if np.isfinite(m["r2"]):
            all_r2.append(m["r2"])
        logger.info("LOO CatBoost held-out=%s MAE=%.3f R2=%.3f", held_out, m["mae"], m["r2"])
    return {
        "model": "CatBoost",
        "loo_mae_mean": round(float(np.mean(all_mae)), 3),
        "loo_r2_mean": round(float(np.mean(all_r2)), 3) if all_r2 else float("nan"),
        "n_folds": len(all_mae),
        "iterations": iterations,
        "depth": depth,
        "learning_rate": lr,
    }


def _train_final(video_frames, feature_cols, train_ids, val_ids, model_dir, **cb_params):
    tr = pd.concat([video_frames[v] for v in train_ids], ignore_index=True)
    vl = pd.concat([video_frames[v] for v in val_ids], ignore_index=True)
    X_tr = tr.reindex(columns=feature_cols, fill_value=0).astype(float).fillna(0)
    y_tr = tr["retention"].values.astype(float)
    X_vl = vl.reindex(columns=feature_cols, fill_value=0).astype(float).fillna(0)
    y_vl = vl["retention"].values.astype(float)
    model = cb.CatBoostRegressor(
        iterations=cb_params.get("iterations", 800),
        depth=cb_params.get("depth", 6),
        learning_rate=cb_params.get("learning_rate", 0.05),
        loss_function="RMSE",
        eval_metric="MAE",
        verbose=100,
        early_stopping_rounds=50,
        random_seed=42,
    )
    model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), use_best_model=True)
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "model_unified.cbm")
    model.save_model(save_path)
    top_pairs = sorted(zip(X_tr.columns, model.get_feature_importance(), strict=True), key=lambda t: -t[1])[:15]
    return {
        "model_path": save_path,
        "n_train": len(tr),
        "n_val": len(vl),
        "n_features": len(feature_cols),
        "n_trees": model.tree_count_,
        "train": _metrics(y_tr, model.predict(X_tr)),
        "val": _metrics(y_vl, model.predict(X_vl)),
        "top_features": {k: round(float(v), 2) for k, v in top_pairs},
        "model": model,
    }


def _plot_predictions(video_frames, model, feature_cols, val_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for vid in sorted(video_frames):
        df = video_frames[vid]
        y_true = df["retention"].values.astype(float)
        y_pred = model.predict(df.reindex(columns=feature_cols, fill_value=0).astype(float).fillna(0))
        split = "val" if vid in val_ids else "train"
        m = _metrics(y_true, y_pred)
        t = np.arange(len(y_true))
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1], sharex=True)
        a1.plot(t, y_true, color="#2196F3", lw=1.5, label="actual")
        a1.plot(t, y_pred, color="#FF5722", lw=1.5, label="predicted", alpha=0.8)
        a1.fill_between(t, y_true, y_pred, alpha=0.1, color="#9C27B0")
        a1.set(ylabel="Retention (%)", title=f"{vid} [{split}] MAE={m['mae']:.2f} R2={m['r2']:.3f}")
        a1.legend(fontsize=9)
        a1.grid(True, alpha=0.3)
        res = y_pred - y_true
        a2.fill_between(t, res, alpha=0.3, color="#4CAF50", where=res >= 0)
        a2.fill_between(t, res, alpha=0.3, color="#F44336", where=res < 0)
        a2.axhline(0, color="black", lw=0.5)
        a2.set(xlabel="sec", ylabel="error")
        a2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"pred_{vid}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)


def run(
    output_dir="output",
    results_dir="function_result/simple",
    top_features=None,
    iterations=800,
    depth=6,
    learning_rate=0.05,
    loo_optuna_use=False,
    loo_optuna_trials=0,
    val_ratio=0.2,
    snapshot_dir="data",
):
    model_dir, plots_dir = os.path.join(results_dir, "models"), os.path.join(results_dir, "plots")
    for d in (results_dir, model_dir, plots_dir):
        os.makedirs(d, exist_ok=True)
    raw = _load_video_frames(output_dir, snapshot_dir=snapshot_dir)
    video_frames = {vid: _add_positional_features(df) for vid, df in raw.items()}
    feature_cols = _select_features(pd.concat(video_frames.values(), ignore_index=True), top_features)
    ids = sorted(video_frames.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids, train_ids = set(ids[:n_val]), set(ids[n_val:])
    ridge_loo = _loo_ridge(video_frames, feature_cols)
    cb_loo = _loo_catboost(
        video_frames, feature_cols, iterations=min(iterations, 500), depth=min(depth, 5), lr=learning_rate, optuna_use=loo_optuna_use, optuna_trials=loo_optuna_trials
    )
    logger.info("LOO Ridge: MAE=%.3f R2=%.3f | CatBoost: MAE=%.3f R2=%.3f", ridge_loo["loo_mae_mean"], ridge_loo["loo_r2_mean"], cb_loo["loo_mae_mean"], cb_loo["loo_r2_mean"])
    final = _train_final(
        video_frames,
        feature_cols,
        train_ids,
        val_ids,
        model_dir,
        iterations=cb_loo.get("iterations", iterations),
        depth=cb_loo.get("depth", depth),
        learning_rate=cb_loo.get("learning_rate", learning_rate),
    )
    _plot_predictions(video_frames, final["model"], feature_cols, val_ids, plots_dir)
    summary = {
        "n_videos": len(video_frames),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "ridge_loo": ridge_loo,
        "catboost_loo": cb_loo,
        "final_train": final["train"],
        "final_val": final["val"],
        "top_features": final["top_features"],
    }
    with open(os.path.join(results_dir, "simple_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(
        [
            {"model": "Ridge LOO", **ridge_loo},
            {"model": "CatBoost LOO", **cb_loo},
            {"model": "CatBoost final (train)", **final["train"]},
            {"model": "CatBoost final (val)", **final["val"]},
        ]
    ).to_csv(os.path.join(results_dir, "simple_summary.csv"), index=False)
    print(f"  CatBoost LOO: MAE={cb_loo['loo_mae_mean']:.3f} R2={cb_loo['loo_r2_mean']:.3f}")
    print(f"  Final train: MAE={final['train']['mae']:.3f} R2={final['train']['r2']:.3f}")
    print(f"  Final val: MAE={final['val']['mae']:.3f} R2={final['val']['r2']:.3f}")
    print(f"  Top: {list(final['top_features'].keys())[:10]}\n  Results: {results_dir}\n{'=' * 60}\n")
    return summary


def main():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output_dir", default="output")
    p.add_argument("--results_dir", default=os.path.join(ROOT, "function_result", "simple"))
    p.add_argument("--top_features", type=int, default=None)
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--loo_optuna_use", action="store_true")
    p.add_argument("--loo_optuna_trials", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--snapshot_dir", default="data")
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
