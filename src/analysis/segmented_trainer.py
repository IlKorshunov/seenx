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
from scipy.optimize import curve_fit

from ..utils.logger import Logger


logger = Logger(show=True).get_logger()
FEATURE_EXCLUDE_COLS = frozenset({"retention", "frame", "time"})
SEGMENT_NAMES = ("first_drop", "dip", "main")
SEGMENT_COLORS = {"first_drop": "#F44336", "dip": "#FF9800", "main": "#2196F3"}
_C_FALLBACK_RATIO = 0.15
CURVE_TYPES = ("hill", "double_exp", "weibull")


def hill_curve(x, a, b, c, d):
    return d + (a - d) / (1.0 + np.power(x / c, b))


def double_exp_curve(x, a, b, c, d, e):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e


def weibull_curve(x, d, lam, k):
    return d * np.exp(-np.power(x / lam, k))


def fit_hill_curve(time_sec, retention):
    y, n = np.clip(retention, 0.0, 100.0), len(retention)
    c_max = _C_FALLBACK_RATIO * n
    popt, _ = curve_fit(
        hill_curve, time_sec, y, p0=[float(y[-1]), 0.8, min(max(1.0, float(n * 0.20)), c_max), float(y[0])], bounds=([0, 0.01, 1, 0], [100, 20, c_max, 100]), maxfev=8000
    )
    return hill_curve(time_sec, *popt), np.array(popt, dtype=float)


def fit_double_exp(time_sec, retention):
    y = np.clip(retention, 0.0, 100.0)
    drop = float(y[0] - y[-1])
    try:
        popt, _ = curve_fit(
            double_exp_curve, time_sec, y, p0=[drop * 0.6, 0.05, drop * 0.3, 0.005, float(y[-1])], bounds=([0, 1e-4, 0, 1e-5, 0], [100, 1.0, 100, 0.5, 100]), maxfev=8000
        )
        return double_exp_curve(time_sec, *popt), np.array(popt, dtype=float)
    except RuntimeError:
        return fit_hill_curve(time_sec, retention)


def fit_weibull(time_sec, retention):
    y, n = np.clip(retention, 0.0, 100.0), len(time_sec)
    try:
        popt, _ = curve_fit(weibull_curve, time_sec + 1.0, y, p0=[float(y[0]), float(n * 0.3), 0.5], bounds=([0, 1, 0.01], [100, n * 2, 5.0]), maxfev=8000)
        return weibull_curve(time_sec + 1.0, *popt), np.array(popt, dtype=float)
    except RuntimeError:
        return fit_hill_curve(time_sec, retention)


def fit_curve(time_sec, retention, curve_type="hill"):
    if curve_type == "double_exp":
        smooth, params = fit_double_exp(time_sec, retention)
    elif curve_type == "weibull":
        smooth, params = fit_weibull(time_sec, retention)
    else:
        smooth, params = fit_hill_curve(time_sec, retention)
        curve_type = "hill"
    return smooth, params, curve_type


def segment_video(time_sec, retention, dip_threshold=5.0, curve_type="hill"):
    n = len(time_sec)
    smooth, params, curve_used = fit_curve(time_sec, retention, curve_type)
    inflection = params[2] if (curve_used == "hill" and len(params) >= 4) else float(time_sec[np.argmax(np.abs(np.gradient(np.gradient(smooth))))])
    labels = np.full(n, "main", dtype=object)
    fd_mask = time_sec < inflection
    labels[fd_mask] = "first_drop"
    labels[(~fd_mask) & (retention - smooth < -dip_threshold)] = "dip"
    return (
        labels,
        smooth,
        {
            "converged": True,
            "curve_type": curve_used,
            "params": {f"p{i}": round(float(v), 4) for i, v in enumerate(params)},
            "inflection_sec": inflection,
            "n_first_drop": int(fd_mask.sum()),
            "n_dip": int((labels == "dip").sum()),
            "n_main": int((labels == "main").sum()),
        },
    )


def prune_features(df, target_col="retention", corr_threshold=0.95, var_threshold=1e-4):
    feat_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE_COLS | {"_segment"}]
    keep = df[feat_cols].astype(float).std().pipe(lambda s: s[s > var_threshold**0.5]).index.tolist()
    if not keep:
        return []
    X = df[keep].astype(float)
    corr_mat, target_corr = X.corr().abs(), X.corrwith(df[target_col].astype(float)).abs()
    to_drop = set()
    for i, ci in enumerate(keep):
        for cj in keep[i + 1 :]:
            if ci in to_drop or cj in to_drop:
                continue
            if corr_mat.loc[ci, cj] > corr_threshold:
                to_drop.add(cj if target_corr.get(ci, 0) >= target_corr.get(cj, 0) else ci)
    return [c for c in keep if c not in to_drop]


def _regression_metrics(y_true, y_pred):
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res, ss_tot = float(np.sum((y_true - y_pred) ** 2)), float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {"mse": round(mse, 3), "mae": round(mae, 3), "r2": round(1.0 - ss_res / ss_tot, 3) if ss_tot >= 1e-6 else float("nan")}


def train_one_segment(segment_name, X_train, y_train, X_val, y_val, model_dir, iterations=800, depth=6, learning_rate=0.05, early_stopping_rounds=50, top=10, sample_weight=None):
    model = cb.CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="RMSE",
        eval_metric="MAE",
        verbose=100,
        early_stopping_rounds=early_stopping_rounds,
        random_seed=42,
    )
    eval_set = (X_val, y_val) if len(X_val) > 0 else None
    pool_tr = cb.Pool(X_train, y_train, weight=sample_weight)
    model.fit(pool_tr, eval_set=eval_set, use_best_model=(eval_set is not None))
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"model_{segment_name}.cbm")
    model.save_model(save_path)
    top_pairs = sorted(zip(X_train.columns, model.get_feature_importance(), strict=True), key=lambda t: -t[1])[:top]
    result = {
        "segment": segment_name,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": X_train.shape[1],
        "n_trees": model.tree_count_,
        "model_path": save_path,
        "train": _regression_metrics(y_train, model.predict(X_train)),
        "top": {k: round(float(v), 2) for k, v in top_pairs},
        "skipped": False,
        "model": model,
    }
    if eval_set:
        result["val"] = _regression_metrics(y_val, model.predict(X_val))
    return result


def _video_sample_weights(video_frames, ids, feature_cols, segment):
    """Compute per-row weights so each video contributes equally to training loss."""
    weights_parts = []
    for v in ids:
        seg_df = video_frames[v][video_frames[v]["_segment"] == segment]
        n = len(seg_df)
        if n > 0:
            weights_parts.append(np.full(n, 1.0 / n))
    if not weights_parts:
        return None
    w = np.concatenate(weights_parts)
    w *= len(w) / w.sum()
    return w


def _loo_folds(video_frames, feature_cols, segment):
    ids = sorted(video_frames.keys())
    folds = []
    for held_out in ids:
        train_ids = [v for v in ids if v != held_out]
        train_parts = [video_frames[v][video_frames[v]["_segment"] == segment] for v in train_ids]
        train_parts = [p for p in train_parts if not p.empty]
        val_df = video_frames[held_out][video_frames[held_out]["_segment"] == segment]
        if not train_parts or val_df.empty:
            continue
        sw = _video_sample_weights(video_frames, train_ids, feature_cols, segment)
        train_df = pd.concat(train_parts, ignore_index=True)
        folds.append(
            (
                train_df[feature_cols].astype(float).fillna(0).values,
                train_df["retention"].values.astype(float),
                val_df[feature_cols].astype(float).fillna(0).values,
                val_df["retention"].values.astype(float),
                held_out,
                sw,
            )
        )
    return folds


def _loo_eval(folds, segment, iterations, depth, learning_rate, log=True):
    all_maes, all_r2s = [], []
    for X_tr, y_tr, X_vl, y_vl, held_out, sw in folds:
        pool_tr = cb.Pool(X_tr, y_tr, weight=sw)
        pred = cb.CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=learning_rate, loss_function="RMSE", verbose=0, random_seed=42).fit(pool_tr).predict(X_vl)
        m = _regression_metrics(y_vl, pred)
        all_maes.append(m["mae"])
        if np.isfinite(m["r2"]):
            all_r2s.append(m["r2"])
    return all_maes, all_r2s


def _loo_cv(video_frames, feature_cols, segment, loo_iterations=500, loo_depth=5, loo_learning_rate=0.05, loo_optuna_use=False, loo_optuna_trials=0):
    folds = _loo_folds(video_frames, feature_cols, segment)
    if not folds:
        return {"loo_mae_mean": float("nan"), "loo_mae_std": float("nan"), "loo_r2_mean": float("nan"), "n_folds": 0}
    if loo_optuna_use:

        def objective(trial):
            maes, _ = _loo_eval(
                folds, segment, trial.suggest_int("iterations", 200, 900), trial.suggest_int("depth", 3, 8), trial.suggest_float("learning_rate", 0.01, 0.2, log=True), log=False
            )
            return float(np.mean(maes)) if maes else float("inf")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=max(1, loo_optuna_trials), show_progress_bar=False)
        loo_iterations, loo_depth, loo_learning_rate = (study.best_params["iterations"], study.best_params["depth"], study.best_params["learning_rate"])
    all_maes, all_r2s = _loo_eval(folds, segment, loo_iterations, loo_depth, loo_learning_rate)
    return {
        "loo_mae_mean": round(float(np.mean(all_maes)), 3),
        "loo_mae_std": round(float(np.std(all_maes)), 3),
        "loo_r2_mean": round(float(np.mean(all_r2s)), 3) if all_r2s else float("nan"),
        "n_folds": len(all_maes),
        "loo_iterations": loo_iterations,
        "loo_depth": loo_depth,
        "loo_learning_rate": loo_learning_rate,
    }


def _optuna_full_tune(raw_frames, feature_cols, n_trials=5, loo_subsample=20):
    ids = sorted(raw_frames.keys())
    rng = np.random.default_rng(42)
    sub_ids = list(rng.choice(ids, size=min(loo_subsample, len(ids)), replace=False)) if len(ids) > loo_subsample else ids
    logger.info("Optuna full tune: %d trials, LOO on %d/%d videos", n_trials, len(sub_ids), len(ids))

    def objective(trial):
        dip_t = trial.suggest_float("dip_threshold", 1.0, 15.0)
        iters, dep = trial.suggest_int("iterations", 150, 500), trial.suggest_int("depth", 3, 7)
        lr = trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
        frames = {}
        for vid, df in raw_frames.items():
            labels, _, _ = segment_video(np.arange(len(df), dtype=float), df["retention"].values.astype(float), dip_t)
            dfx = df.copy()
            dfx["_segment"] = labels
            frames[vid] = dfx
        per_video_maes = []
        for seg in SEGMENT_NAMES:
            for held_out in sub_ids:
                train_ids_seg = [v for v in ids if v != held_out]
                tr = pd.concat([frames[v][frames[v]["_segment"] == seg] for v in train_ids_seg], ignore_index=True)
                vl = frames[held_out][frames[held_out]["_segment"] == seg]
                if tr.empty or vl.empty:
                    continue
                sw = _video_sample_weights(frames, train_ids_seg, feature_cols, seg)
                y_vl = vl["retention"].values.astype(float)
                pool_tr = cb.Pool(tr[feature_cols].astype(float).fillna(0).values, tr["retention"].values.astype(float), weight=sw)
                pred = (
                    cb.CatBoostRegressor(iterations=iters, depth=dep, learning_rate=lr, loss_function="RMSE", verbose=0, random_seed=42)
                    .fit(pool_tr)
                    .predict(vl[feature_cols].astype(float).fillna(0).values)
                )
                per_video_maes.append(float(np.mean(np.abs(y_vl - pred))))
        return float(np.mean(per_video_maes)) if per_video_maes else float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("Full tune best: %s (MAE=%.3f)", study.best_params, study.best_value)
    return study.best_params


def _load_llm_features(vid, snapshot_dir):
    _LLM_ID = {"video_folder", "transcript_path", "drive_file_id"}
    for candidate in [os.path.join(snapshot_dir, vid, "features_llm.json"), os.path.join(snapshot_dir, vid, "transcripts", "features_llm.json")]:
        if not os.path.exists(candidate):
            continue
        try:
            flat = json.load(open(candidate, encoding="utf-8")).get("video_features_flat", {})
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
    frames = {}
    for path in csvs:
        vid = os.path.basename(path).replace("_features.csv", "")
        df = pd.read_csv(path, index_col=0)
        frames[vid] = _broadcast_llm(df, _load_llm_features(vid, snapshot_dir)).dropna(subset=["retention"])
    logger.info("Loaded %d videos", len(frames))
    return frames


def _split_ids(video_frames, val_ratio):
    ids = sorted(video_frames.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return set(ids[n_val:]), set(ids[:n_val])


def _segment_frames(video_frames, dip_threshold, metrics_dir, curve_type="hill"):
    seg_dir = os.path.join(metrics_dir, "segmentation")
    fit_mae_records = []
    for vid, df in video_frames.items():
        t, ret = np.arange(len(df), dtype=float), df["retention"].values.astype(float)
        labels, smooth, meta = segment_video(t, ret, dip_threshold, curve_type=curve_type)
        dfx = df.copy()
        dfx["_segment"] = labels
        video_frames[vid] = dfx
        total_mae, seg_mae = _plot_segmentation(vid, t, ret, labels, smooth, meta, seg_dir)
        fit_mae_records.append(
            {"video_id": vid, "total_mae": round(total_mae, 3), "n_points": len(ret), **{f"mae_{s}": round(seg_mae.get(s, float("nan")), 3) for s in SEGMENT_NAMES}}
        )
    common = sorted(set.intersection(*[set(df.columns) for df in video_frames.values()]))
    logger.info("Segmented %d videos", len(video_frames))
    return {vid: df[common] for vid, df in video_frames.items()}, fit_mae_records


def _select_features(train_df, skip_prune, top_features):
    cols = [c for c in train_df.columns if c not in FEATURE_EXCLUDE_COLS | {"_segment"}] if skip_prune else prune_features(train_df)
    if top_features and top_features < len(cols):
        gm = cb.CatBoostRegressor(iterations=300, depth=5, verbose=0, random_seed=42)
        gm.fit(train_df[cols].astype(float).fillna(0), train_df["retention"].values.astype(float))
        cols = [cols[i] for i in np.argsort(gm.get_feature_importance())[::-1][:top_features]]
    logger.info("Features: %d", len(cols))
    return cols


def _curve_fit_mae(retention, smooth, labels):
    total_mae = float(np.mean(np.abs(retention - smooth)))
    seg_mae = {}
    for seg in SEGMENT_NAMES:
        mask = labels == seg
        if mask.any():
            seg_mae[seg] = float(np.mean(np.abs(retention[mask] - smooth[mask])))
    return total_mae, seg_mae


def _plot_segmentation(video_id, time_sec, retention, labels, smooth, meta, output_dir):
    total_mae, seg_mae = _curve_fit_mae(retention, smooth, labels)
    fig, ax = plt.subplots(figsize=(14, 5))
    for seg in SEGMENT_NAMES:
        mask = labels == seg
        seg_m = seg_mae.get(seg)
        lbl = f"{seg} (n={mask.sum()}, MAE={seg_m:.2f})" if seg_m is not None else f"{seg} (n={mask.sum()})"
        ax.scatter(time_sec[mask], retention[mask], s=8, alpha=0.6, color=SEGMENT_COLORS[seg], label=lbl, zorder=3)
    ax.plot(time_sec, smooth, color="#212121", linewidth=1.8, label=f"{meta.get('curve_type', 'hill')} curve", zorder=4)
    if meta.get("curve_type") == "hill":
        ax.axvline(meta["inflection_sec"], color="#9C27B0", linestyle="--", linewidth=1, label=f"inflection={meta['inflection_sec']:.0f}s")
    ax.set(
        xlabel="Time (sec)",
        ylabel="Retention (%)",
        title=f"{video_id} [{meta.get('curve_type', 'hill')}]  MAE={total_mae:.2f}  ({' '.join(f'{k}={v}' for k, v in meta['params'].items())})",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"seg_{video_id}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return total_mae, seg_mae


def _plot_curve_fit_mae(fit_mae_records, output_dir):
    if not fit_mae_records:
        return
    df = pd.DataFrame(fit_mae_records).sort_values("total_mae", ascending=True)

    overall_mae = float(np.mean(df["total_mae"]))
    seg_means = {s: float(df[f"mae_{s}"].dropna().mean()) for s in SEGMENT_NAMES}

    avg_row = pd.DataFrame(
        [{"video_id": "AVERAGE", "total_mae": round(overall_mae, 3), "n_points": int(df["n_points"].sum()), **{f"mae_{s}": round(seg_means[s], 3) for s in SEGMENT_NAMES}}]
    )
    df_with_avg = pd.concat([df, avg_row], ignore_index=True)
    df_with_avg.to_csv(os.path.join(output_dir, "curve_fit_mae.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(df) * 0.28)), gridspec_kw={"width_ratios": [2, 1]})

    ax_vid = axes[0]
    y_pos = np.arange(len(df))
    ax_vid.barh(y_pos, df["total_mae"].values, color="#607D8B", edgecolor="white", linewidth=0.3)
    ax_vid.set_yticks(y_pos)
    ax_vid.set_yticklabels(df["video_id"].values, fontsize=7)
    ax_vid.axvline(overall_mae, color="#F44336", linestyle="--", linewidth=1.5, label=f"mean={overall_mae:.2f}")
    for i, v in enumerate(df["total_mae"].values):
        ax_vid.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=6.5)
    ax_vid.set(xlabel="MAE (curve fit)", title=f"Curve-fit MAE per video  (overall={overall_mae:.2f})")
    ax_vid.legend(fontsize=9)
    ax_vid.grid(True, alpha=0.3, axis="x")
    ax_vid.invert_yaxis()

    ax_seg = axes[1]
    seg_names = list(SEGMENT_NAMES)
    seg_vals = [seg_means[s] for s in seg_names]
    colors = [SEGMENT_COLORS[s] for s in seg_names]
    b = ax_seg.bar(seg_names, seg_vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(b, seg_vals, strict=True):
        ax_seg.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_seg.set(ylabel="MAE", title="Mean curve-fit MAE per zone")
    ax_seg.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "curve_fit_mae_summary.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Curve-fit MAE: overall=%.3f  %s", overall_mae, "  ".join(f"{s}={seg_means[s]:.3f}" for s in SEGMENT_NAMES))


def _plot_segmented_predictions(video_frames, seg_models, feature_cols, val_ids, results_dir, plots_dir):
    """For each video, stitch per-zone CatBoost predictions into one curve; plot + save metrics."""
    pred_dir = os.path.join(plots_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    all_metrics = []

    for vid in sorted(video_frames.keys()):
        df = video_frames[vid]
        n = len(df)
        y_true = df["retention"].values.astype(float)
        y_pred = np.full(n, np.nan)
        labels = df["_segment"].values

        for seg, model in seg_models.items():
            mask = labels == seg
            if not mask.any():
                continue
            X_seg = df.loc[mask, feature_cols].astype(float).fillna(0)
            y_pred[mask] = model.predict(X_seg)

        valid = ~np.isnan(y_pred)
        if not valid.any():
            continue
        if np.isnan(y_pred).any():
            y_pred[~valid] = np.nanmean(y_pred)

        split = "val" if vid in val_ids else "train"
        m = _regression_metrics(y_true, y_pred)
        rmse = round(float(np.sqrt(m["mse"])), 3)
        all_metrics.append({"video_id": vid, "split": split, "mae": m["mae"], "rmse": rmse, "r2": m["r2"], "n_seconds": n})

        t = np.arange(n)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1], sharex=True)
        a1.plot(t, y_true, color="#2196F3", lw=1.5, label="actual")
        a1.plot(t, y_pred, color="#FF5722", lw=1.5, label="predicted", alpha=0.8)
        a1.fill_between(t, y_true, y_pred, alpha=0.1, color="#9C27B0")

        for seg in SEGMENT_NAMES:
            mask = labels == seg
            if mask.any():
                spans = np.where(np.diff(np.concatenate([[False], mask, [False]]).astype(int)))[0]
                for i in range(0, len(spans), 2):
                    a1.axvspan(spans[i], spans[i + 1] - 1, alpha=0.06, color=SEGMENT_COLORS[seg], label=seg if i == 0 else None)

        a1.set(ylabel="Retention (%)", title=f"{vid} [{split}]  MAE={m['mae']:.2f}  RMSE={rmse:.2f}  R²={m['r2']:.3f}")
        a1.legend(fontsize=8)
        a1.grid(True, alpha=0.3)

        res = y_pred - y_true
        a2.fill_between(t, res, alpha=0.3, color="#4CAF50", where=res >= 0)
        a2.fill_between(t, res, alpha=0.3, color="#F44336", where=res < 0)
        a2.axhline(0, color="black", lw=0.5)
        a2.set(xlabel="sec", ylabel="error")
        a2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(pred_dir, f"pred_{vid}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(results_dir, "prediction_metrics.csv"), index=False)

        val_df = metrics_df[metrics_df["split"] == "val"]
        train_df = metrics_df[metrics_df["split"] == "train"]
        summary = {
            "all_mae": round(float(metrics_df["mae"].mean()), 3),
            "all_rmse": round(float(metrics_df["rmse"].mean()), 3),
            "all_r2": round(float(metrics_df["r2"].mean()), 3),
            "val_mae": round(float(val_df["mae"].mean()), 3) if len(val_df) else None,
            "val_rmse": round(float(val_df["rmse"].mean()), 3) if len(val_df) else None,
            "val_r2": round(float(val_df["r2"].mean()), 3) if len(val_df) else None,
            "train_mae": round(float(train_df["mae"].mean()), 3) if len(train_df) else None,
            "train_rmse": round(float(train_df["rmse"].mean()), 3) if len(train_df) else None,
            "train_r2": round(float(train_df["r2"].mean()), 3) if len(train_df) else None,
            "n_videos": len(metrics_df),
        }
        with open(os.path.join(results_dir, "prediction_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "Prediction summary: all MAE=%.3f RMSE=%.3f | val MAE=%.3f RMSE=%.3f | train MAE=%.3f RMSE=%.3f",
            summary["all_mae"],
            summary["all_rmse"],
            summary.get("val_mae") or 0,
            summary.get("val_rmse") or 0,
            summary.get("train_mae") or 0,
            summary.get("train_rmse") or 0,
        )
    logger.info("Prediction plots saved to %s (%d videos)", pred_dir, len(all_metrics))


def _plot_mae_summary(results, loo_results, output_dir):
    active = [r for r in results if not r.get("skipped")]
    if not active:
        return
    segs = [r["segment"] for r in active]
    t_mae = [r["train"]["mae"] for r in active]
    v_mae = [r.get("val", {}).get("mae", float("nan")) for r in active]
    l_mae = [loo_results.get(r["segment"], {}).get("loo_mae_mean", float("nan")) for r in active]
    x, w = np.arange(len(segs)), 0.28
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, t_mae, w, label="train", color="#2196F3")
    ax.bar(x, v_mae, w, label="val", color="#FF5722")
    ax.bar(x + w, l_mae, w, label="LOO", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(segs, fontsize=11)
    ax.set(ylabel="MAE", title="MAE per zone")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "segment_mae.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(
    output_dir="output",
    results_dir="function_result",
    model_dir=None,
    metrics_dir=None,
    dip_threshold=5.0,
    top_features=None,
    top=10,
    iterations=800,
    depth=6,
    learning_rate=0.05,
    early_stopping_rounds=50,
    loo_iterations=500,
    loo_depth=5,
    loo_learning_rate=0.05,
    loo_optuna_use=False,
    loo_optuna_trials=0,
    optuna_full=False,
    optuna_full_trials=5,
    loo_subsample=20,
    skip_prune=False,
    val_ratio=0.2,
    use_loo=True,
    curve_type="hill",
    snapshot_dir="data",
):
    model_dir = model_dir or os.path.join(results_dir, "models")
    metrics_dir = metrics_dir or os.path.join(results_dir, "plots")
    for d in (results_dir, model_dir, metrics_dir):
        os.makedirs(d, exist_ok=True)
    raw_frames = _load_video_frames(output_dir, snapshot_dir=snapshot_dir)
    if optuna_full:
        common = sorted(set.intersection(*[set(df.columns) for df in raw_frames.values()]))
        raw_common = {vid: df[common] for vid, df in raw_frames.items()}
        best = _optuna_full_tune(
            raw_common, _select_features(pd.concat(raw_common.values(), ignore_index=True), skip_prune, top_features), n_trials=optuna_full_trials, loo_subsample=loo_subsample
        )
        dip_threshold, iterations, depth, learning_rate = (best["dip_threshold"], best["iterations"], best["depth"], best["learning_rate"])
        loo_iterations, loo_depth, loo_learning_rate = iterations, depth, learning_rate
    video_frames, fit_mae_records = _segment_frames(raw_frames, dip_threshold, metrics_dir, curve_type=curve_type)
    _plot_curve_fit_mae(fit_mae_records, metrics_dir)
    train_ids, val_ids = _split_ids(video_frames, val_ratio)
    feature_cols = _select_features(pd.concat([video_frames[v] for v in train_ids], ignore_index=True), skip_prune, top_features)
    results = []
    for seg in SEGMENT_NAMES:
        tr = pd.concat([video_frames[v][video_frames[v]["_segment"] == seg] for v in train_ids], ignore_index=True)
        vl = pd.concat([video_frames[v][video_frames[v]["_segment"] == seg] for v in val_ids], ignore_index=True)
        sw = _video_sample_weights(video_frames, train_ids, feature_cols, seg)
        results.append(
            train_one_segment(
                seg,
                tr[feature_cols].astype(float).fillna(0),
                tr["retention"].values.astype(float),
                vl[feature_cols].astype(float).fillna(0),
                vl["retention"].values.astype(float),
                model_dir=model_dir,
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                early_stopping_rounds=early_stopping_rounds,
                top=top,
                sample_weight=sw,
            )
        )
    seg_models = {r["segment"]: r["model"] for r in results if not r.get("skipped") and "model" in r}
    _plot_segmented_predictions(video_frames, seg_models, feature_cols, val_ids, results_dir, metrics_dir)

    loo_results = {}
    if use_loo:
        for seg in SEGMENT_NAMES:
            loo_results[seg] = _loo_cv(video_frames, feature_cols, seg, loo_iterations, loo_depth, loo_learning_rate, loo_optuna_use, loo_optuna_trials)
    _plot_mae_summary(results, loo_results, metrics_dir)
    hyperparams = {
        "train": {"iterations": iterations, "depth": depth, "learning_rate": learning_rate, "early_stopping_rounds": early_stopping_rounds},
        "dip_threshold": dip_threshold,
        "top_features": top_features,
        "feature_cols": feature_cols,
    }
    if loo_results:
        hyperparams["loo"] = {
            seg: {"iterations": d.get("loo_iterations"), "depth": d.get("loo_depth"), "learning_rate": d.get("loo_learning_rate")} for seg, d in loo_results.items()
        }
    with open(os.path.join(results_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)
    rows = []

    for r in results:
        if r.get("skipped"):
            rows.append({"segment": r["segment"], "skipped": True})
            continue
        row = {
            "segment": r["segment"],
            "n_train": r["n_train"],
            "n_val": r.get("n_val", 0),
            "n_features": r["n_features"],
            "n_trees": r["n_trees"],
            "train_mae": r["train"]["mae"],
            "train_r2": r["train"]["r2"],
        }
        if "val" in r:
            row.update(val_mae=r["val"]["mae"], val_r2=r["val"]["r2"])
        loo = loo_results.get(r["segment"], {})
        row.update(loo_mae_mean=loo.get("loo_mae_mean"), loo_r2_mean=loo.get("loo_r2_mean"), loo_n_folds=loo.get("n_folds"))
        rows.append(row)
        pd.DataFrame([{"feature": k, "importance": v} for k, v in r.get("top", {}).items()]).to_csv(
            os.path.join(results_dir, f"feature_importance_{r['segment']}.csv"), index=False
        )
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, "segmented_summary.csv"), index=False)
    for r in results:
        if r.get("skipped"):
            print(f"  [{r['segment']:<12}] SKIPPED")
            continue
        v, loo = r.get("val", {}), loo_results.get(r["segment"], {})
        print(
            f"[{r['segment']:<12}] train={r['n_train']:>5} val={r.get('n_val', 0):>5} feats={r['n_features']:>3} trees={r['n_trees']:>4} "
            f"train_MAE={r['train']['mae']:.2f}  {'val_MAE=' + str(v.get('mae', '—')) if v else 'no val'}  {'LOO_MAE=' + str(loo.get('loo_mae_mean', '—')) if loo else ''}"
        )
        print(f"top-{top}: {list(r['top'].items())}")
    print(f" Models: {model_dir}\n  Plots: {metrics_dir}")
    return results


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    p.add_argument("--output_dir", default="output")
    p.add_argument("--results_dir", default=os.path.join(ROOT, "function_result", "segmented"))
    p.add_argument("--model_dir", default=None)
    p.add_argument("--metrics_dir", default=None)
    p.add_argument("--dip_threshold", type=float, default=5.0)
    p.add_argument("--top_features", type=int, default=None)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--loo_iterations", type=int, default=500)
    p.add_argument("--loo_depth", type=int, default=5)
    p.add_argument("--loo_learning_rate", type=float, default=0.05)
    p.add_argument("--loo_optuna_use", action="store_true")
    p.add_argument("--loo_optuna_trials", type=int, default=0)
    p.add_argument("--optuna_full", action="store_true")
    p.add_argument("--optuna_full_trials", type=int, default=5)
    p.add_argument("--loo_subsample", type=int, default=20)
    p.add_argument("--skip_prune", action="store_true")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--no_loo", action="store_true")
    p.add_argument("--curve_type", default="hill", choices=list(CURVE_TYPES))
    p.add_argument("--snapshot_dir", default="data")
    args = p.parse_args()
    run(
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        model_dir=args.model_dir,
        metrics_dir=args.metrics_dir,
        dip_threshold=args.dip_threshold,
        top_features=args.top_features,
        top=args.top,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        loo_iterations=args.loo_iterations,
        loo_depth=args.loo_depth,
        loo_learning_rate=args.loo_learning_rate,
        loo_optuna_use=args.loo_optuna_use,
        loo_optuna_trials=args.loo_optuna_trials,
        optuna_full=args.optuna_full,
        optuna_full_trials=args.optuna_full_trials,
        loo_subsample=args.loo_subsample,
        skip_prune=args.skip_prune,
        val_ratio=args.val_ratio,
        use_loo=not args.no_loo,
        curve_type=args.curve_type,
        snapshot_dir=args.snapshot_dir,
    )


if __name__ == "__main__":
    main()
