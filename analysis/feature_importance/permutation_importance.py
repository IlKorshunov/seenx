"""Permutation feature importance.
Compares CatBoost, Ridge and Random Forest after shuffling features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance as sklearn_perm
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from catboost import CatBoostRegressor

from .utils import aggregate_per_video, default_output_dir, feature_group_of, load_all_videos, prepare_X_y, save_importance_csv


def _catboost_model(n_samples: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=300 if n_samples < 20 else 500, depth=4 if n_samples < 20 else 6, learning_rate=0.05, loss_function="RMSE", random_seed=42, verbose=0, allow_writing_files=False
    )


def _ridge_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])


def _rf_model(n_samples: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, max_depth=4 if n_samples < 20 else 8, random_state=42, n_jobs=-1)


MODEL_BUILDERS = {"catboost": _catboost_model, "ridge": lambda _: _ridge_pipeline(), "rf": _rf_model}


def _fit_model(model_name: str, X_arr: np.ndarray, y_arr: np.ndarray):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")
    model = MODEL_BUILDERS[model_name](len(X_arr))
    model.fit(X_arr, y_arr)
    return model


def compute_permutation_importance(X: pd.DataFrame, y: pd.Series, model_name: str = "catboost", n_repeats: int = 20, random_state: int = 42) -> pd.DataFrame:
    X_arr, y_arr = X.values.astype(float), y.values.astype(float)
    model = _fit_model(model_name, X_arr, y_arr)
    perm = sklearn_perm(model, X_arr, y_arr, n_repeats=n_repeats, random_state=random_state, scoring="neg_mean_squared_error")
    return _importance_frame(X.columns.tolist(), perm.importances_mean, perm.importances_std, model_name)


def compute_loo_permutation_importance(X: pd.DataFrame, y: pd.Series, model_name: str = "catboost", n_repeats: int = 10) -> pd.DataFrame:
    n = len(X)
    if n < 5:
        print(f"  LOO: only {n} samples, falling back to full-data permutation importance")
        return compute_permutation_importance(X, y, model_name, n_repeats)

    X_arr, y_arr = X.values.astype(float), y.values.astype(float)
    fold_importances: list[np.ndarray] = []

    for train_idx, test_idx in LeaveOneOut().split(X_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        model = _fit_model(model_name, X_train, y_train)
        perm = sklearn_perm(model, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring="neg_mean_squared_error")
        fold_importances.append(perm.importances_mean)

    importances = np.array(fold_importances)
    return _importance_frame(X.columns.tolist(), importances.mean(axis=0), importances.std(axis=0), f"{model_name}_loo")


def _importance_frame(features: list[str], mean: np.ndarray, std: np.ndarray, model: str) -> pd.DataFrame:
    return pd.DataFrame({"feature": features, "importance_mean": mean, "importance_std": std, "group": [feature_group_of(feature) for feature in features], "model": model}).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_permutation_importance(imp_df: pd.DataFrame, title: str, out_path: str | Path, top_n: int = 30) -> None:
    top = imp_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    y_pos = np.arange(len(top))[::-1]
    ax.barh(y_pos, top["importance_mean"], xerr=top.get("importance_std", None), color="steelblue", alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean decrease in MSE when feature is permuted")
    ax.set_title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_multi_model_comparison(results: dict[str, pd.DataFrame], out_path: str | Path, top_n: int = 20) -> None:
    all_features = set()
    for df in results.values():
        all_features.update(df["feature"].tolist())

    avg_imp: dict[str, float] = {}
    for feat in all_features:
        vals = []
        for df in results.values():
            row = df[df["feature"] == feat]
            if not row.empty:
                vals.append(float(row["importance_mean"].iloc[0]))
        avg_imp[feat] = np.mean(vals) if vals else 0.0

    top_features = sorted(avg_imp, key=avg_imp.get, reverse=True)[:top_n]

    model_names = list(results.keys())
    n_models = len(model_names)
    x = np.arange(len(top_features))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2.colors
    for i, (mname, df) in enumerate(results.items()):
        feat_imp = df.set_index("feature")["importance_mean"]
        vals = [feat_imp.get(f, 0.0) for f in top_features]
        ax.bar(x + i * width, vals, width, label=mname, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean permutation importance (decrease in MSE)")
    ax.set_title(f"Permutation Importance — top {top_n} features across models")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.6)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def run_permutation_importance(
    output_dir: str = "output", results_dir: str | None = None, target: str = "target_avg_retention", top_n: int = 30, use_loo: bool = True, models: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    if results_dir is None:
        results_dir = default_output_dir()
    results_dir = Path(results_dir) / "permutation"
    results_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = ["catboost", "ridge", "rf"]

    print(f"Loading videos from {output_dir}")
    video_dfs = load_all_videos(output_dir)
    print(f"  Found {len(video_dfs)} video(s)")

    agg_df = aggregate_per_video(video_dfs)
    X, y = prepare_X_y(agg_df, target=target)
    print(f"  Feature matrix: {X.shape}")

    all_results: dict[str, pd.DataFrame] = {}

    for mname in models:
        print(f"  Computing permutation importance [{mname}]")
        try:
            imp = compute_loo_permutation_importance(X, y, mname) if use_loo and len(X) >= 5 else compute_permutation_importance(X, y, mname)
            save_importance_csv(imp, results_dir / f"perm_importance_{mname}.csv", sort_by="importance_mean")
            plot_permutation_importance(imp, f"Permutation Importance [{mname}] — {target}", results_dir / f"perm_importance_{mname}.png", top_n)
            all_results[mname] = imp
        except Exception as e:
            print(f"  ERROR for {mname}: {e}")

    if len(all_results) > 1:
        plot_multi_model_comparison(all_results, results_dir / "perm_importance_comparison.png", top_n=min(top_n, 20))

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Permutation feature importance")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--target", default="target_avg_retention", choices=["target_avg_retention", "target_drop_rate", "target_early_drop"])
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--no_loo", action="store_true", help="Disable LOO CV")
    parser.add_argument("--models", nargs="+", default=None, choices=["catboost", "ridge", "rf"])
    args = parser.parse_args()
    run_permutation_importance(output_dir=args.output_dir, results_dir=args.results_dir, target=args.target, top_n=args.top_n, use_loo=not args.no_loo, models=args.models)
