"""SHAP feature importance.
Trains CatBoost, saves SHAP tables and summary/dependence plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")

import shap
from catboost import CatBoostRegressor, Pool
from .utils import aggregate_per_video, default_output_dir, feature_group_of, load_all_videos, prepare_X_y, save_importance_csv


def _train_catboost(X: pd.DataFrame, y: pd.Series) -> CatBoostRegressor:
    n = len(X)
    model = CatBoostRegressor(
        iterations=400 if n < 20 else 600, depth=4 if n < 20 else 6, learning_rate=0.05, loss_function="RMSE", random_seed=42, verbose=0, allow_writing_files=False
    )
    model.fit(X.values, y.values)
    return model


def compute_shap_values_catboost(model: CatBoostRegressor, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    pool = Pool(X.values)
    raw = model.get_feature_importance(pool, type="ShapValues")
    shap_matrix = raw[:, :-1]
    expected_value = float(raw[0, -1])
    return shap_matrix, expected_value


def compute_shap_values_sklearn(model, X: pd.DataFrame, background_samples: int = 50) -> tuple[np.ndarray, float]:
    X_arr = X.values.astype(float)
    background = shap.sample(X_arr, min(background_samples, len(X_arr)))
    explainer = shap.KernelExplainer(model.predict, background)
    shap_vals = explainer.shap_values(X_arr, nsamples=100)
    return shap_vals, float(explainer.expected_value)


def plot_shap_summary(shap_matrix: np.ndarray, X: pd.DataFrame, out_path: str | Path, max_display: int = 25, plot_type: str = "dot") -> None:
    fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.4)))
    shap.summary_plot(shap_matrix, X, plot_type=plot_type, max_display=max_display, show=False, plot_size=None)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def _fallback_shap_bar(shap_matrix: np.ndarray, X: pd.DataFrame, out_path: str | Path, top_n: int = 25) -> None:
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_features = [X.columns[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    ax.barh(top_features[::-1], top_vals[::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance (top {top_n})")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_shap_waterfall(shap_matrix: np.ndarray, X: pd.DataFrame, expected_value: float, sample_idx: int, out_path: str | Path, title: str = "", max_display: int = 15) -> None:
    explanation = shap.Explanation(values=shap_matrix[sample_idx], base_values=expected_value, data=X.iloc[sample_idx].values, feature_names=X.columns.tolist())
    fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.4)))
    shap.waterfall_plot(explanation, max_display=max_display, show=False)
    if title:
        plt.title(title, fontsize=10)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_shap_dependence(shap_matrix: np.ndarray, X: pd.DataFrame, feature: str, out_path: str | Path, interaction_feature: str = "auto") -> None:
    feat_idx = X.columns.tolist().index(feature)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(feat_idx, shap_matrix, X, interaction_index=interaction_feature, ax=ax, show=False)
    ax.set_title(f"SHAP Dependence: {feature}")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def _fallback_dependence(shap_matrix: np.ndarray, X: pd.DataFrame, feature: str, out_path: str | Path) -> None:
    if feature not in X.columns:
        return
    feat_idx = X.columns.tolist().index(feature)
    shap_vals = shap_matrix[:, feat_idx]
    feat_vals = X[feature].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(feat_vals, shap_vals, alpha=0.7, s=40, c="steelblue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"SHAP Dependence: {feature}")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_shap_group_bar(shap_matrix: np.ndarray, X: pd.DataFrame, out_path: str | Path) -> None:
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    group_sums: dict[str, float] = {}
    for i, feat in enumerate(X.columns):
        g = feature_group_of(feat)
        group_sums[g] = group_sums.get(g, 0.0) + mean_abs[i]

    group_df = pd.Series(group_sums).sort_values(ascending=False)
    total = group_df.sum()
    if total > 0:
        group_df = group_df / total * 100

    fig, ax = plt.subplots(figsize=(8, max(4, len(group_df) * 0.5)))
    ax.barh(group_df.index[::-1], group_df.values[::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP| contribution (%)")
    ax.set_title("Feature Group Importance (SHAP)")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def save_shap_html(shap_matrix: np.ndarray, X: pd.DataFrame, expected_value: float, out_path: str | Path, sample_idx: int = 0) -> None:
    html = shap.force_plot(expected_value, shap_matrix[sample_idx], X.iloc[sample_idx], feature_names=X.columns.tolist(), matplotlib=False)
    shap.save_html(str(out_path), html)
    print(f"Saved HTML: {out_path}")


def run_shap_analysis(
    output_dir: str = "output", results_dir: str | None = None, target: str = "target_avg_retention", top_n: int = 25, n_dependence_plots: int = 5
) -> dict[str, object]:
    if results_dir is None:
        results_dir = default_output_dir()
    results_dir = Path(results_dir) / "shap"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading videos from {output_dir}")
    video_dfs = load_all_videos(output_dir)
    print(f"  Found {len(video_dfs)} video(s)")

    agg_df = aggregate_per_video(video_dfs)
    X, y = prepare_X_y(agg_df, target=target)
    print(f"  Feature matrix: {X.shape}")
    
    print("Training CatBoost model")
    model = _train_catboost(X, y)

    print("Computing SHAP values")
    shap_matrix, expected_value = compute_shap_values_catboost(model, X)
    print(f"  SHAP matrix: {shap_matrix.shape}, expected value: {expected_value:.4f}")

    shap_df = pd.DataFrame(shap_matrix, index=X.index, columns=X.columns)
    shap_df.to_csv(results_dir / "shap_values.csv")

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": X.columns.tolist(), "shap_mean_abs": mean_abs_shap, "shap_mean": shap_matrix.mean(axis=0), "group": [feature_group_of(f) for f in X.columns]})
        .sort_values("shap_mean_abs", ascending=False)
        .reset_index(drop=True)
    )
    save_importance_csv(importance_df, results_dir / "shap_importance.csv", sort_by="shap_mean_abs")

    print("Generating summary plots")
    plot_shap_summary(shap_matrix, X, results_dir / "shap_summary_beeswarm.png", max_display=top_n, plot_type="dot")
    plot_shap_summary(shap_matrix, X, results_dir / "shap_summary_bar.png", max_display=top_n, plot_type="bar")

    plot_shap_group_bar(shap_matrix, X, results_dir / "shap_group_bar.png")

    if len(X) >= 2:
        y_pred = model.predict(X.values)
        errors = np.abs(y_pred - y.values)
        best_idx = int(np.argmin(errors))
        worst_idx = int(np.argmax(errors))

        plot_shap_waterfall(
            shap_matrix, X, expected_value, best_idx, results_dir / "waterfall_best_prediction.png", title=f"Best predicted: {X.index[best_idx]} (err={errors[best_idx]:.3f})"
        )
        plot_shap_waterfall(
            shap_matrix, X, expected_value, worst_idx, results_dir / "waterfall_worst_prediction.png", title=f"Worst predicted: {X.index[worst_idx]} (err={errors[worst_idx]:.3f})"
        )

        save_shap_html(shap_matrix, X, expected_value, results_dir / "force_plot_sample0.html", sample_idx=0)

    top_features = importance_df["feature"].head(n_dependence_plots).tolist()
    dep_dir = results_dir / "dependence_plots"
    dep_dir.mkdir(exist_ok=True)
    print(f"Generating {len(top_features)} dependence plots")
    for feat in top_features:
        if feat in X.columns:
            plot_shap_dependence(shap_matrix, X, feat, dep_dir / f"dep_{feat}.png")

    print(f"\nTop 10 features by mean |SHAP|:\n{importance_df.head(10).to_string()}")

    return {"shap_matrix": shap_matrix, "expected_value": expected_value, "importance_df": importance_df, "X": X, "y": y}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep SHAP analysis")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--target", default="target_avg_retention", choices=["target_avg_retention", "target_drop_rate", "target_early_drop"])
    parser.add_argument("--top_n", type=int, default=25)
    parser.add_argument("--n_dependence_plots", type=int, default=5)
    args = parser.parse_args()
    run_shap_analysis(output_dir=args.output_dir, results_dir=args.results_dir, target=args.target, top_n=args.top_n, n_dependence_plots=args.n_dependence_plots)
