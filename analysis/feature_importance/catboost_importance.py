"""CatBoost feature importance.
Computes built-in importance, SHAP, group scores and consensus ranking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")
from catboost import CatBoostRegressor, Pool
from .utils import aggregate_per_video, default_output_dir, feature_group_of, load_all_videos, prepare_X_y, save_importance_csv


def _build_catboost() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )


def _train_full(X: pd.DataFrame, y: pd.Series) -> CatBoostRegressor:
    model = _build_catboost()
    model.fit(X.values, y.values)
    return model


def compute_builtin_importance(X: pd.DataFrame, y: pd.Series, importance_type: str = "PredictionValuesChange") -> pd.DataFrame:
    model = _train_full(X, y)
    pool = Pool(X.values, y.values)
    scores = model.get_feature_importance(pool, type=importance_type)
    return _importance_frame(X.columns.tolist(), scores, importance_type)


def compute_shap_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    model = _train_full(X, y)
    pool = Pool(X.values, y.values)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    shap_matrix = shap_values[:, :-1]
    return (
        pd.DataFrame(
            {
                "feature": X.columns.tolist(),
                "importance": np.abs(shap_matrix).mean(axis=0),
                "shap_mean": shap_matrix.mean(axis=0),
                "group": [feature_group_of(f) for f in X.columns],
                "method": "SHAP_mean_abs",
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    ), shap_matrix


def compute_group_importance(feature_importance: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    grouped = feature_importance.groupby("group")["importance"].sum().sort_values(ascending=False).reset_index()
    if normalize:
        total = grouped["importance"].sum()
        if total > 0:
            grouped["importance_pct"] = grouped["importance"] / total * 100
    return grouped


def _importance_frame(features: list[str], scores: np.ndarray, method: str) -> pd.DataFrame:
    return pd.DataFrame({"feature": features, "importance": scores, "group": [feature_group_of(feature) for feature in features], "method": method}).sort_values("importance", ascending=False).reset_index(drop=True)


def plot_top_features(importance_df: pd.DataFrame, title: str, out_path: str | Path, top_n: int = 30, color_by_group: bool = True) -> None:
    top = importance_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    if color_by_group:
        groups = top["group"].unique()
        palette = plt.cm.tab20.colors
        group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
        colors = [group_colors[g] for g in top["group"]]
    else:
        colors = "steelblue"

    bars = ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1] if color_by_group else colors)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=9)

    if color_by_group:
        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor=group_colors[g], label=g) for g in sorted(group_colors)]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=7, ncol=2)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_group_importance(group_df: pd.DataFrame, title: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, len(group_df) * 0.5)))
    col = "importance_pct" if "importance_pct" in group_df.columns else "importance"
    ax.barh(group_df["group"][::-1], group_df[col][::-1], color="steelblue")
    ax.set_xlabel("Importance (%)" if col == "importance_pct" else "Importance")
    ax.set_title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_shap_beeswarm(X: pd.DataFrame, shap_matrix: np.ndarray, out_path: str | Path, top_n: int = 20) -> None:
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_features = [X.columns[i] for i in top_idx]
    top_shap = shap_matrix[:, top_idx]
    top_X = X.iloc[:, top_idx].values

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    for rank, (feat, shap_col, x_col) in enumerate(zip(top_features[::-1], top_shap.T[::-1], top_X.T[::-1], strict=True)):
        x_norm = (x_col - x_col.min()) / (np.ptp(x_col) + 1e-9)
        jitter = np.random.uniform(-0.2, 0.2, size=len(shap_col))
        sc = ax.scatter(shap_col, np.full_like(shap_col, rank) + jitter, c=x_norm, cmap="RdBu_r", alpha=0.6, s=20, vmin=0, vmax=1)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(f"SHAP Beeswarm — top {top_n} features")
    plt.colorbar(sc, ax=ax, label="Feature value (normalized)")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def run_catboost_importance(output_dir: str = "output", results_dir: str | None = None, target: str = "target_avg_retention", top_n: int = 30) -> dict[str, pd.DataFrame]:
    if results_dir is None:
        results_dir = default_output_dir()
    results_dir = Path(results_dir) / "catboost"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading videos from {output_dir}")
    video_dfs = load_all_videos(output_dir)
    print(f"  Found {len(video_dfs)} video(s): {list(video_dfs.keys())}")

    agg_df = aggregate_per_video(video_dfs)
    X, y = prepare_X_y(agg_df, target=target)
    print(f"  Feature matrix: {X.shape}, target: {target}")

    if len(X) < 2:
        print("WARNING: Only 1 video — importance scores may not be meaningful.")

    results = {}

    print("Computing PredictionValuesChange importance")
    pvc = compute_builtin_importance(X, y, "PredictionValuesChange")
    save_importance_csv(pvc, results_dir / "importance_pvc.csv", sort_by="importance")
    plot_top_features(pvc, f"CatBoost PredictionValuesChange — {target}", results_dir / "importance_pvc.png", top_n)
    results["pvc"] = pvc

    print("Computing LossFunctionChange importance")
    lfc = compute_builtin_importance(X, y, "LossFunctionChange")
    save_importance_csv(lfc, results_dir / "importance_lfc.csv", sort_by="importance")
    plot_top_features(lfc, f"CatBoost LossFunctionChange — {target}", results_dir / "importance_lfc.png", top_n)
    results["lfc"] = lfc

    print("Computing SHAP importance")
    shap_imp, shap_matrix = compute_shap_importance(X, y)
    save_importance_csv(shap_imp, results_dir / "importance_shap.csv", sort_by="importance")
    plot_top_features(shap_imp, f"CatBoost SHAP mean|φ| — {target}", results_dir / "importance_shap.png", top_n)
    plot_shap_beeswarm(X, shap_matrix, results_dir / "shap_beeswarm.png", top_n=min(top_n, 20))
    results["shap"] = shap_imp

    group_imp = compute_group_importance(shap_imp)
    save_importance_csv(group_imp, results_dir / "importance_groups.csv")
    plot_group_importance(group_imp, f"Feature Group Importance (SHAP) — {target}", results_dir / "importance_groups.png")
    results["group"] = group_imp

    consensus = _build_consensus(pvc, lfc, shap_imp)
    save_importance_csv(consensus, results_dir / "importance_consensus.csv", sort_by="avg_rank")
    print(f"\nTop 10 features by consensus rank:\n{consensus.head(10).to_string()}")
    results["consensus"] = consensus

    return results


def _build_consensus(pvc: pd.DataFrame, lfc: pd.DataFrame, shap: pd.DataFrame) -> pd.DataFrame:
    def _rank(df: pd.DataFrame, col: str = "importance") -> pd.Series:
        return df.set_index("feature")[col].rank(ascending=False)

    r_pvc = _rank(pvc).rename("rank_pvc")
    r_lfc = _rank(lfc).rename("rank_lfc")
    r_shap = _rank(shap).rename("rank_shap")
    merged = pd.concat([r_pvc, r_lfc, r_shap], axis=1)
    merged["avg_rank"] = merged.mean(axis=1)
    merged["group"] = [feature_group_of(f) for f in merged.index]
    return merged.sort_values("avg_rank").reset_index().rename(columns={"index": "feature"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CatBoost feature importance")
    parser.add_argument("--output_dir", default="output", help="Directory with *_features.csv")
    parser.add_argument("--results_dir", default=None, help="Where to save results")
    parser.add_argument("--target", default="target_avg_retention", choices=["target_avg_retention", "target_drop_rate", "target_early_drop"])
    parser.add_argument("--top_n", type=int, default=30)
    args = parser.parse_args()
    run_catboost_importance(output_dir=args.output_dir, results_dir=args.results_dir, target=args.target, top_n=args.top_n)
