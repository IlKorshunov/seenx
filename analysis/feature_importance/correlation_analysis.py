"""Correlation feature importance.
Runs target correlations, mutual information and redundancy reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from .utils import NON_FEATURE_COLS, aggregate_per_video, default_output_dir, feature_group_of, load_all_videos, prepare_X_y, save_importance_csv


def compute_feature_target_correlation(X: pd.DataFrame, y: pd.Series, method: str = "spearman") -> pd.DataFrame:
    rows = []
    for col in X.columns:
        x_vals = X[col].values.astype(float)
        y_vals = y.values.astype(float)
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        corr, pval = (np.nan, np.nan) if mask.sum() < 3 else stats.spearmanr(x_vals[mask], y_vals[mask]) if method == "spearman" else stats.pearsonr(x_vals[mask], y_vals[mask])
        rows.append(
            {
                "feature": col,
                "correlation": float(corr) if not np.isnan(corr) else 0.0,
                "abs_correlation": abs(float(corr)) if not np.isnan(corr) else 0.0,
                "p_value": float(pval) if not np.isnan(pval) else 1.0,
                "significant": bool(pval < 0.05) if not np.isnan(pval) else False,
                "group": feature_group_of(col),
                "method": method,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def compute_per_second_correlation(video_dfs: dict[str, pd.DataFrame], target_col: str = "retention", method: str = "spearman") -> pd.DataFrame:
    frames = []
    for df in video_dfs.values():
        if target_col not in df.columns:
            continue
        feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and c != "time_sec" and pd.api.types.is_numeric_dtype(df[c])]
        sub = df[feat_cols + [target_col]].apply(pd.to_numeric, errors="coerce")
        frames.append(sub)

    if not frames:
        raise ValueError("No valid data found")

    pooled = pd.concat(frames, ignore_index=True).dropna(subset=[target_col])
    y = pooled[target_col]
    X = pooled.drop(columns=[target_col])

    return compute_feature_target_correlation(X, y, method=method)


def compute_feature_correlation_matrix(X: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    if method == "spearman":
        corr_matrix, _ = stats.spearmanr(X.values)
        if X.shape[1] == 1:
            return pd.DataFrame([[1.0]], index=X.columns, columns=X.columns)
        return pd.DataFrame(corr_matrix, index=X.columns, columns=X.columns)
    else:
        return X.corr(method="pearson")


def find_redundant_features(corr_matrix: pd.DataFrame, threshold: float = 0.85) -> list[tuple[str, str, float]]:
    cols = corr_matrix.columns.tolist()
    pairs = [(cols[i], cols[j], float(abs(corr_matrix.iloc[i, j]))) for i in range(len(cols)) for j in range(i + 1, len(cols)) if abs(corr_matrix.iloc[i, j]) >= threshold]
    return sorted(pairs, key=lambda x: -x[2])


def compute_mutual_information(X: pd.DataFrame, y: pd.Series, n_neighbors: int = 3, random_state: int = 42) -> pd.DataFrame:
    X_arr = X.values.astype(float)
    y_arr = y.values.astype(float)
    X_arr = np.nan_to_num(X_arr, nan=0.0)
    y_arr = np.nan_to_num(y_arr, nan=0.0)

    mi = mutual_info_regression(X_arr, y_arr, n_neighbors=min(n_neighbors, max(1, len(y) - 1)), random_state=random_state)
    result = (
        pd.DataFrame({"feature": X.columns.tolist(), "mutual_information": mi, "group": [feature_group_of(f) for f in X.columns], "method": "mutual_information"})
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )
    return result


def plot_correlation_bar(corr_df: pd.DataFrame, title: str, out_path: str | Path, top_n: int = 30, col: str = "correlation") -> None:
    top = corr_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors = ["#d73027" if v > 0 else "#4575b4" for v in top[col][::-1]]
    ax.barh(top["feature"][::-1], top[col][::-1], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"{col.replace('_', ' ').title()}")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, out_path: str | Path, title: str = "Feature Correlation Matrix", figsize: tuple[int, int] = (18, 16)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, xticklabels=True, yticklabels=True, square=True, linewidths=0.3, cbar_kws={"shrink": 0.6})
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_group_heatmap(corr_matrix: pd.DataFrame, out_path: str | Path) -> None:
    groups = {}
    for feat in corr_matrix.columns:
        g = feature_group_of(feat)
        groups.setdefault(g, []).append(feat)

    group_names = sorted(groups.keys())
    n = len(group_names)
    group_corr = np.zeros((n, n))

    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            feats1 = [f for f in groups[g1] if f in corr_matrix.columns]
            feats2 = [f for f in groups[g2] if f in corr_matrix.columns]
            if feats1 and feats2:
                sub = corr_matrix.loc[feats1, feats2].values
                group_corr[i, j] = float(np.abs(sub).mean())

    group_df = pd.DataFrame(group_corr, index=group_names, columns=group_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(group_df, cmap="YlOrRd", vmin=0, vmax=1, ax=ax, annot=True, fmt=".2f", annot_kws={"size": 8}, linewidths=0.5)
    ax.set_title("Feature Group Correlation (mean |ρ|)")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_redundancy_network(redundant_pairs: list[tuple[str, str, float]], out_path: str | Path, threshold: float = 0.85) -> None:
    if not redundant_pairs:
        print("No redundant feature pairs found.")
        return

    lines = [f"Highly correlated feature pairs (|ρ| ≥ {threshold}):\n"]
    for a, b, val in redundant_pairs[:50]:
        lines.append(f"  {val:.3f}  {a}  ↔  {b}")

    text = "\n".join(lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path).replace(".png", ".txt"), "w") as f:
        f.write(text)
    print(f"Saved redundancy report: {str(out_path).replace('.png', '.txt')}")


def run_correlation_analysis(
    output_dir: str = "output", results_dir: str | None = None, target: str = "target_avg_retention", top_n: int = 30, redundancy_threshold: float = 0.85
) -> dict[str, pd.DataFrame]:
    if results_dir is None:
        results_dir = default_output_dir()
    results_dir = Path(results_dir) / "correlation"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading videos from {output_dir}")
    video_dfs = load_all_videos(output_dir)
    print(f"  Found {len(video_dfs)} video(s)")

    agg_df = aggregate_per_video(video_dfs)
    X, y = prepare_X_y(agg_df, target=target)
    print(f"  Feature matrix: {X.shape}")

    results = {}

    print("Computing Spearman correlation (aggregated)")
    sp_agg = compute_feature_target_correlation(X, y, method="spearman")
    save_importance_csv(sp_agg, results_dir / "spearman_agg.csv", sort_by="abs_correlation")
    plot_correlation_bar(sp_agg, f"Spearman ρ with {target} (per-video avg)", results_dir / "spearman_agg.png", top_n)
    results["spearman_agg"] = sp_agg

    print("Computing Pearson correlation (aggregated)")
    pe_agg = compute_feature_target_correlation(X, y, method="pearson")
    save_importance_csv(pe_agg, results_dir / "pearson_agg.csv", sort_by="abs_correlation")
    plot_correlation_bar(pe_agg, f"Pearson r with {target} (per-video avg)", results_dir / "pearson_agg.png", top_n)
    results["pearson_agg"] = pe_agg

    print("Computing Spearman correlation (per-second pooled)")
    try:
        sp_ts = compute_per_second_correlation(video_dfs, target_col="retention", method="spearman")
        save_importance_csv(sp_ts, results_dir / "spearman_timeseries.csv", sort_by="abs_correlation")
        plot_correlation_bar(sp_ts, "Spearman ρ with retention (per-second, all videos pooled)", results_dir / "spearman_timeseries.png", top_n)
        results["spearman_ts"] = sp_ts
    except Exception as e:
        print(f"  Per-second correlation failed: {e}")

    print("Computing mutual information")
    mi = compute_mutual_information(X, y)
    save_importance_csv(mi, results_dir / "mutual_information.csv", sort_by="mutual_information")
    plot_correlation_bar(mi, f"Mutual Information with {target}", results_dir / "mutual_information.png", top_n, col="mutual_information")
    results["mutual_info"] = mi

    print("Computing feature–feature correlation matrix")
    corr_mat = compute_feature_correlation_matrix(X, method="spearman")
    corr_mat.to_csv(results_dir / "feature_corr_matrix.csv")
    plot_correlation_heatmap(corr_mat, results_dir / "feature_corr_heatmap.png")
    plot_group_heatmap(corr_mat, results_dir / "group_corr_heatmap.png")
    results["corr_matrix"] = corr_mat

    redundant = find_redundant_features(corr_mat, threshold=redundancy_threshold)
    plot_redundancy_network(redundant, results_dir / "redundant_features.png", redundancy_threshold)
    if redundant:
        red_df = pd.DataFrame(redundant, columns=["feature_a", "feature_b", "correlation"])
        red_df.to_csv(results_dir / "redundant_pairs.csv", index=False)
        print(f"  Found {len(redundant)} redundant pairs (|ρ| ≥ {redundancy_threshold})")

    combined = _build_combined_ranking(sp_agg, pe_agg, mi)
    save_importance_csv(combined, results_dir / "combined_ranking.csv", sort_by="avg_rank")
    print(f"\nTop 10 features by combined correlation ranking:\n{combined.head(10).to_string()}")

    return results


def _build_combined_ranking(spearman: pd.DataFrame, pearson: pd.DataFrame, mi: pd.DataFrame) -> pd.DataFrame:
    def _rank(df: pd.DataFrame, col: str) -> pd.Series:
        return df.set_index("feature")[col].rank(ascending=False)

    r_sp = _rank(spearman, "abs_correlation").rename("rank_spearman")
    r_pe = _rank(pearson, "abs_correlation").rename("rank_pearson")
    r_mi = _rank(mi, "mutual_information").rename("rank_mi")
    merged = pd.concat([r_sp, r_pe, r_mi], axis=1)
    merged["avg_rank"] = merged.mean(axis=1)
    merged["group"] = [feature_group_of(f) for f in merged.index]
    return merged.sort_values("avg_rank").reset_index().rename(columns={"index": "feature"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlation-based feature importance")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--target", default="target_avg_retention", choices=["target_avg_retention", "target_drop_rate", "target_early_drop"])
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--redundancy_threshold", type=float, default=0.85)
    args = parser.parse_args()
    run_correlation_analysis(output_dir=args.output_dir, results_dir=args.results_dir, target=args.target, top_n=args.top_n, redundancy_threshold=args.redundancy_threshold)
