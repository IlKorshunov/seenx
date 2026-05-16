"""Feature-importance orchestrator.
Runs selected methods and builds one consensus ranking/report.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .catboost_importance import run_catboost_importance
from .permutation_importance import run_permutation_importance
from .correlation_analysis import run_correlation_analysis
from .shap_analysis import run_shap_analysis
from .transformer_importance import run_transformer_importance
from matplotlib.patches import Patch  
matplotlib.use("Agg")

from .utils import default_output_dir, feature_group_of


def _run_catboost(output_dir, results_dir, target, top_n) -> dict | None:
    return run_catboost_importance(output_dir=output_dir, results_dir=results_dir, target=target, top_n=top_n)


def _run_permutation(output_dir, results_dir, target, top_n) -> dict | None:
    return run_permutation_importance(output_dir=output_dir, results_dir=results_dir, target=target, top_n=top_n, use_loo=True)


def _run_correlation(output_dir, results_dir, target, top_n) -> dict | None:
    return run_correlation_analysis(output_dir=output_dir, results_dir=results_dir, target=target, top_n=top_n)


def _run_shap(output_dir, results_dir, target, top_n) -> dict | None:   
    return run_shap_analysis(output_dir=output_dir, results_dir=results_dir, target=target, top_n=top_n, n_dependence_plots=min(5, top_n))


def _run_transformer(output_dir, results_dir, target, top_n) -> dict | None:
    return run_transformer_importance(output_dir=output_dir, results_dir=results_dir, target=target, top_n=top_n)


PIPELINE_REGISTRY = {"catboost": _run_catboost, "permutation": _run_permutation, "correlation": _run_correlation, "shap": _run_shap, "transformer": _run_transformer}


def build_master_ranking(pipeline_results: dict[str, dict], results_dir: Path) -> pd.DataFrame:
    all_scores: dict[str, dict[str, float]] = {}

    cb = pipeline_results.get("catboost", {})
    if cb:
        for method_key, df_key in [("cb_pvc", "pvc"), ("cb_lfc", "lfc"), ("cb_shap", "shap")]:
            df = cb.get(df_key)
            if df is not None and "feature" in df.columns and "importance" in df.columns:
                for _, row in df.iterrows():
                    feat = row["feature"]
                    all_scores.setdefault(feat, {})[method_key] = float(row["importance"])

    perm = pipeline_results.get("permutation", {})
    if perm:
        for mname, df in perm.items():
            if df is not None and "feature" in df.columns and "importance_mean" in df.columns:
                for _, row in df.iterrows():
                    feat = row["feature"]
                    all_scores.setdefault(feat, {})[f"perm_{mname}"] = float(row["importance_mean"])

    corr = pipeline_results.get("correlation", {})
    if corr:
        for method_key, df_key, col in [
            ("spearman", "spearman_agg", "abs_correlation"),
            ("pearson", "pearson_agg", "abs_correlation"),
            ("mi", "mutual_info", "mutual_information"),
        ]:
            df = corr.get(df_key)
            if df is not None and "feature" in df.columns and col in df.columns:
                for _, row in df.iterrows():
                    feat = row["feature"]
                    all_scores.setdefault(feat, {})[method_key] = float(row[col])

    shap_res = pipeline_results.get("shap", {})
    if shap_res:
        df = shap_res.get("importance_df")
        if df is not None and "feature" in df.columns and "shap_mean_abs" in df.columns:
            for _, row in df.iterrows():
                feat = row["feature"]
                all_scores.setdefault(feat, {})["shap_deep"] = float(row["shap_mean_abs"])

    trans_res = pipeline_results.get("transformer", {})
    if trans_res:
        for method_key, df_key in [("trans_attn", "attn_importance"), ("trans_grad", "grad_importance")]:
            df = trans_res.get(df_key)
            if df is not None and "feature" in df.columns and "importance" in df.columns:
                for _, row in df.iterrows():
                    feat = row["feature"]
                    all_scores.setdefault(feat, {})[method_key] = float(row["importance"])

    if not all_scores:
        print("WARNING: No importance scores collected for master ranking.")
        return pd.DataFrame()

    score_df = pd.DataFrame(all_scores).T
    score_df.index.name = "feature"

    rank_cols = []
    for col in score_df.columns:
        valid = score_df[col].dropna()
        if len(valid) == 0:
            continue
        rank_col = f"rank_{col}"
        score_df[rank_col] = score_df[col].rank(ascending=False, na_option="bottom")
        rank_cols.append(rank_col)

    if rank_cols:
        score_df["avg_rank"] = score_df[rank_cols].mean(axis=1)
        score_df["n_methods"] = score_df[rank_cols].notna().sum(axis=1)
    else:
        score_df["avg_rank"] = np.nan
        score_df["n_methods"] = 0

    score_df["group"] = [feature_group_of(f) for f in score_df.index]
    result = score_df.sort_values("avg_rank").reset_index()

    out_path = results_dir / "master_ranking.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved master ranking: {out_path}")
    return result


def write_summary_report(master_ranking: pd.DataFrame, pipeline_results: dict[str, dict], results_dir: Path, target: str, elapsed: dict[str, float]) -> None:
    ls = os.linesep
    lines = [
        f"Feature Importance Analysis — Summary Report{ls}",
        f"**Target:** `{target}`{ls}",
        f"**Results directory:** `{results_dir}`{ls}",
        f"{ls}## Pipelines Run{ls}",
    ]

    for name, t in elapsed.items():
        status = "ok" if pipeline_results.get(name) else "failed"
        lines.append(f"- {status} **{name}** ({t:.1f}s){ls}")

    lines.append(f"{ls}## Top 20 Features (Consensus Ranking){ls}{ls}")
    if not master_ranking.empty:
        top20 = master_ranking.head(20)[["feature", "group", "avg_rank", "n_methods"]]
        lines.append(f"| Rank | Feature | Group | Avg Rank | Methods |{ls}")
        lines.append(f"|------|---------|-------|----------|---------|{ls}")
        for i, row in top20.iterrows():
            lines.append(f"| {i + 1} | `{row['feature']}` | {row['group']} | {row['avg_rank']:.1f} | {int(row['n_methods'])} |{ls}")

    lines.append(f"{ls}## Feature Groups (by consensus importance){ls}{ls}")
    if not master_ranking.empty and "group" in master_ranking.columns:
        group_ranks = master_ranking.groupby("group")["avg_rank"].mean().sort_values()
        lines.append(f"| Group | Avg Rank (lower = more important) |{ls}")
        lines.append(f"|-------|-----------------------------------|{ls}")
        for group, rank in group_ranks.items():
            lines.append(f"| {group} | {rank:.1f} |{ls}")

    lines.append(f"{ls}## Output Files{ls}{ls}")
    for subdir in sorted(results_dir.iterdir()):
        if subdir.is_dir():
            files = sorted(subdir.glob("*"))
            lines.append(f"### `{subdir.name}/`{ls}")
            for f in files[:15]:
                lines.append(f"- `{f.name}`{ls}")
            if len(files) > 15:
                lines.append(f"- and {len(files) - 15} more{ls}")
            lines.append(ls)

    report_path = results_dir / "summary_report.md"
    with open(report_path, "w") as f:
        f.writelines(lines)
    print(f"Saved summary report: {report_path}")


def plot_master_ranking(master_ranking: pd.DataFrame, results_dir: Path, top_n: int = 30) -> None:
    if master_ranking.empty:
        return

    top = master_ranking.head(top_n).copy()
    groups = top["group"].unique()
    palette = plt.cm.tab20.colors
    group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    colors = [group_colors[g] for g in top["group"]]

    _, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    max_rank = top["avg_rank"].max()
    bar_vals = (max_rank - top["avg_rank"] + 1)[::-1]
    ax.barh(top["feature"][::-1], bar_vals, color=colors[::-1])
    ax.set_xlabel("Consensus importance (higher = more important)")
    ax.set_title(f"Master Feature Ranking — top {top_n}")
    ax.tick_params(axis="y", labelsize=8)

    legend_elements = [Patch(facecolor=group_colors[g], label=g) for g in sorted(group_colors)]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7, ncol=2)

    plt.tight_layout()
    out_path = results_dir / "master_ranking.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def run_all(
    output_dir: str = "output", results_dir: str | None = None, target: str = "target_avg_retention", top_n: int = 30, pipelines: list[str] | None = None
) -> dict[str, object]:
    if results_dir is None:
        results_dir = default_output_dir()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if pipelines is None:
        pipelines = list(PIPELINE_REGISTRY.keys())

    print(f"[feature_importance] output={output_dir} results={results_dir} target={target} pipelines={pipelines}")

    all_results: dict[str, object] = {}
    elapsed: dict[str, float] = {}

    for name in pipelines:
        if name not in PIPELINE_REGISTRY:
            print(f"WARNING: Unknown pipeline '{name}', skipping.")
            continue

        print(f"[run] {name}")
        t0 = time.time()
        try:
            result = PIPELINE_REGISTRY[name](output_dir, str(results_dir), target, top_n)
            all_results[name] = result
            elapsed[name] = time.time() - t0
            print(f"[ok] {name} {elapsed[name]:.1f}s")
        except Exception as e:
            elapsed[name] = time.time() - t0
            print(f"[fail] {name}: {e}")
            import traceback

            traceback.print_exc()
            all_results[name] = None

    print("[rank] consensus")
    master_ranking = build_master_ranking(all_results, results_dir)
    all_results["master_ranking"] = master_ranking

    if not master_ranking.empty:
        plot_master_ranking(master_ranking, results_dir, top_n)

    write_summary_report(master_ranking, all_results, results_dir, target, elapsed)

    total = sum(elapsed.values())
    print(f"[done] {total:.1f}s results={results_dir}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all feature importance analysis pipelines",
    )
    parser.add_argument("--output_dir", default="output", help="Directory with *_features.csv files (default: output)")
    parser.add_argument("--results_dir", default=None, help="Where to save results (default: analysis/feature_importance/results)")
    parser.add_argument("--target", default="target_avg_retention", choices=["target_avg_retention", "target_drop_rate", "target_early_drop"], help="Target variable to analyze")
    parser.add_argument("--top_n", type=int, default=30, help="Number of top features to show in plots")
    parser.add_argument(
        "--pipelines", nargs="+", default=None, choices=["catboost", "permutation", "correlation", "shap", "transformer"], help="Which pipelines to run (default: all)"
    )
    args = parser.parse_args()
    run_all(output_dir=args.output_dir, results_dir=args.results_dir, target=args.target, top_n=args.top_n, pipelines=args.pipelines)
