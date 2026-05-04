import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from train.common.seq_data_utils import load_all_merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-spikes-dir", default="get_data/spikes")
    parser.add_argument("--diff-threshold", type=float, default=1.0, help="Minimum retention increase (%) to be considered a spike")
    args = parser.parse_args()

    os.makedirs(args.out_spikes_dir, exist_ok=True)

    print("Loading all merged features to find spikes...")
    video_dfs = load_all_merged("output", "data", use_curve_raw=True, emb_pca_components=0)

    all_spikes = []
    all_diffs = []
    all_features = []

    print(f"Scanning for retention spikes (threshold: >{args.diff_threshold}%)...")

    global_max_diff = 0
    for vid, df in video_dfs.items():
        if "time" not in df.columns or "retention" not in df.columns:
            continue

        # Ensure it's sorted by time
        df = df.sort_values("time").reset_index(drop=True)
        # Compute difference using the actual retention curve we use for training
        df["diff"] = pd.to_numeric(df["retention"], errors="coerce").diff()

        max_diff = df["diff"].max()
        if pd.notna(max_diff) and max_diff > global_max_diff:
            global_max_diff = max_diff

        # Find spikes
        spikes = df[df["diff"] > args.diff_threshold].copy()
        for _, row in spikes.iterrows():
            # time might be a string or timedelta
            try:
                sec_val = int(pd.to_timedelta(row["time"]).total_seconds())
            except:
                sec_val = int(row["time"])

            all_spikes.append({"video_id": vid, "sec": sec_val, "retention": row["retention"], "diff": row["diff"]})

        drop_cols = [
            "video_id",
            "time",
            "retention",
            "diff",
            "title",
            "group",
            "time_ratio",
            "duration_sec",
            "log1p_view_count",
            "mean_retention_prior",
            "early_retention_drop_30s",
        ]
        feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

        valid_rows = df.dropna(subset=["diff"])
        # We only keep rows where we have at least some features
        valid_rows = valid_rows.dropna(subset=feature_cols, how="all")

        if not valid_rows.empty:
            # We store the DataFrame slice directly to handle mismatched columns
            valid_rows = valid_rows.copy()
            valid_rows["_target_diff_"] = valid_rows["diff"]
            all_features.append(valid_rows)

    print(f"Global max diff across all videos: {global_max_diff:.3f}%")

    # 1. Save spikes list
    spikes_df = pd.DataFrame(all_spikes)
    if not spikes_df.empty:
        spikes_df = spikes_df.sort_values("diff", ascending=False)
        spikes_csv = Path(args.out_spikes_dir) / "spikes_list.csv"
        spikes_df.to_csv(spikes_csv, index=False)
        print(f"Found {len(spikes_df)} spikes. Saved list to {spikes_csv}")
    else:
        print("No spikes found!")

    # 2. Compute correlations with features
    if all_features:
        print("Computing correlations between positive retention diffs and features...")
        combined_df = pd.concat(all_features, ignore_index=True)

        # Focus on positive spikes
        pos_mask = combined_df["_target_diff_"] > 0
        combined_pos = combined_df[pos_mask]

        correlations = []
        target = combined_pos["_target_diff_"]

        for col in combined_pos.columns:
            if col == "_target_diff_" or col == "diff" or not pd.api.types.is_numeric_dtype(combined_pos[col]):
                continue

            series = combined_pos[col]
            valid = series.notna() & target.notna()
            if valid.sum() > 10:
                std = series[valid].std()
                if std > 1e-6:
                    corr = np.corrcoef(series[valid], target[valid])[0, 1]
                    correlations.append({"feature": col, "corr_with_positive_spike": corr})

        corr_df = pd.DataFrame(correlations).dropna()
        corr_df = corr_df.sort_values("corr_with_positive_spike", ascending=False)
        corr_csv = Path(args.out_spikes_dir) / "spike_feature_correlations.csv"
        corr_df.to_csv(corr_csv, index=False)
        print(f"Saved feature correlations to {corr_csv}")

        print("\nTop 15 features positively correlated with retention spikes:")
        print(corr_df.head(15).to_string(index=False))
    else:
        print("No features found to compute correlations.")


if __name__ == "__main__":
    main()
