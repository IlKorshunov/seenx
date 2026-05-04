#!/usr/bin/env python3
"""
Error analysis for retention prediction model (Idea 4).
- Per-video MAE and residual curves
- Clustering of "bad" videos by features (duration, ad, screencast, etc.)
- Output: error_report.csv, residual plots, cluster analysis
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.logger import Logger


logger = Logger(show=True).get_logger()

FEATURE_EXCLUDE = {"retention", "frame", "time"}


def load_features_and_predict(features_dir: str, model_path: str) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load feature CSVs, run model, return predictions per video."""
    import glob

    import catboost as cb

    csvs = sorted(glob.glob(os.path.join(features_dir, "*_features.csv")))
    if not csvs:
        raise FileNotFoundError(f"No *_features.csv in {features_dir}")

    video_frames = {}
    for path in csvs:
        vid = os.path.basename(path).replace("_features.csv", "")
        df = pd.read_csv(path, index_col=0)
        if "retention" not in df.columns:
            continue
        df = df.dropna(subset=["retention"])
        video_frames[vid] = df

    # Build feature matrix (intersection of columns across videos)
    all_cols = set()
    for df in video_frames.values():
        all_cols.update(df.columns)
    feature_cols = sorted([c for c in all_cols if c not in FEATURE_EXCLUDE])
    X_all = []
    vid_order = []
    for vid in sorted(video_frames.keys()):
        df = video_frames[vid]
        row = df.reindex(columns=feature_cols).astype(float).fillna(0)
        X_all.append(row)
        vid_order.append(vid)
    X_all = pd.concat(X_all, ignore_index=True)

    model = cb.CatBoostRegressor()
    model.load_model(model_path)
    pred_all = model.predict(X_all)

    # Split predictions back per video
    predictions = {}
    idx = 0
    for vid in vid_order:
        n = len(video_frames[vid])
        y_true = video_frames[vid]["retention"].values.astype(float)
        y_pred = pred_all[idx : idx + n]
        predictions[vid] = (y_true, y_pred)
        idx += n

    return video_frames, predictions


def compute_per_video_metrics(video_frames: dict, predictions: dict) -> pd.DataFrame:
    """MAE, MSE, max residual per video + metadata for clustering."""
    rows = []
    for vid in sorted(predictions.keys()):
        y_true, y_pred = predictions[vid]
        df = video_frames[vid]
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        max_residual = float(np.max(np.abs(y_true - y_pred)))
        duration = len(y_true)

        # Metadata for clustering
        row = {"video_id": vid, "mae": mae, "mse": mse, "max_residual": max_residual, "duration_sec": duration}
        # Optional: add feature-based flags if columns exist
        if "ad_density_pct" in df.columns:
            row["ad_density_pct"] = df["ad_density_pct"].iloc[0] if len(df) else 0
        if "screencast_prob" in df.columns:
            row["screencast_mean"] = df["screencast_prob"].mean()
        if "hook_score" in df.columns:
            row["hook_score"] = df["hook_score"].iloc[0] if len(df) else 0
        rows.append(row)

    return pd.DataFrame(rows)


def cluster_bad_videos(metrics_df: pd.DataFrame, n_clusters: int = 3, top_pct_bad: float = 0.3) -> pd.DataFrame:
    """Cluster videos with highest MAE by their features."""
    bad_n = max(1, int(len(metrics_df) * top_pct_bad))
    bad_df = metrics_df.nlargest(bad_n, "mae").copy()

    cluster_cols = [c for c in ["duration_sec", "ad_density_pct", "screencast_mean", "hook_score"] if c in bad_df.columns]
    if not cluster_cols:
        bad_df["cluster"] = 0
        return bad_df

    X = bad_df[cluster_cols].fillna(0).values
    X = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=min(n_clusters, len(bad_df)), random_state=42)
    bad_df["cluster"] = kmeans.fit_predict(X)
    return bad_df


def plot_residual_curves(video_frames: dict, predictions: dict, base_output_dir: str, top_n_worst: int = 5):
    """Plot residual curves for worst videos. Saves to base_output_dir/videos/{vid}/residual/residual.png"""
    metrics = []
    for vid, (y_true, y_pred) in predictions.items():
        mae = float(np.mean(np.abs(y_true - y_pred)))
        metrics.append((vid, mae, y_true, y_pred))
    metrics.sort(key=lambda x: -x[1])

    for i, (vid, mae, y_true, y_pred) in enumerate(metrics[:top_n_worst]):
        t = np.arange(len(y_true))
        residual = y_pred - y_true

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.plot(t, y_true, label="actual", color="#2196F3")
        ax1.plot(t, y_pred, label="predicted", color="#FF5722", alpha=0.8)
        ax1.set(ylabel="Retention (%)", title=f"{vid} — MAE={mae:.2f} pp")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(t, residual, alpha=0.3, color="#4CAF50", where=residual >= 0)
        ax2.fill_between(t, residual, alpha=0.3, color="#F44336", where=residual < 0)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set(xlabel="sec", ylabel="residual (pp)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_dir = os.path.join(base_output_dir, "videos", vid, "residual")
        os.makedirs(plot_dir, exist_ok=True)
        path = os.path.join(plot_dir, "residual.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", path)


def run_error_analysis(features_dir: str = "output", model_path: str = "static/weights/model.cbm", output_dir: str = "my_metrics"):
    """
    output_dir: base dir (e.g. my_metrics). Saves:
      - output_dir/error_analysis/error_report.csv
      - output_dir/error_analysis/worst_videos_clustered.csv
      - output_dir/videos/{vid}/residual/residual.png (for top 5 worst)
    """
    error_dir = os.path.join(output_dir, "error_analysis")
    os.makedirs(error_dir, exist_ok=True)

    logger.info("Loading features and running model...")
    video_frames, predictions = load_features_and_predict(features_dir, model_path)

    logger.info("Computing per-video metrics...")
    metrics_df = compute_per_video_metrics(video_frames, predictions)
    metrics_df.to_csv(os.path.join(error_dir, "error_report.csv"), index=False)
    logger.info("Saved error_report.csv with %d videos", len(metrics_df))

    logger.info("Clustering bad videos...")
    bad_clustered = cluster_bad_videos(metrics_df)
    bad_clustered.to_csv(os.path.join(error_dir, "worst_videos_clustered.csv"), index=False)

    logger.info("Plotting residual curves for worst videos...")
    plot_residual_curves(video_frames, predictions, output_dir)

    logger.info("Error analysis done. Results in %s", output_dir)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error analysis for retention model")
    parser.add_argument("--features_dir", default="output")
    parser.add_argument("--model_path", default="static/weights/model.cbm")
    parser.add_argument("--output_dir", default="my_metrics", help="Base dir; CSVs go to output_dir/error_analysis/, plots to output_dir/videos/{vid}/residual/")
    args = parser.parse_args()
    run_error_analysis(args.features_dir, args.model_path, args.output_dir)
