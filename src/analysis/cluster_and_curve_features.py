import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from train.common.seq_data_utils import load_all_merged


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- CURVE FITTING FUNCTIONS ---
_C_FALLBACK_RATIO = 0.15


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


def get_best_curve(time_sec, retention):
    """Try all 3 curves, return the one with minimum MAE along with its type and parameters."""
    curves = {"hill": fit_hill_curve, "double_exp": fit_double_exp, "weibull": fit_weibull}

    best_mae = float("inf")
    best_type = "hill"
    best_params = np.zeros(5)

    for c_type, c_func in curves.items():
        try:
            pred_y, params = c_func(time_sec, retention)
            mae = np.mean(np.abs(retention - pred_y))
            if mae < best_mae:
                best_mae = mae
                best_type = c_type
                # pad params to length 5
                p_pad = np.zeros(5)
                p_pad[: len(params)] = params
                best_params = p_pad
        except Exception:
            continue

    # mapping type to integer to be used as feature
    type_map = {"hill": 0, "double_exp": 1, "weibull": 2}

    return type_map[best_type], best_params, best_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters-file", default="analysis/video_clustering/kmeans/clusters.json")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    # 1. Load clusters
    cluster_map = {}
    if os.path.exists(args.clusters_file):
        with open(args.clusters_file, encoding="utf-8") as f:
            c_data = json.load(f)
            for vid, v_info in c_data.get("videos", {}).items():
                cluster_map[vid] = v_info.get("cluster_id", v_info.get("kmeans_cluster_id", -1))
        logger.info(f"Loaded clusters for {len(cluster_map)} videos")
    else:
        logger.warning(f"Clusters file not found: {args.clusters_file}")

    # 2. Get retention curves
    logger.info("Loading retention curves to fit parameters...")
    # use_curve_raw=True because we want to fit against the real unsmoothed curve
    video_dfs = load_all_merged(args.output_dir, args.data_dir, use_curve_raw=True, emb_pca_components=0)

    # Compute best curve per video
    curve_features = {}
    for vid, df in video_dfs.items():
        if "time" not in df.columns or "retention" not in df.columns:
            continue

        df_sorted = df.sort_values("time").reset_index(drop=True)
        # Using index * some duration step or just raw index as pseudo time for curve fitting if time_sec is not pure
        t_sec = df_sorted["time"].values if pd.api.types.is_numeric_dtype(df_sorted["time"]) else np.arange(len(df_sorted))
        ret = df_sorted["retention"].values

        c_type, c_params, c_mae = get_best_curve(t_sec, ret)
        curve_features[vid] = {
            "best_curve_type": c_type,
            "curve_p0": c_params[0],
            "curve_p1": c_params[1],
            "curve_p2": c_params[2],
            "curve_p3": c_params[3],
            "curve_p4": c_params[4],
            "curve_fit_mae": c_mae,
        }
    logger.info(f"Computed curve parameters for {len(curve_features)} videos")

    # 3. Update all CSVs in output_dir
    csv_files = glob.glob(os.path.join(args.output_dir, "*_features.csv")) + glob.glob(os.path.join(args.output_dir, "*_features.csv.partial"))

    updated_count = 0
    for path in csv_files:
        vid = os.path.basename(path).replace("_features.csv.partial", "").replace("_features.csv", "")
        df = pd.read_csv(path, index_col=0 if pd.read_csv(path, nrows=0).columns[0] == "Unnamed: 0" else None)

        # Performance warning fix: build a dict of new columns and concat once
        new_cols = {}

        if "video_cluster" in df.columns:
            df.drop(columns=["video_cluster"], inplace=True)

        c_id = cluster_map.get(vid, -1)
        new_cols["video_cluster"] = c_id

        # Remove old curve cols if present
        for col in ["best_curve_type", "curve_p0", "curve_p1", "curve_p2", "curve_p3", "curve_p4", "curve_fit_mae"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        c_feats = curve_features.get(vid, {})
        for k, v in c_feats.items():
            new_cols[k] = v

        # Add all new columns at once to prevent DataFrame fragmentation
        df = pd.concat([df, pd.DataFrame([new_cols] * len(df), index=df.index)], axis=1)

        df.to_csv(path)
        updated_count += 1

    logger.info(f"Updated {updated_count} files in {args.output_dir}")


if __name__ == "__main__":
    main()
