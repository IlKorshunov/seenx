import json
import os

import numpy as np
import pandas as pd
from scipy import stats


MARKUP_DIR = "markup"
OUTPUT_DIR = "output"
REPORT_DIR = "markup/comparison_report"

CONTINUOUS_COLS = [
    "speaker_prob",
    "face_screen_ratio",
    "cinematic",
    "screencast_prob",
    "bumper_score",
    "visual_entropy",
    "viewer_address",
    "crutch_cnt",
    "has_person_mention",
    "has_org_mention",
    "pos_cnt",
    "neg_cnt",
    "hook_score",
    "hook_has_question",
    "hook_has_address",
    "is_ad",
]

CATEGORICAL_COLS = []

SKIP_COLS = {"time", "retention", "time_pct"}


def _parse_time_column(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int).to_numpy()
    td = pd.to_timedelta(series, errors="coerce")
    if td.notna().all():
        return td.dt.total_seconds().astype(int).to_numpy()
    raise ValueError(f"Cannot parse 'time' column: first values = {series.head(3).tolist()}")


def _safe_compare_video(video_id: str) -> pd.DataFrame | None:
    try:
        return compare_video(video_id)
    except Exception as e:
        print(f"WRN: {video_id}: compare failed: {e}", flush=True)
        return None


def aggregate_to_bins(features_df: pd.DataFrame, segments: list[dict]) -> pd.DataFrame:
    if "time" not in features_df.columns:
        raise ValueError("Features CSV has no 'time' column")
    seconds = _parse_time_column(features_df["time"])
    features_df = features_df.drop(columns=list(SKIP_COLS & set(features_df.columns)))
    features_df = features_df.apply(pd.to_numeric, errors="coerce")
    features_df.index = seconds

    rows = []
    for seg in segments:
        start, end = int(seg["start_sec"]), int(seg["end_sec"])
        mask = (features_df.index >= start) & (features_df.index < end)
        chunk = features_df.loc[mask]
        if chunk.empty:
            rows.append({})
        else:
            rows.append(chunk.mean(numeric_only=True).to_dict())
    return pd.DataFrame(rows)


def compute_metrics(gt: pd.Series, ext: pd.Series, col_name: str) -> dict:
    valid = gt.notna() & ext.notna()
    if valid.sum() < 3:
        return {"feature": col_name, "n": int(valid.sum()), "pearson": None, "spearman": None, "mae": None}
    g, e = gt[valid].values.astype(float), ext[valid].values.astype(float)

    pearson_r = float(np.corrcoef(g, e)[0, 1]) if np.std(g) > 1e-9 and np.std(e) > 1e-9 else 0.0
    spearman_r = float(stats.spearmanr(g, e).statistic) if np.std(g) > 1e-9 and np.std(e) > 1e-9 else 0.0
    mae = float(np.mean(np.abs(g - e)))
    return {
        "feature": col_name,
        "n": int(valid.sum()),
        "pearson": round(pearson_r, 3),
        "spearman": round(spearman_r, 3),
        "mae": round(mae, 4),
        "gt_mean": round(float(np.mean(g)), 4),
        "ext_mean": round(float(np.mean(e)), 4),
    }


def compare_video(video_id: str) -> pd.DataFrame | None:
    gt_path = os.path.join(MARKUP_DIR, video_id, "ground_truth.csv")
    seg_path = os.path.join(MARKUP_DIR, video_id, "segments.json")
    feat_path = os.path.join(OUTPUT_DIR, f"{video_id}_features.csv")

    if not os.path.exists(gt_path):
        print(f"WRN: {video_id}: no ground_truth.csv, skipping", flush=True)
        return None
    if not os.path.exists(feat_path):
        print(f"WRN: {video_id}: no features CSV ({feat_path}), skipping", flush=True)
        return None

    gt = pd.read_csv(gt_path)
    with open(seg_path) as f:
        segments = json.load(f)

    features = pd.read_csv(feat_path)
    agg = aggregate_to_bins(features, segments)

    common_cols = [c for c in CONTINUOUS_COLS if c in gt.columns and c in agg.columns]
    if not common_cols:
        print(f"WRN: {video_id}: no common columns to compare", flush=True)
        return None

    results = []
    for col in common_cols:
        metrics = compute_metrics(gt[col], agg[col], col)
        metrics["video_id"] = video_id
        results.append(metrics)

    df = pd.DataFrame(results)
    print(f"INF: {video_id}: compared {len(common_cols)} features across {len(segments)} segments", flush=True)
    return df


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    all_results = []
    for d in sorted(os.listdir(MARKUP_DIR)):
        if os.path.isdir(os.path.join(MARKUP_DIR, d)) and d != "comparison_report":
            result = _safe_compare_video(d)
            if result is not None:
                all_results.append(result)

    if not all_results:
        print("WRN: no comparable videos found; nothing to write", flush=True)
        return

    combined = pd.concat(all_results, ignore_index=True)

    summary = (
        combined.groupby("feature")
        .agg(n_videos=("video_id", "nunique"), n_segments=("n", "sum"), mean_pearson=("pearson", "mean"), mean_spearman=("spearman", "mean"), mean_mae=("mae", "mean"))
        .sort_values("mean_spearman", ascending=False)
        .reset_index()
    )

    combined_path = os.path.join(REPORT_DIR, "per_video_metrics.csv")
    summary_path = os.path.join(REPORT_DIR, "summary.csv")
    combined.to_csv(combined_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Per-video metrics -> {combined_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
