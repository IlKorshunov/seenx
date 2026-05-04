import json
import os

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SMOOTH_WINDOW = 15
_OVERVIEW_LAYOUT = [
    ("Retention", ["retention"], False, False),
    ("Bumper score", ["bumper_score"], False, True),
    ("Edit pace", ["edit_pace"], False, False),
    ("Screencast probability", ["screencast_prob"], False, False),
    ("Speaker probability", ["speaker_prob"], False, False),
    ("Audio dynamics", ["rms", "zcr", "centroid"], True, False),
    ("Motion / optical flow", ["motion_speed", "flow_mag_med"], True, False),
    ("Question density", ["question_density"], True, False),
]
_OVERVIEW_COLORS = {
    "retention": "#2196F3",
    "bumper_score": "#D62728",
    "edit_pace": "#FF7F0E",
    "screencast_prob": "#9467BD",
    "speaker_prob": "#FF5722",
    "rms": "#1F77B4",
    "zcr": "#FF7F0E",
    "centroid": "#2CA02C",
    "motion_speed": "#1F77B4",
    "flow_mag_med": "#17BECF",
    "question_density": "#8C564B",
}
_COMMON_SKIP = {"frame"}


def smooth(series: pd.Series, window: int = SMOOTH_WINDOW) -> pd.Series:
    return series.rolling(window, center=True, min_periods=1).mean()


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").ffill().fillna(0.0)


def _plot_overview(df: pd.DataFrame, time: np.ndarray, title: str, output_path: str) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 16), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    flat_axes = axes.flatten()

    for ax, (panel_title, cols, use_zscore, fill_area) in zip(flat_axes, _OVERVIEW_LAYOUT, strict=True):
        present = [col for col in cols if col in df.columns]
        if not present:
            ax.axis("off")
            continue

        for col in present:
            values = _to_numeric_series(df, col)
            if use_zscore:
                values = (values - values.mean()) / (values.std() + 1e-9)
                values = smooth(values)
            elif col == "question_density":
                values = smooth(values)
            color = _OVERVIEW_COLORS.get(col, "#1F77B4")
            ax.plot(time, values, label=col, color=color, linewidth=1.2, alpha=0.9)
            if fill_area and len(present) == 1:
                ax.fill_between(time, values, alpha=0.15, color=color)

        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("sec")
        ax.grid(True, alpha=0.3)
        if len(present) > 1:
            ax.legend(fontsize=8)
        else:
            ax.legend([present[0]], fontsize=8)

    for ax in flat_axes[len(_OVERVIEW_LAYOUT) :]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _iter_common_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in _COMMON_SKIP:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        cols.append(col)
    return cols


def _plot_common_features(df: pd.DataFrame, time: np.ndarray, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for col in _iter_common_feature_columns(df):
        values = _to_numeric_series(df, col)
        fig, ax = plt.subplots(figsize=(14, 3.5))
        ax.plot(time, values, color="#1F77B4", linewidth=1.0, alpha=0.9, label=col)
        if values.std() > 0:
            ax.plot(time, smooth(values), color="#FF7F0E", linewidth=1.0, alpha=0.8, label=f"{col} (smooth)")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("sec")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{col}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def summarize_video(features_csv: str, meta_json: str, output_dir: str):
    video_id = os.path.basename(features_csv).replace("_features.csv", "")
    df = pd.read_csv(features_csv, index_col=0)
    video_dir = os.path.join(output_dir, "videos", video_id)
    overview_dir = os.path.join(video_dir, "overview")
    common_dir = os.path.join(video_dir, "common")
    os.makedirs(overview_dir, exist_ok=True)
    os.makedirs(common_dir, exist_ok=True)

    meta = {}
    if os.path.exists(meta_json):
        with open(meta_json, encoding="utf-8") as f:
            meta = json.load(f)

    has_retention = "retention" in df.columns
    time = np.arange(len(df))
    plot_title = meta.get("title", video_id)

    _plot_overview(df, time, plot_title, os.path.join(overview_dir, "overview.png"))
    _plot_common_features(df, time, common_dir)

    stats = {"video_id": video_id, "duration_sec": len(df), "n_features": len(df.columns)}
    stats.update(meta)
    if has_retention:
        ret = df["retention"].values
        stats["mean_retention"] = round(float(np.mean(ret)), 2)
        stats["retention_30s"] = round(float(ret[30]), 2) if len(ret) > 30 else None
        stats["avd_sec"] = round(len(ret) * float(np.mean(ret)) / 100, 1)

    return stats


def summarize_all(features_dir: str, data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    all_stats = []

    csvs = sorted(f for f in os.listdir(features_dir) if f.endswith("_features.csv"))
    if not csvs:
        print("no feature CSVs found")
        return

    for csv_name in csvs:
        video_id = csv_name.replace("_features.csv", "")
        meta_path = os.path.join(data_dir, video_id, "meta.json")
        csv_path = os.path.join(features_dir, csv_name)
        print(f"  {video_id}")
        stats = summarize_video(csv_path, meta_path, output_dir)
        all_stats.append(stats)

    stats_df = pd.DataFrame(all_stats)
    stats_path = os.path.join(output_dir, "summary.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="output")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="my_metrics")
    args = parser.parse_args()
    summarize_all(args.features_dir, args.data_dir, args.output_dir)
