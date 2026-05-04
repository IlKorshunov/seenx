import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from .seenx_utils import get_video_duration
from .utils.logger import Logger

logger = Logger(show=True).get_logger()

AlignStrategy = Literal["min_duration", "max_duration", "mean_duration", "extrapolate", "normalized_100"]

NORMALIZED_BASELINE_POINTS = 100
RETENTION_FEATURE_EXCLUDE = {"retention", "frame", "time", "time_pct", "hook_score_x_time_pct", "hook_score_x_time_pct.1", "edit_pace_x_screencast.1", "is_ad_x_viewer_address.1"}

COLORS = {"retention": "#2196F3", "avd": "#FF5722", "retention_mark": "#4CAF50", "baseline": "#FF5722", "baseline_avd": "#E91E63", "std_fill": "#FF9800", "individual": "#9E9E9E"}

STRATEGY_LABELS = {
    "min_duration": "By shortest video",
    "max_duration": "By longest video",
    "mean_duration": "By mean duration",
    "extrapolate": "Extrapolate",
    "normalized_100": "By normalized progress",
}


def compute_avd_from_retention(retention_series: np.ndarray, duration_seconds: float) -> float:
    return float(duration_seconds * np.mean(retention_series) / 100.0)


def calc_retention_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    mse = float(np.mean((y_pred - y_true) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-9))
    return {"mse": round(mse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def compute_retention_at(retention_series: np.ndarray, at_second: float = 30.0) -> float | None:
    idx = int(round(at_second))
    if idx >= len(retention_series):
        return None
    return float(retention_series[idx])


def video_retention_metrics(retention_series: np.ndarray, duration_seconds: float) -> dict:
    avd = compute_avd_from_retention(retention_series, duration_seconds)
    retention_30 = compute_retention_at(retention_series, 30.0)
    return {
        "duration_sec": duration_seconds,
        "avd_sec": round(avd, 2),
        "avd_pct": round(avd / duration_seconds * 100, 2) if duration_seconds > 0 else 0,
        "retention_30": round(retention_30, 2) if retention_30 is not None else None,
        "mean_retention": round(float(np.mean(retention_series)), 2),
    }


def load_channel_retentions_csv(data_dir: str) -> list[dict]:
    channel_data = []
    for name in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, name)
        csv_path, video_path = os.path.join(folder, "retention.csv"), os.path.join(folder, "video.mp4")
        if not (os.path.isfile(csv_path) and os.path.isfile(video_path)):
            continue
        try:
            duration_sec = get_video_duration(video_path)
            csv_df = pd.read_csv(csv_path)
            n_points = int(duration_sec) + 1
            retention_values = np.interp(np.linspace(0, 1, n_points), csv_df["time_ratio"].to_numpy(dtype=float), csv_df["audience_watch_ratio"].to_numpy(dtype=float) * 100.0)
            channel_data.append({"name": name, "retention_series": retention_values, "duration_sec": float(duration_sec), "time_index": np.arange(n_points)})
            logger.info("Loaded %s (%.0fs, %d pts)", name, duration_sec, n_points)
        except Exception as e:
            logger.warning("Failed to load %s: %s", name, e)
    logger.info("Loaded %d retention curves from CSV", len(channel_data))
    return channel_data


def _resample_retention(retention: np.ndarray, target_length: int) -> np.ndarray:
    if len(retention) == target_length:
        return retention.copy()
    return np.interp(np.linspace(0, 1, target_length), np.linspace(0, 1, len(retention)), retention)


def _extrapolate_trend(retention: np.ndarray, target_length: int, trend_window: int = 30) -> np.ndarray:
    if len(retention) >= target_length:
        return _resample_retention(retention, target_length)
    if len(retention) < 2:
        return np.concatenate([retention, np.full(target_length - len(retention), retention[-1])])
    window = min(trend_window, len(retention))
    slope = np.polyfit(np.arange(window), retention[-window:], deg=1)[0]
    extra = retention[-1] + slope * np.arange(1, target_length - len(retention) + 1)
    return np.concatenate([retention, np.clip(extra, 0, 100)])


def _align_retention(retention: np.ndarray, target_length: int, strategy: AlignStrategy, trend_window: int) -> np.ndarray:
    if strategy == "extrapolate":
        return _extrapolate_trend(retention, target_length, trend_window)
    return _resample_retention(retention, target_length)


def _normalized_baseline_metrics(baseline: np.ndarray, time_axis: np.ndarray, duration_seconds: float) -> dict:
    avd = compute_avd_from_retention(baseline, duration_seconds)
    return {
        "duration_sec": duration_seconds,
        "avd_sec": round(avd, 2),
        "avd_pct": round(float(np.mean(baseline)), 2) if duration_seconds > 0 else 0,
        "retention_30": None,
        "retention_30_pct": round(float(np.interp(30.0, time_axis, baseline)), 2),
        "mean_retention": round(float(np.mean(baseline)), 2),
    }


def compute_channel_baseline(channel_data: list[dict], strategy: AlignStrategy = "mean_duration", trend_window: int = 30) -> dict | None:
    if not channel_data:
        raise ValueError("channel_data is empty")
    lengths = [len(d["retention_series"]) for d in channel_data]
    durations = [d["duration_sec"] for d in channel_data]
    is_normalized = strategy == "normalized_100"
    logger.info("Channel stats: %d videos, durations: min=%ds, max=%ds, mean=%ds", len(channel_data), min(durations), max(durations), np.mean(durations))

    if is_normalized:
        target_length = NORMALIZED_BASELINE_POINTS
        time_axis = np.linspace(0, 100, target_length)
        x_label = "Video progress (%)"
        aligned_matrix = np.array([_resample_retention(d["retention_series"], target_length) for d in channel_data])
    else:
        target_length = {"min_duration": min(lengths), "max_duration": max(lengths), "mean_duration": int(round(np.mean(lengths))), "extrapolate": max(lengths)}.get(strategy)
        if target_length is None:
            raise ValueError(f"Unknown strategy: {strategy}")
        time_axis = np.arange(target_length)
        x_label = "Time (seconds)"
        aligned_matrix = np.array([_align_retention(d["retention_series"], target_length, strategy, trend_window) for d in channel_data])

    logger.info("Strategy: %s, target length: %d points", strategy, target_length)
    baseline = np.mean(aligned_matrix, axis=0)
    baseline_metrics = _normalized_baseline_metrics(baseline, time_axis, float(np.mean(durations))) if is_normalized else video_retention_metrics(baseline, target_length - 1)

    return {
        "target_duration": float(np.mean(durations)) if is_normalized else target_length - 1,
        "target_length": target_length,
        "baseline_retention": baseline,
        "baseline_std": np.std(aligned_matrix, axis=0),
        "time_axis": time_axis,
        "x_label": x_label,
        "strategy": strategy,
        "n_videos": len(channel_data),
        "baseline_metrics": baseline_metrics,
        "individual_metrics": [{"name": d["name"], **video_retention_metrics(d["retention_series"], d["duration_sec"])} for d in channel_data],
        "aligned_matrix": aligned_matrix,
    }


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _add_vline(ax, x, color, label):
    ax.axvline(x=x, color=color, linestyle="--", linewidth=1.5, label=label)


def _add_info_box(ax, lines: list[str]):
    ax.text(0.02, 0.02, os.linesep.join(lines), transform=ax.transAxes, fontsize=9, verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))


def _save_and_show(fig, output_path, show):
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", output_path)
    if show:
        plt.show()


def _info_lines(metrics: dict, *, n_videos: int | None = None, strategy_label: str | None = None) -> list[str]:
    lines = []
    if n_videos is not None:
        lines.append(f"Videos: {n_videos}")
    if strategy_label is not None:
        lines.append(f"Strategy: {strategy_label}")
    lines += [
        f"Duration: {format_time(metrics['duration_sec'])}",
        f"AVD: {format_time(metrics['avd_sec'])} ({metrics['avd_pct']}%)",
        f"Mean retention: {metrics['mean_retention']:.1f}%",
    ]
    if metrics.get("retention_30_pct") is not None:
        lines.append(f"Retention at 30%: {metrics['retention_30_pct']:.1f}%")
    elif metrics.get("retention_30") is not None:
        lines.append(f"Retention at 30s: {metrics['retention_30']:.1f}%")
    return lines


def plot_single_retention(retention_series: np.ndarray, duration_sec: float, title: str = "Retention Graph", output_path: str | None = None, show: bool = True) -> Figure:
    metrics = video_retention_metrics(retention_series, duration_sec)
    time_axis = np.arange(len(retention_series))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_axis, retention_series, color=COLORS["retention"], linewidth=2, label="Retention")
    ax.fill_between(time_axis, retention_series, alpha=0.15, color=COLORS["retention"])
    avd_sec = metrics["avd_sec"]
    if avd_sec < len(retention_series):
        _add_vline(ax, avd_sec, COLORS["avd"], f"AVD = {format_time(avd_sec)} ({metrics['avd_pct']}%)")
    ax.set(xlabel="Time (seconds)", ylabel="Retention (%)", title=title, ylim=(0, max(retention_series.max() * 1.05, 105)), xlim=(0, len(retention_series) - 1))
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    _add_info_box(ax, _info_lines(metrics))
    _save_and_show(fig, output_path, show)
    return fig


def plot_channel_baseline(baseline_result: dict, output_path: str | None = None, show_individual: bool = True, show: bool = True) -> Figure:
    baseline, std = baseline_result["baseline_retention"], baseline_result["baseline_std"]
    time_axis, n_videos = baseline_result["time_axis"], baseline_result["n_videos"]
    bm = baseline_result["baseline_metrics"]
    strategy = baseline_result["strategy"]
    strategy_label = STRATEGY_LABELS.get(strategy, strategy)
    fig, ax = plt.subplots(figsize=(14, 7))

    if show_individual:
        for curve in baseline_result["aligned_matrix"]:
            ax.plot(time_axis, curve, alpha=0.25, linewidth=0.8, color=COLORS["individual"])

    ax.fill_between(time_axis, baseline - std, baseline + std, alpha=0.2, color=COLORS["std_fill"], label="+-1 std")
    ax.plot(time_axis, baseline, color=COLORS["baseline"], linewidth=2.5, label=f"Baseline ({n_videos} videos)")

    avd_sec = bm["avd_sec"]
    if strategy != "normalized_100" and avd_sec < len(baseline):
        _add_vline(ax, avd_sec, COLORS["baseline_avd"], f"AVD = {format_time(avd_sec)} ({bm['avd_pct']}%)")

    ax.set(
        xlabel=baseline_result.get("x_label", "Time (seconds)"),
        ylabel="Retention (%)",
        title=f"Channel Retention Baseline{os.linesep}Strategy: {strategy_label}",
        ylim=(0, min(baseline.max() * 1.15, 120)),
        xlim=(time_axis[0], time_axis[-1]),
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    _add_info_box(ax, _info_lines(bm, n_videos=n_videos, strategy_label=strategy_label))
    _save_and_show(fig, output_path, show)
    return fig


def plot_all_strategies(channel_data: list[dict], output_path: str | None = None, show: bool = True) -> Figure:
    strategies: list[AlignStrategy] = ["min_duration", "max_duration", "mean_duration", "extrapolate", "normalized_100"]
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    flat_axes = axes.flatten()

    for ax, strategy in zip(flat_axes, strategies, strict=False):
        result = compute_channel_baseline(channel_data, strategy=strategy)
        if result is None:
            continue

        baseline, std, time_axis = result["baseline_retention"], result["baseline_std"], result["time_axis"]
        metrics = result["baseline_metrics"]

        for curve in result["aligned_matrix"]:
            ax.plot(time_axis, curve, alpha=0.2, linewidth=0.6, color=COLORS["individual"])

        ax.fill_between(time_axis, baseline - std, baseline + std, alpha=0.2, color=COLORS["std_fill"])
        ax.plot(time_axis, baseline, color=COLORS["baseline"], linewidth=2, label="Baseline")
        avd_sec = metrics["avd_sec"]
        if strategy != "normalized_100" and avd_sec < len(baseline):
            _add_vline(ax, avd_sec, COLORS["baseline_avd"], f"AVD = {format_time(avd_sec)}")

        strategy_label = STRATEGY_LABELS.get(strategy, strategy)
        subtitle = f"AVD={format_time(avd_sec)}"
        if metrics.get("retention_30_pct") is not None:
            subtitle += f", 30%={metrics['retention_30_pct']:.1f}%"
        elif metrics.get("retention_30") is not None:
            subtitle += f", 30s={metrics['retention_30']:.1f}%"

        ax.set_title(f"{strategy_label}{os.linesep}{subtitle}", fontsize=11, fontweight="bold")
        ax.set(xlabel=result.get("x_label", "Time (s)"), ylabel="Retention (%)")
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, min(baseline.max() * 1.15, 120))
        ax.legend(fontsize=8)

    for ax in flat_axes[len(strategies) :]:
        ax.axis("off")

    plt.suptitle(f"Comparison of {len(strategies)} retention averaging strategies ({len(channel_data)} videos)", fontsize=15, fontweight="bold", y=1.02)
    _save_and_show(fig, output_path, show)
    return fig


def channel_metrics_table(channel_data: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{"name": d["name"], **video_retention_metrics(d["retention_series"], d["duration_sec"])} for d in channel_data])
    df = df[["name"] + [c for c in df.columns if c != "name"]]
    mean_row = df.select_dtypes(include=[np.number]).mean()
    mean_row["name"] = "MEAN"
    return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
