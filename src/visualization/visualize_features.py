"""
Feature timeline visualization for a single video.

Usage:
    python -m src.visualization.visualize_features \
        --features output/my_video_features.csv \
        --output   output/my_video_features.png \
        --title    "Interns S01E02"

    # or just preview in a window (no --output):
    python -m src.visualization.visualize_features --features output/my_video_features.csv
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

# ---------------------------------------------------------------------------
# Feature catalogue: groups that will become separate panels.
# Each entry: (panel_label, y_label, [(column_name, legend_label, color), ...])
# Only columns that actually exist in the DataFrame are drawn.
# ---------------------------------------------------------------------------
_FEATURE_GROUPS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    ("Retention", "Retention, %", [("retention", "retention", "#1f77b4")]),
    ("Bumper", "Score (0–1)", [("bumper_score", "bumper score", "#d62728")]),
    ("Content structure", "Value", [("edit_pace", "edit pace (cuts/min)", "#ff7f0e"), ("scene_novelty", "scene novelty", "#2ca02c")]),
    ("Visual content", "Probability", [("screencast_prob", "screencast prob", "#9467bd")]),
    ("Speech / engagement", "Density", [("viewer_address", "viewer address", "#8c564b"), ("wps", "words per sec", "#e377c2")]),
    ("Speaker presence", "Probability", [("speaker_probability", "speaker prob", "#17becf"), ("face_screen_ratio", "face ratio", "#bcbd22")]),
]


def _load_csv(csv_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (time_seconds, df) from a *_features.csv produced by aggregate()."""
    df = pd.read_csv(csv_path, index_col=0)

    # Index is timedelta strings like "0 days 00:00:00" or plain seconds.
    try:
        td = pd.to_timedelta(df.index)
        time_sec = td.total_seconds().values
    except Exception:
        try:
            time_sec = df.index.astype(float).values
        except Exception:
            time_sec = np.arange(len(df), dtype=float)

    return time_sec, df


def _active_panels(df: pd.DataFrame) -> list[tuple[str, str, list[tuple[str, str, str]]]]:
    """Return only the panels that have ≥1 column present in df."""
    result = []
    for panel_label, y_label, series in _FEATURE_GROUPS:
        active = [(col, leg, color) for col, leg, color in series if col in df.columns]
        if active:
            result.append((panel_label, y_label, active))
    return result


def _bumper_spans(time_sec: np.ndarray, df: pd.DataFrame, threshold: float = 0.3) -> list[tuple[float, float]]:
    """Return list of (start_sec, end_sec) intervals where bumper_score > threshold."""
    if "bumper_score" not in df.columns:
        return []
    score = df["bumper_score"].fillna(0).values
    spans: list[tuple[float, float]] = []
    in_span = False
    t0 = 0.0
    for i, (t, s) in enumerate(zip(time_sec, score, strict=True)):
        if not in_span and s >= threshold:
            t0 = t
            in_span = True
        elif in_span and s < threshold:
            spans.append((t0, t))
            in_span = False
    if in_span:
        spans.append((t0, float(time_sec[-1])))
    return spans


def plot_features(
    csv_path: str, output_path: str | None = None, title: str | None = None, figsize_per_panel: tuple[float, float] = (18.0, 2.5), bumper_threshold: float = 0.3
) -> None:
    """
    Read a *_features.csv file and produce a multi-panel timeline chart.

    Parameters
    ----------
    csv_path : path to the features CSV (output of aggregate()).
    output_path : if given, saves the figure to this path (PNG/PDF/…);
                  otherwise shows interactively.
    title : optional figure title (defaults to the CSV filename).
    figsize_per_panel : (width, height) in inches for each panel.
    bumper_threshold : bumper_score value above which we shade the panel background.
    """
    time_sec, df = _load_csv(csv_path)
    panels = _active_panels(df)

    if not panels:
        raise ValueError(f"No recognisable feature columns found in {csv_path}.\nAvailable columns: {list(df.columns)}")

    n = len(panels)
    fig_w, panel_h = figsize_per_panel
    fig = plt.figure(figsize=(fig_w, panel_h * n + 0.8))

    gs = GridSpec(
        n,
        1,
        figure=fig,
        hspace=0.08,  # very tight — shared x-axis feel
        top=0.96,
        bottom=0.05,
        left=0.07,
        right=0.97,
    )

    bumper_spans = _bumper_spans(time_sec, df, bumper_threshold)

    axes: list[plt.Axes] = []
    for row, (panel_label, y_label, series_list) in enumerate(panels):
        ax = fig.add_subplot(gs[row], sharex=axes[0] if axes else None)
        axes.append(ax)

        # Shade bumper regions on every panel.
        for t_start, t_end in bumper_spans:
            ax.axvspan(t_start, t_end, color="#d62728", alpha=0.08, zorder=0)

        for col, leg, color in series_list:
            values = df[col].fillna(method="ffill").fillna(0).values.astype(float)
            ax.plot(time_sec, values, color=color, linewidth=0.9, label=leg, alpha=0.85)

        ax.set_ylabel(y_label, fontsize=8, labelpad=4)
        ax.set_title(panel_label, fontsize=9, loc="left", pad=2, fontweight="bold")

        # Legend inside panel, top-right, compact.
        leg_handles = [mpatches.Patch(color=color, label=leg, alpha=0.9) for _, leg, color in series_list]
        ax.legend(handles=leg_handles, fontsize=7, loc="upper right", framealpha=0.6, handlelength=1.2, borderpad=0.4)

        # Only the last panel shows the x-axis ticks/label.
        if row < n - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time, seconds", fontsize=9)

        ax.tick_params(axis="both", labelsize=7)

    # Mark total duration on the x-axis with a dashed vertical line.
    for ax in axes:
        ax.set_xlim(float(time_sec[0]), float(time_sec[-1]))

    # Optional legend for bumper shading (added to the top panel).
    if bumper_spans:
        bump_patch = mpatches.Patch(color="#d62728", alpha=0.15, label="bumper region")
        existing = axes[0].get_legend()
        handles = [h for h in (existing.legend_handles if existing else [])] + [bump_patch]
        axes[0].legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.6, handlelength=1.2, borderpad=0.4)

    fig.suptitle(title or os.path.splitext(os.path.basename(csv_path))[0], fontsize=12, fontweight="bold", y=0.99)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visualise per-second features for a single video.")
    parser.add_argument("--features", "-f", required=True, help="Path to *_features.csv produced by main.py aggregate")
    parser.add_argument("--output", "-o", default=None, help="Where to save the chart (PNG/PDF). If omitted, shows interactively.")
    parser.add_argument("--title", "-t", default=None, help="Chart title (defaults to the CSV filename without extension).")
    parser.add_argument("--bumper-threshold", type=float, default=0.3, help="bumper_score threshold above which to shade the background (default 0.3).")
    args = parser.parse_args(argv)

    plot_features(csv_path=args.features, output_path=args.output, title=args.title, bumper_threshold=args.bumper_threshold)


if __name__ == "__main__":
    main()
