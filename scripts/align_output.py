#!/usr/bin/env python3
"""
Convert raw *_features.csv into human-readable structure with per-feature plots.

Outputs for each video:
  output_aligned/<video_id>/features_readable.csv
  output_aligned/<video_id>/features_long.csv
  output_aligned/<video_id>/features_summary.csv
  output_aligned/<video_id>/plots/<feature>.png

Usage:
  python3 scripts/align_output.py
  python3 scripts/align_output.py --input-dir output --output-dir output_aligned
  python3 scripts/align_output.py --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRIORITY_COLUMNS = ["time", "retention", "edit_pace", "scene_novelty", "bumper_score", "hook_score", "viewer_address", "crutch_cnt", "speech_complexity"]


def _format_time_to_hhmmss(series: pd.Series) -> pd.Series:
    td = pd.to_timedelta(series, errors="coerce")
    if td.notna().sum() == 0:
        return series.astype(str)
    seconds = td.dt.total_seconds().round().astype("Int64")
    hours = (seconds // 3600).astype("Int64")
    minutes = ((seconds % 3600) // 60).astype("Int64")
    secs = (seconds % 60).astype("Int64")
    return hours.astype(str).str.zfill(2) + ":" + minutes.astype(str).str.zfill(2) + ":" + secs.astype(str).str.zfill(2)


def _normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time" in out.columns:
        out["time"] = _format_time_to_hhmmss(out["time"])
        return out
    first_col = out.columns[0]
    if str(first_col).lower().startswith("unnamed"):
        out = out.rename(columns={first_col: "time"})
        out["time"] = _format_time_to_hhmmss(out["time"])
    return out


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    available_priority = [c for c in PRIORITY_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in available_priority]
    return df[available_priority + rest]


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c != "time" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return pd.DataFrame(columns=["feature", "mean", "std", "min", "p10", "median", "p90", "max", "non_zero_pct"])

    rows = []
    n_rows = max(len(df), 1)
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "feature": col,
                "mean": s.mean(),
                "std": s.std(ddof=0),
                "min": s.min(),
                "p10": s.quantile(0.10),
                "median": s.median(),
                "p90": s.quantile(0.90),
                "max": s.max(),
                "non_zero_pct": float((s != 0).sum()) * 100.0 / n_rows,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["std", "mean"], ascending=False).reset_index(drop=True)


def _plot_features(df: pd.DataFrame, plots_dir: Path) -> int:
    numeric_cols = [c for c in df.columns if c != "time" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return 0

    existing_pngs = set(p.stem for p in plots_dir.glob("*.png")) if plots_dir.exists() else set()
    if existing_pngs >= set(numeric_cols):
        print(f"  [skip] plots already generated ({len(existing_pngs)} PNGs)")
        return 0

    plots_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(df))
    retention = df["retention"].values if "retention" in df.columns else None
    count = 0

    for col in numeric_cols:
        out_path = plots_dir / f"{col}.png"
        if out_path.exists():
            continue

        values = df[col].values
        fig, ax1 = plt.subplots(figsize=(14, 4))

        if retention is not None and col != "retention":
            ax2 = ax1.twinx()
            ax2.fill_between(t, retention, alpha=0.08, color="#9E9E9E")
            ax2.plot(t, retention, color="#BDBDBD", linewidth=0.5, alpha=0.4)
            ax2.set_ylabel("retention (%)", color="#9E9E9E", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="#BDBDBD", labelsize=7)

        ax1.plot(t, values, color="#2196F3", linewidth=1.0)
        ax1.set(xlabel="sec", ylabel=col, title=col)
        ax1.grid(True, alpha=0.2)
        ax1.set_xlim(0, len(df) - 1)
        plt.tight_layout()
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        count += 1

    return count


def align_file(input_csv: Path, output_dir: Path, decimals: int, make_plots: bool = True) -> None:
    raw_df = pd.read_csv(input_csv)
    df = _normalize_time_column(raw_df)

    numeric_cols = [c for c in df.columns if c != "time" and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].round(decimals)

    readable_df = _reorder_columns(df)
    id_col = "time" if "time" in readable_df.columns else None
    if id_col:
        long_df = readable_df.melt(id_vars=[id_col], var_name="feature", value_name="value")
    else:
        long_df = readable_df.melt(var_name="feature", value_name="value")
    summary_df = _build_summary(readable_df)
    if not summary_df.empty:
        summary_df = summary_df.round(decimals)

    video_id = input_csv.stem.replace("_features", "")
    video_dir = output_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    readable_df.to_csv(video_dir / "features_readable.csv", index=False)
    long_df.to_csv(video_dir / "features_long.csv", index=False)
    summary_df.to_csv(video_dir / "features_summary.csv", index=False)

    if make_plots:
        raw_for_plot = pd.read_csv(input_csv)
        first_col = raw_for_plot.columns[0]
        if str(first_col).lower().startswith("unnamed"):
            raw_for_plot = raw_for_plot.drop(columns=[first_col])
        n_plots = _plot_features(raw_for_plot, video_dir / "plots")
        print(f"[ok] {input_csv.name} -> {video_dir} ({n_plots} plots)")
    else:
        print(f"[ok] {input_csv.name} -> {video_dir}")


def _collect_inputs(input_path: Path | None, input_dir: Path) -> list[Path]:
    if input_path:
        return [input_path]
    return sorted(input_dir.glob("*_features.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create human-readable aligned feature outputs with plots.")
    parser.add_argument("--input", type=str, default=None, help="Single *_features.csv file to align")
    parser.add_argument("--input-dir", type=str, default="output", help="Directory containing *_features.csv files")
    parser.add_argument("--output-dir", type=str, default="output_aligned", help="Directory for readable outputs")
    parser.add_argument("--decimals", type=int, default=4, help="Decimal precision for numeric values")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve() if args.input else None
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    inputs = _collect_inputs(input_path, input_dir)
    if not inputs:
        print(f"No input files found (input={args.input}, input-dir={input_dir})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in inputs:
        align_file(csv_path, output_dir, decimals=args.decimals, make_plots=not args.no_plots)

    print(f"Done. Aligned files saved to: {output_dir}")


if __name__ == "__main__":
    main()
