from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _dark(fig, title: str, height: int = 700):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", size=13, color="#e6edf3"),
        title=dict(text=title, font=dict(size=18), x=0.5),
        height=height,
        margin=dict(l=40, r=20, t=60, b=40),
    )


def plot_mae_comparison(mae_df: pd.DataFrame, out: Path) -> None:
    summary = mae_df.groupby("curve_type")[["mae_fitted", "mae_predicted"]].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=summary["curve_type"], y=summary["mae_fitted"], name="Fitted (curve_fit)", marker_color="#2196F3"))
    fig.add_trace(go.Bar(x=summary["curve_type"], y=summary["mae_predicted"], name="Predicted (model)", marker_color="#FF5722"))
    fig.update_layout(barmode="group", yaxis_title="MAE (%)")
    _dark(fig, "MAE: Fitted vs Predicted Curves")
    fig.write_html(str(out / "mae_comparison.html"), include_plotlyjs="cdn")


def plot_per_video_mae(mae_df: pd.DataFrame, out: Path) -> None:
    fig = px.strip(mae_df, x="curve_type", y="mae_predicted", hover_data=["video_id"], color="curve_type", color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    _dark(fig, "Predicted Curve MAE per Video")
    fig.write_html(str(out / "mae_per_video.html"), include_plotlyjs="cdn")


def plot_examples(examples: list[dict], out: Path) -> None:
    if not examples:
        return
    n = len(examples)
    fig = make_subplots(rows=n, cols=1, subplot_titles=[f"{e['video_id']} ({e['curve_type']})" for e in examples], vertical_spacing=max(0.02, 0.25 / n))
    for i, ex in enumerate(examples, 1):
        t = np.arange(len(ex["actual"]))
        show = i == 1
        fig.add_trace(go.Scatter(x=t, y=ex["actual"], name="actual", line=dict(color="#2196F3", width=1.5), showlegend=show), row=i, col=1)
        fig.add_trace(go.Scatter(x=t, y=ex["fitted"], name="fitted", line=dict(color="#4CAF50", width=1.5, dash="dash"), showlegend=show), row=i, col=1)
        fig.add_trace(go.Scatter(x=t, y=ex["predicted"], name="predicted", line=dict(color="#FF5722", width=1.5, dash="dot"), showlegend=show), row=i, col=1)
    _dark(fig, "Example Curves: Actual vs Fitted vs Predicted", height=250 * n + 100)
    fig.write_html(str(out / "example_curves.html"), include_plotlyjs="cdn")


def plot_summary_table(mae_df: pd.DataFrame, out: Path) -> None:
    summary = (
        mae_df.groupby("curve_type")
        .agg(
            n=("video_id", "count"),
            mae_fit_mean=("mae_fitted", "mean"),
            mae_fit_std=("mae_fitted", "std"),
            mae_pred_mean=("mae_predicted", "mean"),
            mae_pred_std=("mae_predicted", "std"),
        )
        .reset_index()
    )
    fig = go.Figure(
        go.Table(
            header=dict(
                values=["Curve", "N", "MAE fit (mean)", "MAE fit (std)", "MAE pred (mean)", "MAE pred (std)"],
                fill_color="#161b22",
                font=dict(color="#e6edf3", size=12),
                align="center",
                line_color="#30363d",
            ),
            cells=dict(
                values=[
                    summary["curve_type"],
                    summary["n"],
                    summary["mae_fit_mean"].round(3),
                    summary["mae_fit_std"].round(3),
                    summary["mae_pred_mean"].round(3),
                    summary["mae_pred_std"].round(3),
                ],
                fill_color="#0d1117",
                font=dict(color="#e6edf3", size=11),
                align="center",
                line_color="#21262d",
            ),
        )
    )
    _dark(fig, "Prediction Quality Summary", height=250)
    fig.write_html(str(out / "summary_table.html"), include_plotlyjs="cdn")
