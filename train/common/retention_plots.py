from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

COLOR_ACTUAL = "#2196F3"
COLOR_PRED = "#FF5722"
COLOR_FILL = "#9C27B0"
COLOR_ERR_POS = "#4CAF50"
COLOR_ERR_NEG = "#F44336"
GRID_ALPHA = 0.3
PLOT_DPI = 150


def save_figure(fig: Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(train_losses: list[float], val_losses: list[float], out_path: str, title_suffix: str = "", ylabel: str = "loss") -> None:
    title = f"{title_suffix} Training Curve" if title_suffix else "Training Curve"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="train", color=COLOR_ACTUAL)
    ax.plot(val_losses, label="val", color=COLOR_PRED)
    ax.set(xlabel="epoch", ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_retention_prediction(video_id: str, y_true, y_pred, is_ad, split_name: str, metrics: dict, out_path: str) -> None:
    time_idx = np.arange(len(y_true))
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)
    ax_top.plot(time_idx, y_true, color=COLOR_ACTUAL, label="actual", linewidth=1.2)
    ax_top.plot(time_idx, y_pred, color=COLOR_PRED, label="predicted", alpha=0.8, linewidth=1.2)
    ax_top.fill_between(time_idx, y_true, y_pred, alpha=0.1, color=COLOR_FILL)
    if is_ad is not None and (ad_mask := is_ad > 0.5).any():
        ax_top.fill_between(time_idx, 0, 1, where=ad_mask, alpha=0.15, color="red", label="ad segment")
    ax_top.set(ylabel="Retention (%)", title=f"{video_id} [{split_name}]  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  r={metrics['pearson']:.3f}")
    ax_top.legend(fontsize=9)
    ax_top.grid(True, alpha=GRID_ALPHA)

    residual = y_pred - y_true
    ax_bottom.fill_between(time_idx, residual, alpha=0.3, color=COLOR_ERR_POS, where=residual >= 0)
    ax_bottom.fill_between(time_idx, residual, alpha=0.3, color=COLOR_ERR_NEG, where=residual < 0)
    ax_bottom.axhline(0, color="black", linewidth=0.5)
    ax_bottom.set(xlabel="sec", ylabel="error")
    ax_bottom.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()
    save_figure(fig, out_path)


plot_prediction = plot_retention_prediction
