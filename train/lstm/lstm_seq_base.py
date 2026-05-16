from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train.common.composite_trainer import lr_warmup_cosine, run_composite_training_loop, to_device_batch
from train.common.retention_plots import COLOR_ACTUAL, COLOR_ERR_POS, GRID_ALPHA, plot_retention_prediction, plot_training_curve, save_figure
from train.common.split_utils import apply_train_id_file_filter, resolve_train_val_split


logger = logging.getLogger(__name__)
__all__ = [
    "COLOR_ACTUAL",
    "COLOR_ERR_POS",
    "GRID_ALPHA",
    "apply_train_id_file_filter",
    "lr_warmup_cosine",
    "plot_retention_prediction",
    "plot_training_curve",
    "resolve_train_val_split",
    "run_sequence_training_loop",
    "save_figure",
    "to_device_batch",
]


def run_sequence_training_loop(
    model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, device: torch.device, args: Any, use_engagement_weight: bool = True
) -> tuple[nn.Module, dict[str, Any]]:
    def _forward(batch_model, batch, batch_device):
        features, targets, padding_mask, ad_mask = to_device_batch(batch, batch_device, "features", "retention", "padding_mask", "is_ad")
        spike_triggers = batch["spike_triggers"].to(batch_device)
        video_weight = batch["video_weight"].to(batch_device) if use_engagement_weight else None
        return batch_model(features, src_key_padding_mask=padding_mask), targets, ad_mask, spike_triggers, padding_mask, video_weight

    return run_composite_training_loop(model, train_dl, val_dl, device, args, _forward, enable_swa=True)
