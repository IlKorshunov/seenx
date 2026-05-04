from __future__ import annotations

import copy
import logging
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.common.seq_data_utils import composite_loss


logger = logging.getLogger(__name__)

COLOR_ACTUAL = "#2196F3"
COLOR_PRED = "#FF5722"
COLOR_FILL = "#9C27B0"
COLOR_ERR_POS = "#4CAF50"
COLOR_ERR_NEG = "#F44336"
GRID_ALPHA = 0.3
PLOT_DPI = 150


def to_device_batch(batch: dict, device: torch.device, *keys: str):
    return tuple(batch[key].to(device) for key in keys)


def lr_warmup_cosine(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return 0.5 * (1 + np.cos(np.pi * progress))


def run_sequence_training_loop(
    model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, device: torch.device, args: Any, use_engagement_weight: bool = True
) -> tuple[nn.Module, dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: lr_warmup_cosine(ep, args.warmup_epochs, args.epochs))
    swa_start = args.swa_start_epoch if args.swa_start_epoch > 0 else int(args.epochs * 0.7)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
    swa_active = False

    log_dir = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=log_dir)
    except (ImportError, ModuleNotFoundError):
        writer = None

    best_val_loss = float("inf")
    epochs_without_improve = 0
    best_state: dict[str, torch.Tensor] = {}
    best_state_owner = "model"
    train_losses: list[float] = []
    val_losses: list[float] = []
    t0 = time.time()
    epoch = 0

    for epoch in (epoch_bar := tqdm(range(1, args.epochs + 1), desc="Training", unit="ep")):
        model.train()
        train_loss_sum = 0.0
        train_valid_count = 0
        for batch in tqdm(train_dl, desc=f"Ep {epoch} [train]", leave=False, unit="b"):
            features, targets, padding_mask, ad_mask = to_device_batch(batch, device, "features", "retention", "padding_mask", "is_ad")
            spike_triggers = batch["spike_triggers"].to(device)
            video_weight = batch["video_weight"].to(device) if use_engagement_weight else None
            loss = composite_loss(
                model(features, src_key_padding_mask=padding_mask),
                targets,
                ad_mask,
                spike_triggers,
                padding_mask,
                args.ad_penalty_weight,
                video_weight,
                args.alpha_corr,
                args.alpha_smooth,
                args.alpha_mono,
                args.start_boost_secs,
                args.start_boost_factor,
                args.alpha_delta,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n_valid = (~padding_mask).sum().item()
            train_loss_sum += loss.item() * n_valid
            train_valid_count += n_valid
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_active = True
        else:
            scheduler.step()

        train_losses.append(train_loss_sum / max(train_valid_count, 1))

        eval_model = swa_model if swa_active else model
        eval_model.eval()
        val_loss_sum = 0.0
        val_valid_count = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Ep {epoch} [val]", leave=False, unit="b"):
                features, targets, padding_mask, ad_mask = to_device_batch(batch, device, "features", "retention", "padding_mask", "is_ad")
                spike_triggers = batch["spike_triggers"].to(device)
                loss = composite_loss(
                    eval_model(features, src_key_padding_mask=padding_mask),
                    targets,
                    ad_mask,
                    spike_triggers,
                    padding_mask,
                    1.0,
                    None,
                    args.alpha_corr,
                    0.0,
                    0.0,
                    0,
                    1.0,
                    args.alpha_delta,
                )
                n_valid = (~padding_mask).sum().item()
                val_loss_sum += loss.item() * n_valid
                val_valid_count += n_valid
        val_losses.append(val_loss_sum / max(val_valid_count, 1))

        epoch_bar.set_postfix(train=f"{train_losses[-1]:.4f}", val=f"{val_losses[-1]:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}", swa="on" if swa_active else "off")
        if writer:
            writer.add_scalar("Loss/train", train_losses[-1], epoch)
            writer.add_scalar("Loss/val", val_losses[-1], epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e%s",
                epoch,
                args.epochs,
                train_losses[-1],
                val_losses[-1],
                optimizer.param_groups[0]["lr"],
                " [SWA]" if swa_active else "",
            )

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_without_improve = 0
            if swa_active:
                best_state = {k: v.cpu().clone() for k, v in swa_model.state_dict().items()}
                best_state_owner = "swa"
            else:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_state_owner = "model"
        else:
            epochs_without_improve += 1
            if not swa_active and epochs_without_improve >= args.patience:
                logger.info("Early stop at epoch %d", epoch)
                break

    if writer:
        writer.close()

    if best_state_owner == "swa":
        swa_model.load_state_dict(best_state)
        has_batchnorm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in swa_model.modules())
        if has_batchnorm:
            torch.optim.swa_utils.update_bn(train_dl, swa_model, device=device)
        model = copy.deepcopy(swa_model.module)
    else:
        model.load_state_dict(best_state)

    result = {"train_losses": train_losses, "val_losses": val_losses, "best_val_loss": round(best_val_loss, 6), "epochs_trained": epoch, "elapsed_sec": round(time.time() - t0, 1)}
    return model, result


def save_figure(fig: Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(train_losses: list[float], val_losses: list[float], out_path: str, title_suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="train", color=COLOR_ACTUAL)
    ax.plot(val_losses, label="val", color=COLOR_PRED)
    ax.set(xlabel="epoch", ylabel="composite loss", title=f"{title_suffix} Training Curve")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()
    save_figure(fig, out_path)
    logger.info("Saved %s", out_path)


def plot_retention_prediction(video_id: str, y_true: np.ndarray, y_pred: np.ndarray, is_ad: np.ndarray | None, split_name: str, metrics: dict[str, float], out_path: str) -> None:
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


def resolve_train_val_split(args: Any, video_ids: list[str], output_video_ids: list[str]) -> tuple[list[str], list[str]]:
    if args.eval_video and args.eval_video in video_ids:
        return [v for v in video_ids if v != args.eval_video], [args.eval_video]
    if args.val_first_n_output > 0:
        n_val = min(args.val_first_n_output, len(output_video_ids))
        val_ids = output_video_ids[:n_val]
        val_set = set(val_ids)
        logger.info("Validation split: first %d videos from output", n_val)
        return [v for v in video_ids if v not in val_set], val_ids
    shuffled = list(video_ids)
    np.random.RandomState(args.random_seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def apply_train_id_file_filter(train_ids: list[str], args: Any) -> list[str]:
    path = getattr(args, "train_video_ids_file", "") or ""
    if not path:
        return train_ids
    allow = {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}
    before = len(train_ids)
    filtered = [v for v in train_ids if v in allow]
    logger.info("Train subset from file: %d -> %d videos (%s)", before, len(filtered), path)
    return filtered
