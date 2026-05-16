from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.common.seq_data_utils import composite_loss

logger = logging.getLogger(__name__)

BatchForward = Callable[[nn.Module, dict, torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]]


def to_device_batch(batch: dict, device: torch.device, *keys: str):
    return tuple(batch[key].to(device) for key in keys)


def lr_warmup_cosine(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))


def _maybe_writer(output_dir: str):
    log_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=log_dir)
    except (ImportError, ModuleNotFoundError):
        return None


def _state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.cpu().clone() for k, v in module.state_dict().items()}


def _loss_from_forward(forward_out, args: Any, train: bool):
    pred, target, ad_mask, spike_triggers, padding_mask, video_weight = forward_out
    if train:
        return composite_loss(
            pred,
            target,
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
    return composite_loss(pred, target, ad_mask, spike_triggers, padding_mask, 1.0, None, args.alpha_corr, 0.0, 0.0, 0, 1.0, args.alpha_delta)


def run_composite_training_loop(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    device: torch.device,
    args: Any,
    forward_batch: BatchForward,
    *,
    enable_swa: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: lr_warmup_cosine(ep, args.warmup_epochs, args.epochs))
    swa_start = args.swa_start_epoch if getattr(args, "swa_start_epoch", 0) > 0 else int(args.epochs * 0.7)
    swa_lr = getattr(args, "swa_lr", args.lr)
    swa_model = AveragedModel(model) if enable_swa else None
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr) if enable_swa else None
    swa_active = False
    writer = _maybe_writer(args.output_dir)

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
            forward_out = forward_batch(model, batch, device)
            padding_mask = forward_out[4]
            loss = _loss_from_forward(forward_out, args, train=True)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n_valid = (~padding_mask).sum().item()
            train_loss_sum += loss.item() * n_valid
            train_valid_count += n_valid

        if enable_swa and epoch >= swa_start:
            assert swa_model is not None and swa_scheduler is not None
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_active = True
        else:
            scheduler.step()

        train_losses.append(train_loss_sum / max(train_valid_count, 1))
        eval_model = swa_model if swa_active and swa_model is not None else model
        eval_model.eval()
        val_loss_sum = 0.0
        val_valid_count = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Ep {epoch} [val]", leave=False, unit="b"):
                forward_out = forward_batch(eval_model, batch, device)
                padding_mask = forward_out[4]
                loss = _loss_from_forward(forward_out, args, train=False)
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
            logger.info("Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e%s", epoch, args.epochs, train_losses[-1], val_losses[-1], optimizer.param_groups[0]["lr"], " [SWA]" if swa_active else "")

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_without_improve = 0
            best_state = _state_dict_cpu(swa_model if swa_active and swa_model is not None else model)
            best_state_owner = "swa" if swa_active else "model"
        else:
            epochs_without_improve += 1
            if not swa_active and epochs_without_improve >= args.patience:
                logger.info("Early stop at epoch %d", epoch)
                break

    if writer:
        writer.close()

    if best_state_owner == "swa" and swa_model is not None:
        swa_model.load_state_dict(best_state)
        if any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in swa_model.modules()):
            torch.optim.swa_utils.update_bn(train_dl, swa_model, device=device)
        model = copy.deepcopy(swa_model.module)
    else:
        model.load_state_dict(best_state)

    result = {"train_losses": train_losses, "val_losses": val_losses, "best_val_loss": round(best_val_loss, 6), "epochs_trained": epoch, "elapsed_sec": round(time.time() - t0, 1)}
    return model, result
