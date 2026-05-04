"""
transformer tuning.
python tune_hp/tune_transformer.py --n_trials 50 --max_epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.retention_transformer import RetentionTransformer
from src.datasets import RetentionWindowDataset
from tune_hp.utils import CUDA_DEVICE, create_study, load_tuning_data, mean_loo_score, optimize_study, print_best_trial, save_and_report


def _train_one_fold(
    trial: optuna.Trial,
    params: dict,
    video_frames: dict,
    feature_cols: list[str],
    train_ids: list[str],
    val_id: str,
    max_epochs: int,
    device: torch.device,
    fold_idx: int,
) -> float:
    window_size = params["window_size"]
    stride = params["window_stride"]
    batch_size = params["batch_size"]

    train_ds = RetentionWindowDataset(video_frames, train_ids, feature_cols, window_size=window_size, stride=stride)
    val_ds = RetentionWindowDataset(video_frames, [val_id], feature_cols, window_size=window_size, stride=stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = RetentionTransformer(
        n_features=len(feature_cols), d_model=params["d_model"], n_heads=params["n_heads"], n_layers=params["n_layers"], d_ff=params["d_ff"], dropout=params["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.SmoothL1Loss(reduction="none")

    best_val_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            features = batch["features"].to(device)
            target = batch["retention"].to(device)
            mask = batch["padding_mask"].to(device)
            prediction = model(features, src_key_padding_mask=mask)
            loss_by_second = criterion(prediction, target)
            loss_by_second[mask] = 0.0
            n_valid = (~mask).sum().clamp(min=1)
            loss = loss_by_second.sum() / n_valid
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                target = batch["retention"].to(device)
                mask = batch["padding_mask"].to(device)
                prediction = model(features, src_key_padding_mask=mask)
                loss_by_second = criterion(prediction, target)
                loss_by_second[mask] = 0.0
                n_valid = (~mask).sum().clamp(min=1)
                val_loss_sum += loss_by_second.sum().item()
                val_count += n_valid.item()

        val_smooth_l1 = val_loss_sum / max(val_count, 1)
        best_val_loss = min(best_val_loss, val_smooth_l1)

        global_step = fold_idx * max_epochs + epoch
        trial.report(val_smooth_l1, global_step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


def objective(trial: optuna.Trial, video_frames: dict, feature_cols: list[str], max_epochs: int, device: torch.device) -> float:
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    valid_heads = [head_count for head_count in [2, 4, 8] if d_model % head_count == 0]
    n_heads = trial.suggest_categorical("n_heads", valid_heads)

    window_size = trial.suggest_categorical("window_size", [64, 128, 256])
    max_stride = window_size
    stride_choices = [stride for stride in [32, 64, 128] if stride <= max_stride]
    window_stride = trial.suggest_categorical("window_stride", stride_choices)

    params = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4, step=0.05),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "window_size": window_size,
        "window_stride": window_stride,
    }

    return mean_loo_score(video_frames, lambda train_ids, val_id, fold_idx: _train_one_fold(trial, params, video_frames, feature_cols, train_ids, val_id, max_epochs, device, fold_idx))


def run(features_dir: str, n_trials: int, max_epochs: int, resume: bool):
    video_frames, feature_cols = load_tuning_data(features_dir)
    device = torch.device(CUDA_DEVICE)
    print(f"device={device}")
    study = create_study("transformer", "transformer_retention", resume=resume, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    elapsed = optimize_study(study, lambda trial: objective(trial, video_frames, feature_cols, max_epochs, device), n_trials)
    print_best_trial(study, "LOO SmoothL1", elapsed=elapsed)
    save_and_report("transformer", study.best_params, {"loo_smooth_l1": round(study.best_value, 6)})


def main():
    parser = argparse.ArgumentParser(description="Tune Transformer HPs with Optuna")
    parser.add_argument("--features_dir", default="output")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=50, help="Max epochs per trial fold")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.features_dir, args.n_trials, args.max_epochs, args.resume)


if __name__ == "__main__":
    main()
