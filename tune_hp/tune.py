"""
unified Optuna tuning.
python tune_hp/tune.py --arch multimodal_transformer --n-trials 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.retention_lstm import RetentionLSTM
from src.models.retention_multimodal_lstm import MultimodalRetentionLSTM
from src.models.retention_multimodal_transformer import MultimodalRetentionTransformer
from src.models.retention_transformer import RetentionTransformer
from train.common.seq_data_utils import *
from tune_hp.utils import CUDA_DEVICE, create_study, optimize_study, print_best_trial, print_lines


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCH_CHOICES = ["transformer", "lstm", "multimodal_transformer", "multimodal_lstm"]
MODEL_SPECS: dict[str, tuple[bool, Callable[[int, dict], nn.Module]]] = {
    "transformer": (False, lambda n_features, p: RetentionTransformer(n_features, p["d_model"], p["n_heads"], p["n_layers"], p["d_ff"], p["dropout"])),
    "lstm": (False, lambda n_features, p: RetentionLSTM(n_features, p["hidden_size"], p["n_layers"], p["dropout"])),
    "multimodal_transformer": (True, lambda n_features, p: MultimodalRetentionTransformer(p["d_model"], p["n_heads"], p["n_layers"], p["d_ff"], p["dropout"], n_tabular_features=n_features)),
    "multimodal_lstm": (True, lambda n_features, p: MultimodalRetentionLSTM(p["hidden_size"], p["n_layers"], p["dropout"], n_tabular_features=n_features)),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="multimodal_transformer", choices=ARCH_CHOICES)
    parser.add_argument("--output-dir-features", default="output")
    parser.add_argument("--snapshot-dir", default="data")
    parser.add_argument("--embeddings-root", default="embeddings")
    parser.add_argument("--use-curve-raw", action="store_true", default=True)
    parser.add_argument("--no-use-curve-raw", dest="use_curve_raw", action="store_false")
    parser.add_argument("--val-first-n-output", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--epochs-per-trial", type=int, default=150)
    parser.add_argument("--device", default=CUDA_DEVICE)
    parser.add_argument("--output-dir", default="tune_hp/results")
    parser.add_argument("--study-name", default="")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def _build_model(arch: str, n_features: int, trial: optuna.Trial, device: torch.device):
    is_multimodal, factory = MODEL_SPECS[arch]
    params = {"n_layers": trial.suggest_int("n_layers", 2, 6) if "transformer" in arch else trial.suggest_int("n_layers", 1, 3), "dropout": trial.suggest_float("dropout", 0.1, 0.4)}
    if "transformer" in arch:
        params |= {
            "d_model": trial.suggest_categorical("d_model", [64, 96, 128, 192]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
            "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512]),
        }
    else:
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 96, 128, 192])
    return factory(n_features, params).to(device), is_multimodal


def _batch_spike_triggers(batch, ad_mask: torch.Tensor) -> torch.Tensor:
    spike_triggers = batch.get("spike_triggers")
    if spike_triggers is None:
        return torch.zeros_like(ad_mask)
    return spike_triggers.to(device=ad_mask.device, dtype=ad_mask.dtype)


def _forward_batch(model, batch, device, is_multimodal):
    target = batch["retention"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    ad_mask = batch["is_ad"].to(device)
    spike_triggers = _batch_spike_triggers(batch, ad_mask)
    prediction = model(batch["embeddings"].to(device), tabular=batch["tabular"].to(device), src_key_padding_mask=padding_mask) if is_multimodal else model(batch["features"].to(device), src_key_padding_mask=padding_mask)
    return prediction, target, ad_mask, spike_triggers, padding_mask


def _train_eval(trial: optuna.Trial, model, train_loader, val_loader, device, hyperparams, epochs, is_multimodal):
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=hyperparams["lr"] * 0.01)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            prediction, target, ad_mask, spike_triggers, padding_mask = _forward_batch(model, batch, device, is_multimodal)
            loss = composite_loss(
                prediction,
                target,
                ad_mask,
                spike_triggers,
                padding_mask,
                hyperparams["ad_penalty"],
                None,
                hyperparams["alpha_corr"],
                hyperparams["alpha_smooth"],
                hyperparams["alpha_mono"],
                hyperparams["start_boost_secs"],
                hyperparams["start_boost_factor"],
                hyperparams["alpha_delta"],
                1.0,
                0.6,
                hyperparams["spike_penalty"],
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                prediction, target, ad_mask, spike_triggers, padding_mask = _forward_batch(model, batch, device, is_multimodal)
                batch_loss = composite_loss(
                    prediction, target, ad_mask, spike_triggers, padding_mask, 1.0, None, hyperparams["alpha_corr"], 0.0, 0.0, 0, 1.0, hyperparams["alpha_delta"], 1.0, 0.6, hyperparams["spike_penalty"]
                )
                valid_count = (~padding_mask).sum().item()
                val_loss += batch_loss.item() * valid_count
                val_count += valid_count

        mean_val_loss = val_loss / max(val_count, 1)
        best_val = min(best_val, mean_val_loss)
        trial.report(mean_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val


def _make_datasets(video_dfs, video_embeddings, train_ids, val_ids, feature_cols, normalizer, window_size, window_stride, feature_mask_prob, noise_std, is_multimodal):
    dataset_cls = MultimodalWindowedDataset if is_multimodal else WindowedSeqDataset
    base_args = (video_dfs, video_embeddings, train_ids, feature_cols, normalizer, window_size, window_stride) if is_multimodal else (video_dfs, train_ids, feature_cols, normalizer, window_size, window_stride)
    val_args = (video_dfs, video_embeddings, val_ids, feature_cols, normalizer, window_size, window_stride) if is_multimodal else (video_dfs, val_ids, feature_cols, normalizer, window_size, window_stride)
    return dataset_cls(*base_args, feature_mask_prob=feature_mask_prob, noise_std=noise_std), dataset_cls(*val_args)


def objective(trial: optuna.Trial, args, video_dfs, video_embeddings, train_ids, val_ids, device):
    corr_threshold = trial.suggest_float("redundant_corr_threshold", 0.75, 0.95)
    max_nan_pct = trial.suggest_float("max_nan_pct", 0.15, 0.60)
    min_nonzero_pct = trial.suggest_float("min_nonzero_pct", 0.001, 0.05, log=True)
    top_k = trial.suggest_categorical("top_k_features", [0, 40, 60, 80, 100, 140])
    top_k = top_k if top_k > 0 else None

    feature_cols, _ = filter_features(video_dfs, redundant_corr_threshold=corr_threshold, max_nan_pct=max_nan_pct, min_nonzero_pct=min_nonzero_pct, top_k=top_k)
    n_features = len(feature_cols)
    model, is_multimodal = _build_model(args.arch, n_features, trial, device)

    hyperparams = {
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
        "alpha_corr": trial.suggest_float("alpha_corr", 0.1, 0.5),
        "alpha_smooth": trial.suggest_float("alpha_smooth", 0.1, 0.6),
        "alpha_delta": trial.suggest_float("alpha_delta", 0.1, 0.8),
        "alpha_mono": trial.suggest_float("alpha_mono", 0.0, 0.1),
        "ad_penalty": trial.suggest_float("ad_penalty", 1.0, 25.0),
        "spike_penalty": trial.suggest_float("spike_penalty_weight", 2.0, 25.0),
        "start_boost_secs": trial.suggest_int("start_boost_secs", 5, 30),
        "start_boost_factor": trial.suggest_float("start_boost_factor", 1.0, 4.0),
    }

    window_size = trial.suggest_categorical("window_size", [64, 96, 128, 192])
    window_stride = window_size // 2
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    feature_mask_prob = trial.suggest_float("feature_mask_prob", 0.0, 0.2)
    noise_std = trial.suggest_float("noise_std", 0.0, 0.05)

    normalizer = FeatureNormalizer()
    normalizer.fit({video_id: video_dfs[video_id] for video_id in train_ids}, feature_cols)

    train_ds, val_ds = _make_datasets(video_dfs, video_embeddings, train_ids, val_ids, feature_cols, normalizer, window_size, window_stride, feature_mask_prob, noise_std, is_multimodal)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = _train_eval(trial, model, train_loader, val_loader, device, hyperparams, args.epochs_per_trial, is_multimodal)

    return best_val


def main():
    args = parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.random_seed)
    torch.manual_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    is_multimodal = "multimodal" in args.arch
    embedding_pca_components = 12 if not is_multimodal else 0

    logger.info("Loading data for %s tuning...", args.arch)
    video_dfs = load_all_merged(args.output_dir_features, args.snapshot_dir, use_curve_raw=args.use_curve_raw, emb_pca_components=embedding_pca_components)
    video_ids = sorted(video_dfs.keys())

    video_embeddings = {}
    if is_multimodal:
        video_embeddings = load_aligned_embeddings_for_videos(video_dfs, args.embeddings_root)

    output_dir = Path(args.output_dir_features)
    output_video_ids = sorted(
        feature_path.name.replace("_features.csv", "") for feature_path in output_dir.glob("*_features.csv") if not feature_path.name.endswith(".partial") and feature_path.name.replace("_features.csv", "") in video_dfs
    )

    if args.val_first_n_output > 0:
        val_ids = output_video_ids[:min(args.val_first_n_output, len(output_video_ids))]
        train_ids = [video_id for video_id in video_ids if video_id not in set(val_ids)]
    else:
        rng.shuffle(video_ids)
        val_ids, train_ids = video_ids[:max(1, int(len(video_ids) * 0.15))], video_ids[max(1, int(len(video_ids) * 0.15)):]

    logger.info("Train: %d videos, Val: %d videos", len(train_ids), len(val_ids))

    study_name = args.study_name or f"tune_{args.arch}"
    study = create_study(
        study_name,
        study_name,
        resume=True,
        sampler=optuna.samplers.TPESampler(seed=args.random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )

    logger.info("Starting Optuna study: %s (%d trials)", study_name, args.n_trials)
    optimize_study(study, lambda trial: objective(trial, args, video_dfs, video_embeddings, train_ids, val_ids, device), args.n_trials)

    best = study.best_params
    logger.info("Best params (val_loss=%.4f):", study.best_value)
    for param_name, param_value in sorted(best.items()):
        logger.info("  %s: %s", param_name, param_value)

    best_feature_cols, _ = filter_features(
        video_dfs,
        redundant_corr_threshold=best.get("redundant_corr_threshold", 0.85),
        max_nan_pct=best.get("max_nan_pct", 0.50),
        min_nonzero_pct=best.get("min_nonzero_pct", 0.01),
        top_k=best.get("top_k_features") if best.get("top_k_features", 0) > 0 else None,
    )

    results = {
        "arch": args.arch,
        "best_val_loss": study.best_value,
        "best_params": best,
        "n_trials": len(study.trials),
        "best_n_features": len(best_feature_cols),
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    output_path = os.path.join(args.output_dir, f"{study_name}_best.json")
    Path(output_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results saved to %s", output_path)
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(os.path.join(args.output_dir, f"{study_name}_param_importance.html"))
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(os.path.join(args.output_dir, f"{study_name}_history.html"))

    print_best_trial(study, "Best val loss")
    print_lines(f"Results: {output_path}", "")


if __name__ == "__main__":
    main()
