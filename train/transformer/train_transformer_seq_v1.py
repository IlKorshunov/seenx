"""
Transformer-based per-second retention predictor.

Usage:
    python train/train_transformer_seq.py --eval-video DhFuAhFMvms
    python -m train.transformer.train_transformer_seq_v1 --top-k-features 40 --epochs 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.retention_transformer import RetentionTransformer
from train.common.seq_data_utils import FeatureNormalizer, WindowedSeqDataset, ad_aware_loss, filter_features, load_all_merged, load_video_weights, predict_video, seq_metrics
from train.common.retention_plots import COLOR_ACTUAL as C_BLUE, GRID_ALPHA, plot_prediction, plot_training_curve, save_figure as _save_fig
from train.common.split_utils import resolve_train_val_split


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RetentionTransformer on merged features.")
    add_arg = parser.add_argument
    add_arg("--output-dir-features", default="output")
    add_arg("--snapshot-dir", default="data")
    add_arg("--use-curve-raw", action="store_true", default=True)
    add_arg("--no-use-curve-raw", dest="use_curve_raw", action="store_false")
    add_arg("--output-dir", default="transformer_seq_experiment")
    add_arg("--eval-video", default="")
    add_arg("--val-ratio", type=float, default=0.15)
    add_arg("--val-first-n-output", type=int, default=0, help="Use first N videos from output dir as validation (sorted by video id).")
    add_arg("--top-k-features", type=int, default=0)
    add_arg("--ad-penalty-weight", type=float, default=15.0)
    add_arg("--engagement-weight", action="store_true", default=True, help="Weight loss by video engagement (views/likes/comments) from meta.json.")
    add_arg("--no-engagement-weight", dest="engagement_weight", action="store_false")
    add_arg("--window-size", type=int, default=128)
    add_arg("--window-stride", type=int, default=64)
    add_arg("--d-model", type=int, default=128)
    add_arg("--n-heads", type=int, default=4)
    add_arg("--n-layers", type=int, default=4)
    add_arg("--d-ff", type=int, default=256)
    add_arg("--dropout", type=float, default=0.2)
    add_arg("--epochs", type=int, default=200)
    add_arg("--batch-size", type=int, default=16)
    add_arg("--lr", type=float, default=1e-3)
    add_arg("--weight-decay", type=float, default=1e-4)
    add_arg("--patience", type=int, default=25)
    add_arg("--grad-clip", type=float, default=1.0)
    add_arg("--random-seed", type=int, default=42)
    add_arg("--device", default="cpu")
    add_arg("--apply-smoothing", action="store_true", default=False, help="Apply Savitzky-Golay filter to predictions")
    return parser.parse_args()


def _to_device(batch, device, *tensor_keys):
    return tuple(batch[key].to(device) for key in tensor_keys)


def train_model(model, train_dl, val_dl, device, args, use_engagement_weight: bool = True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    pointwise_loss_fn = nn.MSELoss(reduction="none")
    log_dir = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss, epochs_without_improvement, best_state_dict = float("inf"), 0, {}
    train_losses, val_losses = [], []
    train_start_time = time.time()
    epoch = 0

    for epoch in (epoch_progress := tqdm(range(1, args.epochs + 1), desc="Training", unit="ep")):
        model.train()
        train_loss_sum, train_valid_points = 0.0, 0
        for batch in (train_progress := tqdm(train_dl, desc=f"Ep {epoch} [train]", leave=False, unit="b")):
            features_batch, targets_batch, padding_mask, ad_mask = _to_device(batch, device, "features", "retention", "padding_mask", "is_ad")
            vw = batch["video_weight"].to(device) if use_engagement_weight else None
            loss = ad_aware_loss(
                model(features_batch, src_key_padding_mask=padding_mask), targets_batch, ad_mask, padding_mask, pointwise_loss_fn, args.ad_penalty_weight, video_weight=vw
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            valid_points = (~padding_mask).sum().item()
            train_loss_sum += loss.item() * valid_points
            train_valid_points += valid_points
            train_progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_losses.append(train_loss_sum / max(train_valid_points, 1))

        model.eval()
        val_loss_sum, val_valid_points = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Ep {epoch} [val]", leave=False, unit="b"):
                features_batch, targets_batch, padding_mask, ad_mask = _to_device(batch, device, "features", "retention", "padding_mask", "is_ad")
                loss = ad_aware_loss(model(features_batch, src_key_padding_mask=padding_mask), targets_batch, ad_mask, padding_mask, pointwise_loss_fn, 1.0)
                valid_points = (~padding_mask).sum().item()
                val_loss_sum += loss.item() * valid_points
                val_valid_points += valid_points
        val_losses.append(val_loss_sum / max(val_valid_points, 1))

        epoch_progress.set_postfix(train=f"{train_losses[-1]:.4f}", val=f"{val_losses[-1]:.4f}")
        writer.add_scalar("MAE/train", train_losses[-1], epoch)
        writer.add_scalar("MAE/val", val_losses[-1], epoch)
        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d  train=%.4f  val=%.4f", epoch, args.epochs, train_losses[-1], val_losses[-1])

        if val_losses[-1] < best_val_loss:
            best_val_loss, epochs_without_improvement = val_losses[-1], 0
            best_state_dict = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stop at epoch %d", epoch)
                break

    if writer:
        writer.close()
    model.load_state_dict(best_state_dict)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_mae": round(best_val_loss, 6),
        "epochs_trained": epoch,
        "elapsed_sec": round(time.time() - train_start_time, 1),
    }


def compute_feature_importance(model, feature_cols, video_dfs, val_ids, normalizer, device, out_dir, window_size, n_repeats=5):
    """Permutation importance: shuffle each feature, measure MAE increase over full videos."""
    model.eval()

    baseline_mae_per_video = {}
    for video_id in val_ids:
        y_true, y_pred = predict_video(model, video_dfs[video_id], feature_cols, normalizer, device, window_size)
        baseline_mae_per_video[video_id] = float(np.mean(np.abs(y_pred - y_true)))
    baseline_mae = np.mean(list(baseline_mae_per_video.values()))

    importance = np.zeros(len(feature_cols))
    rng = np.random.RandomState(42)  # pylint: disable=no-member

    for feat_idx, feat_name in enumerate(tqdm(feature_cols, desc="Permutation importance")):
        mae_increases = []
        for _ in range(n_repeats):
            shuffled_maes = []
            for video_id in val_ids:
                shuffled_df = video_dfs[video_id].copy()
                shuffled_df[feat_name] = rng.permutation(shuffled_df[feat_name].values)
                y_true, y_pred = predict_video(model, shuffled_df, feature_cols, normalizer, device, window_size)
                shuffled_maes.append(float(np.mean(np.abs(y_pred - y_true))))
            mae_increases.append(np.mean(shuffled_maes) - baseline_mae)
        importance[feat_idx] = np.mean(mae_increases)

    ranking = sorted(zip(feature_cols, importance, strict=True), key=lambda x: -float(x[1]))
    pd.DataFrame(ranking, columns=["feature", "importance_mae_increase"]).to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    top_n = min(30, len(ranking))
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    top_feature_names, top_feature_scores = zip(*ranking[:top_n], strict=True)
    ax.barh(top_feature_names[::-1], top_feature_scores[::-1], color=C_BLUE)
    ax.set(xlabel="MAE increase when shuffled", title=f"Permutation Feature Importance (top {top_n})")
    ax.grid(True, alpha=GRID_ALPHA, axis="x")
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "feature_importance.png"))
    logger.info("Saved feature importance (%d features, baseline MAE=%.4f)", len(ranking), baseline_mae)


def main():
    args = parse_args()
    device = torch.device(args.device)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading merged data...")
    video_dfs = load_all_merged(args.output_dir_features, args.snapshot_dir, use_curve_raw=args.use_curve_raw)
    video_ids = sorted(video_dfs.keys())
    logger.info("Videos: %s", video_ids)
    output_dir = Path(args.output_dir_features)
    output_video_ids = sorted(
        p.name.replace("_features.csv", "") for p in output_dir.glob("*_features.csv") if not p.name.endswith(".partial") and p.name.replace("_features.csv", "") in video_dfs
    )

    top_k = args.top_k_features or None
    feature_cols, filter_log = filter_features(video_dfs, top_k=top_k)
    Path(os.path.join(args.output_dir, "feature_filter_log.txt")).write_text("\n".join(filter_log), encoding="utf-8")
    logger.info("Features after filtering: %d", len(feature_cols))

    train_ids, val_ids = resolve_train_val_split(args, video_ids, output_video_ids)
    logger.info("Train: %s, Val: %s", train_ids, val_ids)

    normalizer = FeatureNormalizer()
    normalizer.fit({video_id: video_dfs[video_id] for video_id in train_ids}, feature_cols)

    video_weights = None
    if args.engagement_weight:
        video_weights = load_video_weights(train_ids, args.snapshot_dir)

    train_ds = WindowedSeqDataset(video_dfs, train_ids, feature_cols, normalizer, args.window_size, args.window_stride, video_weights=video_weights)
    val_ds = WindowedSeqDataset(video_dfs, val_ids, feature_cols, normalizer, args.window_size, args.window_stride)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info("Train windows: %d, Val windows: %d", len(train_ds), len(val_ds))

    num_features = len(feature_cols)
    model = RetentionTransformer(n_features=num_features, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout).to(device)
    model_param_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Model params: %d (%.1fK)", model_param_count, model_param_count / 1000)

    result = train_model(model, train_dl, val_dl, device, args, use_engagement_weight=args.engagement_weight)
    plot_training_curve(result["train_losses"], result["val_losses"], os.path.join(args.output_dir, "training_curve.png"))

    all_metrics, holdout_rows = {}, []
    for video_id in video_ids:
        split = "val" if video_id in val_ids else "train"
        y_true, y_pred = predict_video(model, video_dfs[video_id], feature_cols, normalizer, device, args.window_size, apply_smoothing=args.apply_smoothing)
        metrics = seq_metrics(y_pred, y_true)
        all_metrics[video_id] = {**metrics, "split": split, "n_seconds": len(y_true)}
        logger.info("%s [%s]  RMSE=%.4f  MAE=%.4f  spearman=%.3f", video_id, split, metrics["rmse"], metrics["mae"], metrics["spearman"])

        is_ad = video_dfs[video_id]["is_ad"].values if "is_ad" in video_dfs[video_id].columns else None
        plot_prediction(video_id, y_true, y_pred, is_ad, split, metrics, os.path.join(args.output_dir, "videos", video_id, "prediction.png"))
        if split == "val":
            holdout_rows.extend(
                {"video": video_id, "second": s, "true_retention": y_true[s], "pred_retention": y_pred[s], "abs_error": abs(y_true[s] - y_pred[s])} for s in range(len(y_true))
            )

    compute_feature_importance(model, feature_cols, video_dfs, val_ids, normalizer, device, args.output_dir, args.window_size)

    assert normalizer.median is not None and normalizer.iqr is not None
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": num_features,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "feature_cols": feature_cols,
            "normalizer_mean": normalizer.median.tolist(),
            "normalizer_std": normalizer.iqr.tolist(),
            "normalizer_median": normalizer.median.tolist(),
            "normalizer_iqr": normalizer.iqr.tolist(),
            "ret_min": normalizer.ret_min,
            "ret_max": normalizer.ret_max,
        },
        os.path.join(args.output_dir, "transformer_model.pt"),
    )
    logger.info("Saved model")

    Path(os.path.join(args.output_dir, "metrics.json")).write_text(
        json.dumps(
            {
                "model": "RetentionTransformer",
                "n_features": num_features,
                "feature_cols": feature_cols,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "best_val_mae": result["best_val_mae"],
                "epochs_trained": result["epochs_trained"],
                "elapsed_sec": result["elapsed_sec"],
                "per_video": all_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved metrics")

    if holdout_rows:
        pd.DataFrame(holdout_rows).to_csv(os.path.join(args.output_dir, "holdout_prediction_vs_true.csv"), index=False)
        logger.info("Saved holdout")
    logger.info("Done. Best val MAE=%.4f, epochs=%d, time=%.0fs", result["best_val_mae"], result["epochs_trained"], result["elapsed_sec"])


if __name__ == "__main__":
    main()
