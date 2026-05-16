from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.retention_transformer import RetentionTransformer
from train.common.seq_data_utils import (
    FeatureNormalizer,
    WindowedSeqDataset,
    filter_features,
    load_all_merged,
    load_video_weights,
    max_time_sec_over_videos,
    plot_mae_summary,
    predict_video,
    resample_video_dfs_to_curve_points,
    seq_metrics,
    time_feature_extra_dim,
)
from train.common.tuned_params_io import merge_tuned_file_into_args
from train.common.retention_plots import plot_retention_prediction, plot_training_curve
from train.common.split_utils import apply_train_id_file_filter, resolve_train_val_split
from train.lstm.lstm_seq_base import run_sequence_training_loop
from train.transformer.transformer_base import build_tuned_feature_filter_kwargs, compute_permutation_feature_importance


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RetentionTransformer on merged features.")
    a = p.add_argument
    a("--output-dir-features", default="output")
    a("--snapshot-dir", default="data")
    a("--use-curve-raw", action="store_true", default=True)
    a("--no-use-curve-raw", dest="use_curve_raw", action="store_false")
    a("--output-dir", default="transformer_exp/latest")
    a("--eval-video", default="")
    a("--val-ratio", type=float, default=0.15)
    a("--val-first-n-output", type=int, default=0)
    a("--top-k-features", type=int, default=0)
    a("--emb-pca-components", type=int, default=36)
    a("--ad-penalty-weight", type=float, default=15.0)
    a("--alpha-corr", type=float, default=0.3)
    a("--alpha-smooth", type=float, default=0.15)
    a("--alpha-delta", type=float, default=0.4)
    a("--alpha-mono", type=float, default=0.03)
    a("--start-boost-secs", type=int, default=15)
    a("--start-boost-factor", type=float, default=2.0)
    a("--smooth-window", type=int, default=7)
    a("--engagement-weight", action="store_true", default=True)
    a("--no-engagement-weight", dest="engagement_weight", action="store_false")
    a("--window-size", type=int, default=128)
    a("--window-stride", type=int, default=64)
    a("--d-model", type=int, default=128)
    a("--n-heads", type=int, default=4)
    a("--n-layers", type=int, default=4)
    a("--d-ff", type=int, default=256)
    a("--dropout", type=float, default=0.2)
    a("--head-type", choices=["cumulative", "sigmoid", "tanh"], default="tanh", help="Output head: cumulative/sigmoid (direct curve); tanh = residual + optional baseline.")
    a("--epochs", type=int, default=200)
    a("--batch-size", type=int, default=16)
    a("--lr", type=float, default=5e-4)
    a("--weight-decay", type=float, default=1e-3)
    a("--patience", type=int, default=30)
    a("--grad-clip", type=float, default=1.0)
    a("--warmup-epochs", type=int, default=10)
    a("--swa-start-epoch", type=int, default=0)
    a("--swa-lr", type=float, default=1e-4)
    a("--feature-mask-prob", type=float, default=0.1)
    a("--noise-std", type=float, default=0.02)
    a("--random-seed", type=int, default=42)
    a("--device", default="cpu")
    a("--tuned-params-json", default="")
    a("--tuned-apply-architecture", action="store_true")
    a("--curve-points", type=int, default=0)
    a("--time-features", choices=["none", "frac", "frac_sec"], default="none")
    a("--min-duration-sec", type=float, default=0)
    a("--max-duration-sec", type=float, default=0)
    a("--no-baseline", action="store_true", default=False)
    a("--train-video-ids-file", default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if getattr(args, "tuned_params_json", ""):
        merge_tuned_file_into_args(
            args,
            args.tuned_params_json,
            model_family="tabular_transformer",
            apply_architecture=args.tuned_apply_architecture,
            save_copy_to=Path(args.output_dir) / "tuned_params_applied.json",
        )

    logger.info("Loading merged data...")
    video_dfs = load_all_merged(
        args.output_dir_features,
        args.snapshot_dir,
        use_curve_raw=args.use_curve_raw,
        embeddings_root="embeddings",
        emb_pca_components=args.emb_pca_components,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )
    if args.curve_points and args.curve_points > 0:
        video_dfs = resample_video_dfs_to_curve_points(video_dfs, args.curve_points)
    video_ids = sorted(video_dfs.keys())
    logger.info("Videos: %s", video_ids)

    output_dir = Path(args.output_dir_features)
    output_video_ids = sorted(
        csv_path.name.replace("_features.csv", "")
        for csv_path in output_dir.glob("*_features.csv")
        if not csv_path.name.endswith(".partial") and csv_path.name.replace("_features.csv", "") in video_dfs
    )

    filter_kw = build_tuned_feature_filter_kwargs(args)
    feature_cols, filter_log = filter_features(video_dfs, **filter_kw)
    Path(os.path.join(args.output_dir, "feature_filter_log.txt")).write_text("\n".join(filter_log), encoding="utf-8")
    logger.info("Features after filtering: %d", len(feature_cols))

    train_ids, val_ids = resolve_train_val_split(args, video_ids, output_video_ids)
    train_ids = apply_train_id_file_filter(train_ids, args)
    logger.info("Train: %s, Val: %s", train_ids, val_ids)

    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)
    ref_sec = max_time_sec_over_videos(video_dfs, train_ids)
    logger.info("Time ref (max time_sec on train): %.1f s", ref_sec)
    video_weights = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None

    train_ds = WindowedSeqDataset(
        video_dfs,
        train_ids,
        feature_cols,
        normalizer,
        args.window_size,
        args.window_stride,
        video_weights=video_weights,
        feature_mask_prob=args.feature_mask_prob,
        noise_std=args.noise_std,
        time_feature_mode=args.time_features,
        ref_time_sec_max=ref_sec,
    )
    val_ds = WindowedSeqDataset(video_dfs, val_ids, feature_cols, normalizer, args.window_size, args.window_stride, time_feature_mode=args.time_features, ref_time_sec_max=ref_sec)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info("Train windows: %d, Val windows: %d", len(train_ds), len(val_ds))

    n_feat = len(feature_cols) + time_feature_extra_dim(args.time_features)
    if args.head_type in ("cumulative", "sigmoid"):
        args.alpha_mono = 0.0
        logger.info("alpha_mono=0 for head_type=%s", args.head_type)

    model = RetentionTransformer(
        n_features=n_feat, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout, head_type=args.head_type
    ).to(device)

    if not args.no_baseline and args.head_type not in ("cumulative", "sigmoid"):
        max_len = max(len(video_dfs[v]) for v in train_ids)
        baseline_sum = np.zeros(max_len, dtype=np.float64)
        baseline_count = np.zeros(max_len, dtype=np.float64)
        for v in train_ids:
            ret = pd.to_numeric(video_dfs[v]["retention"], errors="coerce").fillna(0).values
            baseline_sum[: len(ret)] += ret
            baseline_count[: len(ret)] += 1.0
        baseline_curve = (baseline_sum / np.maximum(baseline_count, 1.0)).astype(np.float32)
        baseline_norm = normalizer.normalize_retention(baseline_curve).astype(np.float32)
        model.set_baseline(torch.tensor(baseline_norm))
        logger.info("Baseline curve set: %d points, raw_mean=%.1f%%, norm_mean=%.4f", len(baseline_curve), baseline_curve.mean(), baseline_norm.mean())
    elif args.head_type in ("cumulative", "sigmoid"):
        logger.info("Baseline skipped (not used for head_type=%s)", args.head_type)
    else:
        logger.info("Baseline disabled (--no-baseline)")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d (%.1fK)", n_params, n_params / 1000)

    model, result = run_sequence_training_loop(model, train_dl, val_dl, device, args, use_engagement_weight=args.engagement_weight)
    model = model.to(device)
    plot_training_curve(result["train_losses"], result["val_losses"], os.path.join(args.output_dir, "training_curve.png"), "Transformer")

    all_metrics: dict = {}
    holdout_rows = []
    for vid in video_ids:
        split_name = "val" if vid in val_ids else "train"
        y_true, y_pred = predict_video(model, video_dfs[vid], feature_cols, normalizer, device, args.window_size, time_feature_mode=args.time_features, ref_time_sec_max=ref_sec)
        metrics = seq_metrics(y_pred, y_true)
        all_metrics[vid] = {**metrics, "split": split_name, "n_seconds": len(y_true)}
        logger.info("%s [%s]  RMSE=%.4f  MAE=%.4f  r=%.3f", vid, split_name, metrics["rmse"], metrics["mae"], metrics["pearson"])
        is_ad = video_dfs[vid]["is_ad"].values if "is_ad" in video_dfs[vid].columns else None
        plot_retention_prediction(vid, y_true, y_pred, is_ad, split_name, metrics, os.path.join(args.output_dir, "videos", vid, "prediction.png"))
        if split_name == "val":
            holdout_rows.extend(
                {"video": vid, "second": sec, "true_retention": y_true[sec], "pred_retention": y_pred[sec], "abs_error": abs(y_true[sec] - y_pred[sec])}
                for sec in range(len(y_true))
            )

    plot_mae_summary(all_metrics, args.output_dir, model_name="Transformer")
    compute_permutation_feature_importance(
        model, feature_cols, video_dfs, val_ids, normalizer, device, args.output_dir, args.window_size, time_feature_mode=args.time_features, ref_time_sec_max=ref_sec
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": n_feat,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "head_type": args.head_type,
            "feature_cols": feature_cols,
            "normalizer_median": normalizer.median.tolist(),
            "normalizer_iqr": normalizer.iqr.tolist(),
            "ret_min": normalizer.ret_min,
            "ret_max": normalizer.ret_max,
        },
        os.path.join(args.output_dir, "transformer_model.pt"),
    )

    Path(os.path.join(args.output_dir, "metrics.json")).write_text(
        json.dumps(
            {
                "model": "RetentionTransformer",
                "n_features": n_feat,
                "feature_cols": feature_cols,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "best_val_loss": result["best_val_loss"],
                "epochs_trained": result["epochs_trained"],
                "elapsed_sec": result["elapsed_sec"],
                "per_video": all_metrics,
                "config": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Done. Best val loss=%.4f, epochs=%d, time=%.0fs", result["best_val_loss"], result["epochs_trained"], result["elapsed_sec"])


if __name__ == "__main__":
    main()
