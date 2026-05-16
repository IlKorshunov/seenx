from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.retention_lstm import RetentionLSTM
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
from train.common.retention_plots import COLOR_ERR_POS, GRID_ALPHA, plot_retention_prediction, plot_training_curve, save_figure
from train.common.split_utils import apply_train_id_file_filter, resolve_train_val_split
from train.lstm.lstm_seq_base import run_sequence_training_loop


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RetentionLSTM on merged features.")
    a = p.add_argument
    a("--output-dir-features", default="output")
    a("--snapshot-dir", default="data")
    a("--use-curve-raw", action="store_true", default=True)
    a("--no-use-curve-raw", dest="use_curve_raw", action="store_false")
    a("--output-dir", default="lstm_exp/latest")
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
    a("--hidden-size", type=int, default=128)
    a("--n-layers", type=int, default=2)
    a("--dropout", type=float, default=0.2)
    a("--bidirectional", action="store_true", default=True)
    a("--no-bidirectional", dest="bidirectional", action="store_false")
    a("--head-type", choices=["cumulative", "sigmoid", "tanh"], default="tanh", help="cumulative/sigmoid=direct curve; tanh=residual+baseline")
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
    a("--curve-points", type=int, default=0)
    a("--time-features", choices=["none", "frac", "frac_sec"], default="none")
    a("--min-duration-sec", type=float, default=0)
    a("--max-duration-sec", type=float, default=0)
    a("--no-baseline", action="store_true", default=False)
    a("--train-video-ids-file", default="")
    return p.parse_args()


def plot_gradient_feature_importance(model, feature_cols, video_dfs, val_ids, normalizer, device, out_dir, window_size):
    model.train()
    importance, n = np.zeros(len(feature_cols)), 0
    for vid in val_ids:
        df = video_dfs[vid]
        fm = df.reindex(columns=feature_cols, fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        if normalizer is not None:
            fm = normalizer.transform(fm).astype(np.float32)
        ret = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values.astype(np.float32)
        wl = min(window_size, len(fm))
        inp = torch.tensor(fm[:wl], dtype=torch.float32).unsqueeze(0).to(device)
        inp.requires_grad_(True)
        tgt = torch.tensor(ret[:wl], dtype=torch.float32).unsqueeze(0).to(device)
        nn.functional.smooth_l1_loss(model(inp), tgt).backward()
        importance += inp.grad.abs().squeeze(0).cpu().numpy().mean(axis=0)
        n += 1
    model.eval()
    if n:
        importance /= n
    ranking = sorted(zip(feature_cols, importance, strict=True), key=lambda t: -t[1])
    pd.DataFrame(ranking, columns=["feature", "gradient_importance"]).to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
    top_n = min(30, len(ranking))
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    names, vals = zip(*ranking[:top_n], strict=True)
    ax.barh(names[::-1], vals[::-1], color=COLOR_ERR_POS)
    ax.set(xlabel="Mean |gradient|", title=f"Feature Importance (top {top_n})")
    ax.grid(True, alpha=GRID_ALPHA, axis="x")
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "feature_importance.png"))
    logger.info("Feature importance: %d features", len(ranking))


def main():
    args = parse_args()
    device, od = torch.device(args.device), args.output_dir
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    os.makedirs(od, exist_ok=True)
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
    outp = Path(args.output_dir_features)
    output_video_ids = sorted(
        p.name.replace("_features.csv", "") for p in outp.glob("*_features.csv") if not p.name.endswith(".partial") and p.name.replace("_features.csv", "") in video_dfs
    )
    feature_cols, filter_log = filter_features(video_dfs, top_k=args.top_k_features or None)
    Path(os.path.join(od, "feature_filter_log.txt")).write_text("\n".join(filter_log), encoding="utf-8")
    logger.info("Features after filtering: %d", len(feature_cols))
    train_ids, val_ids = resolve_train_val_split(args, video_ids, output_video_ids)
    train_ids = apply_train_id_file_filter(train_ids, args)
    logger.info("Train: %s, Val: %s", train_ids, val_ids)
    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)
    ref_sec = max_time_sec_over_videos(video_dfs, train_ids)
    logger.info("Time ref (max time_sec on train): %.1f s", ref_sec)
    vw = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None
    train_ds = WindowedSeqDataset(
        video_dfs,
        train_ids,
        feature_cols,
        normalizer,
        args.window_size,
        args.window_stride,
        video_weights=vw,
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
    model = RetentionLSTM(
        n_features=n_feat, hidden_size=args.hidden_size, n_layers=args.n_layers, dropout=args.dropout, bidirectional=args.bidirectional, head_type=args.head_type
    ).to(device)
    if not args.no_baseline and args.head_type not in ("cumulative", "sigmoid"):
        mx = max(len(video_dfs[v]) for v in train_ids)
        bsum, bcnt = np.zeros(mx, dtype=np.float64), np.zeros(mx, dtype=np.float64)
        for v in train_ids:
            ret = pd.to_numeric(video_dfs[v]["retention"], errors="coerce").fillna(0).values
            bsum[: len(ret)] += ret
            bcnt[: len(ret)] += 1.0
        bc = (bsum / np.maximum(bcnt, 1.0)).astype(np.float32)
        bn = normalizer.normalize_retention(bc).astype(np.float32)
        model.set_baseline(torch.tensor(bn))
        logger.info("Baseline curve set: %d points, raw_mean=%.1f%%, norm_mean=%.4f", len(bc), bc.mean(), bn.mean())
    else:
        logger.info("Baseline skipped" if args.head_type in ("cumulative", "sigmoid") else "Baseline disabled (--no-baseline)")
    logger.info("Model params: %d (%.1fK)", sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters()) / 1000)
    model, result = run_sequence_training_loop(model, train_dl, val_dl, device, args, use_engagement_weight=args.engagement_weight)
    model = model.to(device)
    plot_training_curve(result["train_losses"], result["val_losses"], os.path.join(od, "training_curve.png"), "LSTM")
    all_metrics, holdout_rows = {}, []
    pv_kw = dict(time_feature_mode=args.time_features, ref_time_sec_max=ref_sec)
    for vid in video_ids:
        sn = "val" if vid in val_ids else "train"
        y_true, y_pred = predict_video(model, video_dfs[vid], feature_cols, normalizer, device, args.window_size, **pv_kw)
        m = seq_metrics(y_pred, y_true)
        all_metrics[vid] = {**m, "split": sn, "n_seconds": len(y_true)}
        logger.info("%s [%s]  RMSE=%.4f  MAE=%.4f  r=%.3f", vid, sn, m["rmse"], m["mae"], m["pearson"])
        ia = video_dfs[vid]["is_ad"].values if "is_ad" in video_dfs[vid].columns else None
        plot_retention_prediction(vid, y_true, y_pred, ia, sn, m, os.path.join(od, "videos", vid, "prediction.png"))
        if sn == "val":
            holdout_rows += [
                {"video": vid, "second": s, "true_retention": y_true[s], "pred_retention": y_pred[s], "abs_error": abs(y_true[s] - y_pred[s])} for s in range(len(y_true))
            ]
    plot_mae_summary(all_metrics, od, model_name="LSTM")
    plot_gradient_feature_importance(model, feature_cols, video_dfs, val_ids, normalizer, device, od, args.window_size)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": n_feat,
            "hidden_size": args.hidden_size,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
            "head_type": args.head_type,
            "feature_cols": feature_cols,
            "normalizer_median": normalizer.median.tolist(),
            "normalizer_iqr": normalizer.iqr.tolist(),
            "ret_min": normalizer.ret_min,
            "ret_max": normalizer.ret_max,
        },
        os.path.join(od, "lstm_model.pt"),
    )
    Path(os.path.join(od, "metrics.json")).write_text(
        json.dumps(
            {
                "model": "RetentionLSTM",
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
    if holdout_rows:
        pd.DataFrame(holdout_rows).to_csv(os.path.join(od, "holdout_prediction_vs_true.csv"), index=False)
    logger.info("Done. Best val loss=%.4f, epochs=%d, time=%.0fs", result["best_val_loss"], result["epochs_trained"], result["elapsed_sec"])


if __name__ == "__main__":
    main()
