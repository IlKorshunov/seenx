from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.retention_lstm import RetentionLSTM
from train.common.seq_data_utils import (
    FeatureNormalizer,
    WindowedSeqDataset,
    ad_aware_loss,
    filter_features,
    load_all_merged,
    load_video_weights,
    plot_mae_summary,
    predict_video,
    seq_metrics,
)
from train.common.retention_plots import plot_prediction, plot_training_curve
from train.common.split_utils import apply_train_id_file_filter, resolve_train_val_split


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RetentionLSTM on merged features.")
    a = p.add_argument
    a("--output-dir-features", default="output")
    a("--snapshot-dir", default="data")
    a("--use-curve-raw", action="store_true", default=True)
    a("--no-use-curve-raw", dest="use_curve_raw", action="store_false")
    a("--output-dir", default="lstm_seq_experiment")
    a("--eval-video", default="")
    a("--val-ratio", type=float, default=0.15)
    a("--val-first-n-output", type=int, default=0)
    a("--top-k-features", type=int, default=0)
    a("--ad-penalty-weight", type=float, default=15.0)
    a("--engagement-weight", action="store_true", default=True)
    a("--no-engagement-weight", dest="engagement_weight", action="store_false")
    a("--window-size", type=int, default=128)
    a("--window-stride", type=int, default=64)
    a("--hidden-size", type=int, default=128)
    a("--n-layers", type=int, default=2)
    a("--dropout", type=float, default=0.2)
    a("--bidirectional", action="store_true", default=True)
    a("--no-bidirectional", dest="bidirectional", action="store_false")
    a("--epochs", type=int, default=200)
    a("--batch-size", type=int, default=16)
    a("--lr", type=float, default=1e-3)
    a("--weight-decay", type=float, default=1e-4)
    a("--patience", type=int, default=25)
    a("--grad-clip", type=float, default=1.0)
    a("--random-seed", type=int, default=42)
    a("--device", default="cpu")
    a("--train-video-ids-file", default="")
    a("--min-duration-sec", type=float, default=0)
    a("--max-duration-sec", type=float, default=0)
    a("--apply-smoothing", action="store_true", default=False, help="Apply Savitzky-Golay filter to predictions")
    return p.parse_args()


def train_model(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, device: torch.device, args: argparse.Namespace, use_engagement_weight: bool = True) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = nn.MSELoss(reduction="none")
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(os.path.join(args.output_dir, "tensorboard"), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    except ImportError:
        pass
    best_val, no_improve, best_state = float("inf"), 0, {}
    train_losses, val_losses, t0, epoch = [], [], time.time(), 0
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="ep")
    for epoch in epoch_bar:
        model.train()
        tl, tn = 0.0, 0
        for b in tqdm(train_dl, desc=f"Epoch {epoch} [train]", leave=False, unit="bat"):
            feat, tgt, mask, is_ad = (b["features"].to(device), b["retention"].to(device), b["padding_mask"].to(device), b["is_ad"].to(device))
            spike_triggers = b["spike_triggers"].to(device)
            vw = b["video_weight"].to(device) if use_engagement_weight else None
            pred = model(feat, src_key_padding_mask=mask)
            loss = ad_aware_loss(pred, tgt, is_ad, spike_triggers, mask, crit, args.ad_penalty_weight, video_weight=vw)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            nv = (~mask).sum().item()
            tl, tn = tl + loss.item() * nv, tn + nv
        sched.step()
        train_losses.append(tl / max(tn, 1))
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for b in tqdm(val_dl, desc=f"Epoch {epoch} [val]", leave=False, unit="bat"):
                feat, tgt, mask, is_ad = (b["features"].to(device), b["retention"].to(device), b["padding_mask"].to(device), b["is_ad"].to(device))
                spike_triggers = b["spike_triggers"].to(device)
                nv = (~mask).sum().item()
                vl += ad_aware_loss(model(feat, src_key_padding_mask=mask), tgt, is_ad, spike_triggers, mask, crit, 1.0).item() * nv
                vn += nv
        val_mae = vl / max(vn, 1)
        val_losses.append(val_mae)
        epoch_bar.set_postfix(train=f"{train_losses[-1]:.4f}", val=f"{val_mae:.4f}")
        if writer:
            writer.add_scalar("MAE/train", train_losses[-1], epoch)
            writer.add_scalar("MAE/val", val_mae, epoch)
        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d  train=%.4f  val=%.4f", epoch, args.epochs, train_losses[-1], val_mae)
        if val_mae < best_val:
            best_val, no_improve, best_state = val_mae, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stop at epoch %d", epoch)
                break
    if writer:
        writer.close()
    model.load_state_dict(best_state)
    return {"train_losses": train_losses, "val_losses": val_losses, "best_val_mae": round(best_val, 6), "epochs_trained": epoch, "elapsed_sec": round(time.time() - t0, 1)}


def plot_feature_importance(model, feature_cols, video_dfs, val_ids, normalizer, device, out_dir, window_size):
    model.eval()
    importance, n = np.zeros(len(feature_cols)), 0
    for vid in val_ids:
        df = video_dfs[vid]
        X = df.reindex(columns=feature_cols, fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        if normalizer is not None:
            X = normalizer.transform(X).astype(np.float32)
        y = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values.astype(np.float32)
        ws = min(window_size, len(X))
        inp = torch.tensor(X[:ws], dtype=torch.float32).unsqueeze(0).to(device)
        inp.requires_grad_(True)
        tgt = torch.tensor(y[:ws], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.backends.cudnn.flags(enabled=False):
            nn.functional.smooth_l1_loss(model(inp), tgt).backward()
        importance += inp.grad.abs().squeeze(0).cpu().numpy().mean(axis=0)
        n += 1
    if n:
        importance /= n
    ranking = sorted(zip(feature_cols, importance, strict=True), key=lambda x: -x[1])
    pd.DataFrame(ranking, columns=["feature", "gradient_importance"]).to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
    top_n = min(30, len(ranking))
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    names = [r[0] for r in ranking[:top_n]][::-1]
    vals = [r[1] for r in ranking[:top_n]][::-1]
    ax.barh(names, vals, color="#4CAF50")
    ax.set(xlabel="Mean |gradient|", title=f"Feature Importance (top {top_n})")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance (%d features)", len(ranking))


def main():
    args = parse_args()
    device, od = torch.device(args.device), args.output_dir
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    os.makedirs(od, exist_ok=True)
    logger.info("Loading merged data...")
    video_dfs = load_all_merged(
        args.output_dir_features, args.snapshot_dir, use_curve_raw=args.use_curve_raw, min_duration_sec=args.min_duration_sec, max_duration_sec=args.max_duration_sec
    )
    vids = sorted(video_dfs.keys())
    logger.info("Videos: %s", vids)
    outp = Path(args.output_dir_features)
    output_vids = sorted(
        p.name.replace("_features.csv", "") for p in outp.glob("*_features.csv") if not p.name.endswith(".partial") and p.name.replace("_features.csv", "") in video_dfs
    )
    top_k = args.top_k_features if args.top_k_features > 0 else None
    feature_cols, filter_log = filter_features(video_dfs, top_k=top_k)
    Path(os.path.join(od, "feature_filter_log.txt")).write_text("\n".join(filter_log), encoding="utf-8")
    logger.info("Features after filtering: %d", len(feature_cols))
    train_ids, val_ids = resolve_train_val_split(args, vids, output_vids)
    train_ids = apply_train_id_file_filter(train_ids, args)
    logger.info("Train: %s, Val: %s", train_ids, val_ids)
    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)
    vw = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None
    train_ds = WindowedSeqDataset(video_dfs, train_ids, feature_cols, normalizer, args.window_size, args.window_stride, video_weights=vw)
    val_ds = WindowedSeqDataset(video_dfs, val_ids, feature_cols, normalizer, args.window_size, args.window_stride)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info("Train windows: %d, Val windows: %d", len(train_ds), len(val_ds))
    n_feat = len(feature_cols)
    model = RetentionLSTM(n_features=n_feat, hidden_size=args.hidden_size, n_layers=args.n_layers, dropout=args.dropout, bidirectional=args.bidirectional).to(device)
    logger.info("Model params: %d (%.1fK)", sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters()) / 1000)
    result = train_model(model, train_dl, val_dl, device, args, use_engagement_weight=args.engagement_weight)
    plot_training_curve(result["train_losses"], result["val_losses"], os.path.join(od, "training_curve.png"))
    all_metrics, holdout_rows = {}, []
    for vid in vids:
        sp = "val" if vid in val_ids else "train"
        y_true, y_pred = predict_video(model, video_dfs[vid], feature_cols, normalizer, device, args.window_size, apply_smoothing=args.apply_smoothing)
        m = seq_metrics(y_pred, y_true)
        all_metrics[vid] = {**m, "split": sp, "n_seconds": len(y_true)}
        logger.info("%s [%s]  RMSE=%.4f  MAE=%.4f  spearman=%.3f", vid, sp, m["rmse"], m["mae"], m["spearman"])
        ia = video_dfs[vid]["is_ad"].values if "is_ad" in video_dfs[vid].columns else None
        plot_prediction(vid, y_true, y_pred, ia, sp, m, os.path.join(od, "videos", vid, "prediction.png"))
        if sp == "val":
            holdout_rows += [
                {"video": vid, "second": s, "true_retention": y_true[s], "pred_retention": y_pred[s], "abs_error": abs(y_true[s] - y_pred[s])} for s in range(len(y_true))
            ]
    plot_mae_summary(all_metrics, od, model_name="LSTM-v1")
    plot_feature_importance(model, feature_cols, video_dfs, val_ids, normalizer, device, od, args.window_size)
    mp = os.path.join(od, "lstm_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": n_feat,
            "hidden_size": args.hidden_size,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
            "feature_cols": feature_cols,
            "normalizer_mean": normalizer.median.tolist(),
            "normalizer_std": normalizer.iqr.tolist(),
            "normalizer_median": normalizer.median.tolist(),
            "normalizer_iqr": normalizer.iqr.tolist(),
            "ret_min": normalizer.ret_min,
            "ret_max": normalizer.ret_max,
        },
        mp,
    )
    logger.info("Saved model: %s", mp)
    Path(os.path.join(od, "metrics.json")).write_text(
        json.dumps(
            {
                "model": "RetentionLSTM",
                "n_features": n_feat,
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
    if holdout_rows:
        pd.DataFrame(holdout_rows).to_csv(os.path.join(od, "holdout_prediction_vs_true.csv"), index=False)
        logger.info("Saved holdout: %s", os.path.join(od, "holdout_prediction_vs_true.csv"))
    logger.info("Done. Best val MAE=%.4f, epochs=%d, time=%.0fs", result["best_val_mae"], result["epochs_trained"], result["elapsed_sec"])


if __name__ == "__main__":
    main()
