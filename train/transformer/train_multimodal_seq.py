"""
Multimodal per-second retention predictor: raw embeddings + tabular features.

Supports both MultimodalRetentionTransformer and MultimodalRetentionLSTM.
Embeddings (visual 768 + audio 512 + text 256 = 1536) are projected by learned
per-modality layers inside the model, NOT reduced by PCA.

Usage:
    python train/train_multimodal_seq.py --arch transformer --val-first-n-output 10 --epochs 200
    python -m train.transformer.train_multimodal_seq --arch lstm --val-first-n-output 10 --epochs 200
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm


matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.retention_multimodal_lstm import MultimodalRetentionLSTM
from src.models.retention_multimodal_transformer import MultimodalRetentionTransformer
from train.common.seq_data_utils import (
    FeatureNormalizer,
    MultimodalWindowedDataset,
    composite_loss,
    filter_features,
    load_aligned_embeddings_for_videos,
    load_all_merged,
    load_video_weights,
    max_time_sec_over_videos,
    plot_mae_summary,
    predict_video_multimodal,
    resample_embeddings_to_match_dfs,
    resample_video_dfs_to_curve_points,
    seq_metrics,
    time_feature_extra_dim,
)
from train.common.tuned_params_io import apply_best_params_to_args, merge_tuned_file_into_args


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

C_BLUE, C_ORANGE, C_PURPLE = "#2196F3", "#FF5722", "#9C27B0"
C_GREEN, C_RED = "#4CAF50", "#F44336"
GRID_ALPHA = 0.3
PLOT_DPI = 150


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multimodal retention model (embeddings + tabular).")
    a = p.add_argument
    a("--arch", default="transformer", choices=["transformer", "lstm"])
    a("--output-dir-features", default="output")
    a("--snapshot-dir", default="data")
    a("--embeddings-root", default="embeddings")
    a("--use-curve-raw", action="store_true", default=True)
    a("--no-use-curve-raw", dest="use_curve_raw", action="store_false")
    a("--output-dir", default="")
    a("--tune-first", action="store_true", help="Run Optuna tuning first, then train final model with best params")
    a("--n-trials", type=int, default=50, help="Optuna trials when --tune-first is enabled")
    a("--epochs-per-trial", type=int, default=80, help="Epochs per Optuna trial when --tune-first is enabled")
    a("--tune-output-dir", default="tune_hp/results", help="Directory where Optuna results are stored")
    a("--study-name", default="", help="Optional Optuna study name override")
    a("--tuned-params-json", default="", help="Path to Optuna best JSON (e.g. tune_hp/results/tune_multimodal_lstm_best.json)")
    a("--use-conv-blocks", action="store_true", default=False, help="Use 1D convolutions before/after sequence model")
    a("--apply-smoothing", action="store_true", default=False, help="Apply Savitzky-Golay filter to predictions")
    a("--tuned-apply-architecture", action="store_true", help="Also apply d_model/n_layers/... from tuned JSON (default: keep CLI sizes)")
    a("--global-calibration", action="store_true", help="Apply OLS affine calibration on train preds (default: off; model has learnable scale)")
    a("--train-video-ids-file", default="", help="Optional text file: one train video id per line (subset of train split)")
    a("--eval-video", default="")
    a("--val-ratio", type=float, default=0.15)
    a("--val-first-n-output", type=int, default=0)
    a("--top-k-features", type=int, default=0)

    a("--ad-penalty-weight", type=float, default=15.0)
    a("--alpha-corr", type=float, default=0.3)
    a("--alpha-smooth", type=float, default=0.15)
    a("--alpha-delta", type=float, default=0.4, help="Delta loss weight (supervised first-order diff)")
    a("--alpha-mono", type=float, default=0.03, help="Monotonicity loss weight (penalize upward jumps)")
    a("--start-boost-secs", type=int, default=15, help="Boost loss weight for first N seconds")
    a("--start-boost-factor", type=float, default=2.0)
    a("--smooth-window", type=int, default=7, help="Savitzky-Golay smoothing window for predictions")
    a("--engagement-weight", action="store_true", default=True)
    a("--no-engagement-weight", dest="engagement_weight", action="store_false")

    a("--window-size", type=int, default=128)
    a("--window-stride", type=int, default=64)
    a("--d-model", type=int, default=128, help="Projection dim / hidden size")
    a("--n-heads", type=int, default=4)
    a("--n-layers", type=int, default=4)
    a("--d-ff", type=int, default=256)
    a("--dropout", type=float, default=0.2)

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
    a("--curve-points", type=int, default=0, help="Resample each video to N time points (0=per-second; like LOO curve_points)")
    a("--time-features", choices=["none", "frac", "frac_sec"], default="none", help="Append time_frac [0,1] and/or time_sec/ref_max to tabular input")
    a("--min-duration-sec", type=float, default=0, help="Drop videos shorter than this (seconds); 0=no filter. E.g. 540 for 9 min")
    a("--max-duration-sec", type=float, default=0, help="Drop videos longer than this (seconds); 0=no filter. E.g. 1620 for 27 min")
    return p.parse_args()


def _to_device(batch, device, *keys):
    return tuple(batch[k].to(device) for k in keys)


def _lr_lambda(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / max(total - warmup, 1)))


def train_model(model, train_dl, val_dl, device, args, use_engagement_weight=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: _lr_lambda(ep, args.warmup_epochs, args.epochs))

    swa_start = args.swa_start_epoch if args.swa_start_epoch > 0 else int(args.epochs * 0.7)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
    swa_active = False

    best_val_loss, no_improve, best_state = float("inf"), 0, {}
    best_state_owner = "model"
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in (pbar := tqdm(range(1, args.epochs + 1), desc="Training", unit="ep")):
        model.train()
        tl, tn = 0.0, 0
        for batch in tqdm(train_dl, desc=f"Ep {epoch} [train]", leave=False, unit="b"):
            emb, tab, tgt, pad_mask, ad_mask, spike_triggers = _to_device(batch, device, "embeddings", "tabular", "retention", "padding_mask", "is_ad", "spike_triggers")
            vw = batch["video_weight"].to(device) if use_engagement_weight else None

            pred = model(emb, tabular=tab, src_key_padding_mask=pad_mask)
            loss = composite_loss(
                pred,
                tgt,
                ad_mask,
                spike_triggers,
                pad_mask,
                args.ad_penalty_weight,
                vw,
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

            nv = (~pad_mask).sum().item()
            tl += loss.item() * nv
            tn += nv

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_active = True
        else:
            scheduler.step()

        train_losses.append(tl / max(tn, 1))

        eval_model = swa_model if swa_active else model
        eval_model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Ep {epoch} [val]", leave=False, unit="b"):
                emb, tab, tgt, pad_mask, ad_mask, spike_triggers = _to_device(batch, device, "embeddings", "tabular", "retention", "padding_mask", "is_ad", "spike_triggers")
                pred = eval_model(emb, tabular=tab, src_key_padding_mask=pad_mask)
                loss = composite_loss(pred, tgt, ad_mask, spike_triggers, pad_mask, 1.0, None, args.alpha_corr, 0.0, 0.0, 0, 1.0, args.alpha_delta)
                nv = (~pad_mask).sum().item()
                vl += loss.item() * nv
                vn += nv
        val_losses.append(vl / max(vn, 1))

        pbar.set_postfix(train=f"{train_losses[-1]:.4f}", val=f"{val_losses[-1]:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}", swa="on" if swa_active else "off")
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
            best_val_loss, no_improve = val_losses[-1], 0
            if swa_active:
                best_state = {k: v.cpu().clone() for k, v in swa_model.state_dict().items()}
                best_state_owner = "swa"
            else:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_state_owner = "model"
        else:
            no_improve += 1
            if not swa_active and no_improve >= args.patience:
                logger.info("Early stop at epoch %d", epoch)
                break

    if best_state_owner == "swa":
        swa_model.load_state_dict(best_state)
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in swa_model.modules())
        if has_bn:
            torch.optim.swa_utils.update_bn(train_dl, swa_model, device=device)
        model = copy.deepcopy(swa_model.module)
    else:
        model.load_state_dict(best_state)

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": round(best_val_loss, 6),
        "epochs_trained": epoch,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="train", color=C_BLUE)
    ax.plot(val_losses, label="val", color=C_ORANGE)
    ax.set(xlabel="epoch", ylabel="composite loss", title="Multimodal Training Curve")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()
    _save_fig(fig, out_path)


def plot_prediction(vid, y_true, y_pred, is_ad, split, metrics, out_path):
    t = np.arange(len(y_true))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)
    a1.plot(t, y_true, color=C_BLUE, label="actual", linewidth=1.2)
    a1.plot(t, y_pred, color=C_ORANGE, label="predicted", alpha=0.8, linewidth=1.2)
    a1.fill_between(t, y_true, y_pred, alpha=0.1, color=C_PURPLE)
    if is_ad is not None and (ad := is_ad > 0.5).any():
        a1.fill_between(t, 0, 1, where=ad, alpha=0.15, color="red", label="ad segment")
    a1.set(ylabel="Retention (%)", title=f"{vid} [{split}]  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  r={metrics['pearson']:.3f}")
    a1.legend(fontsize=9)
    a1.grid(True, alpha=GRID_ALPHA)
    res = y_pred - y_true
    a2.fill_between(t, res, alpha=0.3, color=C_GREEN, where=res >= 0)
    a2.fill_between(t, res, alpha=0.3, color=C_RED, where=res < 0)
    a2.axhline(0, color="black", linewidth=0.5)
    a2.set(xlabel="sec", ylabel="error")
    a2.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()
    _save_fig(fig, out_path)


def _tune_arch_name(arch: str) -> str:
    return f"multimodal_{arch}"


def _apply_tuned_params(args: argparse.Namespace) -> argparse.Namespace:
    """Run Optuna first and merge best params into training args."""
    tune_arch = _tune_arch_name(args.arch)
    study_name = args.study_name or f"tune_{tune_arch}"
    tune_script = Path(__file__).resolve().parent.parent / "tune_hp" / "tune.py"
    cmd = [
        sys.executable,
        str(tune_script),
        "--arch",
        tune_arch,
        "--output-dir-features",
        args.output_dir_features,
        "--snapshot-dir",
        args.snapshot_dir,
        "--embeddings-root",
        args.embeddings_root,
        "--val-first-n-output",
        str(args.val_first_n_output),
        "--n-trials",
        str(args.n_trials),
        "--epochs-per-trial",
        str(args.epochs_per_trial),
        "--device",
        args.device,
        "--output-dir",
        args.tune_output_dir,
        "--study-name",
        study_name,
        "--random-seed",
        str(args.random_seed),
    ]
    if args.use_curve_raw:
        cmd.append("--use-curve-raw")
    else:
        cmd.append("--no-use-curve-raw")

    logger.info("Running Optuna tuning first: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    best_path = Path(args.tune_output_dir) / f"{study_name}_best.json"
    if not best_path.exists():
        raise FileNotFoundError(f"Optuna best params file not found: {best_path}")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    best_params = best.get("best_params", {})
    logger.info("Applying tuned params from %s", best_path)
    fam = "multimodal_transformer" if args.arch == "transformer" else "multimodal_lstm"
    apply_best_params_to_args(args, best_params, model_family=fam, apply_architecture=True)

    tuned_out = Path(args.output_dir) / "tuned_params_applied.json"
    tuned_out.parent.mkdir(parents=True, exist_ok=True)
    tuned_out.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved applied tuned params -> %s", tuned_out)
    return args


def main():
    args = parse_args()
    if not args.output_dir:
        base = "transformer_exp" if args.arch == "transformer" else "lstm_exp"
        args.output_dir = os.path.join(base, "multimodal_latest")
    if args.tune_first:
        args = _apply_tuned_params(args)
    elif getattr(args, "tuned_params_json", ""):
        fam = "multimodal_transformer" if args.arch == "transformer" else "multimodal_lstm"
        merge_tuned_file_into_args(
            args, args.tuned_params_json, model_family=fam, apply_architecture=args.tuned_apply_architecture, save_copy_to=Path(args.output_dir) / "tuned_params_applied.json"
        )
    device = torch.device(args.device)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading merged data (no PCA — raw embeddings go to model)...")
    video_dfs = load_all_merged(
        args.output_dir_features,
        args.snapshot_dir,
        use_curve_raw=args.use_curve_raw,
        emb_pca_components=0,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )
    video_ids = sorted(video_dfs.keys())

    logger.info("Loading aligned embeddings...")
    video_embeddings = load_aligned_embeddings_for_videos(video_dfs, args.embeddings_root)
    if args.curve_points and args.curve_points > 0:
        video_dfs = resample_video_dfs_to_curve_points(video_dfs, args.curve_points)
        video_embeddings = resample_embeddings_to_match_dfs(video_embeddings, video_dfs)

    output_dir = Path(args.output_dir_features)
    output_video_ids = sorted(
        p.name.replace("_features.csv", "") for p in output_dir.glob("*_features.csv") if not p.name.endswith(".partial") and p.name.replace("_features.csv", "") in video_dfs
    )

    filter_kwargs = {"top_k": args.top_k_features or None}
    if hasattr(args, "tuned_corr_threshold"):
        filter_kwargs["redundant_corr_threshold"] = args.tuned_corr_threshold
    if hasattr(args, "tuned_nan_pct"):
        filter_kwargs["max_nan_pct"] = args.tuned_nan_pct
    if hasattr(args, "tuned_nonzero_pct"):
        filter_kwargs["min_nonzero_pct"] = args.tuned_nonzero_pct
    if hasattr(args, "tuned_top_k"):
        tk = args.tuned_top_k
        filter_kwargs["top_k"] = tk if tk and tk > 0 else None
    feature_cols, filter_log = filter_features(video_dfs, **filter_kwargs)
    Path(os.path.join(args.output_dir, "feature_filter_log.txt")).write_text("\n".join(filter_log), encoding="utf-8")
    logger.info("Tabular features after filtering: %d", len(feature_cols))

    if args.eval_video and args.eval_video in video_dfs:
        val_ids = [args.eval_video]
        train_ids = [v for v in video_ids if v != args.eval_video]
    elif args.val_first_n_output > 0:
        n_val = min(args.val_first_n_output, len(output_video_ids))
        val_ids = output_video_ids[:n_val]
        train_ids = [v for v in video_ids if v not in set(val_ids)]
    else:
        rng = np.random.RandomState(args.random_seed)
        rng.shuffle(video_ids)
        n_val = max(1, int(len(video_ids) * args.val_ratio))
        val_ids, train_ids = video_ids[:n_val], video_ids[n_val:]
    if getattr(args, "train_video_ids_file", ""):
        allow = {ln.strip() for ln in Path(args.train_video_ids_file).read_text(encoding="utf-8").splitlines() if ln.strip()}
        before = len(train_ids)
        train_ids = [v for v in train_ids if v in allow]
        logger.info("Train subset from file: %d -> %d videos (%s)", before, len(train_ids), args.train_video_ids_file)

    logger.info("Train: %d videos, Val: %d videos", len(train_ids), len(val_ids))

    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)

    ref_sec = max_time_sec_over_videos(video_dfs, train_ids)
    logger.info("Time ref (max time_sec on train): %.1f s", ref_sec)

    video_weights = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None

    train_ds = MultimodalWindowedDataset(
        video_dfs,
        video_embeddings,
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
    val_ds = MultimodalWindowedDataset(
        video_dfs, video_embeddings, val_ids, feature_cols, normalizer, args.window_size, args.window_stride, time_feature_mode=args.time_features, ref_time_sec_max=ref_sec
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info("Train windows: %d, Val windows: %d", len(train_ds), len(val_ds))

    n_tab = len(feature_cols) + time_feature_extra_dim(args.time_features)
    if args.arch == "lstm":
        model = MultimodalRetentionLSTM(hidden_size=args.d_model, n_layers=args.n_layers, dropout=args.dropout, n_tabular_features=n_tab, use_conv_blocks=args.use_conv_blocks).to(
            device
        )
    else:
        model = MultimodalRetentionTransformer(
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout, n_tabular_features=n_tab, use_conv_blocks=args.use_conv_blocks
        ).to(device)

    max_len = max(len(video_dfs[v]) for v in train_ids)
    baseline_acc = np.zeros(max_len, dtype=np.float64)
    baseline_cnt = np.zeros(max_len, dtype=np.float64)
    for v in train_ids:
        ret = pd.to_numeric(video_dfs[v]["retention"], errors="coerce").fillna(0).values
        baseline_acc[: len(ret)] += ret
        baseline_cnt[: len(ret)] += 1.0
    baseline_curve = (baseline_acc / np.maximum(baseline_cnt, 1.0)).astype(np.float32)
    baseline_norm = normalizer.normalize_retention(baseline_curve).astype(np.float32)
    model.set_baseline(torch.tensor(baseline_norm))
    logger.info("Baseline curve set: %d points, raw_mean=%.1f%%, norm_mean=%.4f", len(baseline_curve), baseline_curve.mean(), baseline_norm.mean())

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s, params: %d (%.1fK), tabular features: %d", args.arch, n_params, n_params / 1000, n_tab)

    model, result = train_model(model, train_dl, val_dl, device, args, use_engagement_weight=args.engagement_weight)
    model = model.to(device)
    plot_training_curve(result["train_losses"], result["val_losses"], os.path.join(args.output_dir, "training_curve.png"))

    # Fit global affine calibration on train predictions to fix scale shift
    all_train_true, all_train_pred = [], []
    raw_preds = {}
    for vid in video_ids:
        emb = video_embeddings.get(vid)
        y_true, y_pred = predict_video_multimodal(
            model,
            video_dfs[vid],
            emb,
            feature_cols,
            normalizer,
            device,
            args.window_size,
            smooth_window=args.smooth_window,
            apply_smoothing=args.apply_smoothing,
            time_feature_mode=args.time_features,
            ref_time_sec_max=ref_sec,
        )
        raw_preds[vid] = (y_true, y_pred)
        if vid in train_ids:
            all_train_true.append(y_true)
            all_train_pred.append(y_pred)

    cal_true = np.concatenate(all_train_true)
    cal_pred = np.concatenate(all_train_pred)
    if args.global_calibration:
        A = np.vstack([cal_pred, np.ones(len(cal_pred))]).T
        try:
            cal_a, cal_b = np.linalg.lstsq(A, cal_true, rcond=None)[0]
            cal_a = float(np.clip(cal_a, 0.5, 2.0))
            cal_b = float(np.mean(cal_true) - cal_a * np.mean(cal_pred))
            cal_b = float(np.clip(cal_b, -50.0, 50.0))
        except Exception:
            cal_a, cal_b = 1.0, 0.0
        logger.info("Global calibration: a=%.4f, b=%.4f", cal_a, cal_b)
    else:
        cal_a, cal_b = 1.0, 0.0
        logger.info("Global calibration: disabled (raw model output; use --global-calibration for OLS a,b)")

    all_metrics = {}
    for vid in video_ids:
        split = "val" if vid in val_ids else "train"
        y_true, y_pred = raw_preds[vid]
        y_pred_cal = (cal_a * y_pred + cal_b).astype(y_pred.dtype)
        m = seq_metrics(y_pred_cal, y_true)
        all_metrics[vid] = {**m, "split": split, "n_seconds": len(y_true)}
        logger.info("%s [%s]  RMSE=%.4f  MAE=%.4f  r=%.3f", vid, split, m["rmse"], m["mae"], m["pearson"])
        is_ad = video_dfs[vid]["is_ad"].values if "is_ad" in video_dfs[vid].columns else None
        plot_prediction(vid, y_true, y_pred_cal, is_ad, split, m, os.path.join(args.output_dir, "videos", vid, "prediction.png"))

    model_label = "Multimodal-Transformer" if args.arch == "transformer" else "Multimodal-LSTM"
    plot_mae_summary(all_metrics, args.output_dir, model_name=model_label)

    # --- Feature importance: tabular permutation + modality ablation ---
    logger.info("Computing feature importance...")
    model.eval()

    # Tabular permutation importance
    # Per-video constant features (LLM, hook_score, n_chapters, etc.) need
    # cross-video shuffling: swap the constant value between videos.
    baseline_maes = {}
    for vid in val_ids:
        emb_v = video_embeddings.get(vid)
        yt, yp = predict_video_multimodal(
            model,
            video_dfs[vid],
            emb_v,
            feature_cols,
            normalizer,
            device,
            args.window_size,
            smooth_window=0,
            apply_smoothing=False,
            time_feature_mode=args.time_features,
            ref_time_sec_max=ref_sec,
        )
        baseline_maes[vid] = float(np.mean(np.abs(yp - yt)))
    baseline_mae = float(np.mean(list(baseline_maes.values())))

    def _is_per_video_constant(fn: str) -> bool:
        for vid in val_ids:
            if fn in video_dfs[vid].columns:
                vals = video_dfs[vid][fn].dropna().values
                if len(vals) > 1 and np.std(vals) > 1e-8:
                    return False
        return True

    importance = np.zeros(len(feature_cols))
    rng = np.random.RandomState(42)
    for fi, fn in enumerate(feature_cols):
        is_constant = _is_per_video_constant(fn)
        mae_inc = []
        for _ in range(3):
            if is_constant:
                vid_values = {}
                for vid in val_ids:
                    if fn in video_dfs[vid].columns:
                        vals = video_dfs[vid][fn].dropna().values
                        vid_values[vid] = float(vals[0]) if len(vals) > 0 else 0.0
                    else:
                        vid_values[vid] = 0.0
                shuffled_vals = list(vid_values.values())
                rng.shuffle(shuffled_vals)
                shuffled_map = dict(zip(vid_values.keys(), shuffled_vals, strict=True))
                trial_maes = []
                for vid in val_ids:
                    sdf = video_dfs[vid].copy()
                    if fn in sdf.columns:
                        sdf[fn] = shuffled_map[vid]
                    emb_v = video_embeddings.get(vid)
                    _, yp = predict_video_multimodal(
                        model,
                        sdf,
                        emb_v,
                        feature_cols,
                        normalizer,
                        device,
                        args.window_size,
                        smooth_window=0,
                        apply_smoothing=False,
                        time_feature_mode=args.time_features,
                        ref_time_sec_max=ref_sec,
                    )
                    yt = video_dfs[vid]["retention"].values
                    trial_maes.append(float(np.mean(np.abs(yp - yt))))
            else:
                trial_maes = []
                for vid in val_ids:
                    sdf = video_dfs[vid].copy()
                    if fn in sdf.columns:
                        sdf[fn] = rng.permutation(sdf[fn].values)
                    emb_v = video_embeddings.get(vid)
                    _, yp = predict_video_multimodal(
                        model,
                        sdf,
                        emb_v,
                        feature_cols,
                        normalizer,
                        device,
                        args.window_size,
                        smooth_window=0,
                        apply_smoothing=False,
                        time_feature_mode=args.time_features,
                        ref_time_sec_max=ref_sec,
                    )
                    yt = video_dfs[vid]["retention"].values
                    trial_maes.append(float(np.mean(np.abs(yp - yt))))
            mae_inc.append(np.mean(trial_maes) - baseline_mae)
        importance[fi] = np.mean(mae_inc)

    ranking = sorted(zip(feature_cols, importance, strict=True), key=lambda x: -x[1])
    pd.DataFrame(ranking, columns=["feature", "importance_mae_increase"]).to_csv(os.path.join(args.output_dir, "feature_importance.csv"), index=False)

    top_n = min(30, len(ranking))
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    names, vals = zip(*ranking[:top_n], strict=True)
    ax.barh(names[::-1], vals[::-1], color=C_BLUE)
    ax.set(xlabel="MAE increase when shuffled", title=f"Tabular Feature Importance (top {top_n})")
    ax.grid(True, alpha=GRID_ALPHA, axis="x")
    plt.tight_layout()
    _save_fig(fig, os.path.join(args.output_dir, "feature_importance.png"))
    logger.info("Tabular importance: %d features, baseline MAE=%.4f", len(ranking), baseline_mae)

    # Modality ablation: zero out each modality and measure impact
    from src.utils.embedding_aligner import AUDIO_DIM, TEXT_DIM, VISUAL_DIM

    modality_ablation = {}
    for mod_name, start, end in [("visual", 0, VISUAL_DIM), ("audio", VISUAL_DIM, VISUAL_DIM + AUDIO_DIM), ("text", VISUAL_DIM + AUDIO_DIM, VISUAL_DIM + AUDIO_DIM + TEXT_DIM)]:
        mod_maes = []
        for vid in val_ids:
            emb_v = video_embeddings.get(vid)
            if emb_v is not None:
                emb_ablated = emb_v.copy()
                emb_ablated[:, start:end] = 0.0
            else:
                emb_ablated = None
            _, yp = predict_video_multimodal(
                model,
                video_dfs[vid],
                emb_ablated,
                feature_cols,
                normalizer,
                device,
                args.window_size,
                smooth_window=0,
                apply_smoothing=False,
                time_feature_mode=args.time_features,
                ref_time_sec_max=ref_sec,
            )
            yt = video_dfs[vid]["retention"].values
            mod_maes.append(float(np.mean(np.abs(yp - yt))))
        mod_mae = float(np.mean(mod_maes))
        modality_ablation[mod_name] = {"mae_without": round(mod_mae, 4), "mae_increase": round(mod_mae - baseline_mae, 4)}
        logger.info("Ablation %s: MAE=%.4f (baseline=%.4f, delta=+%.4f)", mod_name, mod_mae, baseline_mae, mod_mae - baseline_mae)

    # Also ablate all embeddings (tabular only)
    tab_only_maes = []
    for vid in val_ids:
        _, yp = predict_video_multimodal(
            model,
            video_dfs[vid],
            None,
            feature_cols,
            normalizer,
            device,
            args.window_size,
            smooth_window=0,
            apply_smoothing=False,
            time_feature_mode=args.time_features,
            ref_time_sec_max=ref_sec,
        )
        yt = video_dfs[vid]["retention"].values
        tab_only_maes.append(float(np.mean(np.abs(yp - yt))))
    tab_only_mae = float(np.mean(tab_only_maes))
    modality_ablation["all_embeddings_zeroed"] = {"mae_without": round(tab_only_mae, 4), "mae_increase": round(tab_only_mae - baseline_mae, 4)}
    logger.info("Ablation all embeddings: MAE=%.4f (delta=+%.4f)", tab_only_mae, tab_only_mae - baseline_mae)

    # Ablation bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    mods = list(modality_ablation.keys())
    deltas = [modality_ablation[m]["mae_increase"] for m in mods]
    colors = [C_ORANGE if d > 0 else C_GREEN for d in deltas]
    ax.barh(mods[::-1], deltas[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.5)
    ax.set(xlabel="MAE increase when zeroed", title="Modality Ablation Importance")
    ax.grid(True, alpha=GRID_ALPHA, axis="x")
    plt.tight_layout()
    _save_fig(fig, os.path.join(args.output_dir, "modality_ablation.png"))

    pd.DataFrame([{"modality": k, **v} for k, v in modality_ablation.items()]).to_csv(os.path.join(args.output_dir, "modality_ablation.csv"), index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": args.arch,
            "n_tabular_features": n_tab,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "feature_cols": feature_cols,
            "normalizer_median": normalizer.median.tolist(),
            "normalizer_iqr": normalizer.iqr.tolist(),
            "ret_min": normalizer.ret_min,
            "ret_max": normalizer.ret_max,
        },
        os.path.join(args.output_dir, f"multimodal_{args.arch}_model.pt"),
    )

    Path(os.path.join(args.output_dir, "metrics.json")).write_text(
        json.dumps(
            {
                "model": f"Multimodal{args.arch.title()}",
                "arch": args.arch,
                "n_tabular_features": n_tab,
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
