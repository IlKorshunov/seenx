"""
BERT-based retention prediction trainer.

Three modes:
    extract  — Extract per-second BERT embeddings from transcripts, then train
               with existing multimodal pipeline (replaces USER2 text branch)
    e2e      — End-to-end LoRA fine-tuning of BERT + temporal regression head
               (text-only model: predicts retention purely from transcript)
    hybrid   — BERT LoRA text branch fused with visual/audio/tabular features

Usage:
    python train/train_bert_seq.py --mode extract --output-dir bert_exp/extract
    python train/train_bert_seq.py --mode e2e --lora-rank 8 --output-dir bert_exp/e2e
    python train/train_bert_seq.py --mode hybrid --output-dir bert_exp/hybrid

Attention visualization (Temporal MHA in the fusion / regression head):
    Add --attention-plots to save heatmaps under {output_dir}/attention_viz/attention/<video_id>/
    For extract/hybrid: runs full predict_video_multimodal inside AttentionCapture.
    For e2e: first window (window-size sec) of BERT embeddings only.
    Standalone (extract/hybrid checkpoints only): load bert_model.pt with multimodal_transformer
    if emb_text_dim matches (see attention_viz --model-type multimodal_transformer).
"""

from __future__ import annotations

import os as _os


_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
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
    seq_metrics,
    time_feature_extra_dim,
)
from train.common.composite_trainer import lr_warmup_cosine as _lr_lambda_warmup_cosine, to_device_batch as _to_device
from train.common.retention_plots import plot_prediction, plot_training_curve
from train.common.split_utils import apply_train_id_file_filter, resolve_train_val_split


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BERT retention predictor.")
    a = p.add_argument

    a("--mode", choices=["extract", "e2e", "hybrid"], default="extract", help="extract: frozen BERT embeddings; e2e: LoRA text-only; hybrid: LoRA + multimodal fusion")
    a("--backbone", default="deberta-base", help="Backbone model name or HF ID")
    a("--lora-rank", type=int, default=8)
    a("--lora-alpha", type=int, default=16)

    a("--output-dir-features", default="output")
    a("--snapshot-dir", default="data")
    a("--embeddings-root", default="embeddings")
    a("--output-dir", default="bert_exp/latest")
    a("--use-curve-raw", action="store_true", default=True)
    a("--no-use-curve-raw", dest="use_curve_raw", action="store_false")

    a("--val-ratio", type=float, default=0.15)
    a("--val-first-n-output", type=int, default=0)
    a("--eval-video", default="")
    a("--train-video-ids-file", default="")
    a("--top-k-features", type=int, default=0)

    a("--alpha-corr", type=float, default=0.3)
    a("--alpha-smooth", type=float, default=0.15)
    a("--alpha-delta", type=float, default=0.4)
    a("--alpha-mono", type=float, default=0.03)
    a("--ad-penalty-weight", type=float, default=15.0)
    a("--start-boost-secs", type=int, default=15)
    a("--start-boost-factor", type=float, default=2.0)
    a("--engagement-weight", action="store_true", default=True)
    a("--no-engagement-weight", dest="engagement_weight", action="store_false")

    a("--window-size", type=int, default=128)
    a("--window-stride", type=int, default=64)
    a("--d-model", type=int, default=256)
    a("--n-heads", type=int, default=4)
    a("--n-layers", type=int, default=4)
    a("--d-ff", type=int, default=512)
    a("--dropout", type=float, default=0.2)

    a("--epochs", type=int, default=200)
    a("--batch-size", type=int, default=8)
    a("--lr", type=float, default=3e-4)
    a("--weight-decay", type=float, default=1e-3)
    a("--patience", type=int, default=30)
    a("--grad-clip", type=float, default=1.0)
    a("--warmup-epochs", type=int, default=10)

    a("--smooth-window", type=int, default=7)
    a("--feature-mask-prob", type=float, default=0.1)
    a("--noise-std", type=float, default=0.02)
    a("--random-seed", type=int, default=42)
    a("--device", default="cuda")
    a("--time-features", choices=["none", "frac", "frac_sec"], default="none")
    a("--min-duration-sec", type=float, default=0)
    a("--max-duration-sec", type=float, default=0)

    a("--attention-plots", action="store_true", default=False, help="After training, save attention heatmaps (Temporal MHA) under output_dir/attention_viz/")
    a("--attention-max-videos", type=int, default=3, help="Max validation videos to run attention viz on (extract/hybrid/e2e)")

    a("--force-bert-extract", action="store_true", default=False, help="Recompute and overwrite bert_embeddings.npy even if already on disk")

    return p.parse_args()


def _ensure_bert_embeddings(video_dfs, args):
    """Extract BERT embeddings for all videos if not cached."""
    from src.extractors.text.bert_feature import extract_bert_embeddings
    from src.utils.config import Config

    config = Config("configs/local.json")
    root = os.path.normpath(os.path.abspath(args.embeddings_root))
    force = getattr(args, "force_bert_extract", False)

    for vid in video_dfs:
        video_path = os.path.join(args.snapshot_dir, vid, "video.mp4")
        if not os.path.exists(video_path):
            logger.warning("Video file not found: %s", video_path)
            continue
        cached = os.path.join(root, vid, "bert_embeddings.npy")
        if not force and os.path.isfile(cached):
            logger.info("Skipping BERT extraction for %s (file exists, not overwriting). Use --force-bert-extract to rebuild.", vid)
            continue
        try:
            extract_bert_embeddings(video_path, config, backbone=args.backbone, embeddings_root=root, force=force)
        except Exception as e:
            logger.error("Failed BERT extraction for %s: %s", vid, e)


def _load_bert_embeddings(video_dfs, embeddings_root):
    """Load cached BERT embeddings, returning dict vid -> (T, hidden_dim)."""
    result = {}
    for vid, df in video_dfs.items():
        path = os.path.join(embeddings_root, vid, "bert_embeddings.npy")
        if os.path.exists(path):
            emb = np.load(path).astype(np.float32)
            n = len(df)
            if len(emb) > n:
                emb = emb[:n]
            elif len(emb) < n:
                emb = np.vstack([emb, np.tile(emb[-1:], (n - len(emb), 1))])
            result[vid] = emb
    logger.info("Loaded BERT embeddings for %d / %d videos", len(result), len(video_dfs))
    return result


def _load_attention_viz():
    """Load attention_viz without importing src.utils (avoids whisper in __init__)."""
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    path = root / "src" / "utils" / "attention_viz.py"
    spec = importlib.util.spec_from_file_location("_attention_viz_bert", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_bert_attention_visualization(
    args,
    mode: str,
    model: nn.Module,
    video_dfs: dict,
    val_ids: list,
    feature_cols: list,
    normalizer: FeatureNormalizer,
    ref_sec: float,
    video_embeddings: dict | None = None,
    bert_embs: dict | None = None,
    emb_dim: int | None = None,
):
    if not getattr(args, "attention_plots", False) or not val_ids:
        return
    max_v = max(1, int(getattr(args, "attention_max_videos", 3)))
    targets = val_ids[:max_v]
    attn = _load_attention_viz()
    out_root = os.path.join(args.output_dir, "attention_viz")
    device = torch.device(args.device)
    model.eval()
    ws = args.window_size

    for vid in targets:
        if vid not in video_dfs:
            continue
        df = video_dfs[vid]
        retention = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values.astype(np.float32)

        with attn.AttentionCapture(model) as cap:
            if mode in ("extract", "hybrid"):
                emb_v = video_embeddings.get(vid) if video_embeddings else None
                if emb_v is None:
                    logger.warning("No multimodal embeddings for attention viz: %s", vid)
                    continue
                d = emb_dim or int(emb_v.shape[1])
                ds = MultimodalWindowedDataset(
                    {vid: df},
                    {vid: emb_v},
                    [vid],
                    feature_cols,
                    normalizer,
                    ws,
                    args.window_stride,
                    feature_mask_prob=0.0,
                    noise_std=0.0,
                    time_feature_mode=args.time_features,
                    ref_time_sec_max=ref_sec,
                    emb_dim=d,
                )
                if len(ds) == 0:
                    continue
                b0 = ds[0]
                emb_t = b0["embeddings"].unsqueeze(0).to(device).float()
                tab = b0["tabular"].unsqueeze(0).to(device).float()
                pm = b0["padding_mask"].unsqueeze(0).to(device)
                _ = model(emb_t, tabular=tab, src_key_padding_mask=pm)
                rl = int((~b0["padding_mask"]).sum().item())
                ret_plot = retention[:rl]
                if ret_plot.size < ws:
                    ret_plot = np.pad(ret_plot, (0, ws - ret_plot.size), mode="edge")
                else:
                    ret_plot = ret_plot[:ws]
            else:
                bert = bert_embs.get(vid) if bert_embs else None
                if bert is None:
                    logger.warning("No BERT embeddings for attention viz: %s", vid)
                    continue
                n = len(df)
                w = min(n, ws)
                chunk = bert[:w].astype(np.float32)
                if w < ws:
                    chunk = np.pad(chunk, ((0, ws - w), (0, 0)), mode="edge")
                emb_t = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)
                pad_mask = torch.zeros(1, ws, dtype=torch.bool, device=device)
                if w < ws:
                    pad_mask[0, w:] = True
                _ = model(emb_t, padding_mask=pad_mask)
                ret_plot = np.pad(retention[:w], (0, ws - w), mode="edge")

        weights = cap.get_weights_numpy()
        if not weights:
            logger.warning("No attention weights captured for %s (mode=%s)", vid, mode)
            continue
        attn.visualize_all(weights, ret_plot, vid, out_root)
        logger.info("Attention viz saved: %s", os.path.join(out_root, "attention", vid))


def _split_train_val(args, video_ids, video_dfs):
    output_dir = Path(args.output_dir_features)
    output_video_ids = sorted(
        p.name.replace("_features.csv", "") for p in output_dir.glob("*_features.csv") if not p.name.endswith(".partial") and p.name.replace("_features.csv", "") in video_dfs
    )
    train_ids, val_ids = resolve_train_val_split(args, video_ids, output_video_ids)
    return apply_train_id_file_filter(train_ids, args), val_ids


def _compute_baseline(video_dfs, train_ids, normalizer):
    max_len = max(len(video_dfs[v]) for v in train_ids)
    acc = np.zeros(max_len, dtype=np.float64)
    cnt = np.zeros(max_len, dtype=np.float64)
    for v in train_ids:
        ret = pd.to_numeric(video_dfs[v]["retention"], errors="coerce").fillna(0).values
        acc[: len(ret)] += ret
        cnt[: len(ret)] += 1.0
    baseline = (acc / np.maximum(cnt, 1.0)).astype(np.float32)
    return normalizer.normalize_retention(baseline).astype(np.float32)


def _training_loop(model, train_dl, val_dl, args, model_name="BERT"):
    """Shared training loop for extract/e2e/hybrid modes."""
    device = torch.device(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: _lr_lambda_warmup_cosine(ep, args.warmup_epochs, args.epochs))

    best_val, no_improve, best_state = float("inf"), 0, {}
    train_losses, val_losses = [], []

    for epoch in (pbar := tqdm(range(1, args.epochs + 1), desc=f"{model_name} Training")):
        model.train()
        tl, tn = 0.0, 0
        for batch in train_dl:
            feat, tgt, pad_mask, ad_mask, spike_triggers = _to_device(batch, device, "tabular", "retention", "padding_mask", "is_ad", "spike_triggers")
            emb = batch["embeddings"].to(device)
            vw = batch["video_weight"].to(device) if args.engagement_weight and "video_weight" in batch else None
            pred = model(emb, tabular=feat, src_key_padding_mask=pad_mask)
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
        scheduler.step()
        train_losses.append(tl / max(tn, 1))

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                feat, tgt, pad_mask, ad_mask, spike_triggers = _to_device(batch, device, "tabular", "retention", "padding_mask", "is_ad", "spike_triggers")
                emb = batch["embeddings"].to(device)
                pred = model(emb, tabular=feat, src_key_padding_mask=pad_mask)
                loss = composite_loss(pred, tgt, ad_mask, spike_triggers, pad_mask, 1.0, None, args.alpha_corr, 0.0, 0.0, 0, 1.0, args.alpha_delta)
                nv = (~pad_mask).sum().item()
                vl += loss.item() * nv
                vn += nv
        val_losses.append(vl / max(vn, 1))

        pbar.set_postfix(train=f"{train_losses[-1]:.4f}", val=f"{val_losses[-1]:.4f}")
        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d  train=%.4f  val=%.4f", epoch, args.epochs, train_losses[-1], val_losses[-1])

        if val_losses[-1] < best_val:
            best_val, no_improve = val_losses[-1], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stop at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    model = model.to(device)
    return model, best_state, best_val, epoch, train_losses, val_losses


def train_extract_mode(args):
    """Mode 'extract': frozen BERT embeddings replace USER2 text in multimodal pipeline."""
    device = torch.device(args.device)

    video_dfs = load_all_merged(
        args.output_dir_features,
        args.snapshot_dir,
        use_curve_raw=args.use_curve_raw,
        emb_pca_components=0,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )

    _ensure_bert_embeddings(video_dfs, args)
    bert_embs = _load_bert_embeddings(video_dfs, args.embeddings_root)

    aligned_embs = load_aligned_embeddings_for_videos(video_dfs, args.embeddings_root)

    video_embeddings = {}
    for vid in video_dfs:
        aligned = aligned_embs.get(vid)
        bert = bert_embs.get(vid)
        if aligned is not None and bert is not None:
            n = min(len(aligned), len(bert), len(video_dfs[vid]))
            vis_aud = aligned[:n, : 768 + 512]
            bert_part = bert[:n]
            video_embeddings[vid] = np.concatenate([vis_aud, bert_part], axis=1)
        elif aligned is not None:
            video_embeddings[vid] = aligned
        elif bert is not None:
            video_embeddings[vid] = bert

    video_ids = sorted(video_dfs.keys())
    feature_cols, filter_log = filter_features(video_dfs, top_k=args.top_k_features or None)
    Path(os.path.join(args.output_dir, "feature_filter_log.txt")).write_text("\n".join(filter_log), encoding="utf-8")

    train_ids, val_ids = _split_train_val(args, video_ids, video_dfs)
    logger.info("Train: %d videos, Val: %d videos", len(train_ids), len(val_ids))

    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)
    ref_sec = max_time_sec_over_videos(video_dfs, train_ids)
    video_weights = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None

    n_tab = len(feature_cols) + time_feature_extra_dim(args.time_features)
    emb_dim = next(iter(video_embeddings.values())).shape[1] if video_embeddings else 1536
    bert_hidden = next(iter(bert_embs.values())).shape[1] if bert_embs else 768
    vis_dim = 768
    aud_dim = 512
    txt_dim = emb_dim - vis_dim - aud_dim if emb_dim > vis_dim + aud_dim else bert_hidden

    from src.models.retention_multimodal_transformer import MultimodalRetentionTransformer

    model = MultimodalRetentionTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        n_tabular_features=n_tab,
        emb_visual_dim=vis_dim,
        emb_audio_dim=aud_dim,
        emb_text_dim=max(txt_dim, 1),
    ).to(device)

    baseline_norm = _compute_baseline(video_dfs, train_ids, normalizer)
    model.set_baseline(torch.tensor(baseline_norm))

    train_ds = MultimodalWindowedDataset(
        video_dfs,
        video_embeddings,
        train_ids,
        feature_cols,
        normalizer,
        args.window_size,
        args.window_stride,
        video_weights=video_weights,
        emb_dim=emb_dim,
        feature_mask_prob=args.feature_mask_prob,
        noise_std=args.noise_std,
        time_feature_mode=args.time_features,
        ref_time_sec_max=ref_sec,
    )
    val_ds = MultimodalWindowedDataset(
        video_dfs,
        video_embeddings,
        val_ids,
        feature_cols,
        normalizer,
        args.window_size,
        args.window_stride,
        emb_dim=emb_dim,
        time_feature_mode=args.time_features,
        ref_time_sec_max=ref_sec,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    logger.info(
        "Model params: %d (%.1fK), emb_dim=%d (vis=%d aud=%d txt=%d)",
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters()) / 1000,
        emb_dim,
        vis_dim,
        aud_dim,
        txt_dim,
    )

    model, best_state, best_val, last_epoch, train_losses, val_losses = _training_loop(model, train_dl, val_dl, args, model_name="BERT-extract")

    plot_training_curve(train_losses, val_losses, os.path.join(args.output_dir, "training_curve.png"))

    all_metrics = {}
    for vid in video_ids:
        split = "val" if vid in val_ids else "train"
        emb_v = video_embeddings.get(vid)
        y_true, y_pred = predict_video_multimodal(
            model,
            video_dfs[vid],
            emb_v,
            feature_cols,
            normalizer,
            device,
            args.window_size,
            smooth_window=args.smooth_window,
            time_feature_mode=args.time_features,
            ref_time_sec_max=ref_sec,
        )
        m = seq_metrics(y_pred, y_true)
        all_metrics[vid] = {**m, "split": split, "n_seconds": len(y_true)}
        logger.info("%s [%s]  MAE=%.4f  r=%.3f", vid, split, m["mae"], m["pearson"])
        is_ad = video_dfs[vid]["is_ad"].values if "is_ad" in video_dfs[vid].columns else None
        plot_prediction(vid, y_true, y_pred, is_ad, split, m, os.path.join(args.output_dir, "videos", vid, "prediction.png"))

    plot_mae_summary(all_metrics, args.output_dir, model_name="BERT-extract")

    run_bert_attention_visualization(args, "extract", model, video_dfs, val_ids, feature_cols, normalizer, ref_sec, video_embeddings=video_embeddings, emb_dim=emb_dim)

    torch.save({"model_state_dict": best_state, "feature_cols": feature_cols, "mode": "extract", "backbone": args.backbone}, os.path.join(args.output_dir, "bert_model.pt"))

    Path(os.path.join(args.output_dir, "metrics.json")).write_text(
        json.dumps(
            {
                "model": "BERT-extract",
                "backbone": args.backbone,
                "per_video": all_metrics,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "best_val_loss": round(best_val, 6),
                "epochs_trained": last_epoch,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Done (extract mode). Best val=%.4f epochs=%d", best_val, last_epoch)
    return all_metrics


def train_e2e_mode(args):
    """Mode 'e2e': text-only retention prediction with BERT + LoRA."""
    device = torch.device(args.device)
    logger.info("E2E mode: text-only BERT retention, backbone=%s lora_rank=%d", args.backbone, args.lora_rank)

    video_dfs = load_all_merged(
        args.output_dir_features,
        args.snapshot_dir,
        use_curve_raw=args.use_curve_raw,
        emb_pca_components=0,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )

    _ensure_bert_embeddings(video_dfs, args)
    bert_embs = _load_bert_embeddings(video_dfs, args.embeddings_root)

    video_ids = sorted(video_dfs.keys())
    feature_cols, filter_log = filter_features(video_dfs, top_k=args.top_k_features or None)
    Path(os.path.join(args.output_dir, "feature_filter_log.txt")).write_text("\n".join(filter_log), encoding="utf-8")

    train_ids, val_ids = _split_train_val(args, video_ids, video_dfs)
    logger.info("Train: %d, Val: %d", len(train_ids), len(val_ids))

    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)
    ref_sec = max_time_sec_over_videos(video_dfs, train_ids)
    video_weights = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None

    bert_hidden = next(iter(bert_embs.values())).shape[1] if bert_embs else 768

    from src.models.bert_retention import BERTRetention

    model = BERTRetention(backbone=args.backbone, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, n_head_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout).to(
        device
    )

    baseline_norm = _compute_baseline(video_dfs, train_ids, normalizer)
    model.set_baseline(torch.tensor(baseline_norm))

    train_ds = MultimodalWindowedDataset(
        video_dfs,
        bert_embs,
        train_ids,
        feature_cols,
        normalizer,
        args.window_size,
        args.window_stride,
        video_weights=video_weights,
        emb_dim=bert_hidden,
        feature_mask_prob=args.feature_mask_prob,
        noise_std=args.noise_std,
        time_feature_mode=args.time_features,
        ref_time_sec_max=ref_sec,
    )
    val_ds = MultimodalWindowedDataset(
        video_dfs,
        bert_embs,
        val_ids,
        feature_cols,
        normalizer,
        args.window_size,
        args.window_stride,
        emb_dim=bert_hidden,
        time_feature_mode=args.time_features,
        ref_time_sec_max=ref_sec,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    trainable = sum(p.numel() for p in model.trainable_parameters())
    total = sum(p.numel() for p in model.parameters())
    logger.info("E2E params: %d trainable / %d total (%.1f%%)", trainable, total, 100 * trainable / total)

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: _lr_lambda_warmup_cosine(ep, args.warmup_epochs, args.epochs))

    best_val, no_improve, best_state = float("inf"), 0, {}
    train_losses, val_losses = [], []

    for epoch in (pbar := tqdm(range(1, args.epochs + 1), desc="BERT-e2e")):
        model.train()
        tl, tn = 0.0, 0
        for batch in train_dl:
            tgt, pad_mask, ad_mask, spike_triggers = _to_device(batch, device, "retention", "padding_mask", "is_ad", "spike_triggers")
            emb = batch["embeddings"].to(device)
            vw = batch["video_weight"].to(device) if args.engagement_weight and "video_weight" in batch else None
            pred = model(emb, padding_mask=pad_mask)
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
        scheduler.step()
        train_losses.append(tl / max(tn, 1))

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                tgt, pad_mask, ad_mask, spike_triggers = _to_device(batch, device, "retention", "padding_mask", "is_ad", "spike_triggers")
                emb = batch["embeddings"].to(device)
                pred = model(emb, padding_mask=pad_mask)
                loss = composite_loss(pred, tgt, ad_mask, spike_triggers, pad_mask, 1.0, None, args.alpha_corr, 0.0, 0.0, 0, 1.0, args.alpha_delta)
                nv = (~pad_mask).sum().item()
                vl += loss.item() * nv
                vn += nv
        val_losses.append(vl / max(vn, 1))

        pbar.set_postfix(train=f"{train_losses[-1]:.4f}", val=f"{val_losses[-1]:.4f}")
        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d  train=%.4f  val=%.4f", epoch, args.epochs, train_losses[-1], val_losses[-1])

        if val_losses[-1] < best_val:
            best_val, no_improve = val_losses[-1], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stop at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    model = model.to(device)
    plot_training_curve(train_losses, val_losses, os.path.join(args.output_dir, "training_curve.png"))

    all_metrics = {}
    for vid in video_ids:
        split = "val" if vid in val_ids else "train"
        bert = bert_embs.get(vid)
        if bert is None:
            continue
        df = video_dfs[vid]
        n = len(df)
        emb = bert[:n] if len(bert) >= n else np.vstack([bert, np.tile(bert[-1:], (n - len(bert), 1))])
        emb_t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            pred = model(emb_t).squeeze(0).cpu().numpy()
        if normalizer is not None:
            pred = normalizer.denormalize_retention(pred)
        y_true = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values[:n]
        y_pred = pred[: len(y_true)]

        m = seq_metrics(y_pred, y_true)
        all_metrics[vid] = {**m, "split": split, "n_seconds": len(y_true)}
        logger.info("%s [%s]  MAE=%.4f  r=%.3f", vid, split, m["mae"], m["pearson"])
        is_ad = df["is_ad"].values if "is_ad" in df.columns else None
        plot_prediction(vid, y_true, y_pred, is_ad, split, m, os.path.join(args.output_dir, "videos", vid, "prediction.png"))

    plot_mae_summary(all_metrics, args.output_dir, model_name="BERT-e2e")

    run_bert_attention_visualization(args, "e2e", model, video_dfs, val_ids, feature_cols, normalizer, ref_sec, bert_embs=bert_embs)

    torch.save(
        {"model_state_dict": best_state, "feature_cols": feature_cols, "mode": "e2e", "backbone": args.backbone, "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha},
        os.path.join(args.output_dir, "bert_model.pt"),
    )

    Path(os.path.join(args.output_dir, "metrics.json")).write_text(
        json.dumps(
            {
                "model": "BERT-e2e",
                "backbone": args.backbone,
                "per_video": all_metrics,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "best_val_loss": round(best_val, 6),
                "epochs_trained": epoch,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Done (e2e). Best val=%.4f epochs=%d", best_val, epoch)
    return all_metrics


def train_hybrid_mode(args):
    """Mode 'hybrid': BERT embeddings + visual/audio/tabular fusion."""
    device = torch.device(args.device)
    logger.info("Hybrid mode: BERT text + visual/audio/tabular fusion")

    video_dfs = load_all_merged(
        args.output_dir_features,
        args.snapshot_dir,
        use_curve_raw=args.use_curve_raw,
        emb_pca_components=0,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )

    _ensure_bert_embeddings(video_dfs, args)
    bert_embs = _load_bert_embeddings(video_dfs, args.embeddings_root)
    aligned_embs = load_aligned_embeddings_for_videos(video_dfs, args.embeddings_root)

    video_embeddings = {}
    for vid in video_dfs:
        bert = bert_embs.get(vid)
        aligned = aligned_embs.get(vid)
        if bert is not None and aligned is not None:
            n = min(len(bert), len(aligned), len(video_dfs[vid]))
            vis_aud = aligned[:n, : 768 + 512]
            video_embeddings[vid] = np.concatenate([vis_aud, bert[:n]], axis=1)
        elif aligned is not None:
            video_embeddings[vid] = aligned
        elif bert is not None:
            video_embeddings[vid] = bert

    video_ids = sorted(video_dfs.keys())
    feature_cols, filter_log = filter_features(video_dfs, top_k=args.top_k_features or None)
    Path(os.path.join(args.output_dir, "feature_filter_log.txt")).write_text("\n".join(filter_log), encoding="utf-8")

    train_ids, val_ids = _split_train_val(args, video_ids, video_dfs)
    logger.info("Train: %d, Val: %d", len(train_ids), len(val_ids))

    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)
    ref_sec = max_time_sec_over_videos(video_dfs, train_ids)
    video_weights = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None

    n_tab = len(feature_cols) + time_feature_extra_dim(args.time_features)
    emb_dim = next(iter(video_embeddings.values())).shape[1] if video_embeddings else 1536
    bert_hidden = next(iter(bert_embs.values())).shape[1] if bert_embs else 768

    from src.models.retention_multimodal_transformer import MultimodalRetentionTransformer

    model = MultimodalRetentionTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        n_tabular_features=n_tab,
        emb_visual_dim=768,
        emb_audio_dim=512,
        emb_text_dim=bert_hidden,
    ).to(device)

    baseline_norm = _compute_baseline(video_dfs, train_ids, normalizer)
    model.set_baseline(torch.tensor(baseline_norm))

    train_ds = MultimodalWindowedDataset(
        video_dfs,
        video_embeddings,
        train_ids,
        feature_cols,
        normalizer,
        args.window_size,
        args.window_stride,
        video_weights=video_weights,
        emb_dim=emb_dim,
        feature_mask_prob=args.feature_mask_prob,
        noise_std=args.noise_std,
        time_feature_mode=args.time_features,
        ref_time_sec_max=ref_sec,
    )
    val_ds = MultimodalWindowedDataset(
        video_dfs,
        video_embeddings,
        val_ids,
        feature_cols,
        normalizer,
        args.window_size,
        args.window_stride,
        emb_dim=emb_dim,
        time_feature_mode=args.time_features,
        ref_time_sec_max=ref_sec,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    logger.info(
        "Hybrid params: %d (%.1fK), emb_dim=%d (vis=768 aud=512 bert=%d)",
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters()) / 1000,
        emb_dim,
        bert_hidden,
    )

    model, best_state, best_val, last_epoch, train_losses, val_losses = _training_loop(model, train_dl, val_dl, args, model_name="BERT-hybrid")

    plot_training_curve(train_losses, val_losses, os.path.join(args.output_dir, "training_curve.png"))

    all_metrics = {}
    for vid in video_ids:
        split = "val" if vid in val_ids else "train"
        emb_v = video_embeddings.get(vid)
        y_true, y_pred = predict_video_multimodal(
            model,
            video_dfs[vid],
            emb_v,
            feature_cols,
            normalizer,
            device,
            args.window_size,
            smooth_window=args.smooth_window,
            time_feature_mode=args.time_features,
            ref_time_sec_max=ref_sec,
        )
        m = seq_metrics(y_pred, y_true)
        all_metrics[vid] = {**m, "split": split, "n_seconds": len(y_true)}
        logger.info("%s [%s]  MAE=%.4f  r=%.3f", vid, split, m["mae"], m["pearson"])
        is_ad = video_dfs[vid]["is_ad"].values if "is_ad" in video_dfs[vid].columns else None
        plot_prediction(vid, y_true, y_pred, is_ad, split, m, os.path.join(args.output_dir, "videos", vid, "prediction.png"))

    plot_mae_summary(all_metrics, args.output_dir, model_name="BERT-hybrid")

    run_bert_attention_visualization(args, "hybrid", model, video_dfs, val_ids, feature_cols, normalizer, ref_sec, video_embeddings=video_embeddings, emb_dim=emb_dim)

    torch.save({"model_state_dict": best_state, "feature_cols": feature_cols, "mode": "hybrid", "backbone": args.backbone}, os.path.join(args.output_dir, "bert_model.pt"))

    Path(os.path.join(args.output_dir, "metrics.json")).write_text(
        json.dumps(
            {
                "model": "BERT-hybrid",
                "backbone": args.backbone,
                "per_video": all_metrics,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "best_val_loss": round(best_val, 6),
                "epochs_trained": last_epoch,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Done (hybrid). Best val=%.4f epochs=%d", best_val, last_epoch)
    return all_metrics


def main():
    args = parse_args()
    args.embeddings_root = os.path.normpath(os.path.abspath(args.embeddings_root))
    args.snapshot_dir = os.path.normpath(os.path.abspath(args.snapshot_dir))
    args.output_dir_features = os.path.normpath(os.path.abspath(args.output_dir_features))
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "extract":
        train_extract_mode(args)
    elif args.mode == "e2e":
        train_e2e_mode(args)
    elif args.mode == "hybrid":
        train_hybrid_mode(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
