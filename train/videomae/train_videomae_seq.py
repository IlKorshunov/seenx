"""
VideoMAE-based retention prediction trainer.

Three modes:
    extract  — Extract and cache VideoMAE per-second embeddings, then train
               with existing multimodal pipeline (replaces CLIP visual branch)
    e2e      — End-to-end LoRA on raw frames (same metrics/plots as extract/hybrid)
    hybrid   — LoRA-tuned VideoMAE fused with tabular/audio/text features

Usage:
    python train/train_videomae_seq.py --mode extract --output-dir videomae_exp/extract
    python train/train_videomae_seq.py --mode e2e --lora-rank 8 --output-dir videomae_exp/e2e
    python train/train_videomae_seq.py --mode hybrid --output-dir videomae_exp/hybrid
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
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
    smooth_predictions,
    time_feature_extra_dim,
)
from train.common.composite_trainer import lr_warmup_cosine as _lr_lambda_warmup_cosine, to_device_batch as _to_device
from train.common.retention_plots import plot_prediction, plot_training_curve
from train.common.split_utils import apply_train_id_file_filter, resolve_train_val_split


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VideoMAE retention predictor.")
    a = p.add_argument

    a("--mode", choices=["extract", "e2e", "hybrid"], default="extract", help="extract: use VideoMAE as frozen feature extractor; e2e: end-to-end LoRA; hybrid: LoRA + fusion")
    a("--backbone", default="videomae-base", help="Backbone model name or HF ID")
    a("--lora-rank", type=int, default=8)
    a("--lora-alpha", type=int, default=16)
    a("--clip-stride", type=int, default=4, help="Stride in seconds for VideoMAE sliding window")

    a("--output-dir-features", default="output")
    a("--snapshot-dir", default="data")
    a("--embeddings-root", default="embeddings")
    a("--output-dir", default="videomae_exp/latest")
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
    a("--global-calibration", action="store_true", default=False)

    return p.parse_args()


def _split_ids(args, video_ids: list[str], output_video_ids: list[str]) -> tuple[list[str], list[str]]:
    train_ids, val_ids = resolve_train_val_split(args, video_ids, output_video_ids)
    return apply_train_id_file_filter(train_ids, args), val_ids


def _ensure_videomae_embeddings(video_dfs, args):
    """Extract VideoMAE embeddings for all videos if not cached."""
    from src.extractors.video.videomae_feature import extract_videomae_embeddings
    from src.utils.config import Config

    config = Config("configs/local.json")
    for vid in video_dfs:
        video_path = os.path.join(args.snapshot_dir, vid, "video.mp4")
        if not os.path.exists(video_path):
            logger.warning("Video file not found: %s", video_path)
            continue
        try:
            extract_videomae_embeddings(video_path, config, backbone=args.backbone, clip_stride=args.clip_stride, embeddings_root=args.embeddings_root)
        except Exception as e:
            logger.error("Failed VideoMAE extraction for %s: %s", vid, e)


def _load_videomae_embeddings(video_dfs, embeddings_root):
    """Load cached VideoMAE embeddings, returning dict vid -> (T, hidden_dim)."""
    result = {}
    for vid, df in video_dfs.items():
        path = os.path.join(embeddings_root, vid, "videomae_embeddings.npy")
        if os.path.exists(path):
            emb = np.load(path).astype(np.float32)
            n = len(df)
            if len(emb) > n:
                emb = emb[:n]
            elif len(emb) < n:
                emb = np.vstack([emb, np.tile(emb[-1:], (n - len(emb), 1))])
            result[vid] = emb
    logger.info("Loaded VideoMAE embeddings for %d / %d videos", len(result), len(video_dfs))
    return result


def _read_rgb_frames_1fps_window(video_path: str, start_sec: int, n_out: int) -> list[np.ndarray]:
    """Read n_out consecutive 1 Hz RGB frames starting at second start_sec; pad with last frame if EOF."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        z = np.zeros((360, 640, 3), dtype=np.uint8)
        return [z.copy() for _ in range(n_out)]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps)))
    out: list[np.ndarray] = []
    video_idx = 0
    sec = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if video_idx % step == 0:
            sec += 1
            if sec < start_sec:
                pass
            elif len(out) < n_out:
                out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if len(out) >= n_out:
                    break
        video_idx += 1
    cap.release()
    while len(out) < n_out:
        fill = out[-1].copy() if out else np.zeros((360, 640, 3), dtype=np.uint8)
        out.append(fill)
    return out[:n_out]


def _frames_to_videomae_inputs(frames_ws: list[np.ndarray], processor, num_frames: int) -> torch.Tensor:
    """Uniform subsample window to num_frames RGB uint8 arrays → processor batch (num_frames, C, H, W)."""
    w = len(frames_ws)
    if w == 0:
        raise ValueError("empty frame list")
    idx = np.linspace(0, w - 1, num_frames).astype(int)
    sub = [frames_ws[i] for i in idx]
    batch = processor(sub, return_tensors="pt")
    return batch["pixel_values"].squeeze(0)


class VideoMAEE2EWindowDataset(Dataset):
    """Sliding windows over videos; loads raw 1 fps RGB, subsamples to VideoMAE input length per sample."""

    def __init__(
        self,
        video_dfs: dict,
        video_ids: list[str],
        snapshot_dir: str,
        window_size: int,
        stride: int,
        normalizer: FeatureNormalizer,
        videomae_num_frames: int,
        processor,
        video_weights: dict | None = None,
    ):
        self.video_dfs = video_dfs
        self.snapshot_dir = snapshot_dir
        self.window_size = window_size
        self.stride = stride
        self.normalizer = normalizer
        self.videomae_num_frames = videomae_num_frames
        self.processor = processor
        self.windows: list[tuple[str, int, int, float, int]] = []

        for vid in video_ids:
            df = video_dfs[vid]
            w = float(video_weights[vid]) if video_weights and vid in video_weights else 1.0
            n = len(df)
            if n <= window_size:
                self.windows.append((vid, 0, n, w, n))
            else:
                for s in range(0, n - window_size + 1, stride):
                    self.windows.append((vid, s, window_size, w, n))
                if (n - window_size) % stride != 0:
                    s = n - window_size
                    self.windows.append((vid, s, window_size, w, n))

        self._y_cache: dict = {}
        self._ad_cache: dict = {}
        for vid in video_ids:
            df = video_dfs[vid]
            y = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values.astype(np.float32)
            y = self.normalizer.normalize_retention(y).astype(np.float32)
            self._y_cache[vid] = y
            self._ad_cache[vid] = df["is_ad"].values.astype(np.float32) if "is_ad" in df.columns else np.zeros(len(df), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        vid, start, real_len, weight, n_full = self.windows[idx]
        ws = self.window_size
        path = os.path.join(self.snapshot_dir, vid, "video.mp4")
        frames = _read_rgb_frames_1fps_window(path, start, ws)
        pixel_values = _frames_to_videomae_inputs(frames, self.processor, self.videomae_num_frames)

        y = self._y_cache[vid][start : start + real_len]
        is_ad = self._ad_cache[vid][start : start + real_len]
        if len(y) < ws:
            pad = ws - len(y)
            y = np.pad(y, (0, pad))
            is_ad = np.pad(is_ad, (0, pad))
            mask = np.array([False] * real_len + [True] * pad)
        else:
            mask = np.zeros(ws, dtype=bool)

        return {
            "pixel_values": pixel_values,
            "retention": torch.from_numpy(y.copy()),
            "is_ad": torch.from_numpy(is_ad.copy()),
            "padding_mask": torch.from_numpy(mask),
            "video_weight": torch.tensor(weight, dtype=torch.float32),
        }


@torch.no_grad()
def predict_video_videomae_e2e(
    model, df: pd.DataFrame, video_path: str, normalizer: FeatureNormalizer, device: torch.device, window_size: int, stride: int, videomae_num_frames: int, smooth_window: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    from src.extractors.video.videomae_feature import _extract_frames_1fps

    model.eval()
    y_true = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values.astype(np.float64)
    n_full = len(df)
    frames_full = _extract_frames_1fps(video_path)
    if not frames_full:
        return y_true, np.zeros(n_full, dtype=np.float64)

    if len(frames_full) < n_full:
        frames_full = frames_full + [frames_full[-1]] * (n_full - len(frames_full))
    elif len(frames_full) > n_full:
        frames_full = frames_full[:n_full]

    processor = model.processor

    if n_full <= window_size:
        seg = frames_full + [frames_full[-1]] * (window_size - len(frames_full))
        seg = seg[:window_size]
        pv = _frames_to_videomae_inputs(seg, processor, videomae_num_frames).unsqueeze(0).to(device)
        pad = torch.zeros(1, window_size, dtype=torch.bool, device=device)
        if n_full < window_size:
            pad[0, n_full:] = True
        pred = model(pv, n_seconds=window_size, padding_mask=pad).squeeze(0).cpu().numpy()[:n_full]
        if normalizer is not None:
            pred = normalizer.denormalize_retention(pred)
        if smooth_window > 1:
            pred = smooth_predictions(pred, window=smooth_window)
        return y_true, pred

    pred_sum = np.zeros(n_full)
    pred_cnt = np.zeros(n_full)
    for s in range(0, n_full - window_size + 1, stride):
        seg = frames_full[s : s + window_size]
        real_len = len(seg)
        if real_len < window_size:
            seg = seg + [seg[-1]] * (window_size - real_len)
        pv = _frames_to_videomae_inputs(seg, processor, videomae_num_frames).unsqueeze(0).to(device)
        pad = torch.zeros(1, window_size, dtype=torch.bool, device=device)
        if real_len < window_size:
            pad[0, real_len:] = True
        p = model(pv, n_seconds=window_size, padding_mask=pad).squeeze(0).cpu().numpy()
        pred_sum[s : s + window_size] += p
        pred_cnt[s : s + window_size] += 1.0
    if (n_full - window_size) % stride != 0:
        s = n_full - window_size
        seg = frames_full[s:]
        real_len = len(seg)
        if real_len < window_size:
            seg = seg + [seg[-1]] * (window_size - real_len)
        pv = _frames_to_videomae_inputs(seg, processor, videomae_num_frames).unsqueeze(0).to(device)
        pad = torch.zeros(1, window_size, dtype=torch.bool, device=device)
        if real_len < window_size:
            pad[0, real_len:] = True
        p = model(pv, n_seconds=window_size, padding_mask=pad).squeeze(0).cpu().numpy()
        pred_sum[s : s + window_size] += p
        pred_cnt[s : s + window_size] += 1.0

    pred = pred_sum / np.maximum(pred_cnt, 1.0)
    if normalizer is not None:
        pred = normalizer.denormalize_retention(pred)
    if smooth_window > 1:
        pred = smooth_predictions(pred, window=smooth_window)
    return y_true, pred


def train_extract_mode(args):
    """Mode 'extract': use VideoMAE as frozen feature extractor, then train multimodal model."""
    device = torch.device(args.device)

    video_dfs = load_all_merged(
        args.output_dir_features,
        args.snapshot_dir,
        use_curve_raw=args.use_curve_raw,
        emb_pca_components=0,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )

    _ensure_videomae_embeddings(video_dfs, args)

    video_embeddings = load_aligned_embeddings_for_videos(video_dfs, args.embeddings_root)
    vmae_embeddings = _load_videomae_embeddings(video_dfs, args.embeddings_root)

    for vid in video_embeddings:
        if vid in vmae_embeddings:
            aligned = video_embeddings[vid]
            vmae = vmae_embeddings[vid]
            n = min(len(aligned), len(vmae))
            aligned = aligned[:n]
            vmae = vmae[:n]
            video_embeddings[vid] = np.concatenate([vmae, aligned[:, 768:]], axis=1)

    video_ids = sorted(video_dfs.keys())
    feature_cols, filter_log = filter_features(video_dfs, top_k=args.top_k_features or None)
    Path(os.path.join(args.output_dir, "feature_filter_log.txt")).write_text("\n".join(filter_log), encoding="utf-8")

    output_dir = Path(args.output_dir_features)
    output_video_ids = sorted(
        p.name.replace("_features.csv", "") for p in output_dir.glob("*_features.csv") if not p.name.endswith(".partial") and p.name.replace("_features.csv", "") in video_dfs
    )

    train_ids, val_ids = _split_ids(args, video_ids, output_video_ids)

    logger.info("Train: %d videos, Val: %d videos", len(train_ids), len(val_ids))

    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)
    ref_sec = max_time_sec_over_videos(video_dfs, train_ids)
    video_weights = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None

    n_tab = len(feature_cols) + time_feature_extra_dim(args.time_features)

    from src.models.retention_multimodal_transformer import MultimodalRetentionTransformer

    model = MultimodalRetentionTransformer(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout, n_tabular_features=n_tab).to(
        device
    )

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

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d (%.1fK)", n_params, n_params / 1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: _lr_lambda_warmup_cosine(ep, args.warmup_epochs, args.epochs))

    best_val, no_improve, best_state = float("inf"), 0, {}
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in (pbar := tqdm(range(1, args.epochs + 1), desc="Training")):
        model.train()
        tl, tn = 0.0, 0
        for batch in train_dl:
            feat, tgt, pad_mask, ad_mask, spike_triggers = _to_device(batch, device, "tabular", "retention", "padding_mask", "is_ad", "spike_triggers")
            emb = batch["embeddings"].to(device)
            vw = batch["video_weight"].to(device) if args.engagement_weight else None
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
    plot_training_curve(train_losses, val_losses, os.path.join(args.output_dir, "training_curve.png"))

    all_metrics = {}
    for vid in video_ids:
        split = "val" if vid in val_ids else "train"
        y_true, y_pred = predict_video_multimodal(
            model,
            video_dfs[vid],
            video_embeddings.get(vid),
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

    plot_mae_summary(all_metrics, args.output_dir, model_name="VideoMAE-extract")

    torch.save({"model_state_dict": best_state, "feature_cols": feature_cols, "mode": "extract", "backbone": args.backbone}, os.path.join(args.output_dir, "videomae_model.pt"))

    Path(os.path.join(args.output_dir, "metrics.json")).write_text(
        json.dumps(
            {
                "model": "VideoMAE-extract",
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

    logger.info("Done (extract mode). Best val=%.4f epochs=%d", best_val, epoch)
    return all_metrics


def train_e2e_mode(args):
    """Mode 'e2e': end-to-end VideoMAE + LoRA fine-tuning on raw frames (tabular not used)."""
    from src.models.video_mae_retention import VIDEOMAE_NUM_FRAMES, VideoMAERetention

    logger.info("End-to-end mode: backbone=%s lora_rank=%d", args.backbone, args.lora_rank)
    logger.info("E2E loads VideoMAE clips in each batch; use a small --batch-size if GPU OOM.")

    device = torch.device(args.device)

    video_dfs = load_all_merged(
        args.output_dir_features,
        args.snapshot_dir,
        use_curve_raw=args.use_curve_raw,
        emb_pca_components=0,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )

    skipped = [v for v in video_dfs if not os.path.exists(os.path.join(args.snapshot_dir, v, "video.mp4"))]
    for v in skipped:
        logger.warning("Skipping %s: no video.mp4 under snapshot", v)
        del video_dfs[v]
    if not video_dfs:
        logger.error("No videos with video.mp4 found. Check --snapshot-dir.")
        return {}

    video_ids = sorted(video_dfs.keys())
    feature_cols, filter_log = filter_features(video_dfs, top_k=args.top_k_features or None)
    Path(os.path.join(args.output_dir, "feature_filter_log.txt")).write_text("\n".join(filter_log), encoding="utf-8")

    output_dir = Path(args.output_dir_features)
    output_video_ids = sorted(
        p.name.replace("_features.csv", "") for p in output_dir.glob("*_features.csv") if not p.name.endswith(".partial") and p.name.replace("_features.csv", "") in video_dfs
    )

    train_ids, val_ids = _split_ids(args, video_ids, output_video_ids)

    logger.info("E2E train: %d videos, val: %d videos", len(train_ids), len(val_ids))

    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)
    max_len = max(len(video_dfs[v]) for v in train_ids)
    baseline_acc = np.zeros(max_len, dtype=np.float64)
    baseline_cnt = np.zeros(max_len, dtype=np.float64)
    for v in train_ids:
        ret = pd.to_numeric(video_dfs[v]["retention"], errors="coerce").fillna(0).values
        baseline_acc[: len(ret)] += ret
        baseline_cnt[: len(ret)] += 1.0
    baseline_curve = (baseline_acc / np.maximum(baseline_cnt, 1.0)).astype(np.float32)
    baseline_norm = normalizer.normalize_retention(baseline_curve).astype(np.float32)

    video_weights = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None

    model = VideoMAERetention(backbone=args.backbone, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, n_head_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout).to(
        device
    )
    model.set_baseline(torch.tensor(baseline_norm))

    trainable = sum(p.numel() for p in model.trainable_parameters())
    total = sum(p.numel() for p in model.parameters())
    logger.info("E2E trainable: %d / %d params (%.1f%%)", trainable, total, 100 * trainable / total)

    train_ds = VideoMAEE2EWindowDataset(
        video_dfs, train_ids, args.snapshot_dir, args.window_size, args.window_stride, normalizer, VIDEOMAE_NUM_FRAMES, model.processor, video_weights=video_weights
    )
    val_ds = VideoMAEE2EWindowDataset(
        video_dfs, val_ids, args.snapshot_dir, args.window_size, args.window_stride, normalizer, VIDEOMAE_NUM_FRAMES, model.processor, video_weights=None
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    opt_params = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: _lr_lambda_warmup_cosine(ep, args.warmup_epochs, args.epochs))

    best_val, no_improve, best_state = float("inf"), 0, {}
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in (pbar := tqdm(range(1, args.epochs + 1), desc="E2E Training")):
        model.train()
        tl, tn = 0.0, 0
        for batch in train_dl:
            pv = batch["pixel_values"].to(device)
            tgt, pad_mask, ad_mask, spike_triggers = _to_device(batch, device, "retention", "padding_mask", "is_ad", "spike_triggers")
            vw = batch["video_weight"].to(device) if args.engagement_weight else None
            pred = model(pv, n_seconds=args.window_size, padding_mask=pad_mask)
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
            nn.utils.clip_grad_norm_(opt_params, args.grad_clip)
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
                pv = batch["pixel_values"].to(device)
                tgt, pad_mask, ad_mask, spike_triggers = _to_device(batch, device, "retention", "padding_mask", "is_ad", "spike_triggers")
                pred = model(pv, n_seconds=args.window_size, padding_mask=pad_mask)
                loss = composite_loss(pred, tgt, ad_mask, pad_mask, 1.0, None, args.alpha_corr, 0.0, 0.0, 0, 1.0, args.alpha_delta)
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
    model.set_baseline(torch.tensor(baseline_norm))
    plot_training_curve(train_losses, val_losses, os.path.join(args.output_dir, "training_curve.png"))

    all_metrics = {}
    for vid in video_ids:
        split = "val" if vid in val_ids else "train"
        vpath = os.path.join(args.snapshot_dir, vid, "video.mp4")
        if not os.path.exists(vpath):
            continue
        y_true, y_pred = predict_video_videomae_e2e(
            model, video_dfs[vid], vpath, normalizer, device, args.window_size, args.window_stride, VIDEOMAE_NUM_FRAMES, smooth_window=args.smooth_window
        )
        m = seq_metrics(y_pred, y_true)
        all_metrics[vid] = {**m, "split": split, "n_seconds": len(y_true)}
        logger.info("%s [%s]  MAE=%.4f  r=%.3f", vid, split, m["mae"], m["pearson"])
        is_ad = video_dfs[vid]["is_ad"].values if "is_ad" in video_dfs[vid].columns else None
        plot_prediction(vid, y_true, y_pred, is_ad, split, m, os.path.join(args.output_dir, "videos", vid, "prediction.png"))

    plot_mae_summary(all_metrics, args.output_dir, model_name="VideoMAE-e2e")

    torch.save(
        {
            "model_state_dict": best_state,
            "feature_cols": feature_cols,
            "mode": "e2e",
            "backbone": args.backbone,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "window_size": args.window_size,
            "baseline_norm": baseline_norm.tolist(),
        },
        os.path.join(args.output_dir, "videomae_model.pt"),
    )

    Path(os.path.join(args.output_dir, "metrics.json")).write_text(
        json.dumps(
            {
                "model": "VideoMAE-e2e",
                "backbone": args.backbone,
                "per_video": all_metrics,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "best_val_loss": round(best_val, 6),
                "epochs_trained": epoch,
                "elapsed_sec": round(time.time() - t0, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Done (e2e). Best val=%.4f epochs=%d time=%.0fs", best_val, epoch, time.time() - t0)
    return all_metrics


def train_hybrid_mode(args):
    """Mode 'hybrid': VideoMAE embeddings + tabular/audio/text fusion with LoRA."""
    logger.info("Hybrid mode: VideoMAE LoRA backbone + multimodal fusion")

    device = torch.device(args.device)
    video_dfs = load_all_merged(
        args.output_dir_features,
        args.snapshot_dir,
        use_curve_raw=args.use_curve_raw,
        emb_pca_components=0,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )

    _ensure_videomae_embeddings(video_dfs, args)
    vmae_embs = _load_videomae_embeddings(video_dfs, args.embeddings_root)
    aligned_embs = load_aligned_embeddings_for_videos(video_dfs, args.embeddings_root)

    video_embeddings = {}
    for vid in video_dfs:
        vmae = vmae_embs.get(vid)
        aligned = aligned_embs.get(vid)
        if vmae is not None and aligned is not None:
            n = min(len(vmae), len(aligned), len(video_dfs[vid]))
            video_embeddings[vid] = np.concatenate([vmae[:n], aligned[:n, 768:]], axis=1)
        elif vmae is not None:
            video_embeddings[vid] = vmae
        elif aligned is not None:
            video_embeddings[vid] = aligned

    video_ids = sorted(video_dfs.keys())
    feature_cols, filter_log = filter_features(video_dfs, top_k=args.top_k_features or None)
    Path(os.path.join(args.output_dir, "feature_filter_log.txt")).write_text("\n".join(filter_log), encoding="utf-8")

    output_dir = Path(args.output_dir_features)
    output_video_ids = sorted(
        p.name.replace("_features.csv", "") for p in output_dir.glob("*_features.csv") if not p.name.endswith(".partial") and p.name.replace("_features.csv", "") in video_dfs
    )

    train_ids, val_ids = _split_ids(args, video_ids, output_video_ids)

    logger.info("Train: %d, Val: %d", len(train_ids), len(val_ids))

    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)
    ref_sec = max_time_sec_over_videos(video_dfs, train_ids)
    video_weights = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None

    n_tab = len(feature_cols) + time_feature_extra_dim(args.time_features)

    from src.models.retention_multimodal_transformer import MultimodalRetentionTransformer

    emb_dim = next(iter(video_embeddings.values())).shape[1] if video_embeddings else 1536
    vis_dim = 768
    aud_dim = min(512, emb_dim - vis_dim) if emb_dim > vis_dim else 0
    txt_dim = emb_dim - vis_dim - aud_dim if emb_dim > vis_dim + aud_dim else 0

    model = MultimodalRetentionTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        n_tabular_features=n_tab,
        emb_visual_dim=vis_dim,
        emb_audio_dim=max(aud_dim, 1),
        emb_text_dim=max(txt_dim, 1),
    ).to(device)

    max_len = max(len(video_dfs[v]) for v in train_ids)
    baseline_acc = np.zeros(max_len, dtype=np.float64)
    baseline_cnt = np.zeros(max_len, dtype=np.float64)
    for v in train_ids:
        ret = pd.to_numeric(video_dfs[v]["retention"], errors="coerce").fillna(0).values
        baseline_acc[: len(ret)] += ret
        baseline_cnt[: len(ret)] += 1.0
    baseline_norm = normalizer.normalize_retention((baseline_acc / np.maximum(baseline_cnt, 1.0)).astype(np.float32)).astype(np.float32)
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

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Hybrid model params: %d (%.1fK), emb_dim=%d", n_params, n_params / 1000, emb_dim)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: _lr_lambda_warmup_cosine(ep, args.warmup_epochs, args.epochs))

    best_val, no_improve, best_state = float("inf"), 0, {}
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in (pbar := tqdm(range(1, args.epochs + 1), desc="Hybrid Training")):
        model.train()
        tl, tn = 0.0, 0
        for batch in train_dl:
            feat, tgt, pad_mask, ad_mask, spike_triggers = _to_device(batch, device, "tabular", "retention", "padding_mask", "is_ad", "spike_triggers")
            emb = batch["embeddings"].to(device)
            vw = batch["video_weight"].to(device) if args.engagement_weight else None
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

    plot_mae_summary(all_metrics, args.output_dir, model_name="VideoMAE-hybrid")

    torch.save({"model_state_dict": best_state, "feature_cols": feature_cols, "mode": "hybrid", "backbone": args.backbone}, os.path.join(args.output_dir, "videomae_model.pt"))

    Path(os.path.join(args.output_dir, "metrics.json")).write_text(
        json.dumps(
            {
                "model": "VideoMAE-hybrid",
                "backbone": args.backbone,
                "per_video": all_metrics,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "best_val_loss": round(best_val, 6),
                "epochs_trained": epoch,
                "elapsed_sec": round(time.time() - t0, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Done (hybrid). Best val=%.4f epochs=%d time=%.0fs", best_val, epoch, time.time() - t0)
    return all_metrics


def main():
    args = parse_args()
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
