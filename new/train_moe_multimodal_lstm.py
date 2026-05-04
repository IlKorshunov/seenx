"""Train mixture-of-experts multimodal LSTM (optional SARIMAX baseline).

Isolated under ``new/``. Reuses the same data loading and loss as
``train_sarima_multimodal_lstm.py`` but swaps the backbone for ``MoeMultimodalRetentionLSTM``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from new.model_moe_multimodal_lstm import MoeMultimodalRetentionLSTM
from new.sarima_trend import ClusterSarimaxTrendProvider
from train.common.seq_data_utils import (
    TIME_FEATURE_MODES,
    FeatureNormalizer,
    _align_embedding_rows,
    _append_time_features_to_matrix,
    _augment_tabular_features,
    _prediction_tabular_X,
    _tabular_array_from_df,
    _time_sec_per_row,
    _window_start_indices,
    composite_loss,
    filter_features,
    load_aligned_embeddings_for_videos,
    load_all_merged,
    load_video_weights,
    max_time_sec_over_videos,
    plot_mae_summary,
    resample_embeddings_to_match_dfs,
    resample_video_dfs_to_curve_points,
    seq_metrics,
    smooth_predictions,
    time_feature_extra_dim,
)


matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

C_BLUE, C_ORANGE, C_PURPLE = "#2196F3", "#FF5722", "#9C27B0"
C_GREEN, C_RED = "#4CAF50", "#F44336"
GRID_ALPHA = 0.3
PLOT_DPI = 150


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE multimodal LSTM (+ optional SARIMAX baseline)")
    a = p.add_argument
    a("--output-dir-features", default="output")
    a("--snapshot-dir", default="data")
    a("--embeddings-root", default="embeddings")
    a("--output-dir", default="new/experiments/moe_multimodal_lstm")
    a("--use-curve-raw", action="store_true", default=True)
    a("--no-use-curve-raw", dest="use_curve_raw", action="store_false")
    a("--val-ratio", type=float, default=0.15)
    a("--val-first-n-output", type=int, default=0)
    a("--eval-video", default="")
    a("--train-video-ids-file", default="")
    a("--top-k-features", type=int, default=0)
    a("--window-size", type=int, default=128)
    a("--window-stride", type=int, default=64)
    a("--hidden-size", type=int, default=256)
    a("--n-layers", type=int, default=3)
    a("--dropout", type=float, default=0.2)
    a("--epochs", type=int, default=200)
    a("--batch-size", type=int, default=16)
    a("--lr", type=float, default=5e-4)
    a("--weight-decay", type=float, default=1e-3)
    a("--patience", type=int, default=30)
    a("--grad-clip", type=float, default=1.0)
    a("--warmup-epochs", type=int, default=10)
    a("--feature-mask-prob", type=float, default=0.1)
    a("--noise-std", type=float, default=0.02)
    a("--random-seed", type=int, default=42)
    a("--device", default="cpu")
    a("--curve-points", type=int, default=0)
    a("--time-features", choices=["none", "frac", "frac_sec"], default="none")
    a("--min-duration-sec", type=float, default=0)
    a("--max-duration-sec", type=float, default=0)
    a("--engagement-weight", action="store_true", default=True)
    a("--no-engagement-weight", dest="engagement_weight", action="store_false")
    a("--ad-penalty-weight", type=float, default=15.0)
    a("--alpha-corr", type=float, default=0.3)
    a("--alpha-smooth", type=float, default=0.15)
    a("--alpha-delta", type=float, default=0.4)
    a("--alpha-mono", type=float, default=0.03)
    a("--start-boost-secs", type=int, default=15)
    a("--start-boost-factor", type=float, default=2.0)
    a("--smooth-window", type=int, default=7)
    a("--apply-smoothing", action="store_true", default=False)
    a("--use-sarimax-baseline", action="store_true", default=True)
    a("--no-sarimax-baseline", dest="use_sarimax_baseline", action="store_false")
    a("--sarima-grid", default="1,0,1;2,0,1;1,1,1;2,1,1;2,1,2", help="SARIMAX order triples when --use-sarimax-baseline")
    a("--n-experts", type=int, default=4)
    a("--cluster-embed-buckets", type=int, default=32, help="Embedding table size; video_cluster is reduced modulo this")
    a("--routing-mode", choices=["soft", "hard_cluster", "hybrid"], default="soft")
    a("--hybrid-alpha", type=float, default=0.5, help="Weight of soft gate in hybrid mode")
    a("--load-balance-weight", type=float, default=0.0, help="Penalty on deviation of mean gate weights from uniform (train only)")
    return p.parse_args()


def _parse_order_grid(spec: str) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        p_i, d_i, q_i = (int(x) for x in item.split(","))
        out.append((p_i, d_i, q_i))
    return out


def _cluster_bucket_from_df(df: pd.DataFrame, n_buckets: int) -> int:
    if n_buckets <= 0:
        return 0
    if "video_cluster" not in df.columns:
        return 0
    s = pd.to_numeric(df["video_cluster"], errors="coerce").dropna()
    if len(s) == 0:
        return 0
    v = int(round(float(s.iloc[0])))
    return int(v % n_buckets)


def _save_fig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="train", color=C_BLUE)
    ax.plot(val_losses, label="val", color=C_ORANGE)
    ax.set(xlabel="epoch", ylabel="composite loss", title="MoE multimodal LSTM training")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()
    _save_fig(fig, out_path)


def plot_prediction(vid, y_true, y_pred, y_prior, split, metrics, out_path, prior_label: str = "prior"):
    t = np.arange(len(y_true))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)
    a1.plot(t, y_true, color=C_BLUE, label="actual", linewidth=1.2)
    a1.plot(t, y_prior, color=C_PURPLE, label=prior_label, alpha=0.85, linewidth=1.0)
    a1.plot(t, y_pred, color=C_ORANGE, label="prediction", alpha=0.9, linewidth=1.2)
    a1.fill_between(t, y_true, y_pred, alpha=0.1, color=C_PURPLE)
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


class MoeMultimodalWindowedDataset(Dataset):
    def __init__(
        self,
        video_dfs: dict[str, pd.DataFrame],
        video_embeddings: dict[str, np.ndarray],
        sarimax_baselines: dict[str, np.ndarray] | None,
        video_ids: list[str],
        feature_cols: list[str],
        normalizer: FeatureNormalizer | None = None,
        window_size: int = 128,
        stride: int = 64,
        video_weights: dict[str, float] | None = None,
        feature_mask_prob: float = 0.0,
        noise_std: float = 0.0,
        emb_dim: int = 1536,
        time_feature_mode: str = "none",
        ref_time_sec_max: float = 1.0,
        cluster_embed_buckets: int = 32,
    ):
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.feature_mask_prob = feature_mask_prob
        self.noise_std = noise_std
        self.emb_dim = emb_dim
        self.time_feature_mode = time_feature_mode if time_feature_mode in TIME_FEATURE_MODES else "none"
        self.ref_time_sec_max = float(max(ref_time_sec_max, 1e-6))
        self.cluster_embed_buckets = max(int(cluster_embed_buckets), 1)
        self.windows = []
        for vid in video_ids:
            df = video_dfs[vid]
            cluster_id = _cluster_bucket_from_df(df, self.cluster_embed_buckets)
            w = float(video_weights[vid]) if video_weights and vid in video_weights else 1.0
            ts_full = _time_sec_per_row(df)
            X, y, is_ad_col, spike_triggers = _tabular_array_from_df(df, feature_cols, normalizer)
            emb = _align_embedding_rows(video_embeddings.get(vid), len(X), emb_dim)
            if sarimax_baselines is not None:
                base_raw = sarimax_baselines[vid].astype(np.float32)
                base = normalizer.normalize_retention(base_raw).astype(np.float32) if normalizer is not None else base_raw
            else:
                base = np.zeros(len(y), dtype=np.float32)
            n = len(X)
            ts_opt = ts_full if self.time_feature_mode != "none" else None
            if n <= window_size:
                self.windows.append((emb, X, y, is_ad_col, spike_triggers, base, n, w, 0, ts_opt, n, cluster_id))
            else:
                for s in _window_start_indices(n, window_size, stride):
                    self.windows.append(
                        (
                            emb[s : s + window_size],
                            X[s : s + window_size],
                            y[s : s + window_size],
                            is_ad_col[s : s + window_size],
                            spike_triggers[s : s + window_size],
                            base[s : s + window_size],
                            window_size,
                            w,
                            s,
                            ts_opt,
                            n,
                            cluster_id,
                        )
                    )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        emb, X, y, is_ad, spike_triggers, base, real_len, weight, start, ts_full, n_full, cluster_id = self.windows[idx]
        ws = self.window_size
        if len(X) < ws:
            pad = ws - len(X)
            emb = np.pad(emb, ((0, pad), (0, 0)))
            X = np.pad(X, ((0, pad), (0, 0)))
            y = np.pad(y, (0, pad))
            is_ad = np.pad(is_ad, (0, pad))
            spike_triggers = np.pad(spike_triggers, (0, pad))
            base = np.pad(base, (0, pad), mode="edge")
            mask = np.array([False] * real_len + [True] * pad)
        else:
            mask = np.zeros(ws, dtype=bool)
        X = _append_time_features_to_matrix(
            _augment_tabular_features(X, self.feature_mask_prob, self.noise_std), start, n_full, ws, real_len, ts_full, self.time_feature_mode, self.ref_time_sec_max
        )
        return {
            "embeddings": torch.from_numpy(emb.copy()),
            "tabular": torch.from_numpy(X),
            "retention": torch.from_numpy(y),
            "is_ad": torch.from_numpy(is_ad),
            "spike_triggers": torch.from_numpy(spike_triggers),
            "sarimax_baseline": torch.from_numpy(base),
            "padding_mask": torch.from_numpy(mask),
            "video_weight": torch.tensor(weight, dtype=torch.float32),
            "cluster_id": torch.tensor(cluster_id, dtype=torch.long),
        }


def _to_device(batch, device, *keys):
    return tuple(batch[k].to(device) for k in keys)


def _lb_penalty(w: torch.Tensor) -> torch.Tensor:
    k = w.size(1)
    mu = w.mean(dim=0)
    tgt = 1.0 / k
    return ((mu - tgt) ** 2).sum()


def train_model(model, train_dl, val_dl, device, args, use_engagement_weight=True, use_sarimax: bool = True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: _lr_lambda(ep, args.warmup_epochs, args.epochs))
    best_val_loss, no_improve, best_state = float("inf"), 0, {}
    train_losses, val_losses = [], []
    t0 = time.time()
    lb_w = float(args.load_balance_weight)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, tn = 0.0, 0
        for batch in train_dl:
            emb, tab, tgt, pad_mask, ad_mask, spike_triggers, baseline, cluster_ids = _to_device(
                batch, device, "embeddings", "tabular", "retention", "padding_mask", "is_ad", "spike_triggers", "sarimax_baseline", "cluster_id"
            )
            vw = batch["video_weight"].to(device) if use_engagement_weight else None
            base_arg = baseline if use_sarimax else None
            if lb_w > 0:
                pred, gw = model(emb, tabular=tab, sarimax_baseline=base_arg, src_key_padding_mask=pad_mask, cluster_ids=cluster_ids, return_expert_weights=True)
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
                loss = loss + lb_w * _lb_penalty(gw)
            else:
                pred = model(emb, tabular=tab, sarimax_baseline=base_arg, src_key_padding_mask=pad_mask, cluster_ids=cluster_ids)
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
                emb, tab, tgt, pad_mask, ad_mask, spike_triggers, baseline, cluster_ids = _to_device(
                    batch, device, "embeddings", "tabular", "retention", "padding_mask", "is_ad", "spike_triggers", "sarimax_baseline", "cluster_id"
                )
                base_arg = baseline if use_sarimax else None
                pred = model(emb, tabular=tab, sarimax_baseline=base_arg, src_key_padding_mask=pad_mask, cluster_ids=cluster_ids)
                loss = composite_loss(pred, tgt, ad_mask, spike_triggers, pad_mask, 1.0, None, args.alpha_corr, 0.0, 0.0, 0, 1.0, args.alpha_delta)
                nv = (~pad_mask).sum().item()
                vl += loss.item() * nv
                vn += nv
        val_losses.append(vl / max(vn, 1))
        logger.info("Epoch %3d/%d train=%.4f val=%.4f lr=%.2e", epoch, args.epochs, train_losses[-1], val_losses[-1], optimizer.param_groups[0]["lr"])
        if val_losses[-1] < best_val_loss:
            best_val_loss, no_improve = val_losses[-1], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stop at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": round(best_val_loss, 6),
        "epochs_trained": epoch,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def _lr_lambda(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / max(total - warmup, 1)))


@torch.no_grad()
def predict_video_moe(
    model: torch.nn.Module,
    df: pd.DataFrame,
    emb: np.ndarray | None,
    feature_cols: list[str],
    normalizer: FeatureNormalizer | None,
    sarimax_baseline_raw: np.ndarray | None,
    device: torch.device,
    window_size: int = 128,
    stride: int = 1,
    emb_dim: int = 1536,
    smooth_window: int = 15,
    apply_smoothing: bool = False,
    time_feature_mode: str = "none",
    ref_time_sec_max: float = 1.0,
    cluster_embed_buckets: int = 32,
    use_sarimax: bool = True,
):
    model.eval()
    cluster_id = _cluster_bucket_from_df(df, cluster_embed_buckets)

    X = _prediction_tabular_X(df, feature_cols, normalizer)
    y_true = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values
    ts_full = _time_sec_per_row(df)
    n_full = len(X)
    emb = _align_embedding_rows(emb, len(X), emb_dim)
    if use_sarimax and sarimax_baseline_raw is not None:
        baseline = normalizer.normalize_retention(sarimax_baseline_raw).astype(np.float32) if normalizer is not None else sarimax_baseline_raw.astype(np.float32)
    else:
        baseline = np.zeros(len(X), dtype=np.float32)
    n = len(X)
    if n <= window_size:
        Xw = X
        real_len = n
        emb_w = emb
        base_w = baseline
        if Xw.shape[0] < window_size:
            pad = window_size - Xw.shape[0]
            Xw = np.pad(Xw, ((0, pad), (0, 0)))
            emb_w = np.pad(emb_w, ((0, pad), (0, 0)))
            base_w = np.pad(base_w, (0, pad), mode="edge")
        Xw = _append_time_features_to_matrix(Xw, 0, n_full, window_size, real_len, ts_full, time_feature_mode, ref_time_sec_max)
        t_emb = torch.tensor(emb_w, dtype=torch.float32).unsqueeze(0).to(device)
        t_tab = torch.tensor(Xw, dtype=torch.float32).unsqueeze(0).to(device)
        t_base = torch.tensor(base_w, dtype=torch.float32).unsqueeze(0).to(device)
        bsz = t_emb.size(0)
        cid_b = torch.full((bsz,), cluster_id, dtype=torch.long, device=device)
        pred = model(t_emb, tabular=t_tab, sarimax_baseline=t_base if use_sarimax else None, cluster_ids=cid_b).squeeze(0).cpu().numpy()[:n]
    else:
        pred_sum, pred_cnt = np.zeros(n), np.zeros(n)
        for s in range(0, n - window_size + 1, stride):
            Xw = _append_time_features_to_matrix(X[s : s + window_size], s, n_full, window_size, window_size, ts_full, time_feature_mode, ref_time_sec_max)
            t_emb = torch.tensor(emb[s : s + window_size], dtype=torch.float32).unsqueeze(0).to(device)
            t_tab = torch.tensor(Xw, dtype=torch.float32).unsqueeze(0).to(device)
            t_base = torch.tensor(baseline[s : s + window_size], dtype=torch.float32).unsqueeze(0).to(device)
            bsz = t_emb.size(0)
            cid_b = torch.full((bsz,), cluster_id, dtype=torch.long, device=device)
            p = model(t_emb, tabular=t_tab, sarimax_baseline=t_base if use_sarimax else None, cluster_ids=cid_b).squeeze(0).cpu().numpy()
            pred_sum[s : s + window_size] += p
            pred_cnt[s : s + window_size] += 1.0
        pred = pred_sum / np.maximum(pred_cnt, 1.0)
    if normalizer is not None:
        pred = normalizer.denormalize_retention(pred)
    if apply_smoothing and smooth_window > 1:
        pred = smooth_predictions(pred, window=smooth_window)
    prior_plot = sarimax_baseline_raw[: len(y_true)] if sarimax_baseline_raw is not None else np.zeros(len(y_true))
    return y_true, pred, prior_plot


def main():
    args = parse_args()
    if args.use_sarimax_baseline:
        try:
            import statsmodels  # noqa: F401
        except Exception as e:
            raise SystemExit("statsmodels is required when --use-sarimax-baseline. Install or pass --no-sarimax-baseline.") from e

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading merged data...")
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

    feature_cols, filter_log = filter_features(video_dfs, top_k=args.top_k_features or None)
    Path(args.output_dir, "feature_filter_log.txt").write_text("\n".join(filter_log), encoding="utf-8")
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
    if args.train_video_ids_file:
        allow = {line.strip() for line in Path(args.train_video_ids_file).read_text(encoding="utf-8").splitlines() if line.strip()}
        train_ids = [vid for vid in train_ids if vid in allow]

    logger.info("Train: %d videos, Val: %d videos", len(train_ids), len(val_ids))

    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)

    sarimax_baselines: dict[str, np.ndarray] | None = None
    trend_meta = None
    if args.use_sarimax_baseline:
        trend_provider = ClusterSarimaxTrendProvider(order_grid=_parse_order_grid(args.sarima_grid)).fit(video_dfs, train_ids)
        trend_meta = trend_provider.describe()
        Path(args.output_dir, "sarimax_trends.json").write_text(json.dumps(trend_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        sarimax_baselines = {vid: trend_provider.get_baseline_for_df(df) for vid, df in video_dfs.items()}

    ref_sec = max_time_sec_over_videos(video_dfs, train_ids)
    video_weights = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None

    train_ds = MoeMultimodalWindowedDataset(
        video_dfs,
        video_embeddings,
        sarimax_baselines,
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
        cluster_embed_buckets=args.cluster_embed_buckets,
    )
    val_ds = MoeMultimodalWindowedDataset(
        video_dfs,
        video_embeddings,
        sarimax_baselines,
        val_ids,
        feature_cols,
        normalizer,
        args.window_size,
        args.window_stride,
        time_feature_mode=args.time_features,
        ref_time_sec_max=ref_sec,
        cluster_embed_buckets=args.cluster_embed_buckets,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MoeMultimodalRetentionLSTM(
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        n_tabular_features=len(feature_cols) + time_feature_extra_dim(args.time_features),
        n_experts=args.n_experts,
        n_cluster_buckets=args.cluster_embed_buckets,
        routing_mode=args.routing_mode,
        hybrid_alpha=args.hybrid_alpha,
    ).to(device)

    model, result = train_model(model, train_dl, val_dl, device, args, use_engagement_weight=args.engagement_weight, use_sarimax=args.use_sarimax_baseline)
    plot_training_curve(result["train_losses"], result["val_losses"], os.path.join(args.output_dir, "training_curve.png"))

    all_metrics = {}
    for vid in video_ids:
        split = "val" if vid in val_ids else "train"
        base_arr = sarimax_baselines[vid] if sarimax_baselines is not None else None
        y_true, y_pred, y_prior = predict_video_moe(
            model,
            video_dfs[vid],
            video_embeddings.get(vid),
            feature_cols,
            normalizer,
            base_arr,
            device,
            args.window_size,
            smooth_window=args.smooth_window,
            apply_smoothing=args.apply_smoothing,
            time_feature_mode=args.time_features,
            ref_time_sec_max=ref_sec,
            cluster_embed_buckets=args.cluster_embed_buckets,
            use_sarimax=args.use_sarimax_baseline,
        )
        m = seq_metrics(y_pred, y_true)
        all_metrics[vid] = {**m, "split": split, "n_seconds": len(y_true)}
        logger.info("%s [%s] RMSE=%.4f MAE=%.4f r=%.3f", vid, split, m["rmse"], m["mae"], m["pearson"])
        plot_prediction(
            vid,
            y_true,
            y_pred,
            y_prior,
            split,
            m,
            os.path.join(args.output_dir, "videos", vid, "prediction.png"),
            prior_label="SARIMAX prior" if args.use_sarimax_baseline else "zero baseline",
        )

    plot_mae_summary(all_metrics, args.output_dir, model_name="MoE+Multimodal-LSTM")

    save_payload = {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "normalizer_median": normalizer.median.tolist(),
        "normalizer_iqr": normalizer.iqr.tolist(),
        "ret_min": normalizer.ret_min,
        "ret_max": normalizer.ret_max,
        "config": vars(args),
    }
    if trend_meta is not None:
        save_payload["sarimax_trends"] = trend_meta
    torch.save(save_payload, os.path.join(args.output_dir, "moe_multimodal_lstm.pt"))

    Path(args.output_dir, "metrics.json").write_text(
        json.dumps(
            {
                "model": "MoE_MultimodalLSTM",
                "feature_cols": feature_cols,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "best_val_loss": result["best_val_loss"],
                "epochs_trained": result["epochs_trained"],
                "elapsed_sec": result["elapsed_sec"],
                "per_video": all_metrics,
                "sarimax_trends": trend_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Done. Best val loss=%.4f", result["best_val_loss"])


if __name__ == "__main__":
    main()
