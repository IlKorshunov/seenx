"""
Transformer v2 + Video temporal features for retention prediction.
Extends the v2 transformer by adding per-second video features from ./output/
resampled to curve_points (50) as time-varying inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID, _point_col, _safe_float, build_rows_with_targets_source, make_feature_matrix, select_train_test
from train.loo.train_retention_lstm_loo import (
    _build_integration_matrix,
    _clip01,
    _compute_percentile_curves,
    _curve_metrics,
    _knn_weighted_baseline,
    _make_time_features,
    _reduce_dim,
    _resolve_device,
    _standardize_apply,
    _standardize_fit,
)
from train.loo.train_retention_transformer_v2_loo import RetentionTransformerV2, _curve_loss, _make_train_val_split, _smooth_postprocess, _warmup_cosine


# ---------------------------------------------------------------------------
# Video feature loading & resampling
# ---------------------------------------------------------------------------

# Columns to exclude from temporal features
_EXCLUDE_COLS = {
    "time",
    "retention",
    # Near-constant or zero-variance columns
    "silence_stretch",
    "beat_sync",
    "beat_sync_ratio",
    "n_chapters",  # constant per video
    "hook_score",
    "hook_has_address",  # constant per video
    "n_ad_segments",  # constant per video
    "ad_density_percent",  # constant per video
}


def _discover_all_feature_columns(output_dir: Path, video_ids: list[str]) -> list[str]:
    """Scan all available CSVs to build the union of feature columns."""
    all_cols = set()
    for vid in video_ids:
        csv_path = output_dir / f"{vid}_features.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, nrows=0)
            cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
            all_cols.update(cols)
        except Exception:
            continue
    # Also scan any remaining CSVs in the directory not in video_ids
    for p in output_dir.glob("*_features.csv"):
        vid = p.stem.replace("_features", "")
        if vid not in set(video_ids):
            try:
                df = pd.read_csv(p, nrows=0)
                cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
                all_cols.update(cols)
            except Exception:
                continue
    return sorted(all_cols)


def _load_video_temporal_features(output_dir: Path, video_id: str, curve_points: int, canonical_cols: list[str]) -> np.ndarray | None:
    """Load per-second features from output CSV and resample to curve_points.
    Uses canonical_cols to ensure consistent column ordering, filling missing with 0.
    """
    csv_path = output_dir / f"{video_id}_features.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None

    # Build array with canonical column ordering, fill missing with 0
    n_seconds = len(df)
    arr = np.zeros((n_seconds, len(canonical_cols)), dtype=float)
    for j, col in enumerate(canonical_cols):
        if col in df.columns:
            arr[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)

    if n_seconds == 0:
        return None

    # Resample to curve_points using linear interpolation
    if n_seconds == curve_points:
        return arr
    x_old = np.linspace(0, 1, n_seconds)
    x_new = np.linspace(0, 1, curve_points)
    resampled = np.zeros((curve_points, len(canonical_cols)), dtype=float)
    for j in range(len(canonical_cols)):
        resampled[:, j] = np.interp(x_new, x_old, arr[:, j])
    return resampled


def load_all_video_temporal_features(output_dir: Path, video_ids: list[str], curve_points: int) -> tuple[np.ndarray, list[str], int]:
    """
    Load temporal features for all videos.
    Returns:
        features: (n_videos, curve_points, n_temporal_features)
        feature_names: list of feature column names
        n_available: number of videos with actual features
    """
    # Discover union of all feature columns across all CSVs
    feat_cols = _discover_all_feature_columns(output_dir, video_ids)

    if not feat_cols:
        print("[video] WARNING: No output CSVs found!")
        return np.zeros((len(video_ids), curve_points, 1)), [], 0

    n_feat = len(feat_cols)
    features = np.zeros((len(video_ids), curve_points, n_feat), dtype=float)
    n_available = 0

    for i, vid in enumerate(video_ids):
        arr = _load_video_temporal_features(output_dir, vid, curve_points, feat_cols)
        if arr is not None:
            features[i] = arr
            n_available += 1

    print(f"[video] Loaded temporal features: {n_available}/{len(video_ids)} videos, {n_feat} features, {curve_points} time points")
    return features, feat_cols, n_available


def _standardize_temporal_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardize temporal features across all videos and time points.
    X shape: (n_videos, curve_points, n_features)
    """
    # Compute stats across videos and time
    flat_train = X_train.reshape(-1, X_train.shape[2])  # (n*T, F)
    mu = flat_train.mean(axis=0)
    sigma = flat_train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    X_train_norm = (X_train - mu[None, None, :]) / sigma[None, None, :]
    X_test_norm = (X_test - mu[None, None, :]) / sigma[None, None, :]
    return X_train_norm, X_test_norm


def _reduce_temporal_dim(X_train: np.ndarray, X_test: np.ndarray, max_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA on temporal features. X shape: (n_videos, curve_points, n_features)
    """
    n_feat = X_train.shape[2]
    if n_feat <= max_dim:
        return X_train.copy(), X_test.copy()

    # Flatten for SVD: (n_videos * curve_points, n_features)
    flat = X_train.reshape(-1, n_feat)
    _, _, Vt = np.linalg.svd(flat, full_matrices=False)
    keep = min(max_dim, Vt.shape[0])
    basis = Vt[:keep].T  # (n_feat, keep)

    T = X_train.shape[1]
    X_tr_red = np.zeros((X_train.shape[0], T, keep), dtype=float)
    X_te_red = np.zeros((X_test.shape[0], T, keep), dtype=float)
    for t in range(T):
        X_tr_red[:, t, :] = X_train[:, t, :] @ basis
        X_te_red[:, t, :] = X_test[:, t, :] @ basis
    return X_tr_red, X_te_red


# ---------------------------------------------------------------------------
# Modified sequence inputs with temporal video features
# ---------------------------------------------------------------------------


def _make_sequence_inputs_with_video(
    X: np.ndarray,
    time_features: np.ndarray,
    knn_baseline: np.ndarray,
    global_mean: np.ndarray,
    integration_strength: np.ndarray,
    video_temporal: np.ndarray,
    percentile_curves: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build sequence inputs combining static features + temporal video features.
    X: (n_videos, static_dim) - static features (broadcast across time)
    video_temporal: (n_videos, curve_points, temporal_dim) - time-varying features
    """
    n, steps = X.shape[0], time_features.shape[0]
    temporal_dim = video_temporal.shape[2] if video_temporal.ndim == 3 else 0

    bl_d1 = np.diff(knn_baseline, prepend=knn_baseline[0])
    bl_d2 = np.diff(knn_baseline, n=2, prepend=[knn_baseline[0], knn_baseline[0]])
    gm_d1 = np.diff(global_mean, prepend=global_mean[0])
    diff_bl = knn_baseline - global_mean

    n_extra = 8
    if percentile_curves is not None:
        n_extra += percentile_curves.shape[0]

    total_dim = X.shape[1] + time_features.shape[1] + n_extra + temporal_dim
    base = np.zeros((n, steps, total_dim), dtype=float)
    c = 0

    # Static features (broadcast)
    base[:, :, c : c + X.shape[1]] = X[:, None, :]
    c += X.shape[1]

    # Time features (shared)
    base[:, :, c : c + time_features.shape[1]] = time_features[None, :, :]
    c += time_features.shape[1]

    # Baselines
    base[:, :, c] = knn_baseline[None, :]
    c += 1
    base[:, :, c] = bl_d1[None, :]
    c += 1
    base[:, :, c] = bl_d2[None, :]
    c += 1
    base[:, :, c] = global_mean[None, :]
    c += 1
    base[:, :, c] = gm_d1[None, :]
    c += 1
    base[:, :, c] = diff_bl[None, :]
    c += 1

    # Integration
    base[:, :, c] = integration_strength
    c += 1
    ad_delta = np.diff(integration_strength, axis=1, prepend=integration_strength[:, :1])
    base[:, :, c] = ad_delta
    c += 1

    # Percentiles
    if percentile_curves is not None:
        for k in range(percentile_curves.shape[0]):
            base[:, :, c] = percentile_curves[k][None, :]
            c += 1

    # Temporal video features (time-varying per video)
    if temporal_dim > 0:
        base[:, :, c : c + temporal_dim] = video_temporal[:, :steps, :]
        c += temporal_dim

    return base


# ---------------------------------------------------------------------------
# Training function (reuses v2 model and loss)
# ---------------------------------------------------------------------------


def _train_one(
    *,
    seed,
    seq_train,
    y_train,
    integ_train,
    baseline_np,
    seq_test,
    integ_test,
    device,
    d_model,
    n_layers,
    n_heads,
    ffn_mult,
    dropout,
    residual_scale,
    conv_kernels,
    curve_points,
    static_dim,
    time_ctx_dim,
    ad_lw,
    ad_sw,
    noise_std,
    feat_noise,
    epochs,
    patience,
    grad_clip,
    log_every,
    lr,
    wd,
    lr_min,
    warmup,
    swa_frac,
    mixup_alpha,
    tta,
    eidx,
    etot,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    xt = torch.tensor(seq_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xte = torch.tensor(seq_test, dtype=torch.float32, device=device)
    it = torch.tensor(integ_train, dtype=torch.float32, device=device)
    ite = torch.tensor(integ_test, dtype=torch.float32, device=device)
    bl = torch.tensor(baseline_np, dtype=torch.float32, device=device)

    tri, vai = _make_train_val_split(len(y_train), seed)
    xf, yf, inf_ = xt[tri], yt[tri], it[tri]
    xv = xt[vai] if len(vai) else None
    yv = yt[vai] if len(vai) else None
    iv = it[vai] if len(vai) else None

    model = RetentionTransformerV2(
        seq_train.shape[2], d_model, n_layers, n_heads, ffn_mult, dropout, residual_scale, curve_points, conv_kernels or None, static_dim, time_ctx_dim
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=max(0.0, wd))
    sched = _warmup_cosine(opt, warmup, epochs, lr_min)

    swa_s, swa_n = None, 0
    swa_ep = max(1, int(epochs * swa_frac))
    tag = f"v2vid-{eidx + 1}/{etot}"
    best_l, best_e, best_sd = float("inf"), 0, None
    live = sys.stdout.isatty()

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        cx, cy, ci = xf, yf, inf_

        if mixup_alpha > 0 and cx.shape[0] > 1:
            lam = max(np.random.beta(mixup_alpha, mixup_alpha), 0.55)
            pm = torch.randperm(cx.shape[0], device=device)
            cx = lam * cx + (1 - lam) * cx[pm]
            cy = lam * cy + (1 - lam) * cy[pm]
            ci = lam * ci + (1 - lam) * ci[pm]
        if feat_noise > 0:
            cx = cx + feat_noise * torch.randn_like(cx)
        if noise_std > 0:
            cy = torch.clamp(cy + noise_std * torch.randn_like(cy), 0, 1)

        pred, res, _ = model(cx, bl, ci)
        loss, stats = _curve_loss(pred, cy, res, ci, ad_lw, ad_sw)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        opt.step()
        sched.step()

        if ep >= swa_ep:
            sd = model.state_dict()
            if swa_s is None:
                swa_s = {k: v.clone() for k, v in sd.items()}
                swa_n = 1
            else:
                swa_n += 1
                for k in swa_s:
                    swa_s[k] += (sd[k] - swa_s[k]) / swa_n

        if xv is not None and yv is not None:
            model.eval()
            with torch.no_grad():
                pv, rv, _ = model(xv, bl, iv)
                ml, _ = _curve_loss(pv, yv, rv, iv, ad_lw, ad_sw)
                mv = ml.item()
        else:
            mv = loss.item()

        if live:
            print(f"[{tag}] ep={ep}/{epochs} train={loss.item():.5f} val={mv:.5f} best={best_l:.5f}", end="\r", flush=True)
        if mv < best_l - 1e-8:
            best_l, best_e = mv, ep
            best_sd = copy.deepcopy(model.state_dict())
        if ep == 1 or ep % log_every == 0:
            if live:
                print()
            print(f"[{tag}] ep={ep}/{epochs} train={loss.item():.5f} val={mv:.5f} best={best_l:.5f} corr={stats['corr']:.5f}")
        if (ep - best_e) >= patience:
            if live:
                print()
            print(f"[{tag}] early_stop ep={ep} best={best_e}")
            break
    if live:
        print()

    if swa_s is not None and swa_n >= 10:
        model.load_state_dict(swa_s)
        print(f"[{tag}] SWA ({swa_n})")
    elif best_sd is not None:
        model.load_state_dict(best_sd)
        print(f"[{tag}] best ep={best_e}")

    ft_ep = max(1, min(best_e // 5, 60))
    fopt = torch.optim.AdamW(model.parameters(), lr=lr * 0.03, weight_decay=max(0.0, wd))
    for _ in range(ft_ep):
        model.train()
        fopt.zero_grad(set_to_none=True)
        pf, rf, _ = model(xt, bl, it)
        fl, _ = _curve_loss(pf, yt, rf, it, ad_lw, ad_sw)
        fl.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        fopt.step()

    model.eval()
    with torch.no_grad():
        trp, _, _ = model(xt, bl, it)
        trp = trp.cpu().numpy()
        preds, ress, ads = [], [], []
        tp, tr_, ta = model(xte, bl, ite)
        preds.append(tp.cpu().numpy()[0])
        ress.append(tr_.cpu().numpy()[0])
        ads.append(ta.cpu().numpy()[0])
        for _ in range(max(0, tta - 1)):
            xn = xte + 0.015 * torch.randn_like(xte)
            tp2, tr2, ta2 = model(xn, bl, ite)
            preds.append(tp2.cpu().numpy()[0])
            ress.append(tr2.cpu().numpy()[0])
            ads.append(ta2.cpu().numpy()[0])

    return trp, np.mean(preds, 0), np.mean(ress, 0), np.mean(ads, 0), best_e, best_l


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer v2 + Video features.")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--snapshot-dir", default="drive_snapshot_90")
    p.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    p.add_argument("--output-features-dir", default="output", help="Directory with per-second video feature CSVs")
    p.add_argument("--limit-videos", type=int, default=90)
    p.add_argument("--train-videos", type=int, default=89)
    p.add_argument("--curve-points", type=int, default=50)
    p.add_argument("--eval-video-folder", default="")
    p.add_argument("--eval-drive-file-id", default="")
    p.add_argument("--output-dir", default="transformer_v2_video_experiment")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-transformer-layers", type=int, default=3)
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--ffn-mult", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--residual-scale", type=float, default=1.25)
    p.add_argument("--conv-kernels", default="3,5,7")
    p.add_argument("--n-sinusoidal", type=int, default=4)
    p.add_argument("--ad-loss-weight", type=float, default=2.5)
    p.add_argument("--ad-slope-weight", type=float, default=1.5)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--knn-temperature", type=float, default=0.5)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    p.add_argument("--feature-max-dim", type=int, default=40)
    p.add_argument("--video-feature-max-dim", type=int, default=20, help="Max PCA dimensions for temporal video features")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--feature-noise-std", type=float, default=0.02)
    p.add_argument("--max-step", type=float, default=0.05)
    p.add_argument("--ensemble-seeds", type=int, default=9)
    p.add_argument("--mixup-alpha", type=float, default=0.3)
    p.add_argument("--lr-min-ratio", type=float, default=0.02)
    p.add_argument("--warmup-epochs", type=int, default=40)
    p.add_argument("--swa-start-frac", type=float, default=0.7)
    p.add_argument("--tta-samples", type=int, default=16)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--max-increase-per-step", type=float, default=0.05)
    return p.parse_args()


def _extract_video_id_from_folder(video_folder: str) -> str:
    """Extract video ID from folder name (same as output CSV naming)."""
    return str(video_folder).strip()


def run_experiment(args) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    output_features_dir = Path(str(args.output_features_dir)).expanduser()

    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    # --- Static features (original pipeline) ---
    x_tr_df = make_feature_matrix(train_df)
    x_te_df = make_feature_matrix(test_df)
    if x_tr_df.empty:
        raise RuntimeError("Empty features")

    Xtr = x_tr_df.to_numpy(float)
    Xte = x_te_df.to_numpy(float)
    mu, sig = _standardize_fit(Xtr)
    Xtr = _standardize_apply(Xtr, mu, sig)
    Xte = _standardize_apply(Xte, mu, sig)
    Xtr, Xte = _reduce_dim(Xtr, Xte, int(args.feature_max_dim))
    static_dim = Xtr.shape[1]
    print(f"[v2vid] Static features: {static_dim} dims")

    # --- Temporal video features from output/ ---
    T = int(args.curve_points)
    train_video_ids = [_extract_video_id_from_folder(r) for r in train_df["video_folder"]]
    test_video_ids = [_extract_video_id_from_folder(r) for r in test_df["video_folder"]]

    vid_tr, feat_names, n_avail_tr = load_all_video_temporal_features(output_features_dir, train_video_ids, T)
    vid_te, _, n_avail_te = load_all_video_temporal_features(output_features_dir, test_video_ids, T)

    # Standardize and reduce temporal features
    if vid_tr.shape[2] > 0:
        vid_tr, vid_te = _standardize_temporal_features(vid_tr, vid_te)
        vid_max_dim = int(args.video_feature_max_dim)
        vid_tr, vid_te = _reduce_temporal_dim(vid_tr, vid_te, vid_max_dim)
        temporal_dim = vid_tr.shape[2]
        print(
            f"[v2vid] Temporal video features: {temporal_dim} dims "
            f"(reduced from {len(feat_names)}, "
            f"train={n_avail_tr}/{len(train_video_ids)}, "
            f"test={n_avail_te}/{len(test_video_ids)})"
        )
    else:
        temporal_dim = 0
        print("[v2vid] WARNING: No temporal video features available!")

    # --- Retention targets ---
    y_train = np.zeros((len(train_df), T), float)
    y_true = np.zeros(T, float)
    for i in range(T):
        col = _point_col(i)
        y_train[:, i] = pd.to_numeric(train_df[col], errors="coerce").fillna(0).to_numpy(float)
        y_true[i] = _safe_float(test_df.iloc[0][col], 0.0)
    y_train, y_true = _clip01(y_train), _clip01(y_true)

    # --- Baselines ---
    n_sin = int(getattr(args, "n_sinusoidal", 4))
    tf = _make_time_features(T, n_sin)
    tcd = tf.shape[1]
    global_mean = np.mean(y_train, axis=0)
    pctls = _compute_percentile_curves(y_train)
    int_tr = _build_integration_matrix(train_df, snapshot_dir, T)
    int_te = _build_integration_matrix(test_df, snapshot_dir, T)

    knn_bl = _clip01(_knn_weighted_baseline(Xtr, Xte, y_train, k=int(args.knn_k), temperature=float(args.knn_temperature)))
    print(f"[v2vid] kNN RMSE to true: {np.sqrt(np.mean((knn_bl - y_true) ** 2)):.4f}")
    print(f"[v2vid] mean RMSE to true: {np.sqrt(np.mean((global_mean - y_true) ** 2)):.4f}")

    # --- Build sequence inputs with video temporal features ---
    seq_tr = _make_sequence_inputs_with_video(Xtr, tf, knn_bl, global_mean, int_tr, vid_tr, pctls)
    seq_te = _make_sequence_inputs_with_video(Xte, tf, knn_bl, global_mean, int_te, vid_te, pctls)
    print(f"[v2vid] Sequence input shape: {seq_tr.shape} (static={static_dim} + time={tcd} + baseline=8+3 + temporal={temporal_dim})")

    # --- Device ---
    device = _resolve_device(getattr(args, "device", "auto"))
    print(f"[v2vid] device={device.type}")
    try:
        torch.set_num_threads(int(getattr(args, "torch_num_threads", 1)))
    except Exception:
        pass

    # --- Training ---
    dm = int(args.d_model)
    nl = int(args.n_transformer_layers)
    nh = int(args.attn_heads)
    fm = int(args.ffn_mult)
    do = float(args.dropout)
    rs = float(args.residual_scale)
    ck = [int(k.strip()) for k in str(args.conv_kernels).split(",") if k.strip()]
    n_ens = int(args.ensemble_seeds)
    n_tta = int(args.tta_samples)
    seeds = [int(args.random_seed) + i * 111 for i in range(n_ens)]

    all_tr, all_te, all_res, all_ad, all_be, all_bl_ = [], [], [], [], [], []

    for idx, s in enumerate(seeds):
        print(f"\n{'=' * 60}\n[v2vid] ensemble {idx + 1}/{n_ens} seed={s}\n{'=' * 60}")
        tr, te, res, ad, be, bl_ = _train_one(
            seed=s,
            seq_train=seq_tr,
            y_train=y_train,
            integ_train=int_tr,
            baseline_np=knn_bl,
            seq_test=seq_te,
            integ_test=int_te,
            device=device,
            d_model=dm,
            n_layers=nl,
            n_heads=nh,
            ffn_mult=fm,
            dropout=do,
            residual_scale=rs,
            conv_kernels=ck,
            curve_points=T,
            static_dim=static_dim,
            time_ctx_dim=tcd,
            ad_lw=float(args.ad_loss_weight),
            ad_sw=float(args.ad_slope_weight),
            noise_std=float(args.noise_std),
            feat_noise=float(args.feature_noise_std),
            epochs=int(args.epochs),
            patience=int(args.patience),
            grad_clip=float(args.grad_clip),
            log_every=int(args.log_every),
            lr=float(args.learning_rate),
            wd=float(args.weight_decay),
            lr_min=float(args.lr_min_ratio),
            warmup=int(args.warmup_epochs),
            swa_frac=float(args.swa_start_frac),
            mixup_alpha=float(args.mixup_alpha),
            tta=n_tta,
            eidx=idx,
            etot=n_ens,
        )
        all_tr.append(tr)
        all_te.append(te)
        all_res.append(res)
        all_ad.append(ad)
        all_be.append(be)
        all_bl_.append(bl_)

    # --- Ensemble aggregation ---
    losses = np.array(all_bl_)
    if losses.max() - losses.min() > 1e-10:
        inv = 1.0 / (losses + 1e-8)
        w = inv / inv.sum()
    else:
        w = np.ones(n_ens) / n_ens
    print(f"[v2vid] weights: {[f'{x:.3f}' for x in w]}")

    train_pred = sum(wi * p for wi, p in zip(w, all_tr, strict=True))
    test_pred = sum(wi * p for wi, p in zip(w, all_te, strict=True))
    test_res = sum(wi * p for wi, p in zip(w, all_res, strict=True))
    test_ad = sum(wi * p for wi, p in zip(w, all_ad, strict=True))

    y_raw = _clip01(test_pred)
    y_pred = _clip01(_smooth_postprocess(y_raw, max_step=float(args.max_step)))

    # --- Save results ---
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "holdout_prediction_vs_true.csv"
    all_df.to_csv(out / "dataset.csv", index=False)

    pd.DataFrame(
        {
            "point_idx": list(range(T)),
            "pred_retention_base": knn_bl,
            "pred_retention_raw": y_raw,
            "pred_retention_residual": test_res,
            "pred_retention_ad_drop": test_ad,
            "integration_strength": int_te[0] if len(int_te) else np.zeros(T),
            "pred_retention": y_pred,
            "true_retention": y_true,
            "abs_error": np.abs(y_pred - y_true),
        }
    ).to_csv(pred_path, index=False)

    metrics = {
        "videos_total_with_target": len(rows),
        "videos_used": len(all_df),
        "train_videos": len(train_df),
        "curve_points": T,
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(test_df.iloc[0]["drive_file_id"]),
        **_curve_metrics(y_pred, y_true),
        "prediction_path": str(pred_path),
        "d_model": dm,
        "n_layers": nl,
        "dropout": do,
        "residual_scale": rs,
        "ensemble_size": n_ens,
        "tta_samples": n_tta,
        "knn_k": int(args.knn_k),
        "knn_temperature": float(args.knn_temperature),
        "best_epochs": all_be,
        "best_losses": all_bl_,
        "ensemble_weights": [float(x) for x in w],
        "train_rmse": float(np.sqrt(np.mean((train_pred - y_train) ** 2))),
        "model_name": "retention_transformer_v2_video",
        "temporal_video_features": temporal_dim,
        "static_features": static_dim,
        "videos_with_video_features": n_avail_tr,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Transformer v2 + Video Features ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
