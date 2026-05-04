"""
AdAwareRetentionTransformer + Video temporal features for retention prediction.
Uses the original transformer (not v2) with temporal smoothing and monotonicity,
extended with per-second video features from ./output/ resampled to curve_points.
"""

from __future__ import annotations

import argparse
import json
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
    _make_time_features,
    _reduce_dim,
    _resolve_device,
    _soft_non_increasing,
    _standardize_apply,
    _standardize_fit,
    _train_single_model,
)


# ---------------------------------------------------------------------------
# Video feature loading & resampling (shared with v2 video script)
# ---------------------------------------------------------------------------

_EXCLUDE_COLS = {"time", "retention", "silence_stretch", "beat_sync", "beat_sync_ratio", "n_chapters", "hook_score", "hook_has_address", "n_ad_segments", "ad_density_percent"}


def _discover_all_feature_columns(output_dir: Path, video_ids: list[str]) -> list[str]:
    all_cols = set()
    for vid in video_ids:
        csv_path = output_dir / f"{vid}_features.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, nrows=0)
            all_cols.update(c for c in df.columns if c not in _EXCLUDE_COLS)
        except Exception:
            continue
    for p in output_dir.glob("*_features.csv"):
        vid = p.stem.replace("_features", "")
        if vid not in set(video_ids):
            try:
                df = pd.read_csv(p, nrows=0)
                all_cols.update(c for c in df.columns if c not in _EXCLUDE_COLS)
            except Exception:
                continue
    return sorted(all_cols)


def _load_video_temporal_features(output_dir: Path, video_id: str, curve_points: int, canonical_cols: list[str]) -> np.ndarray | None:
    csv_path = output_dir / f"{video_id}_features.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None

    n_seconds = len(df)
    arr = np.zeros((n_seconds, len(canonical_cols)), dtype=float)
    for j, col in enumerate(canonical_cols):
        if col in df.columns:
            arr[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)

    if n_seconds == 0:
        return None
    if n_seconds == curve_points:
        return arr
    x_old = np.linspace(0, 1, n_seconds)
    x_new = np.linspace(0, 1, curve_points)
    resampled = np.zeros((curve_points, len(canonical_cols)), dtype=float)
    for j in range(len(canonical_cols)):
        resampled[:, j] = np.interp(x_new, x_old, arr[:, j])
    return resampled


def load_all_video_temporal_features(output_dir: Path, video_ids: list[str], curve_points: int) -> tuple[np.ndarray, list[str], int]:
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


def _standardize_temporal_features(X_train, X_test):
    flat_train = X_train.reshape(-1, X_train.shape[2])
    mu = flat_train.mean(axis=0)
    sigma = flat_train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def _reduce_temporal_dim(X_train, X_test, max_dim):
    n_feat = X_train.shape[2]
    if n_feat <= max_dim:
        return X_train.copy(), X_test.copy()
    flat = X_train.reshape(-1, n_feat)
    _, _, Vt = np.linalg.svd(flat, full_matrices=False)
    keep = min(max_dim, Vt.shape[0])
    basis = Vt[:keep].T
    T = X_train.shape[1]
    X_tr = np.zeros((X_train.shape[0], T, keep), dtype=float)
    X_te = np.zeros((X_test.shape[0], T, keep), dtype=float)
    for t in range(T):
        X_tr[:, t, :] = X_train[:, t, :] @ basis
        X_te[:, t, :] = X_test[:, t, :] @ basis
    return X_tr, X_te


# ---------------------------------------------------------------------------
# Sequence builder: original transformer layout + temporal video features
# ---------------------------------------------------------------------------


def _make_sequence_inputs_with_video(
    X: np.ndarray,
    time_features: np.ndarray,
    primary_baseline: np.ndarray,
    global_mean_curve: np.ndarray,
    integration_strength: np.ndarray,
    video_temporal: np.ndarray,
    percentile_curves: np.ndarray | None = None,
) -> np.ndarray:
    """
    Same layout as train_retention_lstm_loo._make_sequence_inputs
    but with video temporal features appended.
    """
    n, steps = X.shape[0], time_features.shape[0]
    temporal_dim = video_temporal.shape[2] if video_temporal.ndim == 3 else 0

    bl_d1 = np.diff(primary_baseline, prepend=primary_baseline[0])
    bl_d2 = np.diff(primary_baseline, n=2, prepend=[primary_baseline[0], primary_baseline[0]])
    baseline_diff = primary_baseline - global_mean_curve

    n_extra = 7
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

    # Baselines (same as original lstm_loo)
    base[:, :, c] = primary_baseline[None, :]
    c += 1
    base[:, :, c] = bl_d1[None, :]
    c += 1
    base[:, :, c] = bl_d2[None, :]
    c += 1
    base[:, :, c] = global_mean_curve[None, :]
    c += 1
    base[:, :, c] = baseline_diff[None, :]
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
# Main experiment
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer (LSTM-style) + Video features.")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--snapshot-dir", default="drive_snapshot_90")
    p.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    p.add_argument("--output-features-dir", default="output")
    p.add_argument("--limit-videos", type=int, default=90)
    p.add_argument("--train-videos", type=int, default=89)
    p.add_argument("--curve-points", type=int, default=50)
    p.add_argument("--eval-video-folder", default="")
    p.add_argument("--eval-drive-file-id", default="")
    p.add_argument("--output-dir", default="transformer_video_experiment")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-transformer-layers", type=int, default=3)
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--ffn-mult", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.18)
    p.add_argument("--residual-scale", type=float, default=0.8)
    p.add_argument("--conv-kernels", default="3,5,7")
    p.add_argument("--n-sinusoidal", type=int, default=4)
    p.add_argument("--ad-loss-weight", type=float, default=2.5)
    p.add_argument("--ad-slope-weight", type=float, default=1.5)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    p.add_argument("--feature-max-dim", type=int, default=40)
    p.add_argument("--video-feature-max-dim", type=int, default=20)
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--feature-noise-std", type=float, default=0.02)
    p.add_argument("--max-increase-per-step", type=float, default=0.015)
    p.add_argument("--ensemble-seeds", type=int, default=7)
    p.add_argument("--mixup-alpha", type=float, default=0.3)
    p.add_argument("--lr-min-ratio", type=float, default=0.02)
    p.add_argument("--warmup-epochs", type=int, default=40)
    p.add_argument("--swa-start-frac", type=float, default=0.7)
    p.add_argument("--tta-samples", type=int, default=12)
    return p.parse_args()


def run_experiment(args) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    output_features_dir = Path(str(args.output_features_dir)).expanduser()

    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    # --- Static features ---
    X_train_df = make_feature_matrix(train_df)
    X_test_df = make_feature_matrix(test_df)
    if X_train_df.empty:
        raise RuntimeError("Empty features")

    X_train = X_train_df.to_numpy(float)
    X_test = X_test_df.to_numpy(float)
    mu, sigma = _standardize_fit(X_train)
    X_train = _standardize_apply(X_train, mu, sigma)
    X_test = _standardize_apply(X_test, mu, sigma)
    X_train, X_test = _reduce_dim(X_train, X_test, int(args.feature_max_dim))
    static_feature_dim = X_train.shape[1]
    print(f"[tf+vid] Static features: {static_feature_dim} dims")

    # --- Temporal video features ---
    T = int(args.curve_points)
    train_vids = [str(v).strip() for v in train_df["video_folder"]]
    test_vids = [str(v).strip() for v in test_df["video_folder"]]

    vid_tr, feat_names, n_avail_tr = load_all_video_temporal_features(output_features_dir, train_vids, T)
    vid_te, _, n_avail_te = load_all_video_temporal_features(output_features_dir, test_vids, T)

    if vid_tr.shape[2] > 0 and len(feat_names) > 0:
        vid_tr, vid_te = _standardize_temporal_features(vid_tr, vid_te)
        vid_tr, vid_te = _reduce_temporal_dim(vid_tr, vid_te, int(args.video_feature_max_dim))
        temporal_dim = vid_tr.shape[2]
        print(f"[tf+vid] Temporal video features: {temporal_dim} dims (PCA from {len(feat_names)}, train={n_avail_tr}/{len(train_vids)}, test={n_avail_te}/{len(test_vids)})")
    else:
        temporal_dim = 0
        vid_tr = np.zeros((len(train_vids), T, 0))
        vid_te = np.zeros((len(test_vids), T, 0))
        print("[tf+vid] WARNING: No temporal video features available!")

    # --- Targets ---
    y_train = np.zeros((len(train_df), T), float)
    y_true = np.zeros(T, float)
    for i in range(T):
        col = _point_col(i)
        y_train[:, i] = pd.to_numeric(train_df[col], errors="coerce").fillna(0).to_numpy(float)
        y_true[i] = _safe_float(test_df.iloc[0][col], 0.0)
    y_train, y_true = _clip01(y_train), _clip01(y_true)

    # --- Baselines ---
    n_sin = int(getattr(args, "n_sinusoidal", 4))
    time_features_np = _make_time_features(T, n_sin)
    time_ctx_dim = time_features_np.shape[1]
    global_mean_curve = np.mean(y_train, axis=0)
    percentile_curves = _compute_percentile_curves(y_train)
    integration_train_np = _build_integration_matrix(train_df, snapshot_dir, T)
    integration_test_np = _build_integration_matrix(test_df, snapshot_dir, T)

    baseline_curve = global_mean_curve
    print(f"[tf+vid] baseline (global mean) RMSE to true: {np.sqrt(np.mean((baseline_curve - y_true) ** 2)):.4f}")

    # --- Build sequence inputs with video features ---
    seq_train_np = _make_sequence_inputs_with_video(
        X=X_train,
        time_features=time_features_np,
        primary_baseline=baseline_curve,
        global_mean_curve=global_mean_curve,
        integration_strength=integration_train_np,
        video_temporal=vid_tr,
        percentile_curves=percentile_curves,
    )
    seq_test_np = _make_sequence_inputs_with_video(
        X=X_test,
        time_features=time_features_np,
        primary_baseline=baseline_curve,
        global_mean_curve=global_mean_curve,
        integration_strength=integration_test_np,
        video_temporal=vid_te,
        percentile_curves=percentile_curves,
    )
    print(f"[tf+vid] Sequence input shape: {seq_train_np.shape} (static={static_feature_dim} + time={time_ctx_dim} + base=7+3 + temporal={temporal_dim})")

    # --- Device ---
    device = _resolve_device(getattr(args, "device", "auto"))
    print(f"[tf+vid] device={device.type}")
    try:
        torch.set_num_threads(int(args.torch_num_threads))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # --- Training parameters ---
    d_model = int(args.d_model)
    n_layers = int(args.n_transformer_layers)
    n_heads = int(args.attn_heads)
    ffn_mult = int(args.ffn_mult)
    dropout = float(args.dropout)
    residual_scale = float(args.residual_scale)
    ad_lw = float(args.ad_loss_weight)
    ad_sw = float(args.ad_slope_weight)
    noise_std = float(args.noise_std)
    feat_noise = float(args.feature_noise_std)
    max_inc = float(args.max_increase_per_step)
    epochs = int(args.epochs)
    patience = int(args.patience)
    grad_clip = float(args.grad_clip)
    log_every = int(args.log_every)
    n_ens = int(args.ensemble_seeds)
    mixup_alpha = float(args.mixup_alpha)
    lr_min = float(args.lr_min_ratio)
    warmup = int(args.warmup_epochs)
    swa_frac = float(args.swa_start_frac)
    tta_samples = int(args.tta_samples)
    conv_kernels = [int(k.strip()) for k in str(args.conv_kernels).split(",") if k.strip()]
    base_seed = int(args.random_seed)
    seeds = [base_seed + i * 111 for i in range(n_ens)]

    all_tr, all_te, all_res, all_ad = [], [], [], []
    all_be: list[int] = []
    all_bl: list[float] = []

    for idx, s in enumerate(seeds):
        print(f"\n{'=' * 60}")
        print(f"[tf+vid] ensemble {idx + 1}/{n_ens}, seed={s}")
        print(f"{'=' * 60}")
        tr, te, res, ad, be, bl = _train_single_model(
            seed=s,
            seq_train_np=seq_train_np,
            y_train=y_train,
            integration_train_np=integration_train_np,
            baseline_curve_np=baseline_curve,
            seq_test_np=seq_test_np,
            integration_test_np=integration_test_np,
            device=device,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_mult=ffn_mult,
            dropout=dropout,
            residual_scale=residual_scale,
            conv_kernels=conv_kernels,
            curve_points=T,
            static_feature_dim=static_feature_dim,
            time_ctx_dim=time_ctx_dim,
            ad_loss_weight=ad_lw,
            ad_slope_weight=ad_sw,
            noise_std=noise_std,
            feature_noise_std=feat_noise,
            epochs=epochs,
            patience=patience,
            grad_clip=grad_clip,
            log_every=log_every,
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            lr_min_ratio=lr_min,
            warmup_epochs=warmup,
            swa_start_frac=swa_frac,
            mixup_alpha=mixup_alpha,
            tta_samples=tta_samples,
            ens_idx=idx,
            ens_total=n_ens,
        )
        all_tr.append(tr)
        all_te.append(te)
        all_res.append(res)
        all_ad.append(ad)
        all_be.append(be)
        all_bl.append(bl)

    # --- Ensemble ---
    losses = np.array(all_bl)
    if losses.max() - losses.min() > 1e-10:
        inv = 1.0 / (losses + 1e-8)
        weights = inv / inv.sum()
    else:
        weights = np.ones(n_ens) / n_ens
    print(f"[tf+vid] ensemble weights: {[f'{w:.3f}' for w in weights]}")

    train_pred = sum(w * p for w, p in zip(weights, all_tr, strict=True))
    test_pred = sum(w * p for w, p in zip(weights, all_te, strict=True))
    test_residual = sum(w * p for w, p in zip(weights, all_res, strict=True))
    test_ad_drop = sum(w * p for w, p in zip(weights, all_ad, strict=True))

    y_pred_raw = _clip01(test_pred)
    y_pred = _clip01(_soft_non_increasing(y_pred_raw, max_increase=max_inc))

    # --- Save ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    all_df.to_csv(out_dir / "dataset.csv", index=False)

    pd.DataFrame(
        {
            "point_idx": list(range(T)),
            "pred_retention_base": baseline_curve,
            "pred_retention_lstm_raw": y_pred_raw,
            "pred_retention_residual": test_residual,
            "pred_retention_ad_drop": test_ad_drop,
            "integration_strength": integration_test_np[0] if len(integration_test_np) else np.zeros(T),
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": np.abs(y_pred - y_true),
        }
    ).to_csv(pred_path, index=False)

    train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
    metrics = {
        "videos_total_with_target": len(rows),
        "videos_used": len(all_df),
        "train_videos": len(train_df),
        "curve_points": T,
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(test_df.iloc[0]["drive_file_id"]),
        **_curve_metrics(y_pred, y_true),
        "prediction_path": str(pred_path),
        "d_model": d_model,
        "n_transformer_layers": n_layers,
        "dropout": dropout,
        "residual_scale": residual_scale,
        "ensemble_size": n_ens,
        "tta_samples": tta_samples,
        "best_epochs": all_be,
        "best_losses": all_bl,
        "ensemble_weights": [float(w) for w in weights],
        "train_rmse": train_rmse,
        "model_name": "ad_aware_transformer_video",
        "temporal_video_features": temporal_dim,
        "static_features": static_feature_dim,
        "videos_with_video_features": n_avail_tr,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Transformer + Video Features ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
