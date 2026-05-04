"""
Decomposed retention prediction: trend + ad-pattern + content-pattern + meta-combiner.

Stage 1 — Trend:  kNN baseline + CatBoost pointwise + global mean → ridge blend
Stage 2 — Ad:     CatBoost on integration residual (active only where integration > 0.3)
Stage 3 — Content: lightweight Transformer on remaining residual
Stage 4 — Meta:   per-point ridge on OOF predictions from all stages
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from catboost import CatBoostRegressor

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID, _point_col, _safe_float, build_rows_with_targets_source, make_feature_matrix, select_train_test
from train.loo.train_retention_lstm_loo import (
    _build_integration_matrix,
    _clip01,
    _curve_metrics,
    _knn_weighted_baseline,
    _make_time_features,
    _resolve_device,
    _savgol_smooth,
    _standardize_apply,
    _standardize_fit,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decomposed retention pipeline (LOO).")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--snapshot-dir", default="drive_snapshot_90")
    p.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    p.add_argument("--limit-videos", type=int, default=90)
    p.add_argument("--train-videos", type=int, default=89)
    p.add_argument("--curve-points", type=int, default=50)
    p.add_argument("--eval-video-folder", default="")
    p.add_argument("--eval-drive-file-id", default="")
    p.add_argument("--output-dir", default="decomposed_experiment")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    p.add_argument("--oof-folds", type=int, default=5)
    p.add_argument("--meta-l2", type=float, default=0.02)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--knn-temperature", type=float, default=0.5)
    p.add_argument("--cb-iterations", type=int, default=500)
    p.add_argument("--cb-learning-rate", type=float, default=0.05)
    p.add_argument("--cb-depth", type=int, default=6)
    p.add_argument("--ad-cb-iterations", type=int, default=300)
    p.add_argument("--ad-cb-depth", type=int, default=4)
    p.add_argument("--ad-threshold", type=float, default=0.3)
    p.add_argument("--tf-d-model", type=int, default=48)
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--tf-heads", type=int, default=3)
    p.add_argument("--tf-dropout", type=float, default=0.15)
    p.add_argument("--tf-epochs", type=int, default=400)
    p.add_argument("--tf-lr", type=float, default=0.001)
    p.add_argument("--tf-patience", type=int, default=80)
    p.add_argument("--tf-holdout-ensemble", type=int, default=5)
    p.add_argument("--tf-holdout-tta", type=int, default=8)
    p.add_argument("--max-step", type=float, default=0.05)
    p.add_argument("--n-sinusoidal", type=int, default=4)
    p.add_argument("--feature-max-dim", type=int, default=40)
    return p.parse_args()


def _get_target_matrix(df: pd.DataFrame, curve_points: int) -> np.ndarray:
    mat = np.zeros((len(df), curve_points), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        for p in range(curve_points):
            mat[i, p] = _safe_float(row[_point_col(p)], 0.0)
    return _clip01(mat)


def _kfold_indices(n: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        out.append((train, val))
    return out


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    x = np.nan_to_num(x.astype(float), nan=0.0)
    y = np.nan_to_num(y.astype(float), nan=0.0)
    x_aug = np.hstack([x, np.ones((x.shape[0], 1))])
    reg = np.eye(x_aug.shape[1]) * max(l2, 0.0)
    reg[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + reg
    rhs = x_aug.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs + np.eye(lhs.shape[0]) * 1e-8) @ rhs


def _predict_ridge(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1))]) @ beta


def _reduce_dim(X_train: np.ndarray, X_test: np.ndarray, max_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if X_train.shape[1] <= max_dim:
        return X_train.copy(), X_test.copy()
    _, _, Vt = np.linalg.svd(X_train, full_matrices=False)
    keep = int(max(8, min(max_dim, Vt.shape[0])))
    basis = Vt[:keep].T
    return X_train @ basis, X_test @ basis


def _smooth_postprocess(curve: np.ndarray, max_step: float = 0.05) -> np.ndarray:
    smoothed = _savgol_smooth(curve, window=7, order=3)
    out = smoothed.copy()
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        if abs(delta) > max_step:
            out[i] = out[i - 1] + max_step * np.sign(delta)
    return out


def _predict_trend_knn(X_train: np.ndarray, X_pred: np.ndarray, y_train: np.ndarray, k: int, temperature: float) -> np.ndarray:
    """Per-sample kNN baseline."""
    n_pred = X_pred.shape[0]
    out = np.zeros((n_pred, y_train.shape[1]), dtype=float)
    for i in range(n_pred):
        out[i] = _knn_weighted_baseline(X_train, X_pred[i : i + 1], y_train, k=k, temperature=temperature)
    return _clip01(out)


def _predict_trend_catboost(x_train_df: pd.DataFrame, y_train: np.ndarray, x_pred_df: pd.DataFrame, iterations: int, lr: float, depth: int, seed: int) -> np.ndarray:
    n_pred = len(x_pred_df)
    n_points = y_train.shape[1]
    baseline = np.mean(y_train, axis=0)
    residual = y_train - baseline[None, :]
    pred = np.zeros((n_pred, n_points), dtype=float)
    for p in range(n_points):
        tgt = residual[:, p]
        if np.var(tgt) < 1e-12:
            pred[:, p] = float(tgt[0]) if tgt.size else 0.0
            continue
        m = CatBoostRegressor(loss_function="Huber:delta=0.03", eval_metric="MAE", iterations=iterations, learning_rate=lr, depth=depth, random_seed=seed + p, verbose=False)
        m.fit(x_train_df, tgt)
        pred[:, p] = m.predict(x_pred_df)
    return _clip01(baseline[None, :] + pred)


def _predict_trend_mean(y_train: np.ndarray, n_pred: int) -> np.ndarray:
    return np.tile(_clip01(np.mean(y_train, axis=0)), (n_pred, 1))


def _predict_ad_correction(
    x_train_df: pd.DataFrame,
    residual_train: np.ndarray,
    integration_train: np.ndarray,
    trend_train: np.ndarray,
    x_pred_df: pd.DataFrame,
    integration_pred: np.ndarray,
    trend_pred: np.ndarray,
    iterations: int,
    depth: int,
    seed: int,
    threshold: float,
) -> np.ndarray:
    n_points = residual_train.shape[1]
    n_train = len(x_train_df)
    n_pred = len(x_pred_df)
    correction = np.zeros((n_pred, n_points), dtype=float)

    x_tr_np = x_train_df.to_numpy(dtype=float)
    x_pr_np = x_pred_df.to_numpy(dtype=float)

    for p in range(n_points):
        mask = integration_train[:, p] > threshold
        if mask.sum() < 3:
            continue
        feat_tr = np.column_stack([x_tr_np[mask], integration_train[mask, p : p + 1], trend_train[mask, p : p + 1], np.full((mask.sum(), 1), p / max(1, n_points - 1))])
        tgt = residual_train[mask, p]
        if np.var(tgt) < 1e-12:
            continue
        m = CatBoostRegressor(loss_function="Huber:delta=0.05", iterations=iterations, depth=depth, random_seed=seed + p, verbose=False)
        m.fit(feat_tr, tgt)

        if integration_pred[:, p].max() > threshold:
            feat_pr = np.column_stack([x_pr_np, integration_pred[:, p : p + 1], trend_pred[:, p : p + 1], np.full((n_pred, 1), p / max(1, n_points - 1))])
            correction[:, p] = m.predict(feat_pr) * (integration_pred[:, p] > threshold).astype(float)

    return correction


class _ContentTransformer(torch.nn.Module):
    def __init__(self, input_size: int, d_model: int, n_layers: int, n_heads: int, dropout: float, curve_points: int):
        super().__init__()
        self.proj = torch.nn.Sequential(torch.nn.Linear(input_size, d_model), torch.nn.LayerNorm(d_model), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.pos = torch.nn.Parameter(torch.randn(1, curve_points, d_model) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1], :]
        h = self.norm(self.enc(h))
        return self.head(h).squeeze(-1)


def _build_tf_inputs(X: np.ndarray, time_features: np.ndarray, trend: np.ndarray, ad_correction: np.ndarray, integration: np.ndarray) -> np.ndarray:
    n, steps = X.shape[0], time_features.shape[0]
    total = X.shape[1] + time_features.shape[1] + 3
    base = np.zeros((n, steps, total), dtype=float)
    c = 0
    base[:, :, c : c + X.shape[1]] = X[:, None, :]
    c += X.shape[1]
    base[:, :, c : c + time_features.shape[1]] = time_features[None, :, :]
    c += time_features.shape[1]
    base[:, :, c] = trend
    c += 1
    base[:, :, c] = ad_correction
    c += 1
    base[:, :, c] = integration
    return base


def _train_content_transformer(
    seq_np: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    d_model: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    curve_points: int,
    epochs: int,
    lr: float,
    patience: int,
    seed: int,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_t = torch.tensor(seq_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(targets, dtype=torch.float32, device=device)

    n = x_t.shape[0]
    val_n = max(2, int(0.15 * n))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    tr_idx, va_idx = perm[val_n:], perm[:val_n]

    model = _ContentTransformer(seq_np.shape[2], d_model, n_layers, n_heads, dropout, curve_points).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss, best_ep, best_sd = float("inf"), 0, None
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x_t[tr_idx])
        loss = F.smooth_l1_loss(pred, y_t[tr_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if len(va_idx) > 0:
            model.eval()
            with torch.no_grad():
                vl = F.smooth_l1_loss(model(x_t[va_idx]), y_t[va_idx]).item()
        else:
            vl = loss.item()

        if vl < best_loss - 1e-8:
            best_loss, best_ep = vl, ep
            best_sd = copy.deepcopy(model.state_dict())
        if (ep - best_ep) >= patience:
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    model.eval()
    return model


def _predict_content_transformer(model: torch.nn.Module, seq_np: np.ndarray, device: torch.device, n_tta: int = 1) -> np.ndarray:
    model.eval()
    x_t = torch.tensor(seq_np, dtype=torch.float32, device=device)
    preds = []
    with torch.no_grad():
        preds.append(model(x_t).cpu().numpy())
        for _ in range(max(0, n_tta - 1)):
            noised = x_t + 0.01 * torch.randn_like(x_t)
            preds.append(model(noised).cpu().numpy())
    return np.mean(preds, axis=0)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)
    x_train_df = make_feature_matrix(train_df).reset_index(drop=True)
    x_test_df = make_feature_matrix(test_df).reset_index(drop=True)
    if x_train_df.empty:
        raise RuntimeError("Пустая матрица признаков")

    T = int(args.curve_points)
    y_train = _get_target_matrix(train_df, T)
    y_true = _get_target_matrix(test_df, T)[0]
    n_train = len(x_train_df)

    x_train_np = np.nan_to_num(x_train_df.to_numpy(dtype=float), nan=0.0)
    x_test_np = np.nan_to_num(x_test_df.to_numpy(dtype=float), nan=0.0)
    mu, sigma = _standardize_fit(x_train_np)
    x_train_std = _standardize_apply(x_train_np, mu, sigma)
    x_test_std = _standardize_apply(x_test_np, mu, sigma)
    x_train_std, x_test_std = _reduce_dim(x_train_std, x_test_std, int(args.feature_max_dim))
    static_dim = x_train_std.shape[1]

    integration_train = _build_integration_matrix(train_df, snapshot_dir=snapshot_dir, curve_points=T)
    integration_test = _build_integration_matrix(test_df, snapshot_dir=snapshot_dir, curve_points=T)

    n_sin = int(args.n_sinusoidal)
    time_features = _make_time_features(T, n_sinusoidal=n_sin)
    device = _resolve_device(args.device)
    print(f"[decomposed] device={device.type} train={n_train} points={T}")

    folds = _kfold_indices(n_train, int(args.oof_folds), int(args.random_seed))
    n_folds = len(folds)

    oof_trend_knn = np.zeros_like(y_train)
    oof_trend_cb = np.zeros_like(y_train)
    oof_trend_mean = np.zeros_like(y_train)
    oof_trend_blend = np.zeros_like(y_train)
    oof_ad_corr = np.zeros_like(y_train)
    oof_content = np.zeros_like(y_train)

    for fi, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"\n[decomposed] === fold {fi}/{n_folds} train={len(tr_idx)} val={len(va_idx)} ===")
        x_tr_df = x_train_df.iloc[tr_idx].reset_index(drop=True)
        x_va_df = x_train_df.iloc[va_idx].reset_index(drop=True)
        y_tr = y_train[tr_idx]
        x_tr_s = x_train_std[tr_idx]
        x_va_s = x_train_std[va_idx]
        int_tr = integration_train[tr_idx]
        int_va = integration_train[va_idx]

        print(f"[decomposed] fold {fi} stage=1 trend models")
        oof_trend_knn[va_idx] = _predict_trend_knn(x_tr_s, x_va_s, y_tr, args.knn_k, args.knn_temperature)
        oof_trend_cb[va_idx] = _predict_trend_catboost(x_tr_df, y_tr, x_va_df, args.cb_iterations, args.cb_learning_rate, args.cb_depth, args.random_seed + fi * 100)
        oof_trend_mean[va_idx] = _predict_trend_mean(y_tr, len(va_idx))

        oof_trend_blend[va_idx] = (oof_trend_knn[va_idx] + oof_trend_cb[va_idx] + oof_trend_mean[va_idx]) / 3.0

        print(f"[decomposed] fold {fi} stage=2 ad-pattern")
        residual_from_trend = y_tr - (
            (
                _predict_trend_knn(x_tr_s, x_tr_s, y_tr, args.knn_k, args.knn_temperature)
                + _predict_trend_catboost(x_tr_df, y_tr, x_tr_df, args.cb_iterations, args.cb_learning_rate, args.cb_depth, args.random_seed + fi * 100)
                + _predict_trend_mean(y_tr, len(tr_idx))
            )
            / 3.0
        )
        oof_ad_corr[va_idx] = _predict_ad_correction(
            x_tr_df,
            residual_from_trend,
            int_tr,
            (oof_trend_knn[tr_idx] + oof_trend_cb[tr_idx] + oof_trend_mean[tr_idx]) / 3.0 if fi > 1 else np.tile(np.mean(y_tr, axis=0), (len(tr_idx), 1)),
            x_va_df,
            int_va,
            oof_trend_blend[va_idx],
            args.ad_cb_iterations,
            args.ad_cb_depth,
            args.random_seed + fi * 200,
            args.ad_threshold,
        )

        print(f"[decomposed] fold {fi} stage=3 content-pattern transformer")
        trend_tr_approx = (
            _predict_trend_knn(x_tr_s, x_tr_s, y_tr, args.knn_k, args.knn_temperature)
            + _predict_trend_catboost(x_tr_df, y_tr, x_tr_df, args.cb_iterations, args.cb_learning_rate, args.cb_depth, args.random_seed + fi * 100)
            + _predict_trend_mean(y_tr, len(tr_idx))
        ) / 3.0
        ad_tr_approx = _predict_ad_correction(
            x_tr_df,
            y_tr - trend_tr_approx,
            int_tr,
            trend_tr_approx,
            x_tr_df,
            int_tr,
            trend_tr_approx,
            args.ad_cb_iterations,
            args.ad_cb_depth,
            args.random_seed + fi * 200,
            args.ad_threshold,
        )
        content_target_tr = y_tr - trend_tr_approx - ad_tr_approx

        seq_tr = _build_tf_inputs(x_tr_s, time_features, trend_tr_approx, ad_tr_approx, int_tr)
        seq_va = _build_tf_inputs(x_va_s, time_features, oof_trend_blend[va_idx], oof_ad_corr[va_idx], int_va)

        model = _train_content_transformer(
            seq_tr,
            content_target_tr,
            device,
            args.tf_d_model,
            args.tf_layers,
            args.tf_heads,
            args.tf_dropout,
            T,
            args.tf_epochs,
            args.tf_lr,
            args.tf_patience,
            seed=args.random_seed + fi * 300,
        )
        oof_content[va_idx] = _predict_content_transformer(model, seq_va, device, n_tta=1)
        del model
        if device.type != "cpu":
            torch.cuda.empty_cache() if device.type == "cuda" else None

    print("\n[decomposed] training Stage 1 trend ridge blend on OOF")
    trend_betas: list[np.ndarray] = []
    oof_trend_blended = np.zeros_like(y_train)
    for p in range(T):
        x_meta = np.column_stack([oof_trend_knn[:, p], oof_trend_cb[:, p], oof_trend_mean[:, p]])
        beta = _fit_ridge(x_meta, y_train[:, p], args.meta_l2)
        trend_betas.append(beta)
        oof_trend_blended[:, p] = _predict_ridge(x_meta, beta)

    print("[decomposed] training Stage 4 meta-combiner on OOF")
    oof_stage12 = oof_trend_blended + oof_ad_corr
    oof_stage123 = oof_stage12 + oof_content

    meta_betas: list[np.ndarray] = []
    for p in range(T):
        p_frac = p / max(1, T - 1)
        x_meta = np.column_stack(
            [
                oof_trend_blended[:, p],
                oof_stage12[:, p],
                oof_stage123[:, p],
                np.full(n_train, p_frac),
                integration_train[:, p],
                np.abs(oof_stage123[:, p] - oof_trend_blended[:, p]),
            ]
        )
        beta = _fit_ridge(x_meta, y_train[:, p], args.meta_l2)
        meta_betas.append(beta)

    print("\n[decomposed] === holdout prediction ===")

    print("[decomposed] holdout stage=1 trend")
    ho_trend_knn = _predict_trend_knn(x_train_std, x_test_std, y_train, args.knn_k, args.knn_temperature)
    ho_trend_cb = _predict_trend_catboost(x_train_df, y_train, x_test_df, args.cb_iterations, args.cb_learning_rate, args.cb_depth, args.random_seed + 900)
    ho_trend_mean = _predict_trend_mean(y_train, 1)

    ho_trend_blend = np.zeros((1, T), dtype=float)
    for p in range(T):
        x_m = np.column_stack([ho_trend_knn[:, p], ho_trend_cb[:, p], ho_trend_mean[:, p]])
        ho_trend_blend[:, p] = _predict_ridge(x_m, trend_betas[p])
    ho_trend_blend = _clip01(ho_trend_blend)

    print("[decomposed] holdout stage=2 ad-pattern")
    trend_train_full = np.zeros((n_train, T), dtype=float)
    for p in range(T):
        x_m = np.column_stack(
            [
                _predict_trend_knn(x_train_std, x_train_std, y_train, args.knn_k, args.knn_temperature)[:, p],
                _predict_trend_catboost(x_train_df, y_train, x_train_df, args.cb_iterations, args.cb_learning_rate, args.cb_depth, args.random_seed + 900)[:, p],
                _predict_trend_mean(y_train, n_train)[:, p],
            ]
        )
        trend_train_full[:, p] = _predict_ridge(x_m, trend_betas[p])
    trend_train_full = _clip01(trend_train_full)

    residual_from_trend_full = y_train - trend_train_full
    ho_ad_corr = _predict_ad_correction(
        x_train_df,
        residual_from_trend_full,
        integration_train,
        trend_train_full,
        x_test_df,
        integration_test,
        ho_trend_blend,
        args.ad_cb_iterations,
        args.ad_cb_depth,
        args.random_seed + 800,
        args.ad_threshold,
    )

    print("[decomposed] holdout stage=3 content transformer (ensemble)")
    ad_train_full = _predict_ad_correction(
        x_train_df,
        residual_from_trend_full,
        integration_train,
        trend_train_full,
        x_train_df,
        integration_train,
        trend_train_full,
        args.ad_cb_iterations,
        args.ad_cb_depth,
        args.random_seed + 800,
        args.ad_threshold,
    )
    content_target_full = y_train - trend_train_full - ad_train_full
    seq_train_tf = _build_tf_inputs(x_train_std, time_features, trend_train_full, ad_train_full, integration_train)
    seq_test_tf = _build_tf_inputs(x_test_std, time_features, ho_trend_blend, ho_ad_corr, integration_test)

    n_ens = int(args.tf_holdout_ensemble)
    n_tta = int(args.tf_holdout_tta)
    content_preds = []
    for ei in range(n_ens):
        s = args.random_seed + 500 + ei * 111
        print(f"[decomposed] holdout tf ensemble {ei + 1}/{n_ens} seed={s}")
        m = _train_content_transformer(
            seq_train_tf, content_target_full, device, args.tf_d_model, args.tf_layers, args.tf_heads, args.tf_dropout, T, args.tf_epochs, args.tf_lr, args.tf_patience, seed=s
        )
        content_preds.append(_predict_content_transformer(m, seq_test_tf, device, n_tta=n_tta))
        del m
    ho_content = np.mean(content_preds, axis=0)

    print("[decomposed] holdout stage=4 meta-combine")
    ho_s12 = ho_trend_blend + ho_ad_corr
    ho_s123 = ho_s12 + ho_content
    y_pred = np.zeros(T, dtype=float)
    for p in range(T):
        p_frac = p / max(1, T - 1)
        x_m = np.array(
            [[ho_trend_blend[0, p], ho_s12[0, p], ho_s123[0, p], p_frac, integration_test[0, p] if len(integration_test) else 0.0, abs(ho_s123[0, p] - ho_trend_blend[0, p])]]
        )
        y_pred[p] = _predict_ridge(x_m, meta_betas[p])[0]

    y_pred_raw = _clip01(y_pred)
    y_pred_final = _clip01(_smooth_postprocess(y_pred_raw, max_step=float(args.max_step)))

    abs_err = np.abs(y_pred_final - y_true)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    metrics_path = out_dir / "metrics.json"
    all_df.to_csv(out_dir / "dataset.csv", index=False)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(T)),
            "pred_trend_knn": ho_trend_knn[0],
            "pred_trend_cb": ho_trend_cb[0],
            "pred_trend_blend": ho_trend_blend[0],
            "pred_ad_correction": ho_ad_corr[0],
            "pred_content_pattern": ho_content[0],
            "pred_meta_raw": y_pred_raw,
            "integration_strength": integration_test[0] if len(integration_test) else np.zeros(T),
            "pred_retention": y_pred_final,
            "pred_retention_norm": y_pred_final,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )
    result_df.to_csv(pred_path, index=False)

    metrics = {
        "videos_total_with_target": len(rows),
        "videos_used": len(all_df),
        "train_videos": len(train_df),
        "curve_points": T,
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(test_df.iloc[0]["drive_file_id"]),
        **_curve_metrics(y_pred_final, y_true),
        "prediction_path": str(pred_path),
        "oof_folds": int(args.oof_folds),
        "knn_k": args.knn_k,
        "cb_iterations": args.cb_iterations,
        "tf_d_model": args.tf_d_model,
        "tf_layers": args.tf_layers,
        "tf_holdout_ensemble": n_ens,
        "tf_holdout_tta": n_tta,
        "model_name": "decomposed_trend_ad_content_v1",
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Decomposed Retention LOO ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
