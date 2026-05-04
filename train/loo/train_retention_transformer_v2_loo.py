"""
Transformer v2 for retention prediction.
Key differences from v1:
  - kNN-personalised baseline (residuals are smaller → easier to learn)
  - No monotone penalty (retention curves have natural rises)
  - Symmetric step cap post-processing (allows rises up to 0.05/step)
  - No temporal smoothing inside model (more expressive)
  - Larger ensemble (9) + more TTA (16)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID, _point_col, _safe_float, build_rows_with_targets_source, make_feature_matrix, select_train_test
from train.loo.train_retention_lstm_loo import (
    _build_integration_matrix,
    _clip01,
    _compute_percentile_curves,
    _curve_metrics,
    _knn_weighted_baseline,
    _make_time_features,
    _resolve_device,
    _safe_logit,
    _savgol_smooth,
    _standardize_apply,
    _standardize_fit,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer v2 — kNN baseline + non-monotone.")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--snapshot-dir", default="drive_snapshot_90")
    p.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    p.add_argument("--limit-videos", type=int, default=90)
    p.add_argument("--train-videos", type=int, default=89)
    p.add_argument("--curve-points", type=int, default=50)
    p.add_argument("--eval-video-folder", default="")
    p.add_argument("--eval-drive-file-id", default="")
    p.add_argument("--output-dir", default="transformer_v2_experiment")
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


def _smooth_postprocess(curve: np.ndarray, max_step: float = 0.05) -> np.ndarray:
    smoothed = _savgol_smooth(curve, window=7, order=3)
    out = smoothed.copy()
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        if abs(delta) > max_step:
            out[i] = out[i - 1] + max_step * np.sign(delta)
    return out


def _make_sequence_inputs(
    X: np.ndarray, time_features: np.ndarray, knn_baseline: np.ndarray, global_mean: np.ndarray, integration_strength: np.ndarray, percentile_curves: np.ndarray | None = None
) -> np.ndarray:
    n, steps = X.shape[0], time_features.shape[0]
    bl_d1 = np.diff(knn_baseline, prepend=knn_baseline[0])
    bl_d2 = np.diff(knn_baseline, n=2, prepend=[knn_baseline[0], knn_baseline[0]])
    gm_d1 = np.diff(global_mean, prepend=global_mean[0])
    diff_bl = knn_baseline - global_mean

    n_extra = 8
    if percentile_curves is not None:
        n_extra += percentile_curves.shape[0]

    base = np.zeros((n, steps, X.shape[1] + time_features.shape[1] + n_extra), dtype=float)
    c = 0
    base[:, :, c : c + X.shape[1]] = X[:, None, :]
    c += X.shape[1]
    base[:, :, c : c + time_features.shape[1]] = time_features[None, :, :]
    c += time_features.shape[1]
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
    base[:, :, c] = integration_strength
    c += 1
    ad_delta = np.diff(integration_strength, axis=1, prepend=integration_strength[:, :1])
    base[:, :, c] = ad_delta
    c += 1
    if percentile_curves is not None:
        for k in range(percentile_curves.shape[0]):
            base[:, :, c] = percentile_curves[k][None, :]
            c += 1
    return base


class _TemporalConvBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_sizes: list[int], dropout: float):
        super().__init__()
        nb = len(kernel_sizes)
        bc = max(1, channels // nb)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(channels, bc, k, padding=k // 2) for k in kernel_sizes])
        tot = bc * nb
        self.proj = torch.nn.Linear(tot, channels) if tot != channels else torch.nn.Identity()
        self.norm = torch.nn.LayerNorm(channels)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        cat = torch.cat([c(xt) for c in self.convs], dim=1).transpose(1, 2)
        return self.norm(x + self.drop(self.proj(cat)))


class _FeatureGate(torch.nn.Module):
    def __init__(self, fd: int, td: int):
        super().__init__()
        self.g = torch.nn.Sequential(torch.nn.Linear(td, fd * 2), torch.nn.GELU(), torch.nn.Linear(fd * 2, fd), torch.nn.Sigmoid())

    def forward(self, f: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return f * self.g(t)


class RetentionTransformerV2(torch.nn.Module):
    def __init__(
        self,
        input_size,
        d_model=64,
        n_layers=3,
        n_heads=4,
        ffn_mult=2,
        dropout=0.15,
        residual_scale=1.25,
        curve_points=50,
        conv_kernels=None,
        static_feature_dim=0,
        time_ctx_dim=10,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.static_feature_dim = static_feature_dim
        self.time_ctx_dim = time_ctx_dim

        self.feature_gate = _FeatureGate(static_feature_dim, time_ctx_dim) if static_feature_dim > 0 and time_ctx_dim > 0 else None

        self.input_proj = torch.nn.Sequential(torch.nn.Linear(input_size, d_model), torch.nn.LayerNorm(d_model), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.pos_embed = torch.nn.Parameter(torch.randn(1, curve_points, d_model) * 0.02)

        self.pre_conv = _TemporalConvBlock(d_model, conv_kernels, dropout) if conv_kernels else None

        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ffn_mult, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.post_conv = _TemporalConvBlock(d_model, conv_kernels, dropout) if conv_kernels else None

        self.head = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(d_model, 1))
        self.ad_head = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(d_model, 1))

    def forward(self, seq_inputs, baseline_curve, integration_strength):
        x = seq_inputs
        if self.feature_gate is not None and self.static_feature_dim > 0:
            sd, td = self.static_feature_dim, self.time_ctx_dim
            x = torch.cat([self.feature_gate(x[:, :, :sd], x[:, :, sd : sd + td]), x[:, :, sd:]], dim=-1)

        h = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        if self.pre_conv is not None:
            h = self.pre_conv(h)
        h = self.final_norm(self.encoder(h))
        if self.post_conv is not None:
            h = self.post_conv(h)

        raw_res = self.head(h).squeeze(-1)
        smooth_res = self.residual_scale * torch.tanh(raw_res)
        ad_drop = F.softplus(self.ad_head(h).squeeze(-1))
        residual = smooth_res - integration_strength * ad_drop

        bs, steps, _ = seq_inputs.shape
        bl_logit = _safe_logit(baseline_curve)[None, :].expand(bs, steps)
        return torch.sigmoid(bl_logit + residual), residual, ad_drop


def _pearson_loss(pred, target):
    pc = pred - pred.mean(1, keepdim=True)
    tc = target - target.mean(1, keepdim=True)
    return 1.0 - ((pc * tc).sum(1) / torch.sqrt((pc.pow(2).sum(1) + 1e-8) * (tc.pow(2).sum(1) + 1e-8))).mean()


def _spectral_loss(pred, target):
    nk = max(1, pred.shape[1] // 4)
    return (torch.fft.rfft(pred, dim=1)[:, :nk] - torch.fft.rfft(target, dim=1)[:, :nk]).abs().pow(2).mean()


def _endpoint_weights(n, device, hw=2.0, tw=2.5, ramp=5):
    w = torch.ones(n, device=device)
    r = min(ramp, n // 4)
    for i in range(r):
        f = (r - i) / r
        w[i] = 1.0 + (hw - 1.0) * f
        w[-(i + 1)] = 1.0 + (tw - 1.0) * f
    return w


def _curve_loss(pred, target, residual, integ, ad_lw, ad_sw):
    ep_w = _endpoint_weights(pred.shape[1], pred.device)
    pw = (1.0 + ad_lw * integ) * ep_w[None, :]
    data = (pw * F.smooth_l1_loss(pred, target, reduction="none")).mean()

    d = pred[:, 1:] - pred[:, :-1]
    td = target[:, 1:] - target[:, :-1]
    ss = torch.maximum(integ[:, 1:], integ[:, :-1])
    slope = ((1.0 + ad_sw * ss) * F.smooth_l1_loss(d, td, reduction="none")).mean()

    curv = (d[:, 1:] - d[:, :-1]).pow(2).mean() if d.shape[1] > 1 else pred.new_tensor(0.0)
    jitter = (d[:, 1:] + d[:, :-1]).pow(2).mean() if d.shape[1] > 1 else pred.new_tensor(0.0)
    res_pen = residual.pow(2).mean()
    corr = _pearson_loss(pred, target)
    spec = _spectral_loss(pred, target)

    total = data + 0.35 * slope + 0.12 * curv + 0.03 * jitter + 0.008 * res_pen + 0.20 * corr + 0.06 * spec
    return total, {"data": data.item(), "slope": slope.item(), "corr": corr.item(), "total": total.item()}


def _warmup_cosine(opt, warmup, total, lr_min_ratio):
    warmup = max(1, warmup)

    def _lr(step):
        if step < warmup:
            return max(0.01, step / warmup)
        p = (step - warmup) / max(1, total - warmup)
        return lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr)


def _make_train_val_split(n, seed):
    if n < 8:
        idx = np.arange(n)
        return idx, idx[:0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    vs = max(4, min(int(0.15 * n), n - 4))
    return np.sort(perm[vs:]), np.sort(perm[:vs])


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
    tag = f"v2-{eidx + 1}/{etot}"
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


def run_experiment(args) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)
    x_tr_df = make_feature_matrix(train_df)
    x_te_df = make_feature_matrix(test_df)
    if x_tr_df.empty:
        raise RuntimeError("Empty features")

    Xtr = x_tr_df.to_numpy(float)
    Xte = x_te_df.to_numpy(float)
    mu, sig = _standardize_fit(Xtr)
    Xtr = _standardize_apply(Xtr, mu, sig)
    Xte = _standardize_apply(Xte, mu, sig)
    from train.loo.train_retention_lstm_loo import _reduce_dim as _rd

    Xtr, Xte = _rd(Xtr, Xte, int(args.feature_max_dim))
    static_dim = Xtr.shape[1]

    T = int(args.curve_points)
    y_train = np.zeros((len(train_df), T), float)
    y_true = np.zeros(T, float)
    for i in range(T):
        col = _point_col(i)
        y_train[:, i] = pd.to_numeric(train_df[col], errors="coerce").fillna(0).to_numpy(float)
        y_true[i] = _safe_float(test_df.iloc[0][col], 0.0)
    y_train, y_true = _clip01(y_train), _clip01(y_true)

    n_sin = int(getattr(args, "n_sinusoidal", 4))
    tf = _make_time_features(T, n_sin)
    tcd = tf.shape[1]
    global_mean = np.mean(y_train, axis=0)
    pctls = _compute_percentile_curves(y_train)
    int_tr = _build_integration_matrix(train_df, snapshot_dir, T)
    int_te = _build_integration_matrix(test_df, snapshot_dir, T)

    knn_bl = _clip01(_knn_weighted_baseline(Xtr, Xte, y_train, k=int(args.knn_k), temperature=float(args.knn_temperature)))
    print(f"[v2] kNN RMSE to true: {np.sqrt(np.mean((knn_bl - y_true) ** 2)):.4f}")
    print(f"[v2] mean RMSE to true: {np.sqrt(np.mean((global_mean - y_true) ** 2)):.4f}")

    seq_tr = _make_sequence_inputs(Xtr, tf, knn_bl, global_mean, int_tr, pctls)
    seq_te = _make_sequence_inputs(Xte, tf, knn_bl, global_mean, int_te, pctls)

    device = _resolve_device(getattr(args, "device", "auto"))
    print(f"[v2] device={device.type}")
    try:
        torch.set_num_threads(int(getattr(args, "torch_num_threads", 1)))
    except Exception:
        pass

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
        print(f"\n{'=' * 60}\n[v2] ensemble {idx + 1}/{n_ens} seed={s}\n{'=' * 60}")
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

    losses = np.array(all_bl_)
    if losses.max() - losses.min() > 1e-10:
        inv = 1.0 / (losses + 1e-8)
        w = inv / inv.sum()
    else:
        w = np.ones(n_ens) / n_ens
    print(f"[v2] weights: {[f'{x:.3f}' for x in w]}")

    train_pred = sum(wi * p for wi, p in zip(w, all_tr, strict=True))
    test_pred = sum(wi * p for wi, p in zip(w, all_te, strict=True))
    test_res = sum(wi * p for wi, p in zip(w, all_res, strict=True))
    test_ad = sum(wi * p for wi, p in zip(w, all_ad, strict=True))

    y_raw = _clip01(test_pred)
    y_pred = _clip01(_smooth_postprocess(y_raw, max_step=float(args.max_step)))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "holdout_prediction_vs_true.csv"
    all_df.to_csv(out / "dataset.csv", index=False)

    pd.DataFrame(
        {
            "point_idx": list(range(T)),
            "pred_retention_base": knn_bl,
            "pred_retention_lstm_raw": y_raw,
            "pred_retention_residual": test_res,
            "pred_retention_ad_drop": test_ad,
            "integration_strength": int_te[0] if len(int_te) else np.zeros(T),
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
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
        "model_name": "retention_transformer_v2",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Transformer v2 ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
