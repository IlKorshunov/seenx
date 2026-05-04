"""
Shared data utilities for sequence-based retention trainers (Transformer, LSTM).

Handles:
- Loading per-second features from output_aligned/
- Loading per-video LLM features from drive_snapshot_90/
- Merging both into a unified per-second DataFrame
- Filtering out redundant, noisy, and low-information features
- Building windowed PyTorch datasets with is_ad mask for ad-penalty loss
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

NON_FEATURE_COLS = {"time", "retention", "time_sec", "frame", "video_folder", "transcript_path", "drive_file_id"}

LLM_ID_COLS = {"video_folder", "transcript_path", "drive_file_id"}


# ---------------------------------------------------------------------------
# Engagement-based video weights
# ---------------------------------------------------------------------------


def load_video_weights(video_ids: list[str], snapshot_dir: str | Path, weight_min: float = 0.25, weight_max: float = 4.0) -> dict[str, float]:
    """Return per-video loss weights based on engagement (views, likes, comments).

    Score = log1p(views) + 5·log1p(likes) + 10·log1p(comments).
    Weights are normalized so the mean over video_ids equals 1.0,
    then clipped to [weight_min, weight_max] to prevent any single
    video from dominating or being ignored.
    Videos without meta.json get weight 1.0 before normalization.
    """
    snapshot_dir = Path(snapshot_dir)
    raw: dict[str, float] = {}
    for vid in video_ids:
        meta_path = snapshot_dir / vid / "meta.json"
        if not meta_path.exists():
            raw[vid] = 1.0
            continue
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            score = np.log1p(float(m.get("view_count", 0))) + 5.0 * np.log1p(float(m.get("like_count", 0))) + 10.0 * np.log1p(float(m.get("comment_count", 0)))
            raw[vid] = max(score, 1e-3)
        except Exception:
            raw[vid] = 1.0

    scores = np.array([raw[v] for v in video_ids], dtype=np.float64)
    mean_score = scores.mean()
    if mean_score < 1e-9:
        mean_score = 1.0
    weights = {v: float(np.clip(raw[v] / mean_score, weight_min, weight_max)) for v in video_ids}
    logger.info("Video weights (engagement-based): min=%.3f  max=%.3f  mean=%.3f", min(weights.values()), max(weights.values()), np.mean(list(weights.values())))
    return weights


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _extract_numeric_llm_cols(llm: dict) -> dict[str, float]:
    return {
        (f"llm_{k}" if not str(k).startswith("llm_") else str(k)): float(v)
        for k, v in llm.items()
        if k not in LLM_ID_COLS and not str(k).startswith("target__") and isinstance(v, (int, float))
    }


def _load_output_features(vid: str, output_dir: Path) -> pd.DataFrame | None:
    """Load per-second features from output/.

    Supports two layouts:
      - output/<vid>_features.csv  (flat CSV, index=time)
      - output/<vid>/features_readable.csv  (subdirectory layout from output_aligned)
    """
    flat_path, subdir_path = output_dir / f"{vid}_features.csv", output_dir / vid / "features_readable.csv"
    if flat_path.exists():
        df = pd.read_csv(flat_path, index_col=0)
        if df.index.name == "time":
            df = df.reset_index()
    elif subdir_path.exists():
        df = pd.read_csv(subdir_path)
    else:
        return None

    if "retention" not in df.columns:
        return None
    if "time" in df.columns:
        try:
            df["time_sec"] = pd.to_timedelta(df["time"]).dt.total_seconds()
        except Exception:
            pass
    return df


def _load_curve_raw(vid: str, snapshot_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load curve_raw from retention_parsed.json or retention.json.
    Returns (curve_values, time_ratios) or None. curve_values in 0-1 scale."""
    for candidate in [snapshot_dir / vid / "retention_parsed.json", snapshot_dir / vid / "transcripts" / "retention_parsed.json", snapshot_dir / vid / "retention.json"]:
        if not candidate.exists():
            continue
        try:
            ret_data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue

        if isinstance(ret_data, dict) and ret_data.get("status") == "ok":
            raw = ret_data.get("curve_raw", [])
            if not isinstance(raw, list) or len(raw) < 5:
                raw = ret_data.get("curve_20", [])
            if isinstance(raw, list) and len(raw) >= 5:
                curve = np.array([float(v) for v in raw], dtype=np.float64)
                time_ratios = np.linspace(0, 1, len(curve))
                return curve, time_ratios
        elif isinstance(ret_data, list) and len(ret_data) >= 5:
            curve = np.array([float(pt.get("audience_watch_ratio", 0)) for pt in ret_data], dtype=np.float64)
            time_ratios = np.array([float(pt.get("time_ratio", i / (len(ret_data) - 1))) for i, pt in enumerate(ret_data)], dtype=np.float64)
            return curve, time_ratios
    return None


def _load_llm_features(vid: str, snapshot_dir: Path) -> dict | None:
    feat_path = snapshot_dir / vid / "features_llm.json"
    if not feat_path.exists():
        feat_path = snapshot_dir / vid / "transcripts" / "features_llm.json"
    if not feat_path.exists():
        return None
    try:
        payload = json.loads(feat_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    flat = payload.get("video_features_flat", {})
    return flat if isinstance(flat, dict) and flat else None


def load_merged_video(vid: str, output_dir: Path, snapshot_dir: Path, use_curve_raw: bool = False) -> pd.DataFrame | None:
    """Load per-second features + broadcast LLM features onto every row.
    If use_curve_raw and curve_raw available, resample to curve_raw resolution."""
    df = _load_output_features(vid, output_dir)
    if df is None or df.empty:
        return None

    if use_curve_raw:
        curve_data = _load_curve_raw(vid, snapshot_dir)
        if curve_data is not None:
            curve, time_ratios = curve_data
            duration_sec = max(1, len(df) - 1)
            indices = (time_ratios * duration_sec).astype(int).clip(0, len(df) - 1)
            df_resampled = df.iloc[indices].copy().reset_index(drop=True)
            df_resampled["retention"] = curve * 100.0
            df = df_resampled

    llm = _load_llm_features(vid, snapshot_dir)
    if llm is not None:
        llm_cols = _extract_numeric_llm_cols(llm)
        if llm_cols:
            df = pd.concat([df, pd.DataFrame({k: [v] * len(df) for k, v in llm_cols.items()}, index=df.index)], axis=1)

    return df


def _load_snapshot_only_video(vid: str, snapshot_dir: Path) -> pd.DataFrame | None:
    """Build DataFrame from snapshot: curve_raw at native resolution (no interpolation)."""
    curve_data = _load_curve_raw(vid, snapshot_dir)
    if curve_data is None:
        return None
    curve, _ = curve_data

    df = pd.DataFrame({"retention": curve * 100.0})
    llm = _load_llm_features(vid, snapshot_dir)
    if llm is not None:
        llm_cols = _extract_numeric_llm_cols(llm)
        if llm_cols:
            df = pd.concat([df, pd.DataFrame({k: [v] * len(df) for k, v in llm_cols.items()}, index=df.index)], axis=1)

    return df


def load_all_merged(output_dir: str | Path = "output", snapshot_dir: str | Path = "drive_snapshot_90", use_curve_raw: bool = True) -> dict[str, pd.DataFrame]:
    """Load all videos: output/ (full features) + snapshot-only (LLM + retention).
    If use_curve_raw=True, target is curve_raw at native resolution (20-101 points)."""
    output_dir = Path(output_dir)
    snapshot_dir = Path(snapshot_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"output dir not found: {output_dir}")

    video_dfs: dict[str, pd.DataFrame] = {}

    output_vids: set = set()
    for p in sorted(output_dir.iterdir()):
        if p.is_file() and p.name.endswith("_features.csv") and not p.name.endswith(".partial"):
            output_vids.add(p.name.replace("_features.csv", ""))
        elif p.is_dir() and (p / "features_readable.csv").exists():
            output_vids.add(p.name)

    for vid in sorted(output_vids):
        df = load_merged_video(vid, output_dir, snapshot_dir, use_curve_raw=use_curve_raw)
        if df is not None and "retention" in df.columns:
            df = df.dropna(subset=["retention"])
            if len(df) >= 10:
                video_dfs[vid] = df
                logger.info("Loaded %s: %d rows, %d cols (output+llm)", vid, len(df), len(df.columns))

    if snapshot_dir.exists():
        for entry in sorted(snapshot_dir.iterdir()):
            if not entry.is_dir() or entry.name in video_dfs:
                continue
            vid = entry.name
            df = _load_snapshot_only_video(vid, snapshot_dir)
            if df is not None and len(df) >= 10:
                video_dfs[vid] = df
                logger.info("Loaded %s: %d rows, %d cols (llm-only)", vid, len(df), len(df.columns))

    if not video_dfs:
        raise RuntimeError("No valid videos found")
    logger.info("Total videos loaded: %d (%d from output, %d snapshot-only)", len(video_dfs), len(output_vids & set(video_dfs)), len(video_dfs) - len(output_vids & set(video_dfs)))
    return video_dfs


def _load_redundant_pairs(results_dir: Path, threshold: float = 0.85) -> list[tuple[str, str, float]]:
    csv_path = results_dir / "correlation" / "redundant_pairs.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    pairs = []
    for _, row in df.iterrows():
        corr = abs(float(row.get("correlation", 0)))
        if corr >= threshold:
            pairs.append((str(row["feature_a"]), str(row["feature_b"]), corr))
    return pairs


def _load_master_ranking(results_dir: Path) -> dict[str, float]:
    csv_path = results_dir / "master_ranking.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, index_col=0)
    if "avg_rank" not in df.columns:
        return {}
    return df["avg_rank"].to_dict()


def filter_features(
    video_dfs: dict[str, pd.DataFrame],
    results_dir: str | Path = "analysis/feature_importance/results",
    redundant_corr_threshold: float = 0.85,
    min_nonzero_pct: float = 0.01,
    max_nan_pct: float = 0.50,
    top_k: int | None = None,
) -> tuple[list[str], list[str]]:
    """
    Select good feature columns from the merged DataFrames.

    Returns (kept_features, log_lines) where log_lines describes what was dropped.
    """
    results_dir = Path(results_dir)
    log: list[str] = []

    all_cols = set()
    for df in video_dfs.values():
        all_cols.update(df.columns)

    candidates = sorted(
        c
        for c in all_cols
        if c not in NON_FEATURE_COLS and not str(c).startswith("target__") and any(pd.api.types.is_numeric_dtype(df[c]) for df in video_dfs.values() if c in df.columns)
    )
    log.append(f"Initial candidates: {len(candidates)}")

    # 1) Remove zero-variance
    drop_zerovar = set()
    for col in candidates:
        values = []
        for df in video_dfs.values():
            if col in df.columns:
                values.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
        if not values or np.std(values) < 1e-8:
            drop_zerovar.add(col)
    candidates = [c for c in candidates if c not in drop_zerovar]
    log.append(f"Dropped zero-variance ({len(drop_zerovar)}): {sorted(drop_zerovar)}")

    # 2) Remove high-NaN
    drop_nan = set()
    for col in candidates:
        total, nans = 0, 0
        for df in video_dfs.values():
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                total += len(s)
                nans += int(s.isna().sum())
            else:
                total += len(df)
                nans += len(df)
        if total > 0 and nans / total > max_nan_pct:
            drop_nan.add(col)
    candidates = [c for c in candidates if c not in drop_nan]
    log.append(f"Dropped high-NaN ({len(drop_nan)}): {sorted(drop_nan)}")

    # 3) Remove low nonzero percentage
    drop_sparse = set()
    for col in candidates:
        values = []
        for df in video_dfs.values():
            if col in df.columns:
                values.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
        if values:
            nonzero_pct = np.count_nonzero(values) / len(values)
            if nonzero_pct < min_nonzero_pct:
                drop_sparse.add(col)
    candidates = [c for c in candidates if c not in drop_sparse]
    log.append(f"Dropped sparse ({len(drop_sparse)}): {sorted(drop_sparse)}")

    ranking = _load_master_ranking(results_dir)
    pairs = _load_redundant_pairs(results_dir, redundant_corr_threshold)
    drop_redundant = set()
    for fa, fb, corr in pairs:
        if fa not in candidates or fb not in candidates:
            continue
        if fa in drop_redundant or fb in drop_redundant:
            continue
        rank_a = ranking.get(fa, 999)
        rank_b = ranking.get(fb, 999)
        drop = fb if rank_a <= rank_b else fa
        drop_redundant.add(drop)
        log.append(f"  Redundant: {fa} <-> {fb} (rho={corr:.3f}), drop {drop}")
    candidates = [c for c in candidates if c not in drop_redundant]
    log.append(f"Dropped redundant ({len(drop_redundant)}): {sorted(drop_redundant)}")

    if top_k is not None and top_k > 0 and ranking:
        ranked = sorted(candidates, key=lambda c: ranking.get(c, 999))
        dropped_topk = set(ranked[top_k:])
        candidates = ranked[:top_k]
        log.append(f"Top-K filter (k={top_k}), dropped {len(dropped_topk)}")

    log.append(f"Final features: {len(candidates)}")
    log.append(f"Kept: {candidates}")
    return candidates, log


class FeatureNormalizer:
    """Z-score normalization for features; min-max for retention (-> 0..1)."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.ret_min: float = 0.0
        self.ret_max: float = 100.0

    def fit(self, video_dfs: dict[str, pd.DataFrame], feature_cols: list[str]):
        all_values = []
        all_ret = []
        for df in video_dfs.values():
            arr = df.reindex(columns=feature_cols, fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0).values
            all_values.append(arr)
            ret = pd.to_numeric(df["retention"], errors="coerce").dropna().values
            if len(ret):
                all_ret.append(ret)
        stacked = np.vstack(all_values)
        self.mean = stacked.mean(axis=0)
        self.std = stacked.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        if all_ret:
            all_ret = np.concatenate(all_ret)
            self.ret_min = float(np.percentile(all_ret, 1))
            self.ret_max = float(np.percentile(all_ret, 99))
            if self.ret_max - self.ret_min < 1.0:
                self.ret_min, self.ret_max = 0.0, 100.0

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def normalize_retention(self, ret: np.ndarray) -> np.ndarray:
        return (ret - self.ret_min) / (self.ret_max - self.ret_min)

    def denormalize_retention(self, ret: np.ndarray) -> np.ndarray:
        return ret * (self.ret_max - self.ret_min) + self.ret_min


class WindowedSeqDataset(Dataset):
    def __init__(
        self,
        video_dfs: dict[str, pd.DataFrame],
        video_ids: list[str],
        feature_cols: list[str],
        normalizer: FeatureNormalizer | None = None,
        window_size: int = 128,
        stride: int = 64,
        video_weights: dict[str, float] | None = None,
    ):
        self.window_size = window_size
        self.feature_cols = feature_cols
        # (X, y, is_ad, real_len, weight)
        self.windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, float]] = []

        for vid in video_ids:
            df = video_dfs[vid]
            w = float(video_weights[vid]) if video_weights and vid in video_weights else 1.0

            X = df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
            if normalizer is not None:
                nan_mask = np.isnan(X)
                X = np.nan_to_num(X, nan=0.0)
                X = normalizer.transform(X).astype(np.float32)
                X[nan_mask] = 0.0  # missing features -> neutral after z-score
            else:
                X = np.nan_to_num(X, nan=0.0)
            y = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values.astype(np.float32)
            if normalizer is not None:
                y = normalizer.normalize_retention(y).astype(np.float32)

            is_ad_col = df["is_ad"].values.astype(np.float32) if "is_ad" in df.columns else np.zeros(len(df), dtype=np.float32)

            n = len(X)
            if n <= window_size:
                self.windows.append((X, y, is_ad_col, n, w))
            else:
                for s in range(0, n - window_size + 1, stride):
                    self.windows.append((X[s : s + window_size], y[s : s + window_size], is_ad_col[s : s + window_size], window_size, w))
                if (n - window_size) % stride != 0:
                    self.windows.append((X[n - window_size :], y[n - window_size :], is_ad_col[n - window_size :], window_size, w))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        X, y, is_ad, real_len, weight = self.windows[idx]
        ws = self.window_size
        if len(X) < ws:
            pad = ws - len(X)
            X = np.pad(X, ((0, pad), (0, 0)))
            y = np.pad(y, (0, pad))
            is_ad = np.pad(is_ad, (0, pad))
            mask = np.array([False] * real_len + [True] * pad)
        else:
            mask = np.zeros(ws, dtype=bool)
        return {
            "features": torch.tensor(X, dtype=torch.float32),
            "retention": torch.tensor(y, dtype=torch.float32),
            "is_ad": torch.tensor(is_ad, dtype=torch.float32),
            "padding_mask": torch.tensor(mask),
            "video_weight": torch.tensor(weight, dtype=torch.float32),
        }


def ad_aware_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_ad: torch.Tensor,
    padding_mask: torch.Tensor,
    base_criterion: torch.nn.Module,
    ad_overpredict_weight: float = 3.0,
    video_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss with ad-overpredict penalty and optional per-video engagement weight.

    video_weight: (batch_size,) tensor; if provided, each sample's loss is scaled
    by its video's engagement weight (normalized to mean=1 over train set).
    """
    elem_loss = base_criterion(pred, target)
    weights = torch.ones_like(elem_loss)
    weights[(pred > target) & (is_ad > 0.5)] = ad_overpredict_weight
    if video_weight is not None:
        # broadcast (B,) -> (B, T)
        weights = weights * video_weight.view(-1, 1)
    elem_loss = elem_loss * weights
    elem_loss[padding_mask] = 0.0
    valid = (~padding_mask).sum().clamp(min=1)
    return elem_loss.sum() / valid


def seq_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    d_pred = np.diff(y_pred)
    d_true = np.diff(y_true)
    dd_pred = np.diff(y_pred, n=2)
    dd_true = np.diff(y_true, n=2)

    sp = float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman"))
    pe = float(pd.Series(y_pred).corr(pd.Series(y_true), method="pearson"))

    return {
        "spearman": sp if not np.isnan(sp) else 0.0,
        "pearson": pe if not np.isnan(pe) else 0.0,
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(abs_err)),
        "spike_rmse": float(np.sqrt(np.mean((d_pred - d_true) ** 2))) if d_pred.size else 0.0,
        "curvature_rmse": float(np.sqrt(np.mean((dd_pred - dd_true) ** 2))) if dd_pred.size else 0.0,
    }


@torch.no_grad()
def predict_video(
    model: torch.nn.Module, df: pd.DataFrame, feature_cols: list[str], normalizer: FeatureNormalizer | None, device: torch.device, window_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    X = df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    if normalizer is not None:
        X = np.nan_to_num(X, nan=0.0)
        X = normalizer.transform(X).astype(np.float32)
        X[np.isnan(X)] = 0.0
    else:
        X = np.nan_to_num(X, nan=0.0)
    y_true = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values

    n = len(X)
    if n <= window_size:
        t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(t).squeeze(0).cpu().numpy()[:n]
        if normalizer is not None:
            pred = normalizer.denormalize_retention(pred)
        return y_true, pred

    pred_sum, pred_cnt = np.zeros(n), np.zeros(n)
    for s in range(0, n - window_size + 1):
        t = torch.tensor(X[s : s + window_size], dtype=torch.float32).unsqueeze(0).to(device)
        p = model(t).squeeze(0).cpu().numpy()
        pred_sum[s : s + window_size] += p
        pred_cnt[s : s + window_size] += 1.0
    pred = pred_sum / np.maximum(pred_cnt, 1.0)
    if normalizer is not None:
        pred = normalizer.denormalize_retention(pred)
    return y_true, pred
