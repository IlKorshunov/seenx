from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas.api.typing import NaTType
from scipy.signal import savgol_filter
from torch.utils.data import Dataset

from analysis.augmentations import RetentionAugmentation


_EMB_ALIGN_PATH = Path(__file__).resolve().parents[2] / "src" / "utils" / "embedding_aligner.py"
_emb_spec = importlib.util.spec_from_file_location("embedding_aligner", _EMB_ALIGN_PATH)
if _emb_spec is None or _emb_spec.loader is None:
    raise ImportError(f"cannot load {_EMB_ALIGN_PATH}")
_embedding_aligner = importlib.util.module_from_spec(_emb_spec)
_emb_spec.loader.exec_module(_embedding_aligner)
PerModalityPCA = _embedding_aligner.PerModalityPCA
load_aligned_embeddings = _embedding_aligner.load_aligned_embeddings

logger = logging.getLogger(__name__)

NON_FEATURE_COLS = {"time", "retention", "time_sec", "frame", "video_folder", "transcript_path", "drive_file_id", "early_retention_drop_30s"}

HARMFUL_FEATURES = {
    "llm_linguistic__avg_word_length",
    "llm_linguistic__flesch_reading_ease",
    "llm_integration_present",
    "llm_integration_suggested_next_step_present",
    "llm_integration_confidence",
    "frame",
    "ad_density_percent",
    "n_ad_segments",
    "has_org_mention",
    "is_ad_x_viewer_address",
}

LLM_ID_COLS = {"video_folder", "transcript_path", "drive_file_id"}

COUNT_LIKE_FEATURES = {"crutch_cnt", "has_person_mention", "has_org_mention", "object_count", "unique_classes", "viewer_address", "wps", "question_density"}


def load_video_weights(video_ids: list[str], snapshot_dir: str | Path, weight_min: float = 0.25, weight_max: float = 4.0) -> dict[str, float]:
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
    mean_score = max(scores.mean(), 1e-9)
    weights = {v: float(np.clip(raw[v] / mean_score, weight_min, weight_max)) for v in video_ids}
    logger.info("Video weights: min=%.3f  max=%.3f  mean=%.3f", min(weights.values()), max(weights.values()), np.mean(list(weights.values())))
    return weights


def _extract_numeric_llm_cols(llm: dict) -> dict[str, float]:
    return {
        (f"llm_{k}" if not str(k).startswith("llm_") else str(k)): float(v)
        for k, v in llm.items()
        if k not in LLM_ID_COLS and not str(k).startswith("target__") and isinstance(v, (int, float))
    }


def _load_output_features(vid: str, output_dir: Path) -> pd.DataFrame | None:
    flat_path, subdir_path = output_dir / f"{vid}_features.csv", output_dir / vid / "features_readable.csv"
    if flat_path.exists():
        df = pd.read_csv(flat_path, index_col=0)
        df = df.copy()
        if df.index.name == "time":
            df = df.reset_index()
    elif subdir_path.exists():
        df = pd.read_csv(subdir_path)
    else:
        return None
    df = df.copy()
    if "retention" not in df.columns:
        return None
    if "time" in df.columns:
        try:
            df["time_sec"] = pd.to_timedelta(df["time"]).dt.total_seconds()
        except Exception:
            pass
    return df


def _load_curve_raw(vid: str, snapshot_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
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
                return np.array([float(v) for v in raw], dtype=np.float64), np.linspace(0, 1, len(raw))
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


def _read_video_meta(vid: str, snapshot_dir: Path) -> dict:
    candidates = [snapshot_dir / vid / "meta.json", snapshot_dir / vid / "transcripts" / "meta.json"]
    for p in candidates:
        if not p.exists():
            continue
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            return meta if isinstance(meta, dict) else {}
        except Exception:
            continue
    return {}


def _meta_duration_sec(meta: dict, df: pd.DataFrame) -> float:
    if isinstance(meta, dict) and "duration" in meta:
        try:
            val = float(meta["duration"])
            if np.isfinite(val) and val > 0:
                return float(val)
        except Exception:
            pass

    if "time_sec" in df.columns:
        ts = pd.to_numeric(df["time_sec"], errors="coerce").dropna().values
        if len(ts) > 0:
            return float(np.max(ts))
    return float(max(len(df) - 1, 0))


def _meta_view_count(meta: dict) -> float:
    c = meta.get("view_count")
    if c is None:
        return 0.0
    try:
        v = float(c)
    except Exception:
        return 0.0
    return float(v)


def _meta_published_at(meta: dict) -> pd.Timestamp | NaTType:
    raw = meta.get("upload_date") or meta.get("published_at") or meta.get("publishedAt")
    if not raw:
        return pd.NaT
    ts = pd.to_datetime(raw, errors="coerce", utc=True)
    return ts


def _add_video_level_features(video_dfs: dict[str, pd.DataFrame], snapshot_dir: Path) -> dict[str, pd.DataFrame]:
    if not video_dfs:
        return video_dfs

    rows = []
    for vid, df in video_dfs.items():
        meta = _read_video_meta(vid, snapshot_dir)
        ret = pd.to_numeric(df.get("retention", pd.Series(dtype=float)), errors="coerce").dropna().values
        ret_mean = float(np.mean(ret)) if len(ret) > 0 else 0.0
        if len(ret) >= 30:
            r5 = float(np.mean(ret[:5])) if len(ret) >= 5 else float(ret[0])
            r30 = float(np.mean(ret[25:35])) if len(ret) >= 35 else float(np.mean(ret[25:30]))
            early_drop = float(np.clip((r5 - r30) / max(r5, 1e-6), 0.0, 1.0))
        else:
            early_drop = 0.0

        rows.append(
            {
                "vid": vid,
                "published_at": _meta_published_at(meta),
                "ret_mean": ret_mean,
                "early_retention_drop_30s": early_drop,
                "duration_sec": _meta_duration_sec(meta, df),
                "log1p_view_count": float(np.log1p(max(_meta_view_count(meta), 0.0))),
            }
        )

    meta_df = pd.DataFrame(rows)
    overall_ret_mean = float(meta_df["ret_mean"].mean()) if len(meta_df) > 0 else 0.0
    meta_df["mean_retention_prior"] = np.nan

    known = meta_df[meta_df["published_at"].notna()].sort_values(["published_at", "vid"]).copy()
    run_sum, run_cnt = 0.0, 0
    for i, row in known.iterrows():
        prior = (run_sum / run_cnt) if run_cnt > 0 else overall_ret_mean
        meta_df.loc[i, "mean_retention_prior"] = float(prior)
        run_sum += float(row["ret_mean"])
        run_cnt += 1

    meta_df["mean_retention_prior"] = meta_df["mean_retention_prior"].fillna(overall_ret_mean).astype(np.float32)
    by_vid = meta_df.set_index("vid")

    for vid, df in video_dfs.items():
        r = by_vid.loc[vid]
        extra = pd.DataFrame(
            {
                "duration_sec": np.full(len(df), float(r["duration_sec"]), dtype=np.float32),
                "log1p_view_count": np.full(len(df), float(r["log1p_view_count"]), dtype=np.float32),
                "mean_retention_prior": np.full(len(df), float(r["mean_retention_prior"]), dtype=np.float32),
                "early_retention_drop_30s": np.full(len(df), float(r["early_retention_drop_30s"]), dtype=np.float32),
            },
            index=df.index,
        )
        video_dfs[vid] = pd.concat([df, extra], axis=1)

    logger.info("Added per-video features: duration_sec, log1p_view_count, mean_retention_prior, early_retention_drop_30s")
    return video_dfs


def _broadcast_llm(df: pd.DataFrame, llm: dict | None) -> pd.DataFrame:
    if llm is None:
        return df
    llm_cols = _extract_numeric_llm_cols(llm)
    if llm_cols:
        df = pd.concat([df, pd.DataFrame({k: [v] * len(df) for k, v in llm_cols.items()}, index=df.index)], axis=1)
    return df


def load_merged_video(vid: str, output_dir: Path, snapshot_dir: Path, use_curve_raw: bool = False) -> pd.DataFrame | None:
    df = _load_output_features(vid, output_dir)
    if df is None or df.empty:
        return None
    if use_curve_raw:
        curve_data = _load_curve_raw(vid, snapshot_dir)
        if curve_data is not None:
            curve, time_ratios = curve_data
            indices = (time_ratios * max(1, len(df) - 1)).astype(int).clip(0, len(df) - 1)
            df = df.iloc[indices].copy().reset_index(drop=True)
            df["retention"] = curve * 100.0
    return _broadcast_llm(df, _load_llm_features(vid, snapshot_dir))


def _load_snapshot_only_video(vid: str, snapshot_dir: Path) -> pd.DataFrame | None:
    curve_data = _load_curve_raw(vid, snapshot_dir)
    if curve_data is None:
        return None
    df = pd.DataFrame({"retention": curve_data[0] * 100.0})
    return _broadcast_llm(df, _load_llm_features(vid, snapshot_dir))


def _merge_embedding_pca(video_dfs: dict[str, pd.DataFrame], embeddings_root: str | Path = "embeddings", n_components: int = 12) -> dict[str, pd.DataFrame]:
    embeddings_root = Path(embeddings_root)
    if not embeddings_root.exists():
        logger.warning("Embeddings root %s not found, skipping PCA features", embeddings_root)
        return video_dfs

    raw_aligned: dict[str, np.ndarray] = {}
    for vid, df in video_dfs.items():
        try:
            aligned, _ = load_aligned_embeddings(vid, embeddings_root, duration_sec=len(df))
            if aligned.shape[0] > 0 and aligned.shape[1] > 0:
                raw_aligned[vid] = aligned
        except Exception as e:
            logger.debug("No embeddings for %s: %s", vid, e)

    if not raw_aligned:
        logger.warning("No embeddings found for any video, skipping PCA features")
        return video_dfs

    all_aligned = np.vstack(list(raw_aligned.values()))
    pca = PerModalityPCA(n_components=n_components)
    pca.fit(all_aligned)
    evr = pca.explained_variance_ratio()
    for mod, ratios in evr.items():
        logger.info("Embedding PCA %s: %.1f%% explained by %d components", mod, ratios.sum() * 100, len(ratios))

    pca_col_names = [f"emb_vis_pc{i}" for i in range(n_components)] + [f"emb_aud_pc{i}" for i in range(n_components)] + [f"emb_txt_pc{i}" for i in range(n_components)]

    for vid, df in video_dfs.items():
        if vid not in raw_aligned:
            pca_df = pd.DataFrame(np.zeros((len(df), len(pca_col_names)), dtype=np.float32), columns=pca_col_names, index=df.index)
        else:
            reduced = pca.transform(raw_aligned[vid])
            n_rows = min(len(df), reduced.shape[0])
            buf = np.zeros((len(df), len(pca_col_names)), dtype=np.float32)
            buf[:n_rows] = reduced[:n_rows]
            pca_df = pd.DataFrame(buf, columns=pca_col_names, index=df.index)
        video_dfs[vid] = pd.concat([df, pca_df], axis=1)

    logger.info("Embedding PCA: added %d features (%d videos with embeddings / %d total)", len(pca_col_names), len(raw_aligned), len(video_dfs))
    return video_dfs


def load_all_merged(
    output_dir: str | Path = "output",
    snapshot_dir: str | Path = "drive_snapshot_90",
    use_curve_raw: bool = True,
    embeddings_root: str | Path = "embeddings",
    emb_pca_components: int = 12,
    min_duration_sec: float = 0,
    max_duration_sec: float = 0,
) -> dict[str, pd.DataFrame]:
    output_dir, snapshot_dir = Path(output_dir), Path(snapshot_dir)
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
                r = pd.to_numeric(df["retention"], errors="coerce")
                df = df.copy()
                df["retention"] = np.clip(r.fillna(0.0), 0.0, 100.0)
                video_dfs[vid] = df
                logger.info("Loaded %s: %d rows, %d cols (output+llm)", vid, len(df), len(df.columns))

    if snapshot_dir.exists():
        for entry in sorted(snapshot_dir.iterdir()):
            if not entry.is_dir() or entry.name in video_dfs:
                continue
            df = _load_snapshot_only_video(entry.name, snapshot_dir)
            if df is not None and len(df) >= 10:
                video_dfs[entry.name] = df
                logger.info("Loaded %s: %d rows, %d cols (llm-only)", entry.name, len(df), len(df.columns))

    if not video_dfs:
        raise RuntimeError("No valid videos found")
    logger.info("Total videos loaded: %d (%d output, %d snapshot-only)", len(video_dfs), len(output_vids & set(video_dfs)), len(video_dfs) - len(output_vids & set(video_dfs)))

    video_dfs = _add_video_level_features(video_dfs, snapshot_dir)

    if min_duration_sec > 0:
        before = len(video_dfs)
        video_dfs = {vid: df for vid, df in video_dfs.items() if df["duration_sec"].iloc[0] >= min_duration_sec}
        dropped = before - len(video_dfs)
        if dropped:
            logger.info("Dropped %d videos shorter than %.0f s (%d remain)", dropped, min_duration_sec, len(video_dfs))
        if not video_dfs:
            raise RuntimeError(f"No videos remain after min_duration_sec={min_duration_sec}")

    if max_duration_sec > 0:
        before = len(video_dfs)
        video_dfs = {vid: df for vid, df in video_dfs.items() if df["duration_sec"].iloc[0] <= max_duration_sec}
        dropped = before - len(video_dfs)
        if dropped:
            logger.info("Dropped %d videos longer than %.0f s (%d remain)", dropped, max_duration_sec, len(video_dfs))
        if not video_dfs:
            raise RuntimeError(f"No videos remain after max_duration_sec={max_duration_sec}")

    if emb_pca_components > 0:
        video_dfs = _merge_embedding_pca(video_dfs, embeddings_root, emb_pca_components)

    return video_dfs


def _load_redundant_pairs(results_dir: Path, threshold: float = 0.85) -> list[tuple[str, str, float]]:
    csv_path = results_dir / "correlation" / "redundant_pairs.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    return [(str(row["feature_a"]), str(row["feature_b"]), abs(float(row.get("correlation", 0)))) for _, row in df.iterrows() if abs(float(row.get("correlation", 0))) >= threshold]


def _load_master_ranking(results_dir: Path) -> dict[str, float]:
    csv_path = results_dir / "master_ranking.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, index_col=0)
    return df["avg_rank"].to_dict() if "avg_rank" in df.columns else {}


def filter_features(
    video_dfs: dict[str, pd.DataFrame],
    results_dir: str | Path = "analysis/feature_importance/results",
    redundant_corr_threshold: float = 0.85,
    min_nonzero_pct: float = 0.01,
    max_nan_pct: float = 0.50,
    top_k: int | None = None,
) -> tuple[list[str], list[str]]:
    results_dir = Path(results_dir)
    log: list[str] = []

    all_cols = set()
    for df in video_dfs.values():
        all_cols.update(df.columns)
    candidates = sorted(
        c
        for c in all_cols
        if c not in NON_FEATURE_COLS
        and c not in HARMFUL_FEATURES
        and not str(c).startswith("target__")
        and any(pd.api.types.is_numeric_dtype(df[c]) for df in video_dfs.values() if c in df.columns)
    )
    log.append(f"Initial candidates: {len(candidates)}")

    def _gather(col):
        vals = []
        for df in video_dfs.values():
            if col in df.columns:
                vals.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
        return vals

    drop_zerovar = {c for c in candidates if (v := _gather(c)) == [] or np.std(v) < 1e-8}
    candidates = [c for c in candidates if c not in drop_zerovar]
    log.append(f"Dropped zero-variance ({len(drop_zerovar)}): {sorted(drop_zerovar)}")

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

    drop_sparse = set()
    for col in candidates:
        vals = _gather(col)
        if vals and np.count_nonzero(vals) / len(vals) < min_nonzero_pct:
            drop_sparse.add(col)
    candidates = [c for c in candidates if c not in drop_sparse]
    log.append(f"Dropped sparse ({len(drop_sparse)}): {sorted(drop_sparse)}")

    ranking = _load_master_ranking(results_dir)
    pairs = _load_redundant_pairs(results_dir, redundant_corr_threshold)
    drop_redundant = set()
    for fa, fb, corr in pairs:
        if fa not in candidates or fb not in candidates or fa in drop_redundant or fb in drop_redundant:
            continue
        drop = fb if ranking.get(fa, 999) <= ranking.get(fb, 999) else fa
        drop_redundant.add(drop)
        log.append(f"  Redundant: {fa} <-> {fb} (rho={corr:.3f}), drop {drop}")
    candidates = [c for c in candidates if c not in drop_redundant]
    log.append(f"Dropped redundant ({len(drop_redundant)}): {sorted(drop_redundant)}")

    if top_k is not None and top_k > 0 and ranking:
        ranked = sorted(candidates, key=lambda c: ranking.get(c, 999))
        candidates = ranked[:top_k]
        log.append(f"Top-K filter (k={top_k}), dropped {len(ranked) - top_k}")

    log.append(f"Final features: {len(candidates)}")
    log.append(f"Kept: {candidates}")
    return candidates, log


class FeatureNormalizer:
    """Robust normalization: median/IQR features, winsorized p1/p99; log1p for count-like and llm_* columns."""

    def __init__(self):
        self.median: np.ndarray | None = None
        self.iqr: np.ndarray | None = None
        self.log_mask: np.ndarray | None = None
        self.ret_min: float = 0.0
        self.ret_max: float = 100.0

    def fit(self, video_dfs: dict[str, pd.DataFrame], feature_cols: list[str]):
        self.log_mask = np.array([c in COUNT_LIKE_FEATURES or c.startswith("llm_") for c in feature_cols])

        all_values, all_ret = [], []
        for df in video_dfs.values():
            arr = np.array(df.reindex(columns=feature_cols, fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0).values, dtype=np.float64, copy=True)
            arr[:, self.log_mask] = np.log1p(np.abs(arr[:, self.log_mask])) * np.sign(arr[:, self.log_mask])
            all_values.append(arr)
            ret = pd.to_numeric(df["retention"], errors="coerce").dropna().values
            if len(ret):
                all_ret.append(ret)

        stacked = np.vstack(all_values)

        p1 = np.percentile(stacked, 1, axis=0)
        p99 = np.percentile(stacked, 99, axis=0)
        stacked = np.clip(stacked, p1, p99)

        self.median = np.median(stacked, axis=0)
        q25 = np.percentile(stacked, 25, axis=0)
        q75 = np.percentile(stacked, 75, axis=0)
        self.iqr = q75 - q25
        self.iqr[self.iqr < 1e-8] = 1.0

        if all_ret:
            all_ret = np.concatenate(all_ret)
            self.ret_min = float(np.percentile(all_ret, 1))
            self.ret_max = float(np.percentile(all_ret, 99))
            if self.ret_max - self.ret_min < 1.0:
                self.ret_min, self.ret_max = 0.0, 100.0

    def transform(self, arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr, copy=True)
        if self.log_mask is not None:
            arr[:, self.log_mask] = np.log1p(np.abs(arr[:, self.log_mask])) * np.sign(arr[:, self.log_mask])
        return (arr - self.median) / self.iqr

    def normalize_retention(self, ret: np.ndarray) -> np.ndarray:
        return (ret - self.ret_min) / (self.ret_max - self.ret_min)

    def denormalize_retention(self, ret: np.ndarray) -> np.ndarray:
        return ret * (self.ret_max - self.ret_min) + self.ret_min


def _tabular_X_from_df(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").values.astype(np.float32)


def _apply_norm_X(X: np.ndarray, normalizer: FeatureNormalizer | None) -> np.ndarray:
    if normalizer is None:
        return np.nan_to_num(X, nan=0.0)
    nan_mask = np.isnan(X)
    X = np.nan_to_num(X, nan=0.0)
    X = normalizer.transform(X).astype(np.float32)
    X[nan_mask] = 0.0
    return X


def _tabular_array_from_df(df: pd.DataFrame, feature_cols: list[str], normalizer: FeatureNormalizer | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = _apply_norm_X(_tabular_X_from_df(df, feature_cols), normalizer)
    y = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values.astype(np.float32)
    if normalizer is not None:
        y = normalizer.normalize_retention(y).astype(np.float32)
    is_ad = df["is_ad"].values.astype(np.float32) if "is_ad" in df.columns else np.zeros(len(df), dtype=np.float32)

    # Convex-up regions from the actual retention curve: discrete 2nd derivative
    # d2[t]=y[t]-2y[t-1]+y[t-2], then long rolling mean (~25s) so triggers follow
    # sustained upward curvature, not one-step noise. First derivative is not gated.
    y_raw = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values.astype(np.float32)
    n = len(y_raw)
    d2 = np.zeros_like(y_raw)
    if n > 2:
        d2[2:] = y_raw[2:] - 2.0 * y_raw[1:-1] + y_raw[:-2]
    roll_sec = 25
    d2_smooth = d2.copy()
    if n > 0 and roll_sec > 1:
        d2_smooth = pd.Series(d2, dtype=np.float64).rolling(roll_sec, center=True, min_periods=max(1, roll_sec // 3)).mean().values.astype(np.float32)
    d2_smooth_min = 0.02
    spike_triggers = np.zeros_like(y_raw)
    spike_triggers[(d2_smooth > d2_smooth_min) & (is_ad < 0.5)] = 1.0

    return X, y, is_ad, spike_triggers


def _window_start_indices(n: int, window_size: int, stride: int) -> list[int]:
    if n <= window_size:
        return [0]
    starts = list(range(0, n - window_size + 1, stride))
    if (n - window_size) % stride != 0:
        starts.append(n - window_size)
    return starts


def _align_embedding_rows(emb: np.ndarray | None, n_rows: int, emb_dim: int) -> np.ndarray:
    if emb is None:
        return np.zeros((n_rows, emb_dim), dtype=np.float32)
    emb = emb.astype(np.float32)
    if len(emb) < n_rows:
        emb = np.vstack([emb, np.zeros((n_rows - len(emb), emb.shape[1]), dtype=np.float32)])
    elif len(emb) > n_rows:
        emb = emb[:n_rows]
    return emb


def _augment_tabular_features(X: np.ndarray, feature_mask_prob: float, noise_std: float) -> np.ndarray:
    if feature_mask_prob > 0:
        X = RetentionAugmentation.apply_random_augmentation(X, feature_mask_prob)
    if noise_std > 0:
        X = X + np.random.randn(*X.shape).astype(np.float32) * noise_std
    return X.astype(np.float32)


def _prediction_tabular_X(df: pd.DataFrame, feature_cols: list[str], normalizer: FeatureNormalizer | None) -> np.ndarray:
    return _apply_norm_X(_tabular_X_from_df(df, feature_cols), normalizer)


TIME_FEATURE_MODES = ("none", "frac", "frac_sec")


def time_feature_extra_dim(mode: str) -> int:
    return {"none": 0, "frac": 1, "frac_sec": 2}.get(mode, 0)


def _time_sec_per_row(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if "time_sec" in df.columns:
        return pd.to_numeric(df["time_sec"], errors="coerce").fillna(0).values.astype(np.float32)
    return np.arange(n, dtype=np.float32)


def max_time_sec_over_videos(video_dfs: dict[str, pd.DataFrame], video_ids: list[str]) -> float:
    m = 1.0
    for vid in video_ids:
        df = video_dfs.get(vid)
        if df is None or len(df) == 0:
            continue
        ts = _time_sec_per_row(df)
        m = max(m, float(np.nanmax(ts)) if ts.size else float(len(df)))
    return max(m, 1.0)


def resample_dataframe_to_n_points(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or len(df) == n:
        return df.copy()
    if len(df) < 2:
        out = df.copy()
        while len(out) < n:
            out = pd.concat([out, out.iloc[-1:]], ignore_index=True)
        return out.iloc[:n].copy()
    t_old = np.linspace(0.0, 1.0, len(df))
    t_new = np.linspace(0.0, 1.0, n)
    idx_nn = (t_new * (len(df) - 1)).round().astype(int).clip(0, len(df) - 1)
    rows: list[tuple[str, np.ndarray]] = []
    for col in df.columns:
        if col in ("video_folder", "transcript_path", "drive_file_id"):
            rows.append((col, df[col].iloc[idx_nn].values))
            continue
        if df[col].dtype == object:
            rows.append((col, df[col].iloc[idx_nn].values))
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= 2:
            v = s.interpolate(limit_direction="both").fillna(0).values.astype(np.float64)
            rows.append((col, np.interp(t_new, t_old, v).astype(np.float32)))
        else:
            flat = float(s.fillna(0).iloc[0]) if len(s) else 0.0
            rows.append((col, np.full(n, flat, dtype=np.float32)))
    return pd.DataFrame(dict(rows))


def resample_video_dfs_to_curve_points(video_dfs: dict[str, pd.DataFrame], n_points: int) -> dict[str, pd.DataFrame]:
    if n_points <= 0:
        return video_dfs
    out = {vid: resample_dataframe_to_n_points(df, n_points) for vid, df in video_dfs.items()}
    logger.info("Resampled all videos to %d time points (curve_points mode)", n_points)
    return out


def resample_embedding_time_series(emb: np.ndarray, target_len: int) -> np.ndarray:
    if emb.shape[0] == target_len:
        return emb.astype(np.float32)
    if emb.shape[0] < 2:
        return np.vstack([emb] * target_len)[:target_len].astype(np.float32)
    t_old = np.linspace(0.0, 1.0, emb.shape[0])
    t_new = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, emb.shape[1]), dtype=np.float32)
    for d in range(emb.shape[1]):
        out[:, d] = np.interp(t_new, t_old, emb[:, d].astype(np.float64)).astype(np.float32)
    return out


def resample_embeddings_to_match_dfs(video_embeddings: dict[str, np.ndarray], video_dfs: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    return {vid: (emb if vid not in video_dfs else resample_embedding_time_series(emb, len(video_dfs[vid]))) for vid, emb in video_embeddings.items()}


def _append_time_features_to_matrix(
    X: np.ndarray, start: int, n_full: int, ws: int, real_len: int, time_sec_full: np.ndarray | None, mode: str, ref_time_sec_max: float
) -> np.ndarray:
    if mode == "none" or time_feature_extra_dim(mode) == 0:
        return X
    frac = np.zeros(ws, dtype=np.float32)
    for i in range(real_len):
        frac[i] = float(start + i) / max(1, n_full - 1)
    if real_len < ws:
        frac[real_len:] = frac[real_len - 1] if real_len > 0 else 0.0
    parts = [X, frac.reshape(ws, 1)]
    if mode == "frac_sec":
        sec = np.zeros(ws, dtype=np.float32)
        if time_sec_full is not None and len(time_sec_full) >= start + real_len:
            sec[:real_len] = time_sec_full[start : start + real_len] / max(ref_time_sec_max, 1e-6)
        if real_len < ws:
            sec[real_len:] = sec[real_len - 1] if real_len > 0 else 0.0
        parts.append(sec.reshape(ws, 1))
    return np.concatenate(parts, axis=1)


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
        feature_mask_prob: float = 0.0,
        noise_std: float = 0.0,
        time_feature_mode: str = "none",
        ref_time_sec_max: float = 1.0,
    ):
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.feature_mask_prob = feature_mask_prob
        self.noise_std = noise_std
        self.time_feature_mode = time_feature_mode if time_feature_mode in TIME_FEATURE_MODES else "none"
        self.ref_time_sec_max = float(max(ref_time_sec_max, 1e-6))
        self.windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, int, np.ndarray | None, int]] = []
        for vid in video_ids:
            df = video_dfs[vid]
            w = float(video_weights[vid]) if video_weights and vid in video_weights else 1.0
            ts_full = _time_sec_per_row(df)
            X, y, is_ad_col, spike_triggers = _tabular_array_from_df(df, feature_cols, normalizer)
            n = len(X)
            tf = ts_full if self.time_feature_mode != "none" else None
            if n <= window_size:
                self.windows.append((X, y, is_ad_col, spike_triggers, n, w, 0, tf, n))
            else:
                for s in _window_start_indices(n, window_size, stride):
                    self.windows.append(
                        (X[s : s + window_size], y[s : s + window_size], is_ad_col[s : s + window_size], spike_triggers[s : s + window_size], window_size, w, s, tf, n)
                    )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        X, y, is_ad, spike_triggers, real_len, weight, start, ts_full, n_full = self.windows[idx]
        ws = self.window_size
        if len(X) < ws:
            pad = ws - len(X)
            X, y, is_ad = np.pad(X, ((0, pad), (0, 0))), np.pad(y, (0, pad)), np.pad(is_ad, (0, pad))
            spike_triggers = np.pad(spike_triggers, (0, pad))
            mask = np.array([False] * real_len + [True] * pad)
        else:
            mask = np.zeros(ws, dtype=bool)
        X = _append_time_features_to_matrix(
            _augment_tabular_features(X, self.feature_mask_prob, self.noise_std), start, n_full, ws, real_len, ts_full, self.time_feature_mode, self.ref_time_sec_max
        )
        return {
            "features": torch.from_numpy(X),
            "retention": torch.from_numpy(y),
            "is_ad": torch.from_numpy(is_ad),
            "spike_triggers": torch.from_numpy(spike_triggers),
            "padding_mask": torch.from_numpy(mask),
            "video_weight": torch.tensor(weight, dtype=torch.float32),
        }


def _pearson_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B = pred.size(0)
    losses = []
    for i in range(B):
        valid = ~mask[i]
        p, t = pred[i][valid], target[i][valid]
        if len(p) < 5:
            losses.append(torch.tensor(0.0, device=pred.device))
            continue
        p_c = p - p.mean()
        t_c = t - t.mean()
        num = (p_c * t_c).sum()
        den = (p_c.norm() * t_c.norm()).clamp(min=1e-8)
        losses.append(1.0 - num / den)
    return torch.stack(losses).mean()


def _smoothness_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d1 = pred[:, 1:] - pred[:, :-1]
    m1 = ~mask[:, 1:] & ~mask[:, :-1]
    d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    m2 = ~mask[:, 2:] & ~mask[:, 1:-1] & ~mask[:, :-2]
    l1 = (d1[m1] ** 2).mean() if m1.sum() > 0 else torch.tensor(0.0, device=pred.device)
    l2 = (d2[m2] ** 2).mean() if m2.sum() > 0 else torch.tensor(0.0, device=pred.device)
    return 0.5 * l1 + 0.5 * l2


def _monotonicity_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d = pred[:, 1:] - pred[:, :-1]
    m = ~mask[:, 1:] & ~mask[:, :-1]
    if m.sum() < 1:
        return torch.tensor(0.0, device=pred.device)
    increases = F.relu(d[m])
    return (increases**2).mean()


def _delta_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_pred = pred[:, 1:] - pred[:, :-1]
    d_true = target[:, 1:] - target[:, :-1]
    m = ~mask[:, 1:] & ~mask[:, :-1]
    if m.sum() < 1:
        return torch.tensor(0.0, device=pred.device)
    return F.smooth_l1_loss(d_pred[m], d_true[m])


def _start_weight_mask(length: int, boost_secs: int = 15, boost_factor: float = 2.0, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    w = torch.ones(length, device=device)
    w[:boost_secs] = boost_factor
    return w


def _slope_curvature_weights(target: torch.Tensor, padding_mask: torch.Tensor, slope_alpha: float = 1.0, curv_alpha: float = 0.6, max_boost: float = 5.0) -> torch.Tensor:
    B, T = target.shape
    w = torch.ones_like(target)
    if T < 3:
        return w
    slope = torch.zeros_like(target)
    slope[:, 1:] = (target[:, 1:] - target[:, :-1]).abs()
    curv = torch.zeros_like(target)
    curv[:, 2:] = (target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]).abs()
    valid = ~padding_mask
    for arr, alpha in [(slope, slope_alpha), (curv, curv_alpha)]:
        vals = arr[valid]
        if vals.numel() > 0:
            mx = vals.quantile(0.95).clamp(min=1e-6)
            w = w + alpha * (arr / mx).clamp(max=2.0)
    return w.clamp(max=max_boost)


def composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_ad: torch.Tensor,
    spike_triggers: torch.Tensor,
    padding_mask: torch.Tensor,
    ad_overpredict_weight: float = 3.0,
    video_weight: torch.Tensor | None = None,
    alpha_corr: float = 0.3,
    alpha_smooth: float = 0.15,
    alpha_mono: float = 0.03,
    start_boost_secs: int = 15,
    start_boost_factor: float = 2.0,
    alpha_delta: float = 0.4,
    slope_weight_alpha: float = 1.0,
    curv_weight_alpha: float = 0.6,
    spike_penalty_weight: float = 15.0,
) -> torch.Tensor:
    huber = F.smooth_l1_loss(pred, target, reduction="none")

    weights = torch.ones_like(huber)
    weights[(pred > target) & (is_ad > 0.5)] = ad_overpredict_weight

    # New penalty: If spike trigger is active but prediction drops or is flat compared to previous sec
    if spike_penalty_weight > 0 and spike_triggers is not None:
        diff_pred = torch.zeros_like(pred)
        diff_pred[:, 1:] = pred[:, 1:] - pred[:, :-1]

        # Penalize if trigger is active (spike expected) but predicted diff is non-positive
        missing_spike_mask = (spike_triggers > 0.5) & (diff_pred <= 0)
        weights[missing_spike_mask] *= spike_penalty_weight

    if slope_weight_alpha > 0 or curv_weight_alpha > 0:
        sc_w = _slope_curvature_weights(target, padding_mask, slope_weight_alpha, curv_weight_alpha)
        weights = weights * sc_w

    if start_boost_secs > 0:
        sw = _start_weight_mask(pred.size(1), start_boost_secs, start_boost_factor, pred.device)
        weights = weights * sw.unsqueeze(0)

    if video_weight is not None:
        weights = weights * video_weight.view(-1, 1)
    huber = huber * weights
    huber[padding_mask] = 0.0
    valid = (~padding_mask).sum().clamp(min=1)
    huber_term = huber.sum() / valid

    corr_term = _pearson_loss(pred, target, padding_mask) if alpha_corr > 0 else 0.0
    delta_term = _delta_loss(pred, target, padding_mask) if alpha_delta > 0 else 0.0
    smooth_term = _smoothness_loss(pred, padding_mask) if alpha_smooth > 0 else 0.0
    mono_term = _monotonicity_loss(pred, padding_mask) if alpha_mono > 0 else 0.0

    return huber_term + alpha_corr * corr_term + alpha_delta * delta_term + alpha_smooth * smooth_term + alpha_mono * mono_term


def ad_aware_loss(pred, target, is_ad, spike_triggers, padding_mask, base_criterion, ad_overpredict_weight=3.0, video_weight=None):
    return composite_loss(pred, target, is_ad, spike_triggers, padding_mask, ad_overpredict_weight, video_weight)


def smooth_predictions(y_pred: np.ndarray, window: int = 15, polyorder: int = 3) -> np.ndarray:
    if len(y_pred) < window:
        return y_pred
    return savgol_filter(y_pred, window_length=window, polyorder=polyorder).astype(y_pred.dtype)


def calibrate_scale(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    if len(y_pred) < 5:
        return y_pred
    A = np.vstack([y_pred, np.ones(len(y_pred))]).T
    try:
        result = np.linalg.lstsq(A, y_true, rcond=None)
        a, b = result[0]
        a = float(np.clip(a, 0.5, 2.0))
        b = float(np.mean(y_true) - a * np.mean(y_pred))
        b = float(np.clip(b, -50.0, 50.0))
    except Exception:
        return y_pred
    return (a * y_pred + b).astype(y_pred.dtype)


def seq_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    d_pred, d_true = np.diff(y_pred), np.diff(y_true)
    dd_pred, dd_true = np.diff(y_pred, n=2), np.diff(y_true, n=2)
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


def _apply_ad_drop(pred: np.ndarray, is_ad: np.ndarray | None, max_drop: float = 5.0) -> np.ndarray:
    if is_ad is None:
        return pred
    mask = is_ad > 0.5
    if not mask.any():
        return pred
    out = pred.copy()
    out[mask] -= max_drop * is_ad[mask]
    return out


@torch.no_grad()
def predict_video(
    model: torch.nn.Module,
    df: pd.DataFrame,
    feature_cols: list[str],
    normalizer: FeatureNormalizer | None,
    device: torch.device,
    window_size: int = 128,
    stride: int = 1,
    smooth_window: int = 15,
    apply_smoothing: bool = False,
    time_feature_mode: str = "none",
    ref_time_sec_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    X = _prediction_tabular_X(df, feature_cols, normalizer)
    y_true = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values
    is_ad = df["is_ad"].values.astype(np.float32) if "is_ad" in df.columns else None
    ts_full = _time_sec_per_row(df)
    n_full = len(X)
    tf_mode = time_feature_mode if time_feature_mode in TIME_FEATURE_MODES else "none"
    ref_sec = float(max(ref_time_sec_max, 1e-6))

    n = len(X)
    if n <= window_size:
        Xw = X
        real_len = n
        if Xw.shape[0] < window_size:
            pad = window_size - Xw.shape[0]
            Xw = np.pad(Xw, ((0, pad), (0, 0)))
        Xw = _append_time_features_to_matrix(Xw, 0, n_full, window_size, real_len, ts_full, tf_mode, ref_sec)
        t = torch.tensor(Xw, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(t).squeeze(0).cpu().numpy()[:n]
        if normalizer is not None:
            pred = normalizer.denormalize_retention(pred)
        pred = _apply_ad_drop(pred, is_ad)
        if apply_smoothing and smooth_window > 1:
            pred = smooth_predictions(pred, window=smooth_window)
        return y_true, pred

    pred_sum, pred_cnt = np.zeros(n), np.zeros(n)
    for s in range(0, n - window_size + 1, stride):
        Xw = _append_time_features_to_matrix(X[s : s + window_size], s, n_full, window_size, window_size, ts_full, tf_mode, ref_sec)
        t = torch.tensor(Xw, dtype=torch.float32).unsqueeze(0).to(device)
        p = model(t).squeeze(0).cpu().numpy()
        pred_sum[s : s + window_size] += p
        pred_cnt[s : s + window_size] += 1.0
    pred = pred_sum / np.maximum(pred_cnt, 1.0)
    if normalizer is not None:
        pred = normalizer.denormalize_retention(pred)
    pred = _apply_ad_drop(pred, is_ad)
    if apply_smoothing and smooth_window > 1:
        pred = smooth_predictions(pred, window=smooth_window)
    return y_true, pred


def load_aligned_embeddings_for_videos(video_dfs: dict[str, pd.DataFrame], embeddings_root: str | Path = "embeddings") -> dict[str, np.ndarray]:
    embeddings_root = Path(embeddings_root)
    result: dict[str, np.ndarray] = {}
    for vid, df in video_dfs.items():
        try:
            aligned, _ = load_aligned_embeddings(vid, embeddings_root, duration_sec=len(df))
            if aligned.shape[0] > 0 and aligned.shape[1] > 0:
                result[vid] = aligned
        except Exception:
            pass
    logger.info("Loaded aligned embeddings for %d / %d videos", len(result), len(video_dfs))
    return result


class MultimodalWindowedDataset(Dataset):
    def __init__(
        self,
        video_dfs: dict[str, pd.DataFrame],
        video_embeddings: dict[str, np.ndarray],
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
    ):
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.feature_mask_prob = feature_mask_prob
        self.noise_std = noise_std
        self.emb_dim = emb_dim
        self.time_feature_mode = time_feature_mode if time_feature_mode in TIME_FEATURE_MODES else "none"
        self.ref_time_sec_max = float(max(ref_time_sec_max, 1e-6))
        self.windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, int, np.ndarray | None, int]] = []
        for vid in video_ids:
            df = video_dfs[vid]
            w = float(video_weights[vid]) if video_weights and vid in video_weights else 1.0
            ts_full = _time_sec_per_row(df)
            X, y, is_ad_col, spike_triggers = _tabular_array_from_df(df, feature_cols, normalizer)
            emb = _align_embedding_rows(video_embeddings.get(vid), len(X), emb_dim)
            n = len(X)
            ts_opt = ts_full if self.time_feature_mode != "none" else None
            if n <= window_size:
                self.windows.append((emb, X, y, is_ad_col, spike_triggers, n, w, 0, ts_opt, n))
            else:
                for s in _window_start_indices(n, window_size, stride):
                    self.windows.append(
                        (
                            emb[s : s + window_size],
                            X[s : s + window_size],
                            y[s : s + window_size],
                            is_ad_col[s : s + window_size],
                            spike_triggers[s : s + window_size],
                            window_size,
                            w,
                            s,
                            ts_opt,
                            n,
                        )
                    )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        emb, X, y, is_ad, spike_triggers, real_len, weight, start, ts_full, n_full = self.windows[idx]
        ws = self.window_size
        if len(X) < ws:
            pad = ws - len(X)
            emb, X = np.pad(emb, ((0, pad), (0, 0))), np.pad(X, ((0, pad), (0, 0)))
            y, is_ad = np.pad(y, (0, pad)), np.pad(is_ad, (0, pad))
            spike_triggers = np.pad(spike_triggers, (0, pad))
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
            "padding_mask": torch.from_numpy(mask),
            "video_weight": torch.tensor(weight, dtype=torch.float32),
        }


@torch.no_grad()
def predict_video_multimodal(
    model: torch.nn.Module,
    df: pd.DataFrame,
    emb: np.ndarray | None,
    feature_cols: list[str],
    normalizer: FeatureNormalizer | None,
    device: torch.device,
    window_size: int = 128,
    stride: int = 1,
    emb_dim: int = 1536,
    smooth_window: int = 15,
    apply_smoothing: bool = False,
    time_feature_mode: str = "none",
    ref_time_sec_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    X = _prediction_tabular_X(df, feature_cols, normalizer)
    y_true = pd.to_numeric(df["retention"], errors="coerce").fillna(0).values
    ts_full = _time_sec_per_row(df)
    n_full = len(X)
    tf_mode = time_feature_mode if time_feature_mode in TIME_FEATURE_MODES else "none"
    ref_sec = float(max(ref_time_sec_max, 1e-6))
    emb = _align_embedding_rows(emb, len(X), emb_dim)
    n = len(X)
    if n <= window_size:
        Xw = X
        real_len = n
        if Xw.shape[0] < window_size:
            pad = window_size - Xw.shape[0]
            Xw = np.pad(Xw, ((0, pad), (0, 0)))
            emb_w = np.pad(emb, ((0, pad), (0, 0)))
        else:
            emb_w = emb
        Xw = _append_time_features_to_matrix(Xw, 0, n_full, window_size, real_len, ts_full, tf_mode, ref_sec)
        t_emb = torch.tensor(emb_w, dtype=torch.float32).unsqueeze(0).to(device)
        t_tab = torch.tensor(Xw, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(t_emb, tabular=t_tab).squeeze(0).cpu().numpy()[:n]
        if normalizer is not None:
            pred = normalizer.denormalize_retention(pred)
        if apply_smoothing and smooth_window > 1:
            pred = smooth_predictions(pred, window=smooth_window)
        return y_true, pred

    pred_sum, pred_cnt = np.zeros(n), np.zeros(n)
    for s in range(0, n - window_size + 1, stride):
        Xw = _append_time_features_to_matrix(X[s : s + window_size], s, n_full, window_size, window_size, ts_full, tf_mode, ref_sec)
        t_emb = torch.tensor(emb[s : s + window_size], dtype=torch.float32).unsqueeze(0).to(device)
        t_tab = torch.tensor(Xw, dtype=torch.float32).unsqueeze(0).to(device)
        p = model(t_emb, tabular=t_tab).squeeze(0).cpu().numpy()
        pred_sum[s : s + window_size] += p
        pred_cnt[s : s + window_size] += 1.0
    pred = pred_sum / np.maximum(pred_cnt, 1.0)
    if normalizer is not None:
        pred = normalizer.denormalize_retention(pred)
    if apply_smoothing and smooth_window > 1:
        pred = smooth_predictions(pred, window=smooth_window)
    return y_true, pred


def plot_mae_summary(all_metrics: dict, out_dir: str, model_name: str = "Model"):
    if not all_metrics:
        return
    rows = sorted(all_metrics.items(), key=lambda kv: kv[1]["mae"])
    vids = [v for v, _ in rows]
    maes = [m["mae"] for _, m in rows]
    splits = [m["split"] for _, m in rows]
    overall = float(np.mean(maes))

    val_maes = [m for m, s in zip(maes, splits, strict=True) if s == "val"]
    train_maes = [m for m, s in zip(maes, splits, strict=True) if s == "train"]
    val_mean = float(np.mean(val_maes)) if val_maes else float("nan")
    train_mean = float(np.mean(train_maes)) if train_maes else float("nan")

    C_BLUE, C_ORANGE, C_RED, C_GREEN = "#2196F3", "#FF5722", "#F44336", "#4CAF50"

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(vids) * 0.28)), gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    y_pos = np.arange(len(vids))
    colors = [C_ORANGE if s == "val" else C_BLUE for s in splits]
    ax.barh(y_pos, maes, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(vids, fontsize=7)
    ax.axvline(overall, color=C_RED, linestyle="--", linewidth=1.5, label=f"mean={overall:.2f}")
    for i, v in enumerate(maes):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=6.5)
    ax.set(xlabel="MAE", title=f"{model_name} — MAE per video  (overall={overall:.2f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    ax2 = axes[1]
    split_names = ["train", "val", "all"]
    split_vals = [train_mean, val_mean, overall]
    split_colors = [C_BLUE, C_ORANGE, C_GREEN]
    b = ax2.bar(split_names, split_vals, color=split_colors, edgecolor="white")
    for bar, val in zip(b, split_vals, strict=True):
        if np.isfinite(val):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set(ylabel="MAE", title=f"{model_name} — MAE by split")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "mae_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    metric_keys = list(next(iter(all_metrics.values())).keys())
    csv_cols = ["video_id", "split"] + [k for k in metric_keys if k not in ("split", "n_seconds")]
    csv_rows = [{"video_id": v, **all_metrics[v]} for v in vids]
    pd.DataFrame(csv_rows).reindex(columns=csv_cols, fill_value=None).to_csv(os.path.join(out_dir, "mae_summary.csv"), index=False)
    logger.info("%s MAE summary: overall=%.3f  train=%.3f  val=%.3f", model_name, overall, train_mean, val_mean)
