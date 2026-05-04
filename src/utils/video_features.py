from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


LLM_ID_COLS = {"video_folder", "transcript_path", "drive_file_id"}
KEY_TABULAR_COLS = [
    "brightness",
    "sharpness",
    "cinematic",
    "visual_entropy",
    "motion_speed",
    "edit_pace",
    "scene_novelty",
    "speaker_prob",
    "text_prob",
    "screencast_prob",
    "pause_rate",
    "speech_rate_cv",
    "pitch_mean",
    "pitch_std",
    "voiced_frac",
    "hook_score",
    "rms",
    "speech_ratio",
    "silence_stretch",
    "wps",
]
EMBEDDING_TYPES = ["visual", "audio", "text", "bert", "videomae", "seg"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_json(data_dir: Path, vid: str, name: str) -> dict:
    for sub in ("", "transcripts/"):
        p = data_dir / vid / f"{sub}{name}"
        if p.exists():
            return load_json(p)
    return {}


def first_positive(meta: dict, *keys) -> float:
    for key in keys:
        val = pd.to_numeric(meta.get(key), errors="coerce")
        if pd.notna(val) and val > 0:
            return float(val)
    return 0.0


def first_nonneg(meta: dict, *keys, stat_key: str | None = None) -> float:
    for key in keys:
        val = pd.to_numeric(meta.get(key), errors="coerce")
        if pd.notna(val) and val >= 0:
            return float(val)
    if stat_key and isinstance(meta.get("statistics"), dict):
        val = pd.to_numeric(meta["statistics"].get(stat_key), errors="coerce")
        if pd.notna(val) and val >= 0:
            return float(val)
    return 0.0


def meta_nums(meta: dict) -> tuple[float, float, float]:
    dur = first_positive(meta, "duration_sec", "duration_seconds", "video_duration_sec", "duration")
    views = first_nonneg(meta, "view_count", "viewCount", "views", stat_key="viewCount")
    likes = first_nonneg(meta, "like_count", "likeCount", "likes", stat_key="likeCount")
    return dur, views, likes


def llm_numeric(flat: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in flat.items():
        if key in LLM_ID_COLS or str(key).startswith("target__"):
            continue
        if isinstance(val, bool):
            out[f"llm_{key}"] = float(val)
        elif isinstance(val, (int, float)) and np.isfinite(float(val)):
            out[f"llm_{key}" if not str(key).startswith("llm_") else str(key)] = float(val)
    return out


def tab_means(output_dir: Path, vid: str) -> dict[str, float]:
    for path, kw in [(output_dir / f"{vid}_features.csv", {"index_col": 0}), (output_dir / vid / "features_readable.csv", {})]:
        if not path.exists():
            continue
        df = pd.read_csv(path, **kw)
        out: dict[str, float] = {}
        for col in KEY_TABULAR_COLS:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series):
                    out[f"tab_{col}"] = float(series.mean())
        return out
    return {}


def dicts_to_matrix(dicts: list[dict[str, float]], cols: list[str]) -> np.ndarray:
    mat = np.full((len(dicts), len(cols)), np.nan)
    for row_idx, row_dict in enumerate(dicts):
        for col_idx, col_name in enumerate(cols):
            if col_name in row_dict:
                mat[row_idx, col_idx] = row_dict[col_name]
    return mat


def pca_reduce(raw: np.ndarray, n_comp: int, rng: int) -> np.ndarray:
    n_samples, n_features = raw.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reduced = PCA(n_components=min(n_comp, n_samples - 1, n_features), random_state=rng).fit_transform(raw)
    padded = np.zeros((n_samples, n_comp), dtype=np.float64)
    padded[:, : reduced.shape[1]] = reduced
    return padded


def load_embeddings(vids: list[str], emb_dir: Path, modality: str) -> tuple[list[np.ndarray | None], int]:
    vecs: list[np.ndarray | None] = []
    max_dim = 0
    for vid in vids:
        path = emb_dir / vid / f"{modality}_embeddings.npy"
        if path.exists():
            arr = np.load(path).astype(np.float64)
            vec = arr if arr.ndim == 1 else np.nanmean(arr, axis=0)
            max_dim = max(max_dim, vec.shape[0])
            vecs.append(vec)
        else:
            vecs.append(None)
    return vecs, max_dim


def embeddings_to_matrix(vecs: list[np.ndarray | None], n_videos: int, max_dim: int) -> np.ndarray:
    mat = np.zeros((n_videos, max_dim), dtype=np.float64)
    for idx, vec in enumerate(vecs):
        if vec is not None:
            mat[idx, : vec.shape[0]] = vec
    return mat
