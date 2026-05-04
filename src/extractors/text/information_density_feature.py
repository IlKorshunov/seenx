"""
Information density: measures how much new semantic content appears per second.

Uses pre-computed text segment embeddings from embeddings/<vid>/seg_embeddings.npy.
For each second t, computes KL-divergence proxy between the embedding at t
and a running average of previous embeddings. High density = new information.
"""

import json
import math
import os

import numpy as np
import pandas as pd

from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.logger import Logger


logger = Logger(show=True).get_logger()
EMBEDDINGS_ROOT = "embeddings"
_COLS = {"information_density", "cumulative_info"}


def _load_seg_data(video_id: str) -> tuple[np.ndarray | None, list[dict]]:
    emb_path = os.path.join(EMBEDDINGS_ROOT, video_id, "seg_embeddings.npy")
    meta_path = os.path.join(EMBEDDINGS_ROOT, video_id, "seg_meta.json")
    if not os.path.exists(emb_path):
        return None, []
    embeddings = np.load(emb_path, allow_pickle=False).astype(np.float32)
    metadata = []
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    return embeddings, metadata


def _resample_to_1fps(embeddings: np.ndarray, seg_meta: list[dict], duration: int) -> np.ndarray:
    """Resample segment embeddings to per-second using overlap weighting."""
    n_segments = len(embeddings)
    out = np.zeros((duration, embeddings.shape[1]), dtype=np.float32)
    for second_idx in range(duration):
        second_start, second_end = float(second_idx), float(second_idx + 1)
        weights, segment_vectors = [], []
        for segment_idx, segment in enumerate(seg_meta[:n_segments]):
            start_sec, end_sec = float(segment.get("start", 0)), float(segment.get("end", 0))
            overlap = max(0, min(second_end, end_sec) - max(second_start, start_sec))
            if overlap > 1e-6:
                weights.append(overlap)
                segment_vectors.append(embeddings[segment_idx])
        if weights:
            weights = np.array(weights, dtype=np.float32)
            weights /= weights.sum()
            out[second_idx] = np.average(segment_vectors, axis=0, weights=weights)
    return out


def extract_information_density(video_path: str, config: Config, existing_features=None) -> pd.DataFrame:
    if existing_features and _COLS.issubset(set(existing_features)):
        return pd.DataFrame()

    duration = math.ceil(get_video_duration(video_path))
    video_id = os.path.basename(os.path.dirname(video_path))
    embeddings, metadata = _load_seg_data(video_id)

    if embeddings is None or len(metadata) < 2:
        logger.warning("information_density: no embeddings for %s", video_id)
        return pd.DataFrame({column: np.zeros(duration) for column in sorted(_COLS)})

    per_sec = _resample_to_1fps(embeddings, metadata, duration)

    norms = np.linalg.norm(per_sec, axis=1, keepdims=True).clip(min=1e-9)
    normed = per_sec / norms

    density = np.zeros(duration, dtype=np.float64)
    cumulative = np.zeros(duration, dtype=np.float64)
    running_sum = np.zeros(per_sec.shape[1], dtype=np.float64)
    for second_idx in range(duration):
        if norms[second_idx, 0] < 1e-6:
            continue
        if second_idx == 0:
            running_sum += normed[second_idx]
            continue
        running_mean = running_sum / second_idx
        rm_norm = np.linalg.norm(running_mean)
        if rm_norm > 1e-9:
            running_mean /= rm_norm
        cos_sim = float(np.dot(normed[second_idx], running_mean))
        density[second_idx] = 1.0 - cos_sim
        cumulative[second_idx] = cumulative[second_idx - 1] + density[second_idx]
        running_sum += normed[second_idx]

    return pd.DataFrame({"information_density": density, "cumulative_info": cumulative})
