"""Viewer expectation model: does current content match what viewer expected?
expectation_match   : 0-1, cosine similarity to predicted embedding
expectation_surprise: 0-1, inverse of match (1 = maximally unexpected)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, logger, skip_if_exists


_COLS = {"expectation_match", "expectation_surprise"}

_LOOKBACK_SEC = 30
_INTRO_SEC = 15
_DECAY = 0.9


def _cosine_sim(left_vector: np.ndarray, right_vector: np.ndarray) -> float:
    left_norm, right_norm = np.linalg.norm(left_vector), np.linalg.norm(right_vector)
    if left_norm < 1e-12 or right_norm < 1e-12:
        return 0.0
    return float(np.dot(left_vector, right_vector) / (left_norm * right_norm))


def _load_embeddings(video_path: str) -> np.ndarray | None:
    video_id = os.path.basename(os.path.dirname(video_path)) if video_path.endswith(".mp4") else os.path.splitext(os.path.basename(video_path))[0]
    emb_path = Path(video_path).resolve().parents[1] / "embeddings" / video_id / "seg_embeddings.npy"
    if not emb_path.exists():
        project_root = Path(__file__).resolve().parents[3]
        emb_path = project_root / "embeddings" / video_id / "seg_embeddings.npy"
    if emb_path.exists():
        return np.load(emb_path).astype(np.float64)
    return None


def _predict_next_embedding(embeddings: np.ndarray, current_sec: int, lookback: int, decay: float) -> np.ndarray:
    start = max(0, current_sec - lookback)
    if start >= current_sec:
        return embeddings[0]

    window = embeddings[start:current_sec]
    window_size = len(window)
    weights = np.array([decay ** (window_size - 1 - position_idx) for position_idx in range(window_size)])
    weights /= weights.sum() + 1e-12

    predicted = (window * weights[:, None]).sum(axis=0)

    if window_size >= 3:
        velocity = embeddings[current_sec - 1] - embeddings[max(0, current_sec - 3)]
        predicted += velocity * 0.3

    return predicted


def extract_expectation(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "expectation"):
        return pd.DataFrame()

    _, dur = get_segments_and_duration(video_path, config)
    embeddings = _load_embeddings(video_path)

    match = np.full(dur, 0.5, dtype=np.float64)
    surprise = np.full(dur, 0.5, dtype=np.float64)

    if embeddings is None:
        logger.warning("Expectation: no seg_embeddings.npy found, returning defaults")
        return pd.DataFrame({"expectation_match": match, "expectation_surprise": surprise})

    if len(embeddings) < dur:
        pad = np.zeros((dur - len(embeddings), embeddings.shape[1]), dtype=np.float64)
        embeddings = np.vstack([embeddings, pad])
    elif len(embeddings) > dur:
        embeddings = embeddings[:dur]

    for sec in range(_INTRO_SEC, dur):
        predicted = _predict_next_embedding(embeddings, sec, _LOOKBACK_SEC, _DECAY)
        sim = _cosine_sim(embeddings[sec], predicted)
        match[sec] = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
        surprise[sec] = 1.0 - match[sec]

    for sec in range(min(_INTRO_SEC, dur)):
        match[sec] = 1.0
        surprise[sec] = 0.0

    logger.info("Expectation: mean_match=%.3f mean_surprise=%.3f", match.mean(), surprise.mean())

    return pd.DataFrame({"expectation_match": match, "expectation_surprise": surprise})
