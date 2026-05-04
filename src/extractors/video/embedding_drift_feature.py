from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.embedding_aligner import normalize_l2, resample_visual_to_1fps
from ...utils.logger import Logger
from .common import embeddings_dir
from .constants import EMBEDDING_DRIFT_SMOOTH_WINDOW_SEC, EMBEDDINGS_ROOT

logger = Logger(show=True).get_logger()

_COLS = {"embedding_drift", "embedding_drift_smoothed"}


def _load_visual_embeddings(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    return np.load(path, allow_pickle=False).astype(np.float32) if path.is_file() else None


def extract_embedding_drift(video_path: str, _config: Config, existing_features: list | None = None) -> pd.DataFrame:
    if existing_features and _COLS.issubset(set(existing_features)):
        logger.info("Embedding drift already exists, skipping")
        return pd.DataFrame()

    duration = max(1, math.ceil(get_video_duration(video_path)))
    root = embeddings_dir(video_path, EMBEDDINGS_ROOT)
    raw = _load_visual_embeddings(root / "visual_embeddings.npy")
    if raw is None:
        logger.info("No visual embeddings for %s, embedding_drift = 0", root.name)
        return pd.DataFrame({"embedding_drift": np.zeros(duration, dtype=np.float64), "embedding_drift_smoothed": np.zeros(duration, dtype=np.float64)})

    aligned_embeddings = normalize_l2(resample_visual_to_1fps(raw, duration), axis=-1)
    sim = np.sum(aligned_embeddings[1:] * aligned_embeddings[:-1], axis=1)
    drift = np.r_[0.0, 1.0 - np.clip(sim, -1.0, 1.0)].astype(np.float64)
    window = EMBEDDING_DRIFT_SMOOTH_WINDOW_SEC
    smoothed = np.array([drift[max(0, t - window + 1) : t + 1].mean() for t in range(duration)], dtype=np.float64)

    return pd.DataFrame({"embedding_drift": drift, "embedding_drift_smoothed": smoothed})
