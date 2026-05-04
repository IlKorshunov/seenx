from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.embedding_aligner import _load_npy, _load_seg_meta, normalize_l2, resample_audio_to_1fps, resample_text_to_1fps, resample_visual_to_1fps
from ...utils.logger import Logger
from .common import embeddings_dir, rolling_frame_window
from .constants import (
    EMBEDDINGS_ROOT,
    VIDEO_INTELLIGENCE_INTRO_SEC,
    VIDEO_INTELLIGENCE_MIN_PERIODS_ROLL,
    VIDEO_INTELLIGENCE_RHYTHM_LAGS,
    VIDEO_INTELLIGENCE_RHYTHM_WINDOW_SEC,
    VIDEO_INTELLIGENCE_SURPRISE_WINDOW_SEC,
    VIDEO_INTELLIGENCE_SYNC_WINDOW_SEC,
)


logger = Logger(show=True).get_logger()

_OUTPUT_COLS = ("content_rhythm", "visual_audio_sync", "narrative_momentum", "engagement_surprise")
_PRODUCED = set(_OUTPUT_COLS)


def _embedding_drift_series(normalized_embeddings: np.ndarray) -> np.ndarray:
    adjacent_similarity = np.sum(normalized_embeddings[1:] * normalized_embeddings[:-1], axis=1)
    return np.r_[0.0, 1.0 - np.clip(adjacent_similarity, -1.0, 1.0)].astype(np.float64)


def _acf_rhythm_score(window_values: np.ndarray, max_lag: int) -> float:
    """
    Mean absolute lag-1..max_lag autocorrelation of standardized window values.
    """
    window_values = (arr := np.asarray(window_values, dtype=np.float64).ravel())[np.isfinite(arr)]
    if window_values.size < max_lag + 2:
        return 0.0
    centered_values = window_values - np.mean(window_values)
    window_std = float(np.std(centered_values))
    if window_std < 1e-9:
        return 0.0
    normalized_values = centered_values / window_std
    correlations = [float(np.mean(normalized_values[:-lag] * normalized_values[lag:])) for lag in range(1, max_lag + 1)]
    return float(np.mean(np.abs(np.array(correlations, dtype=np.float64))))


def _pearson_correlation(window_a: np.ndarray, window_b: np.ndarray) -> float:
    """Pearson correlation for rolling window."""
    finite_mask = np.isfinite(window_a) & np.isfinite(window_b)
    window_a = window_a[finite_mask]
    window_b = window_b[finite_mask]
    if window_a.size < 3:
        return 0.0
    std_a = float(np.std(window_a))
    std_b = float(np.std(window_b))
    if std_a < 1e-9 or std_b < 1e-9:
        return 0.0
    correlation = float(np.corrcoef(window_a, window_b)[0, 1])
    return correlation if np.isfinite(correlation) else 0.0


def _rolling_zscore(series: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """rolling z-score, same length as series."""
    ser = pd.Series(series, dtype=np.float64)
    roll = ser.rolling(window=window, center=True, min_periods=min_periods)
    mean = roll.mean()
    std = roll.std(ddof=0)
    out = ((ser - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    return np.where(np.isfinite(out), out, 0.0)


def _rolling_seconds(duration: int, window_sec: int, reducer) -> np.ndarray:
    return rolling_frame_window(duration, 1.0, window_sec, 1.0, reducer)


def _visual_signals(visual_embeddings_raw: np.ndarray, duration: int) -> tuple[np.ndarray, np.ndarray]:
    visual_embeddings_norm = normalize_l2(resample_visual_to_1fps(visual_embeddings_raw.astype(np.float32), duration), axis=-1)
    drift = _embedding_drift_series(visual_embeddings_norm)
    rhythm = _rolling_seconds(
        duration, VIDEO_INTELLIGENCE_RHYTHM_WINDOW_SEC, lambda start, end, _span, _duration_min: _acf_rhythm_score(drift[start:end], VIDEO_INTELLIGENCE_RHYTHM_LAGS)
    )
    return drift, rhythm


def _audio_change_signal(audio_embeddings_raw: np.ndarray, duration: int) -> np.ndarray:
    audio_embeddings_aligned = resample_audio_to_1fps(audio_embeddings_raw.astype(np.float32), duration)
    return np.r_[0.0, np.linalg.norm(audio_embeddings_aligned[1:] - audio_embeddings_aligned[:-1], axis=1)].astype(np.float64)


def _text_signals(text_embeddings_raw: np.ndarray, seg_meta, duration: int) -> tuple[np.ndarray, np.ndarray]:
    text_embeddings_norm = normalize_l2(resample_text_to_1fps(text_embeddings_raw.astype(np.float32), seg_meta, duration), axis=-1)
    intro_embeddings = text_embeddings_norm[: min(VIDEO_INTELLIGENCE_INTRO_SEC, duration)]
    valid_intro = np.linalg.norm(intro_embeddings, axis=1) > 1e-9
    narrative_momentum = np.zeros(duration, dtype=np.float64)
    if valid_intro.any():
        intro_ref = normalize_l2(intro_embeddings[valid_intro].mean(axis=0).reshape(1, -1), axis=-1)[0]
        intro_similarity = np.where(np.linalg.norm(text_embeddings_norm, axis=1) > 1e-9, text_embeddings_norm @ intro_ref, 0.0)
        narrative_momentum = np.r_[0.0, intro_similarity[:-1] - intro_similarity[1:]].astype(np.float64)
    valid_pairs = (np.linalg.norm(text_embeddings_norm[:-1], axis=1) > 1e-9) & (np.linalg.norm(text_embeddings_norm[1:], axis=1) > 1e-9)
    pair_shift = 1.0 - np.clip(np.sum(text_embeddings_norm[1:] * text_embeddings_norm[:-1], axis=1), -1.0, 1.0)
    return narrative_momentum, np.r_[0.0, np.where(valid_pairs, pair_shift, 0.0)].astype(np.float64)


def _surprise_score(series: np.ndarray) -> np.ndarray:
    return _rolling_zscore(series, VIDEO_INTELLIGENCE_SURPRISE_WINDOW_SEC, VIDEO_INTELLIGENCE_MIN_PERIODS_ROLL)


def _weighted_surprise(scores: list[tuple[np.ndarray, float]]) -> np.ndarray:
    return sum((score * weight for score, weight in scores), np.zeros_like(scores[0][0])) / sum(weight for _, weight in scores)


def extract_video_intelligence(video_path: str, config: Config, existing_features: list | None = None) -> pd.DataFrame:
    _ = config
    if existing_features and _PRODUCED.issubset(set(existing_features)):
        logger.info("Video intelligence already exists, skipping")
        return pd.DataFrame()

    duration = int(math.ceil(get_video_duration(video_path)))
    root = embeddings_dir(video_path, EMBEDDINGS_ROOT)

    out = {col: np.zeros(duration, dtype=np.float64) for col in _OUTPUT_COLS}

    visual_embeddings_raw = _load_npy(root / "visual_embeddings.npy")
    audio_embeddings_raw = _load_npy(root / "audio_embeddings.npy")
    text_embeddings_raw = _load_npy(root / "seg_embeddings.npy")
    seg_meta = _load_seg_meta(root / "seg_meta.json")
    has_visual, has_audio, has_text = (visual_embeddings_raw is not None, audio_embeddings_raw is not None, text_embeddings_raw is not None and bool(seg_meta))

    drift, out["content_rhythm"] = (
        _visual_signals(cast(np.ndarray, visual_embeddings_raw), duration) if has_visual else (np.zeros(duration, dtype=np.float64), np.zeros(duration, dtype=np.float64))
    )
    audio_change = _audio_change_signal(cast(np.ndarray, audio_embeddings_raw), duration) if has_audio else np.zeros(duration, dtype=np.float64)
    out["visual_audio_sync"] = (
        _rolling_seconds(duration, VIDEO_INTELLIGENCE_SYNC_WINDOW_SEC, lambda start, end, _span, _duration_min: _pearson_correlation(drift[start:end], audio_change[start:end]))
        if has_visual
        else np.zeros(duration, dtype=np.float64)
    )
    out["narrative_momentum"], text_shift = (
        _text_signals(cast(np.ndarray, text_embeddings_raw), seg_meta, duration) if has_text else (np.zeros(duration, dtype=np.float64), np.zeros(duration, dtype=np.float64))
    )
    out["engagement_surprise"] = _weighted_surprise([(_surprise_score(drift), 0.25), (_surprise_score(audio_change), 0.4), (_surprise_score(text_shift), 0.35)]).astype(np.float64)
    return pd.DataFrame(out)
