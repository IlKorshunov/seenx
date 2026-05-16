from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, logger, skip_if_exists
from .constants import SPEECH_INTELLIGIBILITY_COLS, SPEECH_INTELLIGIBILITY_MUMBLE_SCALE, SPEECH_INTELLIGIBILITY_SMOOTH_WINDOW, SPEECH_INTELLIGIBILITY_WPS_FAST


def extract_speech_intelligibility(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(SPEECH_INTELLIGIBILITY_COLS, existing_features, "speech_intelligibility"):
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)

    intelligibility = np.full(duration, 1.0, dtype=np.float64)
    mumble_index = np.zeros(duration, dtype=np.float64)

    words_per_sec = np.zeros(duration, dtype=np.float64)
    conf_sum_per_sec = np.zeros(duration, dtype=np.float64)
    word_count_per_sec = np.zeros(duration, dtype=np.float64)

    for segment in segments:
        words = segment.get("words", [])
        for word in words:
            start = word.get("start", 0.0)
            prob = word.get("probability", 1.0)

            sec_idx = min(duration - 1, max(0, math.floor(start)))

            conf_sum_per_sec[sec_idx] += prob
            word_count_per_sec[sec_idx] += 1
            words_per_sec[sec_idx] += 1

    mask = word_count_per_sec > 0
    intelligibility[mask] = conf_sum_per_sec[mask] / word_count_per_sec[mask]
    wps_norm = np.clip(pd.Series(words_per_sec).rolling(SPEECH_INTELLIGIBILITY_SMOOTH_WINDOW, center=True, min_periods=1).mean().values / SPEECH_INTELLIGIBILITY_WPS_FAST, 0.0, 1.0)
    mumble_index = np.clip((1.0 - intelligibility) * wps_norm * SPEECH_INTELLIGIBILITY_MUMBLE_SCALE, 0.0, 1.0)
    logger.info("Speech intelligibility: mean=%.2f, mumble_mean=%.3f", intelligibility.mean(), mumble_index.mean())

    return pd.DataFrame({"speech_intelligibility": intelligibility, "speech_mumble_index": mumble_index})
