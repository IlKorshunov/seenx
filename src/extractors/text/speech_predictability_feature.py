import math
from collections import Counter

import numpy as np
import pandas as pd

from ...utils.logger import Logger
from .constants import SPEECH_PREDICTABILITY_COL, SPEECH_PREDICTABILITY_WINDOW_SEC
from ._base import get_segments_and_duration


logger = Logger(show=True).get_logger()


def _word_entropy(words: list[str]) -> float:
    if not words:
        return 0.0
    counts = Counter(word.lower() for word in words)
    word_count = len(words)
    entropy = 0.0
    for count in counts.values():
        probability = count / word_count
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy


def extract_speech_predictability(video_path: str, config=None, existing_features: set[str] | None = None) -> pd.DataFrame:
    feature_col = SPEECH_PREDICTABILITY_COL
    if existing_features and feature_col in (existing_features if isinstance(existing_features, set) else set(existing_features)):
        logger.info("%s already exists, skipping", feature_col)
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)
    n_seconds = max(1, int(np.ceil(duration)))

    words_per_second: list[list[str]] = [[] for _ in range(n_seconds)]
    for segment in segments:
        start = segment.get("start", 0.0)
        end = segment.get("end", start)
        text = segment.get("text", "")
        segment_words = [word for word in text.split() if len(word) > 0]
        start_sec = max(0, int(start))
        end_sec = min(n_seconds, int(np.ceil(end)))
        if start_sec >= end_sec:
            end_sec = min(start_sec + 1, n_seconds)
        slot_count = max(1, end_sec - start_sec)
        chunk = max(1, len(segment_words) // slot_count)
        word_idx = 0
        for second_idx in range(start_sec, end_sec):
            words_per_second[second_idx].extend(segment_words[word_idx : word_idx + chunk])
            word_idx += chunk

    result = np.zeros(n_seconds, dtype=np.float32)
    for second_idx in range(n_seconds):
        window_start = max(0, second_idx - SPEECH_PREDICTABILITY_WINDOW_SEC // 2)
        window_end = min(n_seconds, second_idx + SPEECH_PREDICTABILITY_WINDOW_SEC // 2 + 1)
        pooled = [word for context_second_idx in range(window_start, window_end) for word in words_per_second[context_second_idx]]
        result[second_idx] = float(_word_entropy(pooled))

    return pd.DataFrame({feature_col: result})
