from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, logger, seg_bounds, skip_if_exists
from .common import collect_valid_segment_dicts, release_models
from .constants import TOPIC_SHARPNESS_COLS, TOPIC_SHARPNESS_PROMPT, TOPIC_SHARPNESS_WINDOW_OVERLAP, TOPIC_SHARPNESS_WINDOW_SEGMENTS
from .friction_feature import _format_segments, _llm_generate, _load_llm


def _parse_sharpness(response: str) -> dict[int, float]:
    parsed_scores: dict[int, float] = {}
    for line in response.strip().splitlines():
        match = re.match(r"\[(\d+)\]\s*(\d+(?:\.\d+)?)\s*$", line.strip())
        if not match:
            continue
        segment_idx = int(match.group(1))
        score_value = float(np.clip(float(match.group(2)), 0.0, 100.0))
        parsed_scores[segment_idx] = score_value
    return parsed_scores


def extract_topic_sharpness(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(set(TOPIC_SHARPNESS_COLS), existing_features, "topic_sharpness"):
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)
    valid = collect_valid_segment_dicts(segments)
    sharp = np.zeros(duration, dtype=np.float64)

    model, tokenizer = _load_llm()

    for window_start in range(0, len(valid), TOPIC_SHARPNESS_WINDOW_SEGMENTS - TOPIC_SHARPNESS_WINDOW_OVERLAP):
        window_end = min(window_start + TOPIC_SHARPNESS_WINDOW_SEGMENTS, len(valid))
        window = valid[window_start:window_end]
        if not window:
            break

        prompt = TOPIC_SHARPNESS_PROMPT.format(segments=_format_segments(window, start_idx=window_start + 1))
        try:
            response = _llm_generate(model, tokenizer, prompt)
        except (RuntimeError, ValueError) as error:
            logger.warning("Topic sharpness window [%d-%d] failed: %s", window_start + 1, window_end, error)
            continue

        scores = _parse_sharpness(response)
        for local_idx, seg in enumerate(window):
            segment_idx = window_start + local_idx + 1
            if segment_idx not in scores:
                continue
            score_value = scores[segment_idx]
            start_sec, end_sec = seg_bounds(seg, duration)
            sharp[start_sec:end_sec] = np.maximum(sharp[start_sec:end_sec], score_value)

        if window_end >= len(valid):
            break

    release_models(model, tokenizer, device=None)

    logger.info("Topic sharpness: mean=%.1f max=%.0f (0–100)", float(sharp.mean()), float(sharp.max()))
    return pd.DataFrame({"topic_sharpness_0_100": sharp})
