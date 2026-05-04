"""Crutch-word count per segment."""

import re

import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, seg_bounds, skip_if_exists


_COLS = {"crutch_cnt"}
RU_FILLERS = re.compile(r"\b(ээ|ааа|ммм|ну|как бы|типа|вот|короче|значит|собственно|допустим|скажем)\b", re.IGNORECASE)


def extract_speech_fillers(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "crutch_cnt"):
        return pd.DataFrame()
    segments, duration = get_segments_and_duration(video_path, config)
    out = np.zeros(duration, dtype=np.float64)
    for segment in segments:
        match_count = len(RU_FILLERS.findall(segment["text"]))
        if match_count == 0:
            continue
        start_sec, end_sec = seg_bounds(segment, duration)
        out[start_sec:end_sec] = float(match_count)
    return pd.DataFrame({"crutch_cnt": out})
