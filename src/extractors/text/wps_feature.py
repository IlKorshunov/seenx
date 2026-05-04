import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, seg_bounds, skip_if_exists
from .constants import WPS_COLS


def extract_wps(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(WPS_COLS, existing_features, "wps"):
        return pd.DataFrame()
    segments, duration = get_segments_and_duration(video_path, config)
    out = np.zeros(duration, dtype=np.float64)
    for segment in segments:
        segment_duration = segment["end"] - segment["start"]
        if segment_duration <= 0:
            continue
        start_sec, end_sec = seg_bounds(segment, duration)
        out[start_sec:end_sec] = len(segment["text"].split()) / segment_duration
    return pd.DataFrame({"wps": out})
