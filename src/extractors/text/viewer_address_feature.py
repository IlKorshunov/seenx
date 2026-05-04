import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, seg_bounds, skip_if_exists
from .constants import VIEWER_ADDRESS_COLS, VIEWER_ADDRESS_PATTERN


def extract_viewer_address(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(VIEWER_ADDRESS_COLS, existing_features, "viewer_address"):
        return pd.DataFrame()
    segments, duration = get_segments_and_duration(video_path, config)
    out = np.zeros(duration, dtype=np.float64)
    for segment in segments:
        match_count = len(VIEWER_ADDRESS_PATTERN.findall(segment["text"]))
        if match_count == 0:
            continue
        start_sec, end_sec = seg_bounds(segment, duration)
        out[start_sec:end_sec] = float(match_count)
    return pd.DataFrame({"viewer_address": out})
