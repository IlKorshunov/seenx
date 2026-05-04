"""Detects short insert and meme-cut shots from shot boundaries.
Adds a binary short_insert flag and rolling short_insert_rate density."""

import numpy as np
import pandas as pd

from ...utils.config import Config
from ..feature_extractor import VideoFeature
from .common import get_video_fps, shot_bound_pairs
from .constants import SHORT_INSERT_RATE_WINDOW_SEC, SHORT_INSERT_THRESHOLD_SEC


class ShortInsertFeature(VideoFeature):
    SHORT_THRESHOLD_SEC = SHORT_INSERT_THRESHOLD_SEC
    RATE_WINDOW_SEC = SHORT_INSERT_RATE_WINDOW_SEC

    def __init__(self, config: Config):
        self.config = config

    def required_keys(self):
        return {"shot_bounds"}

    def produces_keys(self):
        return {"short_insert", "short_insert_rate"}

    def run(self, video_path, context):
        df = context["data"]
        n_frames = len(df)

        fps = get_video_fps(video_path)
        threshold_frames = self.SHORT_THRESHOLD_SEC * fps
        window_frames = int(self.RATE_WINDOW_SEC * fps)

        is_short = np.zeros(n_frames, dtype=np.float64)
        for shot_start, shot_end in shot_bound_pairs(context["shot_bounds"]):
            if 0 < (shot_end - shot_start) < threshold_frames:
                is_short[shot_start : shot_end + 1] = 1.0

        kernel = np.ones(window_frames, dtype=np.float64) / window_frames
        short_insert_rate = np.convolve(is_short, kernel, mode="same") if n_frames > window_frames else np.full(n_frames, np.mean(is_short), dtype=np.float64)
        df["short_insert"] = pd.Series(is_short, index=df.index, dtype="float64")
        df["short_insert_rate"] = pd.Series(np.clip(short_insert_rate, 0, 1), index=df.index, dtype="float64")
