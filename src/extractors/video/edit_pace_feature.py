import numpy as np
from ...utils.config import Config
from ..feature_extractor import VideoFeature
from .common import get_video_fps, rolling_frame_window
from .constants import EDIT_PACE_MIN_WINDOW_SEC, EDIT_PACE_WINDOW_SEC


class EditPaceFeature(VideoFeature):
    def __init__(self, config: Config):
        self.config = config

    def required_keys(self):
        return {"shot_bounds"}

    def produces_keys(self):
        return {"edit_pace"}

    def run(self, video_path, context):
        df = context["data"]
        cut_frames = np.array(context["shot_bounds"][1::2], dtype=np.float64)
        edit_pace = rolling_frame_window(
            len(df),
            get_video_fps(video_path),
            EDIT_PACE_WINDOW_SEC,
            EDIT_PACE_MIN_WINDOW_SEC,
            lambda start, end, _span_frames, duration_min: float(np.sum((cut_frames >= start) & (cut_frames < end)) / duration_min),
        )
        df["edit_pace"] = edit_pace
