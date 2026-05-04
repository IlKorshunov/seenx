import cv2
import numpy as np

from ...utils.config import Config
from ..feature_extractor import VideoFeature
from .common import open_video_capture, shot_bound_pairs
from .constants import SCENE_NOVELTY_HIST_BINS


class SceneNoveltyFeature(VideoFeature):
    def __init__(self, config: Config):
        self.config = config

    def required_keys(self):
        return {"shot_bounds"}

    def produces_keys(self):
        return {"scene_novelty"}

    @staticmethod
    def _hist(frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, list(SCENE_NOVELTY_HIST_BINS), [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def run(self, video_path, context):
        df = context["data"]
        n_frames = len(df)
        scenes = shot_bound_pairs(context["shot_bounds"])

        mid_frames = {}
        with open_video_capture(video_path) as cap:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for start, end in scenes:
                mid = (start + end) // 2
                mid = min(mid, total - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                _, frame = cap.read()
                mid_frames[(start, end)] = self._hist(frame)

        novelty = np.zeros(n_frames, dtype=np.float64)
        prev_hist = None
        for start, end in scenes:
            h = mid_frames.get((start, end))
            if h is not None and prev_hist is not None:
                novelty[start : end + 1] = np.log1p(
                    cv2.compareHist(prev_hist.reshape(SCENE_NOVELTY_HIST_BINS).astype(np.float32), h.reshape(SCENE_NOVELTY_HIST_BINS).astype(np.float32), cv2.HISTCMP_CHISQR)
                )
            prev_hist = h

        pos = novelty[novelty > 0]
        if len(pos) >= 2:
            median = np.median(pos)
            novelty[novelty > 0] = novelty[novelty > 0] / (novelty[novelty > 0] + median)
        elif len(pos) == 1:
            novelty[novelty > 0] = 0.5

        df["scene_novelty"] = novelty
