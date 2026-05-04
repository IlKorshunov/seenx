"""
Visual entropy — frame visual complexity via Shannon entropy. values from 0 to 8 bits.
"""

import cv2
import numpy as np
from tqdm import tqdm

from ...utils.config import Config
from ..feature_extractor import VideoFeature
from .common import open_video_capture


class VisualEntropyFeature(VideoFeature):
    def __init__(self, config: Config):
        self.config = config

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"visual_entropy"}

    def run(self, video_path, context):
        entropies: list[float] = []

        with open_video_capture(video_path) as cap:
            for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
                ret, frame = cap.read()
                if not ret:
                    break
                hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]).flatten()
                hist /= hist.sum() + 1e-10
                mask = hist > 0
                entropies.append(-float(np.sum(hist[mask] * np.log2(hist[mask]))))

        df = context["data"]
        n = min(len(df), len(entropies))
        df["visual_entropy"] = np.nan
        df.loc[df.index[:n], "visual_entropy"] = entropies[:n]
