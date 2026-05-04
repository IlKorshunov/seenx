import cv2
import numpy as np

from ...utils.config import Config
from ..feature_extractor import VideoFeature
from .common import iter_video_frames


class ColorFeature(VideoFeature):
    def __init__(self, config: Config):
        self.config = config

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"color_temperature", "color_saturation"}

    def run(self, video_path, context):
        temps: list[float] = []
        sats: list[float] = []

        for frame in iter_video_frames(video_path):
            b_mean = float(np.mean(frame[:, :, 0]))
            r_mean = float(np.mean(frame[:, :, 2]))
            temp = float(np.log((r_mean + 1.0) / (b_mean + 1.0)))
            temps.append(temp)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sats.append(float(np.mean(hsv[:, :, 1])))

        df = context["data"]
        n = min(len(df), len(temps), len(sats))
        df["color_temperature"] = np.nan
        df["color_saturation"] = np.nan
        df.loc[df.index[:n], "color_temperature"] = temps[:n]
        df.loc[df.index[:n], "color_saturation"] = sats[:n]
