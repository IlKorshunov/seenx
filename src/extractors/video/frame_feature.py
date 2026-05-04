import cv2
import numpy as np
from tqdm import tqdm

from ...utils.config import Config
from ..feature_extractor import VideoFeature
from .common import open_video_capture
from .constants import VISUAL_COMPLEXITY_ENTROPY_WEIGHT, VISUAL_COMPLEXITY_SHARPNESS_WEIGHT


class FrameQualityFeature(VideoFeature):
    def __init__(self, config: Config):
        self.config = config

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"brightness", "sharpness", "visual_entropy", "visual_complexity", "visual_complexity_gradient", "visual_complexity_acceleration"}

    def run(self, video_path, context):
        frame_features = {"brightness": [], "sharpness": [], "visual_entropy": []}
        with open_video_capture(video_path) as cap:
            for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Extract frame quality"):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
                hist /= hist.sum() + 1e-10
                mask = hist > 0
                frame_features["brightness"].append(float(np.mean(gray)))
                frame_features["sharpness"].append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                frame_features["visual_entropy"].append(-float(np.sum(hist[mask] * np.log2(hist[mask]))))

        df = context["data"]
        for feature_name, values in frame_features.items():
            n = min(len(df), len(values))
            df[feature_name] = np.nan
            df.loc[df.index[:n], feature_name] = values[:n]

        n = min(len(df), len(frame_features["visual_entropy"]), len(frame_features["sharpness"]))
        df["visual_complexity"] = np.nan
        df["visual_complexity_gradient"] = np.nan
        df["visual_complexity_acceleration"] = np.nan

        entropy = np.asarray(frame_features["visual_entropy"][:n], dtype=float)
        sharpness = np.asarray(frame_features["sharpness"][:n], dtype=float)
        ent_norm = (entropy - entropy.mean()) / max(entropy.std(), 1e-9)
        sharp_norm = (sharpness - sharpness.mean()) / max(sharpness.std(), 1e-9)
        complexity = VISUAL_COMPLEXITY_ENTROPY_WEIGHT * ent_norm + VISUAL_COMPLEXITY_SHARPNESS_WEIGHT * sharp_norm
        gradient = np.gradient(complexity) if n > 1 else np.zeros_like(complexity)
        acceleration = np.gradient(gradient) if n > 1 else np.zeros_like(complexity)

        df.loc[df.index[:n], "visual_complexity"] = complexity
        df.loc[df.index[:n], "visual_complexity_gradient"] = gradient
        df.loc[df.index[:n], "visual_complexity_acceleration"] = acceleration
