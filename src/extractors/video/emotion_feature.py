import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ...utils.config import Config
from ...video_dataset import FaceCropVideoDataset
from ..feature_extractor import VideoFeature
from .architectures import ResEmoteNet
from .common import config_device


EMOTION_LABELS = ["happiness", "surprise", "sadness", "anger", "disgust", "fear", "neutral"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EmotionFeature(VideoFeature):
    def __init__(self, config: Config):
        self.device = config_device(config)
        self.batch_size = config.get("batch_size")
        self.speaker_thr = config.get("speaker_probability_threshold")

        self.model = ResEmoteNet().to(self.device)
        self.model.load_state_dict(torch.load(config.get("emotion_model_weights"), map_location=self.device, weights_only=True)["model_state_dict"])
        self.model.eval()

    def required_keys(self):
        return {"frame_face_boxes", "speaker_prob"}

    def produces_keys(self):
        return {"emotion"}

    def _preprocess(self, crops: np.ndarray) -> torch.Tensor:
        def prepare(crop: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
            img3ch = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
            return ((img3ch - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)

        return torch.from_numpy(np.stack([prepare(crop) for crop in crops])).to(self.device)

    def run(self, video_path, context):
        df, total_frames = context["data"], len(context["data"])
        emotions = np.zeros((total_frames, len(EMOTION_LABELS)), dtype=np.float64)
        dataset = FaceCropVideoDataset(
            frame_ids=df.index[df["speaker_prob"] >= self.speaker_thr].tolist(),
            crop_boxes=df["frame_face_boxes"].tolist(),
            video_path=video_path,
            batch_size=self.batch_size,
            transform=self.default_transform,
        )
        with torch.no_grad():
            for frames, indices in tqdm(dataset, desc="Extract emotions"):
                np.maximum.at(emotions, np.asarray(indices, dtype=np.int64), F.softmax(self.model(self._preprocess(frames)), dim=1).cpu().numpy())

        for j, label in enumerate(EMOTION_LABELS):
            df[label] = emotions[:, j]
