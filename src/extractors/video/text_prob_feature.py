import easyocr
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ...utils.config import Config
from ...video_dataset import VideoBatchDataset
from ..feature_extractor import VideoFeature
from .constants import OCR_CONFIDENCE_THRESHOLD


class TextProbFeature(VideoFeature):
    def __init__(self, config: Config):
        self.ocr_reader = easyocr.Reader(["en", "ru"], gpu=torch.cuda.is_available())
        self.batch_size = config.get("text_prob_batch_size") or config.get("batch_size")
        self.config = config

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"text_prob"}

    def run(self, video_path, context):
        df = context["data"]
        if "text_prob" not in df.columns:
            df["text_prob"] = pd.Series(np.nan, index=df.index, dtype="float64")
        dataset = VideoBatchDataset(video_path=video_path, batch_size=self.batch_size, transform=self.default_transform, stride=self.config.get("text_prob_stride"))

        for frames, indices in tqdm(dataset):
            results = self.ocr_reader.readtext_batched(frames, batch_size=self.batch_size)
            for i, res in enumerate(results):
                confident = [c for _, _, c in res if c >= OCR_CONFIDENCE_THRESHOLD]
                text_prob = float(np.mean(confident)) if confident else 0.0
                context["data"].at[indices[i], "text_prob"] = text_prob

        context["data"]["text_prob"] = context["data"]["text_prob"].ffill().bfill()
