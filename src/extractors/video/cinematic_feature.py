import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from ...utils.config import Config
from ...video_dataset import VideoBatchDataset
from ..feature_extractor import VideoFeature


class CinematicFeature(VideoFeature):
    def __init__(self, config: Config):
        self.device = torch.device(config.get("device"))
        self.batch_size = config.get("batch_size")
        self.use_fp16 = self.device.type == "cuda"
        self.processor = CLIPProcessor.from_pretrained(config.get("clip_model"), use_fast=False)
        self.model = CLIPModel.from_pretrained(config.get("clip_model")).to(self.device).eval()
        self._pos_texts = [
            "a cinematic film shot with dramatic lighting",
            "a beautifully composed movie scene",
            "a professional cinema frame with shallow depth of field",
            "a visually stunning wide angle shot",
            "a dramatic cinematic color graded scene",
        ]
        self._neg_texts = [
            "a webcam recording of a person talking",
            "a low quality amateur home video",
            "a screen recording of a computer desktop",
            "a simple static shot of a room",
            "a blurry unfocused snapshot",
        ]

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"cinematic"}

    def run(self, video_path, context):
        df = context["data"]
        if "cinematic" not in df.columns:
            df["cinematic"] = pd.Series([0.0] * len(df), index=df.index, dtype="float64")
        dataset = VideoBatchDataset(video_path=video_path, batch_size=self.batch_size, transform=self.default_transform)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_fp16):
            pos_feat = self.model.get_text_features(**self.processor(text=self._pos_texts, return_tensors="pt", padding=True).to(self.device))
            pos_feat = (pos_feat / pos_feat.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
            pos_feat /= pos_feat.norm(dim=-1, keepdim=True)
            neg_feat = self.model.get_text_features(**self.processor(text=self._neg_texts, return_tensors="pt", padding=True).to(self.device))
            neg_feat = (neg_feat / neg_feat.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
            neg_feat /= neg_feat.norm(dim=-1, keepdim=True)
            txt_feat = torch.cat([pos_feat, neg_feat], dim=0)

            for frames, indices in tqdm(dataset, desc="Extract cinematic probs"):
                image_features = self.model.get_image_features(**self.processor(images=frames, return_tensors="pt").to(self.device))
                probs = ((image_features / image_features.norm(dim=-1, keepdim=True)) @ txt_feat.T).softmax(dim=-1)
                for i, frame_idx in enumerate(indices):
                    df.at[frame_idx, "cinematic"] = float(probs[i, 0])
