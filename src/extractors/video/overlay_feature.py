"""Overlay detection via CLIP zero-shot classification."""

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from ...utils.config import Config
from ...video_dataset import VideoBatchDataset
from ..feature_extractor import VideoFeature
from .constants import OVERLAY_PROB_THRESHOLD


class OverlayFeature(VideoFeature):
    """
    Classifies each frame as one of:
      talking head with overlay
      clean talking head with no overlays
    """

    def __init__(self, config: Config):
        self.device = torch.device(config.get("device"))
        self.batch_size = config.get("batch_size")
        self.config = config
        self.use_fp16 = self.device.type == "cuda"
        self.processor = CLIPProcessor.from_pretrained(config.get("clip_model"), use_fast=False)
        self.model = CLIPModel.from_pretrained(config.get("clip_model")).to(self.device).eval()
        self.pos_texts = [
            "a person talking with an image overlay on screen",
            "a video with a picture-in-picture inset",
            "a person with a photo or graphic overlaid on the video",
            "a talking head with text and images on screen",
            "a video frame with overlaid infographic or illustration",
        ]
        self.neg_texts = ["a person talking to the camera with no overlays", "a clean shot of a person speaking", "a simple talking head video without graphics"]

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"overlay_prob"}

    def run(self, video_path, context):
        df = context["data"]
        df["overlay_prob"] = 0.0

        dataset = VideoBatchDataset(video_path=video_path, batch_size=self.batch_size, transform=self.default_transform)

        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_fp16):
            pos_inp = self.processor(text=self.pos_texts, return_tensors="pt", padding=True).to(self.device)
            neg_inp = self.processor(text=self.neg_texts, return_tensors="pt", padding=True).to(self.device)

            pos_feat = self.model.get_text_features(**pos_inp)
            pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
            pos_feat = pos_feat.mean(dim=0, keepdim=True)
            pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)

            neg_feat = self.model.get_text_features(**neg_inp)
            neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
            neg_feat = neg_feat.mean(dim=0, keepdim=True)
            neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)

            for frames, indices in tqdm(dataset, desc="Extract overlay probs"):
                inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
                img_feat = self.model.get_image_features(**inputs)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                logit_scale = self.model.logit_scale.exp().clamp(1, 100)
                logits = logit_scale * torch.cat([img_feat @ pos_feat.T, img_feat @ neg_feat.T], dim=1)
                probs = logits.softmax(dim=-1)
                for i, idx in enumerate(indices):
                    p = float(probs[i, 0])
                    df.at[idx, "overlay_prob"] = p if p >= OVERLAY_PROB_THRESHOLD else 0.0
