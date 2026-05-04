"""Screencast / demo detection via CLIP zero-shot classification."""

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from ...utils.config import Config
from ...video_dataset import VideoBatchDataset
from ..feature_extractor import VideoFeature


class ScreencastFeature(VideoFeature):
    def __init__(self, config: Config):
        self.device = torch.device(config.get("device"))
        self.batch_size = config.get("batch_size")
        self.config = config
        self.use_fp16 = self.device.type == "cuda"
        self.processor = CLIPProcessor.from_pretrained(config.get("clip_model"), use_fast=False)
        self.model = CLIPModel.from_pretrained(config.get("clip_model")).to(self.device).eval()
        self.pos_texts = ["a screen recording of a computer desktop", "a screenshot of a website or app interface", "a screencast with code editor on screen"]
        self.neg_texts = ["a person talking to the camera", "a talking head vlog", "a close-up of a face speaking"]

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"screencast_prob"}

    def run(self, video_path, context):
        df = context["data"]
        df["screencast_prob"] = 0.0

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

            for frames, indices in tqdm(dataset, desc="Extract screencast probs"):
                inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
                img_feat = self.model.get_image_features(**inputs)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                logit_scale = self.model.logit_scale.exp().clamp(1, 100)
                logits = logit_scale * torch.cat([img_feat @ pos_feat.T, img_feat @ neg_feat.T], dim=1)
                probs = logits.softmax(dim=-1)
                for i, idx in enumerate(indices):
                    df.at[idx, "screencast_prob"] = float(probs[i, 0])
