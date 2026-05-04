"""Aesthetic scoring per frame via CLIP + LAION aesthetic head. Output: aesthetic_score (0-10)."""

import gc
import math
import os
import urllib.request

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor

from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.logger import Logger
from .common import fps_to_frame_step, get_capture_fps, open_video_capture
from .constants import AESTHETIC_HEAD_FILENAME, AESTHETIC_HEAD_URL, CLIP_MODEL_ID


logger = Logger(show=True).get_logger()
_COLS = {"aesthetic_score"}

_cache = os.path.join(torch.hub.get_dir(), AESTHETIC_HEAD_FILENAME)
if not os.path.exists(_cache):
    os.makedirs(os.path.dirname(_cache), exist_ok=True)
    urllib.request.urlretrieve(AESTHETIC_HEAD_URL, _cache)


class _AestheticMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def extract_aesthetic_score(video_path: str, config: Config, existing_features=None) -> pd.DataFrame:
    if existing_features and _COLS.issubset(set(existing_features)):
        return pd.DataFrame()

    duration = math.ceil(get_video_duration(video_path))
    device = torch.device(config.get("device"))

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    head = _AestheticMLP()
    head.load_state_dict(torch.load(_cache, map_location=device, weights_only=True))
    head = head.to(device).eval()

    scores = np.zeros(duration, dtype=np.float64)
    sec_idx, frame_idx = 0, 0

    with open_video_capture(video_path) as cap:
        step = fps_to_frame_step(get_capture_fps(cap))
        while sec_idx < duration:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs = processor(images=[img], return_tensors="pt").to(device)
                with torch.no_grad():
                    emb = torch.nn.functional.normalize(model.get_image_features(**inputs), p=2, dim=1)
                    scores[sec_idx] = np.clip(head(emb).item(), 0.0, 10.0)
                sec_idx += 1
            frame_idx += 1

    del model, processor, head
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame({"aesthetic_score": scores})
