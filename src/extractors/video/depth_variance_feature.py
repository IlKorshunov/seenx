"""Monocular depth variance per frame via Depth Anything V2 Large. Output: depth_variance, depth_mean."""

import gc
import math

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.logger import Logger
from .common import fps_to_frame_step, get_capture_fps, open_video_capture
from .constants import DEPTH_MODEL_ID


logger = Logger(show=True).get_logger()
_COLS = {"depth_variance", "depth_mean"}


def extract_depth_variance(video_path: str, config: Config, existing_features=None) -> pd.DataFrame:
    if existing_features and _COLS.issubset(set(existing_features)):
        return pd.DataFrame()

    duration = math.ceil(get_video_duration(video_path))
    device = torch.device(config.get("device"))
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device).eval()

    variances, means = np.zeros(duration, dtype=np.float64), np.zeros(duration, dtype=np.float64)
    sec_idx, frame_idx = 0, 0

    with open_video_capture(video_path) as cap:
        step = fps_to_frame_step(get_capture_fps(cap))
        while sec_idx < duration:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                inputs = processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_tensors="pt").to(device)
                with torch.no_grad():
                    depth = model(**inputs).predicted_depth.squeeze().cpu().numpy()
                variances[sec_idx], means[sec_idx] = float(np.var(depth)), float(np.mean(depth))
                sec_idx += 1
            frame_idx += 1

    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame({"depth_variance": variances, "depth_mean": means})
