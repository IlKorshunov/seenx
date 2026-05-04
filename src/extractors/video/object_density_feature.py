"""Object density per frame via YOLOv8. Output: object_count, unique_classes."""

import math

import numpy as np
import pandas as pd
from ultralytics import YOLO

from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.logger import Logger
from .common import fps_to_frame_step, get_capture_fps, open_video_capture
from .constants import YOLO_OBJECT_MODEL_ID


logger = Logger(show=True).get_logger()
_COLS = {"object_count", "unique_classes"}


def extract_object_density(video_path: str, config: Config, existing_features=None) -> pd.DataFrame:
    if existing_features and _COLS.issubset(set(existing_features)):
        return pd.DataFrame()

    duration = math.ceil(get_video_duration(video_path))
    yolo = YOLO(YOLO_OBJECT_MODEL_ID)

    obj_counts, cls_counts = np.zeros(duration, dtype=np.float64), np.zeros(duration, dtype=np.float64)
    sec_idx, frame_idx = 0, 0

    with open_video_capture(video_path) as cap:
        step = fps_to_frame_step(get_capture_fps(cap))
        while sec_idx < duration:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                boxes = yolo(frame, verbose=False)[0].boxes
                obj_counts[sec_idx] = len(boxes)
                cls_counts[sec_idx] = len(set(int(c) for c in boxes.cls.cpu().numpy())) if len(boxes) else 0
                sec_idx += 1
            frame_idx += 1

    del yolo
    return pd.DataFrame({"object_count": obj_counts, "unique_classes": cls_counts})
