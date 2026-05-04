import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ...seenx_utils import pad_boxes_square
from ...utils.config import Config
from ...video_dataset import VideoBatchDataset
from ..feature_extractor import VideoFeature


class FaceScreenRatioFeature(VideoFeature):
    def __init__(self, config: Config):
        self.device = torch.device(config.get("device"))
        self.face_detector = YOLO(config.get("face_detector")).to(self.device)
        self.batch_size = config.get("face_screen_batch_size") or config.get("batch_size")

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"face_screen_ratio", "faces_total_ratio", "face_area_ratio"}

    def use_face_detector(self, frames: np.ndarray) -> list[np.ndarray]:
        input_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device).float().div_(255.0)
        with torch.no_grad():
            results = self.face_detector(input_tensor, verbose=False)
        boxes = [res.boxes.xyxy.cpu().numpy().astype(float) for res in results]
        del input_tensor, results
        return boxes

    def run(self, video_path, context):
        dataset = VideoBatchDataset(video_path=video_path, batch_size=self.batch_size, transform=self.default_transform)
        df = context["data"]
        for col in self.produces_keys():
            if col not in df.columns:
                df[col] = pd.Series([0.0] * len(df), index=df.index, dtype="float64")

        if "frame_face_boxes" not in df.columns:
            df["frame_face_boxes"] = pd.Series([None] * len(df), index=df.index, dtype=object)

        for frames, indices in tqdm(dataset, desc="Extract face screen ratio"):
            h, w, _ = frames[0].shape
            frame_area = h * w
            raw_boxes = self.use_face_detector(frames)
            padded_boxes = pad_boxes_square(raw_boxes, w, h)
            for idx, raw, padded in zip(indices, raw_boxes, padded_boxes, strict=True):
                df.at[idx, "frame_face_boxes"] = padded
                if len(raw) == 0:
                    continue
                areas = [max(0, b[2] - b[0]) * max(0, b[3] - b[1]) for b in raw]
                df.at[idx, "face_area_ratio"] = df.at[idx, "face_screen_ratio"] = min(max(areas) / frame_area, 1.0)
                df.at[idx, "faces_total_ratio"] = min(sum(areas) / frame_area, 1.0)
