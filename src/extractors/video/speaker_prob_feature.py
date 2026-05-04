import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ...arcface_client import ArcFaceClient
from ...seenx_utils import pad_boxes_square, resize_crop_center_np
from ...utils.config import Config, get_device
from ...video_dataset import FaceCropVideoDataset, SpecificFramesVideoDataset
from ..feature_extractor import VideoFeature
from .constants import SPEAKER_SAMPLE_EVERY


class SpeakerProbabilityFeature(VideoFeature):
    def __init__(self, config: Config):
        self.device = torch.device(get_device(config))
        self.face_detector = YOLO(config.get("face_detector")).to(self.device)
        self.arcface_client = ArcFaceClient(config.get("face_embedder"))
        self.batch_size = config.get("batch_size")
        self.speaker_thr = config.get("speaker_probability_threshold")
        self.config = config

    def required_keys(self):
        return {"shot_bounds"}

    def produces_keys(self):
        return {"speaker_prob"}

    def use_face_detector(self, frames: np.ndarray) -> list[np.ndarray]:
        input_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device).float().div_(255.0)
        with torch.no_grad():
            results = self.face_detector(input_tensor, verbose=False)
        boxes = [res.boxes.xyxy.cpu().numpy().astype(float) for res in results]
        del input_tensor, results
        return boxes

    def speaker_face_embedding(self) -> np.ndarray:
        img_bgr = cv2.imread(self.config.get("speaker_image_path"))
        img_rgb = resize_crop_center_np(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))[np.newaxis, :, :, :]
        _, h, w, _ = img_rgb.shape
        padded_boxes = pad_boxes_square(self.use_face_detector(img_rgb), w, h)
        x1, y1, x2, y2 = padded_boxes[0].tolist()[0]
        face_crop = cv2.resize(img_rgb[0][y1:y2, x1:x2], (112, 112), interpolation=cv2.INTER_LINEAR)
        return np.array(self.arcface_client.forward(face_crop))

    def vector_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1, emb2 = emb1.reshape(-1), emb2.reshape(-1)
        n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        return float(np.dot(emb1 / n1, emb2 / n2))

    _SAMPLE_EVERY = SPEAKER_SAMPLE_EVERY

    def _build_sample_ids(self, shot_bounds: list[int]) -> list[int]:
        """Shot boundaries + intermediate frames every _SAMPLE_EVERY."""
        sample_set = set(shot_bounds)
        for i in range(0, len(shot_bounds), 2):
            start, end = shot_bounds[i], shot_bounds[i + 1]
            for f in range(start + self._SAMPLE_EVERY, end, self._SAMPLE_EVERY):
                sample_set.add(f)
        return sorted(sample_set)

    def run(self, video_path, context):
        sample_ids = self._build_sample_ids(context["shot_bounds"])

        dataset = SpecificFramesVideoDataset(frame_ids=sample_ids, video_path=video_path, batch_size=self.batch_size, transform=self.default_transform)

        df = context["data"]
        if "frame_face_boxes" not in df.columns:
            df["frame_face_boxes"] = pd.Series([None] * len(df), index=df.index, dtype=object)
        if "speaker_prob" not in df.columns:
            df["speaker_prob"] = pd.Series([0.0] * len(df), index=df.index, dtype="float64")
        for frames, indices in tqdm(dataset, desc="Extract speaker probs"):
            for idx, boxes in zip(indices, pad_boxes_square(self.use_face_detector(frames), frames[0].shape[1], frames[0].shape[0]), strict=True):
                df.at[idx, "frame_face_boxes"] = boxes

        dataset = FaceCropVideoDataset(
            frame_ids=sample_ids, crop_boxes=df["frame_face_boxes"].tolist(), video_path=video_path, batch_size=self.batch_size, transform=self.default_transform
        )

        actual_speaker_embedding = self.speaker_face_embedding()
        for frames, indices in tqdm(dataset, desc="Extract speaker embeddings"):
            embeddings = np.array(self.arcface_client.forward(frames))
            for i, frame_idx in enumerate(indices):
                vec_sim = self.vector_similarity(actual_speaker_embedding, embeddings[i])
                if df.at[frame_idx, "speaker_prob"] < vec_sim:
                    df.at[frame_idx, "speaker_prob"] = vec_sim

        sample_set = set(sample_ids)
        for i in range(0, len(context["shot_bounds"]), 2):
            start = context["shot_bounds"][i]
            end = context["shot_bounds"][i + 1]
            shot_samples = sorted(f for f in sample_set if start <= f <= end)
            xp = np.array(shot_samples, dtype=np.float64)
            fp = np.array([df.at[f, "speaker_prob"] for f in shot_samples])
            fill_start = max(0, start - context["shift"])
            fill_end = min(len(df) - 1, end + context["shift"])
            x_all = np.arange(fill_start, fill_end + 1, dtype=np.float64)
            filled = np.interp(x_all, xp, fp)
            filled = np.where(filled > self.speaker_thr, filled, 0.0)
            df.loc[fill_start:fill_end, "speaker_prob"] = filled
