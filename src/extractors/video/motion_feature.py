import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ...utils.config import Config
from ...video_dataset import SpeakerFilteredVideoDataset
from ..feature_extractor import VideoFeature
from .common import config_device


class MotionSpeedFeature(VideoFeature):
    def __init__(self, config: Config):
        self.device = config_device(config)
        self.pose_model = YOLO(config.get("pose_model")).to(self.device)
        self.batch_size = config.get("batch_size")
        self.kps_thr = config.get("keypoint_confidence_threshold")
        self.speak_thr = config.get("speaker_probability_threshold")

    def required_keys(self):
        return {"speaker_prob"}

    def produces_keys(self):
        return {"motion_speed"}

    def use_pose_model(self, frames: np.ndarray) -> list[np.ndarray]:
        input_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device).float().div_(255.0)
        with torch.no_grad():
            results = self.pose_model(input_tensor, verbose=False)
        keypoints = [res.keypoints.data.cpu().numpy().astype(float) for res in results]
        del input_tensor, results
        return keypoints

    def distance(self, previous_keypoints: np.ndarray, current_keypoints: np.ndarray) -> float:
        if any(keypoints is None or len(keypoints) == 0 for keypoints in (previous_keypoints, current_keypoints)):
            return 0.0
        previous_person, current_person = previous_keypoints[0], current_keypoints[0]
        previous_visible = previous_person[previous_person[:, 2] > self.kps_thr][:, :2]
        current_visible = current_person[current_person[:, 2] > self.kps_thr][:, :2]
        if len(previous_visible) == 0 or len(current_visible) == 0:
            return 0.0
        n_keypoints = min(len(previous_visible), len(current_visible))
        return float(np.linalg.norm(current_visible[:n_keypoints] - previous_visible[:n_keypoints], axis=1).mean())

    def run(self, video_path, context):
        df = context["data"]
        if "frame_keypoints" not in df.columns:
            df["frame_keypoints"] = [None] * len(df)
        if "motion_speed" not in df.columns:
            df["motion_speed"] = 0.0
        dataset = SpeakerFilteredVideoDataset(
            speaker_probs=df["speaker_prob"].tolist(), threshold=self.speak_thr, video_path=video_path, batch_size=self.batch_size, transform=self.default_transform
        )

        for frames, indices in tqdm(dataset, desc="Extract motion speeds"):
            batch_keypoints = self.use_pose_model(frames)
            for frame_idx, keypoints in zip(indices, batch_keypoints, strict=True):
                df.at[frame_idx, "frame_keypoints"] = keypoints
            for frame_idx in indices[indices > 0]:
                df.at[frame_idx, "motion_speed"] = self.distance(df.at[frame_idx - 1, "frame_keypoints"], df.at[frame_idx, "frame_keypoints"])
