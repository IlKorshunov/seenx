import cv2
import numpy as np

from ..seenx_utils import resize_crop_center_np


class VideoFeature:
    def required_keys(self) -> set[str]:
        return set()

    def produces_keys(self) -> set[str]:
        return set()

    def default_transform(self, frame: np.ndarray, size: int = 640) -> np.ndarray:
        """BGR2RGB, resize with crop, add batch dim. Shared by most visual extractors."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_crop_center_np(frame, size)
        return frame[np.newaxis, :, :, :]

    def run(self, video_path: str, context: dict): ...
