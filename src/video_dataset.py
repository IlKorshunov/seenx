from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset

from .utils.logger import Logger

logger = Logger(show=True).get_logger()


class VideoBatchDataset(IterableDataset):
    """
    Sequential video reader yielding batches of frames.
    frames: List[np.ndarray]
    indices: List[int]
    """

    def __init__(
        self,
        video_path: str,
        batch_size: int,
        transform=None,
        start_frame: int = 0,
        end_frame: int | None = None,
        stride: int = 1,
        frame_condition: Callable[[int], bool] | None = None,
        frame_transform: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ):
        self.video_path = video_path
        self.batch_size = batch_size
        self.transform = transform
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.stride = stride
        self.frame_condition = frame_condition
        self.frame_transform = frame_transform
        self.total_processed_frames = None
        self.total_frames = None
        logger.info("Dataset %s: batches_count=%d batch_size=%d - %s frames", video_path, len(self), batch_size, self.total_processed_frames)

    def __len__(self):
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = self.end_frame or total
        self.total_frames = end - self.start_frame
        count = sum(
            (self.frame_condition is None or self.frame_condition(frame_idx)) and (frame_idx - self.start_frame) % self.stride == 0 for frame_idx in range(self.start_frame, end)
        )
        cap.release()
        self.total_processed_frames = count
        return (count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        Yields:
            frames: np.ndarray  (B, H, W, C)  RGB
            indices: List[int]  absolute frame indices
        """
        cap = cv2.VideoCapture(self.video_path)
        end = self.end_frame or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        frames: list[np.ndarray] = []
        indices: list[int] = []
        frame_idx = self.start_frame
        while frame_idx < end:
            ret, frame = cap.read()
            if not ret:
                break
            accept_frame = (self.frame_condition is None or self.frame_condition(frame_idx)) and (frame_idx - self.start_frame) % self.stride == 0
            if accept_frame:
                frame = self.transform(frame) if self.transform is not None else frame
                frame = self.frame_transform(frame, frame_idx) if self.frame_transform is not None else frame
                if frame.shape[0] == 0:
                    frame_idx += 1
                    continue
                frames.append(frame)
                indices.extend([frame_idx] * len(frame))
                if len(frames) == self.batch_size:
                    yield np.concatenate(frames, axis=0), indices
                    frames, indices = [], []
            frame_idx += 1

        if frames:
            yield np.concatenate(frames, axis=0), indices

        cap.release()


class SpeakerFilteredVideoDataset(VideoBatchDataset):
    def __init__(self, speaker_probs, threshold=0.9, **kwargs):
        self.speaker_probs = speaker_probs
        self.threshold = threshold

        super().__init__(frame_condition=self.accept_frame, **kwargs)

    def accept_frame(self, frame_idx: int) -> bool:
        return self.speaker_probs[frame_idx] >= self.threshold


class SpecificFramesVideoDataset(VideoBatchDataset):
    """
    Dataset that yields only specific frames based on a given list of frame_ids.
    """

    def __init__(self, frame_ids: list[int] | np.ndarray | set, **kwargs):
        self.frame_ids = set(frame_ids)
        super().__init__(frame_condition=self.accept_frame, **kwargs)

    def accept_frame(self, frame_idx: int) -> bool:
        return frame_idx in self.frame_ids


class FaceCropVideoDataset(VideoBatchDataset):
    """
    Dataset that yields only specific frames based on a given list of frame_ids.
    """

    def __init__(self, frame_ids: list[int] | np.ndarray | set, crop_boxes: list[list[int]], **kwargs):
        self.frame_ids = set(frame_ids)
        self.crop_boxes = crop_boxes
        super().__init__(frame_condition=self.accept_frame, frame_transform=self._crop_frame, **kwargs)

    def _crop_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        boxes = self.crop_boxes[frame_idx]
        return (
            np.empty((0, 112, 112, 3), dtype=frame.dtype)
            if boxes is None or len(boxes) == 0
            else np.array([cv2.resize(frame[0][y1:y2, x1:x2], (112, 112), interpolation=cv2.INTER_LINEAR) for x1, y1, x2, y2 in boxes.tolist()])
        )

    def accept_frame(self, frame_idx: int) -> bool:
        return frame_idx in self.frame_ids


class EmotionIterableDataset(IterableDataset):
    def __init__(self, video_dataset):
        self.video_dataset = video_dataset
        self.indices = []

    def __iter__(self):
        self.indices.clear()
        for frames, indices in self.video_dataset:
            for frame, idx in zip(frames, indices, strict=True):
                self.indices.append(idx)
                yield Image.fromarray(frame)
