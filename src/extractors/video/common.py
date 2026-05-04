import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

from .constants import DEFAULT_VIDEO_FPS, EMBEDDINGS_ROOT


def video_id(video_path: str) -> str:
    return os.path.basename(os.path.dirname(video_path)) if video_path.endswith(".mp4") else os.path.splitext(os.path.basename(video_path))[0]


def embeddings_dir(video_path: str, embeddings_root: str | Path = EMBEDDINGS_ROOT) -> Path:
    return Path(embeddings_root) / video_id(video_path)


def config_device(config) -> torch.device:
    return torch.device(config.get("device"))


def get_video_fps(video_path: str, fallback: float = DEFAULT_VIDEO_FPS) -> float:
    fps = read_video_fps(video_path)
    return fps if fps is not None else fallback


def get_capture_fps(cap: cv2.VideoCapture, fallback: float = DEFAULT_VIDEO_FPS) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps) if fps and fps > 0 else fallback


def fps_to_frame_step(fps: float) -> int:
    return max(1, int(round(fps)))


def rolling_frame_window(n_frames: int, fps: float, window_sec: float, min_window_sec: float, reducer: Callable[[int, int, int, float], float]) -> np.ndarray:
    window_frames = max(1, int(round(window_sec * fps)))
    min_window_frames = max(1, int(round(min_window_sec * fps)))
    half_window = window_frames // 2
    values = np.zeros(n_frames, dtype=np.float64)

    for frame_idx in range(n_frames):
        start = max(0, frame_idx - half_window)
        end = min(n_frames, frame_idx + half_window + 1)
        span_frames = max(end - start, min_window_frames)
        duration_min = span_frames / fps / 60.0
        values[frame_idx] = reducer(start, end, span_frames, duration_min)
    return values


def shot_bound_pairs(shot_bounds: list[int]) -> list[tuple[int, int]]:
    return list(zip(shot_bounds[0::2], shot_bounds[1::2], strict=True))


def mask_runs(mask: np.ndarray, value: bool = True) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return np.empty((0, 2), dtype=np.int32)
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes - 1, len(mask) - 1]
    keep = mask[starts] == value
    return np.column_stack([starts[keep], ends[keep]]).astype(np.int32)


@contextmanager
def open_video_capture(video_path: str) -> Iterator[cv2.VideoCapture]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        yield cap
    finally:
        cap.release()


def iter_video_frames(video_path: str) -> Iterator[np.ndarray]:
    with open_video_capture(video_path) as cap:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame


def read_video_fps(video_path: str) -> float | None:
    with open_video_capture(video_path) as cap:
        fps = get_capture_fps(cap, fallback=0.0)
    return fps if fps > 0 else None


def iter_1fps_rgb_frames(video_path: str, fallback_fps: float = DEFAULT_VIDEO_FPS) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with open_video_capture(video_path) as cap:
        step = fps_to_frame_step(get_capture_fps(cap, fallback=fallback_fps))
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
    return frames


def require_1fps_rgb_frames(video_path: str, fallback_fps: float = DEFAULT_VIDEO_FPS) -> list[np.ndarray]:
    frames = iter_1fps_rgb_frames(video_path, fallback_fps=fallback_fps)
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")
    return frames
