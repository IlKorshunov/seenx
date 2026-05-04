import cv2
import numpy as np


def get_video_duration(video_path: str, fallback_fps: float = 30.0) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or fallback_fps)
        if fps <= 0:
            raise ValueError(f"Invalid FPS: {fps}")
        return float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / fps
    finally:
        cap.release()


def resize_crop_center_np(frame: np.ndarray, size: int = 640) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected shape (H, W, C), got {frame.shape}")
    h, w, _ = frame.shape
    scale = size / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    h, w, _ = frame.shape
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return frame[top : top + size, left : left + size]


def pad_boxes_square(boxes: list[np.ndarray], w: int, h: int, pad: float = 0.25) -> list[np.ndarray]:
    padded_boxes = []
    for frame_boxes in boxes:
        if len(frame_boxes) == 0:
            padded_boxes.append(frame_boxes)
            continue

        frame_boxes = frame_boxes.astype(np.float32)
        x1, y1, x2, y2 = frame_boxes.T
        bw = x2 - x1
        bh = y2 - y1

        side = np.maximum(bw, bh)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half = (1 + pad) * side / 2

        x1_p = np.clip(cx - half, 0, w)
        y1_p = np.clip(cy - half, 0, h)
        x2_p = np.clip(cx + half, 0, w)
        y2_p = np.clip(cy + half, 0, h)
        padded = np.stack([x1_p, y1_p, x2_p, y2_p], axis=1).astype(np.int32)
        padded_boxes.append(padded)
    return padded_boxes
