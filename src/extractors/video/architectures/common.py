import gc

import numpy as np
import torch

from ....utils.config import Config


def resolve_device(config: Config) -> torch.device:
    raw = str(config.get("device") or "auto").strip().lower()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if raw == "auto" else torch.device("cuda" if raw == "gpu" else raw)


def unload_ensemble(models: tuple, device: str | torch.device | None = "cuda") -> None:
    del models
    gc.collect()
    if device is None:
        should_clear_cuda = torch.cuda.is_available()
    else:
        should_clear_cuda = str(device).split(":", 1)[0] == "cuda" and torch.cuda.is_available()
    if should_clear_cuda:
        torch.cuda.empty_cache()


def robust_minmax(x: np.ndarray, p_low: float = 5.0, p_high: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi - lo < 1e-9:
        return np.clip(x, 0.0, 1.0)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def spread_to_frames(boundary_times_s: np.ndarray, boundary_scores: np.ndarray, n_frames: int, src_fps: float, spread_seconds: float = 0.25) -> np.ndarray:
    sigma_frames = max(1.0, spread_seconds * src_fps)
    centers = np.round(boundary_times_s * src_fps).astype(np.int64)
    return spread_to_frames_idx(centers, boundary_scores, n_frames, sigma_frames)


def spread_to_frames_idx(centers: np.ndarray, scores: np.ndarray, n_frames: int, sigma_frames: float = 3.0) -> np.ndarray:
    out = np.zeros(n_frames, dtype=np.float32)
    half = int(round(3 * sigma_frames))
    for center, score in zip(centers, scores, strict=False):
        start = max(0, int(center) - half)
        end = min(n_frames, int(center) + half + 1)
        if start >= end:
            continue
        idx = np.arange(start, end)
        bump = float(score) * np.exp(-0.5 * ((idx - int(center)) / sigma_frames) ** 2)
        out[start:end] = np.maximum(out[start:end], bump.astype(np.float32))
    return out
