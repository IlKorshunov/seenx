import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from ....utils.config import Config
from ....utils.logger import Logger
from ..common import embeddings_dir
from .common import resolve_device, robust_minmax, spread_to_frames_idx


logger = Logger(show=True).get_logger()


def videomae_boundary_signal(video_path: str, config: Config) -> np.ndarray:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")

    model_name = config.get("videomae_model") or "MCG-NJU/videomae-base"
    sample_fps = float(config.get("videomae_sample_fps") or 8.0)
    frames_per_clip = int(config.get("videomae_frames_per_clip") or 16)
    stride_sampled = int(config.get("videomae_stride") or 8)
    batch = int(config.get("videomae_batch_size") or 4)
    device = resolve_device(config)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    cap = cv2.VideoCapture(video_path)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps <= 0:
        cap.release()
        raise ValueError(f"Could not read FPS from {video_path}")

    cached = embeddings_dir(video_path) / "videomae_embeddings.npy"
    if cached.is_file():
        embs = torch.from_numpy(np.load(cached).astype(np.float32))
        if len(embs) < 2:
            return np.zeros(n_src, dtype=np.float32)
        diss = robust_minmax((1.0 - (F.normalize(embs[:-1], dim=-1) * F.normalize(embs[1:], dim=-1)).sum(-1)).numpy().astype(np.float32))
        return spread_to_frames_idx(np.minimum(np.arange(1, len(diss) + 1) * round(src_fps), n_src - 1), diss, n_src, max(1.0, 0.15 * src_fps))

    step = max(1, int(round(src_fps / sample_fps)))
    sample_idx = list(range(0, n_src, step))

    frames, next_target = [], 0
    for cur in tqdm(range(n_src), desc="VideoMAE-load", unit="frame"):
        ok, frame = cap.read()
        if not ok or next_target >= len(sample_idx):
            break
        if cur == sample_idx[next_target]:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            next_target += 1
    cap.release()

    if len(frames) < frames_per_clip + stride_sampled:
        logger.warning("VideoMAE: video too short for clip-level features — returning zeros")
        return np.zeros(n_src, dtype=np.float32)

    logger.info("VideoMAE: loading %s on %s", model_name, device)
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()

    starts = list(range(0, len(frames) - frames_per_clip + 1, stride_sampled))
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(starts), batch), desc="VideoMAE-encode", unit="batch"):
            clips = [frames[s : s + frames_per_clip] for s in starts[i : i + batch]]
            inp = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in proc(clips, return_tensors="pt").to(device).items()}
            embs.append(F.normalize(model(**inp).last_hidden_state.float().mean(1), dim=-1).cpu())
    embs = torch.cat(embs, dim=0)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if embs.shape[0] < 2:
        return np.zeros(n_src, dtype=np.float32)

    diss = (1.0 - (embs[:-1] * embs[1:]).sum(-1)).numpy().astype(np.float32)
    diss = robust_minmax(diss)

    boundary_src_frames = np.array([sample_idx[(starts[i] + starts[i + 1]) // 2] for i in range(len(diss))], dtype=np.int64)

    return spread_to_frames_idx(centers=boundary_src_frames, scores=diss, n_frames=n_src, sigma_frames=max(1.0, 0.15 * src_fps))
