import gc
import math

import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor

from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.logger import Logger
from .common import embeddings_dir, iter_1fps_rgb_frames
from .constants import CLIP_MODEL_ID

logger = Logger(show=True).get_logger()
_COLS = {"visual_novelty", "visual_topic_shift", "visual_hook_similarity", "visual_global_dist", "visual_momentum", "visual_self_similarity"}


def _derive(embs: np.ndarray) -> dict[str, np.ndarray]:
    n = len(embs)
    sims = embs @ embs.T
    gm = embs.mean(axis=0)
    gm /= max(np.linalg.norm(gm), 1e-9)
    hook = embs[: min(n, 30)].mean(axis=0)
    hook /= max(np.linalg.norm(hook), 1e-9)
    novelty = np.ones(n)
    for i in range(1, n):
        novelty[i] = 1.0 - float(np.mean(sims[i, :i]))
    shift = np.zeros(n)
    shift[1:] = 1.0 - np.diag(sims, k=-1)
    nb = lambda i: embs[max(0, i - 1) : i + 2].mean(axis=0)
    return {
        "visual_novelty": novelty,
        "visual_topic_shift": shift,
        "visual_hook_similarity": embs @ hook,
        "visual_global_dist": 1.0 - (embs @ gm),
        "visual_momentum": np.array([np.mean(shift[max(0, i - 2) : i + 1]) for i in range(n)]),
        "visual_self_similarity": np.array([float(np.dot(embs[i], nb(i) / max(np.linalg.norm(nb(i)), 1e-9))) for i in range(n)]),
    }


def extract_scene_clip(video_path: str, config: Config, existing_features=None) -> pd.DataFrame:
    if existing_features and _COLS.issubset(set(existing_features)):
        return pd.DataFrame()
    duration = math.ceil(get_video_duration(video_path))
    frames = iter_1fps_rgb_frames(video_path)
    device = torch.device(config.get("device"))
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()

    embs_list = []
    for i in range(0, len(frames), 32):
        inputs = processor(images=frames[i : i + 32], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = torch.nn.functional.normalize(model.get_image_features(**inputs), p=2, dim=1)
        embs_list.append(feat.cpu().float().numpy())
    embs = np.vstack(embs_list)

    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out_dir = embeddings_dir(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "visual_embeddings.npy", embs)
    np.save(out_dir / "visual_similarity_matrix.npy", embs @ embs.T)

    feats, cols = _derive(embs), sorted(_COLS)
    n = min(len(embs), duration)
    out = np.zeros((duration, len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        out[:n, j] = feats[c][:n]

    return pd.DataFrame(out, columns=cols)
