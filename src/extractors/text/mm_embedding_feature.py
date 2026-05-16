"""VideoLLaMA2 multimodal embeddings."""

import numpy as np
import pandas as pd

from ...seenx_utils import get_video_duration
from ...utils.logger import Logger
from ...mm.mm_pipeline import multimodal_features


logger = Logger(show=True).get_logger()
MM_EMBED_DIMS = 128

def extract_mm_embeddings(video_path: str, config) -> pd.DataFrame | None:
    if not config.get("use_mm_embeddings", False):
        return None
    features = multimodal_features(video_path, app_config=config)[0]
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    features = np.atleast_2d(features)[:, :MM_EMBED_DIMS]
    duration = int(get_video_duration(video_path)) + 1
    source_times = np.linspace(0, duration - 1, len(features))
    target_times = np.arange(duration, dtype=float)
    return pd.DataFrame({f"mm_embed_{embed_idx}": np.interp(target_times, source_times, features[:, embed_idx]) for embed_idx in range(features.shape[1])})
