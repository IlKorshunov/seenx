import math

import numpy as np
import pandas as pd

from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.logger import Logger
from ...utils.transcript_cache import get_transcript


logger = Logger(show=True).get_logger()


def get_segments_and_duration(video_path: str, config: Config) -> tuple[list[dict], int]:
    result = get_transcript(video_path, config)
    return result["segments"], math.ceil(get_video_duration(video_path))


def seg_bounds(segment: dict, duration: int) -> tuple[int, int]:
    return max(0, math.floor(segment["start"])), min(duration, math.ceil(segment["end"]))


def skip_if_exists(cols: set[str], existing_features: list | None, name: str) -> bool:
    if existing_features and cols.issubset(set(existing_features)):
        logger.info("%s already exist, skipping", name)
        return True
    return False


def empty_df(cols: set[str], duration: int) -> pd.DataFrame:
    return pd.DataFrame({column: np.zeros(max(duration, 1)) for column in cols})


def valid_text_segments(segments: list[dict]) -> list[dict]:
    return [segment for segment in segments if segment.get("text", "").strip()]


def segment_text(segment: dict) -> str:
    return segment.get("text", "").strip()
