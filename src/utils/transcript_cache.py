"""
Whisper transcript cache.
"""

import os

import torch

from .config import Config
from .logger import Logger
import whisper

logger = Logger(show=True).get_logger()
_cache: dict[str, dict] = {}


def get_transcript(video_path: str, config: Config) -> dict:
    logger.info("Looking Whisper transcript cache hit for %s", os.path.basename(video_path))
    key = os.path.abspath(video_path)
    if key in _cache:
        logger.info("Founded Transcript cache hit for %s", os.path.basename(video_path))
        return _cache[key]

    result = whisper.load_model(config.get("whisper_model_size", "large-v3"), device=torch.device(config.get("device"))).transcribe(video_path, word_timestamps=True, language="ru")
    _cache[key] = result
    return result


def clear_cache():
    _cache.clear()
