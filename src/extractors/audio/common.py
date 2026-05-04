import os
from collections.abc import Sequence

import librosa
import numpy as np

from ...audio_utils import extract_audio_to_wav
from ...seenx_utils import get_video_duration
from .consts import EMBEDDINGS_ROOT, STEMS_DIRNAME


def video_id(video_path: str) -> str:
    return os.path.basename(os.path.dirname(video_path))


def embeddings_dir(video_path: str) -> str:
    return os.path.join(EMBEDDINGS_ROOT, video_id(video_path))


def stems_dir(video_path: str) -> str:
    return os.path.join(os.path.dirname(video_path), STEMS_DIRNAME)


def stem_path(video_path: str, stem_name: str) -> str:
    return os.path.join(stems_dir(video_path), stem_name)


def load_video_audio(video_path: str, sample_rate: int, mono: bool = True) -> tuple[np.ndarray, int]:
    wav_path = extract_audio_to_wav(video_path, sr=sample_rate)
    try:
        return librosa.load(wav_path, sr=sample_rate, mono=mono)
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def load_video_audio_or_none(video_path: str, sample_rate: int, logger, feature_name: str, mono: bool = True) -> tuple[np.ndarray, int] | None:
    try:
        waveform, sr = load_video_audio(video_path, sample_rate=sample_rate, mono=mono)
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("%s: failed to load audio from %s: %s", feature_name, video_path, error)
        return None
    if waveform is None or waveform.size == 0:
        logger.warning("%s: empty waveform for %s", feature_name, video_path)
        return None
    return waveform, sr


def skip_if_exists(existing_features, produced: set[str], logger, feature_name: str) -> bool:
    if existing_features and produced.issubset(set(existing_features)):
        logger.info("%s already exists, skipping", feature_name)
        return True
    return False


def video_duration_seconds(video_path: str) -> int:
    return max(0, int(np.ceil(get_video_duration(video_path))))


def prompt_centroids(text_embeddings: np.ndarray, prompt_lengths: Sequence[int], make_centroid) -> np.ndarray:
    prompt_start, centroids = 0, []
    for prompt_count in prompt_lengths:
        centroids.append(make_centroid(text_embeddings, list(range(prompt_start, prompt_start + prompt_count))))
        prompt_start += prompt_count
    return np.stack(centroids)
