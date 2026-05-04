import math

import librosa
import numpy as np
import pandas as pd

from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.logger import Logger
from .common import load_video_audio
from .consts import DEFAULT_AUDIO_SR, FRAME_LENGTH, HOP_LENGTH, LOUDNESS_VARIANCE_WINDOW_SEC


logger = Logger(show=True).get_logger()
LOUDNESS_DYNAMICS_COLS = {"loudness_change", "loudness_variance"}


def extract_loudness_dynamics(video_path: str, config: Config, existing_features: list | None = None) -> pd.DataFrame:
    if existing_features and LOUDNESS_DYNAMICS_COLS.issubset(set(existing_features)):
        logger.info("Loudness dynamics already exist, skipping")
        return pd.DataFrame()

    duration = math.ceil(get_video_duration(video_path))
    waveform, sample_rate = load_video_audio(video_path, sample_rate=DEFAULT_AUDIO_SR)

    rms_fine = librosa.feature.rms(y=waveform, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH).flatten()
    frames_per_sec = sample_rate / HOP_LENGTH
    rms_per_sec = np.array(
        [
            float(np.mean(rms_fine[int(sec * frames_per_sec) : int(min((sec + 1) * frames_per_sec, len(rms_fine)))]))
            if int(sec * frames_per_sec) < int(min((sec + 1) * frames_per_sec, len(rms_fine)))
            else 0.0
            for sec in range(duration)
        ]
    )

    loudness_change = np.abs(np.gradient(rms_per_sec))
    kernel_size = min(5, len(loudness_change))
    loudness_change = np.convolve(loudness_change, np.ones(kernel_size) / kernel_size, mode="same")

    half_w = LOUDNESS_VARIANCE_WINDOW_SEC // 2
    loudness_variance = np.array([float(np.std(rms_per_sec[max(0, sec - half_w) : min(duration, sec + half_w + 1)])) for sec in range(duration)], dtype=np.float64)

    return pd.DataFrame({"loudness_change": loudness_change, "loudness_variance": loudness_variance})
