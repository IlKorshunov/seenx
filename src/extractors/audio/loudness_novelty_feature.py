# dont used in the current pipeline
from __future__ import annotations

import librosa
import numpy as np
import pandas as pd

from ...utils.config import Config
from ...utils.logger import Logger
from .common import load_video_audio_or_none, skip_if_exists, video_duration_seconds
from .consts import DEFAULT_AUDIO_SR, FRAME_LENGTH, HOP_LENGTH, LOUDNESS_MIN_PERIODS_ROLL, LOUDNESS_NOVELTY_ROLLING_SEC, LOUDNESS_Z_DROP, LOUDNESS_Z_SPIKE


logger = Logger(show=True).get_logger()
LOUDNESS_NOVELTY_COLS = {"loudness_zscore", "loudness_spike", "loudness_drop"}


def extract_loudness_novelty(video_path: str, config: Config, existing_features: list | None = None) -> pd.DataFrame:
    if skip_if_exists(existing_features, LOUDNESS_NOVELTY_COLS, logger, "Loudness novelty"):
        return pd.DataFrame()
    duration = video_duration_seconds(video_path)
    loaded_audio = load_video_audio_or_none(video_path, sample_rate=DEFAULT_AUDIO_SR, logger=logger, feature_name="Loudness novelty")
    if loaded_audio is None:
        return pd.DataFrame({column: np.zeros(duration) for column in LOUDNESS_NOVELTY_COLS})
    waveform, sample_rate = loaded_audio

    rms_fine = librosa.feature.rms(y=waveform, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH).flatten()

    frames_per_sec = sample_rate / HOP_LENGTH
    rms_per_sec = np.array(
        [
            float(np.mean(rms_fine[int(sec * frames_per_sec) : int(min((sec + 1) * frames_per_sec, len(rms_fine)))]))
            if int(sec * frames_per_sec) < int(min((sec + 1) * frames_per_sec, len(rms_fine)))
            else 0.0
            for sec in range(duration)
        ],
        dtype=np.float64,
    )

    rms_series = pd.Series(rms_per_sec, dtype=np.float64)
    rolling_rms = rms_series.rolling(window=LOUDNESS_NOVELTY_ROLLING_SEC, center=True, min_periods=LOUDNESS_MIN_PERIODS_ROLL)
    zscore = (rms_series - rolling_rms.mean()) / rolling_rms.std(ddof=0).replace(0.0, np.nan)
    zscore_values = zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    zscore_values = np.where(np.isfinite(zscore_values), zscore_values, 0.0)
    spike = (zscore_values > LOUDNESS_Z_SPIKE).astype(np.int8)
    drop = (zscore_values < LOUDNESS_Z_DROP).astype(np.int8)
    return pd.DataFrame({"loudness_zscore": zscore_values, "loudness_spike": spike, "loudness_drop": drop})
