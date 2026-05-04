import librosa
import numpy as np
import pandas as pd

from ...utils.logger import Logger
from ...seenx_utils import get_video_duration
from .consts import DEFAULT_AUDIO_SR, FRAME_LENGTH, HOP_LENGTH

logger = Logger(show=True).get_logger()
SPECTRAL_FLUX_COL = "spectral_flux"


def extract_spectral_flux(video_path: str, config=None, existing_features: list | None = None) -> pd.DataFrame:
    if existing_features and SPECTRAL_FLUX_COL in (existing_features if isinstance(existing_features, set) else set(existing_features)):
        logger.info("spectral_flux already exists, skipping")
        return pd.DataFrame()
    duration = get_video_duration(video_path)
    n_seconds = max(1, int(np.ceil(duration)))
    y, sr = librosa.load(video_path, sr=DEFAULT_AUDIO_SR, mono=True)
    if len(y) == 0:
        return pd.DataFrame({SPECTRAL_FLUX_COL: np.zeros(n_seconds, dtype=np.float32)})

    S = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
    flux = np.zeros(S.shape[1], dtype=np.float64)
    for t in range(1, S.shape[1]):
        flux[t] = np.sum(np.abs(S[:, t] - S[:, t - 1]))

    frames_per_sec = sr / HOP_LENGTH
    per_second = np.zeros(n_seconds, dtype=np.float32)
    for sec in range(n_seconds):
        start_frame = int(sec * frames_per_sec)
        end_frame = int((sec + 1) * frames_per_sec)
        end_frame = min(end_frame, len(flux))
        if start_frame < end_frame:
            per_second[sec] = float(np.mean(flux[start_frame:end_frame]))

    return pd.DataFrame({SPECTRAL_FLUX_COL: per_second})
