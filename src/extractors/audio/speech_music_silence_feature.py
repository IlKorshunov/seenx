import numpy as np
import pandas as pd

from .consts import MUSIC_CHANGE_PERCENTILE, SILENCE_RMS_THRESHOLD, SILENCE_STRETCH_MIN_SEC, SPECTRAL_EPS, SPEECH_MUSIC_WINDOW_SEC


def extract_speech_music_silence(
    vocal_rms: np.ndarray, music_rms: np.ndarray, window_sec: int = SPEECH_MUSIC_WINDOW_SEC, silence_rms_threshold: float = SILENCE_RMS_THRESHOLD
) -> pd.DataFrame:
    n = len(vocal_rms)
    vocal = np.asarray(vocal_rms, dtype=float).flatten()
    music = np.asarray(music_rms, dtype=float).flatten()
    if len(vocal) != len(music):
        min_len = min(len(vocal), len(music))
        vocal = vocal[:min_len]
        music = music[:min_len]

    window = min(window_sec, n)
    kernel = np.ones(window) / window
    speech_ratio = np.convolve((vocal > music).astype(float), kernel, mode="same")
    speech_ratio = np.clip(speech_ratio, 0, 1)

    silence = (vocal < silence_rms_threshold) & (music < silence_rms_threshold)
    silence_stretch = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        if silence[i]:
            start = i
            while i < n and silence[i]:
                i += 1
            if (i - start) >= SILENCE_STRETCH_MIN_SEC:
                silence_stretch[start:i] = 1.0
        else:
            i += 1
    music_only = ((music > silence_rms_threshold) & (vocal < silence_rms_threshold)).astype(float)
    return pd.DataFrame({"speech_ratio": speech_ratio, "silence_stretch": silence_stretch, "music_only": music_only})


def extract_background_music_features(
    music_rms: np.ndarray,
    music_centroid: np.ndarray,
    music_rolloff: np.ndarray,
    music_present_threshold: float = SILENCE_RMS_THRESHOLD,
    change_percentile: float = MUSIC_CHANGE_PERCENTILE,
) -> pd.DataFrame:
    n = len(music_rms)
    music_rms = np.asarray(music_rms, dtype=float).flatten()
    music_centroid = np.asarray(music_centroid, dtype=float).flatten()
    music_rolloff = np.asarray(music_rolloff, dtype=float).flatten()
    min_len = min(len(music_rms), len(music_centroid), len(music_rolloff), n)
    music_rms = music_rms[:min_len]
    music_centroid = music_centroid[:min_len]
    music_rolloff = music_rolloff[:min_len]

    has_background_music = (music_rms > music_present_threshold).astype(np.float64)

    music_changed = np.zeros(min_len, dtype=np.float64)
    if min_len > 1:
        scale_rms = np.std(music_rms) + SPECTRAL_EPS
        scale_c = np.std(music_centroid) + SPECTRAL_EPS
        scale_r = np.std(music_rolloff) + SPECTRAL_EPS
        diff_rms = np.abs(np.diff(music_rms)) / scale_rms
        diff_c = np.abs(np.diff(music_centroid)) / scale_c
        diff_r = np.abs(np.diff(music_rolloff)) / scale_r
        change_score = diff_rms + diff_c + diff_r
        threshold = np.percentile(change_score, change_percentile)
        both_have_music = has_background_music[:-1].astype(bool) & has_background_music[1:].astype(bool)
        music_changed[1:] = np.where((change_score >= threshold) & (both_have_music > 0), 1.0, 0.0)
    return pd.DataFrame({"has_background_music": has_background_music, "music_changed": music_changed})
