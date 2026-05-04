import os

import librosa
import numpy as np
import pandas as pd

from ...seenx_utils import get_video_duration
from ...utils.logger import Logger
from .clap_zero_shot import encode_text_prompts, load_audio_embeddings, make_centroid, zero_shot_classify
from .common import prompt_centroids, stem_path
from .consts import (
    CLAP_MODEL_ID,
    DEFAULT_AUDIO_SR,
    HOP_LENGTH,
    MUSIC_PROMPTS,
    SFX_PROMPTS,
    SILENCE_PROMPTS,
    SPEECH_PROMPTS,
    STEM_DRUMS,
    STEM_MIXED,
    STEM_OTHER,
    ZERO_SHOT_TEMPERATURE,
)


logger = Logger(show=True).get_logger()
SFX_COL = "sfx_energy"


def _clap_sfx_signal(video_path: str, n_seconds: int, config=None) -> np.ndarray:
    signal = np.zeros(n_seconds, dtype=np.float64)
    try:
        audio_embeddings = load_audio_embeddings(video_path)
        if audio_embeddings is None:
            logger.info("No CLAP audio embeddings for SFX detection")
            return signal

        all_prompts = SFX_PROMPTS + SPEECH_PROMPTS + MUSIC_PROMPTS + SILENCE_PROMPTS
        text_embeddings = encode_text_prompts(all_prompts, model_id=config.get("clap_model", CLAP_MODEL_ID) if config else CLAP_MODEL_ID)
        centroids = prompt_centroids(text_embeddings, [len(SFX_PROMPTS), len(SPEECH_PROMPTS), len(MUSIC_PROMPTS), len(SILENCE_PROMPTS)], make_centroid)
        sfx_probs = zero_shot_classify(audio_embeddings, centroids, temperature=ZERO_SHOT_TEMPERATURE)[:, 0]
        limit = min(n_seconds, len(sfx_probs))
        signal[:limit] = sfx_probs[:limit]
        logger.info("CLAP SFX signal: mean=%.4f max=%.4f >0.1 count=%d", signal.mean(), signal.max(), int((signal > 0.1).sum()))
    except Exception as error:
        logger.warning("CLAP SFX signal failed: %s", error)
    return signal


def _stem_energy_signal(video_path: str, n_seconds: int) -> np.ndarray:
    signal = np.zeros(n_seconds, dtype=np.float64)
    other_path = stem_path(video_path, STEM_OTHER)
    mixed_path = stem_path(video_path, STEM_MIXED)
    drums_path = stem_path(video_path, STEM_DRUMS)

    if not os.path.exists(other_path):
        return signal

    try:
        target_sr = DEFAULT_AUDIO_SR
        other_waveform, _ = librosa.load(other_path, sr=target_sr, mono=True)

        drums_waveform = np.zeros_like(other_waveform)
        if os.path.exists(drums_path):
            try:
                loaded_drums, _ = librosa.load(drums_path, sr=target_sr, mono=True)
                min_length = min(len(loaded_drums), len(other_waveform))
                drums_waveform[:min_length] = loaded_drums[:min_length]
            except Exception:
                pass

        mix_waveform = None
        if os.path.exists(mixed_path):
            try:
                mix_waveform, _ = librosa.load(mixed_path, sr=target_sr, mono=True)
            except Exception:
                pass

        sfx_waveform = other_waveform + drums_waveform
        samples_per_sec = target_sr
        rms_sfx = np.zeros(n_seconds, dtype=np.float64)
        rms_mix = np.ones(n_seconds, dtype=np.float64)

        for sec in range(n_seconds):
            start_sample = sec * samples_per_sec
            end_sample = min((sec + 1) * samples_per_sec, len(sfx_waveform))
            if start_sample >= end_sample:
                continue
            rms_sfx[sec] = float(np.sqrt(np.mean(sfx_waveform[start_sample:end_sample] ** 2)))
            if mix_waveform is not None and end_sample <= len(mix_waveform):
                rms_mix[sec] = max(float(np.sqrt(np.mean(mix_waveform[start_sample:end_sample] ** 2))), 1e-8)

        ratio = rms_sfx / rms_mix if mix_waveform is not None else rms_sfx

        median_ratio = float(np.median(ratio))
        residual = np.maximum(ratio - median_ratio, 0.0)
        percentile_95 = float(np.percentile(residual[residual > 0], 95)) if np.any(residual > 0) else 1.0
        signal = np.clip(residual / max(percentile_95, 1e-8), 0.0, 1.0)
        signal = 1.0 / (1.0 + np.exp(-15.0 * (signal - 0.55)))

        logger.info("Stem energy SFX signal: mean=%.4f max=%.4f >0.1 count=%d", signal.mean(), signal.max(), int((signal > 0.1).sum()))
    except Exception as error:
        logger.warning("Stem energy SFX signal failed: %s", error)
    return signal


def _onset_sfx_signal(video_path: str, n_seconds: int) -> np.ndarray:
    signal = np.zeros(n_seconds, dtype=np.float64)
    other_path = stem_path(video_path, STEM_OTHER)

    if not os.path.exists(other_path):
        return signal

    try:
        waveform, sample_rate = librosa.load(other_path, sr=DEFAULT_AUDIO_SR, mono=True)
        if len(waveform) == 0:
            return signal

        onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate, hop_length=HOP_LENGTH)

        frames_per_sec = sample_rate / HOP_LENGTH
        for sec in range(n_seconds):
            start_frame = int(sec * frames_per_sec)
            end_frame = min(int((sec + 1) * frames_per_sec), len(onset_env))
            if start_frame >= end_frame:
                continue
            signal[sec] = float(np.mean(onset_env[start_frame:end_frame]))

        median_signal = float(np.median(signal))
        residual = np.maximum(signal - median_signal, 0.0)
        percentile_95 = float(np.percentile(residual[residual > 0], 95)) if np.any(residual > 0) else 1.0
        signal = np.clip(residual / max(percentile_95, 1e-8), 0.0, 1.0)
        signal = 1.0 / (1.0 + np.exp(-12.0 * (signal - 0.50)))

        logger.info("Onset SFX signal: mean=%.4f max=%.4f >0.1 count=%d", signal.mean(), signal.max(), int((signal > 0.1).sum()))
    except Exception as error:
        logger.warning("Onset SFX signal failed: %s", error)
    return signal


def extract_sfx_energy(video_path: str, config=None, existing_features=None) -> pd.DataFrame:
    if existing_features and SFX_COL in (existing_features if isinstance(existing_features, set) else set(existing_features)):
        logger.info("sfx_energy already exists, skipping")
        return pd.DataFrame()

    duration = get_video_duration(video_path)
    n_seconds = max(1, int(np.ceil(duration)))

    clap_signal = _clap_sfx_signal(video_path, n_seconds, config)
    stem_signal = _stem_energy_signal(video_path, n_seconds)
    onset_signal = _onset_sfx_signal(video_path, n_seconds)

    has_clap = clap_signal.max() > 0.01

    if has_clap:
        clap_weight, stem_weight, onset_weight = 0.55, 0.25, 0.20
    else:
        clap_weight, stem_weight, onset_weight = 0.0, 0.55, 0.45

    fused = clap_weight * clap_signal + stem_weight * stem_signal + onset_weight * onset_signal
    fused = np.clip(fused, 0.0, 1.0)

    logger.info(
        "sfx_energy fused: mean=%.4f max=%.4f >0.1=%d  weights=(c=%.2f s=%.2f o=%.2f)", fused.mean(), fused.max(), int((fused > 0.1).sum()), clap_weight, stem_weight, onset_weight
    )

    return pd.DataFrame({SFX_COL: fused.astype(np.float32)})
