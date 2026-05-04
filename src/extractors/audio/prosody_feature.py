"""
Привлекательность голоса, тембра
Постановка речи
Output: pitch_mean, pitch_std, voiced_frac, pause_rate, speech_rate_cv
"""

import math
import librosa
import numpy as np
import pandas as pd
from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.logger import Logger
from ...utils.transcript_cache import get_transcript
from .common import load_video_audio
from .consts import DEFAULT_AUDIO_SR, FRAME_LENGTH

logger = Logger(show=True).get_logger()


def extract_prosody(video_path: str, config: Config, existing_features: list | None = None) -> pd.DataFrame:
    target_cols = {"pitch_mean", "pitch_std", "voiced_frac", "pause_rate", "speech_rate_cv"}
    if existing_features and target_cols.issubset(set(existing_features)):
        logger.info("Prosody features already exist, skipping")
        return pd.DataFrame()
    duration = math.ceil(get_video_duration(video_path))
    analysis_window_sec = int(config.get("prosody_analysis_window_sec", 5))
    chunk_sec = int(config.get("prosody_chunk_sec", 300))
    frame_length = int(config.get("prosody_frame_length", FRAME_LENGTH))
    fmin = float(config.get("prosody_fmin", 50.0))
    fmax = float(config.get("prosody_fmax", 500.0))
    pause_gap_sec = float(config.get("prosody_pause_gap_sec", 0.3))
    min_mean_wps = float(config.get("prosody_min_mean_wps", 0.1))

    waveform, sample_rate = load_video_audio(video_path, sample_rate=int(config.get("prosody_sr", DEFAULT_AUDIO_SR)))

    chunk_samples = chunk_sec * sample_rate
    hop_length = frame_length // 4

    pitch_chunks: list[np.ndarray] = []
    voiced_chunks: list[np.ndarray] = []
    total_chunks = 0
    failed_chunks = 0
    failed_seconds = 0.0

    for chunk_start in range(0, len(waveform), chunk_samples):
        chunk = waveform[chunk_start : chunk_start + chunk_samples]
        total_chunks += 1
        try:
            pitch_chunk, voiced_chunk, _ = librosa.pyin(chunk, fmin=fmin, fmax=fmax, sr=sample_rate, frame_length=frame_length, hop_length=hop_length)
        except Exception as error:
            failed_chunks += 1
            failed_seconds += len(chunk) / float(sample_rate)
            n_frames = max(1, int(np.ceil(len(chunk) / max(1, hop_length))))
            pitch_chunk = np.full(n_frames, np.nan, dtype=np.float64)
            voiced_chunk = np.zeros(n_frames, dtype=bool)
            logger.warning("Prosody pyin failed on chunk %d (%.1f sec at %.1f sec): %s", total_chunks, len(chunk) / float(sample_rate), chunk_start / float(sample_rate), error)
        pitch_chunks.append(pitch_chunk)
        voiced_chunks.append(voiced_chunk)

    pitch_values, voiced_flag = np.concatenate(pitch_chunks), np.concatenate(voiced_chunks)
    pitch_frames_per_sec = sample_rate / hop_length
    if total_chunks > 0:
        fail_pct = 100.0 * failed_chunks / total_chunks
        fail_sec_pct = 100.0 * failed_seconds / max(duration, 1e-9)
        logger.info("Prosody pyin failures: %d/%d chunks (%.1f%%), affected duration %.1f%%", failed_chunks, total_chunks, fail_pct, fail_sec_pct)

    pitch_mean, pitch_std, voiced_frac = (np.zeros(duration, dtype=np.float64) for _ in range(3))
    half_window = analysis_window_sec // 2
    for sec in range(duration):
        frame_start = int(max(0, sec - half_window) * pitch_frames_per_sec)
        frame_end = int(min(min(duration, sec + half_window + 1) * pitch_frames_per_sec, len(pitch_values)))
        if frame_end <= frame_start:
            continue

        window_pitch = pitch_values[frame_start:frame_end]
        window_voiced = voiced_flag[frame_start:frame_end]
        valid_pitch = window_pitch[window_voiced & np.isfinite(window_pitch)]
        voiced_frac[sec] = np.sum(window_voiced) / max(len(window_voiced), 1)
        if len(valid_pitch) > 0:
            pitch_mean[sec], pitch_std[sec] = float(np.mean(valid_pitch)), float(np.std(valid_pitch))

    segments = get_transcript(video_path, config)["segments"]
    pause_rate, speech_rate_cv = np.zeros(duration, dtype=np.float64), np.zeros(duration, dtype=np.float64)

    segment_starts = np.array([segment["start"] for segment in segments])
    segment_ends = np.array([segment["end"] for segment in segments])
    segment_wps = np.array([len(segment["text"].split()) / max(segment["end"] - segment["start"], 0.1) for segment in segments])
    for sec in range(duration):
        window_start = max(0.0, sec - half_window)
        window_end = min(float(duration), sec + half_window + 1)
        segment_indices = np.where((segment_ends > window_start) & (segment_starts < window_end))[0]

        if len(segment_indices) >= 2:
            n_pauses = sum((segment_starts[segment_indices[index + 1]] - segment_ends[segment_indices[index]]) > pause_gap_sec for index in range(len(segment_indices) - 1))
            window_duration_min = (window_end - window_start) / 60.0
            pause_rate[sec] = n_pauses / window_duration_min if window_duration_min > 0 else 0.0
            window_wps = segment_wps[segment_indices]
            mean_wps = np.mean(window_wps)
            if mean_wps > min_mean_wps:
                speech_rate_cv[sec] = float(np.std(window_wps) / mean_wps)

    return pd.DataFrame({"pitch_mean": pitch_mean, "pitch_std": pitch_std, "voiced_frac": voiced_frac, "pause_rate": pause_rate, "speech_rate_cv": speech_rate_cv})
