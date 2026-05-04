import math
import os

import cv2
import librosa
import numpy as np
import pandas as pd

from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.logger import Logger
from ..video.shot_segmentation import batch_shot_segmentation
from .common import load_video_audio, stem_path
from .consts import BEAT_SYNC_WINDOW_SEC, BEAT_TOLERANCE_SEC, DEFAULT_AUDIO_SR, STEM_DRUMS, STEM_MIXED


logger = Logger(show=True).get_logger()
BEAT_SYNC_COLS = {"beat_sync", "beat_sync_ratio"}


def _find_music_stem(video_path: str) -> str | None:
    for candidate in (stem_path(video_path, STEM_MIXED), stem_path(video_path, STEM_DRUMS)):
        if os.path.isfile(candidate):
            return stem_path(video_path, STEM_MIXED)
    return None


def _get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        cap.release()
    return fps


def extract_beat_sync(video_path: str, config: Config, existing_features: list | None = None) -> pd.DataFrame:
    if existing_features and BEAT_SYNC_COLS.issubset(set(existing_features)):
        logger.info("beat_sync already exists, skipping")
        return pd.DataFrame()

    duration = math.ceil(get_video_duration(video_path))
    if duration < 60:
        return pd.DataFrame({"beat_sync": np.zeros(max(duration, 1)), "beat_sync_ratio": np.zeros(max(duration, 1))})

    music_stem = _find_music_stem(video_path)
    if music_stem:
        logger.info("Beat-sync: using source-separated music track %s", music_stem)
        y, sr = librosa.load(music_stem, sr=DEFAULT_AUDIO_SR)
    else:
        logger.warning("Beat-sync: no separated music stem found, falling back to full mix")
        y, sr = load_video_audio(video_path, sample_rate=DEFAULT_AUDIO_SR)

    _, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    logger.info("Detected %d beats in %.0f sec audio", len(beat_times), duration)

    if len(beat_times) == 0:
        return pd.DataFrame({"beat_sync": np.zeros(duration), "beat_sync_ratio": np.zeros(duration)})

    scenes = batch_shot_segmentation(video_path, config)
    if len(scenes) <= 1:
        return pd.DataFrame({"beat_sync": np.zeros(duration), "beat_sync_ratio": np.zeros(duration)})
    cut_times = scenes[1:, 0].astype(np.float64) / _get_video_fps(video_path)

    is_synced = np.array([np.min(np.abs(beat_times - ct)) for ct in cut_times]) < BEAT_TOLERANCE_SEC
    ratio = float(np.mean(is_synced))
    logger.info("Beat-sync: %d/%d cuts within %.0fms of a beat (ratio=%.2f)", int(is_synced.sum()), len(cut_times), BEAT_TOLERANCE_SEC * 1000, ratio)

    synced_cut_times = cut_times[is_synced]
    beat_sync = np.zeros(duration, dtype=np.float64)
    beat_sync_ratio = np.zeros(duration, dtype=np.float64)
    half_w = BEAT_SYNC_WINDOW_SEC // 2

    for t in range(duration):
        lo = t - half_w
        hi = t + half_w + 1
        n_cuts = int(np.sum((cut_times >= lo) & (cut_times < hi)))
        n_synced = int(np.sum((synced_cut_times >= lo) & (synced_cut_times < hi)))
        window_min = (min(hi, duration) - max(lo, 0)) / 60.0
        beat_sync[t] = n_synced / window_min if window_min > 0 else 0.0
        beat_sync_ratio[t] = n_synced / n_cuts if n_cuts > 0 else 0.0

    return pd.DataFrame({"beat_sync": beat_sync, "beat_sync_ratio": beat_sync_ratio})
