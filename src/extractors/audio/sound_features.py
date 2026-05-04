"""Audio feature extraction summary.
 rms zcr centroid rolloff
vocal_rms, vocal_zcr, vocal_centroid, vocal_rolloff music_rms, music_zcr, music_centroid, music_rolloff
"""

import os
import shutil
import tempfile

import librosa
import pandas as pd

from ...utils.config import Config
from ...utils.logger import Logger
from .common import stems_dir
from .consts import STEM_FILES
from .source_separation import combine, mp4_to_wav, separate


logger = Logger(show=True).get_logger()
SOUND_FEATURE_NAMES = ("rms", "zcr", "centroid", "rolloff")


def get_vocal_music_features(audio_path: str, config: Config, existing_features: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if existing_features is None:
        existing_features = []
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    wav_file_path = tmp.name
    logger.info(f"Converting mp4 {audio_path} to wav {wav_file_path}")
    mp4_to_wav(audio_path, wav_file_path)
    outp = config.get("source_separation_dir")
    logger.info("Separating %s into music and vocals to %s", wav_file_path, outp)
    ok = separate([wav_file_path], outp=outp, device=config.get("demucs_device"), segment=config.get("demucs_segment"))
    filename, _ = os.path.splitext(os.path.basename(wav_file_path))
    separated_folder = f"{outp}/htdemucs/{filename}"
    if not ok:
        os.unlink(wav_file_path)
        logger.warning("Skipping music/vocal features because Demucs separation failed")
        return pd.DataFrame(), pd.DataFrame()

    music_path, vocal_path = combine(separated_folder)
    vocal_features = sound_features_pipeline(vocal_path, fps=1, prefix="vocal_", existing_features=existing_features)
    music_features = sound_features_pipeline(music_path, fps=1, prefix="music_", existing_features=existing_features)
    os.unlink(wav_file_path)

    target_stems_dir = stems_dir(audio_path)
    os.makedirs(target_stems_dir, exist_ok=True)
    for stem_name in STEM_FILES:
        src = os.path.join(separated_folder, stem_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(target_stems_dir, stem_name))
    logger.info("Saved separated stems to %s", target_stems_dir)
    shutil.rmtree(separated_folder)
    return music_features, vocal_features


def sound_features_pipeline(audio_file_path: str, fps: int = 1, prefix: str = "", existing_features: list | None = None) -> pd.DataFrame:
    if existing_features is None:
        existing_features = []
    feature_names = [f"{prefix}{feature_name}" for feature_name in SOUND_FEATURE_NAMES]
    if all(feature in existing_features for feature in feature_names):
        logger.info(f"Sound features for {prefix} already exist, skipping extraction")
        return pd.DataFrame()

    y, sr = librosa.load(audio_file_path, sr=None)
    logger.info(f"Audio file: {audio_file_path} shape: {y.shape}, sample rate: {sr}")
    frame_length = sr // fps
    hop_length = sr // fps
    logger.info(f"Extracting features with {frame_length=}, {hop_length=}")
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    features = pd.DataFrame({f"{prefix}{feature_name}": values.flatten() for feature_name, values in zip(SOUND_FEATURE_NAMES, (rms, zcr, centroid, rolloff), strict=True)})
    return features
