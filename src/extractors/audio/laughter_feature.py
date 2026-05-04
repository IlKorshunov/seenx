import math
import os
import re
import librosa
import numpy as np
import pandas as pd
from ...seenx_utils import get_video_duration
from ...utils.logger import Logger
from ...utils.transcript_cache import get_transcript
from .clap_zero_shot import encode_text_prompts, load_audio_embeddings, make_centroid, zero_shot_classify
from .common import prompt_centroids, stem_path
from .consts import *


logger = Logger(show=True).get_logger()
LAUGHTER_COL = "laughter_prob"
_LAUGHTER_RE = re.compile(LAUGHTER_RE_PATTERN, re.IGNORECASE)


def _whisper_laughter_signal(video_path, config, n_seconds):
    signal = np.zeros(n_seconds, dtype=np.float64)
    try:
        segments = get_transcript(video_path, config).get("segments", [])
    except Exception as e:
        logger.warning("Whisper transcript unavailable for laughter: %s", e)
        return signal

    for seg in segments:
        if not _LAUGHTER_RE.search(seg.get("text", "")):
            continue
        word_hits = [w for w in seg.get("words", []) if _LAUGHTER_RE.search(w.get("word", ""))]
        spans = [(w.get("start", seg["start"]), w.get("end", seg["end"])) for w in word_hits] or [(seg["start"], seg["end"])]
        for s, e in spans:
            signal[max(0, int(math.floor(s))) : min(n_seconds, int(math.ceil(e)))] = 1.0
    return signal


def _clap_laughter_signal(video_path, n_seconds, config=None):
    signal = np.zeros(n_seconds, dtype=np.float64)
    try:
        audio_emb = load_audio_embeddings(video_path)
        if audio_emb is None:
            return signal
        model_id = (config or {}).get("clap_model", CLAP_MODEL_ID)
        positive = LAUGHTER_PROMPTS
        negative = SPEECH_PROMPTS + MUSIC_PROMPTS + LAUGHTER_SILENCE_PROMPTS
        text_emb = encode_text_prompts(positive + negative, model_id=model_id)
        centroids = prompt_centroids(text_emb, [len(positive), len(negative)], make_centroid)
        probs = zero_shot_classify(audio_emb, centroids, temperature=ZERO_SHOT_TEMPERATURE)[:, 0]
        limit = min(n_seconds, len(probs))
        signal[:limit] = probs[:limit]
        logger.info("CLAP laughter: mean=%.3f max=%.3f >0.5=%d", signal.mean(), signal.max(), int((signal > 0.5).sum()))
    except Exception as e:
        logger.warning("CLAP laughter signal failed: %s", e)
    return signal


def _acoustic_laughter_signal(video_path, n_seconds):
    signal = np.zeros(n_seconds, dtype=np.float64)
    vocal_path = stem_path(video_path, STEM_VOCALS)
    if not os.path.exists(vocal_path):
        return signal
    try:
        waveform, sr = librosa.load(vocal_path, sr=DEFAULT_AUDIO_SR, mono=True)
    except Exception as e:
        logger.warning("Cannot load vocal stem: %s", e)
        return signal
    if len(waveform) == 0:
        return signal

    rms = librosa.feature.rms(y=waveform, hop_length=HOP_LENGTH)[0]
    env_sr = sr / HOP_LENGTH
    win = int(env_sr * 1.0)
    step = max(1, int(env_sr * 0.25))

    scores, centers = [], []
    for start in range(0, max(1, len(rms) - win + 1), step):
        seg = rms[start : start + win]
        if len(seg) < 16 or seg.max() < 1e-6:
            continue
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg)) ** 2
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / env_sr)
        ratio = float(spec[(freqs >= 4.0) & (freqs <= 8.0)].sum() / (spec.sum() + 1e-8))
        scores.append(ratio)
        centers.append((start + win / 2) / env_sr)

    if not scores:
        return signal
    scores = np.asarray(scores)
    centers = np.asarray(centers)
    for sec in range(n_seconds):
        mask = (centers >= sec - 0.5) & (centers < sec + 1.5)
        if mask.any():
            signal[sec] = scores[mask].max()

    return np.clip((signal - 0.10) / 0.25, 0.0, 1.0)


def extract_laughter(video_path, config=None, existing_features=None):
    if existing_features and LAUGHTER_COL in (existing_features if isinstance(existing_features, set) else set(existing_features)):
        logger.info("laughter_prob already exists, skipping")
        return pd.DataFrame()

    n_seconds = math.ceil(get_video_duration(video_path))
    whisper_sig = _whisper_laughter_signal(video_path, config, n_seconds)
    clap_sig = _clap_laughter_signal(video_path, n_seconds, config)
    acoustic_sig = _acoustic_laughter_signal(video_path, n_seconds)
    fused = LAUGHTER_CLAP_WEIGHT * clap_sig + LAUGHTER_ACOUSTIC_WEIGHT * acoustic_sig
    fused = np.where(whisper_sig >= LAUGHTER_REGEX_POSITIVE, 1.0, fused)
    fused = np.clip(fused, 0.0, 1.0)
    logger.info(
        "laughter_prob: mean=%.3f max=%.3f >0.5=%d (whisper=%d, clap mean=%.3f, ac mean=%.3f)",
        fused.mean(),
        fused.max(),
        int((fused > 0.5).sum()),
        int((whisper_sig >= LAUGHTER_REGEX_POSITIVE).sum()),
        clap_sig.mean(),
        acoustic_sig.mean(),
    )
    return pd.DataFrame({LAUGHTER_COL: fused.astype(np.float32)})
