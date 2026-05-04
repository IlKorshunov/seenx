"""Speech emotion recognition via wav2vec2. Output: voice_angry/happy/sad/neutral, voice_dominant_emotion_conf."""

import gc
import math
import numpy as np
import pandas as pd
import torch
from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.logger import Logger
from .common import load_video_audio
from .consts import SPEECH_AUDIO_SR, SPEECH_EMOTION_MODEL_ID
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

logger = Logger(show=True).get_logger()
SPEECH_EMOTION_LABELS = ["angry", "happy", "sad", "neutral"]
SPEECH_EMOTION_CONF_COL = "voice_dominant_emotion_conf"
SPEECH_EMOTION_COLS = {f"voice_{emotion}" for emotion in SPEECH_EMOTION_LABELS} | {SPEECH_EMOTION_CONF_COL}


def extract_speech_emotion(video_path: str, config: Config, existing_features=None) -> pd.DataFrame:
    if existing_features and SPEECH_EMOTION_COLS.issubset(set(existing_features)):
        return pd.DataFrame()

    duration = math.ceil(get_video_duration(video_path))
    waveform, sample_rate = load_video_audio(video_path, sample_rate=SPEECH_AUDIO_SR)
    device = torch.device(config.get("device", "cuda"))
    model_id = config.get("speech_emotion_model", SPEECH_EMOTION_MODEL_ID)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(device).eval()

    chunk_sec, chunk_samples = 5, 5 * sample_rate
    all_probabilities = []
    for chunk_index in range(max(1, math.ceil(len(waveform) / chunk_samples))):
        chunk = waveform[chunk_index * chunk_samples : (chunk_index + 1) * chunk_samples]
        if len(chunk) < sample_rate:
            chunk = np.pad(chunk, (0, sample_rate - len(chunk)))
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
        all_probabilities.append(probabilities)

    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output = np.zeros((duration, len(SPEECH_EMOTION_LABELS) + 1), dtype=np.float64)
    for chunk_index, probabilities in enumerate(all_probabilities):
        start_sec, end_sec = chunk_index * chunk_sec, min((chunk_index + 1) * chunk_sec, duration)
        output[start_sec:end_sec, : len(SPEECH_EMOTION_LABELS)] = probabilities
        output[start_sec:end_sec, -1] = float(np.max(probabilities))

    return pd.DataFrame(output, columns=[f"voice_{emotion}" for emotion in SPEECH_EMOTION_LABELS] + [SPEECH_EMOTION_CONF_COL])
