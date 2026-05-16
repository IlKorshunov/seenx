import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ._base import get_segments_and_duration, seg_bounds, segment_text, skip_if_exists
from .common import release_models
from .constants import TEXT_EMOTION_LABELS, TEXT_SENTIMENT_BATCH_SIZE, TEXT_SENTIMENT_COLS, TEXT_SENTIMENT_MAX_LENGTH, TEXT_SENTIMENT_MODEL_ID


def extract_text_sentiment(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(TEXT_SENTIMENT_COLS, existing_features, "sentiment"):
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)
    texts, seg_ranges = [], []
    for seg in segments:
        text = segment_text(seg)
        start_sec, end_sec = seg_bounds(seg, duration)
        texts.append(text)
        seg_ranges.append((start_sec, end_sec))

    out = np.zeros((duration, len(TEXT_EMOTION_LABELS)), dtype=np.float64)
    device = config.get("device")
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(TEXT_SENTIMENT_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(TEXT_SENTIMENT_MODEL_ID, torch_dtype=dtype).to(device).eval()
    for batch_start in range(0, len(texts), TEXT_SENTIMENT_BATCH_SIZE):
        enc = tokenizer(texts[batch_start : batch_start + TEXT_SENTIMENT_BATCH_SIZE], padding=True, truncation=True, max_length=TEXT_SENTIMENT_MAX_LENGTH, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            probs = torch.sigmoid(model(**enc).logits).cpu().float().numpy()
        for batch_idx, (start_sec, end_sec) in enumerate(seg_ranges[batch_start : batch_start + TEXT_SENTIMENT_BATCH_SIZE]):
            out[start_sec:end_sec] = probs[batch_idx]
    release_models(model, tokenizer, device=device)

    return pd.DataFrame(out, columns=[f"sent_{emotion_label}" for emotion_label in TEXT_EMOTION_LABELS])
