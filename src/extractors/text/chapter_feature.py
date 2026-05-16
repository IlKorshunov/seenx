import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from ._base import get_segments_and_duration, logger, seg_bounds, skip_if_exists
from .common import EMBEDDINGS_ROOT, load_segment_embeddings, release_models, video_id


MODEL_ID = "deepvk/USER2-base"
EMB_DIM = 256
_COLS = {"chapter_id", "n_chapters", "topic_change_rate"}

SHIFT_K = 1.5
MIN_CHAPTER_SEGMENTS = 3
TOPIC_CHANGE_WINDOW_SEC = 60


def _load_saved_embeddings(video_path: str):
    return load_segment_embeddings(video_id(video_path), EMBEDDINGS_ROOT, require_metadata=True)


def _encode_texts(texts, model, tokenizer, device, batch_size=128):
    all_embs = []
    for batch_start in range(0, len(texts), batch_size):
        encoded_inputs = tokenizer(texts[batch_start : batch_start + batch_size], padding=True, truncation=True, max_length=8192, return_tensors="pt").to(device)
        with torch.no_grad():
            model_output = model(**encoded_inputs)
        attention_mask = encoded_inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (model_output.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled[:, :EMB_DIM], p=2, dim=1)
        all_embs.append(pooled.cpu().float().numpy())
    return np.vstack(all_embs)


def extract_chapters(video_path: str, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "chapters"):
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)
    valid = [(seg, *seg_bounds(seg, duration)) for seg in segments if seg.get("text", "").strip()]

    if len(valid) < 2:
        logger.info("Chapters: too few segments (%d), returning defaults", len(valid))
        return pd.DataFrame({"chapter_id": np.zeros(duration), "n_chapters": np.ones(duration), "topic_change_rate": np.zeros(duration)})

    embs, _ = _load_saved_embeddings(video_path)
    if embs is None or len(embs) != len(valid):
        logger.info("Chapters: computing embeddings from scratch (%d segments)", len(valid))
        device = torch.device(config.get("device"))
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()
        embs = _encode_texts([seg["text"].strip() for seg, _, _ in valid], model, tokenizer, device)
        release_models(model, tokenizer, device=device)
    else:
        logger.info("Chapters: loaded saved embeddings (%d segments)", len(embs))

    shift = np.zeros(len(embs))
    sims = np.sum(embs[1:] * embs[:-1], axis=1)
    shift[1:] = 1.0 - sims

    threshold = float(np.mean(shift[1:])) + SHIFT_K * float(np.std(shift[1:]))
    is_boundary = np.zeros(len(embs), dtype=bool)
    for segment_idx in range(1, len(embs)):
        if shift[segment_idx] >= threshold:
            is_boundary[segment_idx] = True

    prev_boundary = 0
    for segment_idx in range(1, len(embs)):
        if is_boundary[segment_idx]:
            if (segment_idx - prev_boundary) < MIN_CHAPTER_SEGMENTS:
                is_boundary[segment_idx] = False
            else:
                prev_boundary = segment_idx

    chapter_ids_seg = np.cumsum(is_boundary).astype(np.float64)
    n_chapters = int(chapter_ids_seg[-1]) + 1

    chapter_id = np.zeros(duration, dtype=np.float64)
    shift_per_sec = np.zeros(duration, dtype=np.float64)
    for segment_idx, (_, start_sec, end_sec) in enumerate(valid):
        chapter_id[start_sec:end_sec] = chapter_ids_seg[segment_idx]
        if is_boundary[segment_idx]:
            mid_sec = (start_sec + end_sec) // 2
            if mid_sec < duration:
                shift_per_sec[mid_sec] = 1.0

    half_window = TOPIC_CHANGE_WINDOW_SEC // 2
    topic_change_rate = np.zeros(duration, dtype=np.float64)
    for second_idx in range(duration):
        window_start, window_end = max(0, second_idx - half_window), min(duration, second_idx + half_window + 1)
        window_min = (window_end - window_start) / 60.0
        topic_change_rate[second_idx] = shift_per_sec[window_start:window_end].sum() / window_min if window_min > 0 else 0.0

    logger.info("Chapters: %d chapters detected (threshold=%.3f), boundaries at segments %s", n_chapters, threshold, list(np.where(is_boundary)[0]))

    return pd.DataFrame({"chapter_id": chapter_id, "n_chapters": np.full(duration, float(n_chapters)), "topic_change_rate": topic_change_rate})
