"""Hook score + per-second binary is_question (rules + embedding fallback)."""

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from ._base import get_segments_and_duration, logger, skip_if_exists
from .common import collect_valid_segments, release_models
from .constants import (
    ADDRESS_WINDOW_SEC,
    CLAIM_WINDOW_SEC,
    HOOK_ADDRESS_W,
    HOOK_CLAIM_W,
    HOOK_DENSITY_W,
    HOOK_NUMBERS_W,
    HOOK_QUESTION_W,
    NUMBER_PATTERN,
    QUESTION_WINDOW_SEC,
    RU_ADDRESS,
    RU_CLAIMS,
)


_COLS = {"hook_score", "hook_has_address", "is_question"}

_QUESTION_WORDS = frozenset(
    {
        "кто",
        "что",
        "где",
        "когда",
        "почему",
        "зачем",
        "как",
        "сколько",
        "какой",
        "какая",
        "какое",
        "какие",
        "каким",
        "каких",
        "какому",
        "чем",
        "чему",
        "куда",
        "откуда",
        "неужели",
        "разве",
        "ли",
    }
)

_QUESTION_MODEL_ID = "deepvk/USER2-base"
_EMB_DIM = 256
_COSINE_THRESHOLD = 0.80
_QUESTION_ANCHORS = [
    "Почему так происходит?",
    "Как это работает?",
    "Что будет, если?",
    "Зачем это нужно?",
    "Вы когда-нибудь задумывались?",
    "Знаете ли вы, что?",
    "В чём причина?",
    "Какой смысл?",
    "А что, если я скажу?",
    "Вы это видели?",
]


def _is_question_by_rules(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    first_word = text.lower().split()[0] if text.split() else ""
    return first_word in _QUESTION_WORDS


def _encode(texts, model, tokenizer, device):
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(**encoded_inputs)
    attention_mask = encoded_inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (model_output.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled[:, :_EMB_DIM], p=2, dim=1)


def _compute_is_question(segments, dur, config):
    """Per-second binary is_question: rules first, cosine similarity fallback."""
    valid = collect_valid_segments(segments, dur)

    if not valid:
        return np.zeros(dur, dtype=np.float64)

    rule_results = [_is_question_by_rules(text) for text, _, _ in valid]
    ambiguous_idx = [segment_idx for segment_idx, rule_result in enumerate(rule_results) if not rule_result]

    if ambiguous_idx:

        device = torch.device(config.get("device"))
        tokenizer = AutoTokenizer.from_pretrained(_QUESTION_MODEL_ID)
        model = AutoModel.from_pretrained(_QUESTION_MODEL_ID, torch_dtype=torch.float32).to(device).eval()

        anchor_embs = _encode(_QUESTION_ANCHORS, model, tokenizer, device)
        anchor_mean = torch.nn.functional.normalize(anchor_embs.mean(dim=0, keepdim=True), p=2, dim=1)

        ambiguous_texts = [valid[segment_idx][0] for segment_idx in ambiguous_idx]
        seg_embs = _encode(ambiguous_texts, model, tokenizer, device)
        sims = (seg_embs @ anchor_mean.T).squeeze(1).cpu().numpy()

        for result_idx, segment_idx in enumerate(ambiguous_idx):
            if sims[result_idx] >= _COSINE_THRESHOLD:
                rule_results[segment_idx] = True

        release_models(model, tokenizer, device=device)

    out = np.zeros(dur, dtype=np.float64)
    for (_, start_sec, end_sec), is_question_segment in zip(valid, rule_results, strict=True):
        if is_question_segment:
            out[start_sec:end_sec] = 1.0

    n_questions = sum(rule_results)
    logger.info("is_question: %d/%d segments classified as questions", n_questions, len(valid))
    return out


def extract_hook_score(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "hook_score"):
        return pd.DataFrame()
    segments, duration = get_segments_and_duration(video_path, config)

    address_score, claim_score, number_score = 0.0, 0.0, 0.0
    total_words = 0
    for segment in segments:
        segment_start_sec, text = segment["start"], segment["text"]
        if segment_start_sec < ADDRESS_WINDOW_SEC and RU_ADDRESS.search(text):
            address_score = 1.0
        if segment_start_sec < CLAIM_WINDOW_SEC:
            if RU_CLAIMS.search(text):
                claim_score = 1.0
            if NUMBER_PATTERN.search(text):
                number_score = 0.5
            total_words += len(text.split())

    wps = total_words / min(CLAIM_WINDOW_SEC, duration)

    is_question = _compute_is_question(segments, duration, config)
    q_hook = float(is_question[: int(QUESTION_WINDOW_SEC)].max()) if duration > 0 else 0.0
    hook = q_hook * HOOK_QUESTION_W + address_score * HOOK_ADDRESS_W + claim_score * HOOK_CLAIM_W + number_score * HOOK_NUMBERS_W + wps * HOOK_DENSITY_W

    logger.info(
        "Hook: score=%.2f Q=%.2f A=%.0f C=%.0f N=%.1f WPS=%.2f, is_question frac=%.3f", hook, q_hook, address_score, claim_score, number_score, wps, float(is_question.mean())
    )
    return pd.DataFrame({"hook_score": np.full(duration, hook), "hook_has_address": np.full(duration, address_score), "is_question": is_question})
