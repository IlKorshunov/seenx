"""LM-based token surprisal (perplexity-related) per second.

Uses a small causal LM from Hugging Face. For each token, surprisal is
-n log p(token | previous tokens) in nats (PyTorch cross-entropy). This differs
from `speech_predictability` (Shannon entropy of word unigrams in a window).

Outputs:
  speech_lm_surprisal     — mean token surprisal (nats) per second
  speech_lm_surprisal_vel — |Δ surprisal| between consecutive seconds (rate of change)

Long transcripts: sliding windows over token ids with overlap; per-token values
are averaged across windows that cover that position (approximate for boundaries).

Config (optional, init.json):
  lm_surprisal_model_id — default "distilgpt2"; for Russian-heavy text try
                          "ai-forever/rugpt3small_based_on_gpt2" or "facebook/mgpt"
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ._base import get_segments_and_duration, logger, skip_if_exists


_COLS = {"speech_lm_surprisal", "speech_lm_surprisal_vel"}

_DEFAULT_MODEL = "distilgpt2"
_DEFAULT_MAX_LEN = 1024
_DEFAULT_STRIDE = 512

_model_cache: dict[str, object] = {}
_tok_cache: dict[str, object] = {}


def _get_lm(model_id: str, device: torch.device):
    if model_id in _model_cache:
        return _model_cache[model_id], _tok_cache[model_id]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    _model_cache[model_id] = model
    _tok_cache[model_id] = tok
    return model, tok


def _build_text_and_char_sec(segments: list[dict], n_seconds: int) -> tuple[str, np.ndarray]:
    """Full transcript string + per-character second index (0 .. n_seconds-1)."""
    chars: list[str] = []
    sec_per_char: list[int] = []

    for segment in segments:
        words = segment.get("words") or []
        for word in words:
            raw = (word.get("word") or word.get("text") or "").strip()
            if not raw:
                continue
            start_sec = float(word.get("start", 0.0))
            second_idx = min(n_seconds - 1, max(0, int(math.floor(start_sec))))
            if chars:
                chars.append(" ")
                sec_per_char.append(second_idx)
            for char in raw:
                chars.append(char)
                sec_per_char.append(second_idx)

    if not chars:
        parts: list[str] = []
        sec_per_char = []
        for segment in segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            start = float(segment.get("start", 0.0))
            second_idx = min(n_seconds - 1, max(0, int(math.floor(start))))
            if parts:
                parts.append(" ")
                sec_per_char.append(second_idx)
            for char in text:
                parts.append(char)
                sec_per_char.append(second_idx)
        chars = parts

    if not chars:
        return "", np.zeros(0, dtype=np.int64)
    return "".join(chars), np.array(sec_per_char, dtype=np.int64)


def _one_pass_surprisal(input_ids: torch.Tensor, model) -> torch.Tensor:
    """Per-token surprisal; index 0 is 0 (no left context)."""
    with torch.no_grad():
        logits = model(input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    cross_entropy = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").view(input_ids.shape[0], -1)
    length = input_ids.shape[1]
    surprisal = torch.zeros(length, device=input_ids.device, dtype=cross_entropy.dtype)
    surprisal[1:] = cross_entropy[0]
    return surprisal


def _sliding_surprisal(input_ids: torch.Tensor, model, max_len: int, stride: int) -> torch.Tensor:
    length = input_ids.shape[1]
    device = input_ids.device
    if length <= max_len:
        return _one_pass_surprisal(input_ids, model)

    total = torch.zeros(length, device=device)
    counts = torch.zeros(length, device=device)
    start = 0
    while start < length:
        end = min(length, start + max_len)
        if end - start < 2:
            break
        chunk = input_ids[:, start:end]
        chunk_surprisal = _one_pass_surprisal(chunk, model)
        total[start:end] += chunk_surprisal
        counts[start:end] += 1.0
        if end >= length:
            break
        start += stride
    return total / counts.clamp(min=1.0)


def extract_speech_lm_surprisal(video_path: str, config=None, existing_features: set[str] | None = None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "speech_lm_surprisal"):
        return pd.DataFrame()

    model_id = _DEFAULT_MODEL
    if config is not None:
        model_id = config.get("lm_surprisal_model_id", _DEFAULT_MODEL)

    segments, duration = get_segments_and_duration(video_path, config)
    n_seconds = max(1, int(np.ceil(duration)))

    out_surp = np.zeros(n_seconds, dtype=np.float64)
    out_vel = np.zeros(n_seconds, dtype=np.float64)

    full_text, char_sec = _build_text_and_char_sec(segments, n_seconds)
    if not full_text.strip():
        return pd.DataFrame({"speech_lm_surprisal": out_surp, "speech_lm_surprisal_vel": out_vel})

    device = torch.device(config.get("device") if config else "cpu")
    model, tokenizer = _get_lm(model_id, device)

    max_model_len = min(_DEFAULT_MAX_LEN, getattr(tokenizer, "model_max_length", _DEFAULT_MAX_LEN) or _DEFAULT_MAX_LEN)
    if max_model_len > 1024:
        max_model_len = 1024
    stride = max(_DEFAULT_STRIDE, max_model_len // 2)

    encoded_inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False, truncation=False, return_offsets_mapping=tokenizer.is_fast)
    input_ids = encoded_inputs["input_ids"].to(device)
    offset_map = encoded_inputs.get("offset_mapping")
    if offset_map is not None:
        offset_map = offset_map[0]

    if input_ids.shape[1] == 0:
        return pd.DataFrame({"speech_lm_surprisal": out_surp, "speech_lm_surprisal_vel": out_vel})

    token_surp = _sliding_surprisal(input_ids, model, max_model_len, stride).detach().cpu().numpy()

    token_count = len(token_surp)
    sec_sum = np.zeros(n_seconds, dtype=np.float64)
    sec_cnt = np.zeros(n_seconds, dtype=np.float64)

    if offset_map is not None and len(offset_map) == token_count:
        for token_idx, (char_start, char_end) in enumerate(offset_map):
            if char_end <= char_start:
                continue
            token_surprisal = float(token_surp[token_idx])
            char_midpoint = (char_start + char_end - 1) // 2
            if char_midpoint >= len(char_sec):
                char_midpoint = len(char_sec) - 1
            if char_midpoint < 0:
                continue
            second_idx = int(char_sec[char_midpoint])
            sec_sum[second_idx] += token_surprisal
            sec_cnt[second_idx] += 1.0
    else:
        for token_idx in range(token_count):
            token_fraction = (token_idx + 0.5) / max(token_count, 1)
            second_idx = min(n_seconds - 1, int(token_fraction * n_seconds))
            sec_sum[second_idx] += float(token_surp[token_idx])
            sec_cnt[second_idx] += 1.0

    mask = sec_cnt > 0
    out_surp[mask] = sec_sum[mask] / sec_cnt[mask]
    unfilled = sec_cnt <= 0
    if unfilled.any():
        global_mean = float(np.mean(token_surp[1:])) if token_count > 1 else 0.0
        out_surp[unfilled] = global_mean

    out_vel[0] = 0.0
    out_vel[1:] = np.abs(np.diff(out_surp))

    logger.info("LM surprisal (%s): mean=%.3f nats, vel_mean=%.3f", model_id, float(np.mean(out_surp)), float(np.mean(out_vel[1:])))

    return pd.DataFrame({"speech_lm_surprisal": out_surp.astype(np.float32), "speech_lm_surprisal_vel": out_vel.astype(np.float32)})
