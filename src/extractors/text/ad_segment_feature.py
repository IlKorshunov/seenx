import math
import os
import re

import numpy as np
import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ._base import get_segments_and_duration, logger, skip_if_exists
from .common import collect_valid_segment_dicts, release_models
from .constants import RU_AD_CTA_PATTERNS, RU_AD_PATTERNS


_COLS = {"is_ad", "ad_segment_length"}
_CNT_GAP_SEG = 3
_LLM_MODEL_ID = "Qwen/Qwen3-4B"
_WINDOW_SIZE = 50
_WINDOW_OVERLAP = 10
_LLM_MAX_NEW_TOKENS = 4096

_AD_PROMPT = """\
Ты анализируешь фрагмент транскрипции YouTube-видео.
Найди все рекламные интеграции в этом фрагменте.

Рекламная интеграция — это когда автор:
- Рекомендует конкретный продукт, игру, приложение, сервис
- Описывает функции или преимущества продукта
- Упоминает промокод, ссылку в описании, скидку

НЕ является рекламой:
- Обсуждение продуктов как часть основной темы видео
- Рекомендации книг/фильмов/музыки по теме
- Призыв подписаться на канал
- Упоминание своих других каналов

Сегменты:
{segments}

Если есть реклама, ответь: START-END (номера первого и последнего рекламного сегмента).
Если несколько блоков: START1-END1, START2-END2
Если рекламы в этом фрагменте нет, ответь: 0"""


def _format_segments(segments: list[dict], start_idx: int = 1) -> str:
    lines = []
    for segment_number, segment in enumerate(segments, start=start_idx):
        text = segment.get("text", "").strip()[:120]
        start_sec, end_sec = segment["start"], segment["end"]
        lines.append(f'[{segment_number}] ({start_sec:.0f}-{end_sec:.0f}с) "{text}"')
    return os.linesep.join(lines)


def _load_llm():
    tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_LLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tokenizer


def _unload_llm(model, tokenizer):
    release_models(model, tokenizer, device=None)


def _llm_generate(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=_LLM_MAX_NEW_TOKENS, do_sample=False, temperature=1.0)
    generated = out[0][inputs["input_ids"].shape[1] :]
    full = tokenizer.decode(generated, skip_special_tokens=False).strip()
    if "</think>" in full:
        return full.split("</think>", 1)[1].strip()
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _detect_ads_llm(segments: list[dict]) -> list[tuple[int, int]]:
    """Scan transcript in windows, return list of (start_seg_idx, end_seg_idx)"""
    if not segments:
        return []
    model, tokenizer = _load_llm()
    all_ranges: list[tuple[int, int]] = []
    for window_start in range(0, len(segments), _WINDOW_SIZE - _WINDOW_OVERLAP):
        window_end = min(window_start + _WINDOW_SIZE, len(segments))
        window = segments[window_start:window_end]
        response = _llm_generate(model, tokenizer, _AD_PROMPT.format(segments=_format_segments(window, start_idx=window_start + 1)))
        ranges = re.findall(r"(\d+)\s*[-–]\s*(\d+)", re.sub(r"<\|[^>]+\|>", "", response).strip())
        for s_str, e_str in ranges:
            start_idx, end_idx = int(s_str) - 1, int(e_str) - 1
            if start_idx <= end_idx:
                all_ranges.append((start_idx, end_idx))
        logger.info("LLM ad window [%d-%d]: %d ranges found", window_start + 1, window_end, len(ranges))
        if window_end >= len(segments):
            break
    _unload_llm(model, tokenizer)

    if not all_ranges:
        return []
    all_ranges.sort()
    merged = [list(all_ranges[0])]
    for start_idx, end_idx in all_ranges[1:]:
        if start_idx <= merged[-1][1] + _CNT_GAP_SEG:
            merged[-1][1] = max(merged[-1][1], end_idx)
        else:
            merged.append([start_idx, end_idx])
    return [(start_idx, end_idx) for start_idx, end_idx in merged]


def _detect_ads_regex(segments: list[dict]) -> list[tuple[int, int]]:
    def _hit(text: str) -> bool:
        return bool(RU_AD_PATTERNS.search(text) or RU_AD_CTA_PATTERNS.search(text))

    hits = [segment_idx for segment_idx, segment in enumerate(segments) if _hit(segment.get("text", ""))]
    if not hits:
        return []
    merged = [[hits[0], hits[0]]]
    for hit_idx in hits[1:]:
        if hit_idx - merged[-1][1] <= 5:
            merged[-1][1] = hit_idx
        else:
            merged.append([hit_idx, hit_idx])
    return [(start_idx, end_idx) for start_idx, end_idx in merged]


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for start_idx, end_idx in ranges[1:]:
        if start_idx <= merged[-1][1] + _CNT_GAP_SEG:
            merged[-1][1] = max(merged[-1][1], end_idx)
        else:
            merged.append([start_idx, end_idx])
    return [(start_idx, end_idx) for start_idx, end_idx in merged]


def extract_ad_segments(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "ad_segments"):
        return pd.DataFrame()
    segments, duration = get_segments_and_duration(video_path, config)
    valid = collect_valid_segment_dicts(segments)
    llm_ranges = _detect_ads_llm(valid)
    regex_ranges = _detect_ads_regex(valid)
    all_ranges = _merge_ranges(llm_ranges + regex_ranges)
    logger.info("Ad detection: %d LLM ranges, %d regex ranges -> %d merged", len(llm_ranges), len(regex_ranges), len(all_ranges))
    is_ad = np.zeros(duration, dtype=np.float64)
    ad_len = np.zeros(duration, dtype=np.float64)

    ad_intervals = []
    for start_segment_idx, end_segment_idx in all_ranges:
        start_sec = max(0, math.floor(valid[start_segment_idx]["start"]))
        end_sec = min(duration, math.ceil(valid[end_segment_idx]["end"]))
        ad_intervals.append((start_sec, end_sec))
        is_ad[start_sec:end_sec] = 1.0
        ad_len[start_sec:end_sec] = end_sec - start_sec

    if ad_intervals:
        percent = float(is_ad.sum()) / max(duration, 1) * 100.0
        ad_times = ", ".join(f"{start_sec}-{end_sec}s" for start_sec, end_sec in ad_intervals)
        logger.info("Ad: %d segments, %.0fs (%.1f%%): %s", len(ad_intervals), is_ad.sum(), percent, ad_times)
    else:
        logger.info("Ad: none detected")
    return pd.DataFrame({"is_ad": is_ad, "ad_segment_length": ad_len})
