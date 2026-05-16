"""Audience friction detection: Qwen3-4B scores problematic segments.
friction_jargon      : 0-1, непонятные термины без объяснения
friction_repetition  : 0-1, повтор уже сказанного
friction_abstract    : 0-1, абстрактные рассуждения без конкретики
friction_digression  : 0-1, уход от заявленной темы
friction_total       : 0-1, агрегированная метрика трения
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._base import get_segments_and_duration, logger, seg_bounds, skip_if_exists
from .common import collect_valid_segment_dicts, release_models


_COLS = {"friction_jargon", "friction_repetition", "friction_abstract", "friction_digression", "friction_total"}

_LLM_MODEL_ID = "Qwen/Qwen3-4B"
_WINDOW_SEGMENTS = 15
_WINDOW_OVERLAP = 5

_FRICTION_PROMPT = """\
Ты анализируешь фрагмент транскрипции YouTube-видео на предмет "точек трения" — \
моментов, где зритель может потерять интерес и уйти.

Фрагмент транскрипта:
{segments}

Оцени КАЖДЫЙ сегмент по 4 критериям от 0.0 до 1.0:
1. jargon: непонятные термины, аббревиатуры без объяснения (0 = всё понятно, 1 = сплошной жаргон)
2. repetition: повтор того, что уже было сказано ранее (0 = новая информация, 1 = полное повторение)
3. abstract: абстрактные рассуждения без примеров и конкретики (0 = конкретно, 1 = чистая абстракция)
4. digression: уход от основной темы видео (0 = по теме, 1 = совсем не по теме)

Ответь СТРОГО в формате, по одной строке на сегмент:
[номер] jargon,repetition,abstract,digression

Пример:
[1] 0.1,0.0,0.3,0.0
[2] 0.8,0.2,0.5,0.1"""


def _format_segments(segments: list[dict], start_idx: int) -> str:
    lines = []
    for segment_number, segment in enumerate(segments, start=start_idx):
        text = segment.get("text", "").strip()[:200]
        start_sec, end_sec = segment["start"], segment["end"]
        lines.append(f'[{segment_number}] ({start_sec:.0f}-{end_sec:.0f}с) "{text}"')
    return  os.linesep.join(lines)


def _load_llm():
    tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_LLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tokenizer


def _llm_generate(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=1.0)
    generated = out[0][inputs["input_ids"].shape[1] :]
    full = tokenizer.decode(generated, skip_special_tokens=False).strip()
    if "</think>" in full:
        return full.split("</think>", 1)[1].strip()
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _parse_scores(response: str) -> dict[int, tuple[float, float, float, float]]:
    result: dict[int, tuple[float, float, float, float]] = {}
    for line in response.strip().splitlines():
        match = re.match(r"\[(\d+)\]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", line)
        if match:
            segment_idx = int(match.group(1))
            scores = tuple(float(np.clip(float(match.group(score_group_idx)), 0.0, 1.0)) for score_group_idx in range(2, 6))
            result[segment_idx] = scores  # type: ignore[assignment]
    return result


def extract_friction(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "friction"):
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)
    valid = collect_valid_segment_dicts(segments)

    jargon = np.zeros(duration, dtype=np.float64)
    repetition = np.zeros(duration, dtype=np.float64)
    abstract = np.zeros(duration, dtype=np.float64)
    digression = np.zeros(duration, dtype=np.float64)

    if not valid:
        return pd.DataFrame(
            {"friction_jargon": jargon, "friction_repetition": repetition, "friction_abstract": abstract, "friction_digression": digression, "friction_total": np.zeros(duration)}
        )

    model, tokenizer = _load_llm()

    for window_start in range(0, len(valid), _WINDOW_SEGMENTS - _WINDOW_OVERLAP):
        window_end = min(window_start + _WINDOW_SEGMENTS, len(valid))
        window = valid[window_start:window_end]
        if not window:
            break

        prompt = _FRICTION_PROMPT.format(segments=_format_segments(window, start_idx=window_start + 1))
        try:
            response = _llm_generate(model, tokenizer, prompt)
        except Exception as error:
            logger.warning("Friction window [%d-%d] failed: %s", window_start + 1, window_end, error)
            continue

        scores = _parse_scores(response)

        for local_idx, seg in enumerate(window):
            segment_idx = window_start + local_idx + 1
            if segment_idx not in scores:
                continue
            jargon_score, repetition_score, abstract_score, digression_score = scores[segment_idx]
            start_sec, end_sec = seg_bounds(seg, duration)
            jargon[start_sec:end_sec] = np.maximum(jargon[start_sec:end_sec], jargon_score)
            repetition[start_sec:end_sec] = np.maximum(repetition[start_sec:end_sec], repetition_score)
            abstract[start_sec:end_sec] = np.maximum(abstract[start_sec:end_sec], abstract_score)
            digression[start_sec:end_sec] = np.maximum(digression[start_sec:end_sec], digression_score)

        if window_end >= len(valid):
            break

    release_models(model, tokenizer, device=None)

    total = (jargon + repetition + abstract + digression) / 4.0

    logger.info("Friction: jargon=%.2f rep=%.2f abstract=%.2f digress=%.2f (means)", jargon.mean(), repetition.mean(), abstract.mean(), digression.mean())

    return pd.DataFrame({"friction_jargon": jargon, "friction_repetition": repetition, "friction_abstract": abstract, "friction_digression": digression, "friction_total": total})
