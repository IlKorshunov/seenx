"""Video section detection: intro / outro via LLM boundary detection.

Primary: Qwen3-4B reads transcript context and returns the boundary segment.
Fallback: GeRaCl + NLI per-segment classification (if LLM fails or unavailable).

is_intro — score [0, 1]: how much this second looks like an intro
is_outro — score [0, 1]: how much this second looks like an outro
"""

import os
import re

import numpy as np
import pandas as pd
import torch

from ._base import get_segments_and_duration, logger, skip_if_exists
from ._zeroshot import ZeroShotTask, classify_segments
from .common import collect_valid_segments_with_mid, release_models

from transformers import AutoModelForCausalLM, AutoTokenizer


_COLS = {"is_intro", "is_outro"}

_LLM_MODEL_ID = "Qwen/Qwen3-4B"
_LLM_MAX_SEGMENTS = 40
_LLM_MAX_NEW_TOKENS = 2048

_INTRO_TASK = ZeroShotTask(
    geracl_labels=["автор представляет тему видео и рассказывает о чём пойдёт речь", "автор уже рассказывает основной контент или завершает видео"],
    nli_hypothesis="Автор представляет тему ролика и объясняет, о чём пойдёт речь дальше",
)

_OUTRO_TASK = ZeroShotTask(
    geracl_labels=["автор подводит итоги, прощается или призывает подписаться", "автор рассказывает основной контент или только начинает видео"],
    nli_hypothesis="Автор подводит итоги, завершает видео или призывает подписаться и поставить лайк",
)

RU_INTRO = re.compile(
    r"(сегодня (поговорим|расскажу|разберём|обсудим|узнаем)"
    r"|в этом (видео|ролике|выпуске)"
    r"|тема (нашего|сегодняшнего|этого)"
    r"|план (на сегодня|видео|ролика)"
    r"|о чём (пойдёт речь|будем говорить|этот ролик)"
    r"|начн[её]м с того"
    r"|приветствую|здравствуйте|всем привет"
    r"|добро пожаловать"
    r"|меня зовут\b"
    r"|я\s.{0,20}\s(канал|блог)\b"
    r"|на (моём|нашем) канале)",
    re.IGNORECASE,
)

RU_OUTRO = re.compile(
    r"(подведём итоги|в заключение|итого\b|резюмируя"
    r"|подпишись|подпишитесь|ставь(те)? лайк"
    r"|до (встречи|свидания|новых|скорого)"
    r"|в следующем (видео|ролике|выпуске)"
    r"|на этом (всё|у меня всё|пока всё)"
    r"|спасибо (за просмотр|что смотрели|что досмотрели)"
    r"|пишите в комментари"
    r"|нажми(те)? на (колокольчик|кнопку)"
    r"|поддержи(те)? канал"
    r"|если (понравилось|было полезно)"
    r"|всем пока\b|пока-пока\b)",
    re.IGNORECASE,
)

_INTRO_BOOST_SEC = 30.0
_INTRO_BOOST_WEIGHT = 0.25
_OUTRO_BOOST_SEC = 30.0
_OUTRO_BOOST_WEIGHT = 0.25

_INTRO_PROMPT = """\
Ты анализируешь транскрипцию начала YouTube-видео.
Определи номер последнего сегмента, который относится к вступлению (интро).

Вступление — это когда автор:
- Приветствует зрителей
- Представляется или упоминает название канала
- Объявляет тему видео
- Делает анонс содержания

Как только автор переходит к основному контенту (рассказ, факты, история) — вступление закончилось.
Рекламная интеграция — это НЕ вступление.

Сегменты:
{segments}

Ответь ОДНИМ числом — номер последнего сегмента вступления.
Если вступления нет, ответь 0."""

_OUTRO_PROMPT = """\
Ты анализируешь транскрипцию конца YouTube-видео.
Определи номер первого сегмента, с которого начинается заключение (аутро).

Заключение — это когда автор:
- Подводит итоги
- Прощается со зрителями
- Призывает подписаться, поставить лайк
- Анонсирует следующие видео

Пока автор рассказывает основной контент — это ещё не заключение.

Сегменты:
{segments}

Ответь ОДНИМ числом — номер первого сегмента заключения.
Если заключения нет, ответь 0."""


def _load_llm():
    tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_LLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tokenizer


def _unload_llm(model, tokenizer):
    release_models(model, tokenizer, device=None)

def _format_segments(segments: list[tuple], start_idx: int = 1) -> str:
    lines = []
    for segment_number, (text, start_sec, end_sec, _) in enumerate(segments, start=start_idx):
        lines.append(f'[{segment_number}] ({start_sec}-{end_sec}с) "{text}"')
    return os.linesep.join(lines)


def _parse_llm_response(text: str, max_val: int) -> int | None:
    match = re.search(r"\d+", text)
    if match is None:
        return None
    value = int(match.group())
    if value < 0 or value > max_val:
        return None
    return value


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


def _run_boundary_prompt(model, tokenizer, valid, prompt_template, edge) -> int | None:
    """Run a single LLM prompt for intro/outro boundary detection.
    Returns timestamp (seconds), 0 (no boundary), or None (failure).
    """
    if not valid:
        return None

    if edge == "start":
        block = valid[:_LLM_MAX_SEGMENTS]
        seg_text = _format_segments(block, start_idx=1)
        max_answer = len(block)
    else:
        block = valid[-_LLM_MAX_SEGMENTS:]
        offset = max(0, len(valid) - _LLM_MAX_SEGMENTS)
        seg_text = _format_segments(block, start_idx=offset + 1)
        max_answer = len(valid)

    prompt = prompt_template.format(segments=seg_text)
    response = _llm_generate(model, tokenizer, prompt)
    logger.info("LLM section response (%s): '%s'", edge, response)

    raw_num = _parse_llm_response(response, max_answer)
    if raw_num is None:
        return None
    if raw_num == 0:
        return 0

    segment_idx = raw_num - 1
    if segment_idx < 0 or segment_idx >= len(valid):
        return None

    if edge == "start":
        _, _, boundary_sec, _ = valid[segment_idx]  
    else:
        _, boundary_sec, _, _ = valid[segment_idx]
    return boundary_sec

_NLI_SEARCH_FRAC = 0.30
_NLI_GAP_SEC = 15


def _detect_boundary_nli(valid: list[tuple], duration: int, edge: str, config) -> int | None:
    """Fallback: per-segment GeRaCl+NLI classification."""
    if not valid:
        return None

    if edge == "start":
        task = _INTRO_TASK
        candidates = [(segment_idx, segment) for segment_idx, segment in enumerate(valid) if segment[3] <= _NLI_SEARCH_FRAC]
    else:
        task = _OUTRO_TASK
        candidates = [(segment_idx, segment) for segment_idx, segment in enumerate(valid) if segment[3] >= 1.0 - _NLI_SEARCH_FRAC]

    if not candidates:
        return 0

    scores = np.zeros(len(valid), dtype=np.float64)

    regex = RU_INTRO if edge == "start" else RU_OUTRO
    classify_idx = []
    for segment_idx, (text, _, _, _) in candidates:
        if regex.search(text):
            scores[segment_idx] = 0.90
        else:
            classify_idx.append(segment_idx)

    if classify_idx:
        texts = [valid[segment_idx][0] for segment_idx in classify_idx]
        raw_scores = classify_segments(texts, task, config)
        for result_idx, segment_idx in enumerate(classify_idx):
            scores[segment_idx] = raw_scores[result_idx]

    per_second = np.zeros(duration, dtype=np.float64)
    for segment_idx, (_, start_sec, end_sec, _) in enumerate(valid):
        if scores[segment_idx] > 0:
            per_second[start_sec:end_sec] = scores[segment_idx]

    if edge == "start":
        last_good = -1
        gap = 0
        for second_idx in range(duration):
            if per_second[second_idx] >= 0.3:
                last_good = second_idx
                gap = 0
            else:
                gap += 1
                if gap > _NLI_GAP_SEC and last_good >= 0:
                    break
        return last_good if last_good >= 0 else 0
    else:
        first_good = duration
        gap = 0
        for second_idx in range(duration - 1, -1, -1):
            if per_second[second_idx] >= 0.3:
                first_good = second_idx
                gap = 0
            else:
                gap += 1
                if gap > _NLI_GAP_SEC and first_good < duration:
                    break
        return first_good if first_good < duration else duration


def _build_linear_curve(duration: int, start_sec: int, end_sec: int, start_score: float, end_score: float) -> np.ndarray:
    scores = np.zeros(duration, dtype=np.float64)
    start_sec = max(0, min(duration, start_sec))
    end_sec = max(0, min(duration, end_sec))
    if start_sec >= end_sec:
        return scores

    span = max(end_sec - start_sec, 1)
    for second_idx in range(start_sec, end_sec):
        fraction = (second_idx - start_sec) / span
        scores[second_idx] = start_score + (end_score - start_score) * fraction
    return scores


def _build_intro_curve(duration: int, boundary_sec: int) -> np.ndarray:
    """Smooth intro score: high near t=0, decays to 0.5 at boundary."""
    return _build_linear_curve(duration, 0, boundary_sec, 0.90, 0.50)


def _build_outro_curve(duration: int, boundary_sec: int) -> np.ndarray:
    """Smooth outro score: grows from 0.5 at boundary to 0.9 at end."""
    return _build_linear_curve(duration, boundary_sec, duration, 0.50, 0.90)


def _apply_regex_boost(scores: np.ndarray, valid: list[tuple], regex, edge: str) -> np.ndarray:
    """Boost score in regions where regex matches."""
    for text, start_sec, end_sec, _ in valid:
        if regex.search(text):
            if (edge == "start" and scores[start_sec:end_sec].any()) or (edge == "end" and scores[start_sec:end_sec].any()):
                scores[start_sec:end_sec] = np.maximum(scores[start_sec:end_sec], 0.85)
    return scores


def _apply_position_boost(scores: np.ndarray, duration: int, edge: str) -> np.ndarray:
    """Small additive boost near the video edge (first/last 30s)."""
    if edge == "start":
        for second_idx in range(min(duration, int(_INTRO_BOOST_SEC))):
            boost = _INTRO_BOOST_WEIGHT * (1.0 - second_idx / _INTRO_BOOST_SEC)
            scores[second_idx] = min(1.0, scores[second_idx] + boost)
    else:
        for second_idx in range(max(0, duration - int(_OUTRO_BOOST_SEC)), duration):
            distance_to_end = duration - 1 - second_idx
            boost = _OUTRO_BOOST_WEIGHT * (1.0 - distance_to_end / _OUTRO_BOOST_SEC)
            scores[second_idx] = min(1.0, scores[second_idx] + boost)
    return scores


def extract_sections(video_path: str, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "sections"):
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)
    valid = collect_valid_segments_with_mid(segments, duration)

    if not valid:
        return pd.DataFrame({"is_intro": np.zeros(duration), "is_outro": np.zeros(duration)})

    W_LLM = 0.8
    W_NLI = 0.2

    intro_llm = None
    outro_llm = None
    model, tokenizer = _load_llm()
    intro_llm = _run_boundary_prompt(model, tokenizer, valid, _INTRO_PROMPT, edge="start")
    outro_llm = _run_boundary_prompt(model, tokenizer, valid, _OUTRO_PROMPT, edge="end")
    _unload_llm(model, tokenizer)

    intro_nli = _detect_boundary_nli(valid, duration, edge="start", config=config)
    outro_nli = _detect_boundary_nli(valid, duration, edge="end", config=config)

    def _weighted_boundary(llm_val, nli_val, default):
        if llm_val is not None and nli_val is not None:
            return int(W_LLM * llm_val + W_NLI * nli_val)
        if llm_val is not None:
            return llm_val
        if nli_val is not None:
            return nli_val
        return default

    intro_end = _weighted_boundary(intro_llm, intro_nli, 0)
    outro_start = _weighted_boundary(outro_llm, outro_nli, duration)

    logger.info("Section boundaries: intro_end=%ds (llm=%s, nli=%s), outro_start=%ds (llm=%s, nli=%s)", intro_end, intro_llm, intro_nli, outro_start, outro_llm, outro_nli)

    is_intro = _build_intro_curve(duration, intro_end)
    is_outro = _build_outro_curve(duration, outro_start)

    is_intro = _apply_regex_boost(is_intro, valid, RU_INTRO, edge="start")
    is_outro = _apply_regex_boost(is_outro, valid, RU_OUTRO, edge="end")
    is_intro = _apply_position_boost(is_intro, duration, edge="start")
    is_outro = _apply_position_boost(is_outro, duration, edge="end")

    intro_secs = int((is_intro > 0.3).sum())
    outro_secs = int((is_outro > 0.3).sum())
    logger.info("sections: intro=%ds (end=%ds), outro=%ds (start=%ds), main=%ds", intro_secs, intro_end, outro_secs, outro_start, duration - intro_secs - outro_secs)

    return pd.DataFrame({"is_intro": is_intro, "is_outro": is_outro})
