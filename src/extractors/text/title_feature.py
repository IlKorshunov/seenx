"""Title analysis: Qwen3-4B scoring + regex heuristics.

title_clickbait : 0-10, насколько кликбейтный заголовок
title_clarity : 0-10, ясность обещания
title_emotional : 0-10, эмоциональная заряженность
title_specificity : 0-10, конкретность
title_urgency : 0-10, срочность
title_len : количество символов
title_has_number  : 1.0 если есть цифры
title_has_question : 1.0 если есть вопрос
title_caps_ratio : доля слов капсом
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._base import get_segments_and_duration, logger, skip_if_exists
from .common import release_models, video_id


_COLS = {"title_clickbait", "title_clarity", "title_emotional", "title_specificity", "title_urgency", "title_len", "title_has_number", "title_has_question", "title_caps_ratio"}

_LLM_MODEL_ID = "Qwen/Qwen3-4B"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_COMMENTS_ROOT = _PROJECT_ROOT / "get_data" / "comments"

_TITLE_PROMPT = """\
Ты — эксперт по YouTube-заголовкам. Оцени заголовок видео по 5 шкалам от 0 до 10.

Заголовок: "{title}"

Критерии:
1. clickbait (0-10): насколько кликбейтный? 0 = честный, 10 = максимальный кликбейт
2. clarity (0-10): насколько понятно, о чём видео? 0 = непонятно, 10 = кристально ясно
3. emotional (0-10): эмоциональная заряженность? 0 = сухой, 10 = максимально эмоциональный
4. specificity (0-10): конкретность? 0 = абстрактный, 10 = точные цифры/факты/детали
5. urgency (0-10): срочность/FOMO? 0 = нет давления, 10 = "смотри прямо сейчас"

Ответь СТРОГО в формате (только 5 чисел через запятую, ничего больше):
clickbait,clarity,emotional,specificity,urgency"""

_CAPS_RE = re.compile(r"\b[А-ЯЁA-Z]{3,}\b")
_NUMBER_RE = re.compile(r"\d+")
_QUESTION_RE = re.compile(r"\?|^(как|зачем|почему|что|кто|где|когда|сколько)\b", re.IGNORECASE)


def _load_title(video_id_value: str) -> str:
    for comments_path in _COMMENTS_ROOT.rglob(f"{video_id_value}/comments.json"):
        try:
            return json.loads(comments_path.read_text(encoding="utf-8")).get("video_title", "")
        except Exception:
            pass
    return ""


def _regex_features(title: str) -> dict[str, float]:
    words = title.split()
    n_words = max(len(words), 1)
    return {
        "title_len": float(len(title)),
        "title_has_number": 1.0 if _NUMBER_RE.search(title) else 0.0,
        "title_has_question": 1.0 if _QUESTION_RE.search(title) else 0.0,
        "title_caps_ratio": float(len(_CAPS_RE.findall(title)) / n_words),
    }


def _llm_score(title: str) -> dict[str, float]:
    keys = ["title_clickbait", "title_clarity", "title_emotional", "title_specificity", "title_urgency"]
    defaults = {key: 5.0 for key in keys}
    if not title.strip():
        return defaults

    tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_LLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    messages = [{"role": "user", "content": _TITLE_PROMPT.format(title=title)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=1.0)
    generated = out[0][inputs["input_ids"].shape[1] :]
    full = tokenizer.decode(generated, skip_special_tokens=False).strip()
    response = full.split("</think>", 1)[1].strip() if "</think>" in full else tokenizer.decode(generated, skip_special_tokens=True).strip()

    release_models(model, tokenizer, device=None)

    result = dict(defaults)
    for score_idx, score_text in enumerate(re.findall(r"(\d+(?:\.\d+)?)", response)[:5]):
        result[keys[score_idx]] = float(np.clip(float(score_text), 0.0, 10.0))
    return result


def extract_title_features(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "title_features"):
        return pd.DataFrame()

    _, duration = get_segments_and_duration(video_path, config)
    video_id_value = video_id(video_path)
    title = _load_title(video_id_value)

    features: dict[str, float] = {}
    features.update(_regex_features(title))
    features.update(_llm_score(title))

    logger.info(
        "Title [%s]: clickbait=%.0f clarity=%.0f emotional=%.0f spec=%.0f urgency=%.0f",
        video_id_value,
        features["title_clickbait"],
        features["title_clarity"],
        features["title_emotional"],
        features["title_specificity"],
        features["title_urgency"],
    )

    return pd.DataFrame({feature_name: np.full(duration, feature_value) for feature_name, feature_value in features.items()})
