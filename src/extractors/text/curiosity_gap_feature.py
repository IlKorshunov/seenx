"""Curiosity-gap phrase detection (regex fast-path + ensemble fallback).

Regex catches obvious patterns instantly.  For remaining segments,
GeRaCl + rubert-NLI ensemble decides via weighted zero-shot scoring.

Output (per-second, 1 Hz):
  curiosity_gap — 1.0 if the segment contains a curiosity-gap phrase, else 0.0
"""

import re

import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, logger, seg_bounds, skip_if_exists
from ._zeroshot import ZeroShotTask, classify_segments


_COLS = {"curiosity_gap"}

_TASK = ZeroShotTask(
    geracl_labels=["автор прямо говорит зрителю подождать или обещает что-то дальше", "автор рассказывает факты и историю без обещаний зрителю"],
    nli_hypothesis="Автор создаёт интригу и просит зрителя подождать или обещает что-то впереди",
)

_THRESHOLD = 0.5

RU_CURIOSITY_GAP = re.compile(
    r"(самое интересное|самое главное|самое важное"
    r"|ты не поверишь|вы не поверите"
    r"|не переключайтесь|не уходите|досмотри"
    r"|сейчас (покажу|узнаете|расскажу|будет)"
    r"|а (дальше|теперь|вот теперь)[\s,.!]*(самое|внимание|начинается)"
    r"|подождите|погодите|стоп"
    r"|внимание[\s!,]|а вот (тут|здесь)"
    r"|но (это|всё|все) ещ[её] не вс[ёе]"
    r"|и вот что (случилось|произошло|было)"
    r"|через (минуту|секунду|пару минут)"
    r"|скоро (узнаете|увидите|поймёте)"
    r"|главный (секрет|вопрос|момент)"
    r"|угадайте|как (думаете|считаете)"
    r"|а знаете (что|ли)|хотите знать"
    r"|обязательно (дождитесь|смотрите до конца))",
    re.IGNORECASE,
)


def extract_curiosity_gap(video_path: str, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "curiosity_gap"):
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)
    valid = []
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        start_sec, end_sec = seg_bounds(segment, duration)
        if start_sec < end_sec:
            valid.append((text, start_sec, end_sec))

    if not valid:
        return pd.DataFrame({"curiosity_gap": np.zeros(duration)})

    results = np.zeros(len(valid), dtype=np.float64)
    regex_hits = 0
    ambiguous_idx = []
    for segment_idx, (text, _, _) in enumerate(valid):
        if RU_CURIOSITY_GAP.search(text):
            results[segment_idx] = 1.0
            regex_hits += 1
        else:
            ambiguous_idx.append(segment_idx)

    if ambiguous_idx:
        ambiguous_texts = [valid[segment_idx][0] for segment_idx in ambiguous_idx]
        scores = classify_segments(ambiguous_texts, _TASK, config)
        for result_idx, segment_idx in enumerate(ambiguous_idx):
            results[segment_idx] = 1.0 if scores[result_idx] >= _THRESHOLD else 0.0

    out = np.zeros(duration, dtype=np.float64)
    for segment_idx, (_, start_sec, end_sec) in enumerate(valid):
        out[start_sec:end_sec] = results[segment_idx]

    n_hits = int((results > 0).sum())
    logger.info("curiosity_gap: %d/%d segments (%d regex, %d ensemble)", n_hits, len(valid), regex_hits, n_hits - regex_hits)
    return pd.DataFrame({"curiosity_gap": out})
