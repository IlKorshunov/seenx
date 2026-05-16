"""Curiosity-gap phrase detection (regex fast-path + ensemble fallback).
curiosity_gap — 1.0 if the segment contains a curiosity-gap phrase, else 0.0
"""

import re

import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, logger, skip_if_exists
from ._zeroshot import ZeroShotTask
from .common import collect_valid_segments, score_regex_then_ensemble, spread_scores_over_timeline


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
    valid = collect_valid_segments(segments, duration)

    if not valid:
        return pd.DataFrame({"curiosity_gap": np.zeros(duration)})

    results, regex_hits = score_regex_then_ensemble(valid, RU_CURIOSITY_GAP, 1.0, _TASK, config, ensemble_threshold=_THRESHOLD, threshold_value=1.0)
    out = spread_scores_over_timeline(valid, results, duration)

    n_hits = int((results > 0).sum())
    logger.info("curiosity_gap: %d/%d segments (%d regex, %d ensemble)", n_hits, len(valid), regex_hits, n_hits - regex_hits)
    return pd.DataFrame({"curiosity_gap": out})
