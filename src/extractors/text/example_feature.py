"""Example / illustration detection (regex fast-path + ensemble)."""

import re

import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, logger, skip_if_exists
from ._zeroshot import ZeroShotTask
from .common import collect_valid_segments, score_regex_then_ensemble, spread_scores_over_timeline


_COLS = {"has_example"}

_TASK = ZeroShotTask(
    geracl_labels=["автор приводит конкретный пример, сравнение или аналогию для объяснения", "автор описывает события или факты без примеров и аналогий"],
    nli_hypothesis="Автор приводит конкретный пример, сравнение или аналогию чтобы объяснить идею",
)

_THRESHOLD = 0.60

RU_EXAMPLE = re.compile(
    r"(например\b|к примеру\b"
    r"|сравним\b"
    r"|смотрите\b"
    r"|допустим\b"
    r"|на примере"
    r"|пример"
    r"|для наглядности"
    r"|по аналогии)",
    re.IGNORECASE,
)


def extract_examples(video_path: str, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "has_example"):
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)
    valid = collect_valid_segments(segments, duration)

    if not valid:
        return pd.DataFrame({"has_example": np.zeros(duration)})

    results, regex_hits = score_regex_then_ensemble(valid, RU_EXAMPLE, 0.9, _TASK, config, ensemble_threshold=_THRESHOLD)
    out = spread_scores_over_timeline(valid, results, duration)

    n_hits = int((results > 0).sum())
    logger.info("has_example: %d/%d segments (%d regex, %d ensemble)", n_hits, len(valid), regex_hits, n_hits - regex_hits)
    return pd.DataFrame({"has_example": out})
