"""Example / illustration detection (regex fast-path + ensemble).

Regex catches clear markers; GeRaCl + rubert-NLI ensemble classifies the rest.

Output (per-second, 1 Hz):
  has_example — score [0, 1]: 0 = no example, 0.85 = regex hit, ensemble score if >= 0.60
"""

import re

import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, logger, seg_bounds, skip_if_exists
from ._zeroshot import ZeroShotTask, classify_segments


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
    valid = []
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        start_sec, end_sec = seg_bounds(segment, duration)
        if start_sec < end_sec:
            valid.append((text, start_sec, end_sec))

    if not valid:
        return pd.DataFrame({"has_example": np.zeros(duration)})

    results = np.zeros(len(valid), dtype=np.float64)
    regex_hits = 0
    ambiguous_idx = []
    for segment_idx, (text, _, _) in enumerate(valid):
        if RU_EXAMPLE.search(text):
            results[segment_idx] = 0.9
            regex_hits += 1
        else:
            ambiguous_idx.append(segment_idx)

    if ambiguous_idx:
        ambiguous_texts = [valid[segment_idx][0] for segment_idx in ambiguous_idx]
        scores = classify_segments(ambiguous_texts, _TASK, config)
        for result_idx, segment_idx in enumerate(ambiguous_idx):
            results[segment_idx] = scores[result_idx] if scores[result_idx] >= _THRESHOLD else 0.0

    out = np.zeros(duration, dtype=np.float64)
    for segment_idx, (_, start_sec, end_sec) in enumerate(valid):
        out[start_sec:end_sec] = results[segment_idx]

    n_hits = int((results > 0).sum())
    logger.info("has_example: %d/%d segments (%d regex, %d ensemble)", n_hits, len(valid), regex_hits, n_hits - regex_hits)
    return pd.DataFrame({"has_example": out})
