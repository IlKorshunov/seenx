import numpy as np

from ._base import seg_bounds
from ._zeroshot import classify_segments


def collect_valid_segments(segments: list[dict], duration: int) -> list[tuple[str, int, int]]:
    valid = []
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        start_sec, end_sec = seg_bounds(segment, duration)
        if start_sec < end_sec:
            valid.append((text, start_sec, end_sec))
    return valid


def score_regex_then_ensemble(
    valid: list[tuple[str, int, int]], regex_pattern, regex_score: float, task, config, ensemble_threshold: float | None = None
) -> tuple[np.ndarray, int]:
    scores = np.zeros(len(valid), dtype=np.float64)
    regex_hits = 0
    ambiguous_idx = []
    for segment_idx, (text, _, _) in enumerate(valid):
        if regex_pattern.search(text):
            scores[segment_idx] = regex_score
            regex_hits += 1
        else:
            ambiguous_idx.append(segment_idx)

    if ambiguous_idx:
        ambiguous_texts = [valid[segment_idx][0] for segment_idx in ambiguous_idx]
        ensemble_scores = classify_segments(ambiguous_texts, task, config)
        for result_idx, segment_idx in enumerate(ambiguous_idx):
            score = ensemble_scores[result_idx]
            scores[segment_idx] = score if ensemble_threshold is None or score >= ensemble_threshold else 0.0

    return scores, regex_hits


def spread_scores_over_timeline(valid: list[tuple[str, int, int]], scores: np.ndarray, duration: int) -> np.ndarray:
    out = np.zeros(duration, dtype=np.float64)
    for segment_idx, (_, start_sec, end_sec) in enumerate(valid):
        out[start_sec:end_sec] = scores[segment_idx]
    return out
