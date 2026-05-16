import json
import os

import numpy as np

from ._base import seg_bounds
from ._zeroshot import classify_segments
from ..video.architectures.common import unload_ensemble


EMBEDDINGS_ROOT = "embeddings"


def video_id(video_path: str) -> str:
    return os.path.basename(os.path.dirname(video_path)) if video_path.endswith(".mp4") else os.path.splitext(os.path.basename(video_path))[0]


def release_models(*models, device=None) -> None:
    unload_ensemble(models, device)


def collect_valid_segments(segments: list[dict], duration: int) -> list[tuple[str, int, int]]:
    valid = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        start_sec, end_sec = seg_bounds(segment, duration)
        if start_sec < end_sec:
            valid.append((text, start_sec, end_sec))
    return valid


def collect_valid_segment_dicts(segments: list[dict]) -> list[dict]:
    return [segment for segment in segments or [] if isinstance(segment, dict) and (segment.get("text") or "").strip()]


def collect_valid_segments_with_mid(segments: list[dict], duration: int) -> list[tuple[str, int, int, float]]:
    valid = []
    for text, start_sec, end_sec in collect_valid_segments(segments, duration):
        mid_fraction = ((start_sec + end_sec) / 2.0) / max(duration, 1)
        valid.append((text, start_sec, end_sec, mid_fraction))
    return valid


def load_segment_embeddings(video_id_value: str, embeddings_root: str = EMBEDDINGS_ROOT, require_metadata: bool = False) -> tuple[np.ndarray | None, list[dict]]:
    emb_path = os.path.join(embeddings_root, video_id_value, "seg_embeddings.npy")
    meta_path = os.path.join(embeddings_root, video_id_value, "seg_meta.json")
    if not os.path.exists(emb_path) or (require_metadata and not os.path.exists(meta_path)):
        return None, []

    embeddings = np.load(emb_path, allow_pickle=False).astype(np.float32)
    metadata = []
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    return embeddings, metadata


def score_regex_then_ensemble(
    valid: list[tuple[str, int, int]],
    regex_pattern,
    regex_score: float,
    task,
    config,
    ensemble_threshold: float | None = None,
    threshold_value: float | None = None,
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
            if ensemble_threshold is None or score >= ensemble_threshold:
                scores[segment_idx] = score if threshold_value is None else threshold_value

    return scores, regex_hits


def spread_scores_over_timeline(valid: list[tuple[str, int, int]], scores: np.ndarray, duration: int) -> np.ndarray:
    out = np.zeros(duration, dtype=np.float64)
    for segment_idx, (_, start_sec, end_sec) in enumerate(valid):
        out[start_sec:end_sec] = scores[segment_idx]
    return out
