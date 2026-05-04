import numpy as np
import pandas as pd
import spacy

from ._base import get_segments_and_duration, skip_if_exists, segment_text, valid_text_segments
from .constants import TEXT_COMPLEXITY_COLS, TEXT_COMPLEXITY_MATTR_WINDOW, TEXT_COMPLEXITY_MIN_WORDS_FOR_DEPTH, TEXT_COMPLEXITY_WINDOW_SEC

_nlp = spacy.load("ru_core_news_sm", disable=["ner", "lemmatizer"])


def _tree_depth(token) -> int:
    children = list(token.children)
    return 0 if not children else 1 + max(_tree_depth(c) for c in children)


def _doc_depth(doc) -> float:
    depths = [_tree_depth(sent.root) for sent in doc.sents]
    return float(np.mean(depths)) if depths else 0.0


def _mattr(words, window=TEXT_COMPLEXITY_MATTR_WINDOW):
    if len(words) < window:
        return len(set(words)) / len(words) if words else 0.0
    return float(np.mean([len(set(words[i : i + window])) / window for i in range(len(words) - window + 1)]))


def normalize_masked(values, mask):
    if mask.sum() < 2:
        return np.zeros_like(values)
    sub = values[mask]
    std = sub.std()
    if std < 1e-3:
        return np.zeros_like(values)
    out = np.zeros_like(values)
    out[mask] = (sub - sub.mean()) / std
    return out


def extract_text_complexity(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(TEXT_COMPLEXITY_COLS, existing_features, "text_complexity"):
        return pd.DataFrame()
    segments, duration = get_segments_and_duration(video_path, config)

    valid_segments = valid_text_segments(segments)
    texts = [segment_text(segment) for segment in valid_segments]

    docs = list(_nlp.pipe(texts, batch_size=64))
    mid_times, depths, ttr_values, word_lens = [], [], [], []
    for segment, doc in zip(valid_segments, docs, strict=True):
        words = [t.text.lower() for t in doc if t.is_alpha and len(t.text) > 1]
        if not words:
            continue
        depth = _doc_depth(doc) if len(words) >= TEXT_COMPLEXITY_MIN_WORDS_FOR_DEPTH else 0.0
        mid_times.append((segment["start"] + segment["end"]) / 2.0)
        depths.append(depth)
        ttr_values.append(_mattr(words))
        word_lens.append(np.mean([len(w) for w in words]))

    mid_times = np.asarray(mid_times)
    depths = np.asarray(depths)
    ttr_values = np.asarray(ttr_values)
    word_lens = np.asarray(word_lens)

    depth_out = np.zeros(duration)
    ttr_out = np.zeros(duration)
    word_len_out = np.zeros(duration)
    has_speech = np.zeros(duration, dtype=bool)

    half = TEXT_COMPLEXITY_WINDOW_SEC // 2
    for sec in range(duration):
        mask = (mid_times >= sec - half) & (mid_times < sec + half)
        if not mask.any():
            continue
        depth_out[sec] = depths[mask].mean()
        ttr_out[sec] = ttr_values[mask].mean()
        word_len_out[sec] = word_lens[mask].mean()
        has_speech[sec] = True

    composite = (normalize_masked(depth_out, has_speech) + normalize_masked(ttr_out, has_speech) + normalize_masked(word_len_out, has_speech)) / 3.0
    return pd.DataFrame({"syntactic_depth": depth_out, "lexical_diversity": ttr_out, "avg_word_length": word_len_out, "speech_complexity": composite})
