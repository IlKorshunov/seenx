"""Cultural references from speech transcript (spaCy NER + regex fallback).

Uses Whisper segments (start/end/text) via get_transcript caching. Detects
PERSON, ORG, EVENT, WORK_OF_ART (English) or PER, ORG, MISC (multilingual wiki).

Output (per-second, 1 Hz):
  has_person_mention — 0/1: PERSON/PER in a segment covering this second
  has_org_mention — 0/1: ORG in a segment covering this second
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, logger, seg_bounds, skip_if_exists


_COLS = {"has_person_mention", "has_org_mention"}

_MAX_TEXT_CHARS = 5000

_PERSON_LABELS = frozenset({"PERSON", "PER"})
_ORG_LABELS = frozenset({"ORG"})
_DENSITY_EXTRA = frozenset({"EVENT", "WORK_OF_ART", "MISC"})

_MULTIWORD_LATIN = re.compile(r"(?<![\w/])([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?![\w/])")
_MULTIWORD_CYR = re.compile(r"(?<![\w/])([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)+)(?![\w/])")
_ORG_HINT = re.compile(r"\b(?:ООО|ИП|АО|ПАО|ЗАО|НАО|LLC|Inc\.?|Ltd\.?|Corp\.?|GmbH|Company|Group|Holdings)\b", re.IGNORECASE)

_nlp: Any = None
_spacy_exhausted = False


def _try_load_spacy(config) -> Any:
    global _nlp, _spacy_exhausted
    if _spacy_exhausted:
        return None
    if _nlp is not None:
        return _nlp

    import spacy

    seen: set[str] = set()
    candidates: list[str] = []
    custom = config.get("cultural_ref_spacy_model") if config is not None else None
    for name in (custom, "xx_ent_wiki_sm", "en_core_web_sm"):
        if name and isinstance(name, str) and name not in seen:
            seen.add(name)
            candidates.append(name)

    for name in candidates:
        try:
            _nlp = spacy.load(name, disable=["parser", "lemmatizer"])
            logger.info("cultural references: loaded spaCy model %r", name)
            return _nlp
        except Exception as exc:
            logger.warning("cultural references: spaCy model %r unavailable (%s)", name, exc)

    _spacy_exhausted = True
    _nlp = None
    logger.info("cultural references: using regex fallback (no spaCy NER model)")
    return None


def _spacy_entity_counts(nlp, text: str) -> tuple[int, bool, bool]:
    snippet = text[:_MAX_TEXT_CHARS]
    doc = nlp(snippet)
    n_density = 0
    has_person = False
    has_org = False
    for ent in doc.ents:
        lab = ent.label_
        if lab in _PERSON_LABELS:
            has_person = True
            n_density += 1
        elif lab in _ORG_LABELS:
            has_org = True
            n_density += 1
        elif lab in _DENSITY_EXTRA:
            n_density += 1
    return n_density, has_person, has_org


def _regex_entity_counts(text: str) -> tuple[int, bool, bool]:
    """Capitalized multi-word phrases as named-entity proxies; org suffix hints."""
    if not text or not text.strip():
        return 0, False, False
    phrases = set(_MULTIWORD_LATIN.findall(text)) | set(_MULTIWORD_CYR.findall(text))
    org_hit = bool(_ORG_HINT.search(text))
    n_phrases = len(phrases)
    n_density = n_phrases + (1 if org_hit and n_phrases == 0 else 0)
    has_org = org_hit or bool(re.search(r"\b\w+\s+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|GmbH|Group)\b", text, re.I))
    has_person = n_phrases > 0
    return n_density, has_person, has_org


def _segment_analysis(nlp, text: str) -> tuple[int, bool, bool]:
    if nlp is not None:
        try:
            return _spacy_entity_counts(nlp, text)
        except Exception as exc:
            logger.warning("cultural references: spaCy failed on segment (%s), regex fallback", exc)
    return _regex_entity_counts(text)


def extract_cultural_references(video_path: str, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "cultural_references"):
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)
    duration = max(int(duration), 1)

    valid: list[tuple[str, int, int]] = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        start_sec, end_sec = seg_bounds(segment, duration)
        if start_sec < end_sec:
            valid.append((text, start_sec, end_sec))

    zeros = {"has_person_mention": np.zeros(duration, dtype=np.float64), "has_org_mention": np.zeros(duration, dtype=np.float64)}

    if not valid:
        logger.info("cultural references: no transcript segments, zeros (%d s)", duration)
        return pd.DataFrame(zeros)

    nlp = _try_load_spacy(config)

    has_person_sec = np.zeros(duration, dtype=np.float64)
    has_org_sec = np.zeros(duration, dtype=np.float64)
    entity_total = 0

    for text, start_sec, end_sec in valid:
        entity_count, has_person, has_org = _segment_analysis(nlp, text)
        entity_total += entity_count
        if has_person:
            has_person_sec[start_sec:end_sec] = 1.0
        if has_org:
            has_org_sec[start_sec:end_sec] = 1.0

    logger.info("cultural references: %d segments, %d entity hits (spaCy=%s)", len(valid), entity_total, nlp is not None)

    return pd.DataFrame({"has_person_mention": has_person_sec, "has_org_mention": has_org_sec})
