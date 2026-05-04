"""Clickbait / semantic gap features: title promise vs actual delivery.

Produces per-second columns:
  - title_transcript_gap   : cosine distance between title+description embedding
                             and a rolling window of transcript text. High values
                             mean the spoken content diverges from the title promise.
                             Constant ~0 when no transcript or no title available.
  - title_delivery_30s     : scalar (broadcast) — gap specifically for the first 30 s.
                             Proxy for "clickbait disappointment" (high = title promise
                             not met in the opening). Available at inference time.
  - title_claim_intensity  : scalar (broadcast) — density of clickbait-style language
                             in the title + first line of description (rules-based).

Data sources:
  - Title & description from get_data/comments/<playlist>/<vid>/comments.json
    (field video_title, video_description).
  - Transcript from the standard pipeline (get_segments_and_duration).
  - Embeddings via deepvk/USER2-base (same model as semantic_embedding_feature).
"""

from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ._base import get_segments_and_duration, logger, seg_bounds, skip_if_exists


_COLS = {"title_transcript_gap", "title_delivery_30s", "title_claim_intensity"}

MODEL_ID = "deepvk/USER2-base"
EMB_DIM = 256

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_COMMENTS_ROOT = _PROJECT_ROOT / "get_data" / "comments"

WINDOW_SEC = 15
DELIVERY_HORIZON_SEC = 30

_CLAIM_RE = re.compile(
    r"(секрет|раскрою|покажу|расскажу|узнаете|научу|объясню"
    r"|никто не знает|мало кто знает|вы не поверите"
    r"|впервые|эксклюзив|уникальн|шок|невероятн|топ[\s\-]?\d"
    r"|правда\s+о|вся\s+правда|запрещён|скрыва|разоблач"
    r"|самый\s+(?:страшн|опасн|шокирующ|удивительн)"
    r"|этого.*не\s+(?:знал|покажут|расскаж)"
    r"|до\s+слёз|до\s+мурашек)",
    re.IGNORECASE,
)

_SUPERLATIVE_RE = re.compile(r"\b(лучш|худш|величайш|сильнейш|главн|невероятн|безумн|гениальн)\w*\b", re.IGNORECASE)

_CAPS_WORD_RE = re.compile(r"\b[А-ЯЁA-Z]{3,}\b")


def _video_id(video_path: str) -> str:
    return os.path.basename(os.path.dirname(video_path)) if video_path.endswith(".mp4") else os.path.splitext(os.path.basename(video_path))[0]


def _find_comments_json(video_id: str) -> Path | None:
    for comments_path in _COMMENTS_ROOT.rglob(f"{video_id}/comments.json"):
        return comments_path
    return None


def _load_title_desc(video_id: str) -> tuple[str, str]:
    comments_path = _find_comments_json(video_id)
    if comments_path is None:
        return "", ""
    try:
        data = json.loads(comments_path.read_text(encoding="utf-8"))
        return data.get("video_title", ""), data.get("video_description", "")
    except Exception:
        return "", ""


def _title_claim_intensity(title: str, desc_first_line: str) -> float:
    combined = f"{title} {desc_first_line}"
    if not combined.strip():
        return 0.0
    words = combined.split()
    n_words = max(len(words), 1)

    claim_hits = len(_CLAIM_RE.findall(combined))
    superlative_hits = len(_SUPERLATIVE_RE.findall(combined))
    caps_hits = len(_CAPS_WORD_RE.findall(combined))

    score = min(claim_hits / n_words * 8.0, 1.0) * 0.5 + min(superlative_hits / n_words * 6.0, 1.0) * 0.25 + min(caps_hits / n_words * 4.0, 1.0) * 0.25
    return float(np.clip(score, 0.0, 1.0))


def _encode_texts(texts: list[str], model, tokenizer, device, batch_size: int = 64) -> np.ndarray:
    all_embs: list[np.ndarray] = []
    for batch_start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[batch_start : batch_start + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled[:, :EMB_DIM], p=2, dim=1)
        all_embs.append(pooled.cpu().float().numpy())
    return np.vstack(all_embs) if all_embs else np.zeros((0, EMB_DIM), dtype=np.float32)


def extract_clickbait_gap(video_path: str, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "clickbait gap"):
        return pd.DataFrame()

    video_id = _video_id(video_path)
    segments, duration = get_segments_and_duration(video_path, config)
    duration = max(duration, 1)

    title, desc = _load_title_desc(video_id)
    desc_first_line = (desc or "").split("\n")[0].strip()

    claim_score = _title_claim_intensity(title, desc_first_line)

    valid = [(seg, *seg_bounds(seg, duration)) for seg in segments if seg.get("text", "").strip()]
    valid = [(segment, start_sec, end_sec) for segment, start_sec, end_sec in valid if start_sec < end_sec]

    if not title or not valid:
        logger.info("Clickbait gap: no title or no transcript for %s, returning zeros", video_id)
        return pd.DataFrame(
            {
                "title_transcript_gap": np.zeros(duration, dtype=np.float32),
                "title_delivery_30s": np.zeros(duration, dtype=np.float32),
                "title_claim_intensity": np.full(duration, claim_score, dtype=np.float32),
            }
        )

    from transformers import AutoModel, AutoTokenizer

    device = torch.device(config.get("device"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()

    title_text = f"{title}. {desc_first_line}" if desc_first_line else title
    title_emb = _encode_texts([title_text], model, tokenizer, device)[0]

    seg_texts = [seg["text"].strip() for seg, _, _ in valid]
    seg_embs = _encode_texts(seg_texts, model, tokenizer, device)

    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_seg_sim = seg_embs @ title_emb

    gap_per_sec = np.zeros(duration, dtype=np.float64)
    count_per_sec = np.zeros(duration, dtype=np.float64)

    for segment_idx, (_, start_sec, end_sec) in enumerate(valid):
        half = WINDOW_SEC
        window_start = max(0, start_sec - half)
        window_end = min(duration, end_sec + half)
        gap_per_sec[window_start:window_end] += 1.0 - per_seg_sim[segment_idx]
        count_per_sec[window_start:window_end] += 1.0

    safe_count = np.maximum(count_per_sec, 1.0)
    title_transcript_gap = (gap_per_sec / safe_count).astype(np.float32)

    early_segments = [(similarity, start_sec, end_sec) for (similarity, (_, start_sec, end_sec)) in zip(per_seg_sim, valid, strict=True) if start_sec < DELIVERY_HORIZON_SEC]
    if early_segments:
        weights = np.array([end_sec - start_sec for _, start_sec, end_sec in early_segments], dtype=np.float64)
        similarities = np.array([similarity for similarity, _, _ in early_segments], dtype=np.float64)
        delivery_30s = float(1.0 - np.average(similarities, weights=weights / weights.sum()))
    else:
        delivery_30s = float(title_transcript_gap[:DELIVERY_HORIZON_SEC].mean()) if duration >= DELIVERY_HORIZON_SEC else 0.0

    logger.info("Clickbait gap %s: claim=%.3f, delivery_30s=%.3f, gap_mean=%.3f", video_id, claim_score, delivery_30s, float(title_transcript_gap.mean()))

    return pd.DataFrame(
        {
            "title_transcript_gap": title_transcript_gap,
            "title_delivery_30s": np.full(duration, delivery_30s, dtype=np.float32),
            "title_claim_intensity": np.full(duration, claim_score, dtype=np.float32),
        }
    )
