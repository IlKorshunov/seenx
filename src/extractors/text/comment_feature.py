"""Per-second features derived from YouTube comments and video description.

Produces columns:
  - desc_chapter_start        : 1 at seconds where an author chapter begins (from description)
  - desc_chapter_boundary_dist: normalized [0,1] distance to nearest chapter boundary
  - timecode_like_weighted_30s: log1p(like_count) of timecoded comments in ±15 s window, normalized
  - comment_question_rate_30s : fraction of timecoded comments in ±15 s window that contain a question
  - comment_density_30s       : count of timecoded comments in ±15 s window, log-scaled
  - comment_positive_rate_30s : fraction of timecoded comments in ±15 s that are enthusiastic/positive
  - comment_reply_depth_30s   : mean reply count of timecoded comment threads in ±15 s, log-scaled
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._base import get_segments_and_duration, logger, skip_if_exists


_COLS = {
    "desc_chapter_start",
    "desc_chapter_boundary_dist",
    "timecode_like_weighted_30s",
    "comment_question_rate_30s",
    "comment_density_30s",
    "comment_positive_rate_30s",
    "comment_aggression_rate_30s",
    "comment_reply_depth_30s",
    "author_reply_rate_video",
    "avg_comment_length_video",
    "complex_words_ratio_video",
}

_HALF_WINDOW = 15

_TIMECODE_RE = re.compile(r"(?<!\d)(?:(\d+):)?([0-5]?\d):([0-5]\d)(?!\d)")

_QUESTION_RE = re.compile(
    r"\?|"
    r"\b(?:что|как|почему|зачем|когда|где|куда|откуда|кто|сколько|"
    r"какой|какая|какое|какие|чей|неужели|разве|"
    r")\b",
    re.IGNORECASE,
)

_POSITIVE_RE = re.compile(
    r"\b(?:круто|класс|топ|огонь|лучший|лучшее|лучшая|лучшие|"
    r"супер|шедевр|великолепно|прекрасно|потрясающе|замечательно|обожаю|"
    r"кайф|шикарно|восторг|восхитительно|браво|молодцы|спасибо)\b",
    re.IGNORECASE,
)

_AGGRESSIVE_RE = re.compile(r"\b(?:бред|чушь|отписка|ужас|говно|хуйня|пиздец|заебал|бесит|тупой|тупая|идиот|дизлайк|хрень|мусор|скучно|нудно|херня|очередной|дно)\b", re.IGNORECASE)

_KNOWN_OWNERS = {"@ivanlyrics"}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_COMMENTS_ROOT = _PROJECT_ROOT / "get_data" / "comments"


def _video_id(video_path: str) -> str:
    return os.path.basename(os.path.dirname(video_path)) if video_path.endswith(".mp4") else os.path.splitext(os.path.basename(video_path))[0]


def _find_comments_json(video_id: str) -> Path | None:
    """Search all playlist dirs for this video's comments."""
    for comments_path in _COMMENTS_ROOT.rglob(f"{video_id}/comments.json"):
        return comments_path
    return None


def _parse_description_timecodes(description: str) -> list[int]:
    """Extract chapter start seconds from video description."""
    timecodes: list[int] = []
    for match in _TIMECODE_RE.finditer(description or ""):
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        timecodes.append(hours * 3600 + minutes * 60 + seconds)
    return sorted(set(timecodes))


def _flatten_comments(threads: list[dict]) -> list[dict[str, Any]]:
    """Flatten threads into a list of {text, timecodes, like_count, n_replies}."""
    out: list[dict[str, Any]] = []
    for thread in threads:
        n_replies = len(thread.get("replies") or [])
        out.append(
            {
                "text": thread.get("text", ""),
                "timecodes": thread.get("timecodes", []),
                "like_count": int(thread.get("like_count", 0)),
                "n_replies": n_replies,
                "updated_at": thread.get("updated_at"),
            }
        )
        for reply in thread.get("replies") or []:
            out.append(
                {
                    "text": reply.get("text", ""),
                    "timecodes": reply.get("timecodes", []),
                    "like_count": int(reply.get("like_count", 0)),
                    "n_replies": 0,
                    "updated_at": reply.get("updated_at"),
                }
            )
    return out


def _rolling_window_features(comments: list[dict[str, Any]], duration: int, half_window: int = _HALF_WINDOW) -> dict[str, np.ndarray]:
    """Compute rolling-window features for timecoded comments."""
    like_weighted = np.zeros(duration, dtype=np.float64)
    question_cnt = np.zeros(duration, dtype=np.float64)
    total_cnt = np.zeros(duration, dtype=np.float64)
    positive_cnt = np.zeros(duration, dtype=np.float64)
    aggression_cnt = np.zeros(duration, dtype=np.float64)
    reply_sum = np.zeros(duration, dtype=np.float64)

    for comment in comments:
        timecodes = comment.get("timecodes") or []
        if not timecodes:
            continue
        is_question = 1.0 if _QUESTION_RE.search(comment["text"]) else 0.0
        is_positive = 1.0 if _POSITIVE_RE.search(comment["text"]) else 0.0
        is_aggression = 1.0 if _AGGRESSIVE_RE.search(comment["text"]) else 0.0
        like_val = math.log1p(comment["like_count"])
        reply_val = math.log1p(comment["n_replies"])

        for timecode in timecodes:
            second = timecode.get("seconds", 0)
            window_start = max(0, second - half_window)
            window_end = min(duration, second + half_window + 1)
            like_weighted[window_start:window_end] += like_val
            question_cnt[window_start:window_end] += is_question
            total_cnt[window_start:window_end] += 1.0
            positive_cnt[window_start:window_end] += is_positive
            aggression_cnt[window_start:window_end] += is_aggression
            reply_sum[window_start:window_end] += reply_val

    safe_total = np.maximum(total_cnt, 1.0)

    max_like = like_weighted.max()
    if max_like > 0:
        like_weighted /= max_like

    return {
        "timecode_like_weighted_30s": like_weighted.astype(np.float32),
        "comment_question_rate_30s": (question_cnt / safe_total).astype(np.float32),
        "comment_density_30s": np.log1p(total_cnt).astype(np.float32),
        "comment_positive_rate_30s": (positive_cnt / safe_total).astype(np.float32),
        "comment_aggression_rate_30s": (aggression_cnt / safe_total).astype(np.float32),
        "comment_reply_depth_30s": (reply_sum / safe_total).astype(np.float32),
    }


def _description_chapter_features(chapter_seconds: list[int], duration: int) -> dict[str, np.ndarray]:
    """Binary chapter-start flag + normalized distance to nearest boundary."""
    chapter_start = np.zeros(duration, dtype=np.float32)
    boundary_dist = np.ones(duration, dtype=np.float32)

    if not chapter_seconds:
        return {"desc_chapter_start": chapter_start, "desc_chapter_boundary_dist": boundary_dist}

    valid_chapter_seconds = [chapter_second for chapter_second in chapter_seconds if 0 <= chapter_second < duration]
    for chapter_second in valid_chapter_seconds:
        chapter_start[chapter_second] = 1.0

    if valid_chapter_seconds:
        boundaries = np.array(valid_chapter_seconds, dtype=np.float64)
        time_axis = np.arange(duration, dtype=np.float64)
        dists = np.abs(time_axis[:, None] - boundaries[None, :]).min(axis=1)
        max_dist = max(float(dists.max()), 1.0)
        boundary_dist = (dists / max_dist).astype(np.float32)

    return {"desc_chapter_start": chapter_start, "desc_chapter_boundary_dist": boundary_dist}


def _video_level_features(data: dict, comments: list[dict[str, Any]], duration: int) -> dict[str, np.ndarray]:
    """Compute video-level features broadcasted to (duration,)."""
    threads = data.get("threads", [])
    author_reply_rate_video = 0.0
    if threads:
        author_reply_count = 0
        for thread in threads:
            replies = thread.get("replies") or []
            if any(reply.get("author") in _KNOWN_OWNERS for reply in replies):
                author_reply_count += 1
        author_reply_rate_video = author_reply_count / len(threads)

    avg_len = 0.0
    complex_ratio = 0.0
    if comments:
        total_len = 0
        total_words = 0
        complex_words = 0
        for comment in comments:
            text = comment.get("text", "")
            total_len += len(text)
            words = [word for word in re.findall(r"\b\w+\b", text) if not word.isdigit()]
            total_words += len(words)
            complex_words += sum(1 for word in words if len(word) >= 8)

        avg_len = total_len / len(comments)
        complex_ratio = complex_words / max(total_words, 1)

    return {
        "author_reply_rate_video": np.full(duration, author_reply_rate_video, dtype=np.float32),
        "avg_comment_length_video": np.full(duration, avg_len, dtype=np.float32),
        "complex_words_ratio_video": np.full(duration, complex_ratio, dtype=np.float32),
    }


def extract_comment_features(video_path: str, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "comment features"):
        return pd.DataFrame()

    video_id = _video_id(video_path)
    _, duration = get_segments_and_duration(video_path, config)
    duration = max(duration, 1)

    comments_path = _find_comments_json(video_id)
    if comments_path is None:
        logger.warning("No comments.json for %s, returning zeros", video_id)
        return pd.DataFrame({column: np.zeros(duration, dtype=np.float32) for column in _COLS})

    try:
        data = json.loads(comments_path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.error("Failed to read %s: %s", comments_path, error)
        return pd.DataFrame({column: np.zeros(duration, dtype=np.float32) for column in _COLS})

    description = data.get("video_description", "")
    threads = data.get("threads", [])
    comments = _flatten_comments(threads)

    chapter_secs = _parse_description_timecodes(description)
    logger.info("Comment features for %s: %d comments, %d desc chapters, dur=%ds", video_id, len(comments), len(chapter_secs), duration)

    result = {}
    result.update(_description_chapter_features(chapter_secs, duration))
    result.update(_rolling_window_features(comments, duration))
    result.update(_video_level_features(data, comments, duration))

    return pd.DataFrame(result)
