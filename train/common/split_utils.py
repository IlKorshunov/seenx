from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def resolve_train_val_split(args: Any, video_ids: list[str], output_video_ids: list[str]) -> tuple[list[str], list[str]]:
    if getattr(args, "eval_video", "") and args.eval_video in video_ids:
        return [v for v in video_ids if v != args.eval_video], [args.eval_video]
    if getattr(args, "val_first_n_output", 0) > 0:
        n_val = min(args.val_first_n_output, len(output_video_ids))
        val_ids = output_video_ids[:n_val]
        logger.info("Validation split: first %d videos from output", n_val)
        return [v for v in video_ids if v not in set(val_ids)], val_ids
    shuffled = list(video_ids)
    np.random.RandomState(args.random_seed).shuffle(shuffled)  # pylint: disable=no-member
    n_val = max(1, int(len(shuffled) * args.val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def apply_train_id_file_filter(train_ids: list[str], args: Any) -> list[str]:
    path = getattr(args, "train_video_ids_file", "") or ""
    if not path:
        return train_ids
    allow = {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}
    filtered = [v for v in train_ids if v in allow]
    logger.info("Train subset from file: %d -> %d videos (%s)", len(train_ids), len(filtered), path)
    return filtered
