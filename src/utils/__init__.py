from __future__ import annotations

import importlib
from typing import Any

from .config import Config
from .logger import Logger

__all__ = ["Config", "Logger", "clear_cache", "get_transcript", "resolve_batch_size", "video_features"]


def __getattr__(name: str) -> Any:
    if name == "resolve_batch_size":
        from .autobatch import resolve_batch_size

        globals()["resolve_batch_size"] = resolve_batch_size
        return resolve_batch_size
    if name == "get_transcript":
        from .transcript_cache import get_transcript

        return get_transcript
    if name == "clear_cache":
        from .transcript_cache import clear_cache

        return clear_cache
    if name == "video_features":
        return importlib.import_module(f"{__name__}.video_features")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*__all__, *globals()})
