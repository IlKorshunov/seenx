import json
import os
from pathlib import Path
from typing import Any


def _resolve_device(value: str) -> str:
    if value == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "gpu":
        return "cuda"
    return value


def get_device(config) -> str:
    """Resolve device string from config (handles both Config and plain dict)."""
    device = config.get("device", "cpu")
    return _resolve_device(device) if isinstance(device, str) else "cpu"


def _resolve_batch_size(key: str, value: Any, default: Any) -> int:
    """Ultralytics-style: -1 = auto (60% GPU), 0.0-1.0 = fraction."""
    if value is None or (isinstance(value, (int, float)) and value >= 1):
        return int(value) if value is not None and value >= 1 else default
    from .autobatch import resolve_batch_size

    if key == "shot_segmentor_batch_size":
        return resolve_batch_size(value, default=64, task="transnet")
    if key == "text_prob_batch_size":
        return resolve_batch_size(value, default=8, task="ocr")
    if key == "face_screen_batch_size":
        return resolve_batch_size(value, default=32, task="yolo")
    return resolve_batch_size(value, default=4, task="default")


def _default_config_path() -> str | None:
    """configs/local.json next to project root (parent of src/)."""
    root = Path(__file__).resolve().parents[2]
    p = root / "configs" / "local.json"
    return str(p) if p.is_file() else None


class Config:
    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = _default_config_path()
        if config_path is None:
            self.config_data: dict[str, Any] = {}
            return
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")

        with open(config_path, encoding="utf-8") as f:
            self.config_data: dict[str, Any] = json.load(f)

    def get(self, key: str, default: Any = None) -> Any:
        value = self.config_data.get(key, default)
        if key in ("device", "demucs_device") and isinstance(value, str):
            return _resolve_device(value)
        if key in ("batch_size", "shot_segmentor_batch_size", "text_prob_batch_size", "face_screen_batch_size"):
            def_val = 64 if key == "shot_segmentor_batch_size" else (32 if key == "face_screen_batch_size" else (8 if key == "text_prob_batch_size" else 4))
            if value is None:
                if key in ("text_prob_batch_size", "face_screen_batch_size"):
                    return _resolve_batch_size(key, -1, def_val)
                return def_val
            if value == -1 or (isinstance(value, (int, float)) and 0 < value < 1):
                return _resolve_batch_size(key, value, def_val)
        return value


if __name__ == "__main__":
    print(f"pwd='{os.getcwd()}'")
    config = Config("configs/init.json")
