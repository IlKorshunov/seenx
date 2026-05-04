import json
import os
from dataclasses import dataclass

import torch


_CONFIGS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
AUDIO_CONFIG_PATH = os.path.join(_CONFIGS_DIR, "bumpers.json")
VIDEO_CONFIG_PATH = os.path.join(_CONFIGS_DIR, "bumpers_video.json")


@dataclass
class BumpCandidate:
    video_id: str
    bumper_type: int
    start_sec: float
    end_sec: float
    duration_sec: float
    audio_score: float


@dataclass
class BumperConfig:
    audio_sr: int = 22050
    chroma_fps: int = 2
    scan_ratio: float = 0.40
    min_bumper_sec: int = 3
    max_bumper_sec: int = 20
    match_threshold: float = 0.90
    mask_score_ratio: float = 0.9
    mask_radius_sec: int = 5
    silence_thresh: float = 0.005
    locate_threshold: float = 0.85
    locate_max_per_video: int = 3
    min_candidate_videos: int = 2

    @classmethod
    def from_json(cls, path: str = AUDIO_CONFIG_PATH) -> "BumperConfig":
        """Load config from JSON file, ignoring unknown keys."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class VideoVerifierConfig:
    clip_model: str = "openai/clip-vit-large-patch14"
    device: str = "auto"
    visual_threshold: float = 0.80
    min_videos_agree: int = 2
    n_frames: int = 5

    @classmethod
    def from_json(cls, path: str = VIDEO_CONFIG_PATH) -> "VideoVerifierConfig":
        """Load config from JSON file, ignoring unknown keys."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
