"""
Combined bumper extraction pipeline: audio fingerprinting → video verification → clip cutting.

Strategy:
  1. Audio stage  (audio.py): chroma-CQT fingerprinting finds candidate intervals
     that repeat across multiple videos. Fast, but may produce false positives.
  2. Video stage  (video.py): CLIP image embeddings verify that candidate frames are
     visually similar across videos. Real bumpers share a distinctive visual identity;
     audio-only FPs do not.
  3. Only verified candidates are cut into clips and written to CSV.

Usage:
    python main.py extract_bumpers --data_dir testing --output_dir result
    python main.py extract_bumpers --data_dir testing --output_dir result --skip_video_verify
"""

import argparse
import os
import subprocess
from collections import defaultdict

import pandas as pd

from ..utils.logger import Logger
from .audio import run_audio_pipeline
from .configs import AUDIO_CONFIG_PATH, VIDEO_CONFIG_PATH, BumpCandidate, BumperConfig, VideoVerifierConfig
from .video import VideoVerifier


logger = Logger(show=True).get_logger()


def cut_segment(video_path: str, start_sec: float, end_sec: float, output_path: str) -> None:
    """Cut [start_sec, end_sec] from video_path and stream-copy to output_path via ffmpeg."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_sec:.2f}", "-to", f"{end_sec:.2f}", "-i", video_path, "-c", "copy", output_path], check=True)


def save_results(candidates: list[BumpCandidate], video_paths: dict[str, str], output_dir: str) -> None:
    """
    Cut clips for all verified candidates and write bumpers.csv to output_dir.
    Clip filename format: {video_id}_type{bumper_type}_{match_idx}.mp4
    """
    type_counters: dict[tuple, int] = defaultdict(int)
    rows = []

    for c in candidates:
        key = (c.video_id, c.bumper_type)
        match_idx = type_counters[key]
        type_counters[key] += 1

        clip = f"{c.video_id}_type{c.bumper_type}_{match_idx}.mp4"
        cut_segment(video_paths[c.video_id], c.start_sec, c.end_sec, os.path.join(output_dir, clip))
        logger.info("  cut %s: %.1f–%.1fs  audio_score=%.3f -> %s", c.video_id, c.start_sec, c.end_sec, c.audio_score, clip)
        rows.append(
            {
                "video_id": c.video_id,
                "bumper_type": c.bumper_type,
                "start_sec": c.start_sec,
                "end_sec": c.end_sec,
                "duration_sec": c.duration_sec,
                "audio_score": c.audio_score,
                "clip_file": clip,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "bumpers.csv")
    df.to_csv(csv_path, index=False)
    logger.info("%d clip(s) saved to %s", len(rows), csv_path)
    logger.info("\n%s", df.to_string(index=False))


def run_bumper_pipeline(args: argparse.Namespace) -> None:
    audio_cfg = BumperConfig.from_json(args.audio_config) if os.path.exists(args.audio_config) else BumperConfig()
    if args.scan_ratio is not None:
        audio_cfg.scan_ratio = args.scan_ratio
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("=== Stage 1: Audio fingerprinting ===")
    candidates, video_paths = run_audio_pipeline(args.data_dir, audio_cfg, max_types=args.max_types)
    if not candidates:
        logger.info("No bumper candidates from audio stage.")
        return
    logger.info("Audio stage: %d candidate interval(s) across %d video(s)", len(candidates), len(video_paths))
    if not args.skip_video_verify:
        logger.info("=== Stage 2: Video verification (CLIP) ===")
        video_cfg = VideoVerifierConfig.from_json(args.video_config) if os.path.exists(args.video_config) else VideoVerifierConfig()
        candidates = VideoVerifier(video_cfg).verify(candidates, video_paths)
        if not candidates:
            logger.info("All candidates rejected by video verification.")
            return
    else:
        logger.info("Video verification skipped (--skip_video_verify)")
    logger.info("=== Stage 3: Cutting clips ===")
    save_results(candidates, video_paths, args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="Bumper extraction: audio fingerprinting + optional CLIP video verification")
    parser.add_argument("--data_dir", default="testing")
    parser.add_argument("--output_dir", default="result")
    parser.add_argument("--audio_config", default=AUDIO_CONFIG_PATH)
    parser.add_argument("--video_config", default=VIDEO_CONFIG_PATH)
    parser.add_argument("--max_types", type=int, default=10, help="Safety cap on bumper types; actual count is auto-detected")
    parser.add_argument("--scan_ratio", type=float, default=None, help="Fraction of video to scan for audio (overrides config)")
    parser.add_argument("--skip_video_verify", action="store_true", help="Skip CLIP verification (audio-only mode)")
    run_bumper_pipeline(parser.parse_args())
