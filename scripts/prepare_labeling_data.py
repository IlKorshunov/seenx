"""Prepare labeling data: extract frames + transcripts for 30-sec segments."""

import json
import math
import os
import subprocess
import sys

import torch
import whisper
from tqdm import tqdm


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PILOT_VIDEOS = [
    "U8e1Pd7aBLY",
    "b5gAWk2QuSU",
    "6Ha1lqRKMig",
    "K5zVqoMMj3Y",
    "3PBo1GiAlbs",
    "bK-1Mfb93uA",
    "3G6JrAq-w4M",
    "Ei6QuJppkb4",
    "gkSAR3jbMjA",
    "ie3vumn9Je4",
    "138P93_Z_u0",
    "1J5zlq2Vs3Y",
    "39R6j9qkzPo",
    "3MncoqWP_qg",
    "3WA4YT8fkx8",
    "4WcGtcKKoWg",
    "6_qPAwOsDi0",
    "7g_56jn0NCA",
    "8__NWVjeyZ8",
    "8YXWaEo201o",
    "9B3N8I4Vidc",
    "BZEZxzRJ-L4",
    "Db6f982wxPg",
    "DhFuAhFMvms",
    "DrS6_CT0XrU",
    "dx7dMwMXSAA",
    "e21aHqRYPKE",
    "ECP4p12A5FY",
    "EFxlvlchMw8",
    "EhAdb6fIEe0",
    "eL9OqXpmvOE",
    "eLSgNBnpeRg",
    "Fr4ZqY97Bco",
    "fwHhuHVNR4I",
    "FYbEJ3oRgFk",
    "FZ4waB-THOA",
    "GfEdSBXxyHA",
    "GthDbmZFW-w",
    "gYHHQNT6100",
    "hnyYg62DY4E",
    "ib29EiVnmJM",
    "iKGmIRAfmCM",
    "iP_siVZjbzc",
    "jPVQgOI33V0",
    "lLoSngvynB8",
    "sujXtr4AVRU",
    "xZnMwJQ4mfc",
    "yKC5nekLvbs",
    "yv4o4Epz1eM",
    "IgJxh-fJTBU",
]

DATA_DIR = "data"
OUTPUT_DIR = "markup"
SEGMENT_SEC = 30
FRAME_INTERVAL_SEC = 2
FRAME_QUALITY = 2

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Loading Whisper large-v3 on {device}")
        _whisper_model = whisper.load_model("large-v3", device=device)
    return _whisper_model


def get_duration(video_path: str) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video_path], capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_frame(video_path: str, time_sec: float, out_path: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-ss", str(time_sec), "-i", video_path, "-frames:v", "1", "-q:v", str(FRAME_QUALITY), out_path], capture_output=True)


def transcribe_video(video_path: str) -> list[dict]:
    model = _get_whisper()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = model.transcribe(video_path, language="ru", word_timestamps=True, fp16=(device == "cuda"))
    return result["segments"]


def segments_for_window(all_segs: list[dict], start: float, end: float) -> str:
    texts = []
    for seg in all_segs:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        texts.append(seg["text"].strip())
    return " ".join(texts).strip()


def prepare_video(video_id: str) -> None:
    video_path = os.path.join(DATA_DIR, video_id, "video.mp4")
    if not os.path.isfile(video_path):
        log(f"  SKIP {video_id}: no video.mp4")
        return

    out_dir = os.path.join(OUTPUT_DIR, video_id)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    segments_path = os.path.join(out_dir, "segments.json")
    if os.path.exists(segments_path):
        log(f"  SKIP {video_id}: already prepared")
        return

    duration = get_duration(video_path)
    n_segments = math.ceil(duration / SEGMENT_SEC)
    log(f"{video_id}: {duration:.0f}s, {n_segments} segments")

    whisper_segs = transcribe_video(video_path)

    segments = []
    total_frames = 0
    for i in tqdm(range(n_segments), desc=f"  {video_id} frames", unit="seg"):
        start = i * SEGMENT_SEC
        end = min((i + 1) * SEGMENT_SEC, duration)

        frame_times = []
        t = start + 1.0
        while t < end - 0.5:
            frame_times.append(min(t, duration - 0.5))
            t += FRAME_INTERVAL_SEC
        if not frame_times:
            frame_times = [min(start + 1.0, duration - 0.5)]

        frame_files = []
        for j, ft in enumerate(frame_times):
            frame_path = os.path.join(frames_dir, f"seg_{i:02d}_f{j:02d}.jpg")
            frame_files.append(os.path.basename(frame_path))
            if not os.path.exists(frame_path):
                extract_frame(video_path, ft, frame_path)
        total_frames += len(frame_files)

        transcript = segments_for_window(whisper_segs, start, end)
        segments.append({"seg_idx": i, "start_sec": round(start, 1), "end_sec": round(end, 1), "n_frames": len(frame_files), "frames": frame_files, "transcript": transcript})

    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    log(f"  Saved {len(segments)} segments + {total_frames} frames")


def log(msg: str) -> None:
    print(msg, flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_vids = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and os.path.isfile(os.path.join(DATA_DIR, d, "video.mp4")))
    log(f"Preparing labeling data for {len(all_vids)} videos\n")
    for vid in tqdm(all_vids, desc="Videos", unit="video"):
        prepare_video(vid)


if __name__ == "__main__":
    main()
