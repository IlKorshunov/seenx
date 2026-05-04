"""Resplit existing 30s segments into 15s bins without re-transcribing."""

import glob
import json
import os


MARKUP_DIR = "markup"
NEW_BIN_SEC = 15


def resplit_video(video_dir: str) -> None:
    seg_path = os.path.join(video_dir, "segments.json")
    if not os.path.exists(seg_path):
        return

    with open(seg_path, encoding="utf-8") as f:
        old_segments = json.load(f)

    # Back up original
    backup_path = seg_path + ".30s_backup"
    if not os.path.exists(backup_path):
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(old_segments, f, ensure_ascii=False, indent=2)

    new_segments = []
    for seg in old_segments:
        start = seg["start_sec"]
        end = seg["end_sec"]
        frames = seg["frames"]
        transcript = seg["transcript"]
        words = transcript.split()

        mid = start + NEW_BIN_SEC
        if mid >= end:
            new_segments.append({"seg_idx": len(new_segments), "start_sec": start, "end_sec": end, "n_frames": len(frames), "frames": frames, "transcript": transcript})
            continue

        # Split frames by time: each frame is at start + 1 + i*2 seconds
        first_half_frames = []
        second_half_frames = []
        for i, fname in enumerate(frames):
            frame_time = start + 1.0 + i * 2.0
            if frame_time < mid:
                first_half_frames.append(fname)
            else:
                second_half_frames.append(fname)

        # Split transcript roughly in half by words
        half_words = len(words) // 2
        first_text = " ".join(words[:half_words])
        second_text = " ".join(words[half_words:])

        new_segments.append(
            {"seg_idx": len(new_segments), "start_sec": start, "end_sec": round(mid, 1), "n_frames": len(first_half_frames), "frames": first_half_frames, "transcript": first_text}
        )
        new_segments.append(
            {"seg_idx": len(new_segments), "start_sec": round(mid, 1), "end_sec": end, "n_frames": len(second_half_frames), "frames": second_half_frames, "transcript": second_text}
        )

    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump(new_segments, f, ensure_ascii=False, indent=2)

    vid = os.path.basename(video_dir)
    print(f"  {vid}: {len(old_segments)} x 30s -> {len(new_segments)} x 15s", flush=True)


def main():
    dirs = sorted(glob.glob(os.path.join(MARKUP_DIR, "*")))
    print(f"Resplitting segments for {len(dirs)} videos\n", flush=True)
    for d in dirs:
        if os.path.isdir(d):
            resplit_video(d)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
