import os
from collections import defaultdict

import librosa
import numpy as np

from ..seenx_utils import get_video_duration
from ..utils.logger import Logger
from .configs import BumpCandidate, BumperConfig


logger = Logger(show=True).get_logger()


def extract_chroma(video_path: str, cfg: BumperConfig, max_sec: float | None = None) -> np.ndarray:
    """
    Load audio and compute L2-normalised chroma-CQT fingerprint.
    Returns array of shape (T, 12) at cfg.chroma_fps frames/sec.
    Silent frames (RMS < cfg.silence_thresh) are zeroed out.
    """
    hop = cfg.audio_sr // cfg.chroma_fps
    y, _ = librosa.load(video_path, sr=cfg.audio_sr, mono=True, duration=max_sec)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop).T
    chroma = librosa.feature.chroma_cqt(y=y, sr=cfg.audio_sr, hop_length=hop).T
    chroma = chroma / (np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-8)

    if rms.shape[0] > chroma.shape[0]:
        rms = rms[: chroma.shape[0]]
    elif rms.shape[0] < chroma.shape[0]:
        rms = np.pad(rms, ((0, chroma.shape[0] - rms.shape[0]), (0, 0)))
    chroma[rms[:, 0] < cfg.silence_thresh] = 0.0
    return chroma


def slide_score(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Slide ref over target and return mean per-frame dot product at each offset.
    Returns array of length (len(target) - len(ref) + 1), or empty if ref is longer.
    Since both inputs are L2-normalised per frame, each value is mean cosine similarity.
    """
    ref_len = ref.shape[0]
    if (target.shape[0] - ref_len + 1) <= 0:
        return np.array([])
    return np.array([np.mean(np.sum(ref * target[t : t + ref_len], axis=1)) for t in range(target.shape[0] - ref_len + 1)])


def find_bumper_template(chroma_a: np.ndarray, chroma_b: np.ndarray, cfg: BumperConfig, mask_a: np.ndarray | None = None, mask_b: np.ndarray | None = None):
    """
    Find the best matching segment between chroma_a and chroma_b, skipping masked frames.
    Tries all durations in [cfg.min_bumper_sec, cfg.max_bumper_sec].
    Returns (template, start_a_sec, start_b_sec, duration_sec, score) or None if score < cfg.match_threshold.
    """
    best = None
    fps = cfg.chroma_fps

    for dur_sec in range(cfg.min_bumper_sec, cfg.max_bumper_sec + 1):
        win = dur_sec * fps
        if win > chroma_a.shape[0] or win > chroma_b.shape[0]:
            continue

        for cur_a_idx in range(0, chroma_a.shape[0] - win, fps):
            if mask_a is not None and np.any(mask_a[cur_a_idx : cur_a_idx + win]):
                continue

            seg = chroma_a[cur_a_idx : cur_a_idx + win]
            scores_b = slide_score(seg, chroma_b)
            if len(scores_b) == 0:
                continue

            if mask_b is not None:
                for tb_idx in range(len(scores_b)):
                    if np.any(mask_b[tb_idx : tb_idx + win]):
                        scores_b[tb_idx] = 0.0

            best_b_idx = int(np.argmax(scores_b))
            score = float(scores_b[best_b_idx])

            if score >= cfg.match_threshold and (best is None or score > best[4]):
                best = (seg.copy(), cur_a_idx / fps, best_b_idx / fps, dur_sec, score)
    return best


def find_all_templates(chromas: dict[str, np.ndarray], video_dirs: list[str], cfg: BumperConfig, max_types: int = 10):
    """
    Iteratively discover distinct bumper templates across all video pairs until none remain above threshold.
    max_types is a safety cap only — the algorithm stops naturally when no new template is found.
    Returns list of (template_chroma, duration_sec, score, (vid_i, vid_j)).
    """
    fps = cfg.chroma_fps
    masks = {vid: np.zeros(chromas[vid].shape[0], dtype=bool) for vid in video_dirs}
    templates = []

    for type_idx in range(max_types):
        best_result, best_pair = None, None

        for i in range(len(video_dirs)):
            for j in range(i + 1, len(video_dirs)):
                vi, vj = video_dirs[i], video_dirs[j]
                result = find_bumper_template(chromas[vi], chromas[vj], cfg, mask_a=masks[vi], mask_b=masks[vj])
                if result and (best_result is None or result[4] > best_result[4]):
                    best_result, best_pair = result, (vi, vj)

        if best_result is None:
            logger.info("No more templates found — detected %d type(s) total", type_idx)
            break

        template, sa, sb, dur_sec, score = best_result
        vi, vj = best_pair
        win = dur_sec * fps
        logger.info("Type %d: %.1fs, score=%.3f (from %s@%.1fs + %s@%.1fs)", type_idx, dur_sec, score, vi, sa, vj, sb)

        mask_r = int(cfg.mask_radius_sec * fps)
        for vid in video_dirs:
            scores = slide_score(template, chromas[vid])
            if len(scores) == 0:
                continue
            for idx in range(len(scores)):
                if np.any(masks[vid][idx : idx + win]):
                    scores[idx] = 0.0
            best_pos = int(np.argmax(scores))
            if scores[best_pos] >= cfg.match_threshold * cfg.mask_score_ratio:
                masks[vid][max(0, best_pos - mask_r) : min(len(masks[vid]), best_pos + win + mask_r)] = True

        templates.append((template, dur_sec, score, best_pair))

    return templates


def locate_in_video(template: np.ndarray, chroma: np.ndarray, cfg: BumperConfig) -> list[tuple[float, float]]:
    """
    Find up to cfg.locate_max_per_video non-overlapping matches of template in chroma.
    Each match must exceed cfg.locate_threshold; found positions are masked before the next search.
    Returns list of (start_sec, score).
    """
    scores = slide_score(template, chroma)
    if len(scores) == 0:
        return []

    fps = cfg.chroma_fps
    win = template.shape[0]
    results = []

    for _ in range(cfg.locate_max_per_video):
        idx = int(np.argmax(scores))
        if scores[idx] < cfg.locate_threshold:
            break
        results.append((idx / fps, float(scores[idx])))
        scores[max(0, idx - int(cfg.mask_radius_sec * fps)) : min(len(scores), idx + win + int(cfg.mask_radius_sec * fps))] = 0.0

    return results


def run_audio_pipeline(data_dir: str, cfg: BumperConfig, max_types: int = 10) -> tuple[list[BumpCandidate], dict[str, str]]:
    """
    Full audio-based bumper detection pipeline.
    Returns (candidates, video_paths); candidates contain all located bumper intervals before any video verification.
    The number of bumper types is detected automatically; max_types is a safety cap.
    """
    video_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, d, "video.mp4"))])
    if len(video_dirs) < 2:
        logger.error("Need at least 2 videos, found %d", len(video_dirs))
        return [], {}

    video_paths = {d: os.path.join(data_dir, d, "video.mp4") for d in video_dirs}
    logger.info("Found %d videos", len(video_dirs))

    logger.info("Extracting audio fingerprints...")
    chromas = {}
    for vid in video_dirs:
        dur = get_video_duration(video_paths[vid])
        chromas[vid] = extract_chroma(video_paths[vid], cfg, max_sec=dur * cfg.scan_ratio)
        logger.info("  %s: %.0fs scanned of %.0fs", vid, dur * cfg.scan_ratio, dur)

    templates = find_all_templates(chromas, video_dirs, cfg, max_types=max_types)
    if not templates:
        logger.info("No bumpers found.")
        return [], video_paths

    logger.info("Found %d bumper type(s). Locating in all videos...", len(templates))

    candidates: list[BumpCandidate] = []
    for type_idx, (template, dur_sec, _, _) in enumerate(templates):
        for vid in video_dirs:
            for start_sec, score in locate_in_video(template, chromas[vid], cfg):
                candidates.append(
                    BumpCandidate(
                        video_id=vid, bumper_type=type_idx, start_sec=round(start_sec, 2), end_sec=round(start_sec + dur_sec, 2), duration_sec=dur_sec, audio_score=round(score, 3)
                    )
                )
                logger.info("  [type %d] %s: %.1f–%.1fs  audio_score=%.3f", type_idx, vid, start_sec, start_sec + dur_sec, score)

    # Discard types present in fewer than min_candidate_videos unique videos.
    # A real repeating bumper must appear in multiple episodes; single-video hits are FPs.
    type_videos: dict[int, set[str]] = defaultdict(set)
    for c in candidates:
        type_videos[c.bumper_type].add(c.video_id)

    before = len(candidates)
    candidates = [c for c in candidates if len(type_videos[c.bumper_type]) >= cfg.min_candidate_videos]
    removed_types = {t for t, vids in type_videos.items() if len(vids) < cfg.min_candidate_videos}

    if removed_types:
        logger.info(
            "Audio post-filter: removed types %s (each found in < %d video(s)); %d → %d candidates", sorted(removed_types), cfg.min_candidate_videos, before, len(candidates)
        )

    return candidates, video_paths
