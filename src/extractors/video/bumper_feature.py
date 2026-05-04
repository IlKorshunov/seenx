import os

import cv2
import numpy as np

from ...utils.config import Config
from ...utils.logger import Logger
from ..feature_extractor import VideoFeature
from .common import get_capture_fps, open_video_capture
from .constants import *

logger = Logger(show=True).get_logger()


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / float(np.linalg.norm(x) + 1e-8)


class BumperFeature(VideoFeature):
    WINDOW_SEC = BUMPER_WINDOW_SEC
    STEP_SEC = BUMPER_STEP_SEC
    MIN_TRIM_SEC = BUMPER_MIN_TRIM_SEC

    HIST_BINS = BUMPER_HIST_BINS
    CROSS_THRESH = BUMPER_CROSS_THRESH
    MAX_BUMPER_SEC = BUMPER_MAX_SEC

    SINGLE_BOUNDARY_PERCENTILE = BUMPER_SINGLE_BOUNDARY_PERCENTILE
    SINGLE_MIN_SEC = BUMPER_SINGLE_MIN_SEC
    SINGLE_MAX_SEC = BUMPER_SINGLE_MAX_SEC
    SINGLE_TOP_K = BUMPER_SINGLE_TOP_K

    def __init__(self, config: Config, data_dir: str = "data"):
        self.config = config
        self.data_dir = data_dir
        self.reference_path = config.get("bumper_reference_path", None)

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"bumper_score"}

    def _frame_hist(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [self.HIST_BINS[0], self.HIST_BINS[1]], [0, 180, 0, 256]).astype(np.float32)
        h = h.flatten()
        return _l2_normalize(h)

    def _sample_hists(self, video_path: str, step_sec: float = 1.0) -> tuple[np.ndarray, float]:
        hists = []
        with open_video_capture(video_path) as cap:
            fps = get_capture_fps(cap)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_sec = total_frames / fps if fps > 0 else 0

            for sec in np.arange(0, total_sec, step_sec):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
                ok, frame = cap.read()
                if not ok:
                    break
                hists.append(self._frame_hist(frame))

        if not hists:
            return np.zeros((0, self.HIST_BINS[0] * self.HIST_BINS[1]), dtype=np.float32), fps
        return np.stack(hists, axis=0), fps

    @staticmethod
    def _window_means(H: np.ndarray, win: int, step: int) -> tuple[np.ndarray, np.ndarray]:
        n, d = H.shape
        if n < win:
            return np.array([], dtype=np.int32), np.zeros((0, d), dtype=np.float32)
        starts = np.arange(0, n - win + 1, step, dtype=np.int32)
        c = np.vstack([np.zeros((1, d), dtype=np.float32), np.cumsum(H, axis=0)])
        sums = c[starts + win] - c[starts]
        means = sums / float(win)
        norms = np.linalg.norm(means, axis=1, keepdims=True) + 1e-8
        means = means / norms
        return starts, means.astype(np.float32)

    def _best_alignment(self, seg: np.ndarray, other: np.ndarray, step: int) -> float:
        L = seg.shape[0]
        if L == 0 or other.shape[0] < L:
            return 0.0
        best = 0.0
        for t in range(0, other.shape[0] - L + 1, step):
            s = float((seg * other[t : t + L]).sum(axis=1).mean())
            if s > best:
                best = s
        return best

    def _segment_score_cross(self, seg: np.ndarray, others: list[np.ndarray]) -> float:
        if not others:
            return 0.0
        scores = [self._best_alignment(seg, oh, step=self.STEP_SEC) for oh in others]
        return float(np.mean(scores)) if scores else 0.0

    def _trim_edges(self, H: np.ndarray, start: int, end: int, others: list[np.ndarray]) -> tuple[int, int]:
        while end - start > self.MIN_TRIM_SEC:
            trimmed = False

            if end - (start + 1) >= self.MIN_TRIM_SEC:
                seg = H[start + 1 : end]
                if self._segment_score_cross(seg, others) >= self.CROSS_THRESH:
                    start += 1
                    trimmed = True

            if (end - 1) - start >= self.MIN_TRIM_SEC:
                seg = H[start : end - 1]
                if self._segment_score_cross(seg, others) >= self.CROSS_THRESH:
                    end -= 1
                    trimmed = True

            if not trimmed:
                break
        return start, end

    @staticmethod
    def _merge_intervals(hits: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        if not hits:
            return []
        hits = sorted(hits, key=lambda x: x[0])
        merged = [list(hits[0])]
        for s, e, sc in hits[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
                merged[-1][2] = max(merged[-1][2], sc)
            else:
                merged.append([s, e, sc])
        return [(a, b, float(c)) for a, b, c in merged]

    def _detect_by_reference(self, H: np.ndarray, ref_H: np.ndarray) -> np.ndarray:
        n = H.shape[0]
        L = ref_H.shape[0]
        if L < self.MIN_TRIM_SEC or n < L:
            return np.zeros(n, dtype=np.float32)

        scores = np.zeros(n, dtype=np.float32)
        for t in range(0, n - L + 1):
            s = float((H[t : t + L] * ref_H).sum(axis=1).mean())
            scores[t : t + L] = np.maximum(scores[t : t + L], s)

        scores = np.clip((scores - self.CROSS_THRESH) / (1.0 - self.CROSS_THRESH + 1e-6), 0, 1)
        logger.info("Reference bumper matching: max_score=%.3f", float(scores.max()))
        return scores

    def _detect_cross_video(self, H: np.ndarray, video_path: str) -> np.ndarray | None:
        other_paths = []
        for name in sorted(os.listdir(self.data_dir)):
            vp = os.path.join(self.data_dir, name, "video.mp4")
            if os.path.isfile(vp) and os.path.abspath(vp) != os.path.abspath(video_path):
                other_paths.append(vp)

        if not other_paths:
            return None
        other_H = []
        for vp in other_paths:
            oh, _ = self._sample_hists(vp, step_sec=1.0)
            if oh.shape[0] > 0:
                other_H.append(oh)

        if not other_H:
            return None

        n = H.shape[0]
        win = min(self.WINDOW_SEC, n)

        starts, cur_means = self._window_means(H, win, self.STEP_SEC)
        if cur_means.shape[0] == 0:
            return np.zeros(n, dtype=np.float32)

        other_means = [self._window_means(oh, win, self.STEP_SEC)[1] for oh in other_H]
        other_means = [m for m in other_means if m.shape[0] > 0]
        if not other_means:
            return None

        per_other_best = []
        for om in other_means:
            per_other_best.append((cur_means @ om.T).max(axis=1))
        mean_best = np.mean(np.stack(per_other_best, axis=0), axis=0)

        hits = []
        for i, s in enumerate(starts):
            sc = float(mean_best[i])
            if sc >= self.CROSS_THRESH:
                hits.append((int(s), int(s + win), sc))

        if not hits:
            logger.info("Cross-video: no bumper found (checked %d other videos)", len(other_means))
            return np.zeros(n, dtype=np.float32)

        merged = self._merge_intervals(hits)
        bumper = np.zeros(n, dtype=np.float32)
        for s, e, sc in merged:
            rs, re = self._trim_edges(H, s, e, other_H)
            duration = re - rs
            if duration > self.MAX_BUMPER_SEC:
                logger.info("Skip segment %d–%d sec (%ds > max %ds)", rs, re, duration, self.MAX_BUMPER_SEC)
                continue
            bumper[rs:re] = max(float(sc), float(bumper[rs:re].max(initial=0.0)))
            logger.info("Bumper: %d–%d sec (%ds, from %d–%d), score=%.3f", rs, re, duration, s, e, sc)

        bumper = np.clip((bumper - self.CROSS_THRESH) / (1.0 - self.CROSS_THRESH + 1e-6), 0, 1)
        return bumper

    def _detect_single_video(self, H: np.ndarray) -> np.ndarray:
        n = H.shape[0]
        bumper = np.zeros(n, dtype=np.float32)
        if n < self.SINGLE_MIN_SEC * 3:
            return bumper

        sim_prev = (H[1:] * H[:-1]).sum(axis=1)
        boundary = 1.0 - sim_prev
        boundary = np.clip(boundary, 0, 2).astype(np.float32)

        if len(boundary) >= 5:
            kernel = np.ones(5, dtype=np.float32) / 5.0
            boundary = np.convolve(boundary, kernel, mode="same")

        thr = float(np.percentile(boundary, self.SINGLE_BOUNDARY_PERCENTILE))
        cut_idx = np.where(boundary >= thr)[0] + 1
        cuts = [0] + cut_idx.tolist() + [n]
        cuts = sorted(set([c for c in cuts if 0 <= c <= n]))

        segs = []
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            L = b - a
            if self.SINGLE_MIN_SEC <= L <= self.SINGLE_MAX_SEC:
                left_b = boundary[a - 1] if a - 1 >= 0 and a - 1 < len(boundary) else 0.0
                right_b = boundary[b - 1] if b - 1 >= 0 and b - 1 < len(boundary) else 0.0
                score = float(left_b + right_b)
                segs.append((a, b, score))

        segs = sorted(segs, key=lambda x: x[2], reverse=True)[: self.SINGLE_TOP_K]
        if not segs:
            logger.info("Single-video bumper detection: no segments found")
            return bumper

        max_score = max(s[2] for s in segs) + 1e-6
        for a, b, sc in segs:
            bumper[a:b] = max(bumper[a:b].max(initial=0.0), sc / max_score)
            logger.info("Single-video bumper segment: %d–%d sec, score=%.3f", a, b, sc / max_score)

        return bumper

    def run(self, video_path, context):
        df = context["data"]
        n_frames = len(df)

        H, fps = self._sample_hists(video_path, step_sec=1.0)
        n_sec = H.shape[0]

        bumper_sec = None

        if self.reference_path and os.path.isfile(self.reference_path):
            ref_H, _ = self._sample_hists(self.reference_path, step_sec=1.0)
            if ref_H.shape[0] > 0:
                bumper_sec = self._detect_by_reference(H, ref_H)
                logger.info("Using reference bumper detection (%s)", self.reference_path)

        if bumper_sec is None:
            cross = self._detect_cross_video(H, video_path)
            if cross is not None and np.any(cross > 0):
                bumper_sec = cross
                logger.info("Using cross-video bumper detection")
            else:
                bumper_sec = self._detect_single_video(H)
                logger.info("Using single-video bumper detection (fallback)")

        if n_sec == 0:
            df["bumper_score"] = 0.0
            return

        bumper_frame = np.interp(np.linspace(0, n_sec - 1, n_frames), np.arange(n_sec), bumper_sec.astype(np.float64))
        df["bumper_score"] = bumper_frame
