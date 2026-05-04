from collections import defaultdict

import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from ..utils.logger import Logger
from .configs import BumpCandidate, VideoVerifierConfig


logger = Logger(show=True).get_logger()


class VideoVerifier:
    def __init__(self, cfg: VideoVerifierConfig):
        device_str = cfg.resolved_device()
        self.device = torch.device(device_str)
        self.cfg = cfg
        logger.info("Loading CLIP model '%s' on %s...", cfg.clip_model, device_str)
        self.processor = CLIPProcessor.from_pretrained(cfg.clip_model, use_fast=False)
        self.model = CLIPModel.from_pretrained(cfg.clip_model).to(self.device).eval()

    def _sample_frames(self, video_path: str, start_sec: float, end_sec: float) -> list[np.ndarray]:
        """
        Sample cfg.n_frames evenly across [start_sec, end_sec] and return them as RGB arrays.
        Skips unreadable frames; returns empty list if none could be read.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        for t in np.linspace(start_sec, end_sec, self.cfg.n_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    @torch.no_grad()
    def _embed_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Compute L2-normalised CLIP image embeddings for a batch of RGB frames.
        Returns array of shape (N, embedding_dim).
        """
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.cpu().numpy()

    def _embed_interval(self, video_path: str, start_sec: float, end_sec: float) -> np.ndarray | None:
        """
        Return a single L2-normalised embedding that represents the full [start_sec, end_sec] interval.
        Samples cfg.n_frames frames evenly, computes CLIP embeddings, then mean-pools them.
        Mean pooling is more robust than a single frame: a bumper looks consistent across all its frames,
        so its mean embedding is stable; a random non-bumper interval has incoherent frames that
        average out and produce a weaker, less distinctive embedding.
        Returns None if fewer than 2 frames could be read.
        """
        frames = self._sample_frames(video_path, start_sec, end_sec)
        if len(frames) < 2:
            return None
        embeddings = self._embed_frames(frames)  # (n_frames, dim)
        mean_emb = embeddings.mean(axis=0)  # (dim,)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        return mean_emb

    def _mean_pairwise_similarity(self, embeddings: np.ndarray) -> float:
        """
        Mean cosine similarity over all unique pairs in the embedding matrix.
        For N=1 returns 1.0 (single video, no cross-video check possible).
        """
        n = embeddings.shape[0]
        if n < 2:
            return 1.0
        sim = embeddings @ embeddings.T
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return float(np.mean([sim[i, j] for i, j in pairs]))

    def verify(self, candidates: list[BumpCandidate], video_paths: dict[str, str]) -> list[BumpCandidate]:
        """
        Filter candidates by cross-video visual similarity using CLIP interval embeddings.
        For each bumper_type: samples cfg.n_frames per video, mean-pools them into one embedding,
        then checks that the mean pairwise cosine similarity across videos >= cfg.visual_threshold.
        Returns filtered list of BumpCandidate.
        """
        by_type: dict[int, list[BumpCandidate]] = defaultdict(list)
        for c in candidates:
            by_type[c.bumper_type].append(c)

        verified: list[BumpCandidate] = []

        for bumper_type, group in sorted(by_type.items()):
            n_videos = len({c.video_id for c in group})

            if n_videos < self.cfg.min_videos_agree:
                logger.info("Type %d: only %d video(s) — REJECTED (not cross-video confirmed)", bumper_type, n_videos)
                continue

            per_video_first: dict[str, BumpCandidate] = {}
            for c in group:
                if c.video_id not in per_video_first:
                    per_video_first[c.video_id] = c

            interval_embeddings, valid_vids = [], []
            for vid, c in per_video_first.items():
                emb = self._embed_interval(video_paths[vid], c.start_sec, c.end_sec)
                if emb is not None:
                    interval_embeddings.append(emb)
                    valid_vids.append(vid)
                    logger.info("  Type %d  %s [%.1f–%.1fs]: sampled %d frames", bumper_type, vid, c.start_sec, c.end_sec, self.cfg.n_frames)

            if len(interval_embeddings) < self.cfg.min_videos_agree:
                logger.warning("Type %d: not enough readable intervals — REJECTED", bumper_type)
                continue

            sim = self._mean_pairwise_similarity(np.stack(interval_embeddings))

            if sim >= self.cfg.visual_threshold:
                logger.info("Type %d: visual_sim=%.3f >= %.2f — CONFIRMED (%d candidates, videos: %s)", bumper_type, sim, self.cfg.visual_threshold, len(group), valid_vids)
                verified.extend(group)
            else:
                logger.info("Type %d: visual_sim=%.3f < %.2f — REJECTED as FP (videos: %s)", bumper_type, sim, self.cfg.visual_threshold, valid_vids)

        logger.info("Video verification: %d / %d candidates kept", len(verified), len(candidates))
        return verified
