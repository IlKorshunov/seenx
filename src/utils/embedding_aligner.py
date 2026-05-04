"""
Align multimodal embeddings to 1 fps and optional projection to common space.

Current extraction rates (разная частота):
  - Visual (CLIP):  ~1 fps   (step=FPS → 1 frame/sec), dim 768
  - Audio (CLAP):   ~0.2 fps (chunk 5 sec), dim 512
  - Text (USER2):   variable (Whisper segments), dim 256

This module resamples all to 1 fps and optionally projects to d_model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np


EMBEDDINGS_ROOT = "embeddings"
VISUAL_DIM = 768
AUDIO_DIM = 512
TEXT_DIM = 256
AUDIO_CHUNK_SEC = 1


def _load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return None


def _load_seg_meta(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def resample_visual_to_1fps(embs: np.ndarray, duration_sec: int) -> np.ndarray:
    """Visual: typically ~1 fps already. Resample to exact duration via linear interpolation."""
    n = len(embs)
    if n == 0:
        return np.zeros((duration_sec, embs.shape[1]), dtype=np.float32)
    if n == duration_sec:
        return embs.astype(np.float32)
    # Map frame indices [0..n-1] to seconds [0..duration_sec-1]
    x_old = np.linspace(0, duration_sec - 1, n, dtype=np.float32)
    x_new = np.arange(duration_sec, dtype=np.float32)
    out = np.zeros((duration_sec, embs.shape[1]), dtype=np.float32)
    for d in range(embs.shape[1]):
        out[:, d] = np.interp(x_new, x_old, embs[:, d].astype(np.float32))
    return out


def resample_audio_to_1fps(embs: np.ndarray, duration_sec: int) -> np.ndarray:
    """
    Audio: 1 embedding per 5 sec. Upsample via linear interpolation.
    Chunk i covers seconds [i*5, (i+1)*5). We interpolate between chunk centers
    to get smooth transitions instead of piecewise-constant replication.
    """
    n = len(embs)
    if n == 0:
        return np.zeros((duration_sec, embs.shape[1]), dtype=np.float32)
    # Chunk centers in seconds: 2.5, 7.5, 12.5, ...
    x_old = (np.arange(n, dtype=np.float32) + 0.5) * AUDIO_CHUNK_SEC
    x_new = np.arange(duration_sec, dtype=np.float32) + 0.5  # second centers
    out = np.zeros((duration_sec, embs.shape[1]), dtype=np.float32)
    for d in range(embs.shape[1]):
        out[:, d] = np.interp(x_new, x_old, embs[:, d].astype(np.float32))
    return out


def resample_text_to_1fps(embs: np.ndarray, seg_meta: list[dict], duration_sec: int) -> np.ndarray:
    """
    Text: variable-length segments. For each second t, take weighted average
    of segment embeddings that overlap t (by overlap duration).
    """
    n_seg = len(embs)
    if n_seg == 0:
        return np.zeros((duration_sec, embs.shape[1]), dtype=np.float32)
    out = np.zeros((duration_sec, embs.shape[1]), dtype=np.float32)
    for t in range(duration_sec):
        t_start, t_end = float(t), float(t + 1)
        weights = []
        embs_sub = []
        for i, seg in enumerate(seg_meta[:n_seg]):
            s, e = float(seg.get("start", 0)), float(seg.get("end", 0))
            overlap = max(0, min(t_end, e) - max(t_start, s))
            if overlap > 1e-6:
                weights.append(overlap)
                embs_sub.append(embs[i])
        if not weights:
            # No segment overlaps (silence/pause) — zero out
            out[t] = 0.0
        else:
            weights = np.array(weights, dtype=np.float32)
            weights /= weights.sum()
            out[t] = np.average(embs_sub, axis=0, weights=weights)
    return out


def load_aligned_embeddings(
    video_id: str,
    embeddings_root: str | Path = EMBEDDINGS_ROOT,
    duration_sec: int | None = None,
    modalities: tuple[Literal["visual", "audio", "text"], ...] = ("visual", "audio", "text"),
) -> tuple[np.ndarray, int]:
    """
    Load and align embeddings to 1 fps.

    Returns:
        aligned: (T, D) where D = sum of modality dims (768+512+256=1536 by default)
        duration_sec: T (inferred from data if not provided)
    """
    root = Path(embeddings_root) / video_id
    # Determine duration first
    if duration_sec is None:
        inferred = 0
        if "visual" in modalities:
            v = _load_npy(root / "visual_embeddings.npy")
            if v is not None:
                inferred = max(inferred, len(v))
        if "audio" in modalities:
            a = _load_npy(root / "audio_embeddings.npy")
            if a is not None:
                inferred = max(inferred, len(a) * AUDIO_CHUNK_SEC)
        if "text" in modalities:
            seg_meta = _load_seg_meta(root / "seg_meta.json")
            if seg_meta:
                inferred = max(inferred, int(np.ceil(max(s.get("end", 0) for s in seg_meta))))
        duration_sec = max(inferred, 1)

    parts = []

    if "visual" in modalities:
        v = _load_npy(root / "visual_embeddings.npy")
        if v is not None:
            parts.append(("visual", resample_visual_to_1fps(v.astype(np.float32), duration_sec)))
        else:
            parts.append(("visual", np.zeros((duration_sec, VISUAL_DIM), dtype=np.float32)))

    if "audio" in modalities:
        a = _load_npy(root / "audio_embeddings.npy")
        if a is not None:
            parts.append(("audio", resample_audio_to_1fps(a.astype(np.float32), duration_sec)))
        else:
            parts.append(("audio", np.zeros((duration_sec, AUDIO_DIM), dtype=np.float32)))

    if "text" in modalities:
        t_embs = _load_npy(root / "seg_embeddings.npy")
        seg_meta = _load_seg_meta(root / "seg_meta.json")
        if t_embs is not None:
            parts.append(("text", resample_text_to_1fps(t_embs.astype(np.float32), seg_meta, duration_sec)))
        else:
            parts.append(("text", np.zeros((duration_sec, TEXT_DIM), dtype=np.float32)))

    if not parts:
        return np.zeros((duration_sec, 0), dtype=np.float32), duration_sec

    # Concatenate: (T, 768+512+256) or subset
    out = np.concatenate([p[1] for p in parts], axis=1)
    return out, duration_sec


def normalize_l2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2 normalize for cosine similarity."""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.clip(n, 1e-9, None)
    return (x / n).astype(np.float32)


def cosine_between_modalities(vis: np.ndarray, aud: np.ndarray, txt: np.ndarray) -> np.ndarray:
    """
    Compute cross-modal cosine similarity per timestep.
    Returns (T, 3): [vis_aud, vis_txt, aud_txt].
    """
    vis = normalize_l2(vis)
    aud = normalize_l2(aud)
    txt = normalize_l2(txt)
    t = vis.shape[0]
    out = np.zeros((t, 3), dtype=np.float32)
    out[:, 0] = (vis * aud).sum(axis=1)
    out[:, 1] = (vis * txt).sum(axis=1)
    out[:, 2] = (aud * txt).sum(axis=1)
    return out


class PerModalityPCA:
    """
    Fit PCA separately per modality, then concatenate reduced representations.
    visual (768) → n_comp, audio (512) → n_comp, text (256) → n_comp → total 3*n_comp.
    """

    def __init__(self, n_components: int = 12):
        from sklearn.decomposition import PCA

        self.n_components = n_components
        self._pcas = {
            "visual": PCA(n_components=n_components, random_state=42),
            "audio": PCA(n_components=n_components, random_state=42),
            "text": PCA(n_components=n_components, random_state=42),
        }
        self._fitted = False

    @staticmethod
    def _split(aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (aligned[..., :VISUAL_DIM], aligned[..., VISUAL_DIM : VISUAL_DIM + AUDIO_DIM], aligned[..., VISUAL_DIM + AUDIO_DIM :])

    def fit(self, aligned: np.ndarray) -> PerModalityPCA:
        """Fit on training data (T_total, 1536) — concatenation of all train videos."""
        vis, aud, txt = self._split(aligned)
        for name, data in [("visual", vis), ("audio", aud), ("text", txt)]:
            n_comp = min(self.n_components, data.shape[0], data.shape[1])
            self._pcas[name].n_components = n_comp
            self._pcas[name].fit(data)
        self._fitted = True
        return self

    def transform(self, aligned: np.ndarray) -> np.ndarray:
        """Transform (T, 1536) → (T, 3*n_components)."""
        assert self._fitted, "Call .fit() first"
        vis, aud, txt = self._split(aligned)
        return np.concatenate([self._pcas["visual"].transform(vis), self._pcas["audio"].transform(aud), self._pcas["text"].transform(txt)], axis=1).astype(np.float32)

    def fit_transform(self, aligned: np.ndarray) -> np.ndarray:
        return self.fit(aligned).transform(aligned)

    def explained_variance_ratio(self) -> dict[str, np.ndarray]:
        return {k: pca.explained_variance_ratio_ for k, pca in self._pcas.items()}


def get_alignment_features(video_id: str, embeddings_root: str | Path = EMBEDDINGS_ROOT, duration_sec: int | None = None) -> np.ndarray | None:
    """
    Lightweight: compute only cross-modal alignment scalars (T, 3).
    Use when you don't need full embeddings, just alignment scores.
    """
    root = Path(embeddings_root) / video_id
    v = _load_npy(root / "visual_embeddings.npy")
    a = _load_npy(root / "audio_embeddings.npy")
    t_embs = _load_npy(root / "seg_embeddings.npy")
    seg_meta = _load_seg_meta(root / "seg_meta.json")

    if v is None or a is None or t_embs is None:
        return None

    dur = duration_sec or max(len(v), len(a) * AUDIO_CHUNK_SEC)
    v1 = resample_visual_to_1fps(v.astype(np.float32), dur)
    a1 = resample_audio_to_1fps(a.astype(np.float32), dur)
    t1 = resample_text_to_1fps(t_embs.astype(np.float32), seg_meta, dur)

    return cosine_between_modalities(v1, a1, t1)
