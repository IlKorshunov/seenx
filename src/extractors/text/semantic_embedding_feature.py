"""Semantic embedding features via deepvk/USER2-base.
Saves per-segment embeddings to embeddings/<video_id>/
semantic_novelty, topic_shift, hook_similarity,
global_topic_dist, semantic_momentum, segment_self_similarity
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from ._base import get_segments_and_duration, seg_bounds, skip_if_exists
from .common import release_models, video_id


MODEL_ID = "deepvk/USER2-base"
EMB_DIM = 256
EMBEDDINGS_ROOT = "embeddings"

_COLS = {"semantic_novelty", "topic_shift", "hook_similarity", "global_topic_dist", "semantic_momentum", "segment_self_similarity"}


def _encode_texts(texts: list[str], model, tokenizer, device, batch_size: int = 128) -> np.ndarray:
    all_embeddings = []
    for batch_start in range(0, len(texts), batch_size):
        encoded_inputs = tokenizer(texts[batch_start : batch_start + batch_size], padding=True, truncation=True, max_length=8192, return_tensors="pt").to(device)
        with torch.no_grad():
            model_output = model(**encoded_inputs)
        attention_mask = encoded_inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (model_output.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled[:, :EMB_DIM], p=2, dim=1)
        all_embeddings.append(pooled.cpu().float().numpy())
    return np.vstack(all_embeddings)


def _save(video_path: str, embeddings: np.ndarray, segments: list[dict]) -> None:
    out_dir = os.path.join(EMBEDDINGS_ROOT, video_id(video_path))
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "seg_embeddings.npy"), embeddings)
    np.save(os.path.join(out_dir, "seg_similarity_matrix.npy"), embeddings @ embeddings.T)
    with open(os.path.join(out_dir, "seg_meta.json"), "w", encoding="utf-8") as metadata_file:
        json.dump([{"start": round(float(segment["start"]), 2), "end": round(float(segment["end"]), 2)} for segment in segments], metadata_file, ensure_ascii=False, indent=2)


def _derive(embeddings: np.ndarray) -> dict[str, np.ndarray]:
    embedding_count = len(embeddings)
    sims = embeddings @ embeddings.T
    global_mean = embeddings.mean(axis=0)
    global_mean /= max(np.linalg.norm(global_mean), 1e-9)
    hook = embeddings[: max(1, min(embedding_count, 3))].mean(axis=0)
    hook /= max(np.linalg.norm(hook), 1e-9)

    novelty = np.ones(embedding_count)
    for segment_idx in range(1, embedding_count):
        novelty[segment_idx] = 1.0 - float(np.mean(sims[segment_idx, :segment_idx]))

    shift = np.zeros(embedding_count)
    shift[1:] = 1.0 - np.diag(sims, k=-1)

    hook_similarity = embeddings @ hook
    global_topic_dist = 1.0 - (embeddings @ global_mean)
    semantic_momentum = np.array([np.mean(shift[max(0, segment_idx - 2) : segment_idx + 1]) for segment_idx in range(embedding_count)])

    segment_self_similarity = np.zeros(embedding_count)
    for segment_idx in range(embedding_count):
        neighbors = embeddings[max(0, segment_idx - 1) : segment_idx] if segment_idx == embedding_count - 1 else embeddings[segment_idx + 1 : segment_idx + 2]
        if 0 < segment_idx < embedding_count - 1:
            neighbors = np.vstack([embeddings[segment_idx - 1], embeddings[segment_idx + 1]])
        neighbor_mean = neighbors.mean(axis=0)
        neighbor_mean /= max(np.linalg.norm(neighbor_mean), 1e-9)
        segment_self_similarity[segment_idx] = float(np.dot(embeddings[segment_idx], neighbor_mean))

    return {
        "semantic_novelty": novelty,
        "topic_shift": shift,
        "hook_similarity": hook_similarity,
        "global_topic_dist": global_topic_dist,
        "semantic_momentum": semantic_momentum,
        "segment_self_similarity": segment_self_similarity,
    }


def extract_semantic_embeddings(video_path, config, existing_features=None) -> pd.DataFrame:
    if skip_if_exists(_COLS, existing_features, "semantic embeddings"):
        return pd.DataFrame()

    segments, duration = get_segments_and_duration(video_path, config)
    valid = [(seg, *seg_bounds(seg, duration)) for seg in segments if seg.get("text", "").strip()]
    valid = [(segment, start_sec, end_sec) for segment, start_sec, end_sec in valid if start_sec < end_sec]
    if not valid:
        return pd.DataFrame({column: np.zeros(duration) for column in sorted(_COLS)})

    device = torch.device(config.get("device"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()

    embeddings = _encode_texts([seg["text"].strip() for seg, _, _ in valid], model, tokenizer, device)
    _save(video_path, embeddings, [seg for seg, _, _ in valid])

    release_models(model, tokenizer, device=device)

    cols = sorted(_COLS)
    derived = _derive(embeddings)
    out = np.zeros((duration, len(cols)), dtype=np.float64)
    for segment_idx, (_, start_sec, end_sec) in enumerate(valid):
        out[start_sec:end_sec] = [derived[column][segment_idx] for column in cols]

    return pd.DataFrame(out, columns=cols)
