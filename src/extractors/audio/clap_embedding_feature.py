import gc
import math
import os

import numpy as np
import pandas as pd
import torch
from transformers import ClapModel, ClapProcessor

from ...seenx_utils import get_video_duration
from ...utils.config import Config
from ...utils.logger import Logger
from .common import embeddings_dir, load_video_audio
from .consts import CHUNK_SEC, CLAP_AUDIO_SR, CLAP_MODEL_ID


logger = Logger(show=True).get_logger()
CLAP_AUDIO_COLS = {"audio_novelty", "audio_topic_shift", "audio_hook_similarity", "audio_global_dist", "audio_momentum", "audio_self_similarity"}


def _encode_chunks(waveform: np.ndarray, sample_rate: int, model, processor, device, batch_size: int = 4) -> np.ndarray:
    chunk_samples = CHUNK_SEC * sample_rate
    chunks = [waveform[start : start + chunk_samples] for start in range(0, max(len(waveform), 1), chunk_samples)]
    chunks = [np.pad(chunk, (0, max(0, sample_rate - len(chunk)))) if len(chunk) < sample_rate else chunk for chunk in chunks]
    embeddings = []
    for batch_start in range(0, len(chunks), batch_size):
        inputs = processor(audios=chunks[batch_start : batch_start + batch_size], sampling_rate=sample_rate, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = torch.nn.functional.normalize(model.get_audio_features(**inputs), p=2, dim=1)
        embeddings.append(features.cpu().float().numpy())
    return np.vstack(embeddings)


def _derive(embeddings: np.ndarray) -> dict[str, np.ndarray]:
    n_chunks = len(embeddings)
    similarities = embeddings @ embeddings.T
    global_mean = embeddings.mean(axis=0)
    global_mean /= max(float(np.linalg.norm(global_mean)), 1e-9)
    hook_mean = embeddings[: min(n_chunks, 30)].mean(axis=0)
    hook_mean /= max(float(np.linalg.norm(hook_mean)), 1e-9)
    novelty = np.array([1.0] + [1.0 - float(np.mean(similarities[index, :index])) for index in range(1, n_chunks)])
    shift = np.zeros(n_chunks)
    shift[1:] = 1.0 - np.diag(similarities, k=-1)

    def normalized_neighbor_mean(index: int) -> np.ndarray:
        neighbor_mean = embeddings[max(0, index - 1) : index + 2].mean(axis=0)
        return neighbor_mean / max(float(np.linalg.norm(neighbor_mean)), 1e-9)

    return {
        "audio_novelty": novelty,
        "audio_topic_shift": shift,
        "audio_hook_similarity": embeddings @ hook_mean,
        "audio_global_dist": 1.0 - (embeddings @ global_mean),
        "audio_momentum": np.array([np.mean(shift[max(0, index - 2) : index + 1]) for index in range(n_chunks)]),
        "audio_self_similarity": np.array([float(np.dot(embeddings[index], normalized_neighbor_mean(index))) for index in range(n_chunks)]),
    }


def extract_clap_embeddings(video_path: str, config: Config, existing_features=None) -> pd.DataFrame:
    if existing_features and CLAP_AUDIO_COLS.issubset(set(existing_features)):
        return pd.DataFrame()

    duration = math.ceil(get_video_duration(video_path))
    waveform, sample_rate = load_video_audio(video_path, sample_rate=CLAP_AUDIO_SR)

    device = torch.device(config.get("device"))
    model_id = config.get("clap_model", CLAP_MODEL_ID)
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id).to(device).eval()

    embeddings, last_error = None, None
    try:
        for batch_size in (4, 2, 1):
            try:
                logger.info("CLAP encoding on GPU with batch_size=%d", batch_size)
                embeddings = _encode_chunks(waveform, sample_rate, model, processor, device, batch_size=batch_size)
                break
            except RuntimeError as error:
                if not any(marker in str(error).upper() for marker in ("CUDA", "CUBLAS", "OUT OF MEMORY")):
                    raise
                last_error = error
                logger.warning("CLAP GPU failed with batch_size=%d: %s", batch_size, error)
                gc.collect()
                torch.cuda.empty_cache()

        if embeddings is None:
            raise RuntimeError("CLAP GPU encoding failed for all batch sizes") from last_error
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    out_dir = embeddings_dir(video_path)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "audio_embeddings.npy"), embeddings)
    np.save(os.path.join(out_dir, "audio_similarity_matrix.npy"), embeddings @ embeddings.T)

    features, columns = _derive(embeddings), sorted(CLAP_AUDIO_COLS)
    output = np.zeros((duration, len(columns)), dtype=np.float64)
    for chunk_index in range(len(embeddings)):
        start_sec, end_sec = chunk_index * CHUNK_SEC, min((chunk_index + 1) * CHUNK_SEC, duration)
        for column_index, column in enumerate(columns):
            output[start_sec:end_sec, column_index] = features[column][chunk_index]
    return pd.DataFrame(output, columns=columns)
