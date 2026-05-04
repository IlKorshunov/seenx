import gc
import os

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from ...utils.logger import Logger
from .common import embeddings_dir
from .consts import CLAP_MODEL_ID, ZERO_SHOT_TEMPERATURE


logger = Logger(show=True).get_logger()

_text_emb_cache: dict[str, np.ndarray] = {}


def load_audio_embeddings(video_path: str) -> np.ndarray | None:
    embeddings_path = os.path.join(embeddings_dir(video_path), "audio_embeddings.npy")
    if not os.path.exists(embeddings_path):
        return None
    embeddings = np.load(embeddings_path)
    return embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-3)


def encode_text_prompts(prompts: list[str], device: str = "cpu", model_id: str = CLAP_MODEL_ID) -> np.ndarray:
    cache_key = "|".join(prompts) + f"|{device}|{model_id}"
    if cache_key in _text_emb_cache:
        return _text_emb_cache[cache_key]

    model, processor = ClapModel.from_pretrained(model_id).eval(), ClapProcessor.from_pretrained(model_id)
    model = model.to(device)
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        text_embeddings = torch.nn.functional.normalize(model.get_text_features(**inputs), p=2, dim=1)
    result = text_embeddings.cpu().float().numpy()

    del model, processor, inputs, text_embeddings
    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    _text_emb_cache[cache_key] = result
    return result


def make_centroid(text_embeddings: np.ndarray, indices: list[int]) -> np.ndarray:
    centroid = text_embeddings[indices].mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-3)
    return centroid


def zero_shot_classify(audio_embeddings: np.ndarray, class_centroids: np.ndarray, temperature: float = ZERO_SHOT_TEMPERATURE) -> np.ndarray:
    logits = (audio_embeddings @ class_centroids.T) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)
