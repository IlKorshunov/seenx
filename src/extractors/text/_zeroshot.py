"""Shared zero-shot classification ensemble.
curiosity_gap, storytelling, example, viewer_engagement, and section extractors.
"""

from dataclasses import dataclass

import numpy as np
import torch
from geracl import GeraclHF, ZeroShotClassificationPipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..video.architectures.common import unload_ensemble

_GERACL_ID = "deepvk/GeRaCl-USER2-base"
_NLI_ID = "cointegrated/rubert-base-cased-nli-threeway"

W_GERACL = 0.75
W_NLI = 0.25


@dataclass
class ZeroShotTask:
    geracl_labels: list[str]
    nli_hypothesis: str


def load_ensemble(device: str) -> tuple:
    geracl_pipeline = ZeroShotClassificationPipeline(GeraclHF.from_pretrained(_GERACL_ID).to(device).eval(), AutoTokenizer.from_pretrained(_GERACL_ID), device=device, progress_bar=False)
    return geracl_pipeline, AutoTokenizer.from_pretrained(_NLI_ID), AutoModelForSequenceClassification.from_pretrained(_NLI_ID).to(device).eval()


def classify_segments(texts: list[str], task: ZeroShotTask, config, batch_size: int = 64, preloaded: tuple | None = None) -> np.ndarray:
    if not texts:
        return np.array([], dtype=np.float64)

    device = config.get("device")
    owns_models = preloaded is None
    geracl_pipeline, nli_tokenizer, nli_model = load_ensemble(device) if owns_models else preloaded

    labels_per_text = [task.geracl_labels for _ in range(len(texts))]
    geracl_similarities = geracl_pipeline.get_similarities(texts, labels_per_text, same_labels=False, batch_size=batch_size)
    geracl_scores = torch.softmax(torch.cat(geracl_similarities).view(-1, len(task.geracl_labels)), dim=1)[:, 0].cpu().numpy()

    nli_scores = np.zeros(len(texts), dtype=np.float64)
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        for batch_idx, text in enumerate(batch):
            encoded_inputs = nli_tokenizer(text, task.nli_hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                logits = nli_model(**encoded_inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
            nli_scores[batch_start + batch_idx] = float(probs[0])

    if owns_models:
        unload_ensemble((geracl_pipeline, nli_tokenizer, nli_model), device)

    ensemble = W_GERACL * geracl_scores + W_NLI * nli_scores
    return ensemble.astype(np.float64)
