"""Shared zero-shot classification ensemble: GeRaCl + rubert-NLI.

Loads both models once, scores every segment, returns weighted ensemble.
Used by curiosity_gap, storytelling, example, viewer_engagement, and section extractors.

In batch mode, call ``load_ensemble`` once, pass the result to ``classify_segments``
via *preloaded*, and call ``unload_ensemble`` when done with all videos.
"""

import gc
from dataclasses import dataclass

import numpy as np
import torch


_GERACL_ID = "deepvk/GeRaCl-USER2-base"
_NLI_ID = "cointegrated/rubert-base-cased-nli-threeway"

W_GERACL = 0.75
W_NLI = 0.25


@dataclass
class ZeroShotTask:
    geracl_labels: list[str]
    nli_hypothesis: str


def load_ensemble(device: str) -> tuple:
    """Load GeRaCl + NLI models. Returns (geracl_pipeline, nli_tokenizer, nli_model)."""
    from geracl import GeraclHF, ZeroShotClassificationPipeline
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    geracl_tokenizer = AutoTokenizer.from_pretrained(_GERACL_ID)
    geracl_model = GeraclHF.from_pretrained(_GERACL_ID).to(device).eval()
    geracl_pipeline = ZeroShotClassificationPipeline(geracl_model, geracl_tokenizer, device=device, progress_bar=False)

    nli_tokenizer = AutoTokenizer.from_pretrained(_NLI_ID)
    nli_model = AutoModelForSequenceClassification.from_pretrained(_NLI_ID).to(device).eval()

    return geracl_pipeline, nli_tokenizer, nli_model


def unload_ensemble(models: tuple, device: str = "cuda") -> None:
    """Delete models and free GPU memory."""
    del models
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def classify_segments(texts: list[str], task: ZeroShotTask, config, batch_size: int = 64, preloaded: tuple | None = None) -> np.ndarray:
    """Return ensemble scores (0..1) for each text. Higher = more positive.

    If *preloaded* is given (geracl_pipeline, nli_tokenizer, nli_model), those models are used
    and NOT freed afterwards (caller is responsible for ``unload_ensemble``).
    """
    if not texts:
        return np.array([], dtype=np.float64)

    device = config.get("device")
    owns_models = preloaded is None
    if owns_models:
        geracl_pipeline, nli_tokenizer, nli_model = load_ensemble(device)
    else:
        geracl_pipeline, nli_tokenizer, nli_model = preloaded

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
        del geracl_pipeline, nli_tokenizer, nli_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    ensemble = W_GERACL * geracl_scores + W_NLI * nli_scores
    return ensemble.astype(np.float64)
