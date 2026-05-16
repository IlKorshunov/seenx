"""Semantic analysis of YouTube comments: top improvement ideas + constructive criticism.
Uses deepvk/USER2-base embeddings to embed all comments, then:
  1. Zero-shot classifies each comment into {suggestion, criticism, praise, neutral}
  2. Clusters suggestion+criticism comments
  3. Picks a representative comment per cluster (closest to centroid, weighted by likes)
  4. Optionally summarises each cluster

Output per video: get_data/comments/<playlist>/<video_id>/insights.json
python get_data/comment_insights.py # all videos
python get_data/comment_insights.py --video-id [vid] # one video
python get_data/comment_insights.py --summarize # use LLM to summarise clusters
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModel, AutoTokenizer
from transformers import pipeline as hf_pipeline

_ROOT = Path(__file__).resolve().parent.parent
_COMMENTS_ROOT = _ROOT / "get_data" / "comments"

EMBED_MODEL_ID = "deepvk/USER2-base"
EMBED_DIM = 256
MAX_TOKENS = 128
BATCH_SIZE = 64

MIN_COMMENT_LEN = 15
MAX_CLUSTERS = 5
MIN_CLUSTER_SIZE = 2
CLUSTER_DISTANCE_THRESHOLD = 1.2

_SUGGESTION_KW = re.compile(
    r"\b(?:сдела\w*|сними\w*|расскажи\w*|разбер\w*|предлага\w*|добав\w*|"
    r"идея\s+для|можно\s+бы|жд[уеё]\w*|просим|пожалуйста)\b",
    re.IGNORECASE,
)

_CRITICISM_KW = re.compile(
    r"\b(?:ошиб\w*|неправильно|неверно|не\s+соглас\w*|разочарован\w*|"
    r"плох\w*|затянут\w*|скучно|слишком\s+\w+|зря|не\s+стоило|"
    r"испорт\w*|меша\w*|раздража\w*|хуже|деградир\w*|упал\w*\s+качество)\b",
    re.IGNORECASE,
)


def _load_embed_model(device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    model = AutoModel.from_pretrained(EMBED_MODEL_ID).eval().to(device)
    return tokenizer, model


@torch.no_grad()
def _encode_texts(texts: list[str], tokenizer, model, device: str = "cpu", batch_size: int = BATCH_SIZE) -> np.ndarray:
    all_embs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device)
        out = model(**enc)
        emb = out.last_hidden_state[:, 0]
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(all_embs) if all_embs else np.zeros((0, EMBED_DIM), dtype=np.float32)


def _keyword_classify(texts: list[str]) -> list[str]:
    """Fast keyword-based classification ."""
    labels: list[str] = []
    for text in texts:
        if _SUGGESTION_KW.search(text):
            labels.append("suggestion")
        elif _CRITICISM_KW.search(text):
            labels.append("criticism")
        else:
            labels.append("other")
    return labels


def _cluster_and_pick(texts: list[str], embs: np.ndarray, likes: list[int], max_clusters: int = MAX_CLUSTERS) -> list[dict[str, Any]]:
    """Agglomerative clustering, then pick representative per cluster."""
    if len(texts) < MIN_CLUSTER_SIZE:
        if texts:
            return [{"text": texts[0], "likes": likes[0], "cluster_size": 1}]
        return []

    n_clusters = max(2, min(max_clusters, len(texts) // MIN_CLUSTER_SIZE))
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    labels = clustering.fit_predict(embs)

    results: list[dict[str, Any]] = []
    for cid in range(n_clusters):
        mask = labels == cid
        if mask.sum() < 1:
            continue
        idxs = np.where(mask)[0]
        cluster_embs = embs[idxs]
        centroid = cluster_embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9

        sims = cluster_embs @ centroid
        like_boost = np.array([np.log1p(likes[i]) for i in idxs])
        scores = sims + 0.3 * (like_boost / max(like_boost.max(), 1e-9))
        best_global = int(idxs[int(scores.argmax())])
        results.append({"representative": texts[best_global], "likes": likes[best_global], "cluster_size": int(mask.sum()), "sample_comments": [texts[int(j)] for j in idxs[np.argsort(-scores)[:3]]]})

    results.sort(key=lambda r: (-r["cluster_size"], -r["likes"]))
    return results[:max_clusters]


def _llm_summarize(clusters: list[dict[str, Any]], category: str, model_id: str = "google/gemma-3-1b-it", device: str = "cpu") -> list[dict[str, Any]]:
    """Optional: summarise each cluster with a small HF text-gen model."""
    pipe = hf_pipeline("text-generation", model=model_id, device=device, torch_dtype=torch.float16 if device != "cpu" else torch.float32, max_new_tokens=120)
    label_ru = "предложение по улучшению" if category == "suggestion" else "конструктивная критика"
    for cluster in clusters:
        comments_block = os.linesep.join(f"- {c}" for c in cluster["sample_comments"][:5])
        prompt = (
            f"Ниже несколько комментариев к YouTube-видео, объединённых одной темой ({label_ru}).{os.linesep}"
            f"Кратко сформулируй суть в 1-2 предложениях на русском:{os.linesep}{os.linesep}"
            f"{comments_block}{os.linesep}{os.linesep}Суть:"
        )
        try:
            generated = pipe(prompt, do_sample=False)[0]["generated_text"]
            cluster["summary"] = generated.split("Суть:")[-1].strip()
        except Exception as e:
            cluster["summary"] = f"[error: {e}]"

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return clusters


def analyze_video(comments_path: Path, *, tokenizer, model, device: str = "cpu", summarize: bool = False, llm_model: str = "google/gemma-3-1b-it") -> dict[str, Any]:
    data = json.loads(comments_path.read_text(encoding="utf-8"))
    video_id = data.get("video_id", comments_path.parent.name)
    title = data.get("video_title", "")
    threads = data.get("threads", [])

    all_texts: list[str] = []
    all_likes: list[int] = []
    for thread in threads:
        text = (thread.get("text") or "").strip()
        if len(text) >= MIN_COMMENT_LEN:
            all_texts.append(text)
            all_likes.append(int(thread.get("like_count", 0)))
        for reply in thread.get("replies") or []:
            text = (reply.get("text") or "").strip()
            if len(text) >= MIN_COMMENT_LEN:
                all_texts.append(text)
                all_likes.append(int(reply.get("like_count", 0)))

    if not all_texts:
        return {"video_id": video_id, "video_title": title, "total_comments_analyzed": 0, "suggestions": [], "criticisms": []}

    labels = _keyword_classify(all_texts)

    sugg_idx = [i for i, l in enumerate(labels) if l == "suggestion"]
    crit_idx = [i for i, l in enumerate(labels) if l == "criticism"]

    sugg_texts = [all_texts[i] for i in sugg_idx]
    crit_texts = [all_texts[i] for i in crit_idx]

    texts_to_embed = sugg_texts + crit_texts
    if texts_to_embed:
        print(f"Encoding {len(texts_to_embed)} candidate comments ({len(sugg_texts)} suggestion, {len(crit_texts)} criticism)")
        embs = _encode_texts(texts_to_embed, tokenizer, model, device)
        sugg_embs = embs[: len(sugg_texts)]
        crit_embs = embs[len(sugg_texts) :]
    else:
        sugg_embs = np.zeros((0, EMBED_DIM), dtype=np.float32)
        crit_embs = np.zeros((0, EMBED_DIM), dtype=np.float32)

    sugg_clusters = _cluster_and_pick(sugg_texts, sugg_embs, [all_likes[i] for i in sugg_idx])
    crit_clusters = _cluster_and_pick(crit_texts, crit_embs, [all_likes[i] for i in crit_idx])

    if summarize:
        sugg_clusters = _llm_summarize(sugg_clusters, "suggestion", llm_model, device)
        crit_clusters = _llm_summarize(crit_clusters, "criticism", llm_model, device)

    label_counts = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1

    return {
        "video_id": video_id,
        "video_title": title,
        "total_comments_analyzed": len(all_texts),
        "label_distribution": label_counts,
        "suggestions": sugg_clusters,
        "criticisms": crit_clusters,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic analysis of YouTube comments")
    ap.add_argument("--video-id", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="Use LLM to summarise clusters")
    ap.add_argument("--llm-model", type=str, default="google/gemma-3-1b-it")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {EMBED_MODEL_ID}")
    tokenizer, model = _load_embed_model(device)

    if args.video_id:
        paths = list(_COMMENTS_ROOT.rglob(f"{args.video_id}/comments.json"))
        if not paths:
            raise ValueError(f"No comments.json for {args.video_id}") # temporary error
    else:
        paths = sorted(_COMMENTS_ROOT.rglob("*/comments.json"))

    print(f"Found {len(paths)} video to analyze")

    for cp in paths:
        out_path = cp.parent / "insights.json"
        if out_path.exists() and not args.force:
            print(f"[skip] {cp.parent.name} (insights.json exists)")
            continue

        print(f"[{cp.parent.name}] Analyzing")
        result = analyze_video(cp, tokenizer=tokenizer, model=model, device=device, summarize=args.summarize, llm_model=args.llm_model)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"suggestion clusters: {len(result['suggestions'])}")
        for cluster in result["suggestions"]:
            print(f" {cluster['representative']}")
        print(f"criticism clusters: {len(result['criticisms'])}")
        for cluster in result["criticisms"]:
            print(f" {cluster['representative']}")
        print(f"DONE: {out_path}")


if __name__ == "__main__":
    main()
