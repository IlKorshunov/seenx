"""Semantic analysis of YouTube comments: top improvement ideas + constructive criticism.

Uses deepvk/USER2-base embeddings (same model as the rest of the pipeline) to
embed all comments, then:
  1. Zero-shot classifies each comment into {suggestion, criticism, praise, neutral}
  2. Clusters suggestion+criticism comments via Agglomerative Clustering
  3. Picks a representative comment per cluster (closest to centroid, weighted by likes)
  4. Optionally summarises each cluster with an HF text-generation model

Output per video: get_data/comments/<playlist>/<video_id>/insights.json

Usage:
  python get_data/comment_insights.py                          # all videos
  python get_data/comment_insights.py --video-id pfacUo-6wNs   # one video
  python get_data/comment_insights.py --force                   # overwrite
  python get_data/comment_insights.py --summarize               # use LLM to summarise clusters
  python get_data/comment_insights.py --summarize --llm-model google/gemma-3-1b-it
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModel, AutoTokenizer


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
    r"\b(?:сделайте|сделай|снимите|снимай|расскажите|разберите|"
    r"хотелось\s*бы|хочу\s+(?:вторую|продолжение|часть)|предлагаю|"
    r"не\s+хватает|добавьте|добавить|было\s+бы\s+(?:круто|здорово|классно|интересно)|"
    r"идея\s+для|можно\s+(?:было\s+бы|ещё|еще)|ждём?|жду|"
    r"просим|пожалуйста\s+(?:сделайте|снимите|расскажите)|"
    r"please\s+(?:make|do|cover))\b",
    re.IGNORECASE,
)

_CRITICISM_KW = re.compile(
    r"\b(?:ошиб(?:ка|лись|ся)|неправильно|неверно|не\s+согласен|не\s+согласна|"
    r"разочарован|плохо[йе]?\s+(?:звук|качество|монтаж)|затянут[оа]?|скучно|"
    r"слишком\s+(?:много|длинно|долго)|зря|не\s+стоило|"
    r"к\s+сожалению|испортил[иа]?|мешает|раздражает|"
    r"хуже|деградир|упал[оа]?\s+качество|"
    r"wrong|mistake|error|bad\s+(?:audio|quality))\b",
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
    """Fast keyword-based classification into suggestion/criticism/other."""
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

    n_clusters = min(max_clusters, len(texts) // MIN_CLUSTER_SIZE)
    n_clusters = max(n_clusters, 1)

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
        best_local = int(scores.argmax())
        best_global = int(idxs[best_local])

        sample_idxs = idxs[np.argsort(-scores)[:3]]
        samples = [texts[int(j)] for j in sample_idxs]

        results.append({"representative": texts[best_global], "likes": likes[best_global], "cluster_size": int(mask.sum()), "sample_comments": samples})

    results.sort(key=lambda r: (-r["cluster_size"], -r["likes"]))
    return results[:max_clusters]


def _llm_summarize(clusters: list[dict[str, Any]], category: str, model_id: str = "google/gemma-3-1b-it", device: str = "cpu") -> list[dict[str, Any]]:
    """Optional: summarise each cluster with a small HF text-gen model."""
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("transformers pipeline not available, skipping LLM summarization", file=sys.stderr)
        return clusters

    pipe = hf_pipeline("text-generation", model=model_id, device=device, torch_dtype=torch.float16 if device != "cpu" else torch.float32, max_new_tokens=120)

    label_ru = "предложение по улучшению" if category == "suggestion" else "конструктивная критика"
    for cluster in clusters:
        comments_block = "\n".join(f"- {c}" for c in cluster["sample_comments"][:5])
        prompt = (
            f"Ниже несколько комментариев к YouTube-видео, объединённых одной темой ({label_ru}).\n"
            f"Кратко сформулируй суть в 1-2 предложениях на русском:\n\n"
            f"{comments_block}\n\nСуть:"
        )
        try:
            out = pipe(prompt, do_sample=False)
            generated = out[0]["generated_text"]
            summary = generated.split("Суть:")[-1].strip()
            cluster["summary"] = summary
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
        print(f"  Encoding {len(texts_to_embed)} candidate comments ({len(sugg_texts)} suggestion, {len(crit_texts)} criticism)...", file=sys.stderr)
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
    print(f"Loading {EMBED_MODEL_ID} on {device}...", file=sys.stderr)
    tokenizer, model = _load_embed_model(device)

    if args.video_id:
        paths = list(_COMMENTS_ROOT.rglob(f"{args.video_id}/comments.json"))
        if not paths:
            print(f"No comments.json for {args.video_id}", file=sys.stderr)
            sys.exit(1)
    else:
        paths = sorted(_COMMENTS_ROOT.rglob("*/comments.json"))

    print(f"Found {len(paths)} video(s) to analyze", file=sys.stderr)

    for cp in paths:
        out_path = cp.parent / "insights.json"
        if out_path.exists() and not args.force:
            print(f"[skip] {cp.parent.name} (insights.json exists)", file=sys.stderr)
            continue

        print(f"[{cp.parent.name}] Analyzing...", file=sys.stderr)
        result = analyze_video(cp, tokenizer=tokenizer, model=model, device=device, summarize=args.summarize, llm_model=args.llm_model)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        n_sugg = sum(c["cluster_size"] for c in result["suggestions"])
        n_crit = sum(c["cluster_size"] for c in result["criticisms"])
        print(
            f"  -> {result['total_comments_analyzed']} comments, "
            f"{len(result['suggestions'])} suggestion clusters ({n_sugg} comments), "
            f"{len(result['criticisms'])} criticism clusters ({n_crit} comments)",
            file=sys.stderr,
        )
        print(f"  -> {out_path}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
