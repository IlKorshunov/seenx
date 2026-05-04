#!/usr/bin/env python3
"""Заполнить только недостающие эмбеддинги по всем модальностям.

Не пересчитывает то, что уже есть:
  - visual: если есть embeddings/<vid>/visual_embeddings.npy — CLIP не гоняется;
    при необходимости дорисовывает similarity + heatmap из сохранённого npy.
  - audio: аналогично для audio_embeddings.npy (CLAP).
  - text (markup): если есть text_embeddings.npy — пропуск; иначе нужен
    markup/<vid>/segments.json с transcript.
  - seg (Whisper-сегменты): если есть seg_embeddings.npy — пропуск; иначе
    extract_semantic_embeddings (USER2 + Whisper).

Запуск из корня репозитория:
  python scripts/fill_missing_embeddings.py
  python scripts/fill_missing_embeddings.py --only visual,audio
  python scripts/fill_missing_embeddings.py --config configs/local.json
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import librosa
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch


def _plot(sim: np.ndarray, labels: list, title: str, path: str) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.3), max(7, n * 0.25)))
    ax.imshow(sim, cmap="RdYlGn_r", vmin=-0.1, vmax=1.0, aspect="auto", origin="lower")
    ax.set(xticks=range(n), yticks=range(n), xlabel="segment", ylabel="segment", title=title)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(ax.images[0], ax=ax, label="cosine sim")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _unload_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _video_ids(data_dir: str) -> list[str]:
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(data_dir, "*/video.mp4")))


def _ensure_visual_sidecars(out_dir: str, vid: str) -> bool:
    """Достроить matrix + heatmap из уже сохранённого visual_embeddings.npy."""
    emb_path = os.path.join(out_dir, "visual_embeddings.npy")
    sim_path = os.path.join(out_dir, "visual_similarity_matrix.npy")
    heat_path = os.path.join(out_dir, "visual_heatmap.png")
    if not os.path.isfile(emb_path):
        return False
    if os.path.isfile(sim_path) and os.path.isfile(heat_path):
        return True
    embs = np.load(emb_path)
    bin_sec = 30
    n_bins = max(1, math.ceil(len(embs) / bin_sec))
    binned = np.array([embs[i * bin_sec : min((i + 1) * bin_sec, len(embs))].mean(0) for i in range(n_bins)])
    binned = binned / np.linalg.norm(binned, axis=1, keepdims=True).clip(min=1e-9)
    sim = binned @ binned.T
    os.makedirs(out_dir, exist_ok=True)
    np.save(sim_path, sim)
    labels = [f"{i * bin_sec}s" for i in range(n_bins)]
    _plot(sim, labels, f"{vid} — visual similarity ({n_bins} bins)", heat_path)
    print(f"  {vid}: visual sidecars from existing npy ({n_bins} bins)", flush=True)
    return True


def _ensure_audio_sidecars(out_dir: str, vid: str) -> bool:
    emb_path = os.path.join(out_dir, "audio_embeddings.npy")
    sim_path = os.path.join(out_dir, "audio_similarity_matrix.npy")
    heat_path = os.path.join(out_dir, "audio_heatmap.png")
    if not os.path.isfile(emb_path):
        return False
    if os.path.isfile(sim_path) and os.path.isfile(heat_path):
        return True
    embs = np.load(emb_path)
    bin_sec = 5
    n_bins = max(1, math.ceil(len(embs) / bin_sec))
    binned = np.array([embs[i * bin_sec : min((i + 1) * bin_sec, len(embs))].mean(0) for i in range(n_bins)])
    binned = binned / np.linalg.norm(binned, axis=1, keepdims=True).clip(min=1e-9)
    sim = binned @ binned.T
    os.makedirs(out_dir, exist_ok=True)
    np.save(sim_path, sim)
    labels = [f"{i * bin_sec}s" for i in range(n_bins)]
    _plot(sim, labels, f"{vid} — audio similarity ({n_bins} bins, {len(embs)} 1s chunks)", heat_path)
    print(f"  {vid}: audio sidecars from existing npy ({len(embs)} chunks)", flush=True)
    return True


def _ensure_text_sidecars(out_dir: str, vid: str, markup_dir: str) -> bool:
    emb_path = os.path.join(out_dir, "text_embeddings.npy")
    sim_path = os.path.join(out_dir, "text_similarity_matrix.npy")
    heat_path = os.path.join(out_dir, "text_heatmap.png")
    if not os.path.isfile(emb_path):
        return False
    if os.path.isfile(sim_path) and os.path.isfile(heat_path):
        return True
    seg_path = os.path.join(markup_dir, vid, "segments.json")
    if not os.path.isfile(seg_path):
        print(f"  {vid}: text sidecars skipped (no markup for labels: {seg_path})", flush=True)
        return False
    with open(seg_path, encoding="utf-8") as f:
        segs = json.load(f)
    pairs = [(i, s.get("transcript", "").strip()) for i, s in enumerate(segs) if s.get("transcript", "").strip()]
    if len(pairs) < 2:
        return False
    idx, _texts = zip(*pairs, strict=True)
    embs = np.load(emb_path)
    if embs.shape[0] != len(idx):
        print(f"  {vid}: text sidecars skipped (embedding rows {embs.shape[0]} != markup segments {len(idx)})", flush=True)
        return False
    sim = embs @ embs.T
    os.makedirs(out_dir, exist_ok=True)
    np.save(sim_path, sim)
    labels = [f"{segs[i].get('start_sec', i)}s" for i in idx]
    _plot(sim, labels, f"{vid} — text similarity ({len(labels)} seg)", heat_path)
    print(f"  {vid}: text sidecars from existing npy", flush=True)
    return True


def run_text(video_ids: list[str], embeddings_root: str, data_dir: str, markup_dir: str) -> None:
    from transformers import AutoModel, AutoTokenizer

    model_id = "deepvk/USER2-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    need_any = False
    for vid in video_ids:
        out_dir = os.path.join(embeddings_root, vid)
        emb = os.path.join(out_dir, "text_embeddings.npy")
        if os.path.isfile(emb):
            _ensure_text_sidecars(out_dir, vid, markup_dir)
            continue
        seg_path = os.path.join(markup_dir, vid, "segments.json")
        if not os.path.isfile(seg_path):
            continue
        with open(seg_path, encoding="utf-8") as f:
            segs = json.load(f)
        pairs = [(i, s.get("transcript", "").strip()) for i, s in enumerate(segs) if s.get("transcript", "").strip()]
        if len(pairs) < 2:
            continue
        need_any = True
        break
    if not need_any:
        print("\n=== TEXT (markup): nothing to compute ===", flush=True)
        for vid in video_ids:
            out_dir = os.path.join(embeddings_root, vid)
            if os.path.isfile(os.path.join(out_dir, "text_embeddings.npy")):
                _ensure_text_sidecars(out_dir, vid, markup_dir)
        return

    print("\n=== TEXT (USER2-base, markup segments) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()

    def encode(texts: list[str], tokenizer, model, device: str) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), 64):
            enc = tokenizer(texts[i : i + 64], padding=True, truncation=True, max_length=8192, return_tensors="pt").to(device)
            with torch.no_grad():
                h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled[:, :256], p=2, dim=1)
            embs.append(pooled.cpu().float().numpy())
        return np.vstack(embs)

    for vid in video_ids:
        out_dir = os.path.join(embeddings_root, vid)
        emb_path = os.path.join(out_dir, "text_embeddings.npy")
        if os.path.isfile(emb_path):
            print(f"  {vid}: text skip (text_embeddings.npy exists)", flush=True)
            _ensure_text_sidecars(out_dir, vid, markup_dir)
            continue
        seg_path = os.path.join(markup_dir, vid, "segments.json")
        if not os.path.isfile(seg_path):
            continue
        with open(seg_path, encoding="utf-8") as f:
            segs = json.load(f)
        pairs = [(i, s.get("transcript", "").strip()) for i, s in enumerate(segs) if s.get("transcript", "").strip()]
        if len(pairs) < 2:
            continue
        idx, texts = zip(*pairs, strict=True)
        embs = encode(list(texts), tokenizer, model, device)
        sim = embs @ embs.T
        os.makedirs(out_dir, exist_ok=True)
        np.save(emb_path, embs)
        np.save(os.path.join(out_dir, "text_similarity_matrix.npy"), sim)
        labels = [f"{segs[i].get('start_sec', i)}s" for i in idx]
        _plot(sim, labels, f"{vid} — text similarity ({len(labels)} seg)", os.path.join(out_dir, "text_heatmap.png"))
        print(f"  {vid}: text OK ({len(labels)} seg)", flush=True)

    del model, tokenizer
    _unload_gpu()


def run_visual(video_ids: list[str], embeddings_root: str, data_dir: str) -> None:
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-large-patch14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    need_model = any(not os.path.isfile(os.path.join(embeddings_root, vid, "visual_embeddings.npy")) for vid in video_ids)
    if not need_model:
        print("\n=== VISUAL (CLIP): all visual_embeddings.npy present ===", flush=True)
        for vid in video_ids:
            _ensure_visual_sidecars(os.path.join(embeddings_root, vid), vid)
        return

    print("\n=== VISUAL (CLIP) ===", flush=True)
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()

    for vid in video_ids:
        out_dir = os.path.join(embeddings_root, vid)
        emb_path = os.path.join(out_dir, "visual_embeddings.npy")
        if os.path.isfile(emb_path):
            print(f"  {vid}: visual skip (visual_embeddings.npy exists)", flush=True)
            _ensure_visual_sidecars(out_dir, vid)
            continue
        vp = os.path.join(data_dir, vid, "video.mp4")
        cap = cv2.VideoCapture(vp)
        step = max(1, int(round(cap.get(cv2.CAP_PROP_FPS) or 30.0)))
        frames, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        if len(frames) < 2:
            print(f"  {vid}: visual skip (too few frames)", flush=True)
            continue

        embs = []
        for i in range(0, len(frames), 32):
            inputs = processor(images=frames[i : i + 32], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feat = torch.nn.functional.normalize(model.get_image_features(**inputs), p=2, dim=1)
            embs.append(feat.cpu().float().numpy())
        embs = np.vstack(embs)

        os.makedirs(out_dir, exist_ok=True)
        np.save(emb_path, embs)
        _ensure_visual_sidecars(out_dir, vid)
        del embs, frames
        _unload_gpu()

    del model, processor
    _unload_gpu()


def run_audio(video_ids: list[str], embeddings_root: str, data_dir: str) -> None:
    from transformers import ClapModel, ClapProcessor

    from src.audio_utils import extract_audio_to_wav

    model_id = "laion/larger_clap_music_and_speech"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_sec = 1
    need_model = any(not os.path.isfile(os.path.join(embeddings_root, vid, "audio_embeddings.npy")) for vid in video_ids)
    if not need_model:
        print("\n=== AUDIO (CLAP): all audio_embeddings.npy present ===", flush=True)
        for vid in video_ids:
            _ensure_audio_sidecars(os.path.join(embeddings_root, vid), vid)
        return

    print("\n=== AUDIO (CLAP) ===", flush=True)
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id).to(device).eval()

    for vid in video_ids:
        out_dir = os.path.join(embeddings_root, vid)
        emb_path = os.path.join(out_dir, "audio_embeddings.npy")
        if os.path.isfile(emb_path):
            print(f"  {vid}: audio skip (audio_embeddings.npy exists)", flush=True)
            _ensure_audio_sidecars(out_dir, vid)
            continue
        vp = os.path.join(data_dir, vid, "video.mp4")
        wav_path = extract_audio_to_wav(vp, sr=48000)
        y, sr = librosa.load(wav_path, sr=48000)
        os.unlink(wav_path)

        cs = chunk_sec * sr
        chunks = [y[i * cs : (i + 1) * cs] for i in range(max(1, math.ceil(len(y) / cs)))]
        chunks = [np.pad(c, (0, max(0, sr - len(c)))) if len(c) < sr else c for c in chunks]

        embs = []
        for i in range(0, len(chunks), 4):
            inputs = processor(audios=chunks[i : i + 4], sampling_rate=sr, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feat = torch.nn.functional.normalize(model.get_audio_features(**inputs), p=2, dim=1)
            embs.append(feat.cpu().float().numpy())
        embs = np.vstack(embs)

        os.makedirs(out_dir, exist_ok=True)
        np.save(emb_path, embs)
        _ensure_audio_sidecars(out_dir, vid)
        del embs, chunks
        _unload_gpu()

    del model, processor
    _unload_gpu()


def run_seg(video_ids: list[str], data_dir: str, embeddings_root: str, config_path: str) -> None:
    from src.extractors.text.semantic_embedding_feature import extract_semantic_embeddings
    from src.utils.config import Config

    print("\n=== SEG (USER2 + Whisper -> seg_embeddings.npy) ===", flush=True)
    config = Config(config_path)
    for vid in video_ids:
        out_dir = os.path.join(embeddings_root, vid)
        seg_path = os.path.join(out_dir, "seg_embeddings.npy")
        if os.path.isfile(seg_path):
            print(f"  {vid}: seg skip (seg_embeddings.npy exists)", flush=True)
            continue
        vp = os.path.join(data_dir, vid, "video.mp4")
        if not os.path.isfile(vp):
            continue
        print(f"  {vid}: computing seg_embeddings ...", flush=True)
        try:
            extract_semantic_embeddings(vp, config=config, existing_features=None)
            print(f"  {vid}: seg OK", flush=True)
        except Exception as e:
            print(f"  {vid}: seg FAILED: {e}", flush=True)
        _unload_gpu()


def main() -> None:
    p = argparse.ArgumentParser(description="Fill only missing multimodal embeddings.")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--embeddings-root", type=str, default="embeddings")
    p.add_argument("--markup-dir", type=str, default="markup")
    p.add_argument("--config", type=str, default="configs/local.json", help="For seg / Whisper semantic path")
    p.add_argument("--only", type=str, default="text,visual,audio,seg", help="Comma-separated: text, visual, audio, seg (default: all)")
    args = p.parse_args()
    only = {x.strip().lower() for x in args.only.split(",") if x.strip()}

    data_dir = os.path.abspath(args.data_dir)
    emb_root = os.path.abspath(args.embeddings_root)
    markup_dir = os.path.abspath(args.markup_dir)

    if not os.path.isfile(args.config):
        print(f"ERROR: config not found: {args.config}", flush=True)
        sys.exit(1)

    video_ids = _video_ids(data_dir)
    print(f"Found {len(video_ids)} videos under {data_dir}/", flush=True)

    if "text" in only:
        run_text(video_ids, emb_root, data_dir, markup_dir)
    if "visual" in only:
        run_visual(video_ids, emb_root, data_dir)
    if "audio" in only:
        run_audio(video_ids, emb_root, data_dir)
    if "seg" in only:
        run_seg(video_ids, data_dir, emb_root, args.config)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
