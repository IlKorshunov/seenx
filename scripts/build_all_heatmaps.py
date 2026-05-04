"""Build text/video/audio similarity heatmaps for all videos in data/.

Runs modalities sequentially to fit within GPU memory:
  1) Text (USER2-base) -> unload
  2) Visual (CLIP) -> unload
  3) Audio (CLAP) -> unload
"""

import gc
import glob
import json
import math
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import librosa
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EMBEDDINGS_ROOT = "embeddings"
DATA_DIR = "data"
MARKUP_DIR = "markup"
FORCE_REBUILD = os.environ.get("FORCE_REBUILD", "0") == "1"


def _plot(sim, labels, title, path):
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


def _unload_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _already_done(out_dir: str, files: list[str]) -> bool:
    if FORCE_REBUILD:
        return False
    return all(os.path.exists(os.path.join(out_dir, f)) for f in files)


def _video_ids():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(DATA_DIR, "*/video.mp4")))


# ── Pass 1: Text ──────────────────────────────────────────────────────────────
def _run_text(video_ids):
    print("\n=== TEXT (USER2-base) ===", flush=True)
    from transformers import AutoModel, AutoTokenizer

    model_id = "deepvk/USER2-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()

    def encode(texts, tokenizer, model, device):
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
        out_dir = os.path.join(EMBEDDINGS_ROOT, vid)
        if _already_done(out_dir, ["text_embeddings.npy", "text_similarity_matrix.npy", "text_heatmap.png"]):
            print(f"  {vid}: text skip (already exists)", flush=True)
            continue
        seg_path = os.path.join(MARKUP_DIR, vid, "segments.json")
        if not os.path.exists(seg_path):
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
        np.save(os.path.join(out_dir, "text_embeddings.npy"), embs)
        np.save(os.path.join(out_dir, "text_similarity_matrix.npy"), sim)
        labels = [f"{segs[i].get('start_sec', i)}s" for i in idx]
        _plot(sim, labels, f"{vid} — text similarity ({len(labels)} seg)", os.path.join(out_dir, "text_heatmap.png"))
        print(f"  {vid}: {len(labels)} seg", flush=True)

    del model, tokenizer
    _unload_gpu()


# ── Pass 2: Visual ────────────────────────────────────────────────────────────
def _run_visual(video_ids):
    print("\n=== VISUAL (CLIP) ===", flush=True)
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-large-patch14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()

    for vid in video_ids:
        out_dir = os.path.join(EMBEDDINGS_ROOT, vid)
        if _already_done(out_dir, ["visual_embeddings.npy", "visual_similarity_matrix.npy", "visual_heatmap.png"]):
            print(f"  {vid}: visual skip (already exists)", flush=True)
            continue
        vp = os.path.join(DATA_DIR, vid, "video.mp4")
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
            continue

        embs = []
        for i in range(0, len(frames), 32):
            inputs = processor(images=frames[i : i + 32], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feat = torch.nn.functional.normalize(model.get_image_features(**inputs), p=2, dim=1)
            embs.append(feat.cpu().float().numpy())
        embs = np.vstack(embs)

        bin_sec = 30
        n_bins = max(1, math.ceil(len(embs) / bin_sec))
        binned = np.array([embs[i * bin_sec : min((i + 1) * bin_sec, len(embs))].mean(0) for i in range(n_bins)])
        binned = binned / np.linalg.norm(binned, axis=1, keepdims=True).clip(min=1e-9)
        sim = binned @ binned.T

        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "visual_embeddings.npy"), embs)
        np.save(os.path.join(out_dir, "visual_similarity_matrix.npy"), sim)
        labels = [f"{i * bin_sec}s" for i in range(n_bins)]
        _plot(sim, labels, f"{vid} — visual similarity ({n_bins} bins)", os.path.join(out_dir, "visual_heatmap.png"))
        print(f"  {vid}: {n_bins} bins ({len(embs)} frames)", flush=True)

        del embs, frames
        _unload_gpu()

    del model, processor
    _unload_gpu()


# ── Pass 3: Audio ─────────────────────────────────────────────────────────────
def _run_audio(video_ids):
    print("\n=== AUDIO (CLAP) ===", flush=True)
    from transformers import ClapModel, ClapProcessor

    from src.audio_utils import extract_audio_to_wav

    model_id = "laion/larger_clap_music_and_speech"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id).to(device).eval()
    chunk_sec = 1

    for vid in video_ids:
        out_dir = os.path.join(EMBEDDINGS_ROOT, vid)
        if _already_done(out_dir, ["audio_embeddings.npy", "audio_similarity_matrix.npy", "audio_heatmap.png"]):
            print(f"  {vid}: audio skip (already exists)", flush=True)
            continue
        vp = os.path.join(DATA_DIR, vid, "video.mp4")
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
        np.save(os.path.join(out_dir, "audio_embeddings.npy"), embs)

        # Bin to 5s for similarity matrix and heatmap (full 1s matrix is too large)
        bin_sec = 5
        n_bins = max(1, math.ceil(len(embs) / bin_sec))
        binned = np.array([embs[i * bin_sec : min((i + 1) * bin_sec, len(embs))].mean(0) for i in range(n_bins)])
        binned = binned / np.linalg.norm(binned, axis=1, keepdims=True).clip(min=1e-9)
        sim = binned @ binned.T
        np.save(os.path.join(out_dir, "audio_similarity_matrix.npy"), sim)
        labels = [f"{i * bin_sec}s" for i in range(n_bins)]
        _plot(sim, labels, f"{vid} — audio similarity ({n_bins} bins, {len(embs)} 1s chunks)", os.path.join(out_dir, "audio_heatmap.png"))
        print(f"  {vid}: {len(embs)} chunks", flush=True)

        del embs, chunks
        _unload_gpu()

    del model, processor
    _unload_gpu()


def main():
    video_ids = _video_ids()
    print(f"Found {len(video_ids)} videos in {DATA_DIR}/", flush=True)
    _run_text(video_ids)
    _run_visual(video_ids)
    _run_audio(video_ids)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
