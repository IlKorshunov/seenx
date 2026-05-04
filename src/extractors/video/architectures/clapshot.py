import os
import subprocess
import tempfile

import cv2
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

from ....utils.config import Config
from ....utils.logger import Logger
from ...audio.consts import CHUNK_SEC
from ..common import embeddings_dir
from .common import resolve_device, robust_minmax, spread_to_frames


logger = Logger(show=True).get_logger()


def _load_audio_mono(video_path: str, sr: int = 48_000) -> np.ndarray:
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr), "-vn", "-loglevel", "error", tmp_wav], check=True)
        wav, _ = sf.read(tmp_wav, dtype="float32")
        return wav
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def _video_n_frames_and_fps(video_path: str) -> tuple[int, float]:
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return n, fps


def clap_boundary_signal(video_path: str, config: Config) -> np.ndarray:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")

    model_name = config.get("clap_model") or "laion/larger_clap_general"
    chunk_seconds = float(config.get("clap_chunk_seconds") or 1.0)
    batch = int(config.get("clap_batch_size") or 32)
    device = resolve_device(config)

    n_frames, src_fps = _video_n_frames_and_fps(video_path)
    if src_fps <= 0:
        raise ValueError(f"Could not read FPS from {video_path}")

    cached = embeddings_dir(video_path) / "audio_embeddings.npy"
    if cached.is_file():
        embs = torch.from_numpy(np.load(cached).astype(np.float32))
        if len(embs) < 2:
            return np.zeros(n_frames, dtype=np.float32)
        diss = robust_minmax((1.0 - (embs[:-1] * embs[1:]).sum(-1)).numpy().astype(np.float32))
        return spread_to_frames((np.arange(len(diss)) + 1) * CHUNK_SEC, diss, n_frames, src_fps, spread_seconds=0.25)

    logger.info("CLAP: extracting audio from %s", video_path)
    sr, wav = 48_000, _load_audio_mono(video_path)
    chunk_len = int(chunk_seconds * sr)
    n_chunks = wav.size // chunk_len
    if n_chunks < 2:
        return np.zeros(n_frames, dtype=np.float32)

    wav = wav[: n_chunks * chunk_len].reshape(n_chunks, chunk_len)

    logger.info("CLAP: loading %s on %s", model_name, device)
    proc = ClapProcessor.from_pretrained(model_name)
    clap = ClapModel.from_pretrained(model_name).to(device).eval()

    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, n_chunks, batch), desc="CLAP", unit="batch"):
            inputs = proc(audios=list(wav[i : i + batch]), sampling_rate=sr, return_tensors="pt").to(device)
            embs.append(F.normalize(clap.get_audio_features(**inputs).float(), dim=-1).cpu())
    embs = torch.cat(embs, dim=0)

    del clap
    if device.type == "cuda":
        torch.cuda.empty_cache()

    diss = (1.0 - (embs[:-1] * embs[1:]).sum(-1)).numpy().astype(np.float32)
    diss = robust_minmax(diss)
    boundary_times = (np.arange(len(diss)) + 1) * chunk_seconds
    return spread_to_frames(boundary_times_s=boundary_times, boundary_scores=diss, n_frames=n_frames, src_fps=src_fps, spread_seconds=0.25)
