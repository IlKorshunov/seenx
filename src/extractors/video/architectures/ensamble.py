import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ....utils.config import Config
from ....utils.logger import Logger
from ....video_dataset import VideoBatchDataset
from ..shot_segmentation import predictions_to_scenes
from .clapshot import clap_boundary_signal
from .common import resolve_device
from .raftshot import raft_boundary_signal
from .transnetv2 import TransNetV2
from .videomaeshot import videomae_boundary_signal


logger = Logger(show=True).get_logger()


DEFAULT_ENSEMBLE_WEIGHTS = {"transnet": 0.7, "clap": 0.25, "videomae": 0.7, "raft": 0.6}


def transnet_boundary_signal(video_path: str, config: Config) -> np.ndarray:
    weights_path = config.get("shot_segmentor")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"TransNetV2 weights not found at {weights_path}")

    device = resolve_device(config)
    model = TransNetV2()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval().to(device)

    def transform(frame: np.ndarray) -> np.ndarray:
        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (48, 27), interpolation=cv2.INTER_AREA)[np.newaxis, :, :, :]

    ds = VideoBatchDataset(video_path, batch_size=config.get("shot_segmentor_batch_size"), transform=transform)
    out = np.zeros(ds.total_processed_frames or 0, dtype=np.float32)
    with torch.no_grad():
        for frames_batch, frame_indices in tqdm(ds, desc="TransNet", unit="batch"):
            preds = torch.sigmoid(model(torch.tensor(frames_batch, dtype=torch.uint8).to(device).unsqueeze(0))[0])
            out[frame_indices] = preds.cpu().numpy().squeeze(0).flatten()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def ensemble_shot_segmentation(video_path: str, config: Config) -> np.ndarray:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")

    disabled = set(config.get("ensemble_disable") or [])
    if config.get("use_clap") is False:
        disabled.add("clap")
    weights = dict(DEFAULT_ENSEMBLE_WEIGHTS)
    weights.update(config.get("ensemble_weights") or {})

    signals: dict[str, np.ndarray] = {}

    if "transnet" not in disabled:
        logger.info("[ensemble] TransNetV2 ...")
        signals["transnet"] = transnet_boundary_signal(video_path, config)

    if "clap" not in disabled:
        logger.info("[ensemble] CLAP ...")
        try:
            signals["clap"] = clap_boundary_signal(video_path, config)
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            logger.warning("CLAP failed (%s) — skipping", exc)

    if "videomae" not in disabled:
        logger.info("[ensemble] VideoMAE ...")
        try:
            signals["videomae"] = videomae_boundary_signal(video_path, config)
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            logger.warning("VideoMAE failed (%s) — skipping", exc)

    if "raft" not in disabled:
        logger.info("[ensemble] RAFT ...")
        try:
            signals["raft"] = raft_boundary_signal(video_path, config)
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            logger.warning("RAFT failed (%s) — skipping", exc)

    if not signals:
        raise RuntimeError("All ensemble methods disabled or failed")

    min_len = min(s.shape[0] for s in signals.values())
    for k in list(signals.keys()):
        signals[k] = signals[k][:min_len]

    used_weights = {m: float(weights.get(m, 1.0)) for m in signals}
    total_w = sum(used_weights.values()) or 1.0
    combined = np.zeros(min_len, dtype=np.float32)
    for m, sig in signals.items():
        combined += (used_weights[m] / total_w) * sig

    min_agree = config.get("ensemble_min_agree")
    if min_agree:
        per_thr = float(config.get("ensemble_per_method_threshold") or 0.4)
        agree = np.zeros(min_len, dtype=np.int32)
        for sig in signals.values():
            agree += (sig >= per_thr).astype(np.int32)
        combined = combined * (agree >= int(min_agree)).astype(np.float32)

    threshold = float(config.get("ensemble_threshold") or 0.5)
    scenes = predictions_to_scenes(combined, threshold=threshold)

    logger.info("Ensemble: methods=%s weights=%s n_scenes=%d", list(signals.keys()), used_weights, len(scenes))
    scene_ranges = ", ".join([f"{s}:{e}" for s, e in scenes])
    logger.info("Detected scenes (start_frame, end_frame): %s", scene_ranges)
    return scenes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=False)
    args = parser.parse_args()
    scenes = ensemble_shot_segmentation(args.video_path, config=Config(args.config))
