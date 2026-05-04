import argparse
import os

import numpy as np

from ...models.video_mae_retention import VideoMAEFeatureExtractor
from ...utils.config import Config
from ...utils.logger import Logger
from .common import config_device, embeddings_dir, require_1fps_rgb_frames, video_id
from .constants import EMBEDDINGS_ROOT


logger = Logger(show=True).get_logger()


def extract_videomae_embeddings(
    video_path: str, config: Config, backbone: str = "videomae-base", clip_stride: int = 4, embeddings_root: str = EMBEDDINGS_ROOT, force: bool = False
) -> np.ndarray:
    vid = video_id(video_path)
    out_dir = embeddings_dir(video_path, embeddings_root)
    out_path = out_dir / "videomae_embeddings.npy"

    if not force and out_path.is_file():
        logger.info("VideoMAE embeddings cached: %s", out_path)
        return np.load(out_path)

    device = config_device(config)
    logger.info("Extracting VideoMAE embeddings for %s on %s ...", vid, device)
    frames = require_1fps_rgb_frames(video_path)
    extractor = VideoMAEFeatureExtractor(backbone=backbone, device=device)
    embeddings = extractor.extract(frames, clip_stride=clip_stride)
    if len(embeddings) != len(frames):
        raise ValueError(f"VideoMAE extractor returned {len(embeddings)} embeddings for {len(frames)} 1 FPS frames in {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    logger.info("Saved VideoMAE embeddings: %s  shape=%s", out_path, embeddings.shape)
    return embeddings


def extract_videomae_batch(
    data_dir: str = "data",
    config_path: str = "configs/local.json",
    backbone: str = "videomae-base",
    clip_stride: int = 4,
    embeddings_root: str = EMBEDDINGS_ROOT,
    force: bool = False,
):
    config = Config(config_path)
    video_dirs = sorted(d for d in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, d, "video.mp4")))
    logger.info("Found %d videos in %s", len(video_dirs), data_dir)
    for vid in video_dirs:
        try:
            extract_videomae_embeddings(os.path.join(data_dir, vid, "video.mp4"), config, backbone=backbone, clip_stride=clip_stride, embeddings_root=embeddings_root, force=force)
        except Exception as e:
            logger.error("Failed to extract %s: %s", vid, e)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract VideoMAE embeddings for all videos")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--config", default="configs/local.json")
    p.add_argument("--backbone", default="videomae-base")
    p.add_argument("--clip-stride", type=int, default=4)
    p.add_argument("--embeddings-root", default=EMBEDDINGS_ROOT)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    extract_videomae_batch(
        data_dir=args.data_dir, config_path=args.config, backbone=args.backbone, clip_stride=args.clip_stride, embeddings_root=args.embeddings_root, force=args.force
    )
