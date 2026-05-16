import os
import numpy as np
import torch
import argparse
from ...models.bert_retention import DEFAULT_BACKBONE, BERTFeatureExtractor
from ...utils.config import Config
from ._base import get_segments_and_duration, logger, seg_bounds
from .common import release_models, video_id


EMBEDDINGS_ROOT = "embeddings"


def _resolve_embeddings_root(embeddings_root: str) -> str:
    return os.path.normpath(os.path.abspath(embeddings_root))


def extract_bert_embeddings(video_path: str, config: Config, backbone: str = DEFAULT_BACKBONE, embeddings_root: str = EMBEDDINGS_ROOT, force: bool = False) -> np.ndarray:
    embeddings_root = _resolve_embeddings_root(embeddings_root)
    video_id_value = video_id(video_path)
    out_dir = os.path.join(embeddings_root, video_id_value)
    out_path = os.path.join(out_dir, "bert_embeddings.npy")

    if not force and os.path.isfile(out_path):
        logger.info("Using existing BERT embeddings: %s", out_path)
        return np.load(out_path)

    segments, duration = get_segments_and_duration(video_path, config)
    valid = [(seg, start, end) for seg in segments if seg.get("text", "").strip() for start, end in [seg_bounds(seg, duration)] if start < end]

    device = torch.device(config.get("device"))
    logger.info("Extracting BERT embeddings for %s (%d segments, %ds) on %s", video_id_value, len(valid), duration, device)

    extractor = BERTFeatureExtractor(backbone=backbone, device=device)
    texts = [seg["text"].strip() for seg, _, _ in valid]
    seg_meta = [{"start": float(seg["start"]), "end": float(seg["end"])} for seg, _, _ in valid]
    embeddings = extractor.extract(texts, seg_meta, duration)

    release_models(extractor, device=device)
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = out_path[:-4] + ".tmp.npy" if out_path.endswith(".npy") else out_path + ".tmp.npy"
    np.save(tmp_path, embeddings)
    os.replace(tmp_path, out_path)
    logger.info("Saved BERT embeddings: %s  shape=%s", out_path, embeddings.shape)
    return embeddings


def extract_bert_batch(
    data_dir: str = "data", config_path: str = "configs/local.json", backbone: str = DEFAULT_BACKBONE, embeddings_root: str = EMBEDDINGS_ROOT, force: bool = False
):
    embeddings_root = _resolve_embeddings_root(embeddings_root)
    config = Config(config_path)
    video_dirs = sorted(dirname for dirname in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, dirname, "video.mp4")))
    logger.info("Found %d videos in %s", len(video_dirs), data_dir)

    for video_id in video_dirs:
        video_path = os.path.join(data_dir, video_id, "video.mp4")
        try:
            extract_bert_embeddings(video_path, config, backbone=backbone, embeddings_root=embeddings_root, force=force)
        except Exception as error:
            logger.error("Failed to extract %s: %s", video_id, error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract BERT embeddings for all videos")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--embeddings-root", default=EMBEDDINGS_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    extract_bert_batch(data_dir=args.data_dir, config_path=args.config, backbone=args.backbone, embeddings_root=args.embeddings_root, force=args.force)
