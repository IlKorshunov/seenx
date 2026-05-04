import argparse
import os
from functools import partial

import torch
from transformers import AutoConfig

from ..utils.logger import Logger
from .mm_arch import encode_images_or_videos, temporal_aggregator
from .mm_constants import NUM_FRAMES
from .mm_models import build_audio_tower, build_vision_tower
from .mm_processing import process_video
from .mm_projector import build_audio_projector, build_vision_projector


logger = Logger(show=True).get_logger()


def unfreeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def multimodal_features(video_path: str, app_config=None):
    device = torch.device(app_config.get("device")) if app_config is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    working_dir = os.getcwd()
    model_path = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    config = AutoConfig.from_pretrained(model_path)

    config.num_frames = NUM_FRAMES
    config.mm_audio_tower = f"{working_dir}/audio_tower.bin"
    config.mm_projector_a_type = "mlp2x_gelu"
    config.mm_vision_tower = "google/siglip-so400m-patch14-384"
    config.mm_projector_type = "stc_connector_v35"

    logger.info(f"Configuration loaded: {config}")

    logger.info("Loading vision tower...")
    num_frames = config.num_frames if hasattr(config, "num_frames") else NUM_FRAMES
    vision_tower = build_vision_tower(config).half().to(device)
    vision_tower.requires_grad_(False)

    logger.info("Building vision projector...")
    mm_projector = build_vision_projector(config).half().to(device)
    unfreeze_module(mm_projector)

    logger.info("Loading audio tower...")
    audio_tower, audio_tower_cfg = build_audio_tower(config)
    audio_tower = audio_tower.half().to(device)
    audio_tower.requires_grad_(False)

    logger.info("Building audio projector...")
    mm_projector_a = build_audio_projector(config).half().to(device)
    unfreeze_module(mm_projector_a)

    processor = {"video": partial(process_video, processor=vision_tower.image_processor, aspect_ratio=None, num_frames=num_frames)}
    logger.info("Processing video...")
    image_or_video = processor["video"](video_path, s=0, e=10, va=True)

    tensor = {k: v.half().to(device) for k, v in image_or_video.items()}
    tensor = [(tensor, "video")]

    X_video, X_audio, select_audio_id, select_videoimage_id = [], [], [], []
    for idx, data_list in enumerate(tensor):
        if isinstance(data_list[0], dict):
            assert data_list[1] == "video"
            X_audio.append(data_list[0]["audio"])
            select_audio_id.append(True)
            X_video.append((data_list[0]["video"], "video"))
            select_videoimage_id.append(True)
        else:
            if data_list[1] == "audio":
                X_audio.append(data_list[0])
                select_audio_id.append(True)
                select_videoimage_id.append(False)
            elif data_list[1] in {"video", "image"}:
                X_video.append(data_list)
                select_videoimage_id.append(True)
                select_audio_id.append(False)
            else:
                raise NotImplementedError

    logger.info("Encoding multimodal features...")
    if X_audio:
        Xa_features = torch.cat(X_audio, dim=0)
        audio_padding_mask = torch.zeros(Xa_features.shape, device=device).bool()
        audio_embedding, T, F = audio_tower.extract_features(Xa_features, padding_mask=audio_padding_mask)
        Xa_features = mm_projector_a(audio_embedding)
        Xa_features = Xa_features.view(len(X_audio), -1, Xa_features.shape[-1])

    if X_video:
        X_features = encode_images_or_videos(vision_tower, X_video, config)
        X_features = temporal_aggregator(mm_projector, config, X_features)

    mm_features = []
    idx_a, idx_v = 0, 0
    for audio_idx, videoimage_idx in zip(select_audio_id, select_videoimage_id, strict=True):
        if audio_idx and videoimage_idx:
            mm_features.append(torch.cat([X_features[idx_v], Xa_features[idx_a]], dim=0))
            idx_a += 1
            idx_v += 1
        elif audio_idx:
            mm_features.append(Xa_features[idx_a])
            idx_a += 1
        elif videoimage_idx:
            mm_features.append(X_features[idx_v])
            idx_v += 1
        else:
            raise NotImplementedError

    print("Multimodal features shape:", mm_features[0].shape)
    return mm_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="/kaggle/input/seenx-data/videos/faceless_youtube_channel_ideas.mp4")

    args = parser.parse_args()
    mm_features = multimodal_features(video_path=args.video_path)
