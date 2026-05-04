import einops
import torch

from ..utils.logger import Logger
from .mm_constants import NUM_FRAMES


logger = Logger(show=True).get_logger()


def temporal_aggregator(mm_projector, config, frames_features):
    projector_type = config.mm_projector_type
    if projector_type in {"mlp2x_gelu", "linear"}:
        return mm_projector(frames_features.mean(1))
    if projector_type in {"spatial_conv", "spatial_pool"} or "tc_connector" in projector_type or "tp_connector" in projector_type:
        return mm_projector(frames_features)
    raise ValueError(f"Unsupported projector type {projector_type}")


def encode_images_or_videos(vision_tower, images, config):
    num_frames = config.num_frames if hasattr(config, "num_frames") else NUM_FRAMES
    data_batch = [data.expand(num_frames, -1, -1, -1) if modal == "image" else data for data, modal in images]
    data_batch = torch.stack(data_batch, dim=0)
    assert len(data_batch.size()) == 5
    batch_size = data_batch.size(0)
    frames = einops.rearrange(data_batch, "b t c h w -> (b t) c h w")
    frames_features = vision_tower(frames)
    return einops.rearrange(frames_features, "(b t) n h -> b t n h", b=batch_size)
