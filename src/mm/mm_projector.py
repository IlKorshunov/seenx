import os
import re

import einops
import torch
import torch.nn as nn
from timm.models.layers import LayerNorm2d
from timm.models.regnet import RegStage
from transformers import TRANSFORMERS_CACHE


def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    cache_dir = TRANSFORMERS_CACHE if cache_dir is None else cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    return os.path.join(repo_cache, "snapshots", revision)


def load_mm_projector(model_path, cache_dir=None, token=None):
    if os.path.exists(os.path.join(model_path, "mm_projector.bin")):
        is_local = True
        folder = model_path
    else:
        is_local = False
        folder = parse_snapshot_folder(model_path, cache_dir=cache_dir, repo_type="model")
        if not os.path.exists(os.path.join(folder, "mm_projector.bin")):
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

    mm_projector_weights = torch.load(os.path.join(folder, "mm_projector.bin"), map_location="cpu")
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    return mm_projector_weights


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")
    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    if projector_type == "stc_connector":
        return STCConnector(config)
    if projector_type == "stp_connector":
        return STPConnector(config)
    if projector_type == "stc_connector_v35":
        return STCConnectorV35(config)
    if projector_type == "spatial_conv":
        return SpatialConv(config)
    if projector_type == "spatial_pool":
        return SpatialPool(config)
    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


def build_audio_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_a_type", "linear")
    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size_a, config.hidden_size_a)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size_a, config.hidden_size_a))
        return nn.Sequential(*modules)
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size_a, config.hidden_size_a)
    if projector_type == "identity":
        return IdentityMap()


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class STCConnector(nn.Module):
    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.downsample = downsample
        if depth != 0:
            self.s1 = RegStage(depth=depth, in_chs=encoder_hidden_size, out_chs=hidden_size, stride=1, dilation=1, act_layer=nn.SiLU, norm_layer=LayerNorm2d)
        else:
            self.s1 = nn.Identity()
        self.sampler = nn.Sequential(nn.Conv3d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=downsample, stride=downsample, padding=1, bias=True), nn.SiLU())
        if depth != 0:
            self.s2 = RegStage(depth=depth, in_chs=hidden_size, out_chs=hidden_size, stride=1, dilation=1, act_layer=nn.SiLU, norm_layer=LayerNorm2d)
        else:
            self.s2 = nn.Identity()
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

    def forward(self, x):
        t = x.size(1)
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        x = self.sampler(x)
        new_t = x.size(2)
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)
        return x


class STPConnector(STCConnector):
    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(nn.AvgPool3d(downsample), nn.SiLU())


class STCConnectorV35(STCConnector):
    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(
            nn.Conv3d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=downsample, stride=downsample, padding=0, bias=True), nn.SiLU()
        )


class SpatialConv(STCConnector):
    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)


class SpatialPool(STPConnector):
    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
