import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

from ..utils.logger import Logger


logger = Logger(show=True).get_logger()


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


ROOT = os.environ.get("WORKING_DIR", str(Path(__file__).resolve().parent.parent.parent.parent))
path = os.path.join(ROOT, "VideoLLaMA2")
logger.info(f"Adding {path} to sys.path")
add_path(path)

from videollama2.model.beats.BEATs import BEATs, BEATsConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, load_pretrained=False):
        super().__init__()

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        config = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        config._attn_implementation = "sdpa"

        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name) if load_pretrained else CLIPVisionModel(config=config)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            return image_features[:, 1:]
        if self.select_feature == "cls_patch":
            return image_features
        raise ValueError(f"Unexpected select feature: {self.select_feature}")

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            return [self.feature_select(self.vision_tower(image.unsqueeze(0), output_hidden_states=True)).to(image.dtype) for image in images]
        return self.feature_select(self.vision_tower(images, output_hidden_states=True)).to(images.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, load_pretrained=False):
        super().__init__()

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)

        config = SiglipVisionConfig.from_pretrained(self.vision_tower_name)
        config._attn_implementation = "sdpa"

        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name) if load_pretrained else SiglipVisionModel(config=config)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            return image_features
        raise ValueError(f"Unexpected select feature: {self.select_feature}")

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            return [self.feature_select(self.vision_tower(image.unsqueeze(0), output_hidden_states=True)).to(image.dtype) for image in images]
        return self.feature_select(self.vision_tower(images, output_hidden_states=True)).to(images.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    if "clip" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "siglip" in vision_tower:
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    raise ValueError(f"Unknown vision tower: {vision_tower}")


def build_audio_tower(audio_tower_cfg, delay_load=False, **kwargs) -> tuple[nn.Module, BEATsConfig]:
    audio_tower = getattr(audio_tower_cfg, "mm_audio_tower", getattr(audio_tower_cfg, "audio_tower", None))
    if not delay_load:
        beats_checkpoint = torch.load(audio_tower, map_location="cpu", weights_only=False)
        beats_cfg = BEATsConfig(beats_checkpoint["cfg"]) if "cfg" in beats_checkpoint else BEATsConfig()
        beats = BEATs(beats_cfg)
        if not audio_tower.endswith(".bin"):
            print(beats.load_state_dict(beats_checkpoint["model"]))
        else:
            filtered_checkpoint = {}
            prefix = "model.audio_tower."
            for key, value in beats_checkpoint.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix) :]
                    filtered_checkpoint[new_key] = value
            print(beats.load_state_dict(filtered_checkpoint, strict=False))
    else:
        beats_cfg = BEATsConfig()
        beats = BEATs(beats_cfg)
    return beats, beats_cfg
