from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, TimesformerModel, VideoMAEImageProcessor, VideoMAEModel
from transformers.modeling_utils import PreTrainedModel

from .model_base import BaselineModule, apply_baseline, apply_tabular_gate, build_deviation_head, build_tabular_gate
from .model_base import SinusoidalPositionalEncoding as SinusoidalPE


VIDEOMAE_HIDDEN = 768
VIDEOMAE_NUM_FRAMES = 16
DEFAULT_BACKBONE = "MCG-NJU/videomae-base-finetuned-kinetics"
BACKBONE_REGISTRY = {
    "videomae-base": "MCG-NJU/videomae-base-finetuned-kinetics",
    "videomae-small": "MCG-NJU/videomae-small-finetuned-kinetics400",
    "timesformer-base": "facebook/timesformer-base-finetuned-k400",
}


def _load_backbone(backbone_name: str, device: torch.device):
    model_id = BACKBONE_REGISTRY.get(backbone_name, backbone_name)
    if "timesformer" in model_id.lower():
        _m = TimesformerModel.from_pretrained(model_id, local_files_only=True)
        model: PreTrainedModel = cast(PreTrainedModel, _m)
        model.to(device) 
        processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
    else:
        _m = VideoMAEModel.from_pretrained(model_id, local_files_only=True)
        model = cast(PreTrainedModel, _m)
        model.to(device) 
        processor = VideoMAEImageProcessor.from_pretrained(model_id, local_files_only=True)
    return model, processor, model.config.hidden_size


def _apply_lora(model: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        import logging

        logging.getLogger(__name__).warning("peft not installed, LoRA will not be applied to VideoMAE backbone")
        return model
    target_modules = [n for n, _ in model.named_modules() if any(k in n for k in ("query", "key", "value", "qkv", "q_proj", "k_proj", "v_proj"))]
    if not target_modules:
        target_modules = ["attention.self.query", "attention.self.key", "attention.self.value"]
    return get_peft_model(
        cast(PreTrainedModel, model), LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(set(target_modules)), bias="none", task_type=None)
    )


class TemporalRegressionHead(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2, d_ff: int = 512, dropout: float = 0.2, max_deviation: float = 1.0):
        super().__init__()
        self.max_deviation = max_deviation
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True),
            num_layers=n_layers,
        )
        self.head = build_deviation_head(d_model, dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return torch.tanh(self.head(self.temporal(self.pos_enc(h), src_key_padding_mask=mask)).squeeze(-1)) * self.max_deviation


def _interp_to_seconds(h: torch.Tensor, n_seconds: int) -> torch.Tensor:
    return F.interpolate(h.permute(0, 2, 1), size=n_seconds, mode="linear", align_corners=False).permute(0, 2, 1)


class VideoMAEFeatureExtractor(nn.Module):
    def __init__(self, backbone: str = DEFAULT_BACKBONE, device: torch.device | None = None):
        super().__init__()
        self._device = device or torch.device("cpu")
        self.backbone, self.processor, self.hidden_dim = _load_backbone(backbone, self._device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, frames: list[np.ndarray], clip_stride: int = 4) -> np.ndarray:
        T, clip_len = len(frames), VIDEOMAE_NUM_FRAMES
        emb_sum = np.zeros((T, self.hidden_dim), dtype=np.float64)
        emb_cnt = np.zeros(T, dtype=np.float64)
        for start in range(0, max(1, T - clip_len + 1), clip_stride):
            end = min(start + clip_len, T)
            clip = frames[start:end]
            if len(clip) < clip_len:
                clip = clip + [clip[-1]] * (clip_len - len(clip))
            hidden = self.backbone(self.processor(clip, return_tensors="pt")["pixel_values"].to(self._device)).last_hidden_state
            if hidden.dim() == 3 and hidden.size(1) > 1:
                n_t = min(end - start, hidden.size(1))
                step = hidden.size(1) / n_t
                for i in range(n_t):
                    emb_sum[start + i] += hidden[0, int(i * step)].cpu().float().numpy()
                    emb_cnt[start + i] += 1.0
            else:
                pooled = hidden.mean(dim=1)[0].cpu().float().numpy()
                for i in range(end - start):
                    emb_sum[start + i] += pooled
                    emb_cnt[start + i] += 1.0
        return (emb_sum / np.maximum(emb_cnt, 1.0)[:, None]).astype(np.float32)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values).last_hidden_state


class VideoMAERetention(BaselineModule):
    def __init__(
        self, backbone: str = DEFAULT_BACKBONE, lora_rank: int = 8, lora_alpha: int = 16, n_head_layers: int = 2, d_ff: int = 512, dropout: float = 0.2, max_deviation: float = 1.0
    ):
        super().__init__()
        self.backbone, self.processor, hidden_dim = _load_backbone(backbone, torch.device("cpu"))
        self.backbone = _apply_lora(self.backbone, rank=lora_rank, alpha=lora_alpha)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.head = TemporalRegressionHead(d_model=hidden_dim, n_layers=n_head_layers, d_ff=d_ff, dropout=dropout, max_deviation=max_deviation)

    def forward(self, pixel_values: torch.Tensor, n_seconds: int, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = _interp_to_seconds(self.proj(self.backbone(pixel_values).last_hidden_state), n_seconds)
        return apply_baseline(self.head(h, mask=padding_mask), self._baseline, n_seconds)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class VideoMAEHybridRetention(BaselineModule):
    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.2,
        n_tabular_features: int = 0,
        audio_dim: int = 512,
        text_dim: int = 256,
        max_deviation: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_deviation = max_deviation
        self.backbone, self.processor, backbone_dim = _load_backbone(backbone, torch.device("cpu"))
        self.backbone = _apply_lora(self.backbone, rank=lora_rank, alpha=lora_alpha)

        def _proj(in_d):
            return nn.Sequential(nn.Linear(in_d, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))

        self.vis_proj, self.aud_proj, self.txt_proj = _proj(backbone_dim), _proj(audio_dim), _proj(text_dim)
        self.mod_emb = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.mod_emb, std=0.02)
        self.fusion = nn.Linear(3 * d_model, d_model)
        self.tabular_proj, self.tabular_gate = build_tabular_gate(n_tabular_features, d_model, dropout)
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True),
            num_layers=n_layers,
        )
        self.deviation_head = build_deviation_head(d_model, dropout)
        for name, p in self.named_parameters():
            if "backbone" not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_video(self, pixel_values: torch.Tensor, n_seconds: int) -> torch.Tensor:
        return _interp_to_seconds(self.vis_proj(self.backbone(pixel_values).last_hidden_state), n_seconds) + self.mod_emb[0]

    def forward(
        self, vis_h: torch.Tensor, audio_emb: torch.Tensor, text_emb: torch.Tensor, tabular: torch.Tensor | None = None, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.fusion(torch.cat([vis_h, self.aud_proj(audio_emb) + self.mod_emb[1], self.txt_proj(text_emb) + self.mod_emb[2]], dim=-1))
        h = apply_tabular_gate(h, tabular, self.tabular_proj, self.tabular_gate)
        h = self.encoder(self.pos_enc(h), src_key_padding_mask=src_key_padding_mask)
        return apply_baseline(torch.tanh(self.deviation_head(h).squeeze(-1)) * self.max_deviation, self._baseline, vis_h.size(1))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
