from __future__ import annotations

import torch
import torch.nn as nn

from .model_base import BaselineModule, GatedFeatureProjection, SinusoidalPositionalEncoding, apply_baseline, build_deviation_head, init_model_weights


class RetentionTransformer(BaselineModule):
    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.2,
        max_len: int = 4096,
        max_deviation: float = 1.0,
        head_type: str = "tanh",
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.head_type = head_type
        self.max_deviation = max_deviation
        self.input_proj = GatedFeatureProjection(n_features, d_model, dropout)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = build_deviation_head(d_model, dropout)
        init_model_weights(self, skip_gate=True)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(self.pos_enc(self.input_proj(x)), src_key_padding_mask=src_key_padding_mask)
        out = self.output_head(h).squeeze(-1)
        if self.head_type == "cumulative":
            return torch.clamp(100.0 + torch.cumsum(out, dim=1), 0.0, 100.0)
        if self.head_type == "sigmoid":
            return torch.sigmoid(out) * 100.0
        return apply_baseline(torch.tanh(out) * self.max_deviation, self._baseline, x.size(1))
