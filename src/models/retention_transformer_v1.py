from __future__ import annotations

import torch
import torch.nn as nn

from .model_base import SinusoidalPositionalEncoding, init_model_weights


class RetentionTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 4, d_ff: int = 256, dropout: float = 0.2, max_len: int = 4096):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.input_proj = nn.Sequential(nn.Linear(n_features, d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        init_model_weights(self)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.output_head(self.encoder(self.pos_enc(self.input_proj(x)), src_key_padding_mask=src_key_padding_mask)).squeeze(-1)
