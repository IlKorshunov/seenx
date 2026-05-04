"""
GatedFeatureProjection --> Residual BiLSTM --> LayerNorm --> Temporal Attention --> MLP head --> retention (B, T)
Supports head_type: cumulative | sigmoid | tanh (with optional baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_base import BaselineModule, MultiHeadTemporalAttention, apply_baseline, build_deviation_head, init_model_weights, lstm_forward_packed


class RetentionLSTM(BaselineModule):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        n_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
        n_attn_heads: int = 4,
        max_deviation: float = 1.0,
        head_type: str = "tanh",
    ):
        super().__init__()
        self.max_deviation = max_deviation
        self.head_type = head_type

        self.gate = nn.Sequential(nn.Linear(n_features, n_features), nn.Sigmoid())
        self.input_proj = nn.Sequential(nn.Linear(n_features, hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0.0
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.residual_proj = nn.Linear(hidden_size, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.attention = MultiHeadTemporalAttention(out_dim, n_attn_heads, dropout)
        self.deviation_head = build_deviation_head(out_dim, dropout)
        init_model_weights(self, skip_gate=True)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        proj = self.input_proj(self.gate(x) * x)
        h = lstm_forward_packed(self.lstm, proj, src_key_padding_mask, x.size(1))
        h = self.layer_norm(h + self.residual_proj(proj))
        h = self.attention(h)
        out = self.deviation_head(h).squeeze(-1)

        if self.head_type == "cumulative":
            return torch.clamp(100.0 + torch.cumsum(out, dim=1), 0.0, 100.0)
        if self.head_type == "sigmoid":
            return torch.sigmoid(out) * 100.0
        return apply_baseline(torch.tanh(out) * self.max_deviation, self._baseline, x.size(1))
