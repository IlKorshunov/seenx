"""Minimal BiLSTM: Linear projection --> BiLSTM --> LayerNorm+Linear --> retention (B, T)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_base import init_model_weights, lstm_forward_packed


class RetentionLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, n_layers: int = 2, dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_proj = nn.Sequential(nn.Linear(n_features, hidden_size), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0.0
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.output_head = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, 1))
        init_model_weights(self)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = lstm_forward_packed(self.lstm, self.input_proj(x), src_key_padding_mask, x.size(1))
        return self.output_head(h).squeeze(-1)
