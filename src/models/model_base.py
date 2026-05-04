"""Shared building blocks for retention prediction models (LSTM / Transformer)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


VISUAL_DIM = 768
AUDIO_DIM = 512
TEXT_DIM = 256


class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        return x + self.dropout(self.attn(normed, normed, normed)[0])


class ModalityProjection(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GatedFeatureProjection(nn.Module):
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(n_features, n_features), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(n_features, d_model), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.gate(x) * x)


class SinusoidalPositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class BaselineModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._baseline: torch.Tensor | None = None

    def set_baseline(self, baseline: torch.Tensor):
        self._baseline = baseline


def build_deviation_head(in_dim: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(in_dim // 2, 1))


def build_tabular_gate(n_tabular: int, hidden_size: int, dropout: float = 0.2):
    if n_tabular <= 0:
        return None, None
    return (nn.Sequential(nn.Linear(n_tabular, hidden_size), nn.GELU(), nn.Dropout(dropout)), nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid()))


def apply_tabular_gate(h: torch.Tensor, tabular: torch.Tensor | None, proj: nn.Module | None, gate: nn.Module | None) -> torch.Tensor:
    if tabular is not None and proj is not None and gate is not None:
        tab_h = proj(tabular)
        return h + gate(torch.cat([h, tab_h], dim=-1)) * tab_h
    return h


def lstm_forward_packed(lstm: nn.LSTM, x: torch.Tensor, mask: torch.Tensor | None, total_length: int) -> torch.Tensor:
    if mask is None:
        h, _ = lstm(x)
        return h
    lengths = (~mask).sum(dim=1).cpu()
    packed = nn.utils.rnn.pack_padded_sequence(x, lengths.clamp(min=1), batch_first=True, enforce_sorted=False)
    out_packed, _ = lstm(packed)
    h, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=total_length)
    return h


def apply_baseline(deviation: torch.Tensor, baseline: torch.Tensor | None, T: int) -> torch.Tensor:
    if baseline is None:
        return deviation
    bl = baseline.to(deviation.device)
    bl = torch.cat([bl, bl[-1].repeat(T - bl.numel())], dim=0) if bl.numel() < T else bl[:T]
    return bl.unsqueeze(0) + deviation


def split_embeddings(x: torch.Tensor):
    return x[..., :VISUAL_DIM], x[..., VISUAL_DIM : VISUAL_DIM + AUDIO_DIM], x[..., VISUAL_DIM + AUDIO_DIM :]


def init_model_weights(module: nn.Module, skip_gate: bool = False):
    for name, p in module.named_parameters():
        if "lstm" in name and p.dim() >= 2:
            nn.init.orthogonal_(p)
        elif skip_gate and "gate" in name:
            continue
        elif p.dim() > 1:
            nn.init.xavier_uniform_(p)
