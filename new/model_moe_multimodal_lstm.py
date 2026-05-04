"""Mixture-of-experts multimodal LSTM for retention (experimental, under ``new/``).

Shared temporal backbone (fusion + LSTM + attention) with K lightweight expert
heads. A gate mixes experts using either a learned softmax over the pooled
sequence representation, a hard one-hot route from ``video_cluster``, or a
convex combination of both (hybrid).

Optional ``sarimax_baseline`` matches the SARIMAX hybrid: final output is
baseline + mixture deviation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_base import (
    AUDIO_DIM,
    TEXT_DIM,
    VISUAL_DIM,
    ModalityProjection,
    MultiHeadTemporalAttention,
    apply_tabular_gate,
    build_tabular_gate,
    init_model_weights,
    lstm_forward_packed,
    split_embeddings,
)


class PreConvBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_t = x.transpose(1, 2)
        out = (self.conv3(x_t) + self.conv5(x_t) + self.conv7(x_t)).transpose(1, 2)
        out = self.dropout(torch.nn.functional.gelu(self.proj(out)))
        return self.norm(residual + out)


def _masked_mean_pool(h: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    """h: [B,T,D], padding_mask: [B,T] True = pad. Returns [B,D]."""
    if padding_mask is None:
        return h.mean(dim=1)
    valid = (~padding_mask).to(h.dtype).unsqueeze(-1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (h * valid).sum(dim=1) / denom


class ExpertDeviationHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        hid = max(in_dim // 2, 32)
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hid), nn.GELU(), nn.Dropout(dropout), nn.Linear(hid, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class MoeMultimodalRetentionLSTM(nn.Module):
    """Multimodal LSTM with K deviation experts and a cluster-aware gate."""

    def __init__(
        self,
        hidden_size: int = 320,
        n_layers: int = 3,
        dropout: float = 0.25,
        bidirectional: bool = True,
        n_tabular_features: int = 0,
        n_attn_heads: int = 4,
        emb_visual_dim: int = VISUAL_DIM,
        emb_audio_dim: int = AUDIO_DIM,
        emb_text_dim: int = TEXT_DIM,
        use_conv_blocks: bool = False,
        n_experts: int = 4,
        n_cluster_buckets: int = 32,
        cluster_embed_dim: int = 32,
        routing_mode: str = "soft",
        hybrid_alpha: float = 0.5,
    ):
        super().__init__()
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        routing_mode = routing_mode.lower()
        if routing_mode not in ("soft", "hard_cluster", "hybrid"):
            raise ValueError("routing_mode must be soft | hard_cluster | hybrid")
        self.routing_mode = routing_mode
        self.hybrid_alpha = float(hybrid_alpha)
        self.n_experts = n_experts
        d = hidden_size
        self.vis_proj = ModalityProjection(emb_visual_dim, d, dropout)
        self.aud_proj = ModalityProjection(emb_audio_dim, d, dropout)
        self.txt_proj = ModalityProjection(emb_text_dim, d, dropout)
        self.mod_emb = nn.Parameter(torch.zeros(3, d))
        nn.init.normal_(self.mod_emb, std=0.02)

        self.fusion = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout))
        self.tabular_proj, self.tabular_gate = build_tabular_gate(n_tabular_features, d, dropout)
        self.pre_conv = PreConvBlock(d, dropout) if use_conv_blocks else nn.Identity()

        self.lstm = nn.LSTM(input_size=d, hidden_size=d, num_layers=n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0.0)
        out_dim = d * (2 if bidirectional else 1)
        self.residual_proj = nn.Linear(d, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.post_conv = PreConvBlock(out_dim, dropout) if use_conv_blocks else nn.Identity()
        self.attention = MultiHeadTemporalAttention(out_dim, n_attn_heads, dropout)

        self.n_cluster_buckets = max(int(n_cluster_buckets), 1)
        self.cluster_embed = nn.Embedding(self.n_cluster_buckets, cluster_embed_dim)
        gate_in = out_dim + cluster_embed_dim
        self.gate = nn.Sequential(nn.Linear(gate_in, gate_in // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(gate_in // 2, n_experts))
        self.experts = nn.ModuleList(ExpertDeviationHead(out_dim, dropout) for _ in range(n_experts))
        self.max_deviation = 1.0
        init_model_weights(self)

    def _mixture_weights(self, pooled: torch.Tensor, cluster_ids: torch.Tensor | None) -> torch.Tensor:
        """Returns [B, K] non-negative weights summing to 1."""
        b = pooled.size(0)
        device = pooled.device
        if cluster_ids is None:
            cid = torch.zeros(b, dtype=torch.long, device=device)
        else:
            cid = cluster_ids.long().clamp(0, self.n_cluster_buckets - 1)
        cemb = self.cluster_embed(cid)
        logits = self.gate(torch.cat([pooled, cemb], dim=-1))
        w_soft = F.softmax(logits, dim=-1)

        if self.routing_mode == "soft":
            return w_soft

        expert_idx = cid % self.n_experts
        w_hard = F.one_hot(expert_idx, num_classes=self.n_experts).to(dtype=pooled.dtype)

        if self.routing_mode == "hard_cluster":
            return w_hard

        a = self.hybrid_alpha
        return a * w_soft + (1.0 - a) * w_hard

    def forward(
        self,
        embeddings: torch.Tensor,
        tabular: torch.Tensor | None = None,
        sarimax_baseline: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        return_expert_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        vis, aud, txt = split_embeddings(embeddings)
        h = self.fusion(torch.cat([self.vis_proj(vis) + self.mod_emb[0], self.aud_proj(aud) + self.mod_emb[1], self.txt_proj(txt) + self.mod_emb[2]], dim=-1))
        h = apply_tabular_gate(h, tabular, self.tabular_proj, self.tabular_gate)
        h = self.pre_conv(h)
        start_hidden = h
        t_len = embeddings.size(1)
        h = lstm_forward_packed(self.lstm, h, src_key_padding_mask, t_len)
        h = self.layer_norm(h + self.residual_proj(start_hidden))
        h = self.post_conv(h)
        h = self.attention(h)

        pooled = _masked_mean_pool(h, src_key_padding_mask)
        weights = self._mixture_weights(pooled, cluster_ids)

        expert_stack = torch.stack([expert(h) for expert in self.experts], dim=-1)
        deviation = (expert_stack * weights.unsqueeze(1)).sum(dim=-1)
        deviation = torch.tanh(deviation) * self.max_deviation

        if sarimax_baseline is None:
            out = deviation
        else:
            out = sarimax_baseline.to(deviation.device) + deviation
        if return_expert_weights:
            return out, weights
        return out
