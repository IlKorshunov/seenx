"""
Multimodal BiLSTM: per-modality projection --> fusion --> optional tabular gate
--> Residual BiLSTM --> LayerNorm --> Temporal Attention --> MLP deviation + baseline --> retention (B, T)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_base import (
    AUDIO_DIM,
    TEXT_DIM,
    VISUAL_DIM,
    BaselineModule,
    ModalityProjection,
    MultiHeadTemporalAttention,
    apply_baseline,
    apply_tabular_gate,
    build_deviation_head,
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
        # x shape: (B, T, d_model)
        residual = x
        x_t = x.transpose(1, 2)
        c3 = self.conv3(x_t)
        c5 = self.conv5(x_t)
        c7 = self.conv7(x_t)
        out = (c3 + c5 + c7).transpose(1, 2)
        out = self.dropout(torch.nn.functional.gelu(self.proj(out)))
        return self.norm(residual + out)


class MultimodalRetentionLSTM(BaselineModule):
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        d = hidden_size
        self.vis_proj = ModalityProjection(emb_visual_dim, d, dropout)
        self.aud_proj = ModalityProjection(emb_audio_dim, d, dropout)
        self.txt_proj = ModalityProjection(emb_text_dim, d, dropout)
        self.mod_emb = nn.Parameter(torch.zeros(3, d))
        nn.init.normal_(self.mod_emb, std=0.02)
        self.fusion = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout))
        self.tabular_proj, self.tabular_gate = build_tabular_gate(n_tabular_features, d, dropout)
        self.use_conv_blocks = use_conv_blocks
        if use_conv_blocks:
            self.pre_conv = PreConvBlock(d, dropout)
        else:
            self.pre_conv = nn.Identity()
        self.lstm = nn.LSTM(input_size=d, hidden_size=d, num_layers=n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0.0)
        out_dim = d * (2 if bidirectional else 1)
        self.residual_proj = nn.Linear(d, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        if use_conv_blocks:
            self.post_conv = PreConvBlock(out_dim, dropout)
        else:
            self.post_conv = nn.Identity()
        self.attention = MultiHeadTemporalAttention(out_dim, n_attn_heads, dropout)
        self.deviation_head = build_deviation_head(out_dim, dropout)
        self.max_deviation = 1.0
        init_model_weights(self)

    def forward(self, embeddings: torch.Tensor, tabular: torch.Tensor | None = None, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        vis, aud, txt = split_embeddings(embeddings)
        h = self.fusion(torch.cat([self.vis_proj(vis) + self.mod_emb[0], self.aud_proj(aud) + self.mod_emb[1], self.txt_proj(txt) + self.mod_emb[2]], dim=-1))
        h = apply_tabular_gate(h, tabular, self.tabular_proj, self.tabular_gate)
        h = self.pre_conv(h)
        start_hidden = h
        h = lstm_forward_packed(self.lstm, h, src_key_padding_mask, embeddings.size(1))
        h = self.layer_norm(h + self.residual_proj(start_hidden))
        h = self.post_conv(h)
        h = self.attention(h)
        deviation = torch.tanh(self.deviation_head(h).squeeze(-1)) * self.max_deviation
        return apply_baseline(deviation, self._baseline, embeddings.size(1))
