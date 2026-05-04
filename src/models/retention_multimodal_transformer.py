from __future__ import annotations

import torch
import torch.nn as nn

from .model_base import (
    AUDIO_DIM,
    TEXT_DIM,
    VISUAL_DIM,
    BaselineModule,
    ModalityProjection,
    SinusoidalPositionalEncoding,
    apply_baseline,
    apply_tabular_gate,
    build_deviation_head,
    build_tabular_gate,
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
        c3 = self.conv3(x_t)
        c5 = self.conv5(x_t)
        c7 = self.conv7(x_t)
        out = (c3 + c5 + c7).transpose(1, 2)
        out = self.dropout(torch.nn.functional.gelu(self.proj(out)))
        return self.norm(residual + out)


class MultimodalRetentionTransformer(BaselineModule):
    def __init__(
        self,
        d_model: int = 320,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 668,
        dropout: float = 0.25,
        max_len: int = 4096,
        use_modality_embeddings: bool = True,
        n_tabular_features: int = 0,
        emb_visual_dim: int = VISUAL_DIM,
        emb_audio_dim: int = AUDIO_DIM,
        emb_text_dim: int = TEXT_DIM,
        use_conv_blocks: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_modality_embeddings = use_modality_embeddings
        self.vis_proj = ModalityProjection(emb_visual_dim, d_model, dropout)
        self.aud_proj = ModalityProjection(emb_audio_dim, d_model, dropout)
        self.txt_proj = ModalityProjection(emb_text_dim, d_model, dropout)
        if use_modality_embeddings:
            self.mod_emb = nn.Parameter(torch.zeros(3, d_model))
            nn.init.normal_(self.mod_emb, std=0.02)
        self.fusion = nn.Linear(3 * d_model, d_model)
        self.tabular_proj, self.tabular_gate = build_tabular_gate(n_tabular_features, d_model, dropout)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.use_conv_blocks = use_conv_blocks
        if use_conv_blocks:
            self.pre_conv = PreConvBlock(d_model, dropout)
            self.post_conv = PreConvBlock(d_model, dropout)
        else:
            self.pre_conv = nn.Identity()
            self.post_conv = nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.deviation_head = build_deviation_head(d_model, dropout)
        self.max_deviation = 1.0
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, embeddings: torch.Tensor, tabular: torch.Tensor | None = None, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        vis, aud, txt = split_embeddings(embeddings)
        h_vis, h_aud, h_txt = self.vis_proj(vis), self.aud_proj(aud), self.txt_proj(txt)
        if self.use_modality_embeddings:
            h_vis, h_aud, h_txt = h_vis + self.mod_emb[0], h_aud + self.mod_emb[1], h_txt + self.mod_emb[2]
        h = apply_tabular_gate(self.fusion(torch.cat([h_vis, h_aud, h_txt], dim=-1)), tabular, self.tabular_proj, self.tabular_gate)
        h = self.pre_conv(h)
        h = self.encoder(self.pos_enc(h), src_key_padding_mask=src_key_padding_mask)
        h = self.post_conv(h)
        return apply_baseline(torch.tanh(self.deviation_head(h).squeeze(-1)) * self.max_deviation, self._baseline, embeddings.size(1))
