from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_BACKBONE = "deepvk/deberta-v1-base"

BACKBONE_REGISTRY = {
    "deberta-base": "deepvk/deberta-v1-base",
    "rubert-base": "ai-forever/ruBert-base",
    "rubert-large": "ai-forever/ruBert-large",
    "multilingual-e5": "intfloat/multilingual-e5-base",
    "labse": "sentence-transformers/LaBSE",
}


def _load_backbone(backbone_name: str, device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    model_id = BACKBONE_REGISTRY.get(backbone_name, backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_pretrained(model_id, local_files_only=True).to(device)
    hidden_dim = model.config.hidden_size
    return model, tokenizer, hidden_dim


def _apply_lora(model: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
    """Wrap a HuggingFace model with LoRA adapters on attention projections."""
    from peft import LoraConfig, get_peft_model

    target_modules = []
    for name, mod in model.named_modules():
        if "Linear" not in type(mod).__name__:
            continue
        if any(k in name for k in ("query", "key", "value", "q_proj", "k_proj", "v_proj", "query_proj", "key_proj", "value_proj", "in_proj")):
            target_modules.append(name)

    if not target_modules:
        target_modules = ["attention.self.in_proj"]

    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(set(target_modules)), bias="none", task_type=None)
    return get_peft_model(model, config)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TemporalRegressionHead(nn.Module):
    """Transformer encoder layers + MLP for per-second retention from text hidden states."""

    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2, d_ff: int = 512, dropout: float = 0.2, max_deviation: float = 1.0):
        super().__init__()
        self.max_deviation = max_deviation
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.pos_enc(h)
        h = self.temporal(h, src_key_padding_mask=mask)
        return torch.tanh(self.head(h).squeeze(-1)) * self.max_deviation


def _encode_segments(texts: list[str], model, tokenizer, device, batch_size: int = 32, max_length: int = 512) -> list[np.ndarray]:
    """Encode a list of text segments, returning per-segment mean-pooled embeddings.

    Returns list of (hidden_dim,) arrays, one per segment.
    """
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
            for j in range(pooled.size(0)):
                all_embs.append(pooled[j].cpu().float().numpy())
    return all_embs


def _align_segments_to_1fps(embs: list[np.ndarray], seg_meta: list[dict], duration_sec: int) -> np.ndarray:
    """Align per-segment embeddings to per-second via overlap-weighted average."""
    if not embs:
        return np.zeros((duration_sec, embs[0].shape[0] if embs else 768), dtype=np.float32)
    hidden_dim = embs[0].shape[0]
    out = np.zeros((duration_sec, hidden_dim), dtype=np.float64)
    weight = np.zeros(duration_sec, dtype=np.float64)

    for i, seg in enumerate(seg_meta[: len(embs)]):
        s, e = float(seg.get("start", 0)), float(seg.get("end", 0))
        for t in range(max(0, int(s)), min(duration_sec, int(np.ceil(e)))):
            t_start, t_end = float(t), float(t + 1)
            overlap = max(0.0, min(t_end, e) - max(t_start, s))
            if overlap > 1e-6:
                out[t] += embs[i] * overlap
                weight[t] += overlap

    weight = np.maximum(weight, 1e-12)
    return (out / weight[:, None]).astype(np.float32)


class BERTFeatureExtractor(nn.Module):
    """Mode A: frozen BERT backbone -> per-second text embeddings."""

    def __init__(self, backbone: str = DEFAULT_BACKBONE, device: torch.device | None = None):
        super().__init__()
        self._device = device or torch.device("cpu")
        self.backbone, self.tokenizer, self.hidden_dim = _load_backbone(backbone, self._device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, texts: list[str], seg_meta: list[dict], duration_sec: int, batch_size: int = 32) -> np.ndarray:
        """Extract per-second embeddings from transcript segments.

        Args:
            texts: list of segment text strings
            seg_meta: list of {"start": float, "end": float} dicts
            duration_sec: video duration in seconds

        Returns:
            (T, hidden_dim) float32 numpy array
        """
        if not texts:
            return np.zeros((duration_sec, self.hidden_dim), dtype=np.float32)

        embs = _encode_segments(texts, self.backbone, self.tokenizer, self._device, batch_size=batch_size)
        return _align_segments_to_1fps(embs, seg_meta, duration_sec)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class BERTRetention(nn.Module):
    """Mode B: BERT + LoRA -> per-second retention from transcript only.

    Processes transcript segments through LoRA-adapted BERT, aligns to per-second,
    then feeds through a temporal regression head.
    """

    def __init__(
        self, backbone: str = DEFAULT_BACKBONE, lora_rank: int = 8, lora_alpha: int = 16, n_head_layers: int = 2, d_ff: int = 512, dropout: float = 0.2, max_deviation: float = 1.0
    ):
        super().__init__()
        self.backbone, self.tokenizer, self.hidden_dim = _load_backbone(backbone, torch.device("cpu"))
        self.backbone = _apply_lora(self.backbone, rank=lora_rank, alpha=lora_alpha)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.head = TemporalRegressionHead(d_model=self.hidden_dim, n_layers=n_head_layers, d_ff=d_ff, dropout=dropout, max_deviation=max_deviation)
        self._baseline = None

    def set_baseline(self, baseline: torch.Tensor):
        self._baseline = baseline

    def forward(self, text_emb: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            text_emb: (B, T, hidden_dim) pre-aligned per-second text embeddings
            padding_mask: (B, T) True=pad
        """
        h = self.proj(text_emb)
        deviation = self.head(h, mask=padding_mask)

        if self._baseline is not None:
            T = text_emb.size(1)
            bl = self._baseline.to(h.device)
            if bl.numel() < T:
                bl = torch.cat([bl, bl[-1:].expand(T - bl.numel())])
            else:
                bl = bl[:T]
            return bl.unsqueeze(0) + deviation
        return deviation

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class BERTHybridRetention(nn.Module):
    """Mode C: BERT LoRA text branch fused with visual + audio + tabular.

    Similar to MultimodalRetentionTransformer but the text branch is a
    pretrained BERT model with LoRA instead of precomputed USER2 embeddings.
    """

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
        visual_dim: int = 768,
        audio_dim: int = 512,
        max_deviation: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_deviation = max_deviation

        self.backbone, self.tokenizer, backbone_dim = _load_backbone(backbone, torch.device("cpu"))
        self.backbone = _apply_lora(self.backbone, rank=lora_rank, alpha=lora_alpha)
        self.txt_proj = nn.Sequential(nn.Linear(backbone_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.vis_proj = nn.Sequential(nn.Linear(visual_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.aud_proj = nn.Sequential(nn.Linear(audio_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))

        self.mod_emb = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.mod_emb, std=0.02)
        self.fusion = nn.Linear(3 * d_model, d_model)

        self.tabular_proj = None
        self.tabular_gate = None
        if n_tabular_features > 0:
            self.tabular_proj = nn.Sequential(nn.Linear(n_tabular_features, d_model), nn.GELU(), nn.Dropout(dropout))
            self.tabular_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.deviation_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self._baseline = None
        self._init_weights()

    def set_baseline(self, baseline: torch.Tensor):
        self._baseline = baseline

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "backbone" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, txt_h: torch.Tensor, visual_emb: torch.Tensor, audio_emb: torch.Tensor, tabular: torch.Tensor | None = None, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            txt_h: (B, T, backbone_dim) per-second BERT embeddings (pre-aligned)
            visual_emb: (B, T, visual_dim)
            audio_emb: (B, T, audio_dim)
            tabular: (B, T, n_tab) optional
            src_key_padding_mask: (B, T) optional
        """
        h_txt = self.txt_proj(txt_h) + self.mod_emb[0]
        h_vis = self.vis_proj(visual_emb) + self.mod_emb[1]
        h_aud = self.aud_proj(audio_emb) + self.mod_emb[2]

        fused = torch.cat([h_txt, h_vis, h_aud], dim=-1)
        h = self.fusion(fused)

        if tabular is not None and self.tabular_proj is not None:
            tab_h = self.tabular_proj(tabular)
            gate = self.tabular_gate(torch.cat([h, tab_h], dim=-1))
            h = h + gate * tab_h

        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        deviation = torch.tanh(self.deviation_head(h).squeeze(-1)) * self.max_deviation

        if self._baseline is not None:
            T = txt_h.size(1)
            bl = self._baseline.to(txt_h.device)
            if bl.numel() < T:
                bl = torch.cat([bl, bl[-1:].expand(T - bl.numel())])
            else:
                bl = bl[:T]
            return bl.unsqueeze(0) + deviation
        return deviation

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
