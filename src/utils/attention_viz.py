"""
Universal attention weight extraction and visualization for all retention models.

Works with: RetentionTransformer, RetentionLSTM, MultimodalRetentionTransformer,
            MultimodalRetentionLSTM, VideoMAERetention, VideoMAEHybridRetention.

Uses forward hooks on nn.MultiheadAttention modules — no model code changes needed.

Usage:
    from src.utils.attention_viz import AttentionCapture, plot_attention_summary

    with AttentionCapture(model) as cap:
        output = model(inputs)
        weights = cap.get_weights()   # {layer_name: (B, H, T, T)}

    plot_attention_summary(weights, "video_id", "output/attention/video_id/")
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


class AttentionCapture:
    """Context manager that hooks into all nn.MultiheadAttention layers and captures weights.

    Usage:
        with AttentionCapture(model) as cap:
            model(input)
            weights = cap.get_weights()  # dict[str, Tensor(B, H, T, T)]
    """

    def __init__(self, model: nn.Module):
        self._model = model
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._weights: dict[str, torch.Tensor] = {}

    def __enter__(self) -> AttentionCapture:
        self._weights.clear()
        self._hooks.clear()

        for name, module in self._model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                self._register_hook(name, module)
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        return False

    def _register_hook(self, name: str, mha: nn.MultiheadAttention):
        original_forward = mha.forward

        capture = self

        def hooked_forward(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            out, attn_w = original_forward(*args, **kwargs)
            capture._weights[name] = attn_w.detach()
            return out, attn_w

        mha.forward = hooked_forward
        self._hooks.append(_ForwardOverrideHook(mha, original_forward))

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Return captured attention weights: {layer_name: (B, n_heads, T_q, T_k)}."""
        return dict(self._weights)

    def get_weights_numpy(self) -> dict[str, np.ndarray]:
        """Return captured attention weights as numpy arrays."""
        return {k: v.cpu().float().numpy() for k, v in self._weights.items()}


class _ForwardOverrideHook:
    """Mimics RemovableHook interface to restore original forward on remove()."""

    def __init__(self, module: nn.Module, original_forward):
        self.module = module
        self.original_forward = original_forward

    def remove(self):
        self.module.forward = self.original_forward


def attention_rollout(weights: dict[str, np.ndarray]) -> np.ndarray:
    """Compute attention rollout across all layers.

    Multiplies attention matrices from bottom to top, adding residual identity.
    Returns (T, T) rollout matrix (first batch element).
    """
    sorted_names = sorted(weights.keys())
    rollout = None
    for name in sorted_names:
        w = weights[name]
        if w.ndim == 4:
            w = w[0]
        attn = w.mean(axis=0)
        T = attn.shape[0]
        attn_with_residual = 0.5 * attn + 0.5 * np.eye(T)
        row_sums = attn_with_residual.sum(axis=-1, keepdims=True)
        attn_with_residual = attn_with_residual / np.maximum(row_sums, 1e-12)
        if rollout is None:
            rollout = attn_with_residual
        else:
            if rollout.shape != attn_with_residual.shape:
                break
            rollout = attn_with_residual @ rollout
    return rollout


def attention_entropy(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute per-head per-timestep attention entropy.

    Returns: {layer_name: (n_heads, T)} entropy values.
    """
    result = {}
    for name, w in weights.items():
        if w.ndim == 4:
            w = w[0]
        eps = 1e-12
        ent = -np.sum(w * np.log(w + eps), axis=-1)
        result[name] = ent
    return result


# ── Visualization functions ────────────────────────────────────────────────


def _lazy_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_attention_heatmap(weights: dict[str, np.ndarray], layer: str, head: int, video_id: str, output_path: str):
    """Plot a single attention head heatmap (T_q x T_k)."""
    plt = _lazy_mpl()
    w = weights[layer]
    if w.ndim == 4:
        w = w[0]
    attn = w[head]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Key (sec)")
    ax.set_ylabel("Query (sec)")
    ax.set_title(f"{video_id} — {layer} head {head}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_summary(weights: dict[str, np.ndarray], video_id: str, output_dir: str):
    """Plot all heads/layers in a grid and save individual heatmaps."""
    plt = _lazy_mpl()
    os.makedirs(output_dir, exist_ok=True)

    sorted_layers = sorted(weights.keys())
    n_layers = len(sorted_layers)
    if n_layers == 0:
        return

    sample_w = weights[sorted_layers[0]]
    if sample_w.ndim == 4:
        sample_w = sample_w[0]
    n_heads = sample_w.shape[0]

    for li, layer in enumerate(sorted_layers):
        for hi in range(n_heads):
            plot_attention_heatmap(weights, layer, hi, video_id, os.path.join(output_dir, f"layer_{li}_head_{hi}.png"))

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(3 * n_heads, 3 * n_layers), squeeze=False)
    for li, layer in enumerate(sorted_layers):
        w = weights[layer]
        if w.ndim == 4:
            w = w[0]
        for hi in range(n_heads):
            ax = axes[li][hi]
            ax.imshow(w[hi], aspect="auto", cmap="viridis", interpolation="nearest")
            if li == 0:
                ax.set_title(f"H{hi}", fontsize=9)
            if hi == 0:
                ax.set_ylabel(f"L{li}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"Attention Summary — {video_id}", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "attention_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_over_retention(weights: dict[str, np.ndarray], retention: np.ndarray, video_id: str, output_path: str):
    """Plot attention rollout overlaid on the retention curve.

    Shows which seconds the model attends to when predicting each second.
    """
    plt = _lazy_mpl()
    rollout = attention_rollout(weights)
    if rollout is None:
        return

    T = rollout.shape[0]
    retention = retention[:T]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[1, 2], sharex=True)

    ax1.plot(range(T), retention, color="#2196F3", linewidth=1.5, label="retention")
    ax1.set_ylabel("Retention")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    im = ax2.imshow(rollout, aspect="auto", cmap="magma", interpolation="nearest", extent=[0, T, T, 0])
    ax2.set_xlabel("Attended-to second (key)")
    ax2.set_ylabel("Predicting second (query)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(f"Attention Rollout vs Retention — {video_id}", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_entropy_chart(weights: dict[str, np.ndarray], video_id: str, output_path: str):
    """Plot per-head attention entropy over time.

    Low entropy = focused attention, high entropy = diffuse/uniform attention.
    """
    plt = _lazy_mpl()
    ent_dict = attention_entropy(weights)
    sorted_layers = sorted(ent_dict.keys())
    n_layers = len(sorted_layers)
    if n_layers == 0:
        return

    fig, axes = plt.subplots(n_layers, 1, figsize=(14, 3 * n_layers), sharex=True, squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for li, layer in enumerate(sorted_layers):
        ax = axes[li][0]
        ent = ent_dict[layer]
        n_heads, T = ent.shape
        for hi in range(n_heads):
            ax.plot(range(T), ent[hi], alpha=0.7, linewidth=1, color=colors[hi % len(colors)], label=f"head {hi}")
        ax.set_ylabel(f"L{li} entropy")
        ax.legend(fontsize=7, ncol=min(n_heads, 4), loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Second")
    fig.suptitle(f"Attention Entropy — {video_id}", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_attention_weights(weights: dict[str, np.ndarray], output_path: str):
    """Save raw attention weights to .npz for further analysis."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **weights)


def visualize_all(weights: dict[str, np.ndarray], retention: np.ndarray, video_id: str, output_dir: str):
    """Run all visualizations for a single video."""
    out = os.path.join(output_dir, "attention", video_id)
    os.makedirs(out, exist_ok=True)

    plot_attention_summary(weights, video_id, out)
    plot_attention_over_retention(weights, retention, video_id, os.path.join(out, "attention_vs_retention.png"))
    plot_attention_entropy_chart(weights, video_id, os.path.join(out, "attention_entropy.png"))
    save_attention_weights(weights, os.path.join(out, "attention_weights.npz"))


# ── CLI entry point ────────────────────────────────────────────────────────


def _cli():
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    p = argparse.ArgumentParser(description="Visualize attention weights from a trained retention model checkpoint.")
    p.add_argument("--model-path", required=True, help="Path to model .pt checkpoint")
    p.add_argument("--model-type", choices=["transformer", "lstm", "multimodal_transformer", "multimodal_lstm", "videomae"], default="transformer")
    p.add_argument("--video-id", default=None, help="Single video ID (or --all)")
    p.add_argument("--all", action="store_true", help="Run on all videos")
    p.add_argument("--output-dir", default="output/attention_viz")
    p.add_argument("--output-dir-features", default="output")
    p.add_argument("--snapshot-dir", default="data")
    p.add_argument("--embeddings-root", default="embeddings")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    feature_cols = checkpoint.get("feature_cols", [])

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from train.common.seq_data_utils import FeatureNormalizer, load_aligned_embeddings_for_videos, load_all_merged, predict_video_multimodal

    video_dfs = load_all_merged(args.output_dir_features, args.snapshot_dir, use_curve_raw=True, emb_pca_components=0)

    model = _build_model(args.model_type, checkpoint, device)
    if model is None:
        print(f"Could not build model of type '{args.model_type}'")
        return

    model.eval()

    video_ids = [args.video_id] if args.video_id else sorted(video_dfs.keys())

    need_embeddings = args.model_type in ("multimodal_transformer", "multimodal_lstm", "videomae")
    video_embeddings = {}
    if need_embeddings:
        video_embeddings = load_aligned_embeddings_for_videos(video_dfs, args.embeddings_root)

    normalizer = FeatureNormalizer()
    normalizer.fit(video_dfs, feature_cols)

    for vid in video_ids:
        if vid not in video_dfs:
            print(f"Video {vid} not found in loaded data, skipping")
            continue

        df = video_dfs[vid]
        retention = df["retention"].values.astype(np.float32)

        with AttentionCapture(model) as cap:
            if need_embeddings:
                emb = video_embeddings.get(vid)
                predict_video_multimodal(model, df, emb, feature_cols, normalizer, device, 128)
            else:
                from train.common.seq_data_utils import predict_video

                predict_video(model, df, feature_cols, normalizer, device, 128)
            weights = cap.get_weights_numpy()

        if not weights:
            print(f"No attention weights captured for {vid}")
            continue

        print(f"Captured {len(weights)} attention layers for {vid}")
        visualize_all(weights, retention, vid, args.output_dir)
        print(f"  Saved to {os.path.join(args.output_dir, 'attention', vid)}/")


def _build_model(model_type: str, checkpoint: dict, device: torch.device):
    """Reconstruct model from checkpoint."""
    state = checkpoint.get("model_state_dict", checkpoint)
    feature_cols = checkpoint.get("feature_cols", [])
    n_features = len(feature_cols)

    if model_type == "transformer":
        from src.models import RetentionTransformer

        d_model = _infer_dim(state, "input_proj.proj.0.weight", dim=0)
        model = RetentionTransformer(n_features=n_features, d_model=d_model)
    elif model_type == "lstm":
        from src.models import RetentionLSTM

        hidden = _infer_dim(state, "input_proj.0.weight", dim=0)
        model = RetentionLSTM(n_features=n_features, hidden_size=hidden)
    elif model_type == "multimodal_transformer":
        from src.models import MultimodalRetentionTransformer
        from src.models.retention_multimodal_transformer import TEXT_DIM as _DEFAULT_TEXT_DIM

        d_model = _infer_dim(state, "vis_proj.proj.0.weight", dim=0)
        n_tab = _infer_dim(state, "tabular_proj.0.weight", dim=1) if "tabular_proj.0.weight" in state else 0
        txt_dim = _DEFAULT_TEXT_DIM
        if "txt_proj.proj.0.weight" in state:
            txt_dim = state["txt_proj.proj.0.weight"].shape[1]
        model = MultimodalRetentionTransformer(d_model=d_model, n_tabular_features=n_tab, emb_text_dim=txt_dim)
    elif model_type == "multimodal_lstm":
        from src.models import MultimodalRetentionLSTM

        hidden = _infer_dim(state, "vis_proj.proj.0.weight", dim=0)
        n_tab = _infer_dim(state, "tabular_proj.0.weight", dim=1) if "tabular_proj.0.weight" in state else 0
        model = MultimodalRetentionLSTM(hidden_size=hidden, n_tabular_features=n_tab)
    else:
        return None

    model.load_state_dict(state, strict=False)
    return model.to(device)


def _infer_dim(state: dict, key: str, dim: int = 0) -> int:
    if key in state:
        return state[key].shape[dim]
    return 256


if __name__ == "__main__":
    _cli()
