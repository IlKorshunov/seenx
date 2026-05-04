"""FT-Transformer feature importance.
Uses CLS attention and input gradients to rank tabular features.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from .utils import aggregate_per_video, default_output_dir, feature_group_of, load_all_videos, prepare_X_y, save_importance_csv


    class FeatureTokenizer(nn.Module):
        def __init__(self, n_features: int, d_model: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, d_model))
            self.bias = nn.Parameter(torch.zeros(n_features, d_model))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

    class FTTransformer(nn.Module):
        def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1) -> None:
            super().__init__()
            self.n_features = n_features
            self.d_model = d_model

            self.tokenizer = FeatureTokenizer(n_features, d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

            self.attn_weights: list[torch.Tensor] = []
            self._hooks: list = []

        def forward(self, x: torch.Tensor, return_attn: bool = False) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
            batch = x.size(0)
            tokens = self.tokenizer(x)
            cls = self.cls_token.expand(batch, -1, -1)
            seq = torch.cat([cls, tokens], dim=1)

            attn_list = []
            if return_attn:
                h = seq
                for layer in self.encoder.layers:
                    h_norm = layer.norm1(h)
                    attn_out, attn_w = layer.self_attn(
                        h_norm,
                        h_norm,
                        h_norm,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    attn_list.append(attn_w.detach())
                    h = h + layer.dropout1(attn_out)
                    h_norm2 = layer.norm2(h)
                    h = h + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(h_norm2)))))
                encoded = h
            else:
                encoded = self.encoder(seq)

            cls_out = encoded[:, 0, :]
            pred = self.head(cls_out).squeeze(-1)

            return pred, attn_list if return_attn else None


def extract_attention_importance(model: FTTransformer, X_arr: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, attn_list = model(X_t, return_attn=True)

    cls_to_feat_per_layer = []
    feat_to_feat_per_layer = []

    for attn_w in attn_list:
        attn_mean = attn_w.mean(dim=1).mean(dim=0)
        cls_row = attn_mean[0, 1:].cpu().numpy()
        cls_to_feat_per_layer.append(cls_row)
        feat_block = attn_mean[1:, 1:].cpu().numpy()
        feat_to_feat_per_layer.append(feat_block)

    return np.mean(cls_to_feat_per_layer, axis=0), np.mean(feat_to_feat_per_layer, axis=0)


def extract_gradient_importance(model: FTTransformer, X_arr: np.ndarray, y_arr: np.ndarray, device: str = "cpu") -> np.ndarray:
    X_t = torch.tensor(X_arr, dtype=torch.float32, device=device, requires_grad=True)
    y_t = torch.tensor(y_arr, dtype=torch.float32, device=device)

    model.eval()
    pred, _ = model(X_t, return_attn=False)
    loss = nn.MSELoss()(pred, y_t)
    loss.backward()

    grads = X_t.grad.abs().cpu().numpy()
    return grads.mean(axis=0)


def plot_attention_importance(importance_df: pd.DataFrame, title: str, out_path: str | Path, top_n: int = 30) -> None:
    top = importance_df.head(top_n).copy()
    groups = top["group"].unique()
    palette = plt.cm.tab20.colors
    group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    colors = [group_colors[g] for g in top["group"]]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=9)

    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=group_colors[g], label=g) for g in sorted(group_colors)]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7, ncol=2)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_attention_heatmap(attn_matrix: np.ndarray, feature_names: list[str], out_path: str | Path, top_n: int = 25) -> None:
    n = min(top_n, len(feature_names))
    row_sums = attn_matrix.sum(axis=1)
    top_idx = np.argsort(row_sums)[::-1][:n]
    top_names = [feature_names[i] for i in top_idx]
    sub_matrix = attn_matrix[np.ix_(top_idx, top_idx)]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(7, n * 0.4)))
    im = ax.imshow(sub_matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(top_names, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_names, fontsize=7)
    ax.set_title(f"Feature-to-Feature Attention (top {n})")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_combined_importance(attn_df: pd.DataFrame, grad_df: pd.DataFrame, out_path: str | Path, top_n: int = 25) -> None:
    def _norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    attn_norm = _norm(attn_df.set_index("feature")["importance"])
    grad_norm = _norm(grad_df.set_index("feature")["importance"])

    combined = pd.DataFrame({"attention": attn_norm, "gradient": grad_norm}).fillna(0.0)
    combined["avg"] = combined.mean(axis=1)
    combined = combined.sort_values("avg", ascending=False).head(top_n)

    x = np.arange(len(combined))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, combined["attention"], width, label="Attention (CLS→feat)", color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, combined["gradient"], width, label="Gradient saliency", color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(combined.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalized importance")
    ax.set_title(f"FT-Transformer: Attention vs Gradient Importance — top {top_n}")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.6)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_training_curve(losses: list[float], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("FT-Transformer Training Curve")
    ax.set_yscale("log")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def _train_with_tracking(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    device: str = "cpu",
) -> tuple[FTTransformer, list[float]]:
    n_features = X_arr.shape[1]
    model = FTTransformer(n_features=n_features, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout).to(device)

    X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_arr, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(X_arr)), shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    epoch_losses = []
    model.train()
    for epoch in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred, _ = model(xb, return_attn=False)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        epoch_losses.append(ep_loss / max(n_batches, 1))
        scheduler.step()

    model.eval()
    return model, epoch_losses


def compute_loo_transformer_importance(
    X: pd.DataFrame, y: pd.Series, scaler: StandardScaler, epochs: int = 150, d_model: int = 32, n_heads: int = 2, n_layers: int = 1, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    X_arr = scaler.transform(X.values.astype(float))
    y_arr = y.values.astype(float)
    n = len(X_arr)

    if n < 5:
        print(f"  LOO: only {n} samples, training on full data")
        model, _ = _train_with_tracking(X_arr, y_arr, d_model=d_model, n_heads=n_heads, n_layers=n_layers, epochs=epochs, device=device)
        attn_imp, _ = extract_attention_importance(model, X_arr, device)
        grad_imp = extract_gradient_importance(model, X_arr, y_arr, device)
        return attn_imp, grad_imp

    loo = LeaveOneOut()
    attn_folds = []
    grad_folds = []

    for fold_i, (train_idx, test_idx) in enumerate(loo.split(X_arr)):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        model, _ = _train_with_tracking(X_train, y_train, d_model=d_model, n_heads=n_heads, n_layers=n_layers, epochs=epochs, device=device)
        attn_imp, _ = extract_attention_importance(model, X_test, device)
        grad_imp = extract_gradient_importance(model, X_test, y_test, device)
        attn_folds.append(attn_imp)
        grad_folds.append(grad_imp)
        print(f"    LOO fold {fold_i + 1}/{n} done")

    return np.mean(attn_folds, axis=0), np.mean(grad_folds, axis=0)


def run_transformer_importance(
    output_dir: str = "output",
    results_dir: str | None = None,
    target: str = "target_avg_retention",
    top_n: int = 30,
    epochs: int = 200,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    use_loo: bool = False,
    device: str | None = None,
) -> dict[str, object]: 
    if results_dir is None:
        results_dir = default_output_dir()
    results_dir = Path(results_dir) / "transformer"
    results_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading videos from {output_dir}")
    video_dfs = load_all_videos(output_dir)
    print(f"  Found {len(video_dfs)} video(s)")

    agg_df = aggregate_per_video(video_dfs)
    X, y = prepare_X_y(agg_df, target=target)
    print(f"  Feature matrix: {X.shape}, target: {target}")

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_arr = scaler.fit_transform(X.values.astype(float))
    y_arr = y.values.astype(float)

    if len(X) < 10:
        d_model = min(d_model, 32)
        n_heads = min(n_heads, 2)
        n_layers = min(n_layers, 1)
        epochs = max(epochs, 300)
        print(f"  Small dataset: using d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

    while d_model % n_heads != 0:
        n_heads -= 1
    print(f"  Model config: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, epochs={epochs}")

    if use_loo and len(X) >= 5:
        print("Computing LOO cross-validated transformer importance")
        attn_imp_arr, grad_imp_arr = compute_loo_transformer_importance(
            X, y, scaler, epochs=max(100, epochs // 2), d_model=d_model, n_heads=n_heads, n_layers=n_layers, device=device
        )
        print("Training final model on full data for attention heatmap")
        model, losses = _train_with_tracking(X_arr, y_arr, d_model=d_model, n_heads=n_heads, n_layers=n_layers, epochs=epochs, device=device)
        _, attn_matrix = extract_attention_importance(model, X_arr, device)
    else:
        print(f"Training FT-Transformer ({epochs} epochs)")
        model, losses = _train_with_tracking(X_arr, y_arr, d_model=d_model, n_heads=n_heads, n_layers=n_layers, epochs=epochs, device=device)
        final_loss = losses[-1]
        print(f"  Final training loss: {final_loss:.6f}")

        print("Extracting attention importance")
        attn_imp_arr, attn_matrix = extract_attention_importance(model, X_arr, device)

        print("Extracting gradient importance")
        grad_imp_arr = extract_gradient_importance(model, X_arr, y_arr, device)

    attn_df = (
        pd.DataFrame({"feature": feature_names, "importance": attn_imp_arr, "group": [feature_group_of(f) for f in feature_names], "method": "attention_cls"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    grad_df = (
        pd.DataFrame({"feature": feature_names, "importance": grad_imp_arr, "group": [feature_group_of(f) for f in feature_names], "method": "gradient_saliency"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    save_importance_csv(attn_df, results_dir / "transformer_attention_importance.csv", sort_by="importance")
    save_importance_csv(grad_df, results_dir / "transformer_gradient_importance.csv", sort_by="importance")

    attn_matrix_df = pd.DataFrame(attn_matrix, index=feature_names, columns=feature_names)
    attn_matrix_df.to_csv(results_dir / "transformer_attention_matrix.csv")

    print("Generating plots")
    plot_attention_importance(attn_df, f"FT-Transformer Attention Importance (CLS→feat) — {target}", results_dir / "transformer_attention_importance.png", top_n)
    plot_attention_importance(grad_df, f"FT-Transformer Gradient Saliency — {target}", results_dir / "transformer_gradient_importance.png", top_n)
    plot_attention_heatmap(attn_matrix, feature_names, results_dir / "transformer_attention_heatmap.png", top_n=min(top_n, 25))
    plot_combined_importance(attn_df, grad_df, results_dir / "transformer_combined_importance.png", top_n=min(top_n, 25))
    if not use_loo:
        plot_training_curve(losses, results_dir / "transformer_training_curve.png")

    print(f"\nTop 10 features (attention):\n{attn_df.head(10)[['feature', 'importance', 'group']].to_string()}")
    print(f"\nTop 10 features (gradient):\n{grad_df.head(10)[['feature', 'importance', 'group']].to_string()}")

    return {"attn_importance": attn_df, "grad_importance": grad_df, "attn_matrix": attn_matrix, "model": model}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FT-Transformer feature importance",
    )
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--target", default="target_avg_retention", choices=["target_avg_retention", "target_drop_rate", "target_early_drop"])
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--use_loo", action="store_true", help="LOO cross-validation")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()
    run_transformer_importance(
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        target=args.target,
        top_n=args.top_n,
        epochs=args.epochs,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        use_loo=args.use_loo,
        device=args.device,
    )
