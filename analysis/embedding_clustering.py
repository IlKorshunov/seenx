import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from analysis.augmentations import AugmentedRetentionDataset, RetentionDataset

N_STEPS = 100
EMBEDDING_FILES = ["audio_embeddings.npy", "bert_embeddings.npy", "visual_embeddings.npy", "videomae_embeddings.npy", "seg_embeddings.npy"]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class RetentionTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

    def forward(self, src: torch.Tensor, return_embeddings: bool = False):
        x = self.input_proj(src).transpose(0, 1)
        x = self.pos_encoder(x).transpose(0, 1)
        return self.decoder(self.transformer_encoder(x).mean(dim=1)) if not return_embeddings else (self.decoder(self.transformer_encoder(x).mean(dim=1)), self.transformer_encoder(x).mean(dim=1))


def _sample_embedding_sequence(arr: np.ndarray, n_steps: int) -> np.ndarray:
    arr = arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)
    idx = np.linspace(0, len(arr) - 1, n_steps).round().astype(int)
    sampled = arr[idx].astype(np.float32, copy=False)
    return sampled / (np.linalg.norm(sampled, axis=1, keepdims=True) + 1e-6)


def _pca_reduce(embeddings: np.ndarray, pca_dim: int) -> np.ndarray:
    n_components = min(pca_dim, embeddings.shape[0] - 1, embeddings.shape[1])
    return PCA(n_components=n_components, random_state=42).fit_transform(embeddings) if n_components > 0 else embeddings


def _scaled_modality_blocks(sequences_by_vid: dict[str, dict[str, np.ndarray]], valid_vids: list[str], dims_by_file: dict[str, int], n_steps: int) -> list[np.ndarray]:
    blocks = []
    for fname in EMBEDDING_FILES:
        dim = dims_by_file.get(fname, 0)
        if dim == 0:
            continue
        block = np.zeros((len(valid_vids), n_steps * dim), dtype=np.float32)
        present_rows = []
        for row_idx, vid in enumerate(valid_vids):
            if fname in sequences_by_vid[vid]:
                block[row_idx] = sequences_by_vid[vid][fname].reshape(-1)
                present_rows.append(row_idx)
        if present_rows:
            block[present_rows] = StandardScaler().fit_transform(block[present_rows])
        blocks.append(block)
    return blocks


def extract_precomputed_embeddings(vids: list[str], emb_dir: Path, n_steps: int = N_STEPS, pca_dim: int = 128) -> tuple[np.ndarray, list[str]]:
    sequences_by_vid: dict[str, dict[str, np.ndarray]] = {}
    dims_by_file: dict[str, int] = {}
    valid_vids = []

    for vid in vids:
        vid_dir = emb_dir / vid
        if not vid_dir.exists():
            continue

        video_parts = {}
        for fname in EMBEDDING_FILES:
            path = vid_dir / fname
            if path.exists():
                try:
                    arr = np.load(path)
                    if arr.size > 0:
                        sampled = _sample_embedding_sequence(arr, n_steps)
                        video_parts[fname] = sampled
                        dims_by_file[fname] = max(dims_by_file.get(fname, 0), sampled.shape[1])
                except Exception:
                    pass

        if video_parts:
            sequences_by_vid[vid] = video_parts
            valid_vids.append(vid)

    if not sequences_by_vid:
        return np.zeros((0, 0)), []

    blocks = _scaled_modality_blocks(sequences_by_vid, valid_vids, dims_by_file, n_steps)
    return _pca_reduce(np.hstack(blocks), pca_dim), valid_vids


def extract_video_embeddings(df, feature_cols, scaler, model, device):
    embeddings = []
    video_ids = []

    model.eval()
    with torch.no_grad():
        for video_id in sorted(df["video_id"].unique()):
            video_data = df[df["video_id"] == video_id].sort_values("interval_idx").iloc[:N_STEPS]
            if len(video_data) == 0:
                continue

            features = video_data[feature_cols].values
            features = scaler.transform(features)

            seq_len = len(features)
            if seq_len < N_STEPS:
                features = np.vstack([features, np.zeros((N_STEPS - seq_len, len(feature_cols)))])

            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

            _, emb = model(features_tensor, return_embeddings=True)
            embeddings.append(emb.cpu().numpy().flatten())
            video_ids.append(video_id)

    return np.vstack(embeddings), video_ids


def find_optimal_clusters(embeddings: np.ndarray, min_clusters: int = 2, max_clusters: int = 10, output_dir: Path = None) -> int:
    inertias = []
    silhouette_scores = []
    max_k = min(max_clusters, len(embeddings) - 1)
    min_k = max(2, min_clusters)
    if max_k < min_k:
        return min_k

    K_range = range(min_k, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))

    if output_dir is not None:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Elbow Method (Inertia)", "Silhouette Analysis"))

        fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode="lines+markers", name="Inertia"), row=1, col=1)
        fig.update_xaxes(title_text="Number of clusters", row=1, col=1)
        fig.update_yaxes(title_text="Inertia", row=1, col=1)

        fig.add_trace(go.Scatter(x=list(K_range), y=silhouette_scores, mode="lines+markers", name="Silhouette", marker=dict(color="red")), row=1, col=2)
        fig.update_xaxes(title_text="Number of clusters", row=1, col=2)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

        fig.update_layout(title_text="Optimal Clusters Analysis", template="plotly_dark", height=400)

        fig.write_html(str(output_dir / "optimal_clusters.html"))

        plt.style.use("dark_background")
        _, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(list(K_range), inertias, marker="o", color="dodgerblue")
        axes[0].set_title("Elbow Method (Inertia)")
        axes[0].set_xlabel("Number of clusters")
        axes[0].set_ylabel("Inertia")
        axes[0].grid(True, alpha=0.2)

        axes[1].plot(list(K_range), silhouette_scores, marker="o", color="crimson")
        axes[1].set_title("Silhouette Analysis")
        axes[1].set_xlabel("Number of clusters")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(output_dir / "optimal_clusters.png", dpi=150)
        plt.close()

    optimal_k = K_range[np.argmax(silhouette_scores)]
    return optimal_k


class ClusterAwareTransformer:
    def __init__(self, n_clusters: int, input_dim: int, device: str, model_config: dict = None):
        self.n_clusters = max(2, n_clusters)
        self.device = device
        self.models = {}
        self.cluster_assignments = {}

        if model_config is None:
            model_config = {"d_model": 64, "nhead": 4, "num_layers": 3, "dim_feedforward": 256, "dropout": 0.1}

        for i in range(self.n_clusters):
            self.models[i] = RetentionTransformer(input_dim=input_dim, **model_config).to(device)

    def fit_clusters(self, embeddings: np.ndarray):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(StandardScaler().fit_transform(embeddings))

        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for c, cnt in zip(unique, counts, strict=True):
            print(f"Cluster {c}: {cnt} videos")

        return self.cluster_labels

    def train_for_cluster(
        self,
        cluster_id: int,
        train_videos: list,
        val_videos: list,
        df,
        feature_cols: list,
        scaler,
        epochs: int = 200,
        lr: float = 1e-3,
        patience: int = 20,
        use_augmentation: bool = True,
    ):
        cluster_train_videos = [v for v in train_videos if self.cluster_assignments.get(v, 0) == cluster_id]
        cluster_val_videos = [v for v in val_videos if self.cluster_assignments.get(v, 0) == cluster_id]

        if not cluster_val_videos and val_videos:
            cluster_val_videos = val_videos[:1]

        if len(cluster_train_videos) < 2:
            warnings.warn(f"Not enough training videos for cluster {cluster_id}")
            return None, float("inf")

        train_dataset = AugmentedRetentionDataset(
            df, feature_cols, cluster_train_videos, scaler, fit_scaler=False, max_seq_len=100, augment=use_augmentation, augment_prob=0.3, num_augmentations=1
        )
        val_dataset = RetentionDataset(df, feature_cols, cluster_val_videos, scaler, fit_scaler=False, max_seq_len=100)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=min(4, len(val_dataset)), shuffle=False)

        model = self.models[cluster_id]
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        criterion = nn.HuberLoss(delta=0.05)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for _ in range(epochs):
            model.train()
            train_loss = 0
            for seq, target in train_loader:
                seq, target = seq.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(seq)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for seq, target in val_loader:
                    seq, target = seq.to(self.device), target.to(self.device)
                    output = model(seq)
                    val_loss += criterion(output, target).item()

            val_loss /= max(1, len(val_loader))

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, best_val_loss

    def predict(self, video_id: str, features_tensor: torch.Tensor):
        cluster = self.cluster_assignments.get(video_id, 0)
        model = self.models[cluster]
        model.eval()
        with torch.no_grad():
            return model(features_tensor.to(self.device)).cpu().numpy().flatten()
