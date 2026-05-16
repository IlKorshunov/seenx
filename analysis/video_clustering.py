from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import umap
from dtaidistance import dtw_ndim
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, pairwise_distances, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from analysis.augmentations import RetentionDataset
from analysis.embedding_clustering import RetentionTransformer, extract_precomputed_embeddings, extract_video_embeddings, find_optimal_clusters
from src.utils.embedding_aligner import load_aligned_embeddings
from src.utils.video_features import EMBEDDING_TYPES, dicts_to_matrix, embeddings_to_matrix, find_json, llm_numeric, load_embeddings, meta_nums, pca_reduce, tab_means


BLOCK_WEIGHTS = {"meta": 0.15, "llm": 0.25, "embeddings": 0.40, "tabular": 0.20}
ALL_STRATEGIES = ["kmeans", "gmm", "dbscan", "spectral", "dtw", "retention"]
METRIC_LABELS = {"calinski_harabasz": "CH", "davies_bouldin": "DB", "F1/F0": "F1/F0", "Phi0": "\u03a60"}
METRIC_KEYS = ("F0", "F1", "F1/F0", "Phi0", "calinski_harabasz", "davies_bouldin", "H_clust")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video clustering")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--embeddings-dir", type=Path, default=Path("embeddings"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--out-root", type=Path, default=Path("analysis/video_clustering"))
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--emb-pca-dim", type=int, default=16)
    p.add_argument("--min-k", type=int, default=2)
    p.add_argument("--max-k", type=int, default=8)
    p.add_argument("--strategy", choices=ALL_STRATEGIES + ["all"], default="all")
    p.add_argument("--dbscan-min-samples", type=int, default=3)
    p.add_argument("--dtw-downsample-sec", type=int, default=5)
    p.add_argument("--dtw-pca-dim", type=int, default=48)
    p.add_argument("--dtw-band-ratio", type=float, default=0.2)
    p.add_argument("--dtw-meta-weight", type=float, default=0.3)
    p.add_argument("--use-dynamic-embs", action="store_true", help="Train a base retention transformer to extract dynamic sequence embeddings (can be unstable/noisy)")
    return p.parse_args()


def _scale_block(mat: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(mat))


def _k_range(mn: int, mx: int, n: int) -> list[int]:
    return list(range(max(2, mn), min(mx, n - 1) + 1))


def _n_unique(labels: np.ndarray) -> int:
    return len(set(labels.tolist()))


def _load_video_data(vids: list[str], data_dir: Path, output_dir: Path):
    meta_triples: list[tuple[float, float, float]] = []
    llm_per_video: list[dict[str, float]] = []
    tab_per_video: list[dict[str, float]] = []
    llm_keys: set[str] = set()
    tab_keys: set[str] = set()
    for video_id in vids:
        meta = find_json(data_dir, video_id, "meta.json")
        try:
            flat = find_json(data_dir, video_id, "features_llm.json").get("video_features_flat", {})
        except Exception as e:
            print(f"[Warning] Failed to load LLM features for {video_id}: {e}")
            flat = {}
        meta_triples.append(meta_nums(meta))
        llm_dict = llm_numeric(flat) if flat else {}
        llm_keys.update(llm_dict)
        llm_per_video.append(llm_dict)
        tab_dict = tab_means(output_dir, video_id)
        tab_keys.update(tab_dict)
        tab_per_video.append(tab_dict)
    llm_cols, tab_cols = sorted(llm_keys), sorted(tab_keys)
    log_meta = np.array([[np.log1p(dur), np.log1p(views), np.log1p(likes)] for dur, views, likes in meta_triples], dtype=np.float64)
    display = [{"video_id": vid, "duration_sec": dur, "view_count": views, "like_count": likes} for vid, (dur, views, likes) in zip(vids, meta_triples, strict=True)]
    return (log_meta, llm_cols, dicts_to_matrix(llm_per_video, llm_cols), tab_cols, dicts_to_matrix(tab_per_video, tab_cols), display)


def build_feature_matrix(vids: list[str], data_dir: Path, emb_dir: Path, output_dir: Path, emb_pca_dim: int, rng: int) -> tuple[np.ndarray, list[str], list[dict]]:
    n_videos = len(vids)
    log_meta, llm_cols, llm_mat, tab_cols, tab_mat, display = _load_video_data(vids, data_dir, output_dir)
    emb_parts, emb_names = [], []
    for modality in EMBEDDING_TYPES:
        vecs, max_dim = load_embeddings(vids, emb_dir, modality)
        mat = embeddings_to_matrix(vecs, n_videos, max_dim)
        emb_parts.append(pca_reduce(mat, emb_pca_dim, rng))
        emb_names.extend(f"pca_{modality}_{j}" for j in range(emb_pca_dim))
    emb_block = np.hstack(emb_parts)
    parts, all_names = [], []
    for category, mat, names in [
        ("meta", log_meta, ["log1p_dur", "log1p_views", "log1p_likes"]),
        ("llm", llm_mat, llm_cols),
        ("embeddings", emb_block, emb_names),
        ("tabular", tab_mat, tab_cols),
    ]:
        if mat.shape[1] == 0:
            continue
        scaled = _scale_block(mat)
        parts.append(scaled * (BLOCK_WEIGHTS.get(category, 1.0) / np.sqrt(mat.shape[1])))
        all_names.extend(names)
    return (np.hstack(parts) if parts else np.zeros((n_videos, 0))), all_names, display


def build_dtw_distances(
    vids: list[str], emb_dir: Path, data_dir: Path, output_dir: Path, downsample: int, pca_dim: int, band_ratio: float, meta_weight: float, rng: int, display: list[dict]
) -> tuple[np.ndarray, list[dict]]:
    raw_seqs = []
    for vid in vids:
        aligned, _ = load_aligned_embeddings(vid, str(emb_dir), duration_sec=None)
        seq = aligned.astype(np.float64)
        n_chunks = max(1, seq.shape[0] // downsample)
        raw_seqs.append(seq[: n_chunks * downsample].reshape(n_chunks, downsample, seq.shape[1]).mean(axis=1))
    pool = np.vstack(raw_seqs)
    n_comp = max(1, min(pca_dim, pool.shape[0] - 1, pool.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = PCA(n_components=n_comp, random_state=rng).fit(pool)
    seqs = [np.ascontiguousarray(pca.transform(s), dtype=np.double) for s in raw_seqs]
    window = max(2, int(max(s.shape[0] for s in seqs) * band_ratio))
    D_emb = np.zeros((len(seqs), len(seqs)))
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            D_emb[i, j] = D_emb[j, i] = dtw_ndim.distance(seqs[i], seqs[j], window=window)
    if (mx := D_emb.max()) > 0:
        D_emb /= mx
    log_meta, _, llm_mat, _, tab_mat, _ = _load_video_data(vids, data_dir, output_dir)
    D_meta = pairwise_distances(_scale_block(np.hstack([log_meta, llm_mat, tab_mat])), metric="euclidean")
    if (mx := D_meta.max()) > 0:
        D_meta /= mx
    return (1 - meta_weight) * D_emb + meta_weight * D_meta, display


def cluster_kmeans(X: np.ndarray, mn: int, mx: int, rng: int):
    best = (np.zeros(X.shape[0], dtype=np.int32), 2, -1.0)
    for k in _k_range(mn, mx, X.shape[0]):
        lb = KMeans(n_clusters=k, random_state=rng, n_init=10).fit_predict(X).astype(np.int32)
        if _n_unique(lb) < 2:
            continue
        if (s := float(silhouette_score(X, lb))) > best[2]:
            best = (lb, k, s)
    return best


def cluster_gmm(X: np.ndarray, mn: int, mx: int, rng: int):
    best_k, best_bic, best_lb = 2, np.inf, np.zeros(X.shape[0], dtype=np.int32)
    for k in _k_range(mn, mx, X.shape[0]):
        gmm = GaussianMixture(n_components=k, random_state=rng, n_init=3, covariance_type="full")
        lb = gmm.fit_predict(X).astype(np.int32)
        bic = gmm.bic(X)
        if bic < best_bic and _n_unique(lb) > 1:
            best_bic, best_k, best_lb = bic, k, lb
    sil = float(silhouette_score(X, best_lb)) if _n_unique(best_lb) > 1 else 0.0
    return best_lb, best_k, sil


def cluster_dbscan(X: np.ndarray, min_samples: int = 5):
    dists = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0][:, -1])
    best = (np.zeros(X.shape[0], dtype=np.int32), 1, -1.0)
    for pct in range(10, 95, 5):
        eps = max(float(np.percentile(dists, pct)), 0.01)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X).astype(np.int32)
        n_clusters = len(set(labels.tolist()) - {-1})
        valid = labels >= 0
        if n_clusters < 2 or valid.sum() < 3:
            continue
        if (s := float(silhouette_score(X[valid], labels[valid]))) > best[2]:
            best = (labels, n_clusters, s)
    return best


def cluster_spectral(X: np.ndarray, mn: int, mx: int, rng: int):
    best = (np.zeros(X.shape[0], dtype=np.int32), 2, -1.0)
    nn = max(2, min(10, X.shape[0] - 1))
    for k in _k_range(mn, mx, X.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lb = SpectralClustering(n_clusters=k, random_state=rng, affinity="nearest_neighbors", n_neighbors=nn).fit_predict(X).astype(np.int32)
        if _n_unique(lb) < 2:
            continue
        if (s := float(silhouette_score(X, lb))) > best[2]:
            best = (lb, k, s)
    return best


def cluster_agglo(D: np.ndarray, mn: int, mx: int):
    best = (np.zeros(D.shape[0], dtype=np.int32), 2, -1.0)
    for k in _k_range(mn, mx, D.shape[0]):
        lb = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(D).astype(np.int32)
        if _n_unique(lb) < 2:
            continue
        if (s := float(silhouette_score(D, lb, metric="precomputed"))) > best[2]:
            best = (lb, k, s)
    return best


def metric_f0_f1(Dv: np.ndarray, lb: np.ndarray) -> tuple[float, float, float | None]:
    n = len(lb)
    triu = np.triu(np.ones((n, n), dtype=bool))
    same = lb[:, None] == lb[None, :]
    intra_mask, inter_mask = same & triu, (~same) & triu
    f0 = float(Dv[intra_mask].sum() / intra_mask.sum()) if intra_mask.sum() else 0.0
    f1 = float(Dv[inter_mask].sum() / inter_mask.sum()) if inter_mask.sum() else 0.0
    return f0, f1, (round(f1 / f0, 4) if f0 > 1e-10 else None)


def metric_phi0(Xv: np.ndarray, lb: np.ndarray, unique_k: list, n: int, K: int) -> float:
    total = sum(np.sum(np.linalg.norm(Xv[lb == k] - Xv[lb == k].mean(axis=0), axis=1) ** 2) for k in unique_k)
    return round(float(total / (n * K)), 6)


def metric_h_clust(lb: np.ndarray, unique_k: list) -> float:
    counts = np.array([int(np.sum(lb == k)) for k in unique_k])
    probs = counts / counts.sum()
    return round(float(-np.sum(probs * np.log(probs + 1e-12))), 4)


def metric_sklearn_features(Xv: np.ndarray, lb: np.ndarray) -> dict:
    return {
        "silhouette": round(float(silhouette_score(Xv, lb)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(Xv, lb)), 2),
        "davies_bouldin": round(float(davies_bouldin_score(Xv, lb)), 4),
    }


def metric_sklearn_precomputed(Dv: np.ndarray, lb: np.ndarray) -> dict:
    return {"silhouette": round(float(silhouette_score(Dv, lb, metric="precomputed")), 4)}


def compute_metrics(labels: np.ndarray, X: np.ndarray | None = None, D: np.ndarray | None = None) -> dict:
    valid = labels >= 0
    lb = labels[valid]
    unique_k = sorted(set(lb.tolist()))
    K, n, n_noise = len(unique_k), int(valid.sum()), int((~valid).sum())
    base = {"k": K, "n_valid": n, "n_noise": n_noise}
    if K < 2:
        return base
    if D is not None:
        Dv = D[np.ix_(valid, valid)]
    else:
        assert X is not None
        Dv = pairwise_distances(X[valid])
    f0, f1, f1f0 = metric_f0_f1(Dv, lb)
    out = {**base, "F0": round(f0, 4), "F1": round(f1, 4), "F1/F0": f1f0, "H_clust": metric_h_clust(lb, unique_k)}
    if X is not None:
        Xv = X[valid]
        out["Phi0"] = metric_phi0(Xv, lb, unique_k, n, K)
        out.update(metric_sklearn_features(Xv, lb))
    else:
        out.update(metric_sklearn_precomputed(Dv, lb))
    return out


def compute_entropy_external(labels: np.ndarray, true_labels: np.ndarray) -> dict:
    n = len(labels)
    classes, clusters = sorted(set(true_labels.tolist())), sorted(set(labels.tolist()))
    mc = np.array([np.sum(true_labels == c) for c in classes])
    nk = np.array([np.sum(labels == k) for k in clusters])
    H_class = float(-np.sum((mc / n) * np.log((mc / n) + 1e-12)))
    H_clust = float(-np.sum((nk / n) * np.log((nk / n) + 1e-12)))
    H_cond = 0.0
    for ki, k in enumerate(clusters):
        for c in classes:
            nck = int(np.sum((true_labels == c) & (labels == k)))
            if nck > 0:
                H_cond -= (nck / n) * np.log(nck / nk[ki] + 1e-12)
    return {"H_class": round(H_class, 4), "H_clust": round(H_clust, 4), "H_class|clust": round(H_cond, 4)}


def project_2d(X_or_D: np.ndarray, rng: int, precomputed: bool = False) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            umap.UMAP(n_components=2, n_neighbors=min(15, X_or_D.shape[0] - 1), min_dist=0.1, metric="precomputed" if precomputed else "euclidean", random_state=rng)
            .fit_transform(X_or_D)
            .astype(np.float64)
        )


def save_results(vids: list[str], labels: np.ndarray, xy: np.ndarray, display: list[dict], strategy: str, metrics: dict, out: Path) -> None:
    n = len(vids)
    summary = [
        {
            "cluster": int(c),
            "n": len(idx := np.where(labels == c)[0]),
            "mean_dur_s": float(np.mean([display[i]["duration_sec"] for i in idx])),
            "mean_views": float(np.mean([display[i]["view_count"] for i in idx])),
        }
        for c in sorted(set(labels.tolist()))
    ]
    pd.DataFrame(summary).to_csv(out / "cluster_summary.csv", index=False)
    (out / "clusters.json").write_text(
        json.dumps(
            {
                "config": {"strategy": strategy, "n_videos": n, "metrics": metrics},
                "videos": {
                    vids[i]: {
                        "cluster_id": int(labels[i]),
                        "x": round(float(xy[i, 0]), 4),
                        "y": round(float(xy[i, 1]), 4),
                        **{k: v for k, v in display[i].items() if k != "video_id"},
                    }
                    for i in range(n)
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sil = metrics.get("silhouette", 0)
    df = pd.DataFrame(
        {
            "video_id": vids,
            "cluster": labels.astype(str),
            "duration_s": [r["duration_sec"] for r in display],
            "views": [r["view_count"] for r in display],
            "UMAP 1": xy[:, 0],
            "UMAP 2": xy[:, 1],
        }
    )
    df.to_csv(out / "clusters.csv", index=False)
    title = f"{strategy.upper()} · k={metrics.get('k', '?')} · sil={sil:.3f} · {n} videos"
    m_text = " · ".join(f"{METRIC_LABELS.get(key, key)}={v}" for key in METRIC_KEYS if (v := metrics.get(key)) is not None)

    plt.style.use("default")
    fig_plt, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    clusters = np.unique(labels)
    cmap = plt.get_cmap("tab10")

    for i, c in enumerate(clusters):
        mask = labels == c
        n_samples = np.sum(mask)
        color = cmap(i % 10)
        ax.scatter(xy[mask, 0], xy[mask, 1], label=f"Cluster {c} ({n_samples})", c=[color], alpha=0.85, edgecolors="white", linewidths=0.4, s=42)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.figtext(0.02, 0.02, m_text, fontsize=8, color="#4b5563", family="monospace")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(out / "cluster_viz.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_comparison(results: dict, out: Path) -> None:
    strategies = list(results.keys())
    header_keys = ["k", "silhouette", "F0", "F1", "F1/F0", "Phi0", "calinski_harabasz", "davies_bouldin", "H_clust", "n_noise"]
    pd.DataFrame([{"strategy": strategy, **{key: results[strategy].get(key) for key in header_keys}} for strategy in strategies]).to_csv(out / "comparison.csv", index=False)

    plt.style.use("default")
    fig_plt, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    x = np.arange(len(strategies))
    width = 0.35

    sil_scores = [results[s].get("silhouette", 0) or 0 for s in strategies]
    f1f0_scores = [results[s].get("F1/F0", 0) or 0 for s in strategies]

    ax.bar(x - width / 2, sil_scores, width, label="Silhouette ↑", color="#2563eb")
    ax.bar(x + width / 2, f1f0_scores, width, label="F1/F0 ↑", color="#db2777")

    ax.set_ylabel("Score")
    ax.set_title("Clustering Strategy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_facecolor("white")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def discover_video_ids(data_dir: Path, emb_dir: Path, output_dir: Path) -> list[str]:
    ids: set[str] = set()
    for directory in (data_dir, emb_dir):
        if directory.is_dir():
            ids.update(child.name for child in directory.iterdir() if child.is_dir())
    if output_dir.is_dir():
        ids.update(p.stem.removesuffix("_features") for p in output_dir.glob("*_features.csv"))
        ids.update(child.name for child in output_dir.iterdir() if child.is_dir() and (child / "features_readable.csv").exists())
    return sorted(ids)


def load_all_video_sequences(vids: list[str], output_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    dfs = []
    for vid in vids:
        path = output_dir / f"{vid}_features.csv"
        if not path.exists():
            path = output_dir / vid / "features_readable.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["video_id"] = vid
            if "interval_idx" not in df.columns:
                df["interval_idx"] = np.arange(len(df))
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(), []
    full_df = pd.concat(dfs, ignore_index=True)
    exclude_cols = {"video_id", "interval_idx", "target", "timestamp", "time", "timestamp_start", "timestamp_end"}
    feature_cols = [c for c in full_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(full_df[c])]
    full_df[feature_cols] = full_df[feature_cols].fillna(0.0)
    return full_df, feature_cols


def main() -> None:
    args = parse_args()
    rng = args.random_state
    args.out_root.mkdir(parents=True, exist_ok=True)
    vids = discover_video_ids(args.data_dir, args.embeddings_dir, args.output_dir)
    assert vids, "No videos found"
    strategies = ALL_STRATEGIES if args.strategy == "all" else [args.strategy]
    feature_based = [s for s in strategies if s not in ("dtw", "retention")]
    X, display, xy_feat = None, None, None
    if feature_based:
        X, _, display = build_feature_matrix(vids, args.data_dir, args.embeddings_dir, args.output_dir, args.emb_pca_dim, rng)
        assert X.shape[1] > 0, "Empty feature matrix"
        print(f"[features] {len(vids)} videos × {X.shape[1]} features (emb types: {len(EMBEDDING_TYPES)}, pca={args.emb_pca_dim})")
        xy_feat = project_2d(X, rng)
    all_results: dict[str, dict] = {}
    run = {
        "kmeans": lambda: cluster_kmeans(X, args.min_k, args.max_k, rng),
        "gmm": lambda: cluster_gmm(X, args.min_k, args.max_k, rng),
        "dbscan": lambda: cluster_dbscan(X, min_samples=args.dbscan_min_samples),
        "spectral": lambda: cluster_spectral(X, args.min_k, args.max_k, rng),
    }
    for strategy in feature_based:
        labels, k, _ = run[strategy]()
        metrics = compute_metrics(labels, X=X)
        (sub := args.out_root / strategy).mkdir(parents=True, exist_ok=True)
        save_results(vids, labels, xy_feat, display, strategy, metrics, sub)
        all_results[strategy] = metrics
        noise = f" noise={metrics['n_noise']}" if metrics.get("n_noise") else ""
        print(
            f"  [{strategy:>10}] k={metrics['k']:>2}  sil={metrics.get('silhouette', 0):>7.4f}"
            f"  F0={metrics.get('F0', 0):.4f}  F1={metrics.get('F1', 0):.4f}"
            f"  F1/F0={metrics.get('F1/F0', 'n/a')}{noise}"
        )
    if "dtw" in strategies:
        if display is None:
            _, _, display = build_feature_matrix(vids, args.data_dir, args.embeddings_dir, args.output_dir, args.emb_pca_dim, rng)
        D_dtw, _ = build_dtw_distances(
            vids, args.embeddings_dir, args.data_dir, args.output_dir, args.dtw_downsample_sec, args.dtw_pca_dim, args.dtw_band_ratio, args.dtw_meta_weight, rng, display
        )
        labels, k, _ = cluster_agglo(D_dtw, args.min_k, args.max_k)
        metrics = compute_metrics(labels, D=D_dtw)
        (sub := args.out_root / "dtw").mkdir(parents=True, exist_ok=True)
        save_results(vids, labels, project_2d(D_dtw, rng, precomputed=True), display, "dtw", metrics, sub)
        all_results["dtw"] = metrics
        print(f"  [{'dtw':>10}] k={metrics['k']:>2}  sil={metrics.get('silhouette', 0):>7.4f}  F0={metrics.get('F0', 0):.4f}  F1={metrics.get('F1', 0):.4f}")

    if "retention" in strategies:
        (sub := args.out_root / "retention").mkdir(parents=True, exist_ok=True)

        print("  [retention] Extracting rich precomputed multimodal embeddings from disk")
        emb_matrix, ext_vids = extract_precomputed_embeddings(vids, args.embeddings_dir)

        seq_emb_matrix = None
        seq_vids = []
        if args.use_dynamic_embs:
            df, feature_cols = load_all_video_sequences(vids, args.output_dir)
            if not df.empty and feature_cols:
                device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
                print(f"  [retention] Training base sequence model on {device}")

                base_model = RetentionTransformer(input_dim=len(feature_cols), d_model=64, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1).to(device)

                temp_dataset = RetentionDataset(df, feature_cols, vids, scaler=None, fit_scaler=True, max_seq_len=100)
                scaler = temp_dataset.scaler

                temp_loader = DataLoader(temp_dataset, batch_size=8, shuffle=True)
                optimizer = optim.AdamW(base_model.parameters(), lr=1e-3)
                criterion = nn.MSELoss()

                for _ in range(10):
                    base_model.train()
                    for seq, target in temp_loader:
                        seq, target = seq.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = base_model(seq)
                        if output.shape != target.shape:
                            target = target.mean(dim=1, keepdim=True)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                print("  [retention] Extracting sequence embeddings")
                seq_emb_matrix, seq_vids = extract_video_embeddings(df, feature_cols, scaler, base_model, device)
            else:
                print("  [retention] No sequence data found for dynamic embeddings.")
        else:
            print("  [retention] Skipping dynamic sequence embeddings (--use-dynamic-embs is False).")

        combined_vids = sorted(list(set(ext_vids) | set(seq_vids)))
        combined_embs = []
        final_vids = []

        ext_idx = {v: i for i, v in enumerate(ext_vids)}
        seq_idx = {v: i for i, v in enumerate(seq_vids)}

        ext_dim = emb_matrix.shape[1] if len(emb_matrix) > 0 else 0
        seq_dim = seq_emb_matrix.shape[1] if seq_emb_matrix is not None and len(seq_emb_matrix) > 0 else 0

        for v in combined_vids:
            parts = []
            if ext_dim > 0:
                parts.append(emb_matrix[ext_idx[v]] if v in ext_idx else np.zeros(ext_dim))
            if seq_dim > 0:
                parts.append(seq_emb_matrix[seq_idx[v]] if v in seq_idx else np.zeros(seq_dim))

            if parts:
                combined_embs.append(np.concatenate(parts))
                final_vids.append(v)

        combined_embs = np.array(combined_embs)

        if len(combined_embs) < 2:
            print("  [retention] Not enough embeddings extracted to cluster.")
        else:
            optimal_k = find_optimal_clusters(combined_embs, min_clusters=args.min_k, max_clusters=args.max_k, output_dir=sub)

            scaler = StandardScaler()
            emb_scaled = scaler.fit_transform(combined_embs)
            kmeans = KMeans(n_clusters=optimal_k, random_state=rng, n_init=10)
            labels = kmeans.fit_predict(emb_scaled)

            vid_to_label = dict(zip(final_vids, labels, strict=True))
            full_labels = np.array([vid_to_label.get(v, -1) for v in vids])

            metrics = compute_metrics(full_labels, X=combined_embs)
            xy_retention = project_2d(combined_embs, rng)

            if display is None:
                _, _, display = build_feature_matrix(vids, args.data_dir, args.embeddings_dir, args.output_dir, args.emb_pca_dim, rng)

            save_results(vids, full_labels, xy_retention, display, "retention", metrics, sub)
            all_results["retention"] = metrics
            print(f"  [{'retention':>10}] k={metrics.get('k', 0):>2}  sil={metrics.get('silhouette', 0):>7.4f}  F0={metrics.get('F0', 0):.4f}  F1={metrics.get('F1', 0):.4f}")

    if len(all_results) > 1:
        save_comparison(all_results, args.out_root)
        print(f"\nComparison → {args.out_root / 'comparison.png'}")


if __name__ == "__main__":
    main()
