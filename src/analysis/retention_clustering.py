import os

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dtaidistance import dtw
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..retention_analysis import _resample_retention, load_channel_retentions_csv
from ..utils.logger import Logger


logger = Logger(show=True).get_logger()


def normalize_curve(curve, target_len=100):
    return _resample_retention(curve, target_len)


def extract_shape_features(curve):
    n = len(curve)
    if n < 2:
        return np.zeros(6)
    k, idx_30 = max(1, n // 10), int(n * 0.3)
    return np.array([(curve[k] - curve[0]) / k, (curve[-1] - curve[-k - 1]) / k, np.var(curve), np.min(curve), np.mean(curve), curve[0] - curve[idx_30] if idx_30 < n else 0])


def cluster_curves_by_shape(channel_data, n_clusters=4, use_dtw=False, target_len=100):
    curves = np.array([normalize_curve(d["retention_series"], target_len) for d in channel_data])
    if use_dtw:
        try:
            n = len(curves)
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    dist[i, j] = dist[j, i] = dtw.distance(curves[i], curves[j])
            labels = fcluster(linkage(squareform(dist), method="average"), n_clusters, criterion="maxclust") - 1
        except ImportError:
            use_dtw = False
    if not use_dtw:
        labels = KMeans(n_clusters=min(n_clusters, len(curves)), random_state=42).fit_predict(StandardScaler().fit_transform(np.array([extract_shape_features(c) for c in curves])))
    labels = np.asarray(labels, dtype=int)
    cluster_name_map = {}
    for k in sorted(set(labels)):
        sub = curves[labels == k]
        if len(sub) == 0:
            cluster_name_map[k] = f"cluster_{k}"
            continue
        sl_start = np.mean([(c[1] - c[0]) if len(c) > 1 else 0 for c in sub])
        sl_end = np.mean([(c[-1] - c[-2]) if len(c) > 1 else 0 for c in sub])
        mn = np.mean([np.min(c) for c in sub])
        name = "sharp_drop" if sl_start < -0.5 else ("flat" if abs(sl_start) < 0.2 and abs(sl_end) < 0.1 else ("with_dips" if mn < 30 else "smooth"))
        cluster_name_map[k] = name
    return labels, cluster_name_map


def plot_cluster_curves(channel_data, labels, cluster_name_map, output_path=None):
    unique = sorted(set(labels))[:4]
    n_rows = (len(unique) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    axes = np.atleast_2d(axes)
    for idx, k in enumerate(unique):
        ax = axes[idx // 2, idx % 2]
        for i, d in enumerate(channel_data):
            if labels[i] == k:
                ax.plot(np.arange(len(d["retention_series"])), d["retention_series"], alpha=0.5, linewidth=1)
        ax.set(ylim=(0, 105), ylabel="Retention (%)", xlabel="Time (sec)", title=f"{cluster_name_map.get(k, f'cluster_{k}')} (n={(labels == k).sum()})")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def run_retention_clustering(data_dir="data", n_clusters=4, output_dir="my_metrics/retention_clusters"):
    channel_data = load_channel_retentions_csv(data_dir)
    if len(channel_data) < n_clusters:
        return pd.DataFrame()
    labels, cmap = cluster_curves_by_shape(channel_data, n_clusters=n_clusters)
    df = pd.DataFrame(
        [
            {"video_id": d["name"], "duration_sec": d["duration_sec"], "cluster": int(labels[i]), "cluster_name": cmap.get(int(labels[i]), f"cluster_{labels[i]}")}
            for i, d in enumerate(channel_data)
        ]
    )
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "retention_clusters.csv"), index=False)
    plot_cluster_curves(channel_data, labels, cmap, output_path=os.path.join(output_dir, "cluster_curves.png"))
    plt.close()
    return df
