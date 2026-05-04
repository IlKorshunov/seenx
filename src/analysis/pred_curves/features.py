from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from ...utils.video_features import EMBEDDING_TYPES, embeddings_to_matrix, find_json, llm_numeric, load_embeddings, meta_nums, tab_means


def build_feature_df(vids: list[str], data_dir: Path, emb_dir: Path, output_dir: Path, emb_pca_dim: int = 16, rng: int = 42) -> pd.DataFrame:
    per_video: list[dict[str, float]] = []
    emb_raw: dict[str, tuple[list, int]] = {}

    for vid in vids:
        meta = find_json(data_dir, vid, "meta.json")
        flat = find_json(data_dir, vid, "features_llm.json").get("video_features_flat", {})
        dur, views, likes = meta_nums(meta)
        feats: dict[str, float] = {"log1p_duration": float(np.log1p(dur)), "log1p_views": float(np.log1p(views)), "log1p_likes": float(np.log1p(likes))}
        if flat:
            feats.update(llm_numeric(flat))
        feats.update(tab_means(output_dir, vid))
        per_video.append(feats)

    for mod in EMBEDDING_TYPES:
        emb_raw[mod] = load_embeddings(vids, emb_dir, mod)

    n = len(vids)
    for mod in EMBEDDING_TYPES:
        vecs, max_dim = emb_raw[mod]
        if max_dim == 0:
            for i in range(n):
                for j in range(emb_pca_dim):
                    per_video[i][f"pca_{mod}_{j}"] = 0.0
            continue
        mat = embeddings_to_matrix(vecs, n, max_dim)
        n_comp = min(emb_pca_dim, n - 1, max_dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reduced = PCA(n_components=n_comp, random_state=rng).fit_transform(mat)
        for i in range(n):
            for j in range(emb_pca_dim):
                per_video[i][f"pca_{mod}_{j}"] = float(reduced[i, j]) if j < n_comp else 0.0

    df = pd.DataFrame(per_video, index=vids)
    df.index.name = "video_id"
    imputed = SimpleImputer(strategy="median").fit_transform(df)
    return pd.DataFrame(imputed, index=df.index, columns=df.columns)
