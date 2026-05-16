import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# pylint: disable=protected-access


ROOT = Path(__file__).resolve().parents[1]


def load_video_clustering(monkeypatch):
    pytest.importorskip("sklearn")
    for name in ["torch", "torch.nn", "torch.optim", "torch.utils", "torch.utils.data"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["torch.utils.data"].DataLoader = object

    umap = types.ModuleType("umap")
    umap.UMAP = lambda *args, **kwargs: types.SimpleNamespace(fit_transform=lambda x: np.zeros((x.shape[0], 2)))
    monkeypatch.setitem(sys.modules, "umap", umap)

    dtaidistance = types.ModuleType("dtaidistance")
    dtaidistance.dtw_ndim = types.SimpleNamespace(distance=lambda a, b, window=None: float(abs(len(a) - len(b))))
    monkeypatch.setitem(sys.modules, "dtaidistance", dtaidistance)

    embedding_clustering = types.ModuleType("analysis.embedding_clustering")
    embedding_clustering.RetentionTransformer = object
    embedding_clustering.extract_precomputed_embeddings = lambda *args, **kwargs: (np.zeros((0, 0)), [])
    embedding_clustering.extract_video_embeddings = lambda *args, **kwargs: (np.zeros((0, 0)), [])
    embedding_clustering.find_optimal_clusters = lambda *args, **kwargs: 2
    monkeypatch.setitem(sys.modules, "analysis.embedding_clustering", embedding_clustering)

    aligner = types.ModuleType("src.utils.embedding_aligner")
    aligner.load_aligned_embeddings = lambda *args, **kwargs: (np.zeros((1, 1)), [])
    monkeypatch.setitem(sys.modules, "src.utils.embedding_aligner", aligner)

    video_features = types.ModuleType("src.utils.video_features")
    video_features.EMBEDDING_TYPES = []
    video_features.dicts_to_matrix = lambda rows, cols: np.zeros((len(rows), len(cols)))
    video_features.embeddings_to_matrix = lambda vecs, n, max_dim: np.zeros((n, max_dim))
    video_features.find_json = lambda *args, **kwargs: {}
    video_features.llm_numeric = lambda flat: {}
    video_features.load_embeddings = lambda *args, **kwargs: ({}, 0)
    video_features.meta_nums = lambda meta: (0.0, 0.0, 0.0)
    video_features.pca_reduce = lambda mat, dim, rng: np.zeros((mat.shape[0], dim))
    video_features.tab_means = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "src.utils.video_features", video_features)

    spec = importlib.util.spec_from_file_location("analysis.video_clustering", ROOT / "analysis/video_clustering.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["analysis.video_clustering"] = module
    spec.loader.exec_module(module)
    return module


def test_scale_block_imputes_median_and_standardizes(monkeypatch):
    vc = load_video_clustering(monkeypatch)
    scaled = vc._scale_block(np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 8.0]]))

    assert scaled.shape == (3, 2)
    assert np.allclose(scaled.mean(axis=0), [0.0, 0.0])


def test_metric_f0_f1_and_single_cluster_metrics(monkeypatch):
    vc = load_video_clustering(monkeypatch)
    d = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 5.0], [4.0, 5.0, 0.0]])

    f0, f1, ratio = vc.metric_f0_f1(d, np.array([0, 0, 1]))

    assert f0 == pytest.approx(1.0 / 3.0)
    assert f1 == pytest.approx(3.0)
    assert ratio == 9.0
    assert vc.compute_metrics(np.array([0, 0, 0]), X=np.zeros((3, 2))) == {"k": 1, "n_valid": 3, "n_noise": 0}


def test_compute_metrics_ignores_noise_labels(monkeypatch):
    vc = load_video_clustering(monkeypatch)
    x = np.array([[0.0], [0.1], [4.0], [4.1], [99.0]])
    labels = np.array([0, 0, 1, 1, -1])

    metrics = vc.compute_metrics(labels, X=x)

    assert metrics["k"] == 2
    assert metrics["n_valid"] == 4
    assert metrics["n_noise"] == 1
    assert metrics["F0"] < metrics["F1"]
    assert metrics["F1/F0"] is not None


def test_entropy_external_reports_conditional_entropy(monkeypatch):
    vc = load_video_clustering(monkeypatch)
    perfect = vc.compute_entropy_external(np.array([0, 0, 1, 1]), np.array([10, 10, 20, 20]))
    mixed = vc.compute_entropy_external(np.array([0, 1, 0, 1]), np.array([10, 10, 20, 20]))

    assert perfect["H_class|clust"] == pytest.approx(0.0)
    assert mixed["H_class|clust"] > perfect["H_class|clust"]
