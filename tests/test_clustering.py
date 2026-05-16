import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import train.clustering.content_cluster_specialists as content_specialists
from train.clustering.cluster_specialists_multimodal import bucket, build_duration_clusters, duration_sec, video_id_from_payload, write_duration_clusters
from train.clustering.content_cluster_specialists import group_video_ids_by_content_cluster


def write_features(path, video_folder, duration):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"video_features_flat": {"video_folder": video_folder, "duration_seconds": duration}}), encoding="utf-8")


class TestDurationClustering:
    def test_bucket_boundaries_are_upper_exclusive(self):
        assert bucket(719.9, 720, 1320) == (0, "short")
        assert bucket(720, 720, 1320) == (1, "medium")
        assert bucket(1319.9, 720, 1320) == (1, "medium")
        assert bucket(1320, 720, 1320) == (2, "long")

    def test_video_id_prefers_flat_payload_and_falls_back_to_transcript_parent(self, tmp_path):
        payload = {"video_features_flat": {"video_folder": "  from_json  "}}
        assert video_id_from_payload(tmp_path / "abc" / "transcripts" / "features_llm.json", payload) == "from_json"
        assert video_id_from_payload(tmp_path / "abc" / "transcripts" / "features_llm.json", {}) == "abc"
        assert video_id_from_payload(tmp_path / "plain" / "features_llm.json", {}) == "plain"

    def test_duration_uses_flat_payload_only(self):
        assert duration_sec({"video_features_flat": {"duration_seconds": "12.5"}, "source": {"duration_seconds": 99}}) == 12.5
        assert duration_sec({"source": {"duration_seconds": 99}}) == 0.0

    def test_build_duration_clusters_skips_bad_json_and_deduplicates_lists(self, tmp_path):
        write_features(tmp_path / "v1" / "transcripts" / "features_llm.json", "v1", 10)
        write_features(tmp_path / "v2" / "transcripts" / "features_llm.json", "v2", 800)
        write_features(tmp_path / "v3" / "transcripts" / "features_llm.json", "v3", 1500)
        write_features(tmp_path / "dup_a" / "features_llm.json", "v1", 20)
        (tmp_path / "bad" / "transcripts").mkdir(parents=True)
        (tmp_path / "bad" / "transcripts" / "features_llm.json").write_text("{bad", encoding="utf-8")

        videos, train_lists = build_duration_clusters(tmp_path, short_max=720, medium_max=1320)

        assert set(videos) == {"v1", "v2", "v3"}
        assert videos["v1"]["cluster_name"] == "short"
        assert train_lists == {"short": ["v1"], "medium": ["v2"], "long": ["v3"], "all": []}

    def test_write_duration_clusters_writes_both_artifacts(self, tmp_path):
        write_features(tmp_path / "data" / "v1" / "features_llm.json", "v1", 60)
        args = SimpleNamespace(features_root=Path("data"), clusters_json=Path("out/clusters.json"), lists_json=Path("out/lists.json"), short_max_sec=100, medium_max_sec=200)

        train_lists = write_duration_clusters(args, tmp_path)

        assert train_lists["short"] == ["v1"]
        assert json.loads((tmp_path / "out" / "clusters.json").read_text(encoding="utf-8"))["videos"]["v1"]["cluster_name"] == "short"
        assert json.loads((tmp_path / "out" / "lists.json").read_text(encoding="utf-8"))["short"] == ["v1"]


class TestContentClusterGrouping:
    def test_parse_args_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["content_cluster_specialists.py"])
        args = content_specialists.parse_args()

        assert args.arch == "lstm"
        assert args.clusters_json == Path("analysis/video_clustering/retention/clusters.json")
        assert args.min_videos == 5
        assert args.n_heads == 4
        assert args.d_ff == 512
        assert args.clustering_strategy == "retention"

    def test_groups_video_ids_by_cluster_id_as_ints(self):
        clusters = {"videos": {"a": {"cluster_id": "2"}, "b": {"cluster_id": 1}, "c": {"cluster_id": 2}}}
        assert group_video_ids_by_content_cluster(clusters) == {2: ["a", "c"], 1: ["b"]}

    def test_missing_cluster_id_is_a_contract_error(self):
        with pytest.raises(KeyError):
            group_video_ids_by_content_cluster({"videos": {"a": {}}})

    def test_main_runs_optional_clustering_skips_small_clusters_and_trains_large_cluster(self, tmp_path, monkeypatch):
        clusters_json = tmp_path / "clusters.json"
        clusters_json.write_text(json.dumps({"videos": {"small": {"cluster_id": 1}, "big_a": {"cluster_id": 2}, "big_b": {"cluster_id": 2}}}), encoding="utf-8")
        args = SimpleNamespace(
            arch="transformer",
            repo_root=tmp_path,
            output_base=Path("out"),
            run_clustering_first=True,
            data_dir=Path("data"),
            embeddings_dir=Path("embeddings"),
            features_output_dir=Path("features"),
            cluster_out_root=Path("analysis/video_clustering"),
            cluster_min_k=2,
            cluster_max_k=3,
            clustering_strategy="retention",
            clusters_json=clusters_json,
            min_videos=2,
            n_heads=4,
            d_ff=512,
        )
        commands = []
        monkeypatch.setattr(content_specialists, "parse_args", lambda: args)
        monkeypatch.setattr(content_specialists, "run_command", lambda cmd, root: commands.append(cmd))
        monkeypatch.setattr(content_specialists, "train_command", lambda *call_args, **kwargs: ["train", str(kwargs["output_dir"])])

        content_specialists.main()

        assert len(commands) == 2
        assert commands[0][-1] == "retention"
        assert commands[1][0] == "train"
        assert (tmp_path / "out" / "2" / "train_video_ids.txt").read_text(encoding="utf-8").splitlines() == ["big_a", "big_b"]
        meta = json.loads((tmp_path / "out" / "cluster_runs_meta.json").read_text(encoding="utf-8"))
        assert meta["clusters"]["2"]["n_train_videos"] == 2

    def test_main_exits_when_clusters_json_is_missing(self, tmp_path, monkeypatch):
        args = SimpleNamespace(arch="lstm", repo_root=tmp_path, output_base=Path("out"), run_clustering_first=False, clusters_json=Path("missing.json"))
        monkeypatch.setattr(content_specialists, "parse_args", lambda: args)
        with pytest.raises(SystemExit, match="Missing clusters file"):
            content_specialists.main()


def test_embedding_sampling_and_pca_pipeline(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from analysis.embedding_clustering import _sample_embedding_sequence, extract_precomputed_embeddings, find_optimal_clusters

    sampled = _sample_embedding_sequence(np.array([3.0, 4.0]), n_steps=3)
    assert sampled.shape == (3, 2)
    assert np.allclose(sampled, np.array([[0.6, 0.8], [0.6, 0.8], [0.6, 0.8]], dtype=np.float32))

    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    np.save(tmp_path / "v1" / "audio_embeddings.npy", np.arange(8, dtype=np.float32).reshape(4, 2))
    np.save(tmp_path / "v1" / "bert_embeddings.npy", np.arange(6, dtype=np.float32).reshape(3, 2))
    np.save(tmp_path / "v2" / "audio_embeddings.npy", np.ones((2, 2), dtype=np.float32))

    embeddings, valid_vids = extract_precomputed_embeddings(["missing", "v1", "v2"], tmp_path, n_steps=4, pca_dim=2)

    assert valid_vids == ["v1", "v2"]
    assert embeddings.shape == (2, 1)
    assert find_optimal_clusters(np.ones((2, 3)), min_clusters=2, max_clusters=8) == 2


def test_comment_keyword_classification_and_single_cluster_pick():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from get_data.comment_insights import _cluster_and_pick, _keyword_classify

    assert _keyword_classify(["сделайте продолжение", "это ошибка", "обычный комментарий"]) == ["suggestion", "criticism", "other"]
    assert _cluster_and_pick(["один комментарий"], np.ones((1, 3)), [7]) == [{"text": "один комментарий", "likes": 7, "cluster_size": 1}]
