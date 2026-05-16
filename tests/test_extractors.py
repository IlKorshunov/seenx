import json
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_text_package_stubs(monkeypatch):
    for name in ["src", "src.extractors", "src.extractors.text", "src.utils"]:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
    seenx_utils = types.ModuleType("src.seenx_utils")
    seenx_utils.get_video_duration = lambda video_path: 10  # type: ignore[attr-defined]
    config = types.ModuleType("src.utils.config")
    config.Config = dict  # type: ignore[attr-defined]
    logger = types.ModuleType("src.utils.logger")
    logger.Logger = lambda show=True: types.SimpleNamespace(get_logger=lambda: types.SimpleNamespace(info=lambda *args, **kwargs: None))  # type: ignore[attr-defined]
    transcript_cache = types.ModuleType("src.utils.transcript_cache")
    transcript_cache.get_transcript = lambda video_path, config: {"segments": []}  # type: ignore[attr-defined]
    video_arch = types.ModuleType("src.extractors.video.architectures.common")
    video_arch.unload_ensemble = lambda models, device=None: None  # type: ignore[attr-defined]
    for name, module in {
        "src.seenx_utils": seenx_utils,
        "src.utils.config": config,
        "src.utils.logger": logger,
        "src.utils.transcript_cache": transcript_cache,
        "src.extractors.video": types.ModuleType("src.extractors.video"),
        "src.extractors.video.architectures": types.ModuleType("src.extractors.video.architectures"),
        "src.extractors.video.architectures.common": video_arch,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return load_module("src.extractors.text._base", "src/extractors/text/_base.py")


def install_fake_zeroshot(monkeypatch, scores):
    module = types.ModuleType("src.extractors.text._zeroshot")
    module.classify_segments = lambda texts, task, config: np.asarray(scores[: len(texts)], dtype=np.float64)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.extractors.text._zeroshot", module)


class TestTextBaseHelpers:
    def test_seg_bounds_clamps_to_video_duration(self, monkeypatch):
        base = install_text_package_stubs(monkeypatch)
        assert base.seg_bounds({"start": -1.2, "end": 3.1}, duration=3) == (0, 3)
        assert base.seg_bounds({"start": 2.9, "end": 2.1}, duration=10) == (2, 3)

    def test_skip_if_exists_only_when_all_columns_exist(self, monkeypatch):
        base = install_text_package_stubs(monkeypatch)
        assert base.skip_if_exists({"a", "b"}, ["a", "b", "c"], "feature") is True
        assert base.skip_if_exists({"a", "missing"}, ["a", "b"], "feature") is False
        assert base.skip_if_exists({"a"}, None, "feature") is False

    def test_empty_df_uses_at_least_one_row_and_zero_values(self, monkeypatch):
        base = install_text_package_stubs(monkeypatch)
        df = base.empty_df({"x", "y"}, duration=0)
        assert len(df) == 1
        assert set(df.columns) == {"x", "y"}
        assert float(df.to_numpy().sum()) == 0.0

    def test_valid_text_segments_and_segment_text_strip_whitespace(self, monkeypatch):
        base = install_text_package_stubs(monkeypatch)
        segments = [{"text": " hello "}, {"text": "   "}, {"text": ""}, {}, {"text": "world"}]
        assert base.valid_text_segments(segments) == [{"text": " hello "}, {"text": "world"}]
        assert base.segment_text({"text": "  trimmed  "}) == "trimmed"
        assert base.segment_text({}) == ""


class TestTextCommonHelpers:
    def test_collect_valid_segments_and_midpoints(self, monkeypatch):
        install_text_package_stubs(monkeypatch)
        install_fake_zeroshot(monkeypatch, [])
        common = load_module("src.extractors.text.common", "src/extractors/text/common.py")

        segments = ["bad", {"text": " a ", "start": 0.2, "end": 2.1}, {"text": "empty", "start": 5, "end": 4}, {"text": "   ", "start": 0, "end": 1}]

        assert common.collect_valid_segment_dicts(segments) == [{"text": " a ", "start": 0.2, "end": 2.1}, {"text": "empty", "start": 5, "end": 4}]
        assert common.collect_valid_segments(segments, duration=10) == [("a", 0, 3)]
        assert common.collect_valid_segments_with_mid(segments, duration=10) == [("a", 0, 3, 0.15)]

    def test_load_segment_embeddings_missing_and_with_metadata(self, tmp_path, monkeypatch):
        install_text_package_stubs(monkeypatch)
        install_fake_zeroshot(monkeypatch, [])
        common = load_module("src.extractors.text.common", "src/extractors/text/common.py")

        assert common.load_segment_embeddings("missing", embeddings_root=str(tmp_path)) == (None, [])

        video_dir = tmp_path / "vid"
        video_dir.mkdir()
        np.save(video_dir / "seg_embeddings.npy", np.ones((2, 3), dtype=np.float64))
        assert common.load_segment_embeddings("vid", embeddings_root=str(tmp_path), require_metadata=True) == (None, [])

        (video_dir / "seg_meta.json").write_text(json.dumps([{"start": 0, "end": 1}]), encoding="utf-8")
        emb, meta = common.load_segment_embeddings("vid", embeddings_root=str(tmp_path), require_metadata=True)
        assert emb.dtype == np.float32
        assert emb.shape == (2, 3)
        assert meta == [{"start": 0, "end": 1}]

    def test_score_regex_then_ensemble_and_spread_timeline(self, monkeypatch):
        install_text_package_stubs(monkeypatch)
        install_fake_zeroshot(monkeypatch, [0.7, 0.2])
        common = load_module("src.extractors.text.common", "src/extractors/text/common.py")

        class Pattern:
            def search(self, text):
                return "regex" in text

        valid = [("regex hit", 0, 2), ("model high", 2, 4), ("model low", 4, 5)]
        scores, regex_hits = common.score_regex_then_ensemble(valid, Pattern(), regex_score=1.0, task=None, config={}, ensemble_threshold=0.5, threshold_value=0.9)

        assert regex_hits == 1
        assert np.allclose(scores, [1.0, 0.9, 0.0])
        assert np.allclose(common.spread_scores_over_timeline(valid, scores, duration=6), [1.0, 1.0, 0.9, 0.9, 0.0, 0.0])


class TestEmotionFusion:
    def test_no_modalities_returns_zero_columns(self):
        compute_ekman_fusion = load_module("src.extractors.emotion_fusion", "src/extractors/emotion_fusion.py").compute_ekman_fusion
        out = compute_ekman_fusion(pd.DataFrame(index=[10, 11]))
        assert list(out.columns) == ["ekman_joy", "ekman_excitement", "ekman_sadness", "ekman_neutral", "ekman_intensity"]
        assert float(out.to_numpy().sum()) == 0.0
        assert list(out.index) == [10, 11]

    def test_voice_and_text_are_weighted_when_available(self):
        compute_ekman_fusion = load_module("src.extractors.emotion_fusion", "src/extractors/emotion_fusion.py").compute_ekman_fusion
        df = pd.DataFrame({"voice_happy": [1.0, 0.0], "sent_sadness": [0.0, 0.8], "sent_neutral": [0.2, 0.0]})
        out = compute_ekman_fusion(df)

        assert out.loc[0, "ekman_joy"] == pytest.approx(0.5)
        assert out.loc[0, "ekman_neutral"] == pytest.approx(0.1)
        assert out.loc[1, "ekman_sadness"] == pytest.approx(0.8)
        assert out["ekman_intensity"].tolist() == pytest.approx([0.5, 0.8])
