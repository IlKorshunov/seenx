"""Tests for src/retention_analysis.py"""

import os
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


matplotlib.use("Agg")

from src.retention_analysis import (
    _extrapolate_trend,
    _format_time,
    _resample_retention,
    channel_metrics_table,
    compute_avd_from_retention,
    compute_channel_baseline,
    compute_retention_at,
    load_channel_retentions,
    plot_all_strategies,
    plot_channel_baseline,
    plot_single_retention,
    video_retention_metrics,
)


@pytest.fixture
def flat_retention():
    """Constant 50% retention for 100 seconds."""
    return np.full(101, 50.0)


@pytest.fixture
def linear_retention():
    """Linearly decreasing retention: 100% -> 0% over 100 seconds."""
    return np.linspace(100, 0, 101)


@pytest.fixture
def sample_channel_data(flat_retention, linear_retention):
    return [
        {"name": "video_a", "retention_series": flat_retention, "duration_sec": 100.0, "time_index": np.arange(101)},
        {"name": "video_b", "retention_series": linear_retention, "duration_sec": 100.0, "time_index": np.arange(101)},
    ]


class TestComputeAvdFromRetention:
    def test_flat_50(self, flat_retention):
        assert compute_avd_from_retention(flat_retention, 100.0) == pytest.approx(50.0)

    def test_full_retention(self):
        ret = np.full(61, 100.0)
        assert compute_avd_from_retention(ret, 60.0) == pytest.approx(60.0)

    def test_zero_retention(self):
        ret = np.zeros(61)
        assert compute_avd_from_retention(ret, 60.0) == pytest.approx(0.0)

    def test_linear_retention(self, linear_retention):
        assert compute_avd_from_retention(linear_retention, 100.0) == pytest.approx(50.0)


class TestComputeRetentionAt:
    def test_at_30s(self, flat_retention):
        assert compute_retention_at(flat_retention, 30.0) == pytest.approx(50.0)

    def test_at_0s(self, linear_retention):
        assert compute_retention_at(linear_retention, 0.0) == pytest.approx(100.0)

    def test_video_too_short(self):
        short = np.array([100.0, 90.0, 80.0])
        assert compute_retention_at(short, 30.0) is None

    def test_at_last_second(self, linear_retention):
        assert compute_retention_at(linear_retention, 100.0) == pytest.approx(0.0)


class TestVideoRetentionMetrics:
    def test_keys(self, flat_retention):
        m = video_retention_metrics(flat_retention, 100.0)
        assert set(m.keys()) == {"duration_sec", "avd_sec", "avd_pct", "retention_30", "mean_retention"}

    def test_flat_values(self, flat_retention):
        m = video_retention_metrics(flat_retention, 100.0)
        assert m["duration_sec"] == 100.0
        assert m["avd_sec"] == pytest.approx(50.0)
        assert m["avd_pct"] == pytest.approx(50.0)
        assert m["mean_retention"] == pytest.approx(50.0)
        assert m["retention_30"] == pytest.approx(50.0)

    def test_short_video_no_30s(self):
        ret = np.full(11, 80.0)
        m = video_retention_metrics(ret, 10.0)
        assert m["retention_30"] is None

    def test_zero_duration(self):
        ret = np.array([100.0])
        m = video_retention_metrics(ret, 0.0)
        assert m["avd_pct"] == 0


class TestResampleRetention:
    def test_same_length(self):
        arr = np.array([100.0, 50.0, 0.0])
        result = _resample_retention(arr, 3)
        np.testing.assert_array_equal(result, arr)

    def test_upsample(self):
        arr = np.array([100.0, 0.0])
        result = _resample_retention(arr, 3)
        assert len(result) == 3
        assert result[0] == pytest.approx(100.0)
        assert result[1] == pytest.approx(50.0)
        assert result[2] == pytest.approx(0.0)

    def test_downsample(self):
        arr = np.array([100.0, 50.0, 0.0])
        result = _resample_retention(arr, 2)
        assert len(result) == 2
        assert result[0] == pytest.approx(100.0)
        assert result[-1] == pytest.approx(0.0)


class TestExtrapolateTrend:
    def test_longer_than_target_resamples_not_truncates(self):
        """Shrinking longer curves uses interpolation over full video [start..end], not [:target]."""
        arr = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
        result = _extrapolate_trend(arr, 3)
        expected = _resample_retention(arr, 3)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, expected)

    def test_extends_with_trend(self):
        arr = np.array([100.0, 90.0, 80.0])
        result = _extrapolate_trend(arr, 5)
        assert len(result) == 5
        np.testing.assert_array_equal(result[:3], arr)
        assert result[3] < result[2]
        assert result[4] < result[3]

    def test_clamped_to_zero(self):
        arr = np.array([10.0, 5.0, 1.0])
        result = _extrapolate_trend(arr, 10)
        assert len(result) == 10
        assert np.all(result >= 0)

    def test_same_length(self):
        arr = np.array([80.0, 60.0, 40.0])
        result = _extrapolate_trend(arr, 3)
        np.testing.assert_array_equal(result, arr)


class TestFormatTime:
    def test_zero(self):
        assert _format_time(0) == "0:00"

    def test_30_seconds(self):
        assert _format_time(30) == "0:30"

    def test_90_seconds(self):
        assert _format_time(90) == "1:30"

    def test_exact_minute(self):
        assert _format_time(120) == "2:00"

    def test_float_truncates(self):
        assert _format_time(90.7) == "1:30"


class TestLoadChannelRetentions:
    @patch("src.retention_analysis.get_video_duration", return_value=120.0)
    @patch("src.retention_analysis.parse_retention")
    @patch("src.retention_analysis.glob.glob")
    def test_loads_html_files(self, mock_glob, mock_parse, mock_dur, tmp_path):
        mock_glob.return_value = ["/data/html/vid1.html", "/data/html/vid2.html"]

        ret_df = pd.DataFrame({"retention": np.linspace(100, 50, 61)})
        mock_parse.return_value = ret_df

        (tmp_path / "vid1.mp4").write_bytes(b"x")

        result = load_channel_retentions("/data/html/", video_dir=str(tmp_path))

        assert len(result) == 2
        assert result[0]["name"] == "vid1"
        assert result[1]["name"] == "vid2"
        assert len(result[0]["retention_series"]) == 61

    @patch("src.retention_analysis.glob.glob", return_value=[])
    def test_empty_dir(self, mock_glob):
        result = load_channel_retentions("/empty/")
        assert result == []

    @patch("src.retention_analysis.parse_retention", side_effect=ValueError("bad html"))
    @patch("src.retention_analysis.glob.glob", return_value=["/data/bad.html"])
    def test_skips_broken_files(self, mock_glob, mock_parse):
        result = load_channel_retentions("/data/")
        assert result == []


class TestComputeChannelBaseline:
    def test_empty_data(self):
        assert compute_channel_baseline([]) is None

    def test_mean_duration_strategy(self, sample_channel_data):
        result = compute_channel_baseline(sample_channel_data, strategy="mean_duration")
        assert result is not None
        assert result["strategy"] == "mean_duration"
        assert result["n_videos"] == 2
        assert result["baseline_retention"].shape == (101,)

    def test_min_duration_strategy(self, sample_channel_data):
        result = compute_channel_baseline(sample_channel_data, strategy="min_duration")
        assert result["target_length"] == 101

    def test_max_duration_strategy(self, sample_channel_data):
        result = compute_channel_baseline(sample_channel_data, strategy="max_duration")
        assert result["target_length"] == 101

    def test_extrapolate_strategy(self, sample_channel_data):
        result = compute_channel_baseline(sample_channel_data, strategy="extrapolate")
        assert result is not None
        assert result["strategy"] == "extrapolate"

    def test_baseline_is_average(self, sample_channel_data):
        """With flat=50 and linear avg=50, baseline mean should be ~50."""
        result = compute_channel_baseline(sample_channel_data, strategy="mean_duration")
        baseline_mean = np.mean(result["baseline_retention"])
        assert baseline_mean == pytest.approx(50.0, abs=1.0)

    def test_baseline_metrics_present(self, sample_channel_data):
        result = compute_channel_baseline(sample_channel_data)
        bm = result["baseline_metrics"]
        assert "avd_sec" in bm
        assert "avd_pct" in bm
        assert "mean_retention" in bm

    def test_individual_metrics(self, sample_channel_data):
        result = compute_channel_baseline(sample_channel_data)
        assert len(result["individual_metrics"]) == 2
        assert result["individual_metrics"][0]["name"] == "video_a"

    def test_different_lengths(self):
        """Videos with different lengths should still compute baseline."""
        short = np.full(51, 60.0)
        long = np.full(201, 40.0)
        data = [
            {"name": "short", "retention_series": short, "duration_sec": 50.0, "time_index": np.arange(51)},
            {"name": "long", "retention_series": long, "duration_sec": 200.0, "time_index": np.arange(201)},
        ]
        result = compute_channel_baseline(data, strategy="mean_duration")
        assert result is not None
        expected_len = int(round(np.mean([51, 201])))
        assert result["target_length"] == expected_len


class TestChannelMetricsTable:
    def test_basic_table(self, sample_channel_data):
        df = channel_metrics_table(sample_channel_data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert df.iloc[-1]["name"] == "MEAN"

    def test_columns(self, sample_channel_data):
        df = channel_metrics_table(sample_channel_data)
        assert "name" in df.columns
        assert "avd_sec" in df.columns
        assert "duration_sec" in df.columns

    def test_mean_row_values(self, sample_channel_data):
        df = channel_metrics_table(sample_channel_data)
        mean_row = df[df["name"] == "MEAN"].iloc[0]
        assert mean_row["duration_sec"] == pytest.approx(100.0)


class TestPlots:
    def test_plot_single_retention(self, flat_retention):
        fig = plot_single_retention(flat_retention, 100.0, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_single_retention_saves_file(self, flat_retention, tmp_path):
        out = str(tmp_path / "test.png")
        fig = plot_single_retention(flat_retention, 100.0, output_path=out, show=False)
        assert os.path.exists(out)
        plt.close(fig)

    def test_plot_channel_baseline(self, sample_channel_data):
        result = compute_channel_baseline(sample_channel_data)
        fig = plot_channel_baseline(result, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_all_strategies(self, sample_channel_data):
        fig = plot_all_strategies(sample_channel_data, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
