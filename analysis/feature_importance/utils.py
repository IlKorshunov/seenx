"""Feature-importance helpers.
Loads feature CSVs, builds per-video rows and maps features to groups.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_GROUPS: dict[str, list[str]] = {
    "visual_quality": ["brightness", "sharpness", "cinematic", "visual_entropy", "color_temperature", "color_saturation"],
    "visual_motion": [
        "motion_speed",
        "edit_pace",
        "scene_novelty",
        "motion_speed_chg_5s",
        "motion_speed_abs_step_mean_5s",
        "edit_pace_chg_5s",
        "edit_pace_abs_step_mean_5s",
        "scene_novelty_chg_5s",
        "scene_novelty_abs_step_mean_5s",
        "motion_spike",
        "flow_mag_med",
        "radial_med",
        "radial_ratio",
        "frame",
        "short_insert",
        "short_insert_rate",
    ],
    "visual_content": ["speaker_prob", "face_screen_ratio", "face_area_ratio", "text_prob", "screencast_prob", "bumper_score"],
    "emotion": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    "audio_basic": ["rms", "zcr", "centroid", "rolloff"],
    "audio_music": ["music_rms", "music_zcr", "music_centroid", "music_rolloff", "beat_sync", "beat_sync_ratio", "music_only"],
    "audio_vocal": ["vocal_rms", "vocal_zcr", "vocal_centroid", "vocal_rolloff"],
    "audio_speech": ["speech_ratio", "silence_stretch", "wps", "pitch_mean", "pitch_std", "voiced_frac", "pause_rate", "speech_rate_cv"],
    "audio_loudness": ["loudness_change", "loudness_variance", "spectral_flux", "laughter_prob", "sfx_energy"],
    "text_complexity": ["syntactic_depth", "lexical_diversity", "avg_word_length", "speech_complexity", "speech_predictability"],
    "text_content": ["viewer_address", "crutch_cnt", "has_person_mention", "has_org_mention", "pos_cnt", "neg_cnt", "emo_cnt"],
    "hook": ["hook_score", "hook_has_question", "hook_has_address"],
    "ad": ["is_ad", "ad_segment_length"],
    "temporal": ["time_pct"],
    "interaction": ["edit_pace_x_screencast", "hook_score_x_time_pct", "is_ad_x_viewer_address"],
}

ALL_FEATURES: list[str] = [f for feats in FEATURE_GROUPS.values() for f in feats]
NON_FEATURE_COLS = {"time", "retention"}


def load_features_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time_sec"] = pd.to_timedelta(df["time"]).dt.total_seconds()
    return df


def load_all_videos(output_dir: str | Path = "output") -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    paths = sorted(glob.glob(str(output_dir / "*_features.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_features.csv found in {output_dir}")
    return {Path(path).stem.replace("_features", ""): load_features_csv(path) for path in paths}


def aggregate_per_video(video_dfs: dict[str, pd.DataFrame], agg: str = "mean") -> pd.DataFrame:
    aggregators = {"mean": lambda frame: frame.mean(), "median": lambda frame: frame.median(), "std": lambda frame: frame.std().fillna(0.0), "max": lambda frame: frame.max()}
    if agg not in aggregators:
        raise ValueError(f"Unknown agg: {agg}")
    rows = []
    for video_id, df in video_dfs.items():
        feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS and col != "time_sec" and pd.api.types.is_numeric_dtype(df[col])]
        row = aggregators[agg](df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0))
        if "retention" in df.columns:
            retention = df["retention"].apply(pd.to_numeric, errors="coerce").dropna().values
            cutoff = max(1, int(0.2 * len(retention)))
            row["target_avg_retention"] = float(retention.mean()) if len(retention) else np.nan
            row["target_drop_rate"] = float((retention[0] - retention[-1]) / max(retention[0], 1e-6)) if len(retention) else np.nan
            row["target_early_drop"] = float(retention[:cutoff].mean()) if len(retention) else np.nan
        row["video_id"] = video_id
        rows.append(row)
    return pd.DataFrame(rows).set_index("video_id")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {c for c in df.columns if c.startswith("target_")}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def get_target_column(df: pd.DataFrame, target: str = "target_avg_retention") -> pd.Series:
    return df[target].dropna()


def prepare_X_y(df: pd.DataFrame, target: str = "target_avg_retention") -> tuple[pd.DataFrame, pd.Series]:
    feat_cols = get_feature_columns(df)
    valid = df[target].notna()
    X = df.loc[valid, feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df.loc[valid, target].astype(float)
    return X, y


def feature_group_of(feature: str) -> str:
    for group, feats in FEATURE_GROUPS.items():
        if feature in feats:
            return group
    return "unknown"


def save_importance_csv(importance: pd.DataFrame, out_path: str | Path, sort_by: str | None = None) -> None:
    if sort_by and sort_by in importance.columns:
        importance = importance.sort_values(sort_by, ascending=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out_path)
    print(f"Saved: {out_path}")


def default_output_dir(base: str = "analysis/feature_importance/results") -> Path:
    root = Path(__file__).parent.parent.parent
    return root / base
