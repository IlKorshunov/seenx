import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .extractors.audio import (
    extract_clap_embeddings,
    extract_background_music_features,
    extract_beat_sync,
    extract_laughter,
    extract_loudness_dynamics,
    extract_prosody,
    extract_sfx_energy,
    extract_spectral_flux,
    extract_speech_music_silence,
    get_vocal_music_features,
    sound_features_pipeline,
)
from .extractors.emotion_fusion import compute_ekman_fusion
from .extractors.text import (
    extract_ad_segments,
    extract_chapters,
    extract_clickbait_gap,
    extract_comment_features,
    extract_cultural_references,
    extract_curiosity_gap,
    extract_examples,
    extract_hook_score,
    extract_information_density,
    extract_mm_embeddings,
    extract_sections,
    extract_semantic_embeddings,
    extract_speech_fillers,
    extract_speech_intelligibility,
    extract_speech_lm_surprisal,
    extract_speech_predictability,
    extract_storytelling,
    extract_text_complexity,
    extract_text_sentiment,
    extract_topic_sharpness,
    extract_viewer_address,
    extract_viewer_engagement,
    extract_wps,
)
from .extractors.video import (
    BumperFeature,
    ColorFeature,
    EditPaceFeature,
    FaceScreenRatioFeature,
    FrameQualityFeature,
    MotionSpeedFeature,
    OverlayFeature,
    SceneNoveltyFeature,
    ScreencastFeature,
    ShortInsertFeature,
    SpeakerProbabilityFeature,
    TextProbFeature,
    ZoomFeatureExtractor,
    extract_object_density,
)
from .extractors.video.video_intelligence_feature import extract_video_intelligence
from .extractors.video.common import embeddings_dir as video_embeddings_dir
from .extractors.video.videomae_feature import extract_videomae_embeddings
from .extractors.video.zoom_features import mask_flow_at_cuts
from .seenx_utils import get_video_duration
from .speaker_features import run_feature_pipeline
from .utils.config import Config
from .utils.logger import Logger
from .utils.transcript_cache import clear_cache as clear_transcript_cache


logger = Logger(show=True).get_logger()

CHECKPOINT_SUFFIX = ".partial"
FAILED_SUFFIX = ".failed_features.json"


def _checkpoint_path(output_path: str) -> str:
    return output_path + CHECKPOINT_SUFFIX


def _failed_path(output_path: str) -> str:
    return output_path + FAILED_SUFFIX


def _save_checkpoint(df: pd.DataFrame, output_path: str) -> None:
    path = _checkpoint_path(output_path)
    df.to_csv(path, index=True)
    logger.info("Checkpoint saved (%d cols) -> %s", len(df.columns), path)


def _load_failed_features(output_path: str) -> set[str]:
    path = _failed_path(output_path)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        cols = set(json.load(f))
    logger.info("Loaded %d previously failed feature cols -> will re-extract", len(cols))
    return cols


def _save_failed_features(output_path: str, cols: set[str]) -> None:
    path = _failed_path(output_path)
    if not cols:
        if os.path.exists(path):
            os.remove(path)
        return
    with open(path, "w") as f:
        json.dump(sorted(cols), f)
    logger.info("Saved %d failed feature cols -> %s", len(cols), path)


# Seconds: aligned with local spike MA window; edit_pace / motion / scene_novelty dynamics
_VIS_DYNAMICS_WINDOW_SEC = 5
_VIS_DYNAMICS_SPIKE_SOURCES = {"motion_speed": "motion_spike", "pitch_mean": "pitch_spike", "brightness": "brightness_spike", "visual_entropy": "entropy_spike"}
_VIS_DYNAMICS_DELTA_BASES = ("motion_speed", "edit_pace", "scene_novelty")


def _append_local_spike_and_visual_dynamics(acc: pd.DataFrame) -> None:
    """Spike ratios (value / rolling mean) + rolling change stats for montage / motion / scene pace."""
    W = _VIS_DYNAMICS_WINDOW_SEC
    for col, new_col in _VIS_DYNAMICS_SPIKE_SOURCES.items():
        if col not in acc.columns:
            continue
        ma = acc[col].rolling(window=W, min_periods=1).mean()
        acc[new_col] = (acc[col] / (ma + 1e-6)).astype(np.float32)
    for col in _VIS_DYNAMICS_DELTA_BASES:
        if col not in acc.columns:
            continue
        s = pd.to_numeric(acc[col], errors="coerce").fillna(0.0).astype(np.float64)
        acc[f"{col}_chg_{W}s"] = (s - s.shift(W)).fillna(0.0).astype(np.float32)
        acc[f"{col}_abs_step_mean_{W}s"] = s.diff().abs().rolling(W, min_periods=1).mean().fillna(0.0).astype(np.float32)


def _load_checkpoint(output_path: str) -> tuple[pd.DataFrame, list]:
    path = _checkpoint_path(output_path)
    if not os.path.exists(path):
        return pd.DataFrame(), []
    df = pd.read_csv(path, index_col=0)
    features = df.columns.tolist()
    logger.info("Resuming from checkpoint (%d cols): %s", len(features), path)
    return df, features


def get_retention(video_path: str, html_path: str = None) -> pd.DataFrame:
    """
    Load or build audience retention. When html_path is a CSV (e.g. retention.csv):
    values are YouTube audience_watch_ratio * 100, interpolated to 1 fps via
    np.interp; this is ground truth and should not be smoothed.

    Formula (CSV case). Given CSV columns time_ratio τ and audience_watch_ratio r,
    video duration T seconds, for each integer second t in 0..T:
      τ_t = t / T,
      retention(t) = 100 * interp(τ_t, τ_csv, r_csv),
    where interp is piecewise linear (np.interp).
    """
    video_duration = get_video_duration(video_path)
    if html_path is None:
        logger.info("Creating empty audience retention data")
        retention_index = pd.to_timedelta(np.arange(0, int(video_duration) + 1, 1), unit="s")
        retention = pd.DataFrame(index=retention_index)
    else:
        logger.info("Loading audience retention from CSV: %s", html_path)
        csv_df = pd.read_csv(html_path)
        n_points = int(video_duration) + 1
        time_seconds = np.arange(n_points)
        retention_pct = np.interp(np.linspace(0, 1, n_points), csv_df["time_ratio"].values, csv_df["audience_watch_ratio"].values * 100.0)
        retention_index = pd.to_timedelta(time_seconds, unit="s")
        retention = pd.DataFrame({"retention": retention_pct}, index=retention_index)
        retention.index.name = "time"
    return retention


def _should_run(step_cols: set[str], only: set[str] | None, existing: set[str]) -> bool:
    missing = step_cols - existing
    want_overwrite = only and (step_cols & only & existing)
    if not missing and not want_overwrite:
        return False
    return only is None or bool(missing & only) or bool(want_overwrite)


def _prepare_shot_boundary_embeddings(video_path: str, config: Config) -> None:
    if config.get("use_shot_ensemble") is False or config.get("shot_boundary_ensemble") is False:
        return
    disabled = set(config.get("ensemble_disable") or [])
    root = video_embeddings_dir(video_path)
    if config.get("use_clap") is not False and "clap" not in disabled and not (root / "audio_embeddings.npy").is_file():
        logger.info("Preparing CLAP audio embeddings before shot boundary ensemble")
        try:
            extract_clap_embeddings(video_path=video_path, config=config, existing_features=None)
        except (ImportError, RuntimeError, OSError, ValueError) as error:
            logger.warning("Could not prepare CLAP embeddings before shot ensemble: %s", error)
    if "videomae" not in disabled and not (root / "videomae_embeddings.npy").is_file():
        logger.info("Preparing VideoMAE embeddings before shot boundary ensemble")
        try:
            extract_videomae_embeddings(video_path=video_path, config=config)
        except (ImportError, RuntimeError, OSError, ValueError) as error:
            logger.warning("Could not prepare VideoMAE embeddings before shot ensemble: %s", error)


def aggregate(
    video_path: str,
    audio_path: str,
    output_path: str,
    config: Config,
    html_path: str = None,
    data_dir: str = "data",
    only: set[str] | None = None,
    skip_comment_features: bool = False,
    skip_emotion_features: bool = False,
):
    accumulated, existing_features = _load_checkpoint(output_path)
    if accumulated.empty and os.path.exists(output_path):
        try:
            accumulated = pd.read_csv(output_path, index_col=0)
            existing_features = accumulated.columns.tolist()
            logger.info("Loaded existing output (%d cols): %s", len(existing_features), output_path)
        except pd.errors.EmptyDataError:
            logger.warning("Existing output file is empty, starting fresh: %s", output_path)
            accumulated = pd.DataFrame()
            existing_features = []
    existing_features_set = set(existing_features)

    _FACE_EMOTION_COLS = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}
    _VOICE_EMOTION_COLS = {"voice_angry", "voice_happy", "voice_sad", "voice_neutral", "voice_dominant_emotion_conf"}
    _TEXT_EMOTION_COLS = {
        f"sent_{e}"
        for e in (
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        )
    }
    _EKMAN_COLS = {f"ekman_{e}" for e in ("joy", "excitement", "sadness", "neutral")} | {"ekman_intensity"}
    _RAW_EMOTION_COLS = _FACE_EMOTION_COLS | _VOICE_EMOTION_COLS | _TEXT_EMOTION_COLS

    _ALL_EXPECTED = (
        {"retention"}
        | {
            "brightness",
            "sharpness",
            "speaker_prob",
            "face_screen_ratio",
            "faces_total_ratio",
            "text_prob",
            "motion_speed",
            "edit_pace",
            "scene_novelty",
            "screencast_prob",
            "overlay_prob",
            "bumper_score",
            "visual_entropy",
            "visual_complexity",
            "visual_complexity_gradient",
            "visual_complexity_acceleration",
        }
        | {"rms", "zcr", "centroid", "rolloff"}
        | {
            "music_rms",
            "music_zcr",
            "music_centroid",
            "music_rolloff",
            "vocal_rms",
            "vocal_zcr",
            "vocal_centroid",
            "vocal_rolloff",
            "speech_ratio",
            "silence_stretch",
            "music_only",
            "has_background_music",
            "music_changed",
        }
        | {"flow_mag_med", "radial_med", "radial_ratio"}
        | {"wps"}
        | {"viewer_address"}
        | {"crutch_cnt"}
        | {"pitch_mean", "pitch_std", "voiced_frac", "speech_rate_cv", "pause_rate"}
        | {"beat_sync", "beat_sync_ratio"}
        | {"is_ad", "ad_segment_length"}
        | {"syntactic_depth", "lexical_diversity", "avg_word_length", "speech_complexity"}
        | {"has_person_mention", "has_org_mention"}
        | _EKMAN_COLS
        | {"hook_score", "hook_has_address", "is_question"}
        | {"loudness_change", "loudness_variance"}
        | {"semantic_novelty", "topic_shift", "hook_similarity", "semantic_momentum", "segment_self_similarity"}
        | {"object_count", "unique_classes"}
        | {"chapter_id", "n_chapters", "topic_change_rate"}
        | {"speech_intelligibility", "speech_mumble_index"}
        | {"curiosity_gap"}
        | {"storytelling"}
        | {"viewer_engagement"}
        | {"has_example"}
        | {"question_density"}
        | {"is_intro", "is_outro"}
        | {"edit_pace_x_screencast", "is_ad_x_viewer_address"}
        | {"content_rhythm", "visual_audio_sync", "narrative_momentum", "engagement_surprise"}
        | {"color_temperature", "color_saturation", "face_area_ratio"}
        | {"short_insert", "short_insert_rate"}
        | {"spectral_flux"}
        | {"laughter_prob"}
        | {"sfx_energy"}
        | {"speech_predictability"}
        | {"speech_lm_surprisal", "speech_lm_surprisal_vel"}
        | {
            "desc_chapter_start",
            "desc_chapter_boundary_dist",
            "timecode_like_weighted_30s",
            "comment_question_rate_30s",
            "comment_density_30s",
            "comment_positive_rate_30s",
            "comment_reply_depth_30s",
            "comment_aggression_rate_30s",
            "author_reply_rate_video",
            "avg_comment_length_video",
            "complex_words_ratio_video",
        }
        | {"title_transcript_gap", "title_delivery_30s", "title_claim_intensity"}
    )
    if _ALL_EXPECTED.issubset(existing_features_set):
        logger.info("All features already computed (%d cols), skipping %s", len(existing_features_set), output_path)
        return accumulated

    if "retention" not in existing_features:
        retention = get_retention(video_path=video_path, html_path=html_path)
        if retention.index.name is None:
            retention.index.name = "time"
    else:
        retention = accumulated[["retention"]].copy()

    logger.info("Retention shape: %s", retention.shape)
    t_total = time.time()

    def map_to_retention(features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame(index=retention.index)
        mapped = pd.DataFrame(index=retention.index)
        for col in features.columns:
            fp = features[col].astype("float64").values
            mapped[col] = np.interp(np.linspace(0, len(features) - 1, len(retention)), np.arange(len(features)), fp)
        return mapped

    def add_to_accumulated(mapped: pd.DataFrame) -> None:
        nonlocal accumulated, existing_features, existing_features_set
        new_cols = [c for c in mapped.columns if c not in accumulated.columns]
        overwrite_cols = [c for c in mapped.columns if c in accumulated.columns and c in only] if only else []
        if new_cols:
            accumulated = pd.concat([accumulated, mapped[new_cols]], axis=1)
        if overwrite_cols:
            accumulated[overwrite_cols] = mapped[overwrite_cols]
        if new_cols or overwrite_cols:
            existing_features = accumulated.columns.tolist()
            existing_features_set = set(existing_features)
            _save_checkpoint(accumulated, output_path)

    if accumulated.empty:
        accumulated = retention.copy()
        _save_checkpoint(accumulated, output_path)

    # Drop columns that previously failed so they get re-extracted.
    prev_failed = _load_failed_features(output_path)
    if prev_failed:
        drop_cols = prev_failed & existing_features_set
        if drop_cols:
            accumulated = accumulated.drop(columns=list(drop_cols), errors="ignore")
            existing_features = accumulated.columns.tolist()
            existing_features_set = set(existing_features)
            logger.info("Dropped %d previously-failed cols for re-extraction: %s", len(drop_cols), sorted(drop_cols))
            _save_checkpoint(accumulated, output_path)

    # Drop broken feature columns (all-zero or constant) so they get re-extracted.
    # Skip columns that are constant by design (one value per video).
    _SKIP_QUALITY_CHECK = {"retention", "time", "time_pct", "hook_score", "hook_has_address", "n_chapters"}
    bad_cols = []
    for c in accumulated.columns:
        if c in _SKIP_QUALITY_CHECK:
            continue
        if not pd.api.types.is_numeric_dtype(accumulated[c]):
            continue
        vals = accumulated[c].dropna()
        if len(vals) == 0 or (vals == 0).all() or (float(vals.std()) < 1e-9 and len(vals) > 10):
            bad_cols.append(c)
    if bad_cols:
        accumulated = accumulated.drop(columns=bad_cols, errors="ignore")
        existing_features = accumulated.columns.tolist()
        existing_features_set = set(existing_features)
        logger.info("Dropped %d broken cols (all-zero or constant) for re-extraction: %s", len(bad_cols), sorted(bad_cols))
        _save_checkpoint(accumulated, output_path)

    _FACE_DEPENDENT = {"speaker_prob", "face_screen_ratio", "faces_total_ratio", "face_area_ratio"}
    if "speaker_prob" in existing_features_set and "speaker_prob" in accumulated.columns:
        sp = accumulated["speaker_prob"]
        zero_ratio = float((sp == 0).sum()) / max(len(sp), 1)
        if zero_ratio > 0.50:
            bad_face = _FACE_DEPENDENT & existing_features_set
            if bad_face:
                accumulated = accumulated.drop(columns=list(bad_face), errors="ignore")
                existing_features = accumulated.columns.tolist()
                existing_features_set = set(existing_features)
                logger.info("speaker_prob zero_ratio=%.1f%% — dropped %d face-dependent cols for re-extraction: %s", zero_ratio * 100, len(bad_face), sorted(bad_face))
                _save_checkpoint(accumulated, output_path)

    _VISUAL_COLS = {
        "brightness",
        "sharpness",
        "speaker_prob",
        "face_screen_ratio",
        "faces_total_ratio",
        "text_prob",
        "motion_speed",
        "edit_pace",
        "scene_novelty",
        "screencast_prob",
        "overlay_prob",
        "bumper_score",
        "visual_entropy",
        "visual_complexity",
        "visual_complexity_gradient",
        "visual_complexity_acceleration",
        "color_temperature",
        "color_saturation",
        "face_area_ratio",
        "short_insert",
        "short_insert_rate",
    }
    if _should_run(_VISUAL_COLS, only, existing_features_set):
        _prepare_shot_boundary_embeddings(video_path, config)
        t0 = time.time()
        logger.info("Extracting visual features")
        all_passes = [
            FrameQualityFeature(config),
            ColorFeature(config),
            SpeakerProbabilityFeature(config),
            FaceScreenRatioFeature(config),
            TextProbFeature(config),
            MotionSpeedFeature(config),
            # EmotionFeature(config),  # TODO: re-enable when face emotion model is chosen
            EditPaceFeature(config),
            ShortInsertFeature(config),
            SceneNoveltyFeature(config),
            ScreencastFeature(config),
            OverlayFeature(config),
            BumperFeature(config, data_dir=data_dir),
        ]
        if only is not None:
            all_passes = [p for p in all_passes if p.produces_keys() & only]
        speaker_features = run_feature_pipeline(video_path, config, passes=all_passes, existing_features=existing_features_set)
        speaker_features = speaker_features.drop(columns=["frame_face_boxes", "frame_keypoints", "frame_idx"], errors="ignore")
        logger.info("Visual features done in %.1fs", time.time() - t0)
        mapped_visual = map_to_retention(speaker_features)
        for col in mapped_visual.columns:
            add_to_accumulated(mapped_visual[[col]])
        del all_passes, speaker_features, mapped_visual
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GPU memory freed after visual features")

    # --- Demucs (raw stem features only) ---
    _DEMUCS_RAW = {"music_rms", "music_zcr", "music_centroid", "music_rolloff", "vocal_rms", "vocal_zcr", "vocal_centroid", "vocal_rolloff"}
    if _should_run(_DEMUCS_RAW, only, existing_features_set):
        t0 = time.time()
        logger.info("Extracting demucs features")
        try:
            music_features, vocal_features = get_vocal_music_features(audio_path=audio_path, config=config, existing_features=existing_features)
            music_mapped = map_to_retention(music_features)
            vocal_mapped = map_to_retention(vocal_features)
            add_to_accumulated(music_mapped)
            add_to_accumulated(vocal_mapped)
            logger.info("Demucs done in %.1fs", time.time() - t0)
        except Exception as e:
            logger.error("Demucs FAILED (%.1fs): %s — skipped", time.time() - t0, e)

    # --- Derived from Demucs stems (no Demucs re-run needed) ---
    _SMS_COLS = {"speech_ratio", "silence_stretch", "music_only"}
    need_sms = _SMS_COLS - set(accumulated.columns)
    if need_sms and all(c in accumulated.columns for c in ["vocal_rms", "music_rms"]):
        sms_df = extract_speech_music_silence(accumulated["vocal_rms"].values, accumulated["music_rms"].values)
        sms_df.index = retention.index
        add_to_accumulated(sms_df)

    need_bgm = {"has_background_music", "music_changed"} - set(accumulated.columns)
    if need_bgm and all(c in accumulated.columns for c in ["music_rms", "music_centroid", "music_rolloff"]):
        bgm_df = extract_background_music_features(accumulated["music_rms"].values, accumulated["music_centroid"].values, accumulated["music_rolloff"].values)
        bgm_df.index = retention.index
        add_to_accumulated(bgm_df)

    def _zoom_extract():
        zf = ZoomFeatureExtractor(video_path=video_path, config=config).run()
        zf = zf.drop(columns=["frame_idx"], errors="ignore")
        return mask_flow_at_cuts(zf, video_path, config)

    _STEPS = [
        ("sound", {"rms", "zcr", "centroid", "rolloff"}, lambda: sound_features_pipeline(audio_file_path=audio_path, existing_features=existing_features)),
        ("zoom", {"flow_mag_med", "radial_med", "radial_ratio"}, _zoom_extract),
        ("wps", {"wps"}, lambda: extract_wps(video_path=video_path, config=config, existing_features=existing_features)),
        ("viewer address", {"viewer_address"}, lambda: extract_viewer_address(video_path=video_path, config=config, existing_features=existing_features)),
        ("speech fillers", {"crutch_cnt"}, lambda: extract_speech_fillers(video_path=video_path, config=config, existing_features=existing_features)),
        (
            "prosody",
            {"pitch_mean", "pitch_std", "voiced_frac", "speech_rate_cv", "pause_rate"},
            lambda: extract_prosody(video_path=video_path, config=config, existing_features=existing_features),
        ),
        ("beat-sync", {"beat_sync", "beat_sync_ratio"}, lambda: extract_beat_sync(video_path=video_path, config=config, existing_features=existing_features)),
        ("ad segments", {"is_ad", "ad_segment_length"}, lambda: extract_ad_segments(video_path=video_path, config=config, existing_features=existing_features)),
        (
            "text complexity",
            {"syntactic_depth", "lexical_diversity", "avg_word_length", "speech_complexity"},
            lambda: extract_text_complexity(video_path=video_path, config=config, existing_features=existing_features),
        ),
        (
            "cultural refs",
            {"has_person_mention", "has_org_mention"},
            lambda: extract_cultural_references(video_path=video_path, config=config, existing_features=existing_features),
        ),
        (
            "sentiment",
            {
                f"sent_{e}"
                for e in (
                    "admiration",
                    "amusement",
                    "anger",
                    "annoyance",
                    "approval",
                    "caring",
                    "confusion",
                    "curiosity",
                    "desire",
                    "disappointment",
                    "disapproval",
                    "disgust",
                    "embarrassment",
                    "excitement",
                    "fear",
                    "gratitude",
                    "grief",
                    "joy",
                    "love",
                    "nervousness",
                    "optimism",
                    "pride",
                    "realization",
                    "relief",
                    "remorse",
                    "sadness",
                    "surprise",
                    "neutral",
                )
            },
            lambda: extract_text_sentiment(video_path=video_path, config=config, existing_features=existing_features),
        ),
        ("hook score", {"hook_score", "hook_has_address", "is_question"}, lambda: extract_hook_score(video_path=video_path, config=config, existing_features=existing_features)),
        (
            "loudness dynamics",
            {"loudness_change", "loudness_variance"},
            lambda: extract_loudness_dynamics(video_path=video_path, config=config, existing_features=existing_features),
        ),
        (
            "semantic embeddings",
            {"semantic_novelty", "topic_shift", "hook_similarity", "global_topic_dist", "semantic_momentum", "segment_self_similarity"},
            lambda: extract_semantic_embeddings(video_path=video_path, config=config, existing_features=existing_features),
        ),
        ("object density", {"object_count", "unique_classes"}, lambda: extract_object_density(video_path=video_path, config=config, existing_features=existing_features)),
        (
            "video intelligence",
            {"content_rhythm", "visual_audio_sync", "narrative_momentum", "engagement_surprise"},
            lambda: extract_video_intelligence(video_path=video_path, config=config, existing_features=existing_features),
        ),
        ("chapters", {"chapter_id", "n_chapters", "topic_change_rate"}, lambda: extract_chapters(video_path=video_path, config=config, existing_features=existing_features)),
        ("curiosity gap", {"curiosity_gap"}, lambda: extract_curiosity_gap(video_path=video_path, config=config, existing_features=existing_features)),
        ("topic sharpness (Qwen)", {"topic_sharpness_0_100"}, lambda: extract_topic_sharpness(video_path=video_path, config=config, existing_features=existing_features)),
        (
            "speech intelligibility",
            {"speech_intelligibility", "speech_mumble_index"},
            lambda: extract_speech_intelligibility(video_path=video_path, config=config, existing_features=existing_features),
        ),
        ("storytelling", {"storytelling"}, lambda: extract_storytelling(video_path=video_path, config=config, existing_features=existing_features)),
        ("viewer engagement", {"viewer_engagement"}, lambda: extract_viewer_engagement(video_path=video_path, config=config, existing_features=existing_features)),
        ("examples", {"has_example"}, lambda: extract_examples(video_path=video_path, config=config, existing_features=existing_features)),
        ("sections", {"is_intro", "is_outro"}, lambda: extract_sections(video_path=video_path, config=config, existing_features=existing_features)),
        (
            "information density",
            {"information_density", "cumulative_info"},
            lambda: extract_information_density(video_path=video_path, config=config, existing_features=existing_features),
        ),
        ("spectral flux", {"spectral_flux"}, lambda: extract_spectral_flux(video_path=video_path, config=config, existing_features=existing_features)),
        ("laughter", {"laughter_prob"}, lambda: extract_laughter(video_path=video_path, config=config, existing_features=existing_features)),
        ("sfx energy", {"sfx_energy"}, lambda: extract_sfx_energy(video_path=video_path, config=config, existing_features=existing_features)),
        ("speech predictability", {"speech_predictability"}, lambda: extract_speech_predictability(video_path=video_path, config=config, existing_features=existing_features)),
        (
            "speech lm surprisal",
            {"speech_lm_surprisal", "speech_lm_surprisal_vel"},
            lambda: extract_speech_lm_surprisal(video_path=video_path, config=config, existing_features=existing_features),
        ),
        (
            "comment features",
            {
                "desc_chapter_start",
                "desc_chapter_boundary_dist",
                "timecode_like_weighted_30s",
                "comment_question_rate_30s",
                "comment_density_30s",
                "comment_positive_rate_30s",
                "comment_reply_depth_30s",
                "comment_aggression_rate_30s",
                "author_reply_rate_video",
                "avg_comment_length_video",
                "complex_words_ratio_video",
            },
            lambda: extract_comment_features(video_path=video_path, config=config, existing_features=existing_features),
        ),
        (
            "clickbait gap",
            {"title_transcript_gap", "title_delivery_30s", "title_claim_intensity"},
            lambda: extract_clickbait_gap(video_path=video_path, config=config, existing_features=existing_features),
        ),
    ]

    if _EKMAN_COLS.issubset(existing_features_set):
        existing_features_set |= _RAW_EMOTION_COLS

    failed_cols: set[str] = set()
    for name, cols, fn in _STEPS:
        if skip_comment_features and name == "comment features":
            logger.info("Skipping comment features (--skip-comment-features)")
            continue
        if skip_emotion_features and name == "sentiment":
            logger.info("Skipping text sentiment / sent_* (--skip-emotion-features)")
            continue
        if not _should_run(cols, only, existing_features_set):
            continue
        t0 = time.time()
        logger.info("Extracting %s", name)
        try:
            result = fn()
            if result is not None and not result.empty:
                add_to_accumulated(map_to_retention(result))
            logger.info("%s done in %.1fs", name, time.time() - t0)
        except Exception as e:
            logger.error("%s FAILED (%.1fs): %s — skipped, will retry next run", name, time.time() - t0, e)
            failed_cols |= cols
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if _should_run({"mm_embed"}, only, existing_features_set):
        try:
            mm = extract_mm_embeddings(video_path, config)
            if mm is not None and not mm.empty:
                add_to_accumulated(map_to_retention(mm))
        except Exception as e:
            logger.error("MM embeddings FAILED: %s — skipped", e)

    clear_transcript_cache()

    # --- Refine is_intro / is_outro using Hill curve when retention is available ---
    if "is_intro" in accumulated.columns and "retention" in accumulated.columns and accumulated["retention"].std() > 1e-3:
        try:
            from .analysis.segmented_trainer import fit_hill_curve

            time_sec = np.arange(len(accumulated), dtype=np.float64)
            _, params = fit_hill_curve(time_sec, accumulated["retention"].values)
            inflection_sec = params[2]
            cap = int(min(inflection_sec * 2.0, len(accumulated)))
            intro_vals = accumulated["is_intro"].values.copy()
            if cap < len(intro_vals):
                intro_vals[cap:] = 0.0
                accumulated["is_intro"] = intro_vals
                logger.info("Hill-curve intro cap: inflection=%.0fs, hard cap=%ds", inflection_sec, cap)
        except Exception as e:
            logger.warning("Hill-curve intro refinement failed: %s", e)

    if "edit_pace" in accumulated.columns and "screencast_prob" in accumulated.columns:
        accumulated["edit_pace_x_screencast"] = accumulated["edit_pace"] * accumulated["screencast_prob"]
    if "is_ad" in accumulated.columns and "viewer_address" in accumulated.columns:
        accumulated["is_ad_x_viewer_address"] = accumulated["is_ad"] * accumulated["viewer_address"]

    _QD_WINDOW_SEC = 30
    if "is_question" in accumulated.columns and "question_density" not in accumulated.columns:
        iq = accumulated["is_question"].values.astype(np.float64)
        q_starts = (np.diff(iq, prepend=0) > 0).astype(np.float64)
        rolling_count = np.convolve(q_starts, np.ones(_QD_WINDOW_SEC), mode="same")
        accumulated["question_density"] = rolling_count / (_QD_WINDOW_SEC / 60.0)
        logger.info("Derived question_density from is_question (window=%ds)", _QD_WINDOW_SEC)

    # --- Ekman emotion fusion (face + voice + text → 7 Ekman + intensity) ---
    if not skip_emotion_features:
        if not _EKMAN_COLS.issubset(set(accumulated.columns)):
            has_any_emotion = bool(_RAW_EMOTION_COLS & set(accumulated.columns))
            if has_any_emotion:
                logger.info("Computing Ekman emotion fusion")
                ekman_df = compute_ekman_fusion(accumulated)
                for col in ekman_df.columns:
                    accumulated[col] = ekman_df[col]

        # Drop raw emotion columns — only keep fused Ekman in final output.
        drop_raw = sorted(_RAW_EMOTION_COLS & set(accumulated.columns))
        if drop_raw and _EKMAN_COLS.issubset(set(accumulated.columns)):
            accumulated = accumulated.drop(columns=drop_raw, errors="ignore")
            logger.info("Dropped %d raw emotion cols (Ekman fusion present): %s", len(drop_raw), drop_raw)
    else:
        logger.info("Skipping Ekman fusion and raw-emotion drop (--skip-emotion-features)")

    _DEPRECATED_EKMAN = {"ekman_anger", "ekman_disgust", "ekman_fear", "ekman_surprise"}
    _REMOVED_EXPENSIVE = {
        "aesthetic_score",
        "depth_variance",
        "depth_mean",
        "audio_novelty",
        "audio_topic_shift",
        "audio_hook_similarity",
        "audio_global_dist",
        "audio_momentum",
        "audio_self_similarity",
        "visual_topic_shift",
        "visual_hook_similarity",
        "visual_global_dist",
        "visual_momentum",
        "visual_self_similarity",
        "voice_angry",
        "voice_happy",
        "voice_sad",
        "voice_neutral",
        "voice_dominant_emotion_conf",
    }
    # Redundant with loudness_dynamics + video_intelligence (internal drift / RMS z-channels)
    _REDUNDANT_DUPLICATE = {"embedding_drift", "embedding_drift_smoothed", "loudness_zscore", "loudness_spike", "loudness_drop"}
    _REDUNDANT_COLS = {"visual_novelty", "global_topic_dist"} | _DEPRECATED_EKMAN | _REMOVED_EXPENSIVE | _REDUNDANT_DUPLICATE
    drop_redundant = sorted(_REDUNDANT_COLS & set(accumulated.columns))
    if drop_redundant:
        accumulated = accumulated.drop(columns=drop_redundant, errors="ignore")
        logger.info("Dropped %d redundant/deprecated/removed cols: %s", len(drop_redundant), drop_redundant)

    _append_local_spike_and_visual_dynamics(accumulated)

    _save_failed_features(output_path, failed_cols)

    logger.info("Total aggregation time: %.1fs (%.1f min)", time.time() - t_total, (time.time() - t_total) / 60)
    if failed_cols:
        logger.warning("Failed features (%d cols): %s — will retry on next run", len(failed_cols), sorted(failed_cols))

    checkpoint_path = _checkpoint_path(output_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("Removed checkpoint (run complete)")

    return accumulated


# ---------------------------------------------------------------------------
# Batch-mode: one model loaded → process all videos → unload → next model
# ---------------------------------------------------------------------------

_FACE_EMOTION_COLS = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}
_VOICE_EMOTION_COLS = {"voice_angry", "voice_happy", "voice_sad", "voice_neutral", "voice_dominant_emotion_conf"}
_TEXT_EMOTION_COLS = {
    f"sent_{e}"
    for e in (
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    )
}
_EKMAN_COLS_BATCH = {f"ekman_{e}" for e in ("joy", "excitement", "sadness", "neutral")} | {"ekman_intensity"}
_RAW_EMOTION_COLS_BATCH = _FACE_EMOTION_COLS | _VOICE_EMOTION_COLS | _TEXT_EMOTION_COLS

_BATCH_GROUPS = [
    (
        "visual",
        {
            "brightness",
            "sharpness",
            "speaker_prob",
            "face_screen_ratio",
            "faces_total_ratio",
            "text_prob",
            "motion_speed",
            "edit_pace",
            "scene_novelty",
            "screencast_prob",
            "overlay_prob",
            "bumper_score",
            "visual_entropy",
            "visual_complexity",
            "visual_complexity_gradient",
            "visual_complexity_acceleration",
            "face_area_ratio",
        },
        "~4 GB",
    ),
    ("zoom", {"flow_mag_med", "radial_med", "radial_ratio"}, "~500 MB"),
    ("sound", {"rms", "zcr", "centroid", "rolloff"}, "0 (CPU)"),
    ("demucs", {"music_rms", "music_zcr", "music_centroid", "music_rolloff", "vocal_rms", "vocal_zcr", "vocal_centroid", "vocal_rolloff"}, "~2-4 GB"),
    ("audio_basic", {"beat_sync", "beat_sync_ratio", "loudness_change", "loudness_variance", "pitch_mean", "pitch_std", "voiced_frac", "speech_rate_cv", "pause_rate"}, "0 (CPU)"),
    (
        "whisper_text",
        {
            "wps",
            "viewer_address",
            "crutch_cnt",
            "is_ad",
            "ad_segment_length",
            "title_transcript_gap",
            "title_delivery_30s",
            "title_claim_intensity",
            "syntactic_depth",
            "lexical_diversity",
            "avg_word_length",
            "speech_complexity",
            "has_person_mention",
            "has_org_mention",
        },
        "~3 GB",
    ),
    ("audio_extra", {"spectral_flux", "laughter_prob", "sfx_energy", "speech_predictability", "speech_lm_surprisal", "speech_lm_surprisal_vel"}, "~1 GB"),
    (
        "text_sentiment",
        {
            f"sent_{e}"
            for e in (
                "admiration",
                "amusement",
                "anger",
                "annoyance",
                "approval",
                "caring",
                "confusion",
                "curiosity",
                "desire",
                "disappointment",
                "disapproval",
                "disgust",
                "embarrassment",
                "excitement",
                "fear",
                "gratitude",
                "grief",
                "joy",
                "love",
                "nervousness",
                "optimism",
                "pride",
                "realization",
                "relief",
                "remorse",
                "sadness",
                "surprise",
                "neutral",
            )
        },
        "~1.5 GB",
    ),
    (
        "text_embedding",
        {
            "semantic_novelty",
            "topic_shift",
            "hook_similarity",
            "global_topic_dist",
            "semantic_momentum",
            "segment_self_similarity",
            "hook_score",
            "hook_has_address",
            "is_question",
            "chapter_id",
            "n_chapters",
            "topic_change_rate",
        },
        "~500 MB",
    ),
    ("text_zeroshot", {"curiosity_gap", "storytelling", "viewer_engagement", "has_example", "is_intro", "is_outro", "topic_sharpness_0_100"}, "~1.5 GB"),
    ("object_density", {"object_count", "unique_classes"}, "~100 MB"),
    ("video_intelligence", {"content_rhythm", "visual_audio_sync", "narrative_momentum", "engagement_surprise"}, "~500 MB"),
    (
        "comment_social",
        {
            "desc_chapter_start",
            "desc_chapter_boundary_dist",
            "timecode_like_weighted_30s",
            "comment_question_rate_30s",
            "comment_density_30s",
            "comment_positive_rate_30s",
            "comment_reply_depth_30s",
            "comment_aggression_rate_30s",
            "author_reply_rate_video",
            "avg_comment_length_video",
            "complex_words_ratio_video",
        },
        "0 (CPU)",
    ),
    ("speech_intelligibility", {"speech_intelligibility", "speech_mumble_index"}, "0 (CPU)"),
]


def _discover_videos(data_dir: str) -> list[dict]:
    """Find all videos in data_dir that have both video.mp4 and retention.csv."""
    videos = []
    for name in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, name)
        if not os.path.isdir(d):
            continue
        video = os.path.join(d, "video.mp4")
        ret = os.path.join(d, "retention.csv")
        if os.path.isfile(video) and os.path.isfile(ret):
            videos.append({"vid": name, "video_path": video, "retention_path": ret, "output_path": os.path.join("output", f"{name}_features.csv")})
    return videos


def _batch_load_video(v: dict) -> tuple[pd.DataFrame, set]:
    """Load checkpoint or existing output for a video, return (accumulated, existing_set)."""
    acc, feats = _load_checkpoint(v["output_path"])
    if acc.empty and os.path.exists(v["output_path"]):
        acc = pd.read_csv(v["output_path"], index_col=0)
        feats = acc.columns.tolist()
    if acc.empty:
        retention = get_retention(v["video_path"], v["retention_path"])
        if retention.index.name is None:
            retention.index.name = "time"
        acc = retention.copy()
        _save_checkpoint(acc, v["output_path"])
    return acc, set(feats if feats else acc.columns.tolist())


def _batch_map_to_retention(retention_index, features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(index=retention_index)
    mapped = pd.DataFrame(index=retention_index)
    for col in features.columns:
        fp = features[col].astype("float64").values
        mapped[col] = np.interp(np.linspace(0, len(features) - 1, len(retention_index)), np.arange(len(features)), fp)
    return mapped


def _batch_add_cols(acc: pd.DataFrame, mapped: pd.DataFrame, output_path: str, only: set | None = None) -> pd.DataFrame:
    new_cols = [c for c in mapped.columns if c not in acc.columns]
    overwrite_cols = [c for c in mapped.columns if c in acc.columns and c in only] if only else []
    if new_cols:
        acc = pd.concat([acc, mapped[new_cols]], axis=1)
    if overwrite_cols:
        acc[overwrite_cols] = mapped[overwrite_cols]
    if new_cols or overwrite_cols:
        _save_checkpoint(acc, output_path)
    return acc


def _gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _batch_run_group_for_video(
    group_name: str, v: dict, acc: pd.DataFrame, existing: set, config: Config, preloaded_zeroshot=None, only: set | None = None
) -> tuple[pd.DataFrame, set, set]:
    """Run a single feature group for a single video.

    Returns (updated_acc, updated_existing, failed_cols).
    """
    video_path = v["video_path"]
    output_path = v["output_path"]
    retention_idx = acc.index
    existing_list = list(existing)
    failed = set()

    def _map(feat_df):
        return _batch_map_to_retention(retention_idx, feat_df)

    def _add(mapped):
        nonlocal acc, existing
        acc = _batch_add_cols(acc, mapped, output_path, only=only)
        existing = set(acc.columns.tolist())

    def _run_step(name, cols, fn):
        nonlocal acc, existing
        missing = cols - existing
        want_overwrite = only and (cols & only)
        if not missing and not want_overwrite:
            return
        t0 = time.time()
        logger.info("  [%s] Extracting %s", v["vid"], name)
        try:
            result = fn()
            if result is not None and not result.empty:
                _add(_map(result))
            logger.info("  [%s] %s done in %.1fs", v["vid"], name, time.time() - t0)
        except Exception as e:
            logger.error("  [%s] %s FAILED (%.1fs): %s", v["vid"], name, time.time() - t0, e)
            failed.update(cols)

    if group_name == "visual":
        core = {"brightness", "sharpness", "speaker_prob"}
        group_cols = {
            "brightness",
            "sharpness",
            "speaker_prob",
            "face_screen_ratio",
            "faces_total_ratio",
            "text_prob",
            "motion_speed",
            "edit_pace",
            "scene_novelty",
            "screencast_prob",
            "overlay_prob",
            "bumper_score",
            "visual_entropy",
            "visual_complexity",
            "visual_complexity_gradient",
            "visual_complexity_acceleration",
            "color_temperature",
            "color_saturation",
            "face_area_ratio",
            "short_insert",
            "short_insert_rate",
        }
        if not (core - existing) and not (only and (group_cols & only)):
            return acc, existing, failed
        all_passes = [
            FrameQualityFeature(config),
            ColorFeature(config),
            SpeakerProbabilityFeature(config),
            FaceScreenRatioFeature(config),
            TextProbFeature(config),
            MotionSpeedFeature(config),
            EditPaceFeature(config),
            ShortInsertFeature(config),
            SceneNoveltyFeature(config),
            ScreencastFeature(config),
            OverlayFeature(config),
            BumperFeature(config, data_dir=os.path.dirname(os.path.dirname(video_path))),
        ]
        if only:
            all_passes = [p for p in all_passes if p.produces_keys() & only]
        try:
            t0 = time.time()
            _prepare_shot_boundary_embeddings(video_path, config)
            speaker_features = run_feature_pipeline(video_path, config, passes=all_passes, existing_features=existing)
            speaker_features = speaker_features.drop(columns=["frame_face_boxes", "frame_keypoints", "frame_idx"], errors="ignore")
            mapped = _map(speaker_features)
            for col in mapped.columns:
                _add(mapped[[col]])
            logger.info("  [%s] visual done in %.1fs", v["vid"], time.time() - t0)
        except Exception as e:
            logger.error("  [%s] visual FAILED: %s", v["vid"], e)
            failed.update({"brightness", "sharpness", "speaker_prob"})
        finally:
            del all_passes
            _gpu_cleanup()

    elif group_name == "zoom":

        def _zoom():
            zf = ZoomFeatureExtractor(video_path=video_path, config=config).run()
            zf = zf.drop(columns=["frame_idx"], errors="ignore")
            return mask_flow_at_cuts(zf, video_path, config)

        _run_step("zoom", {"flow_mag_med", "radial_med", "radial_ratio"}, _zoom)

    elif group_name == "sound":
        _run_step("sound", {"rms", "zcr", "centroid", "rolloff"}, lambda: sound_features_pipeline(audio_file_path=video_path, existing_features=existing_list))

    elif group_name == "demucs":
        cols = {"music_rms", "music_zcr", "music_centroid", "music_rolloff", "vocal_rms", "vocal_zcr", "vocal_centroid", "vocal_rolloff"}
        if cols - existing:
            try:
                t0 = time.time()
                music_features, vocal_features = get_vocal_music_features(audio_path=video_path, config=config, existing_features=existing_list)
                _add(_map(music_features))
                _add(_map(vocal_features))
                logger.info("  [%s] demucs done in %.1fs", v["vid"], time.time() - t0)
            except Exception as e:
                logger.error("  [%s] demucs FAILED: %s", v["vid"], e)
                failed.update(cols)

        # Derived features from demucs stems
        need_sms = {"speech_ratio", "silence_stretch", "music_only"} - set(acc.columns)
        if need_sms and all(c in acc.columns for c in ["vocal_rms", "music_rms"]):
            sms_df = extract_speech_music_silence(acc["vocal_rms"].values, acc["music_rms"].values)
            sms_df.index = retention_idx
            _add(sms_df)
        need_bgm = {"has_background_music", "music_changed"} - set(acc.columns)
        if need_bgm and all(c in acc.columns for c in ["music_rms", "music_centroid", "music_rolloff"]):
            bgm_df = extract_background_music_features(acc["music_rms"].values, acc["music_centroid"].values, acc["music_rolloff"].values)
            bgm_df.index = retention_idx
            _add(bgm_df)

    elif group_name == "audio_basic":
        _run_step(
            "prosody",
            {"pitch_mean", "pitch_std", "voiced_frac", "speech_rate_cv", "pause_rate"},
            lambda: extract_prosody(video_path=video_path, config=config, existing_features=existing_list),
        )
        _run_step("beat-sync", {"beat_sync", "beat_sync_ratio"}, lambda: extract_beat_sync(video_path=video_path, config=config, existing_features=existing_list))
        _run_step(
            "loudness dynamics", {"loudness_change", "loudness_variance"}, lambda: extract_loudness_dynamics(video_path=video_path, config=config, existing_features=existing_list)
        )

    elif group_name == "audio_extra":
        _run_step("spectral flux", {"spectral_flux"}, lambda: extract_spectral_flux(video_path=video_path, config=config, existing_features=existing_list))
        _run_step("laughter", {"laughter_prob"}, lambda: extract_laughter(video_path=video_path, config=config, existing_features=existing_list))
        _run_step("sfx energy", {"sfx_energy"}, lambda: extract_sfx_energy(video_path=video_path, config=config, existing_features=existing_list))
        _run_step("speech predictability", {"speech_predictability"}, lambda: extract_speech_predictability(video_path=video_path, config=config, existing_features=existing_list))
        _run_step(
            "speech lm surprisal",
            {"speech_lm_surprisal", "speech_lm_surprisal_vel"},
            lambda: extract_speech_lm_surprisal(video_path=video_path, config=config, existing_features=existing_list),
        )

    elif group_name == "whisper_text":
        _run_step("wps", {"wps"}, lambda: extract_wps(video_path=video_path, config=config, existing_features=existing_list))
        _run_step("viewer address", {"viewer_address"}, lambda: extract_viewer_address(video_path=video_path, config=config, existing_features=existing_list))
        _run_step("speech fillers", {"crutch_cnt"}, lambda: extract_speech_fillers(video_path=video_path, config=config, existing_features=existing_list))
        _run_step("ad segments", {"is_ad", "ad_segment_length"}, lambda: extract_ad_segments(video_path=video_path, config=config, existing_features=existing_list))
        _run_step(
            "clickbait gap",
            {"title_transcript_gap", "title_delivery_30s", "title_claim_intensity"},
            lambda: extract_clickbait_gap(video_path=video_path, config=config, existing_features=existing_list),
        )
        _run_step(
            "text complexity",
            {"syntactic_depth", "lexical_diversity", "avg_word_length", "speech_complexity"},
            lambda: extract_text_complexity(video_path=video_path, config=config, existing_features=existing_list),
        )
        _run_step(
            "cultural refs", {"has_person_mention", "has_org_mention"}, lambda: extract_cultural_references(video_path=video_path, config=config, existing_features=existing_list)
        )
        clear_transcript_cache()

    elif group_name == "text_sentiment":
        _run_step("sentiment", _TEXT_EMOTION_COLS, lambda: extract_text_sentiment(video_path=video_path, config=config, existing_features=existing_list))
        clear_transcript_cache()

    elif group_name == "text_embedding":
        _run_step(
            "semantic embeddings",
            {"semantic_novelty", "topic_shift", "hook_similarity", "global_topic_dist", "semantic_momentum", "segment_self_similarity"},
            lambda: extract_semantic_embeddings(video_path=video_path, config=config, existing_features=existing_list),
        )
        _run_step(
            "hook score", {"hook_score", "hook_has_address", "is_question"}, lambda: extract_hook_score(video_path=video_path, config=config, existing_features=existing_list)
        )
        _run_step("chapters", {"chapter_id", "n_chapters", "topic_change_rate"}, lambda: extract_chapters(video_path=video_path, config=config, existing_features=existing_list))
        clear_transcript_cache()

    elif group_name == "text_zeroshot":
        _run_step("curiosity gap", {"curiosity_gap"}, lambda: extract_curiosity_gap(video_path=video_path, config=config, existing_features=existing_list))
        _run_step("topic sharpness (Qwen)", {"topic_sharpness_0_100"}, lambda: extract_topic_sharpness(video_path=video_path, config=config, existing_features=existing_list))
        _run_step("storytelling", {"storytelling"}, lambda: extract_storytelling(video_path=video_path, config=config, existing_features=existing_list))
        _run_step("viewer engagement", {"viewer_engagement"}, lambda: extract_viewer_engagement(video_path=video_path, config=config, existing_features=existing_list))
        _run_step("examples", {"has_example"}, lambda: extract_examples(video_path=video_path, config=config, existing_features=existing_list))
        _run_step("sections", {"is_intro", "is_outro"}, lambda: extract_sections(video_path=video_path, config=config, existing_features=existing_list))
        _run_step(
            "information density",
            {"information_density", "cumulative_info"},
            lambda: extract_information_density(video_path=video_path, config=config, existing_features=existing_list),
        )
        clear_transcript_cache()

    elif group_name == "object_density":
        _run_step("object density", {"object_count", "unique_classes"}, lambda: extract_object_density(video_path=video_path, config=config, existing_features=existing_list))

    elif group_name == "video_intelligence":
        _run_step(
            "video intelligence",
            {"content_rhythm", "visual_audio_sync", "narrative_momentum", "engagement_surprise"},
            lambda: extract_video_intelligence(video_path=video_path, config=config, existing_features=existing_list),
        )

    elif group_name == "comment_social":
        _cols = {
            "desc_chapter_start",
            "desc_chapter_boundary_dist",
            "timecode_like_weighted_30s",
            "comment_question_rate_30s",
            "comment_density_30s",
            "comment_positive_rate_30s",
            "comment_reply_depth_30s",
            "comment_aggression_rate_30s",
            "author_reply_rate_video",
            "avg_comment_length_video",
            "complex_words_ratio_video",
        }
        _run_step("comment features", _cols, lambda: extract_comment_features(video_path=video_path, config=config, existing_features=existing_list))

    elif group_name == "speech_intelligibility":
        _run_step(
            "speech intelligibility",
            {"speech_intelligibility", "speech_mumble_index"},
            lambda: extract_speech_intelligibility(video_path=video_path, config=config, existing_features=existing_list),
        )

    return acc, existing, failed


def _batch_finalize(acc: pd.DataFrame, output_path: str, *, skip_emotion_features: bool = False) -> pd.DataFrame:
    """Compute derived features and write final CSV."""
    if "edit_pace" in acc.columns and "screencast_prob" in acc.columns:
        acc["edit_pace_x_screencast"] = acc["edit_pace"] * acc["screencast_prob"]
    if "is_ad" in acc.columns and "viewer_address" in acc.columns:
        acc["is_ad_x_viewer_address"] = acc["is_ad"] * acc["viewer_address"]

    _QD_WINDOW_SEC = 30
    if "is_question" in acc.columns and "question_density" not in acc.columns:
        iq = acc["is_question"].values.astype(np.float64)
        q_starts = (np.diff(iq, prepend=0) > 0).astype(np.float64)
        rolling_count = np.convolve(q_starts, np.ones(_QD_WINDOW_SEC), mode="same")
        acc["question_density"] = rolling_count / (_QD_WINDOW_SEC / 60.0)

    if not skip_emotion_features:
        if not _EKMAN_COLS_BATCH.issubset(set(acc.columns)):
            has_any = bool(_RAW_EMOTION_COLS_BATCH & set(acc.columns))
            if has_any:
                ekman_df = compute_ekman_fusion(acc)
                for col in ekman_df.columns:
                    acc[col] = ekman_df[col]

        drop_raw = sorted(_RAW_EMOTION_COLS_BATCH & set(acc.columns))
        if drop_raw and _EKMAN_COLS_BATCH.issubset(set(acc.columns)):
            acc = acc.drop(columns=drop_raw, errors="ignore")

    _DEPRECATED_EKMAN = {"ekman_anger", "ekman_disgust", "ekman_fear", "ekman_surprise"}
    _REMOVED_EXPENSIVE = {
        "aesthetic_score",
        "depth_variance",
        "depth_mean",
        "audio_novelty",
        "audio_topic_shift",
        "audio_hook_similarity",
        "audio_global_dist",
        "audio_momentum",
        "audio_self_similarity",
        "visual_topic_shift",
        "visual_hook_similarity",
        "visual_global_dist",
        "visual_momentum",
        "visual_self_similarity",
        "voice_angry",
        "voice_happy",
        "voice_sad",
        "voice_neutral",
        "voice_dominant_emotion_conf",
    }
    _REDUNDANT_DUPLICATE = {"embedding_drift", "embedding_drift_smoothed", "loudness_zscore", "loudness_spike", "loudness_drop"}
    _REDUNDANT_COLS = {"visual_novelty", "global_topic_dist"} | _DEPRECATED_EKMAN | _REMOVED_EXPENSIVE | _REDUNDANT_DUPLICATE
    drop_redundant = sorted(_REDUNDANT_COLS & set(acc.columns))
    if drop_redundant:
        acc = acc.drop(columns=drop_redundant, errors="ignore")

    if "is_intro" in acc.columns and "retention" in acc.columns and acc["retention"].std() > 1e-3:
        try:
            from .analysis.segmented_trainer import fit_hill_curve

            time_sec = np.arange(len(acc), dtype=np.float64)
            _, params = fit_hill_curve(time_sec, acc["retention"].values)
            inflection_sec = params[2]
            cap = int(min(inflection_sec * 2.0, len(acc)))
            intro_vals = acc["is_intro"].values.copy()
            if cap < len(intro_vals):
                intro_vals[cap:] = 0.0
                acc["is_intro"] = intro_vals
        except Exception:
            pass

    _append_local_spike_and_visual_dynamics(acc)

    checkpoint = _checkpoint_path(output_path)
    if os.path.exists(checkpoint):
        os.remove(checkpoint)

    acc.to_csv(output_path, index=True)
    return acc


def aggregate_batch(
    data_dir: str, output_dir: str, config: Config, only: set | None = None, skip_comment_features: bool = False, skip_emotion_features: bool = False
) -> dict[str, pd.DataFrame]:
    """Batch-mode aggregation: load each model once, process all videos, unload.

    Instead of loading/unloading models per video, iterates over model groups
    and processes all videos for each group before moving on. This avoids
    redundant model loading when processing many videos.

    Returns dict of {video_id: final_dataframe}.
    """
    os.makedirs(output_dir, exist_ok=True)
    t_total = time.time()

    videos = _discover_videos(data_dir)
    if not videos:
        logger.error("No videos found in %s", data_dir)
        return {}

    for v in videos:
        v["output_path"] = os.path.join(output_dir, f"{v['vid']}_features.csv")

    logger.info("=== BATCH MODE: %d videos, %d feature groups ===", len(videos), len(_BATCH_GROUPS))

    all_failed: dict[str, set] = {v["vid"]: set() for v in videos}
    video_data: dict[str, tuple[pd.DataFrame, set]] = {}

    for v in videos:
        acc, existing = _batch_load_video(v)
        video_data[v["vid"]] = (acc, existing)
        logger.info("  loaded %s (%d existing cols)", v["vid"], len(existing))

    for group_name, group_cols, vram_est in _BATCH_GROUPS:
        if skip_comment_features and group_name == "comment_social":
            logger.info("[group comment_social] skipped (--skip-comment-features)")
            continue
        if skip_emotion_features and group_name == "text_sentiment":
            logger.info("[group text_sentiment] skipped (--skip-emotion-features)")
            continue
        # When --only is set, run only groups that produce at least one requested column
        # (avoids loading e.g. text_sentiment when the user only wants comments / speech).
        if only and not (group_cols & only):
            logger.info("[group %s] not in --only, skipping", group_name)
            continue

        any_need = False
        for v in videos:
            _, existing = video_data[v["vid"]]
            missing = group_cols - existing
            want_overwrite = only and (group_cols & only)
            if missing or want_overwrite:
                any_need = True
                break

        if not any_need:
            logger.info("[group %s] all videos done, skipping", group_name)
            continue

        logger.info("[group %s] VRAM %s — processing %d videos ...", group_name, vram_est, len(videos))
        t_group = time.time()

        for v in videos:
            acc, existing = video_data[v["vid"]]
            missing = group_cols - existing
            want_overwrite = only and (group_cols & only)
            if not missing and not want_overwrite:
                logger.info("  [%s] %s — already done, skip", v["vid"], group_name)
                continue

            acc, existing, failed = _batch_run_group_for_video(group_name, v, acc, existing, config, only=only)
            video_data[v["vid"]] = (acc, existing)
            all_failed[v["vid"]] |= failed

        _gpu_cleanup()
        logger.info("[group %s] done in %.1fs", group_name, time.time() - t_group)

    results = {}
    for v in videos:
        acc, _ = video_data[v["vid"]]
        acc = _batch_finalize(acc, v["output_path"], skip_emotion_features=skip_emotion_features)
        _save_failed_features(v["output_path"], all_failed[v["vid"]])
        results[v["vid"]] = acc
        logger.info("Finalized %s (%d cols)", v["vid"], len(acc.columns))

    logger.info("=== BATCH COMPLETE: %d videos in %.1fs (%.1f min) ===", len(videos), time.time() - t_total, (time.time() - t_total) / 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--retention_path", type=str, required=False)
    parser.add_argument("-v", "--video_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("-c", "--config_path", type=str, required=False)
    parser.add_argument("-d", "--data_dir", type=str, default="data")

    args = parser.parse_args()

    config = Config(config_path=args.config_path)

    aggregated_df = aggregate(
        video_path=args.video_path, audio_path=args.video_path, output_path=args.output_path, config=config, html_path=args.retention_path, data_dir=args.data_dir
    )

    aggregated_df.to_csv(args.output_path, index=True)
