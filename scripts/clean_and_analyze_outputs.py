from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/home/kolya/ilya/seenx-ml")
OUTPUT_DIR = ROOT / "output"
REPORT_CSV = OUTPUT_DIR / "feature_quality_report.csv"
REPORT_MD = OUTPUT_DIR / "feature_quality_summary.md"
CSV_GLOB = "*_features.csv"

KEEP_COLS = {
    "time",
    "retention",
    "brightness",
    "sharpness",
    "speaker_prob",
    "face_screen_ratio",
    "faces_total_ratio",
    "text_prob",
    "motion_speed",
    "ekman_joy",
    "ekman_excitement",
    "ekman_sadness",
    "ekman_neutral",
    "ekman_intensity",
    "edit_pace",
    "motion_speed_chg_5s",
    "motion_speed_abs_step_mean_5s",
    "edit_pace_chg_5s",
    "edit_pace_abs_step_mean_5s",
    "scene_novelty_chg_5s",
    "scene_novelty_abs_step_mean_5s",
    "motion_spike",
    "pitch_spike",
    "brightness_spike",
    "entropy_spike",
    "scene_novelty",
    "screencast_prob",
    "bumper_score",
    "visual_entropy",
    "rms",
    "zcr",
    "centroid",
    "rolloff",
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
    "flow_mag_med",
    "radial_med",
    "radial_ratio",
    "wps",
    "viewer_address",
    "crutch_cnt",
    "pitch_mean",
    "pitch_std",
    "voiced_frac",
    "speech_rate_cv",
    "pause_rate",
    "beat_sync",
    "beat_sync_ratio",
    "is_ad",
    "ad_segment_length",
    "syntactic_depth",
    "lexical_diversity",
    "avg_word_length",
    "speech_complexity",
    "has_person_mention",
    "has_org_mention",
    "sent_admiration",
    "sent_amusement",
    "sent_anger",
    "sent_annoyance",
    "sent_approval",
    "sent_caring",
    "sent_confusion",
    "sent_curiosity",
    "sent_desire",
    "sent_disappointment",
    "sent_disapproval",
    "sent_disgust",
    "sent_embarrassment",
    "sent_excitement",
    "sent_fear",
    "sent_gratitude",
    "sent_grief",
    "sent_joy",
    "sent_love",
    "sent_nervousness",
    "sent_neutral",
    "sent_optimism",
    "sent_pride",
    "sent_realization",
    "sent_relief",
    "sent_remorse",
    "sent_sadness",
    "sent_surprise",
    "hook_score",
    "hook_has_address",
    "is_question",
    "loudness_change",
    "loudness_variance",
    "semantic_novelty",
    "topic_shift",
    "hook_similarity",
    "semantic_momentum",
    "segment_self_similarity",
    "voice_angry",
    "voice_happy",
    "voice_sad",
    "voice_neutral",
    "voice_dominant_emotion_conf",
    "visual_topic_shift",
    "visual_hook_similarity",
    "visual_momentum",
    "visual_self_similarity",
    "object_count",
    "unique_classes",
    "audio_novelty",
    "audio_topic_shift",
    "audio_hook_similarity",
    "audio_momentum",
    "audio_self_similarity",
    "aesthetic_score",
    "depth_variance",
    "depth_mean",
    "edit_pace_x_screencast",
    "is_ad_x_viewer_address",
}

LEGACY_DROP_EXACT = {
    "cinematic",
    "frame",
    "hook_has_question",
    "question_score",
    "question_density",
    "filler_density",
    "cultural_ref_cnt",
    "hook_score_x_time_pct",
    "time_pct",
    "sentiment_polarity",
    "sentiment_intensity",
    "pos_cnt",
    "neg_cnt",
    "total_emotional_cnt",
}

DUPLICATE_SUFFIX_RE = re.compile(r"\.\d+$")
EPS = 1e-12


@dataclass
class FeatureStats:
    feature: str
    rows: int
    videos_present: int
    std: float
    min_val: float
    max_val: float
    range_val: float
    zero_frac: float
    one_frac: float
    const_video_frac: float
    low_std_video_frac: float
    unique_values: int
    is_binary_like: bool
    flags: str


def iter_output_files() -> list[Path]:
    return sorted(OUTPUT_DIR.glob(CSV_GLOB))


def should_drop_column(name: str) -> bool:
    if DUPLICATE_SUFFIX_RE.search(name):
        return True
    if name in LEGACY_DROP_EXACT:
        return True
    return name not in KEEP_COLS


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_outputs(files: list[Path]) -> dict[str, list[str]]:
    removed_by_file: dict[str, list[str]] = {}
    for path in files:
        fieldnames, rows = read_csv_rows(path)
        keep = [name for name in fieldnames if not should_drop_column(name)]
        removed = [name for name in fieldnames if should_drop_column(name)]
        if removed:
            cleaned_rows = [{k: row.get(k, "") for k in keep} for row in rows]
            write_csv_rows(path, keep, cleaned_rows)
        removed_by_file[path.name] = removed
    return removed_by_file


def as_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def approx_unique_count(values: list[float]) -> int:
    return len({round(v, 8) for v in values})


def collect_feature_stats(files: list[Path]) -> list[FeatureStats]:
    per_feature_values: dict[str, list[float]] = {}
    per_feature_video_values: dict[str, list[list[float]]] = {}

    for path in files:
        fieldnames, rows = read_csv_rows(path)
        numeric_columns = [c for c in fieldnames if c != "time"]
        video_values: dict[str, list[float]] = {c: [] for c in numeric_columns}
        for row in rows:
            for col in numeric_columns:
                value = as_float(row.get(col, ""))
                if value is not None:
                    video_values[col].append(value)

        for col, values in video_values.items():
            if not values:
                continue
            per_feature_values.setdefault(col, []).extend(values)
            per_feature_video_values.setdefault(col, []).append(values)

    total_videos = len(files)
    stats: list[FeatureStats] = []
    for feature, values in sorted(per_feature_values.items()):
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        sigma = std(values)
        unique_values = approx_unique_count(values)
        zero_frac = sum(abs(v) <= EPS for v in values) / len(values)
        one_frac = sum(abs(v - 1.0) <= EPS for v in values) / len(values)

        by_video = per_feature_video_values[feature]
        const_video_frac = sum(std(v) <= 1e-9 for v in by_video) / total_videos
        low_std_video_frac = sum(std(v) <= 1e-4 for v in by_video) / total_videos
        is_binary_like = all(abs(v) <= EPS or abs(v - 1.0) <= EPS for v in values)

        flags: list[str] = []
        if sigma < 1e-3 and range_val < 0.02:
            flags.append("near_constant_global")
        if const_video_frac >= 0.30:
            flags.append("constant_many_videos")
        if low_std_video_frac >= 0.50:
            flags.append("low_std_many_videos")
        if zero_frac >= 0.95:
            flags.append("mostly_zero")
        if is_binary_like and one_frac <= 0.01:
            flags.append("rare_positive_binary")
        if unique_values <= 3 and not is_binary_like:
            flags.append("very_low_cardinality")

        stats.append(
            FeatureStats(
                feature=feature,
                rows=len(values),
                videos_present=len(by_video),
                std=sigma,
                min_val=min_val,
                max_val=max_val,
                range_val=range_val,
                zero_frac=zero_frac,
                one_frac=one_frac,
                const_video_frac=const_video_frac,
                low_std_video_frac=low_std_video_frac,
                unique_values=unique_values,
                is_binary_like=is_binary_like,
                flags="|".join(flags),
            )
        )

    stats.sort(key=lambda s: (0 if s.flags else 1, s.std, s.range_val, s.feature))
    return stats


def write_report_csv(stats: list[FeatureStats]) -> None:
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "feature",
                "rows",
                "videos_present",
                "std",
                "min",
                "max",
                "range",
                "zero_frac",
                "one_frac",
                "const_video_frac",
                "low_std_video_frac",
                "unique_values",
                "is_binary_like",
                "flags",
            ]
        )
        for s in stats:
            writer.writerow(
                [
                    s.feature,
                    s.rows,
                    s.videos_present,
                    f"{s.std:.12g}",
                    f"{s.min_val:.12g}",
                    f"{s.max_val:.12g}",
                    f"{s.range_val:.12g}",
                    f"{s.zero_frac:.6f}",
                    f"{s.one_frac:.6f}",
                    f"{s.const_video_frac:.6f}",
                    f"{s.low_std_video_frac:.6f}",
                    s.unique_values,
                    int(s.is_binary_like),
                    s.flags,
                ]
            )


def pick_group(stats: list[FeatureStats], predicate, limit: int = 20) -> list[FeatureStats]:
    return [s for s in stats if predicate(s)][:limit]


def write_report_md(files: list[Path], removed_by_file: dict[str, list[str]], stats: list[FeatureStats]) -> None:
    removed_union = sorted({col for cols in removed_by_file.values() for col in cols})
    near_constant = pick_group(stats, lambda s: "near_constant_global" in s.flags)
    const_many = pick_group(stats, lambda s: "constant_many_videos" in s.flags and s.feature not in {x.feature for x in near_constant})
    mostly_zero = pick_group(stats, lambda s: "mostly_zero" in s.flags and s.feature not in {x.feature for x in near_constant})
    rare_binary = pick_group(stats, lambda s: "rare_positive_binary" in s.flags)

    lines: list[str] = []
    lines.append("# Output Feature Cleanup And Quality Report")
    lines.append("")
    lines.append(f"- Files scanned: {len(files)}")
    lines.append(f"- Files cleaned: {sum(bool(v) for v in removed_by_file.values())}")
    lines.append(f"- Removed legacy columns: {', '.join(removed_union) if removed_union else 'none'}")
    lines.append(f"- Full numeric report: `{REPORT_CSV.name}`")
    lines.append("")

    def add_section(title: str, rows: list[FeatureStats]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("- none")
            lines.append("")
            return
        for s in rows:
            lines.append(f"- `{s.feature}`: std={s.std:.4g}, range={s.range_val:.4g}, zero_frac={s.zero_frac:.1%}, const_video_frac={s.const_video_frac:.1%}, flags={s.flags}")
        lines.append("")

    add_section("Near-Constant Global Features", near_constant)
    add_section("Constant In Many Videos", const_many)
    add_section("Mostly Zero Features", mostly_zero)
    add_section("Rare Positive Binary Features", rare_binary)

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    files = iter_output_files()
    if not files:
        raise SystemExit("No output CSV files found.")

    removed_by_file = clean_outputs(files)
    stats = collect_feature_stats(files)
    write_report_csv(stats)
    write_report_md(files, removed_by_file, stats)

    removed_union = sorted({col for cols in removed_by_file.values() for col in cols})
    print(f"cleaned_files={sum(bool(v) for v in removed_by_file.values())}/{len(files)}")
    print(f"removed_columns={','.join(removed_union)}")
    print(f"report_csv={REPORT_CSV}")
    print(f"report_md={REPORT_MD}")


if __name__ == "__main__":
    main()
