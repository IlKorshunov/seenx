"""Actionable «one-screen» report: N prioritized items with timecodes.

For a single video produces a short markdown report:
  1. Feature-to-retention correlation table (Pearson/Spearman across all channel videos).
  2. Top-N worst zones with timecodes + anomalous feature deltas.
  3. Qwen-generated short recommendations per zone.

All major knobs are CLI arguments — nothing is hard-coded.

CLI:
  python -m src.analysis.actionable_report \\
      --target VIDEO_ID \\
      --output-dir output \\
      --data-dir train/drive_snapshot_90 \\
      --top-n 3 \\
      --retention-source auto \\
      --corr-method spearman
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from transformers import AutoModelForCausalLM, AutoTokenizer


_LLM_MODEL_ID = "Qwen/Qwen3-4B"

_INTERPRETABLE_FEATURES = [
    "edit_pace",
    "wps",
    "speech_intelligibility",
    "speech_mumble_index",
    "friction_total",
    "friction_jargon",
    "friction_abstract",
    "friction_repetition",
    "friction_digression",
    "curiosity_gap",
    "storytelling",
    "viewer_engagement",
    "information_density",
    "semantic_novelty",
    "speech_lm_surprisal",
    "motion_speed",
    "brightness",
    "speaker_prob",
    "visual_entropy",
    "hook_score",
    "scene_novelty",
    "spectral_flux",
    "laughter_prob",
    "speech_predictability",
    "speech_complexity",
    "beat_sync",
    "loudness_change",
    "pitch_mean",
    "pitch_std",
    "is_ad",
    "screencast_prob",
]


def _load_features(output_dir: Path, vid: str) -> pd.DataFrame | None:
    for path, kw in [(output_dir / f"{vid}_features.csv", {"index_col": 0}), (output_dir / vid / "features_readable.csv", {})]:
        if path.exists():
            return pd.read_csv(path, **kw)
    return None


def _load_retention_studio(data_dir: Path, vid: str) -> np.ndarray | None:
    for subdir in ("", "transcripts/"):
        csv_path = data_dir / vid / f"{subdir}retention_source.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if {"time_ratio", "audience_watch_ratio"} <= set(df.columns):
                return np.interp(np.linspace(0, 1, 100), df["time_ratio"].values, df["audience_watch_ratio"].values * 100.0)
    for jname in ("retention.json", "retention_parsed.json"):
        for sub in ("", "transcripts/"):
            jp = data_dir / vid / f"{sub}{jname}"
            if jp.exists():
                raw = json.loads(jp.read_text(encoding="utf-8"))
                if isinstance(raw, list) and raw:
                    vals = np.array([float(r.get("audienceWatchRatio", r) if isinstance(r, dict) else r) for r in raw])
                    if vals.max() <= 1.5:
                        vals *= 100.0
                    return np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(vals)), vals)
    return None


def _retention_from_features(features: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in features.columns:
        return None
    ser = pd.to_numeric(features[column], errors="coerce")
    if ser.isna().all():
        return None
    vals = ser.ffill().bfill().fillna(0.0).values.astype(np.float64)
    if np.nanmax(vals) <= 1.5:
        vals *= 100.0
    return vals


def _resolve_retention(
    vid: str, data_dir: Path, features: pd.DataFrame | None, source: Literal["auto", "studio", "features"], retention_column: str
) -> tuple[np.ndarray | None, str]:
    studio = _load_retention_studio(data_dir, vid)
    from_csv = _retention_from_features(features, retention_column) if features is not None else None
    if source == "studio":
        return (studio, "studio") if studio is not None else (None, "none")
    if source == "features":
        return (from_csv, "features") if from_csv is not None else (None, "none")
    if studio is not None:
        return studio, "studio"
    if from_csv is not None:
        return from_csv, "features"
    return None, "none"


def _load_title(vid: str) -> str:
    root = Path(__file__).resolve().parents[2]
    return next((json.loads(p.read_text(encoding="utf-8")).get("video_title", vid) for p in (root / "get_data" / "comments").rglob(f"{vid}/comments.json")), vid)


def _discover_vids(output_dir: Path) -> list[str]:
    return sorted(p.stem.replace("_features", "") for p in output_dir.glob("*_features.csv"))


def compute_feature_retention_correlations(
    output_dir: Path,
    data_dir: Path,
    retention_source: Literal["auto", "studio", "features"] = "auto",
    retention_column: str = "retention",
    method: Literal["pearson", "spearman"] = "spearman",
    features_list: list[str] | None = None,
) -> pd.DataFrame:
    """Per-feature correlation with retention across all videos on the channel."""
    if features_list is None:
        features_list = list(_INTERPRETABLE_FEATURES)
    vids = _discover_vids(output_dir)
    rows: list[dict] = []
    for feat in features_list:
        all_feat_vals: list[float] = []
        all_ret_vals: list[float] = []
        for vid in vids:
            df = _load_features(output_dir, vid)
            if df is None or feat not in df.columns:
                continue
            ret, _ = _resolve_retention(vid, data_dir, df, retention_source, retention_column)
            if ret is None:
                continue
            feat_col = pd.to_numeric(df[feat], errors="coerce").dropna()
            if feat_col.empty:
                continue
            mean_feat = float(feat_col.mean())
            mean_ret = float(ret.mean())
            all_feat_vals.append(mean_feat)
            all_ret_vals.append(mean_ret)
        if len(all_feat_vals) < 5:
            continue
        if method == "spearman":
            corr, pval = sp_stats.spearmanr(all_feat_vals, all_ret_vals)
        else:
            corr, pval = sp_stats.pearsonr(all_feat_vals, all_ret_vals)
        rows.append(
            {"feature": feat, "correlation": round(float(corr), 4), "p_value": round(float(pval), 6), "n_videos": len(all_feat_vals), "direction": "+" if corr > 0 else "-"}
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("correlation", key=abs, ascending=False).reset_index(drop=True)
    return result


def _find_problem_zones(
    retention: np.ndarray, features: pd.DataFrame, top_n: int, drop_threshold_pct: float, min_drop_duration: int, features_list: list[str] | None = None
) -> list[dict]:
    if features_list is None:
        features_list = list(_INTERPRETABLE_FEATURES)
    duration_sec = len(features)
    smoothed = pd.Series(retention).rolling(5, center=True, min_periods=1).mean().values
    gradient = np.gradient(smoothed)
    zones: list[dict] = []
    in_drop = False
    drop_start = 0
    n_ret = len(retention)

    for i, g in enumerate(gradient):
        thresh = -drop_threshold_pct / 100 * max(retention[0], 1.0)
        if g < thresh:
            if not in_drop:
                in_drop = True
                drop_start = i
        else:
            if in_drop and (i - drop_start) >= min_drop_duration:
                _add_zone(zones, retention, features, features_list, drop_start, i, duration_sec, n_ret)
            in_drop = False

    if in_drop and (n_ret - drop_start) >= min_drop_duration:
        _add_zone(zones, retention, features, features_list, drop_start, n_ret - 1, duration_sec, n_ret)

    zones.sort(key=lambda z: z["magnitude"], reverse=True)
    return zones[:top_n]


def _add_zone(zones, retention, features, features_list, start_idx, end_idx, duration_sec, n_ret):
    sec_start = int(start_idx / max(n_ret, 1) * duration_sec)
    sec_end = int(end_idx / max(n_ret, 1) * duration_sec)
    magnitude = float(retention[start_idx] - retention[min(end_idx, n_ret - 1)])
    available = [f for f in features_list if f in features.columns]
    anomalies: list[dict] = []
    for feat in available:
        col = pd.to_numeric(features[feat], errors="coerce")
        mu, std = col.mean(), col.std()
        if std < 1e-6:
            continue
        zone_vals = col.iloc[sec_start : max(sec_start + 1, sec_end)]
        zone_mean = zone_vals.mean()
        z = (zone_mean - mu) / std
        if abs(z) > 0.8:
            anomalies.append({"feature": feat, "zone_mean": round(float(zone_mean), 3), "global_mean": round(float(mu), 3), "z_score": round(float(z), 2)})
    anomalies.sort(key=lambda a: abs(a["z_score"]), reverse=True)
    zones.append(
        {
            "start_sec": sec_start,
            "end_sec": sec_end,
            "from_ret": round(float(retention[start_idx]), 1),
            "to_ret": round(float(retention[min(end_idx, n_ret - 1)]), 1),
            "magnitude": round(magnitude, 1),
            "anomalies": anomalies[:7],
        }
    )


def _format_timecode(sec: int) -> str:
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _build_qwen_prompt(title: str, zones: list[dict], correlations: pd.DataFrame, top_n: int, retention_src: str) -> str:
    corr_lines = ""
    if not correlations.empty:
        top_corr = correlations.head(10)
        corr_lines = "Корреляции признаков с retention по всему каналу (топ-10):\n"
        for _, row in top_corr.iterrows():
            corr_lines += f"  {row['feature']}: r={row['correlation']:+.3f} (p={row['p_value']:.4f})\n"

    zone_lines = ""
    for i, z in enumerate(zones, 1):
        tc0 = _format_timecode(z["start_sec"])
        tc1 = _format_timecode(z["end_sec"])
        zone_lines += f"\nЗона {i}: [{tc0} — {tc1}], retention {z['from_ret']}% → {z['to_ret']}% (падение {z['magnitude']}%)\n"
        if z["anomalies"]:
            zone_lines += "  Аномальные признаки:\n"
            for a in z["anomalies"]:
                zone_lines += f"    - {a['feature']}: {a['zone_mean']} (vs среднее {a['global_mean']}, z={a['z_score']:+.1f})\n"

    retention_note = "Retention — предсказание модели (Studio ещё недоступен)." if retention_src == "features" else "Retention — реальные данные YouTube Studio."

    return f"""\
Ты — YouTube-аналитик. Автор канала просит короткий чек-лист: что исправить в видео.

Заголовок: "{title}"
{retention_note}

{corr_lines}
{zone_lines}
Дай ровно {top_n} конкретных пунктов. Каждый пункт:
- Привязан к таймкоду (или диапазону).
- Содержит 1-2 предложения: что не так + что сделать.
- Опирается на корреляции и аномалии выше.

Формат: нумерованный список, кратко, по-русски. Никакой воды.\
"""


def _qwen_generate(prompt: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_LLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=False, temperature=1.0)
    generated = out[0][inputs["input_ids"].shape[1] :]
    full = tokenizer.decode(generated, skip_special_tokens=False).strip()
    response = full.split("</think>", 1)[1].strip() if "</think>" in full else tokenizer.decode(generated, skip_special_tokens=True).strip()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return response


def generate_actionable_report(
    target_id: str,
    output_dir: Path,
    data_dir: Path,
    out_dir: Path,
    top_n: int = 3,
    retention_source: Literal["auto", "studio", "features"] = "auto",
    retention_column: str = "retention",
    corr_method: Literal["pearson", "spearman"] = "spearman",
    use_qwen: bool = True,
    drop_threshold_pct: float = 3.0,
    min_drop_duration: int = 3,
    features_list: list[str] | None = None,
) -> str:
    features = _load_features(output_dir, target_id)
    if features is None:
        return f"No features CSV for {target_id}."

    retention, ret_src = _resolve_retention(target_id, data_dir, features, retention_source, retention_column)
    if retention is None:
        return f"No retention for {target_id} (source={retention_source})."

    title = _load_title(target_id)

    correlations = compute_feature_retention_correlations(output_dir, data_dir, retention_source, retention_column, corr_method, features_list)

    zones = _find_problem_zones(retention, features, top_n, drop_threshold_pct, min_drop_duration, features_list)

    md_lines = [f"# Чек-лист: {title}", "", f"> Retention ({ret_src}): {int(retention[0])}% → {int(retention[-1])}% (среднее {int(retention.mean())}%)", ""]

    if not correlations.empty:
        md_lines.append("## Корреляции признаков с retention (канал)")
        md_lines.append("")
        md_lines.append("| Признак | r | p | Направление |")
        md_lines.append("|---------|---|---|-------------|")
        for _, row in correlations.head(15).iterrows():
            md_lines.append(f"| {row['feature']} | {row['correlation']:+.3f} | {row['p_value']:.4f} | {row['direction']} |")
        md_lines.append("")

    if not zones:
        md_lines.append("Значительных зон падения не обнаружено.")
    else:
        md_lines.append(f"## Проблемные зоны (топ-{len(zones)})")
        md_lines.append("")
        for i, z in enumerate(zones, 1):
            tc0 = _format_timecode(z["start_sec"])
            tc1 = _format_timecode(z["end_sec"])
            md_lines.append(f"### {i}. [{tc0} — {tc1}] retention {z['from_ret']}% → {z['to_ret']}% (−{z['magnitude']}%)")
            if z["anomalies"]:
                for a in z["anomalies"][:5]:
                    md_lines.append(f"- **{a['feature']}**: {a['zone_mean']} (vs {a['global_mean']}, z={a['z_score']:+.1f})")
            md_lines.append("")

    if use_qwen and zones:
        prompt = _build_qwen_prompt(title, zones, correlations, top_n, ret_src)
        qwen_text = _qwen_generate(prompt)
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("## Рекомендации")
        md_lines.append("")
        md_lines.append(qwen_text)

    report = "\n".join(md_lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{target_id}_actionable.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Actionable report ({len(zones)} zones, top-{top_n}) → {report_path}")

    if not correlations.empty:
        corr_path = out_dir / "feature_retention_correlations.csv"
        correlations.to_csv(corr_path, index=False)
        print(f"Correlations → {corr_path}")

    return report


def main():
    p = argparse.ArgumentParser(description="Actionable one-screen report")
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--data-dir", type=Path, default=Path("train/drive_snapshot_90"))
    p.add_argument("--out", type=Path, default=Path("actionable_reports"))
    p.add_argument("--top-n", type=int, default=3, help="Number of action items")
    p.add_argument("--retention-source", choices=("auto", "studio", "features"), default="auto")
    p.add_argument("--retention-column", type=str, default="retention")
    p.add_argument("--corr-method", choices=("pearson", "spearman"), default="spearman")
    p.add_argument("--no-qwen", action="store_true")
    p.add_argument("--drop-threshold", type=float, default=3.0, help="Min gradient threshold %% to detect a drop")
    p.add_argument("--min-drop-duration", type=int, default=3, help="Min consecutive points in gradient for a drop zone")
    args = p.parse_args()

    generate_actionable_report(
        target_id=args.target,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        out_dir=args.out,
        top_n=args.top_n,
        retention_source=args.retention_source,
        retention_column=args.retention_column,
        corr_method=args.corr_method,
        use_qwen=not args.no_qwen,
        drop_threshold_pct=args.drop_threshold,
        min_drop_duration=args.min_drop_duration,
    )


if __name__ == "__main__":
    main()
