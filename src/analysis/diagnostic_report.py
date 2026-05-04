"""Diagnostic report generator: Qwen3-4B analyzes retention drops.

For each video with retention data, finds significant drop zones,
correlates them with feature anomalies, and generates a markdown
report with actionable recommendations for the channel author.

Usage:
  python -m src.analysis.diagnostic_report --data-dir data --output-dir output

Output:
  diagnostics/{video_id}_report.md per video
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_LLM_MODEL_ID = "Qwen/Qwen3-4B"

_DROP_THRESHOLD_PCT = 3.0
_MIN_DROP_DURATION = 3

_KEY_FEATURES = [
    "edit_pace",
    "speech_ratio",
    "viewer_engagement",
    "information_density",
    "friction_total",
    "friction_jargon",
    "friction_abstract",
    "friction_repetition",
    "curiosity_gap",
    "storytelling",
    "expectation_surprise",
    "saliency_mean",
    "brightness",
    "motion_speed",
    "wps",
]

_REPORT_PROMPT = """\
Ты — эксперт по YouTube-аналитике. Проанализируй видео и дай конкретные рекомендации автору.

Заголовок: "{title}"

Retention curve (0-100%, 100 точек от начала до конца видео):
{retention_summary}

Обнаруженные зоны падения retention:
{drops}

Для каждой зоны падения дай:
1. Вероятную причину (на основе признаков)
2. Конкретную рекомендацию автору — что изменить в монтаже/подаче/структуре
3. Пример хорошей практики

Пиши на русском, кратко, по делу. Формат: markdown с заголовками ## для каждой зоны."""


def _load_retention(data_dir: Path, vid: str) -> np.ndarray | None:
    csv = data_dir / vid / "retention.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if {"time_ratio", "audience_watch_ratio"} <= set(df.columns):
            return np.interp(np.linspace(0, 1, 100), df["time_ratio"].values, df["audience_watch_ratio"].values * 100.0)
    for jname in ("retention.json", "retention_parsed.json"):
        jp = data_dir / vid / jname
        if jp.exists():
            raw = json.loads(jp.read_text(encoding="utf-8"))
            if isinstance(raw, list) and raw:
                vals = np.array([float(r.get("audienceWatchRatio", r) if isinstance(r, dict) else r) for r in raw])
                if vals.max() <= 1.5:
                    vals *= 100.0
                return np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(vals)), vals)
    return None


def _load_features(output_dir: Path, vid: str) -> pd.DataFrame | None:
    for path, kw in [(output_dir / f"{vid}_features.csv", {"index_col": 0}), (output_dir / vid / "features_readable.csv", {})]:
        if path.exists():
            return pd.read_csv(path, **kw)
    return None


def _load_title(vid: str) -> str:
    root = Path(__file__).resolve().parents[2]
    return next((json.loads(p.read_text(encoding="utf-8")).get("video_title", vid) for p in (root / "get_data" / "comments").rglob(f"{vid}/comments.json")), vid)


def _find_drops(retention: np.ndarray, duration_sec: int) -> list[dict]:
    smoothed = pd.Series(retention).rolling(5, center=True, min_periods=1).mean().values
    gradient = np.gradient(smoothed)

    drops = []
    in_drop = False
    drop_start = 0

    for i, g in enumerate(gradient):
        if g < -_DROP_THRESHOLD_PCT / 100 * retention[0]:
            if not in_drop:
                in_drop = True
                drop_start = i
        else:
            if in_drop and (i - drop_start) >= _MIN_DROP_DURATION:
                sec_start = int(drop_start / 100 * duration_sec)
                sec_end = int(i / 100 * duration_sec)
                magnitude = float(retention[drop_start] - retention[min(i, 99)])
                drops.append(
                    {
                        "start_pct": drop_start,
                        "end_pct": i,
                        "start_sec": sec_start,
                        "end_sec": sec_end,
                        "magnitude": round(magnitude, 1),
                        "from_ret": round(float(retention[drop_start]), 1),
                        "to_ret": round(float(retention[min(i, 99)]), 1),
                    }
                )
            in_drop = False

    return drops[:5]


def _feature_anomalies(features: pd.DataFrame, start_sec: int, end_sec: int) -> list[str]:
    available = [f for f in _KEY_FEATURES if f in features.columns]
    if not available:
        return []

    anomalies = []
    for feat in available:
        col = pd.to_numeric(features[feat], errors="coerce")
        global_mean = col.mean()
        global_std = col.std()
        if global_std < 1e-6:
            continue
        zone = col.iloc[start_sec:end_sec]
        zone_mean = zone.mean()
        z = (zone_mean - global_mean) / global_std
        if abs(z) > 1.0:
            direction = "выше" if z > 0 else "ниже"
            anomalies.append(f"{feat}: {zone_mean:.2f} ({direction} среднего {global_mean:.2f}, z={z:+.1f})")

    return sorted(anomalies, key=lambda s: abs(float(s.split("z=")[1].rstrip(")"))), reverse=True)[:5]


def _format_drops(drops: list[dict], features: pd.DataFrame | None) -> str:
    lines = []
    for i, d in enumerate(drops, 1):
        lines.append(f"Зона {i}: {d['start_sec']}с — {d['end_sec']}с (retention {d['from_ret']}% → {d['to_ret']}%, падение {d['magnitude']}%)")
        if features is not None:
            anomalies = _feature_anomalies(features, d["start_sec"], d["end_sec"])
            if anomalies:
                lines.append("  Аномальные признаки:")
                for a in anomalies:
                    lines.append(f"    - {a}")
    return "\n".join(lines)


def _retention_summary(retention: np.ndarray) -> str:
    points = [f"{int(retention[i])}%" for i in range(0, 100, 10)]
    return f"Начало: {points[0]}, " + ", ".join(f"{i * 10}%: {p}" for i, p in enumerate(points)) + f", Конец: {int(retention[-1])}%"


def _generate_report(title: str, retention: np.ndarray, drops: list[dict], features: pd.DataFrame | None, duration: int) -> str:
    if not drops:
        return f"# {title}\n\nЗначительных падений retention не обнаружено.\n"

    tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_LLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prompt = _REPORT_PROMPT.format(title=title, retention_summary=_retention_summary(retention), drops=_format_drops(drops, features))
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=1.0)
    generated = out[0][inputs["input_ids"].shape[1] :]
    full = tokenizer.decode(generated, skip_special_tokens=False).strip()
    response = full.split("</think>", 1)[1].strip() if "</think>" in full else tokenizer.decode(generated, skip_special_tokens=True).strip()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    header = f"# Диагностика: {title}\n\n"
    header += f"**Retention:** {int(retention[0])}% → {int(retention[-1])}% "
    header += f"(средний: {int(retention.mean())}%)\n\n"
    header += f"**Зон падения:** {len(drops)}\n\n---\n\n"

    return header + response


def generate_reports(data_dir: Path, output_dir: Path, out_dir: Path, video_ids: list[str] | None = None) -> list[str]:
    if video_ids is None:
        video_ids = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and (d / "retention.csv").exists())

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for vid in video_ids:
        retention = _load_retention(data_dir, vid)
        if retention is None:
            continue

        features = _load_features(output_dir, vid)
        title = _load_title(vid)
        duration = len(features) if features is not None else len(retention)
        drops = _find_drops(retention, duration)

        print(f"[{vid}] {len(drops)} drop zones, retention {int(retention[0])}% → {int(retention[-1])}%")

        report = _generate_report(title, retention, drops, features, duration)
        report_path = out_dir / f"{vid}_report.md"
        report_path.write_text(report, encoding="utf-8")
        generated.append(str(report_path))

    print(f"\nGenerated {len(generated)} reports → {out_dir}")
    return generated


def main():
    p = argparse.ArgumentParser(description="Generate retention diagnostic reports")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--out", type=Path, default=Path("diagnostics"))
    p.add_argument("--video-id", type=str, nargs="*", default=None)
    args = p.parse_args()

    generate_reports(args.data_dir, args.output_dir, args.out, args.video_id)


if __name__ == "__main__":
    main()
