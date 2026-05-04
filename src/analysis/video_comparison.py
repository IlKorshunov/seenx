"""Compare a target video with its best-performing neighbours on the same channel.

At **inference** time you often have no real YouTube retention yet — only a
**predicted** curve (e.g. model output written into the `retention` column of
`*_features.csv`). Use ``--retention-source features`` so ranking and deltas use
that column. With ``auto`` (default), Studio/Analytics retention is used when
present, otherwise the features column is used.

Pipeline:
1. Find videos with features + a resolvable retention curve (Studio or predicted).
2. Similarity via embeddings (cosine) + duration; rank by similarity + predicted/actual quality.
3. Top-K neighbours; deltas along interpretable axes.
4. Optional Qwen markdown report.

CLI:
  python -m src.analysis.video_comparison \\
      --target VIDEO_ID \\
      --data-dir train/drive_snapshot_90 \\
      --output-dir output \\
      --embeddings-dir embeddings \\
      --retention-source features \\
      --top-k 3

Assumes each video is longer than one minute (duration_sec > 60) for indexing
along the retention curve; guard rails for shorter clips are omitted.
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
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.video_features import EMBEDDING_TYPES, embeddings_to_matrix, find_json, load_embeddings, meta_nums


_LLM_MODEL_ID = "Qwen/Qwen3-4B"

_COMPARISON_AXES = [
    ("intro_length_sec", "Длина интро (секунды до окончания is_intro=1)"),
    ("mean_retention", "Среднее удержание (%)"),
    ("retention_30s", "Удержание на 30-й секунде (%)"),
    ("retention_60s", "Удержание на 60-й секунде (%)"),
    ("mean_edit_pace", "Плотность монтажа (edit_pace)"),
    ("mean_wps", "Скорость речи (слов/сек)"),
    ("mean_speech_intelligibility", "Разборчивость речи"),
    ("mean_speech_mumble_index", "Индекс «зажёвывания»"),
    ("mean_friction_total", "Общее трение (friction_total)"),
    ("mean_curiosity_gap", "Curiosity gap"),
    ("mean_storytelling", "Storytelling"),
    ("mean_viewer_engagement", "Вовлечение зрителей"),
    ("mean_information_density", "Информационная плотность"),
    ("mean_semantic_novelty", "Семантическая новизна"),
    ("mean_speech_lm_surprisal", "LM surprisal (неожиданность речи)"),
    ("mean_motion_speed", "Скорость движения в кадре"),
    ("mean_brightness", "Яркость кадра"),
    ("mean_speaker_prob", "Доля кадров со спикером"),
    ("duration_sec", "Длительность видео (сек)"),
    ("ad_fraction", "Доля рекламных сегментов"),
    ("hook_score_first_30", "Hook score (первые 30 с)"),
]

_SCALAR_FEATURES = [
    "edit_pace",
    "wps",
    "speech_intelligibility",
    "speech_mumble_index",
    "friction_total",
    "curiosity_gap",
    "storytelling",
    "viewer_engagement",
    "information_density",
    "semantic_novelty",
    "speech_lm_surprisal",
    "motion_speed",
    "brightness",
    "speaker_prob",
]


def _interp_studio_df(df: pd.DataFrame) -> np.ndarray | None:
    if {"time_ratio", "audience_watch_ratio"} <= set(df.columns):
        return np.interp(np.linspace(0, 1, 100), df["time_ratio"].values, df["audience_watch_ratio"].values * 100.0)
    return None


def _load_retention(data_dir: Path, vid: str) -> np.ndarray | None:
    base = data_dir / vid
    for csv_path in (base / "retention_source.csv", base / "transcripts" / "retention_source.csv", base / "retention.csv"):
        if csv_path.exists() and (arr := _interp_studio_df(pd.read_csv(csv_path))) is not None:
            return arr
    for jname in ("retention.json", "retention_parsed.json"):
        jp = base / jname
        if not jp.exists():
            jp = base / "transcripts" / jname
        if jp.exists():
            raw = json.loads(jp.read_text(encoding="utf-8"))
            if isinstance(raw, list) and raw:
                vals = np.array([float(r.get("audienceWatchRatio", r) if isinstance(r, dict) else r) for r in raw])
                if vals.max() <= 1.5:
                    vals = vals * 100.0
                return np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(vals)), vals)
    return None


def _normalize_retention_scale(vals: np.ndarray) -> np.ndarray:
    out = np.asarray(vals, dtype=np.float64)
    return out * 100.0 if np.nanmax(out) <= 1.5 else out


def _retention_from_features_df(features: pd.DataFrame, column: str = "retention") -> np.ndarray | None:
    if column not in features.columns:
        return None
    ser = pd.to_numeric(features[column], errors="coerce").ffill().bfill().fillna(0.0)
    if ser.isna().all():
        return None
    n = len(ser)
    raw = _normalize_retention_scale(ser.values.astype(np.float64))
    xp = np.linspace(0.0, 1.0, 100)
    return np.full(100, float(raw[0])) if n < 2 else np.interp(xp, np.linspace(0.0, 1.0, n), raw)


def _resolve_retention(
    vid: str, data_dir: Path, features: pd.DataFrame | None, source: Literal["auto", "studio", "features"], retention_column: str
) -> tuple[np.ndarray | None, str]:
    studio = _load_retention(data_dir, vid)
    from_csv = _retention_from_features_df(features, retention_column) if features is not None else None
    if source == "studio":
        return (studio, "studio") if studio is not None else (None, "none")
    if source == "features":
        return (from_csv, "predicted_features") if from_csv is not None else (None, "none")
    if studio is not None:
        return studio, "studio"
    if from_csv is not None:
        return from_csv, "predicted_features"
    return None, "none"


def _load_features(output_dir: Path, vid: str) -> pd.DataFrame | None:
    cands = [(output_dir / f"{vid}_features.csv", {"index_col": 0}), (output_dir / vid / "features_readable.csv", {})]
    return next((pd.read_csv(p, **kw) for p, kw in cands if p.exists()), None)


def _discover_videos(data_dir: Path, output_dir: Path) -> list[str]:
    from_output = {p.stem.replace("_features", "") for p in output_dir.glob("*_features.csv")} if output_dir.exists() else set()
    from_data = {d.name for d in data_dir.iterdir() if d.is_dir()} if data_dir.exists() else set()
    return sorted(from_output | from_data)


def _build_embedding_matrix(vids: list[str], emb_dir: Path) -> np.ndarray:
    parts = [embeddings_to_matrix(vecs, len(vids), max_dim) for modality in EMBEDDING_TYPES for vecs, max_dim in [load_embeddings(vids, emb_dir, modality)] if max_dim > 0]
    return np.hstack(parts) if parts else np.zeros((len(vids), 1))


def _compute_video_stats(features: pd.DataFrame, retention: np.ndarray, duration_sec: float) -> dict[str, float]:
    n = len(features)
    intro_end = 0
    if "is_intro" in features.columns:
        intro_col = features["is_intro"].values
        for i in range(min(n, int(duration_sec))):
            if intro_col[i] > 0.5:
                intro_end = i + 1
            elif intro_end > 0:
                break
    first_30_idx = min(n, int(30 / duration_sec * n))

    stats: dict[str, float] = {
        "duration_sec": duration_sec,
        "mean_retention": float(retention.mean()),
        "retention_30s": float(retention[min(int(30 / duration_sec * 100), 99)]),
        "retention_60s": float(retention[min(int(60 / duration_sec * 100), 99)]),
        "intro_length_sec": float(intro_end),
        **{f"mean_{feat}": float(pd.to_numeric(features[feat], errors="coerce").mean()) if feat in features.columns else float("nan") for feat in _SCALAR_FEATURES},
        "ad_fraction": float(features["is_ad"].mean()) if "is_ad" in features.columns else 0.0,
        "hook_score_first_30": (
            float(pd.to_numeric(features["hook_score"].iloc[: max(1, first_30_idx)], errors="coerce").mean()) if "hook_score" in features.columns else float("nan")
        ),
    }
    return stats


def find_neighbours(target_id: str, vids: list[str], emb_dir: Path, retentions: dict[str, np.ndarray], durations: dict[str, float], top_k: int = 3) -> list[tuple[str, float]]:
    if target_id not in vids:
        return []
    emb_matrix = _build_embedding_matrix(vids, emb_dir)
    ti = vids.index(target_id)
    sims = cosine_similarity(emb_matrix[ti : ti + 1], emb_matrix)[0]
    target_dur = durations.get(target_id, 0)
    target_ret_mean = float(retentions[target_id].mean())
    scored = []
    for i, vid in enumerate(vids):
        if vid == target_id or vid not in retentions:
            continue
        dur = durations.get(vid, 0)
        dur_ratio = min(target_dur, dur) / max(target_dur, dur) if target_dur > 0 and dur > 0 else 0.5
        scored.append((vid, float(sims[i]) * 0.5 + dur_ratio * 0.2 + max(0, float(retentions[vid].mean()) - target_ret_mean) / 100.0 * 0.3))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]


def compare_videos(target_stats: dict[str, float], neighbour_stats: dict[str, float], neighbour_id: str) -> list[dict]:
    deltas = []
    for key, label in _COMPARISON_AXES:
        t_val, n_val = target_stats.get(key, float("nan")), neighbour_stats.get(key, float("nan"))
        if np.isnan(t_val) or np.isnan(n_val):
            continue
        diff = n_val - t_val
        if abs(diff) < 1e-6:
            continue
        pct = diff / abs(t_val) * 100 if abs(t_val) > 1e-6 else 0.0
        deltas.append(
            {
                "axis": key,
                "label": label,
                "target_val": round(t_val, 3),
                "neighbour_val": round(n_val, 3),
                "diff": round(diff, 3),
                "diff_pct": round(pct, 1),
                "neighbour_id": neighbour_id,
            }
        )
    deltas.sort(key=lambda d: abs(d["diff_pct"]), reverse=True)
    return deltas


def _format_comparison_table(target_id: str, comparisons: dict[str, list[dict]]) -> str:
    blocks = [f"# Сравнение: {target_id}\n"]
    for nid, deltas in comparisons.items():
        rows = "\n".join(
            "| {lb} | {tv} | {nv} | {sg}{df} | {sg}{dp}% |".format(
                lb=d["label"], tv=d["target_val"], nv=d["neighbour_val"], df=d["diff"], dp=d["diff_pct"], sg="+" if d["diff"] > 0 else ""
            )
            for d in deltas[:15]
        )
        blocks += [f"\n## vs {nid}\n", f"| Ось | {target_id} | {nid} | Δ | Δ% |\n|------|------------|-------|---|----|\n", rows]
    return "\n".join(blocks)


_COMPARISON_PROMPT = """\
Ты — эксперт по YouTube-аналитике. Автор канала хочет понять, почему его видео \
"{target_id}" удерживает зрителей хуже, чем похожие ролики на его же канале.

{retention_context}

Вот таблица сравнения target-видео с {n_neighbours} лучшими соседями:

{table}

На основе этих данных:
1. Выдели ТОП-3 самых значимых отличия, которые вероятнее всего объясняют разницу в удержании.
2. Для каждого дай конкретную, действенную рекомендацию автору (что изменить в следующем ролике).
3. Если видишь паттерн (например, «у всех успешных видео короче интро»), укажи его явно.

Пиши кратко, по-русски, в формате markdown. Без воды.\
"""


def generate_comparison_report(
    target_id: str, comparisons: dict[str, list[dict]], target_stats: dict[str, float], neighbour_stats: dict[str, dict[str, float]], retention_context: str
) -> str:
    table = _format_comparison_table(target_id, comparisons)
    tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_LLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    stats_block = "\n".join(
        [f"Статистика target ({target_id}):"]
        + [f"  {k}: {v:.3f}" for k, v in sorted(target_stats.items())]
        + [ln for nid, ns in neighbour_stats.items() for ln in ([f"\nСтатистика {nid}:"] + [f"  {k}: {v:.3f}" for k, v in sorted(ns.items())])]
    )
    prompt = _COMPARISON_PROMPT.format(target_id=target_id, retention_context=retention_context, n_neighbours=len(comparisons), table=f"{table}\n\n{stats_block}")
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
    return f"{table}\n\n---\n\n## Рекомендации (Qwen)\n\n{response}"


def _retention_md_banner(sources: dict[str, str], retention_column: str) -> tuple[str, str]:
    kinds = set(sources.values()) - {"none"}
    if kinds == {"studio"}:
        return (
            "> **Удержание:** реальные данные YouTube (Studio / Analytics), 100 точек.\n",
            "Кривые удержания — **реальные** (YouTube Studio). Сравнение отражает фактическое поведение зрителей.",
        )
    if kinds == {"predicted_features"}:
        return (
            f"> **Удержание:** предсказанная моделью кривая из колонки `{retention_column}` в `*_features.csv` (на момент анализа реального retention ещё нет).\n",
            "Кривые удержания — **предсказанные моделью** (колонка в CSV), не факт YouTube. Рекомендации опираются на эту оценку; после публикации сверяйте с Studio.",
        )
    if not kinds:
        return "> **Удержание:** источник не определён.\n", "Источник кривых удержания неизвестен."
    return (
        f"> **Удержание:** смешанный источник — часть роликов из Studio, часть из предсказанной колонки `{retention_column}` в features CSV.\n",
        "Источники кривых **разные** (часть реальная Studio, часть предсказание в CSV). Учитывай это при интерпретации сравнения.",
    )


def run_comparison(
    target_id: str,
    data_dir: Path,
    output_dir: Path,
    emb_dir: Path,
    out_dir: Path,
    top_k: int = 3,
    use_qwen: bool = True,
    retention_source: Literal["auto", "studio", "features"] = "auto",
    retention_column: str = "retention",
) -> str:
    vids = _discover_videos(data_dir, output_dir)
    if target_id not in vids:
        return f"Video {target_id} not found among {len(vids)} candidates."

    retentions, retention_sources, durations, features_cache = {}, {}, {}, {}

    for vid in vids:
        meta = find_json(data_dir, vid, "meta.json") or find_json(data_dir, vid, "metadata.json")
        dur, _, _ = meta_nums(meta)
        feat = _load_features(output_dir, vid)
        if feat is not None:
            features_cache[vid] = feat
            if dur <= 0:
                dur = float(len(feat))
        if dur > 0:
            durations[vid] = dur
        ret, src = _resolve_retention(vid, data_dir, feat, retention_source, retention_column)
        if ret is not None:
            retentions[vid], retention_sources[vid] = ret, src

    if target_id not in retentions:
        return (
            f"No retention curve for target {target_id}. "
            f"Add Studio CSV under data-dir, or ensure `{retention_column}` exists in "
            f"{target_id}_features.csv and use --retention-source features (or auto)."
        )

    neighbours = find_neighbours(target_id, vids, emb_dir, retentions, durations, top_k)
    if not neighbours:
        return "No suitable neighbours found."

    target_feat = features_cache.get(target_id)
    if target_feat is None:
        return f"No features CSV for target {target_id}."

    target_stats = _compute_video_stats(target_feat, retentions[target_id], durations.get(target_id, float(len(target_feat))))
    comparisons, all_neighbour_stats = {}, {}
    md_banner, llm_ctx = _retention_md_banner(retention_sources, retention_column)
    print(md_banner.strip())
    print(f"\nTarget: {target_id}  (retention mean={target_stats['mean_retention']:.1f}%)")

    for nid, score in neighbours:
        n_feat = features_cache.get(nid)
        if n_feat is None:
            continue
        n_stats = _compute_video_stats(n_feat, retentions[nid], durations.get(nid, float(len(n_feat))))
        all_neighbour_stats[nid] = n_stats
        comparisons[nid] = compare_videos(target_stats, n_stats, nid)
        print(f"  Neighbour: {nid}  (score={score:.3f}, retention mean={n_stats['mean_retention']:.1f}%, src={retention_sources.get(nid, '?')})")

    report = generate_comparison_report(target_id, comparisons, target_stats, all_neighbour_stats, llm_ctx) if use_qwen else _format_comparison_table(target_id, comparisons)
    report = f"{md_banner}\n{report}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{target_id}_comparison.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {report_path}")
    return report


def main():
    p = argparse.ArgumentParser(description="Compare video with best-performing neighbours")
    p.add_argument("--target", type=str, required=True, help="Target video ID")
    p.add_argument("--data-dir", type=Path, default=Path("train/drive_snapshot_90"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--embeddings-dir", type=Path, default=Path("embeddings"))
    p.add_argument("--out", type=Path, default=Path("comparisons"))
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--no-qwen", action="store_true", help="Skip Qwen report generation")
    p.add_argument(
        "--retention-source",
        choices=("auto", "studio", "features"),
        default="auto",
        help="auto: Studio if present else column in features; features: only predicted CSV column; studio: only YouTube files",
    )
    p.add_argument("--retention-column", type=str, default="retention", help="Column in *_features.csv for predicted retention (inference without Studio)")
    args = p.parse_args()
    run_comparison(
        target_id=args.target,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        emb_dir=args.embeddings_dir,
        out_dir=args.out,
        top_k=args.top_k,
        use_qwen=not args.no_qwen,
        retention_source=args.retention_source,
        retention_column=args.retention_column,
    )


if __name__ == "__main__":
    main()
