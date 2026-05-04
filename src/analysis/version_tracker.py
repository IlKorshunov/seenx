"""Version tracker: save prediction snapshots, compare before/after.

Storage: `{out_dir}/{video_id}/v{N}.json`.

CLI:
  python -m src.analysis.version_tracker save --target VIDEO_ID --output-dir output --data-dir train/drive_snapshot_90
  python -m src.analysis.version_tracker diff --target VIDEO_ID
  python -m src.analysis.version_tracker list --target VIDEO_ID
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_LLM_MODEL_ID = "Qwen/Qwen3-4B"

_TRACKED_FEATURES = [
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
    "hook_score",
    "is_ad",
    "scene_novelty",
    "visual_entropy",
]


def _load_features(output_dir: Path, vid: str) -> pd.DataFrame | None:
    cands = [(output_dir / f"{vid}_features.csv", {"index_col": 0}), (output_dir / vid / "features_readable.csv", {})]
    return next((pd.read_csv(p, **kw) for p, kw in cands if p.exists()), None)


def _retention_from_features(features: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in features.columns:
        return None
    ser = pd.to_numeric(features[column], errors="coerce")
    if ser.isna().all():
        return None
    vals = ser.ffill().bfill().fillna(0.0).values.astype(np.float64)
    return vals * 100.0 if np.nanmax(vals) <= 1.5 else vals


def _load_retention_studio(data_dir: Path, vid: str) -> np.ndarray | None:
    base = data_dir / vid
    for csv_path in (base / "retention_source.csv", base / "transcripts" / "retention_source.csv"):
        if csv_path.exists() and {"time_ratio", "audience_watch_ratio"} <= set((df := pd.read_csv(csv_path)).columns):
            return np.interp(np.linspace(0, 1, 100), df["time_ratio"].values, df["audience_watch_ratio"].values * 100.0)
    return None


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


def _snapshot_dir(out_dir: Path, vid: str) -> Path:
    return out_dir / vid


def _list_snapshots(out_dir: Path, vid: str) -> list[Path]:
    d = _snapshot_dir(out_dir, vid)
    return sorted(d.glob("v*.json"), key=lambda p: int(p.stem[1:])) if d.exists() else []


def _next_version(out_dir: Path, vid: str) -> int:
    return max((int(p.stem[1:]) for p in _list_snapshots(out_dir, vid)), default=0) + 1


def _compute_snapshot(features: pd.DataFrame, retention: np.ndarray, retention_src: str, tracked_features: list[str] | None = None) -> dict:
    tf = tracked_features or list(_TRACKED_FEATURES)
    curve = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(retention)), retention)
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "retention_source": retention_src,
        "retention_curve_100": [round(float(v), 2) for v in curve],
        "retention_mean": round(float(retention.mean()), 2),
        "feature_means": {f: round(float(pd.to_numeric(features[f], errors="coerce").mean()), 4) for f in tf if f in features.columns},
        "n_seconds": len(features),
    }


def save_snapshot(
    target_id: str,
    output_dir: Path,
    data_dir: Path,
    out_dir: Path,
    retention_source: Literal["auto", "studio", "features"] = "auto",
    retention_column: str = "retention",
    note: str = "",
    tracked_features: list[str] | None = None,
) -> Path:
    features = _load_features(output_dir, target_id)
    if features is None:
        raise FileNotFoundError(f"No features CSV for {target_id}")
    retention, ret_src = _resolve_retention(target_id, data_dir, features, retention_source, retention_column)
    if retention is None:
        raise ValueError(f"No retention for {target_id}")
    snap = _compute_snapshot(features, retention, ret_src, tracked_features)
    ver = _next_version(out_dir, target_id)
    snap |= {"version": ver, "note": note}
    d = _snapshot_dir(out_dir, target_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"v{ver}.json"
    path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Snapshot v{ver} saved → {path}")
    return path


def _format_timecode(sec: int) -> str:
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _retention_zone_bullets(ret_delta: np.ndarray, n_sec: int) -> tuple[list[str], list[str]]:
    improved, degraded = [], []
    for i in range(0, 100, 10):
        mean_d = float(ret_delta[i : min(i + 10, 100)].mean())
        tc0 = _format_timecode(int(i / 100 * n_sec))
        tc1 = _format_timecode(int(min(i + 10, 100) / 100 * n_sec))
        if mean_d > 1.0:
            improved.append(f"  - [{tc0}–{tc1}]: +{mean_d:.1f}%")
        elif mean_d < -1.0:
            degraded.append(f"  - [{tc0}–{tc1}]: {mean_d:.1f}%")
    return improved, degraded


def diff_snapshots(target_id: str, out_dir: Path, v_old: int | None = None, v_new: int | None = None, use_qwen: bool = True) -> str:
    snaps = _list_snapshots(out_dir, target_id)
    if len(snaps) < 2:
        return f"Need at least 2 snapshots for {target_id}, found {len(snaps)}."
    sd = _snapshot_dir(out_dir, target_id)
    if v_old is not None and v_new is not None:
        old_path, new_path = sd / f"v{v_old}.json", sd / f"v{v_new}.json"
    else:
        old_path, new_path = snaps[-2], snaps[-1]
    old, new = json.loads(old_path.read_text(encoding="utf-8")), json.loads(new_path.read_text(encoding="utf-8"))
    ret_delta = np.array(new["retention_curve_100"]) - np.array(old["retention_curve_100"])
    n_sec = old.get("n_seconds", 100)
    improved, degraded = _retention_zone_bullets(ret_delta, n_sec)

    md = [
        f"# До/После: {target_id}",
        "",
        f"> v{old['version']} ({old['timestamp'][:19]}) → v{new['version']} ({new['timestamp'][:19]})",
        "",
        f"Retention mean: {old['retention_mean']}% → {new['retention_mean']}% (Δ {new['retention_mean'] - old['retention_mean']:+.1f}%)",
        "",
    ]
    if improved:
        md += ["**Улучшилось:**", *improved, ""]
    if degraded:
        md += ["**Ухудшилось:**", *degraded, ""]

    of, nf = old.get("feature_means", {}), new.get("feature_means", {})
    feat_deltas = sorted(
        [(k, float(of[k]), float(nf[k]), float(nf[k]) - float(of[k])) for k in of.keys() & nf.keys() if abs(float(nf[k]) - float(of[k])) > 1e-4],
        key=lambda t: abs(t[3]),
        reverse=True,
    )
    if feat_deltas:
        md += [
            "## Изменения признаков",
            "",
            "| Признак | Было | Стало | Δ |",
            "|---------|------|-------|---|",
            *[f"| {k} | {ov:.4f} | {nv:.4f} | {'+' if d > 0 else ''}{d:.4f} |" for k, ov, nv, d in feat_deltas[:20]],
            "",
        ]

    if use_qwen and (improved or degraded or feat_deltas):
        md += ["---", "", "## Вывод (Qwen)", "", _qwen_generate(_build_diff_prompt(target_id, old, new, improved, degraded, feat_deltas))]

    report_path = _snapshot_dir(out_dir, target_id) / f"diff_v{old['version']}_v{new['version']}.md"
    report_path.write_text(report := "\n".join(md), encoding="utf-8")
    print(f"Diff report → {report_path}")
    return report


def _build_diff_prompt(target_id: str, old: dict, new: dict, improved: list[str], degraded: list[str], feat_deltas: list[tuple[str, float, float, float]]) -> str:
    parts = [f'Автор переделал монтаж видео "{target_id}".', f"Retention mean: {old['retention_mean']}% → {new['retention_mean']}%.", ""]
    if improved:
        parts += ["Зоны, где retention вырос:", *improved]
    if degraded:
        parts += ["Зоны, где retention упал:", *degraded]
    if feat_deltas:
        parts += ["\nИзменения признаков:", *[f"  {k}: {ov:.3f} → {nv:.3f} (Δ{d:+.3f})" for k, ov, nv, d in feat_deltas[:10]]]
    parts.append("\nКратко (3-5 предложений, по-русски):\n1. Что именно автор, вероятно, изменил?\n2. Помогло ли это удержанию? Если нет — что ещё стоит попробовать?")
    return "\n".join(parts)


def _qwen_generate(prompt: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_LLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False, temperature=1.0)
    generated = out[0][inputs["input_ids"].shape[1] :]
    full = tokenizer.decode(generated, skip_special_tokens=False).strip()
    response = full.split("</think>", 1)[1].strip() if "</think>" in full else tokenizer.decode(generated, skip_special_tokens=True).strip()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return response


def main():
    p = argparse.ArgumentParser(description="Video version tracker")
    sub = p.add_subparsers(dest="command", required=True)
    save_p = sub.add_parser("save")
    save_p.add_argument("--target", type=str, required=True)
    save_p.add_argument("--output-dir", type=Path, default=Path("output"))
    save_p.add_argument("--data-dir", type=Path, default=Path("train/drive_snapshot_90"))
    save_p.add_argument("--out", type=Path, default=Path("version_history"))
    save_p.add_argument("--retention-source", choices=("auto", "studio", "features"), default="auto")
    save_p.add_argument("--retention-column", type=str, default="retention")
    save_p.add_argument("--note", type=str, default="")
    diff_p = sub.add_parser("diff")
    diff_p.add_argument("--target", type=str, required=True)
    diff_p.add_argument("--out", type=Path, default=Path("version_history"))
    diff_p.add_argument("--v-old", type=int, default=None)
    diff_p.add_argument("--v-new", type=int, default=None)
    diff_p.add_argument("--no-qwen", action="store_true")
    list_p = sub.add_parser("list")
    list_p.add_argument("--target", type=str, required=True)
    list_p.add_argument("--out", type=Path, default=Path("version_history"))
    args = p.parse_args()
    if args.command == "save":
        save_snapshot(args.target, args.output_dir, args.data_dir, args.out, args.retention_source, args.retention_column, args.note)
    elif args.command == "diff":
        print(diff_snapshots(args.target, args.out, args.v_old, args.v_new, use_qwen=not args.no_qwen))
    else:
        snaps = _list_snapshots(args.out, args.target)
        if not snaps:
            print(f"No snapshots for {args.target}")
        else:
            for s in snaps:
                data = json.loads(s.read_text(encoding="utf-8"))
                note = f" — {data['note']}" if data.get("note") else ""
                print(f"  {s.stem}: {data['timestamp'][:19]}  ret_mean={data['retention_mean']}%{note}")


if __name__ == "__main__":
    main()
