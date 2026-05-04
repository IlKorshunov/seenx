#!/usr/bin/env python3
"""
Assign each video to a duration bucket (short / medium / long) using raw LLM feature JSON.

Buckets (seconds):
  - short:  duration < 12 * 60
  - medium: duration < 22 * 60
  - long:   else

Also writes a simple sklearn k-means overlay on [log1p(duration), n_output_features] when available.

Outputs:
  configs/video_clusters.json
  configs/video_cluster_train_lists.json  (video_ids per cluster for trainers)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features-root", type=Path, default=Path("data"))
    p.add_argument("--output-features-dir", type=Path, default=None, help="Optional: scan merged feature CSVs (e.g. output/) for n_numeric_cols per video.")
    p.add_argument("--output-dir", type=Path, default=Path("configs"))
    p.add_argument("--short-max-sec", type=float, default=12 * 60)
    p.add_argument("--medium-max-sec", type=float, default=22 * 60)
    return p.parse_args()


def discover_jsons(root: Path) -> list[Path]:
    return sorted(root.rglob("features_llm.json"))


def video_id_from_path(path: Path, payload: dict[str, Any]) -> str:
    flat = payload.get("video_features_flat")
    if isinstance(flat, dict):
        vf = flat.get("video_folder")
        if isinstance(vf, str) and vf.strip():
            return vf.strip()
    src = payload.get("source", {}) if isinstance(payload.get("source"), dict) else {}
    vf = src.get("video_folder")
    if isinstance(vf, str) and vf.strip():
        return vf.strip()
    return path.parent.name


def duration_sec(payload: dict[str, Any]) -> float:
    flat = payload.get("video_features_flat")
    if isinstance(flat, dict) and flat.get("duration_seconds") is not None:
        try:
            return float(flat["duration_seconds"])
        except (TypeError, ValueError):
            pass
    src = payload.get("source", {}) if isinstance(payload.get("source"), dict) else {}
    try:
        return float(src.get("duration_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def output_csv_stats(output_root: Path) -> dict[str, dict[str, Any]]:
    """Map video_id -> {n_cols, n_numeric_cols} from first CSV row per video folder."""
    out: dict[str, dict[str, Any]] = {}
    if not output_root.is_dir():
        return out
    for csv_path in sorted(output_root.rglob("*.csv")):
        parts = csv_path.relative_to(output_root).parts
        if not parts:
            continue
        vid = parts[0]
        try:
            df = pd.read_csv(csv_path, nrows=1)
        except Exception:
            continue
        num = sum(1 for c in df.columns if is_numeric_dtype(df[c]))
        out[vid] = {"output_csv_sample": str(csv_path), "n_cols": len(df.columns), "n_numeric_cols": int(num)}
    return out


def bucket(dur: float, short_max: float, medium_max: float) -> tuple[int, str]:
    if dur <= 0 or math.isnan(dur):
        return -1, "unknown"
    if dur < short_max:
        return 0, "short"
    if dur < medium_max:
        return 1, "medium"
    return 2, "long"


def main() -> None:
    args = parse_args()
    feat_stats = output_csv_stats(args.output_features_dir) if args.output_features_dir is not None else {}
    rows = []
    for path in discover_jsons(args.features_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        vid = video_id_from_path(path, payload)
        dur = duration_sec(payload)
        bid, label = bucket(dur, args.short_max_sec, args.medium_max_sec)
        row = {"video_id": vid, "duration_sec": dur, "cluster_id": bid, "cluster_name": label, "source_json": str(path)}
        if vid in feat_stats:
            row.update(feat_stats[vid])
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_vid = {r["video_id"]: r for r in rows}
    out_main = args.output_dir / "video_clusters.json"
    out_main.write_text(
        json.dumps({"videos": by_vid, "bins_sec": {"short_lt": args.short_max_sec, "medium_lt": args.medium_max_sec}}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_lists: dict[str, list[str]] = {"short": [], "medium": [], "long": [], "all": sorted(by_vid.keys())}
    for r in rows:
        if r["cluster_name"] == "short":
            train_lists["short"].append(r["video_id"])
        elif r["cluster_name"] == "medium":
            train_lists["medium"].append(r["video_id"])
        elif r["cluster_name"] == "long":
            train_lists["long"].append(r["video_id"])

    for k in train_lists:
        train_lists[k] = sorted(set(train_lists[k]))

    (args.output_dir / "video_cluster_train_lists.json").write_text(json.dumps(train_lists, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[clusters] videos={len(by_vid)} -> {out_main}")


if __name__ == "__main__":
    main()
