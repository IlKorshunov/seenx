#!/usr/bin/env python3
"""
Build duration clusters (short/medium/long) from video metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from train.clustering.specialist_utils import REPO_ROOT, resolve_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--features-root", type=Path, default=Path("data"))
    p.add_argument("--lists-json", type=Path, default=Path("configs/video_cluster_train_lists.json"), help="Where duration train lists are written.")
    p.add_argument("--clusters-json", type=Path, default=Path("configs/video_clusters.json"), help="Where per-video duration cluster metadata is written.")
    p.add_argument("--short-max-sec", type=float, default=12 * 60)
    p.add_argument("--medium-max-sec", type=float, default=22 * 60)
    return p.parse_args()


def video_id_from_payload(path: Path, payload: dict[str, Any]) -> str:
    block = payload.get("video_features_flat")
    if isinstance(block, dict) and isinstance(raw := block.get("video_folder"), str) and (video_id := raw.strip()):
        return video_id
    parent = path.parent
    if parent.name == "transcripts":
        return parent.parent.name
    return parent.name


def duration_sec(payload: dict[str, Any]) -> float:
    flat = payload.get("video_features_flat")
    if isinstance(flat, dict) and flat.get("duration_seconds") is not None:
        return float(flat["duration_seconds"])
    return 0.0


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def bucket(duration: float, short_max: float, medium_max: float) -> tuple[int, str]:
    return (0, "short") if duration < short_max else (1, "medium") if duration < medium_max else (2, "long")


def build_duration_clusters(features_root: Path, short_max: float, medium_max: float) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    videos: dict[str, dict[str, Any]] = {}
    train_lists: dict[str, list[str]] = {"short": [], "medium": [], "long": [], "all": []}
    for path in sorted(features_root.rglob("features_llm.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            video_id = video_id_from_payload(path, payload)
            duration = duration_sec(payload)
        except Exception:
            continue
        cluster_id, cluster_name = bucket(duration, short_max, medium_max)
        videos[video_id] = {"video_id": video_id, "duration_sec": duration, "cluster_id": cluster_id, "cluster_name": cluster_name, "source_json": str(path)}
        train_lists[cluster_name].append(video_id)
    return videos, {name: sorted(set(ids)) for name, ids in train_lists.items()}


def write_duration_clusters(args: argparse.Namespace, root: Path) -> dict[str, list[str]]:
    videos, train_lists = build_duration_clusters(resolve_path(args.features_root, root), args.short_max_sec, args.medium_max_sec)
    clusters_json = resolve_path(args.clusters_json, root)
    lists_json = resolve_path(args.lists_json, root)
    write_json(clusters_json, {"videos": videos, "tresh_sec": {"short_tr": args.short_max_sec, "medium_tr": args.medium_max_sec}})
    write_json(lists_json, train_lists)
    print(f"[clusters] videos={len(videos)} -> {clusters_json}")
    return train_lists

if __name__ == "__main__":
    args = parse_args()
    write_duration_clusters(args, args.repo_root.resolve())
