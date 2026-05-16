#!/usr/bin/env python3
"""
Train one multimodal model per content cluster
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from train.clustering.specialist_utils import add_common_train_args, default_output_base, rel_path, resolve_path, run_command, train_command, write_ids, write_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-content-cluster multimodal LSTM or Transformer specialists.")
    p.add_argument("--arch", choices=["lstm", "transformer"], default="lstm", help="Multimodal backbone (same train_multimodal_seq.py as global experiments).")
    p.add_argument("--clusters-json", type=Path, default=Path("analysis/video_clustering/retention/clusters.json"))
    add_common_train_args(p, output_base=None, d_model=256, n_layers=3, epochs=600, batch_size=16)
    p.add_argument("--n-heads", type=int, default=4, help="Transformer only (--arch transformer).")
    p.add_argument("--d-ff", type=int, default=512, help="Transformer only (--arch transformer).")
    p.add_argument("--min-videos", type=int, default=5, help="Skip clusters with fewer videos.")
    p.add_argument("--run-clustering-first", action="store_true", help="Run analysis/video_clustering.py before training (writes clusters.json + viz).")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--embeddings-dir", type=Path, default=Path("embeddings"))
    p.add_argument("--features-output-dir", type=Path, default=Path("output"))
    p.add_argument("--cluster-out-root", type=Path, default=Path("analysis/video_clustering"), help="--out-root for analysis/video_clustering.py")
    p.add_argument("--cluster-min-k", type=int, default=6, help="Passed to video_clustering.py when --run-clustering-first.")
    p.add_argument("--cluster-max-k", type=int, default=8, help="Passed to video_clustering.py when --run-clustering-first.")
    p.add_argument("--clustering-strategy", default="retention", help="Passed to video_clustering.py --strategy (e.g. retention = pooled multimodal embeddings).")
    return p.parse_args()


def group_video_ids_by_content_cluster(clusters: dict) -> dict[int, list[str]]:
    cluster2videos: dict[int, list[str]] = {}
    for video_id, video_metadata in clusters["videos"].items():
        cluster2videos.setdefault(int(video_metadata["cluster_id"]), []).append(str(video_id))
    return cluster2videos


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_base = args.output_base or default_output_base(args.arch)

    if args.run_clustering_first:
        cluster_py = root / "analysis" / "video_clustering.py"
        cmd_cluster = [
            sys.executable,
            str(cluster_py),
            "--data-dir",
            str(args.data_dir),
            "--embeddings-dir",
            str(args.embeddings_dir),
            "--output-dir",
            str(args.features_output_dir),
            "--out-root",
            str(args.cluster_out_root),
            "--min-k",
            str(args.cluster_min_k),
            "--max-k",
            str(args.cluster_max_k),
            "--strategy",
            str(args.clustering_strategy),
        ]
        run_command(cmd_cluster, root)

    path = resolve_path(args.clusters_json, root).resolve()
    if not path.is_file():
        raise SystemExit(f"Missing clusters file: {path} (run analysis/video_clustering.py first)")

    cluster2videos = group_video_ids_by_content_cluster(json.loads(path.read_text(encoding="utf-8")))
    meta: dict = {"arch": args.arch, "clusters": {}, "clusters_json": str(path), "min_videos": args.min_videos}

    for curClust in sorted(cluster2videos.keys()):
        ids = sorted(cluster2videos[curClust])
        if len(ids) < args.min_videos:
            print(f"cluster_id={curClust} videos={len(ids)} (need at least {args.min_videos} videos)")
            continue
        out = (root / output_base / f"{curClust}").resolve()
        list_file = out / "train_video_ids.txt"
        write_ids(list_file, ids)
        run_command(train_command(args, root, arch=args.arch, output_dir=out, list_file=list_file, extra_args=["--n-heads", str(args.n_heads), "--d-ff", str(args.d_ff)] if args.arch == "transformer" else []), root)
        meta["clusters"][str(curClust)] = {"out_dir": rel_path(out, root), "n_train_videos": len(ids), "list_file": rel_path(list_file, root)}
    write_meta(root / output_base / "cluster_runs_meta.json", meta)

if __name__ == "__main__":
    main()