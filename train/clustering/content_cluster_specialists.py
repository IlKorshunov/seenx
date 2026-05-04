#!/usr/bin/env python3
"""
Train one multimodal model (LSTM or Transformer) per content cluster from
analysis/video_clustering/retention/clusters.json (embedding-based retention clustering from analysis/video_clustering.py --strategy retention).

Clusters with fewer than --min-videos are skipped (no training, no metrics).

Each video entry may use cluster_id (video_clustering.py) or kmeans_cluster_id (legacy).

Usage (usually via run_all_experiments.sh):
  python train/content_cluster_specialists.py --arch lstm --repo-root . ...
  python train/content_cluster_specialists.py --arch transformer ...  # omit --run-clustering-first if clusters.json is fresh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-content-cluster multimodal LSTM or Transformer specialists.")
    p.add_argument("--arch", choices=["lstm", "transformer"], default="lstm", help="Multimodal backbone (same train_multimodal_seq.py as global experiments).")
    p.add_argument("--clusters-json", type=Path, default=Path("analysis/video_clustering/retention/clusters.json"))
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent.parent)
    p.add_argument("--output-base", type=Path, default=None, help="Default: experiments/lstm_exp/content_cluster_specialists or transformer_exp/... by --arch")
    p.add_argument("--output-dir-features", default="output")
    p.add_argument("--snapshot-dir", default="data")
    p.add_argument("--embeddings-root", default="embeddings")
    p.add_argument("--val-first-n-output", type=int, default=10)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4, help="Transformer only (--arch transformer).")
    p.add_argument("--d-ff", type=int, default=512, help="Transformer only (--arch transformer).")
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-videos", type=int, default=5, help="Skip clusters with fewer videos.")
    p.add_argument("--run-clustering-first", action="store_true", help="Run analysis/video_clustering.py before training (writes clusters.json + viz).")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--embeddings-dir", type=Path, default=Path("embeddings"))
    p.add_argument("--features-output-dir", type=Path, default=Path("output"))
    p.add_argument("--cluster-out-root", type=Path, default=Path("analysis/video_clustering"), help="--out-root for analysis/video_clustering.py")
    p.add_argument("--cluster-min-k", type=int, default=6, help="Passed to video_clustering.py when --run-clustering-first.")
    p.add_argument("--cluster-max-k", type=int, default=8, help="Passed to video_clustering.py when --run-clustering-first.")
    p.add_argument("--clustering-strategy", default="retention", help="Passed to video_clustering.py --strategy (e.g. retention = pooled multimodal embeddings).")
    p.add_argument("--tuned-params-json", type=Path, default=Path("tune_hp/results/tune_multimodal_lstm_best.json"), help="Optional; omit with --no-tuned to skip")
    p.add_argument("--no-tuned", action="store_true")
    return p.parse_args()


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(ids)) + ("\n" if ids else ""), encoding="utf-8")


def group_videos_by_cluster(payload: dict) -> dict[int, list[str]]:
    videos = payload.get("videos") or {}
    by_c: dict[int, list[str]] = {}
    for vid, row in videos.items():
        if not isinstance(row, dict):
            continue
        cid = row.get("kmeans_cluster_id")
        if cid is None:
            cid = row.get("cluster_id")
        if cid is None:
            continue
        c = int(cid)
        by_c.setdefault(c, []).append(str(vid))
    return by_c


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root))
    except ValueError:
        return str(p.resolve())


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()

    output_base = args.output_base
    if output_base is None:
        output_base = Path("experiments/lstm_exp/content_cluster_specialists") if args.arch == "lstm" else Path("experiments/transformer_exp/content_cluster_specialists")

    if args.run_clustering_first:
        cluster_py = root / "analysis" / "video_clustering.py"
        if not cluster_py.is_file():
            raise SystemExit(f"[content_cluster_specialists] Missing {cluster_py}")
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
        print("[run]", " ".join(cmd_cluster))
        subprocess.run(cmd_cluster, cwd=str(root), check=True)

    path = (root / args.clusters_json).resolve() if not args.clusters_json.is_absolute() else args.clusters_json
    if not path.is_file():
        raise SystemExit(f"[content_cluster_specialists] Missing clusters file: {path} (run analysis/video_clustering.py first)")

    payload = json.loads(path.read_text(encoding="utf-8"))
    by_cluster = group_videos_by_cluster(payload)

    py = sys.executable
    script = root / "train" / "train_multimodal_seq.py"
    tuned_json = args.tuned_params_json
    if tuned_json is None and not args.no_tuned:
        rel = Path("tune_hp/results/tune_multimodal_lstm_best.json") if args.arch == "lstm" else Path("tune_hp/results/tune_multimodal_transformer_best.json")
        tuned_json = (root / rel).resolve()
    tuned: list[str] = []
    if not args.no_tuned and tuned_json is not None and tuned_json.is_file():
        tuned = ["--tuned-params-json", str(tuned_json)]

    meta: dict = {"arch": args.arch, "clusters": {}, "clusters_json": str(path), "min_videos": args.min_videos}

    for cid in sorted(by_cluster.keys()):
        ids = sorted(by_cluster[cid])
        if len(ids) < args.min_videos:
            print(f"[skip] cluster_id={cid} videos={len(ids)} (need >={args.min_videos})")
            continue
        out = (root / args.output_base / f"kmeans_{cid}").resolve()
        list_file = out / "train_video_ids.txt"
        write_ids(list_file, ids)
        cmd = [
            py,
            str(script),
            "--arch",
            args.arch,
            "--output-dir",
            _rel(out, root),
            "--output-dir-features",
            args.output_dir_features,
            "--snapshot-dir",
            args.snapshot_dir,
            "--embeddings-root",
            args.embeddings_root,
            "--val-first-n-output",
            str(args.val_first_n_output),
            "--d-model",
            str(args.d_model),
            "--n-layers",
            str(args.n_layers),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--train-video-ids-file",
            _rel(list_file, root),
            *tuned,
        ]
        if args.arch == "transformer":
            cmd.extend(["--n-heads", str(args.n_heads), "--d-ff", str(args.d_ff)])
        if os.environ.get("CURVE_POINTS", "0") != "0":
            cmd.extend(["--curve-points", os.environ["CURVE_POINTS"]])
        if os.environ.get("TIME_FEATURES", "none") != "none":
            cmd.extend(["--time-features", os.environ["TIME_FEATURES"]])
        if os.environ.get("MIN_DURATION_SEC", "0") != "0":
            cmd.extend(["--min-duration-sec", os.environ["MIN_DURATION_SEC"]])
        if os.environ.get("MAX_DURATION_SEC", "0") != "0":
            cmd.extend(["--max-duration-sec", os.environ["MAX_DURATION_SEC"]])
        cmd.extend(["--patience", os.environ.get("PATIENCE", "30")])
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, cwd=str(root), check=True)
        meta["clusters"][str(cid)] = {"out_dir": _rel(out, root), "n_train_videos": len(ids), "list_file": _rel(list_file, root)}

    meta_path = (root / output_base / "content_cluster_runs_meta.json").resolve()
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] meta -> {meta_path}")


if __name__ == "__main__":
    main()
