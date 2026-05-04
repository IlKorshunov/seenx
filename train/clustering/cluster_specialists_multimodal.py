#!/usr/bin/env python3
"""
Train one multimodal LSTM per duration cluster (short/medium/long) + reuse general model path.

Requires:
  configs/video_cluster_train_lists.json  (from train/compute_video_clusters.py)

Writes list files under experiment dir and invokes train_multimodal_seq.py per cluster.
Final weighted blend is left to inference (stack checkpoints with weights); see docstring
in ensemble section of README or use equal weights on val RMSE^-1.

Usage:
  python train/cluster_specialists_multimodal.py --epochs 200 --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lists-json", type=Path, default=Path("configs/video_cluster_train_lists.json"))
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent.parent)
    p.add_argument("--output-base", type=Path, default=Path("experiments/lstm_exp/cluster_specialists"))
    p.add_argument("--output-dir-features", default="output")
    p.add_argument("--snapshot-dir", default="data")
    p.add_argument("--embeddings-root", default="embeddings")
    p.add_argument("--val-first-n-output", type=int, default=10)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tuned-params-json", type=Path, default=Path("tune_hp/results/tune_multimodal_lstm_best.json"), help="Optional; omit with --no-tuned to skip")
    p.add_argument("--no-tuned", action="store_true")
    return p.parse_args()


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.repo_root
    data = json.loads(args.lists_json.read_text(encoding="utf-8"))
    py = sys.executable
    script = root / "train" / "train_multimodal_seq.py"
    tuned: list[str] = []
    if not args.no_tuned and args.tuned_params_json.is_file():
        tuned = ["--tuned-params-json", str(args.tuned_params_json)]

    meta = {"clusters": {}, "general_note": "Train global model separately (e.g. lstm_v3_multimodal)."}
    for name in ("short", "medium", "long"):
        ids = data.get(name, [])
        if len(ids) < 3:
            print(f"[skip] cluster={name} videos={len(ids)} (need >=3)")
            continue
        out = args.output_base / name
        list_file = out / "train_video_ids.txt"
        write_ids(list_file, ids)
        cmd = [
            py,
            str(script),
            "--arch",
            "lstm",
            "--output-dir",
            str(out),
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
            str(list_file),
            *tuned,
        ]
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, cwd=str(root), check=True)
        meta["clusters"][name] = {"out_dir": str(out), "n_train_videos": len(ids), "list_file": str(list_file)}

    meta_path = args.output_base / "cluster_runs_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] meta -> {meta_path}")


if __name__ == "__main__":
    main()
