from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def add_common_train_args(parser: argparse.ArgumentParser, *, output_base: Path | None, d_model: int, n_layers: int, epochs: int, batch_size: int) -> None:
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-base", type=Path, default=output_base)
    parser.add_argument("--output-dir-features", default="output")
    parser.add_argument("--snapshot-dir", default="data")
    parser.add_argument("--embeddings-root", default="embeddings")
    parser.add_argument("--val-first-n-output", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=d_model)
    parser.add_argument("--n-layers", type=int, default=n_layers)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tuned-params-json", type=Path, default=None, help="Optional; omit with --no-tuned to skip")
    parser.add_argument("--no-tuned", action="store_true")


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(ids)) + ("\n" if ids else ""), encoding="utf-8")


def rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def default_output_base(arch: str) -> Path:
    return Path(f"experiments/{arch}_exp/clusters")


def tuned_args(args: argparse.Namespace, root: Path, arch: str) -> list[str]:
    if args.no_tuned:
        return []
    tuned_json = args.tuned_params_json or Path(f"tune_hp/results/tune_multimodal_{arch}_best.json")
    tuned_json = resolve_path(tuned_json, root)
    return ["--tuned-params-json", str(tuned_json)] if tuned_json.is_file() else []


def env_train_args() -> list[str]:
    pairs = [
        ("CURVE_POINTS", "--curve-points", "0"),
        ("TIME_FEATURES", "--time-features", "none"),
        ("MIN_DURATION_SEC", "--min-duration-sec", "0"),
        ("MAX_DURATION_SEC", "--max-duration-sec", "0"),
    ]
    extra = []
    for env_key, arg_name, disabled in pairs:
        value = os.environ.get(env_key, disabled)
        if value != disabled:
            extra.extend([arg_name, value])
    return [*extra, "--patience", os.environ.get("PATIENCE", "30")]


def train_command(args: argparse.Namespace, root: Path, *, arch: str, output_dir: Path, list_file: Path, extra_args: list[str] | None = None, include_env: bool = True) -> list[str]:
    script = root / "train" / "transformer" / "train_multimodal_seq.py"
    if not script.is_file():
        script = root / "train" / "train_multimodal_seq.py"
    cmd = [
        sys.executable,
        str(script),
        "--arch",
        arch,
        "--output-dir",
        rel_path(output_dir, root),
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
        rel_path(list_file, root),
        *tuned_args(args, root, arch),
        *(extra_args or []),
    ]
    return [*cmd, *env_train_args()] if include_env else cmd


def run_command(cmd: list[str], root: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(root), check=True)


def write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] meta -> {path}")
