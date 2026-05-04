"""
orchestrator: runs all tuning steps and prints a comparison.
python tune_hp/run_all.py --features_dir output --skip_transformer --skip_features
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from utils import print_lines, print_params


TUNE_DIR = Path(__file__).resolve().parent
BEST_DIR = TUNE_DIR / "best_params"


def _run_script(name: str, args: list[str]) -> bool:
    script = TUNE_DIR / name
    cmd = [sys.executable, str(script)] + args
    print(f"Running: {' '.join(cmd)}")
    started_at = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - started_at
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"[{status}] {name} finished in {elapsed:.0f}s")
    return result.returncode == 0


def _print_summary():
    print_lines("SUMMARY")
    for name in ["catboost", "transformer", "ensemble", "features"]:
        path = BEST_DIR / f"{name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            metrics = data.get("metrics", {})
            params = data.get("params", {})
            score = metrics.get("loo_mae", metrics.get("mae", metrics.get("loo_smooth_l1", metrics.get("ensemble_loss", "N/A"))))
            print(f"{name:15s}  score = {score}")
            print_params(params, indent="    ")
        else:
            print_lines(f"{name:15s}  (not tuned)")


def main():
    parser = argparse.ArgumentParser(description="Run all HP tuning steps")
    parser.add_argument("--features_dir", default="output")
    parser.add_argument("--cb_trials", type=int, default=100, help="CatBoost trials")
    parser.add_argument("--tf_trials", type=int, default=50, help="Transformer trials")
    parser.add_argument("--tf_max_epochs", type=int, default=50, help="Max epochs per transformer trial")
    parser.add_argument("--ens_trials", type=int, default=50, help="Ensemble trials")
    parser.add_argument("--feat_trials", type=int, default=40, help="Feature tuning trials")
    parser.add_argument("--skip_transformer", action="store_true")
    parser.add_argument("--skip_ensemble", action="store_true")
    parser.add_argument("--skip_features", action="store_true")
    args = parser.parse_args()

    started_at = time.time()
    results = {}

    results["catboost"] = _run_script("tune_catboost.py", ["--features_dir", args.features_dir, "--n_trials", str(args.cb_trials)])

    if not args.skip_transformer:
        results["transformer"] = _run_script(
            "tune_transformer.py", ["--features_dir", args.features_dir, "--n_trials", str(args.tf_trials), "--max_epochs", str(args.tf_max_epochs)]
        )

    if not args.skip_ensemble:
        results["ensemble"] = _run_script("tune_ensemble.py", ["--features_dir", args.features_dir, "--n_trials", str(args.ens_trials)])

    if not args.skip_features:
        results["features"] = _run_script("tune_features.py", ["--features_dir", args.features_dir, "--n_trials", str(args.feat_trials)])

    total = time.time() - started_at
    print(f"{os.linesep}Total tuning time: {total:.0f}s ({total / 60:.1f} min)")

    _print_summary()

    print_lines("optuna-dashboard sqlite:///tune_hp/studies/catboost.db")
    print_lines("optuna-dashboard sqlite:///tune_hp/studies/transformer.db")


if __name__ == "__main__":
    main()
