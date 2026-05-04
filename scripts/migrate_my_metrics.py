#!/usr/bin/env python3
"""
Migrate existing my_metrics files to the new per-video directory structure.

Old layout:
  my_metrics/{video_id}.png
  my_metrics/baseline_{video_id}.png
  my_metrics/pred_{video_id}.png
  my_metrics/new_features_{video_id}.png
  my_metrics/error_analysis/residual_{i}_{video_id}.png

New layout:
  my_metrics/videos/{video_id}/summary/overview.png
  my_metrics/videos/{video_id}/baseline/comparison.png
  my_metrics/videos/{video_id}/prediction/pred.png
  my_metrics/videos/{video_id}/new_features/overview.png
  my_metrics/videos/{video_id}/residual/residual.png

Usage:
  python scripts/migrate_my_metrics.py [--dry-run] [--metrics-dir my_metrics]
"""

import argparse
import shutil
from pathlib import Path


def migrate(metrics_dir: str = "my_metrics", dry_run: bool = False) -> None:
    metrics_dir = Path(metrics_dir)
    if not metrics_dir.exists():
        print(f"Directory {metrics_dir} does not exist")
        return

    moved = 0

    # Known non-video root-level PNGs (keep in place)
    KEEP_ROOT = {"feature_importance.png", "training_curves.png", "transformer_training_curves.png"}

    # 1. {video_id}.png -> videos/{video_id}/summary/overview.png
    for f in metrics_dir.glob("*.png"):
        if f.name in KEEP_ROOT or f.stem.startswith(("baseline_", "pred_", "new_features_")):
            continue
        # Remaining PNGs at root are summarize output: {video_id}.png
        vid = f.stem
        dst = metrics_dir / "videos" / vid / "summary" / "overview.png"
        if dry_run:
            print(f"Would move {f} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst))
            print(f"Moved {f} -> {dst}")
        moved += 1

    # 2. baseline_{video_id}.png -> videos/{video_id}/baseline/comparison.png
    for f in metrics_dir.glob("baseline_*.png"):
        vid = f.stem.replace("baseline_", "")
        dst = metrics_dir / "videos" / vid / "baseline" / "comparison.png"
        if dry_run:
            print(f"Would move {f} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst))
            print(f"Moved {f} -> {dst}")
        moved += 1

    # 3. pred_{video_id}.png -> videos/{video_id}/prediction/pred.png
    for f in metrics_dir.glob("pred_*.png"):
        if "transformer" in f.stem:
            vid = f.stem.replace("pred_transformer_", "")
            dst = metrics_dir / "videos" / vid / "prediction" / "transformer_pred.png"
        else:
            vid = f.stem.replace("pred_", "")
            dst = metrics_dir / "videos" / vid / "prediction" / "pred.png"
        if dry_run:
            print(f"Would move {f} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst))
            print(f"Moved {f} -> {dst}")
        moved += 1

    # 4. new_features_{video_id}.png -> videos/{video_id}/new_features/overview.png
    for f in metrics_dir.glob("new_features_*.png"):
        vid = f.stem.replace("new_features_", "")
        dst = metrics_dir / "videos" / vid / "new_features" / "overview.png"
        if dry_run:
            print(f"Would move {f} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst))
            print(f"Moved {f} -> {dst}")
        moved += 1

    # 5. error_analysis/residual_{i}_{video_id}.png -> videos/{video_id}/residual/residual.png
    err_dir = metrics_dir / "error_analysis"
    if err_dir.exists():
        for f in err_dir.glob("residual_*.png"):
            # residual_0_3G6JrAq-w4M.png -> vid = 3G6JrAq-w4M
            parts = f.stem.split("_", 2)
            if len(parts) >= 3:
                vid = parts[2]
                dst = metrics_dir / "videos" / vid / "residual" / "residual.png"
                if dry_run:
                    print(f"Would move {f} -> {dst}")
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dst))
                    print(f"Moved {f} -> {dst}")
                moved += 1

    print(f"\nTotal: {moved} files migrated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate my_metrics to new structure")
    parser.add_argument("--metrics-dir", default="my_metrics", help="Path to my_metrics")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    args = parser.parse_args()
    migrate(args.metrics_dir, args.dry_run)
