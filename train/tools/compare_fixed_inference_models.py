from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train.common.train_utils import anchor_mean as _anchor_mean
from train.common.train_utils import clamp01 as _clamp01
from train.common.train_utils import curve_metrics as _metrics
from train.common.train_utils import point_col as _point_col
from train.common.train_utils import safe_float as _safe_float
from train.common.train_utils import tail_mean as _tail_mean
from train.loo.train_retention_hybrid_loo import run_experiment as run_hybrid
from train.loo.train_retention_local_knn_loo import run_experiment as run_local_knn
from train.loo.train_retention_ranker_loo import run_experiment as run_ranker
from train.loo.train_retention_regressor_loo import build_rows_with_targets_source, select_train_test
from train.loo.train_retention_regressor_loo import run_experiment as run_regressor
from train.loo.train_retention_shape_only_loo import run_experiment as run_shape_only
from train.loo.train_retention_stacked_loo import run_experiment as run_stacked


SNAPSHOT_DIR = "drive_snapshot_90"
ROOT_FOLDER_ID = "1aIqGRHTsO9kNBrOXRRz9XV8kD0Ru8zSV"
ENV_FILE = ".env"

LIMIT_VIDEOS = 90
TRAIN_VIDEOS = 89
CURVE_POINTS = 100

EVAL_VIDEO_FOLDER = "DhFuAhFMvms"
EVAL_DRIVE_FILE_ID = ""

RANDOM_SEED = 42
OUTPUT_ROOT = "fixed_inference_comparison"

BASELINE_NORM_ANCHOR_POINTS = 1
BASELINE_TAIL_ANCHOR_POINTS = 2


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value).strip())
    return cleaned or "unknown_eval_video"


def _extract_true_curve(pred_path: Path) -> np.ndarray:
    df = pd.read_csv(pred_path)
    if "true_retention" not in df.columns:
        raise RuntimeError(f"В {pred_path} нет колонки true_retention")
    return df["true_retention"].to_numpy(dtype=float)


def _extract_pred_curve(pred_path: Path) -> np.ndarray:
    df = pd.read_csv(pred_path)
    if "pred_retention" in df.columns:
        return df["pred_retention"].to_numpy(dtype=float)
    if "pred_retention_norm" in df.columns:
        return df["pred_retention_norm"].to_numpy(dtype=float)
    raise RuntimeError(f"В {pred_path} нет колонки pred_retention/pred_retention_norm")


def _build_train_baseline_curve(train_df: pd.DataFrame, curve_points: int) -> np.ndarray:
    curves: list[np.ndarray] = []
    norm_curves: list[np.ndarray] = []
    levels: list[float] = []
    tails: list[float] = []

    for _, row in train_df.iterrows():
        curve = np.array([_safe_float(row[_point_col(i)], 0.0) for i in range(curve_points)], dtype=float)
        curve = _clamp01(curve)
        curves.append(curve)

        level = _anchor_mean(curve, BASELINE_NORM_ANCHOR_POINTS)
        tail = _tail_mean(curve, BASELINE_TAIL_ANCHOR_POINTS)
        levels.append(level)
        tails.append(min(level, tail))

        denom = max(level, 1e-6)
        norm = _clamp01(curve / denom)
        if norm.size > 0:
            norm[0] = 1.0
        norm_curves.append(norm)

    if not curves:
        raise RuntimeError("Пустой train_df: baseline построить нельзя")

    norm_mean = _clamp01(np.mean(np.vstack(norm_curves), axis=0))
    if norm_mean.size > 0:
        norm_mean[0] = 1.0
    level_mean = float(np.mean(np.array(levels, dtype=float)))
    tail_mean = float(np.mean(np.array(tails, dtype=float)))
    tail_mean = min(level_mean, tail_mean)

    baseline = tail_mean + (level_mean - tail_mean) * norm_mean
    baseline = _clamp01(baseline)
    return baseline


def _save_comparison_plot(output_root: Path, curves: dict[str, np.ndarray], y_true: np.ndarray, video_id: str) -> None:
    x = np.linspace(0.0, 1.0, len(y_true))
    plt.figure(figsize=(11, 6))
    plt.plot(x, y_true, marker="o", linewidth=2.5, label="true")
    for name, curve in curves.items():
        plt.plot(x, curve, marker="o", linewidth=2, label=name)
    plt.title(f"Fixed Holdout Comparison ({video_id}): baseline vs models")
    plt.xlabel("Video progress (0..1)")
    plt.ylabel("Retention")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "comparison_curve_plot.png", dpi=160)
    plt.close()


def _save_error_by_point_plot(output_root: Path, curves: dict[str, np.ndarray], y_true: np.ndarray, video_id: str) -> None:
    x = np.linspace(0.0, 1.0, len(y_true))
    plt.figure(figsize=(11, 6))
    for name, curve in curves.items():
        err = np.abs(curve - y_true)
        plt.plot(x, err, marker="o", linewidth=2, label=name)
    plt.title(f"Absolute Error by Point ({video_id})")
    plt.xlabel("Video progress (0..1)")
    plt.ylabel("|pred - true|")
    plt.xlim(0.0, 1.0)
    plt.ylim(bottom=0.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "comparison_abs_error_by_point.png", dpi=160)
    plt.close()


def _save_metrics_bar_plot(output_root: Path, metrics_df: pd.DataFrame, video_id: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    fig.suptitle(f"Model Metrics Comparison ({video_id})", y=1.02)

    data_rmse = metrics_df.sort_values("rmse", ascending=True)
    axes[0].bar(data_rmse["model"], data_rmse["rmse"], color="#1f77b4")
    axes[0].set_title("RMSE (lower is better)")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)

    data_mae = metrics_df.sort_values("mae", ascending=True)
    axes[1].bar(data_mae["model"], data_mae["mae"], color="#ff7f0e")
    axes[1].set_title("MAE (lower is better)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)

    data_spear = metrics_df.sort_values("spearman", ascending=False)
    axes[2].bar(data_spear["model"], data_spear["spearman"], color="#2ca02c")
    axes[2].set_title("Spearman (higher is better)")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].grid(axis="y", alpha=0.25)

    data_spike = metrics_df.sort_values("spike_rmse", ascending=True)
    axes[3].bar(data_spike["model"], data_spike["spike_rmse"], color="#9467bd")
    axes[3].set_title("Spike RMSE on dY (lower is better)")
    axes[3].tick_params(axis="x", rotation=35)
    axes[3].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_root / "comparison_metrics_bars.png", dpi=160)
    plt.close(fig)


def _ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def main() -> None:
    eval_key_raw = EVAL_VIDEO_FOLDER if str(EVAL_VIDEO_FOLDER).strip() else EVAL_DRIVE_FILE_ID
    eval_key = _safe_name(eval_key_raw if str(eval_key_raw).strip() else "unknown_eval_video")
    output_root = Path(OUTPUT_ROOT) / eval_key
    output_root.mkdir(parents=True, exist_ok=True)

    # Общий конфиг для train/test split.
    split_args = _ns(limit_videos=LIMIT_VIDEOS, train_videos=TRAIN_VIDEOS, eval_video_folder=EVAL_VIDEO_FOLDER, eval_drive_file_id=EVAL_DRIVE_FILE_ID)

    rows = build_rows_with_targets_source(root_folder_id=ROOT_FOLDER_ID, env_file=Path(ENV_FILE), curve_points=CURVE_POINTS, snapshot_dir=Path(SNAPSHOT_DIR))
    all_df, train_df, test_df = select_train_test(rows, split_args)
    _ = all_df  # explicit: useful for future extensions
    y_true = np.array([_safe_float(test_df.iloc[0][_point_col(i)], 0.0) for i in range(CURVE_POINTS)], dtype=float)
    y_true = _clamp01(y_true)

    # 1) Baseline
    baseline_curve = _build_train_baseline_curve(train_df=train_df, curve_points=CURVE_POINTS)

    # 2) Regressor
    reg_dir = output_root / "regressor"
    _ = run_regressor(
        _ns(
            env_file=ENV_FILE,
            snapshot_dir=SNAPSHOT_DIR,
            root_folder_id=ROOT_FOLDER_ID,
            limit_videos=LIMIT_VIDEOS,
            train_videos=TRAIN_VIDEOS,
            curve_points=CURVE_POINTS,
            eval_video_folder=EVAL_VIDEO_FOLDER,
            eval_drive_file_id=EVAL_DRIVE_FILE_ID,
            output_dir=str(reg_dir),
            iterations=700,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_SEED,
            delta_blend=0.65,
            delta_max_step=0.25,
        )
    )
    reg_curve = _extract_pred_curve(reg_dir / "holdout_prediction_vs_true.csv")

    # 3) Ranker
    rank_dir = output_root / "ranker"
    _ = run_ranker(
        _ns(
            env_file=ENV_FILE,
            snapshot_dir=SNAPSHOT_DIR,
            root_folder_id=ROOT_FOLDER_ID,
            limit_videos=LIMIT_VIDEOS,
            train_videos=TRAIN_VIDEOS,
            eval_video_folder=EVAL_VIDEO_FOLDER,
            eval_drive_file_id=EVAL_DRIVE_FILE_ID,
            output_dir=str(rank_dir),
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_SEED,
            curve_points=CURVE_POINTS,
        )
    )
    rank_curve = _extract_pred_curve(rank_dir / "holdout_prediction_vs_true.csv")

    # 4) Hybrid
    hyb_dir = output_root / "hybrid"
    _ = run_hybrid(
        _ns(
            env_file=ENV_FILE,
            snapshot_dir=SNAPSHOT_DIR,
            root_folder_id=ROOT_FOLDER_ID,
            limit_videos=LIMIT_VIDEOS,
            train_videos=TRAIN_VIDEOS,
            curve_points=CURVE_POINTS,
            eval_video_folder=EVAL_VIDEO_FOLDER,
            eval_drive_file_id=EVAL_DRIVE_FILE_ID,
            output_dir=str(hyb_dir),
            ranker_iterations=600,
            ranker_learning_rate=0.05,
            ranker_depth=6,
            level_iterations=600,
            level_learning_rate=0.05,
            level_depth=6,
            random_seed=RANDOM_SEED,
            ensemble_seeds="42,52,62",
            shape_ranker_weight=0.7,
            shape_anchor_points=1,
            level_anchor_points=2,
            tail_anchor_points=2,
            disable_tail_floor=False,
            enable_affine_calibration=True,
        )
    )
    hyb_curve = _extract_pred_curve(hyb_dir / "holdout_prediction_vs_true.csv")

    # 5) Stacked
    st_dir = output_root / "stacked"
    _ = run_stacked(
        _ns(
            env_file=ENV_FILE,
            snapshot_dir=SNAPSHOT_DIR,
            root_folder_id=ROOT_FOLDER_ID,
            limit_videos=LIMIT_VIDEOS,
            train_videos=TRAIN_VIDEOS,
            curve_points=CURVE_POINTS,
            eval_video_folder=EVAL_VIDEO_FOLDER,
            eval_drive_file_id=EVAL_DRIVE_FILE_ID,
            output_dir=str(st_dir),
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_SEED,
            oof_folds=5,
            meta_l2=0.02,
            shape_anchor_points=1,
            level_anchor_points=2,
            tail_anchor_points=2,
        )
    )
    st_curve = _extract_pred_curve(st_dir / "holdout_prediction_vs_true.csv")

    # 6) Local-kNN residual (feature-similar neighbors + local shape residuals)
    lk_dir = output_root / "local_knn"
    _ = run_local_knn(
        _ns(
            env_file=ENV_FILE,
            snapshot_dir=SNAPSHOT_DIR,
            root_folder_id=ROOT_FOLDER_ID,
            limit_videos=LIMIT_VIDEOS,
            train_videos=TRAIN_VIDEOS,
            curve_points=CURVE_POINTS,
            eval_video_folder=EVAL_VIDEO_FOLDER,
            eval_drive_file_id=EVAL_DRIVE_FILE_ID,
            output_dir=str(lk_dir),
            neighbors_k=12,
            distance_temperature=1.0,
            residual_strength=0.70,
            spike_gain=1.10,
            smooth_window=7,
            auto_tune=True,
            tune_folds=5,
        )
    )
    lk_curve = _extract_pred_curve(lk_dir / "holdout_prediction_vs_true.csv")

    # 7) Shape-only (учим только форму: дельты и кривизну, без уровня)
    sh_dir = output_root / "shape_only"
    _ = run_shape_only(
        _ns(
            env_file=ENV_FILE,
            snapshot_dir=SNAPSHOT_DIR,
            root_folder_id=ROOT_FOLDER_ID,
            limit_videos=LIMIT_VIDEOS,
            train_videos=TRAIN_VIDEOS,
            curve_points=CURVE_POINTS,
            eval_video_folder=EVAL_VIDEO_FOLDER,
            eval_drive_file_id=EVAL_DRIVE_FILE_ID,
            output_dir=str(sh_dir),
            iterations=700,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_SEED,
            shape_anchor_points=1,
            fixed_anchor_value=1.0,
            delta_max_step=0.24,
            curvature_max_step=0.18,
            curvature_blend=0.78,
            shape_max=1.30,
            spike_sensitivity=1.45,
            spike_threshold_quantile=0.70,
            spike_max_amplify=1.8,
        )
    )
    sh_curve = _extract_pred_curve(sh_dir / "holdout_prediction_vs_true.csv")

    rows_out: list[dict[str, float | str]] = []
    curve_map = {"baseline": baseline_curve, "regressor": reg_curve, "ranker": rank_curve, "hybrid": hyb_curve, "stacked": st_curve, "local_knn": lk_curve, "shape_only": sh_curve}
    for name, curve in curve_map.items():
        rows_out.append({"model": name, **_metrics(curve, y_true)})

    metrics_df = pd.DataFrame(rows_out).sort_values("rmse", ascending=True).reset_index(drop=True)
    metrics_df.to_csv(output_root / "comparison_metrics.csv", index=False)

    curves = {
        "baseline": baseline_curve,
        "regressor": _clamp01(reg_curve),
        "ranker": _clamp01(rank_curve),
        "hybrid": _clamp01(hyb_curve),
        "stacked": _clamp01(st_curve),
        "local_knn": _clamp01(lk_curve),
        "shape_only": _clamp01(sh_curve),
    }
    _save_comparison_plot(output_root=output_root, curves=curves, y_true=y_true, video_id=eval_key)
    _save_error_by_point_plot(output_root=output_root, curves=curves, y_true=y_true, video_id=eval_key)
    _save_metrics_bar_plot(output_root=output_root, metrics_df=metrics_df, video_id=eval_key)

    summary = {
        "snapshot_dir": SNAPSHOT_DIR,
        "eval_video_folder": EVAL_VIDEO_FOLDER,
        "eval_drive_file_id": EVAL_DRIVE_FILE_ID,
        "limit_videos": LIMIT_VIDEOS,
        "train_videos": TRAIN_VIDEOS,
        "curve_points": CURVE_POINTS,
        "output_root": str(output_root),
        "models": rows_out,
    }
    (output_root / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Fixed Inference Comparison Complete ===")
    print(f"Output: {output_root}")
    print("Top by RMSE:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
