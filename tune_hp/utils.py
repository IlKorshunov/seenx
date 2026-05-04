"""
shared tuning helpers.
data, metrics, Optuna paths, JSON output.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_EXCLUDE_COLS = frozenset({"retention", "frame", "time"})
STUDIES_DIR = Path(__file__).resolve().parent / "studies"
BEST_PARAMS_DIR = Path(__file__).resolve().parent / "best_params"
CUDA_DEVICE = "cuda"
RANDOM_SEED = 42
CATBOOST_LOSS = "RMSE"
CATBOOST_EVAL = "MAE"
SMOOTH_L1_BETA = 1.0


def load_video_frames(features_dir: str = "output") -> dict[str, pd.DataFrame]:
    csvs = sorted(glob.glob(os.path.join(features_dir, "*_features.csv")))
    if not csvs:
        raise FileNotFoundError(f"No *_features.csv in {features_dir}")

    video_frames: dict[str, pd.DataFrame] = {}
    for path in csvs:
        video_id = os.path.basename(path).replace("_features.csv", "")
        frame = pd.read_csv(path, index_col=0)
        if "retention" not in frame.columns:
            continue
        video_frames[video_id] = frame.dropna(subset=["retention"])

    if not video_frames:
        raise RuntimeError("No videos with retention column found")
    return video_frames


def get_feature_cols(video_frames: dict[str, pd.DataFrame]) -> list[str]:
    sample = next(iter(video_frames.values()))
    return [column for column in sample.columns if column not in FEATURE_EXCLUDE_COLS]


def loo_video_splits(video_ids: list[str]) -> list[tuple[list[str], str]]:
    return [([video_id for video_id in video_ids if video_id != held_out], held_out) for held_out in video_ids]


def load_tuning_data(features_dir: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    video_frames = load_video_frames(features_dir)
    feature_cols = get_feature_cols(video_frames)
    print(f"Loaded {len(video_frames)} videos, {len(feature_cols)} features")
    return video_frames, feature_cols


def mean_loo_score(video_frames: dict[str, pd.DataFrame], score_fold: Callable[[list[str], str, int], float]) -> float:
    fold_scores = [score_fold(train_ids, val_id, fold_idx) for fold_idx, (train_ids, val_id) in enumerate(loo_video_splits(sorted(video_frames.keys())))]
    return float(np.mean(fold_scores))


def build_xy(video_frames: dict[str, pd.DataFrame], video_ids: list[str], feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.concat([video_frames[video_id] for video_id in video_ids], ignore_index=True)
    features = frame[feature_cols].astype(float).fillna(0)
    target = frame["retention"].values.astype(float)
    return features, target


def compute_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((prediction - target) ** 2))
    mae = float(np.mean(np.abs(prediction - target)))
    ss_res = float(np.sum((target - prediction) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-9)
    return {"mse": round(mse, 6), "mae": round(mae, 6), "r2": round(r2, 6)}


def smooth_l1_loss_np(target: np.ndarray, prediction: np.ndarray, beta: float = SMOOTH_L1_BETA) -> float:
    abs_error = np.abs(prediction - target)
    loss = np.where(abs_error < beta, 0.5 * abs_error**2 / beta, abs_error - 0.5 * beta)
    return float(np.mean(loss))


def rmse_np(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def ensemble_loss(target: np.ndarray, prediction: np.ndarray) -> float:
    return 0.5 * rmse_np(target, prediction) + 0.5 * smooth_l1_loss_np(target, prediction)


def save_best_params(name: str, params: dict[str, Any], metrics: dict[str, float]) -> Path:
    BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    out = BEST_PARAMS_DIR / f"{name}.json"
    payload = {"params": params, "metrics": metrics}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return out


def load_best_params(name: str) -> dict[str, Any]:
    path = BEST_PARAMS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No best params found at {path}")
    return json.loads(path.read_text())


def get_study_path(name: str) -> str:
    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{STUDIES_DIR / name}.db"


def create_study(name: str, study_name: str, resume: bool = True, pruner: optuna.pruners.BasePruner | None = None, sampler: optuna.samplers.BaseSampler | None = None) -> optuna.Study:
    return optuna.create_study(study_name=study_name, storage=get_study_path(name), direction="minimize", load_if_exists=resume, pruner=pruner, sampler=sampler)


def optimize_study(study: optuna.Study, objective: Callable[[optuna.Trial], float], n_trials: int) -> float:
    started_at = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return time.time() - started_at


def print_best_trial(study: optuna.Study, metric_name: str, elapsed: float | None = None) -> None:
    elapsed_text = f" ({elapsed:.0f}s total)" if elapsed is not None else ""
    print(f"{os.linesep}Best trial #{study.best_trial.number}{elapsed_text}:")
    print(f"{metric_name} = {study.best_value:.4f}")
    print_params(study.best_params)


def save_and_report(name: str, params: dict[str, Any], metrics: dict[str, float]) -> Path:
    path = save_best_params(name, params, metrics)
    print(f"{os.linesep}Best params saved to {path}")
    return path


def print_lines(*lines: object) -> None:
    print(os.linesep.join(str(line) for line in lines))


def print_params(params: dict[str, Any], indent: str = "  ") -> None:
    for key, value in params.items():
        value_str = f"{value:.6g}" if isinstance(value, float) else str(value)
        print(f"{indent}{key} = {value_str}")
