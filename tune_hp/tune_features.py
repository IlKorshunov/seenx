"""
feature-weight tuning.
python tune_hp/tune_features.py --n_trials 40 --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.extractors.text.config import HOOK_ADDRESS_W, HOOK_CLAIM_W, HOOK_DENSITY_W, HOOK_NUMBERS_W, HOOK_QUESTION_W
from tune_hp.utils import CATBOOST_LOSS, RANDOM_SEED, build_xy, create_study, load_tuning_data, mean_loo_score, print_best_trial, save_and_report


HOOK_COMPONENTS = ("hook_score", "hook_has_address", "is_question")
HOOK_RESIDUAL_DEFAULT_W = HOOK_CLAIM_W + HOOK_NUMBERS_W + HOOK_DENSITY_W


def _apply_feature_overrides(video_frames: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.DataFrame]:
    modified = {}
    for video_id, frame in video_frames.items():
        frame = frame.copy()

        if set(HOOK_COMPONENTS) <= set(frame.columns):
            question_weight = params["hook_w_question"]
            address_weight = params["hook_w_address"]
            residual_weight = params["hook_w_residual"]
            original_score = float(frame["hook_score"].iloc[0])
            address_value = float(frame["hook_has_address"].iloc[0])
            question_value = float(frame["is_question"].max())
            residual_value = max(original_score - question_value * HOOK_QUESTION_W - address_value * HOOK_ADDRESS_W, 0.0) / HOOK_RESIDUAL_DEFAULT_W
            frame["hook_score"] = question_value * question_weight + address_value * address_weight + residual_value * residual_weight

        modified[video_id] = frame
    return modified


def objective(trial: optuna.Trial, video_frames: dict, feature_cols: list[str]) -> float:
    params = {
        "hook_w_question": trial.suggest_float("hook_w_question", 0.05, 0.5),
        "hook_w_address": trial.suggest_float("hook_w_address", 0.05, 0.3),
        "hook_w_residual": trial.suggest_float("hook_w_residual", 0.2, 1.0),
    }

    modified_frames = _apply_feature_overrides(video_frames, params)

    def score_fold(train_ids: list[str], val_id: str, _fold_idx: int) -> float:
        train_features, train_target = build_xy(modified_frames, train_ids, feature_cols)
        val_features, val_target = build_xy(modified_frames, [val_id], feature_cols)
        model = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, loss_function=CATBOOST_LOSS, random_seed=RANDOM_SEED, verbose=0)
        model.fit(train_features, train_target)
        return float(np.mean(np.abs(model.predict(val_features) - val_target)))

    return mean_loo_score(modified_frames, score_fold)


def run(features_dir: str, n_trials: int, resume: bool):
    video_frames, feature_cols = load_tuning_data(features_dir)
    study = create_study("features", "features_retention", resume=resume)
    study.optimize(lambda trial: objective(trial, video_frames, feature_cols), n_trials=n_trials, show_progress_bar=True)
    print_best_trial(study, "LOO MAE")
    save_and_report("features", study.best_params, {"loo_mae": round(study.best_value, 6)})


def main():
    parser = argparse.ArgumentParser(description="Tune feature extraction HPs with Optuna")
    parser.add_argument("--features_dir", default="output")
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.features_dir, args.n_trials, args.resume)

if __name__ == "__main__": main()
