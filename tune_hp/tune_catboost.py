"""
catboost tuning.
python tune_hp/tune_catboost.py --n_trials 100 --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostRegressor
from optuna.integration import CatBoostPruningCallback


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tune_hp.utils import CATBOOST_EVAL, CATBOOST_LOSS, RANDOM_SEED, build_xy, compute_metrics, create_study, load_tuning_data, mean_loo_score, print_best_trial, save_and_report


def objective(trial: optuna.Trial, video_frames: dict, feature_cols: list[str]) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 300, 3000, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
    }

    def score_fold(train_ids: list[str], val_id: str, fold_idx: int) -> float:
        train_features, train_target = build_xy(video_frames, train_ids, feature_cols)
        val_features, val_target = build_xy(video_frames, [val_id], feature_cols)
        pruning_callback = CatBoostPruningCallback(trial, CATBOOST_EVAL)
        model = CatBoostRegressor(**params, loss_function=CATBOOST_LOSS, eval_metric=CATBOOST_EVAL, random_seed=RANDOM_SEED, verbose=0, early_stopping_rounds=50)
        model.fit(train_features, train_target, eval_set=(val_features, val_target), callbacks=[pruning_callback])
        pruning_callback.check_pruned()
        fold_mae = float(np.mean(np.abs(model.predict(val_features) - val_target)))
        trial.report(fold_mae, fold_idx)
        if trial.should_prune(): raise optuna.TrialPruned()
        return fold_mae

    return mean_loo_score(video_frames, score_fold)


def run(features_dir: str, n_trials: int, resume: bool):
    video_frames, feature_cols = load_tuning_data(features_dir)
    study = create_study("catboost", "catboost_retention", resume=resume, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study.optimize(lambda trial: objective(trial, video_frames, feature_cols), n_trials=n_trials, show_progress_bar=True)
    print_best_trial(study, "MAE")
    all_features, all_target = build_xy(video_frames, sorted(video_frames.keys()), feature_cols)
    final_model = CatBoostRegressor(**study.best_params, loss_function=CATBOOST_LOSS, eval_metric=CATBOOST_EVAL, random_seed=RANDOM_SEED, verbose=0)
    final_model.fit(all_features, all_target)
    save_and_report("catboost", study.best_params, {"loo_mae": round(study.best_value, 6), **compute_metrics(all_target, final_model.predict(all_features))})


def main():
    parser = argparse.ArgumentParser(description="Tune CatBoost HPs with Optuna")
    parser.add_argument("--features_dir", default="output")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Resume previous study")
    args = parser.parse_args()
    run(args.features_dir, args.n_trials, args.resume)

if __name__ == "__main__":
    main()
