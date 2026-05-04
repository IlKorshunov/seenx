"""
Hyperparameter optimization for CatBoost using Optuna.
Optimizes RMSE for target_avg_retention.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from analysis.feature_importance.utils import aggregate_per_video, load_all_videos, prepare_X_y


def objective(trial, X, y):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "loss_function": "RMSE",
        "verbose": False,
        "random_seed": 42,
    }

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    X_values = X.values
    y_values = y.values

    for train_index, val_index in kf.split(X_values):
        X_train, X_val = X_values[train_index], X_values[val_index]
        y_train, y_val = y_values[train_index], y_values[val_index]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


def run_optimization(features_dir, n_trials=50):
    print(f"Loading features from {features_dir}...")

    # Use robust loading utils from analysis package
    try:
        video_dfs = load_all_videos(features_dir)
    except FileNotFoundError:
        print(f"No features found in {features_dir}")
        return

    print(f"Loaded {len(video_dfs)} videos.")

    # Aggregate to 1 row per video (mean features) -> predict mean retention
    # This is the standard task for now.
    # If we want frame-level, we would need a different approach (stacking all frames).
    agg_df = aggregate_per_video(video_dfs, agg="mean")

    target = "target_avg_retention"
    if target not in agg_df.columns:
        print(f"Target {target} not found in aggregated data!")
        # Fallback to creating it if possible (aggregate_per_video usually adds it)
        return

    X, y = prepare_X_y(agg_df, target=target)

    if len(X) < 5:
        print("Not enough data for optimization (need at least 5 samples for CV).")
        return

    print(f"Optimization on {len(X)} samples, target: {target}")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("\nBest trial:")
    print(f"  Value (RMSE): {study.best_value:.4f}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Save best params
    import json

    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("Saved best_params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="output")
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

    run_optimization(args.features_dir, args.n_trials)
