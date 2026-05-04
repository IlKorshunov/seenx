"""
ensemble alpha tuning.
python tune_hp/tune_ensemble.py --n_trials 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catboost import CatBoostRegressor
from src.models.retention_transformer import RetentionTransformer
from tune_hp.utils import CUDA_DEVICE, compute_metrics, create_study, ensemble_loss, load_tuning_data, print_lines, save_and_report, smooth_l1_loss_np


def _predict_catboost(model: CatBoostRegressor, frame, feature_cols: list[str]) -> np.ndarray:
    return model.predict(frame[feature_cols].astype(float).fillna(0))


@torch.no_grad()
def _predict_transformer(model: RetentionTransformer, frame, feature_cols: list[str], device: torch.device) -> np.ndarray:
    features = frame[feature_cols].astype(float).fillna(0).values
    n_seconds = len(features)
    window_size = 128

    if n_seconds <= window_size:
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        return model(features_tensor).squeeze(0).cpu().numpy()[:n_seconds]

    pred_sum = np.zeros(n_seconds, dtype=np.float64)
    pred_count = np.zeros(n_seconds, dtype=np.float64)
    for start_second in range(0, n_seconds - window_size + 1):
        chunk = torch.tensor(features[start_second : start_second + window_size], dtype=torch.float32).unsqueeze(0).to(device)
        prediction = model(chunk).squeeze(0).cpu().numpy()
        pred_sum[start_second : start_second + window_size] += prediction
        pred_count[start_second : start_second + window_size] += 1.0
    return pred_sum / np.maximum(pred_count, 1.0)


def objective(trial: optuna.Trial, video_frames: dict, feature_cols: list[str], cb_model: CatBoostRegressor, tf_model: RetentionTransformer, device: torch.device) -> float:
    alpha = trial.suggest_float("alpha", 0.0, 1.0)

    video_ids = sorted(video_frames.keys())
    fold_losses: list[float] = []

    for video_id in video_ids:
        frame = video_frames[video_id]
        target = frame["retention"].values.astype(float)

        catboost_pred = _predict_catboost(cb_model, frame, feature_cols)
        transformer_pred = _predict_transformer(tf_model, frame, feature_cols, device)

        min_len = min(len(target), len(catboost_pred), len(transformer_pred))
        blend_pred = alpha * transformer_pred[:min_len] + (1 - alpha) * catboost_pred[:min_len]
        fold_losses.append(ensemble_loss(target[:min_len], blend_pred))
    return float(np.mean(fold_losses))


def run(features_dir: str, catboost_path: str, transformer_path: str, n_trials: int):
    video_frames, feature_cols = load_tuning_data(features_dir)
    device = torch.device(CUDA_DEVICE)

    cb_model = CatBoostRegressor()
    cb_model.load_model(catboost_path)
    print(f"Loaded CatBoost from {catboost_path}")

    checkpoint = torch.load(transformer_path, map_location=device, weights_only=False)
    tf_model = RetentionTransformer(
        n_features=checkpoint["n_features"],
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        d_ff=checkpoint["d_ff"],
        dropout=checkpoint["dropout"],
    ).to(device)
    tf_model.load_state_dict(checkpoint["model_state_dict"])
    tf_model.eval()

    study = create_study("ensemble", "ensemble_retention", resume=True)
    study.optimize(lambda trial: objective(trial, video_frames, feature_cols, cb_model, tf_model, device), n_trials=n_trials, show_progress_bar=True)

    best_alpha = study.best_params["alpha"]
    target_parts, blend_parts = [], []
    for frame in video_frames.values():
        target = frame["retention"].values.astype(float)
        catboost_pred = _predict_catboost(cb_model, frame, feature_cols)
        transformer_pred = _predict_transformer(tf_model, frame, feature_cols, device)
        min_len = min(len(target), len(catboost_pred), len(transformer_pred))
        target_parts.append(target[:min_len])
        blend_parts.append(best_alpha * transformer_pred[:min_len] + (1 - best_alpha) * catboost_pred[:min_len])
    target_all = np.concatenate(target_parts)
    blend_all = np.concatenate(blend_parts)
    metrics = compute_metrics(target_all, blend_all) | {"smooth_l1": round(smooth_l1_loss_np(target_all, blend_all), 6), "ensemble_loss": round(study.best_value, 6)}
    print_lines(f"{os.linesep}Best alpha = {best_alpha:.3f} (ensemble_loss = {study.best_value:.4f})", "  alpha=0 → pure CatBoost, alpha=1 → pure Transformer")

    save_and_report("ensemble", study.best_params, metrics)


def main():
    parser = argparse.ArgumentParser(description="Tune ensemble alpha with Optuna")
    parser.add_argument("--features_dir", default="output")
    parser.add_argument("--catboost_path", default="static/weights/model.cbm")
    parser.add_argument("--transformer_path", default="static/weights/model_transformer.pt")
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()
    run(args.features_dir, args.catboost_path, args.transformer_path, args.n_trials)


if __name__ == "__main__":
    main()
