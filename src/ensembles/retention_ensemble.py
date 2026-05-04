from __future__ import annotations

import numpy as np

from ..retention_analysis import calc_retention_metrics
from ..utils.logger import Logger


logger = Logger(show=True).get_logger()


class EnsemblePredictor:
    def __init__(self, catboost_trainer, transformer_trainer):
        self.cb_trainer = catboost_trainer
        self.tf = transformer_trainer
        self.alpha = 0.5

    def _cb_predict_video(self, video_id: str) -> tuple[np.ndarray, np.ndarray]:
        y_true, y_abs, y_delta_blend = self.cb_trainer._predict_components(video_id)  # pylint: disable=protected-access
        y_cb = self.cb_trainer.alpha * y_abs + (1 - self.cb_trainer.alpha) * y_delta_blend
        return y_true, y_cb

    def tune_alpha(self, _video_frames, val_ids):
        best_alpha, best_mae = 0.5, float("inf")
        for alpha in np.arange(0.0, 1.05, 0.1):
            errors = []
            for video_id in val_ids:
                y_true, y_cb = self._cb_predict_video(video_id)
                _, y_tf = self.tf.predict_video(video_id)
                min_len = min(len(y_true), len(y_cb), len(y_tf))
                errors.append(np.abs(alpha * y_tf[:min_len] + (1 - alpha) * y_cb[:min_len] - y_true[:min_len]))
            mae = float(np.mean(np.concatenate(errors)))
            if mae < best_mae:
                best_mae, best_alpha = mae, float(alpha)
        self.alpha = round(best_alpha, 2)
        logger.info("Ensemble alpha=%.2f (val MAE=%.4f)", self.alpha, best_mae)
        return self.alpha

    def predict_video(self, _video_frames, video_id):
        y_true, y_cb = self._cb_predict_video(video_id)
        _, y_tf = self.tf.predict_video(video_id)
        min_len = min(len(y_true), len(y_cb), len(y_tf))
        return y_true[:min_len], self.alpha * y_tf[:min_len] + (1 - self.alpha) * y_cb[:min_len]

    def evaluate(self, video_frames, video_ids, split):
        y_true, y_pred = zip(*[self.predict_video(video_frames, video_id) for video_id in video_ids], strict=True)
        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        metrics = calc_retention_metrics(y_true, y_pred)
        logger.info("Ensemble %s — MSE=%.4f MAE=%.4f R2=%.4f (alpha=%.2f)", split, metrics["mse"], metrics["mae"], metrics["r2"], self.alpha)
        return {**metrics, "alpha": self.alpha}
