import argparse
import os

import catboost as cb
import matplotlib.pyplot as plt
import pandas as pd

from ..aggregator import aggregate
from ..utils.config import Config
from ..utils.logger import Logger


logger = Logger(show=True).get_logger()


class Predictor:
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config = Config(config_path=config_path)
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} does not exist.")

        model = cb.CatBoostRegressor()
        model.load_model(self.model_path)
        return model

    def predict(self, video_path):
        features = aggregate(video_path=video_path, audio_path=video_path, output_path="", config=self.config)
        features["time"] = pd.to_timedelta(features["time"]).dt.total_seconds()
        train_features = features.drop(columns=[c for c in ["retention", "frame"] if c in features.columns])
        logger.info(f"Extracted features shape: {train_features.shape}")
        logger.info(f"Model feature names: {self.model.feature_names_}")
        logger.info(f"Input feature names: {train_features.columns.tolist()}")
        predictions = self.model.predict(train_features)
        video_id = os.path.basename(os.path.dirname(video_path)) or os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join("my_metrics", "videos", video_id, "prediction")
        os.makedirs(out_dir, exist_ok=True)
        self.draw_retention_plot(features, predictions, output_path=os.path.join(out_dir, "pred.png"))
        return predictions

    def draw_retention_plot(self, features, predictions, output_path="retention_plot.png"):
        plt.figure(figsize=(10, 6))
        time = features["time"].astype(int)
        plt.plot(time, predictions)
        plt.title("Predicted Retention over Frames")
        plt.xlabel("Frame")
        plt.ylabel("Predicted Retention")
        if "retention" in features.columns:
            plt.plot(time, features["retention"], label="Actual Retention", alpha=0.5)
            plt.legend()
        plt.grid()
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    predictor = Predictor(model_path=args.model_path, config_path=args.config_path)
    predictor.predict(args.video_path)
    logger.info("Prediction completed.")
