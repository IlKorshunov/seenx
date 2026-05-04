import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ...utils.config import Config
from ...utils.logger import Logger
from ...video_dataset import VideoBatchDataset
from .architectures import TransNetV2
from .common import mask_runs


logger = Logger(show=True).get_logger()


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
    cuts = predictions > threshold
    scenes = mask_runs(cuts, value=False)
    scenes[:, 1] = np.minimum(scenes[:, 1] + (scenes[:, 1] < len(cuts) - 1), len(cuts) - 1)
    return scenes if len(scenes) else np.array([[0, len(predictions) - 1]], dtype=np.int32)


def batch_shot_segmentation(video_path: str, config: Config) -> np.ndarray:
    if config.get("use_shot_ensemble") is not False and config.get("shot_boundary_ensemble") is not False:
        from .architectures.ensamble import ensemble_shot_segmentation

        return ensemble_shot_segmentation(video_path, config)

    transnet_weights_path = config.get("shot_segmentor")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")
    if not os.path.exists(transnet_weights_path):
        raise FileNotFoundError(f"TransNetV2 weights not found at {transnet_weights_path}")

    model = TransNetV2()
    model.load_state_dict(torch.load(transnet_weights_path, map_location=torch.device(config.get("device"))))
    model.eval().to(torch.device(config.get("device")))

    def transform(frame: np.ndarray) -> np.ndarray:
        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (48, 27), interpolation=cv2.INTER_AREA)[np.newaxis, :, :, :]

    dataset = VideoBatchDataset(video_path, batch_size=config.get("shot_segmentor_batch_size"), transform=transform)
    n_processed = dataset.total_processed_frames or 0
    all_frame_pred = np.zeros(n_processed, dtype=np.float32)
    for frames_batch, frame_indices in tqdm(dataset):
        with torch.no_grad():
            batch_tensor = torch.tensor(frames_batch, dtype=torch.uint8).to(torch.device(config.get("device")))
            preds = torch.sigmoid(model(batch_tensor.unsqueeze(0))[0]).cpu().numpy().squeeze(0)
            all_frame_pred[frame_indices] = preds.flatten()

    scene_bounds = predictions_to_scenes(all_frame_pred)
    scene_ranges = ", ".join([f"{start}:{end}" for start, end in scene_bounds])
    logger.info("Detected scenes (start_frame, end_frame): %s", scene_ranges)
    return scene_bounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=False)
    args = parser.parse_args()
    scene_bounds = batch_shot_segmentation(args.video_path, config=Config(args.config))
