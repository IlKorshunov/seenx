import argparse

import cv2
import numpy as np
import pandas as pd
import torch

from .extractors.feature_extractor import VideoFeature
from .extractors.video import *
from .extractors.video import batch_shot_segmentation
from .utils.config import Config
from .utils.logger import Logger
import gc

logger = Logger(show=True).get_logger()


def run_feature_pipeline(video_path: str, config: Config, passes: list[VideoFeature], existing_features: set) -> pd.DataFrame:
    vid_cap = cv2.VideoCapture(video_path)
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_cap.release()

    shift = config.get("shot_bound_shift_frames")
    shot_bounds = batch_shot_segmentation(video_path, config)
    shot_bounds[:, 0] = np.clip(shot_bounds[:, 0] + shift, 0, total_frames - 1)
    shot_bounds[:, 1] = np.clip(shot_bounds[:, 1] - shift, 0, total_frames - 1)
    context = {"data": pd.DataFrame({"frame_idx": np.arange(total_frames, dtype=np.int32)}), "shift": shift, "shot_bounds": shot_bounds.flatten().tolist()}

    failed_passes = []
    for feature_pass in passes:
        keys = feature_pass.produces_keys()
        if keys <= existing_features:
            continue
        missing = feature_pass.required_keys() - context.keys() - set(context["data"].columns)
        if missing:
            logger.warning("Skipping %s: missing deps %s", feature_pass.__class__.__name__, missing)
            failed_passes.append((feature_pass.__class__.__name__, f"missing deps: {missing}"))
            continue
        try:
            feature_pass.run(video_path, context)
        except Exception as e:
            logger.error("Pass %s failed: %s — filling produced keys with 0.0", feature_pass.__class__.__name__, e)
            context["data"] = context["data"].assign(**{key: 0.0 for key in keys if key not in context["data"].columns})
            failed_passes.append((feature_pass.__class__.__name__, str(e)))
        torch.cuda.empty_cache()

    if failed_passes:
        logger.warning("Feature pipeline: %d/%d passes failed: %s", len(failed_passes), len(passes), ", ".join(f"{name}({err[:60]})" for name, err in failed_passes))

    for feature_pass in passes:
        for attr, obj in list(vars(feature_pass).items()):
            if isinstance(obj, torch.nn.Module):
                obj.cpu()
                delattr(feature_pass, attr)
    del context["shot_bounds"]
    gc.collect()
    torch.cuda.empty_cache()
    return context["data"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file.")
    args = parser.parse_args()
    config = Config(args.config)
    features_df = run_feature_pipeline(
        args.video,
        config,
        passes=[FrameQualityFeature(config), SpeakerProbabilityFeature(config), FaceScreenRatioFeature(config), TextProbFeature(config), MotionSpeedFeature(config)],
        existing_features=set(),
    )
    features_df.to_csv("features.csv", index=False)
