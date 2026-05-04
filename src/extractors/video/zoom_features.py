import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ...utils.config import Config
from ...utils.logger import Logger
from ...video_dataset import VideoBatchDataset
from .shot_segmentation import batch_shot_segmentation

from utils.utils import InputPadder
from raft import RAFT

logger = Logger(show=True).get_logger()

ROOT = os.environ.get("WORKING_DIR", str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, os.path.join(ROOT, "RAFT/core"))


class _RaftArgs:
    """Minimal namespace that RAFT(args) expects."""

    def __init__(self, *, small: bool, mixed_precision: bool, alternate_corr: bool = False, dropout: float = 0.0):
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr
        self.dropout = dropout

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


class ZoomFeatureExtractor:
    def __init__(self, video_path: str, config: Config):
        self.video_path = video_path
        self.config = config
        self.device = torch.device(config.get("device"))
        self.flow_stride = config.get("flow_stride", 8)
        self.batch_size = config.get("batch_size")

        weights_path = config.get("optical_flow_model")
        is_small = "small" in os.path.basename(weights_path)
        self.use_amp = self.device.type == "cuda"
        self._raft_args = _RaftArgs(small=is_small, mixed_precision=self.use_amp)
        self.model = self._load_model(weights_path)

    def _load_model(self, weights_path: str):
        model = torch.nn.DataParallel(RAFT(self._raft_args))
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model = model.module.to(self.device).eval()
        return model

    @staticmethod
    def transform(frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, fx=0.5, fy=0.5)
        return frame[np.newaxis, :, :, :]

    @staticmethod
    def make_center_grid(h, w, device, stride):
        y, x = torch.meshgrid(torch.arange(0, h, stride, device=device), torch.arange(0, w, stride, device=device), indexing="ij")
        dx, dy = x - (w * 0.5), y - (h * 0.5)
        base_dist = torch.sqrt(dx**2 + dy**2) + 1e-6
        return x, y, dx, dy, base_dist

    @staticmethod
    def compute_flow_features(flow, grid, flow_stride):
        _, _, dx, dy, base_dist = grid
        fx = flow[:, 0][:, ::flow_stride, ::flow_stride]
        fy = flow[:, 1][:, ::flow_stride, ::flow_stride]
        mag = torch.sqrt(fx**2 + fy**2)
        radial = (fx * dx + fy * dy) / base_dist
        return (mag.flatten(1).median(dim=1).values, radial.flatten(1).median(dim=1).values, (radial > 0).float().mean(dim=(1, 2)))

    def _process_batch(self, fs: np.ndarray, indices: list[int], grid: tuple, features: list[dict], start_idx: int):
        fs_tensor = torch.from_numpy(fs).permute(0, 3, 1, 2).float().to(self.device)
        img1, img2 = fs_tensor[:-1], fs_tensor[1:]
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            _, flow_up = self.model(img1, img2, iters=20, test_mode=True)
        flow_up = padder.unpad(flow_up)
        mag, radial, ratio = self.compute_flow_features(flow_up, grid, self.flow_stride)
        mag, radial, ratio = mag.cpu().numpy(), radial.cpu().numpy(), ratio.cpu().numpy()
        features.extend({"frame": indices[i + start_idx], "flow_mag_med": float(mag[i]), "radial_med": float(radial[i]), "radial_ratio": float(ratio[i])} for i in range(len(mag)))
        del fs_tensor, img1, img2, flow_up
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def run(self) -> pd.DataFrame:
        dataset = VideoBatchDataset(video_path=self.video_path, batch_size=self.batch_size, transform=self.transform, stride=10)
        iterator = iter(dataset)
        frames, indices = next(iterator)
        _, h, w, _ = frames.shape
        grid = self.make_center_grid(h, w, self.device, self.flow_stride)
        features = [{"frame": 0, "flow_mag_med": 0.0, "radial_med": 0.0, "radial_ratio": 0.0}]
        self._process_batch(fs=frames, indices=indices, grid=grid, features=features, start_idx=1)
        prev_last = frames[-1:]
        prev_index = indices[-1]
        for frames, indices in tqdm(iterator, total=len(dataset) - 1):
            frames, indices = np.concatenate([prev_last, frames], axis=0), [prev_index] + indices
            self._process_batch(fs=frames, indices=indices, grid=grid, features=features, start_idx=1)
            prev_last = frames[-1:]
        return pd.DataFrame(features).set_index("frame").reindex(range(dataset.total_frames)).ffill().bfill().reset_index()


def mask_flow_at_cuts(df: pd.DataFrame, video_path: str, config: Config, radius: int = 1) -> pd.DataFrame:
    """Zero out flow features on frames within `radius` of a shot boundary."""
    flow_cols = [c for c in ("flow_mag_med", "radial_med", "radial_ratio") if c in df.columns]
    if not flow_cols:
        return df

    scenes = batch_shot_segmentation(video_path, config)
    cut_frames = set()
    for start, end in scenes:
        for offset in range(-radius, radius + 1):
            cut_frames.add(start + offset)
            cut_frames.add(end + offset)

    mask = df.index.isin(cut_frames)
    if not mask.any():
        return df

    df = df.copy()
    df.loc[mask, flow_cols] = np.nan
    df[flow_cols] = df[flow_cols].ffill().bfill().fillna(0.0)
    logger.info("Masked flow features at %d frames near %d shot boundaries", int(mask.sum()), len(scenes))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = Config(args.config)
    extractor = ZoomFeatureExtractor(video_path=args.video, config=config)
    df = extractor.run()
    print(df)
