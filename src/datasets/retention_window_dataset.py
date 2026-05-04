from __future__ import annotations

# pylint: disable=import-error

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RetentionWindowDataset(Dataset):
    def __init__(self, video_frames: dict[str, pd.DataFrame], video_ids: list[str], feature_cols: list[str], window_size: int = 128, stride: int = 64):
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.windows = []
        for video_id in video_ids:
            frame = video_frames[video_id]
            features = frame.reindex(columns=feature_cols, fill_value=0).astype(float).fillna(0).values
            retention = frame["retention"].values.astype(float)
            n_rows = len(features)
            if n_rows <= window_size:
                self.windows.append((features, retention, n_rows))
                continue
            for start in range(0, n_rows - window_size + 1, stride):
                self.windows.append((features[start : start + window_size], retention[start : start + window_size], window_size))
            if (n_rows - window_size) % stride != 0:
                self.windows.append((features[n_rows - window_size :], retention[n_rows - window_size :], window_size))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        features, retention, real_len = self.windows[idx]
        if len(features) < self.window_size:
            pad = self.window_size - len(features)
            features = np.pad(features, ((0, pad), (0, 0)))
            retention = np.pad(retention, (0, pad))
            mask = np.array([False] * real_len + [True] * pad)
        else:
            mask = np.zeros(self.window_size, dtype=bool)
        return {"features": torch.tensor(features, dtype=torch.float32), "retention": torch.tensor(retention, dtype=torch.float32), "padding_mask": torch.tensor(mask)}
