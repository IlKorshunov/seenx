import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class RetentionAugmentation:
    @staticmethod
    def apply_random_augmentation(features: np.ndarray, augmentation_prob: float = 0.5) -> np.ndarray:
        if np.random.rand() > augmentation_prob:
            return features.copy()

        aug_features = features.copy()

        if aug_features.ndim == 0 or (aug_features.ndim == 1 and len(aug_features) <= 1):
            return aug_features

        if aug_features.ndim == 1:
            noise = np.random.normal(0, 0.05, size=aug_features.shape)
            return aug_features + noise

        seq_len, num_features = aug_features.shape

        if np.random.rand() < 0.3:
            mask_len = max(1, int(seq_len * 0.1))
            start = np.random.randint(0, max(1, seq_len - mask_len))
            aug_features[start : start + mask_len, :] = 0.0

        if np.random.rand() < 0.3:
            num_masked_features = max(1, int(num_features * 0.1))
            feature_indices = np.random.choice(num_features, num_masked_features, replace=False)
            aug_features[:, feature_indices] = 0.0

        if np.random.rand() < 0.4:
            std = np.std(aug_features, axis=0)
            noise = np.random.normal(0, 0.05 * (std + 1e-6), size=aug_features.shape)
            aug_features += noise

        if np.random.rand() < 0.2 and seq_len > 3:
            scale_factor = np.random.uniform(0.8, 1.2)
            new_len = max(3, int(seq_len * scale_factor))

            x_old = np.linspace(0, 1, seq_len)
            x_new = np.linspace(0, 1, new_len)

            scaled_features = np.zeros((new_len, num_features))
            for i in range(num_features):
                scaled_features[:, i] = np.interp(x_new, x_old, aug_features[:, i])

            if new_len >= seq_len:
                aug_features = scaled_features[:seq_len, :]
            else:
                pad_len = seq_len - new_len
                padding = np.tile(scaled_features[-1, :], (pad_len, 1))
                aug_features = np.vstack([scaled_features, padding])

        return aug_features


class RetentionDataset(Dataset):
    def __init__(self, data, feature_cols, video_ids, scaler, fit_scaler=True, max_seq_len=100):
        self.feature_cols = feature_cols
        self.video_ids = video_ids
        self.max_seq_len = max_seq_len

        self.sequences = []
        self.targets = []

        all_features = []

        for video_id in video_ids:
            video_data = data[data["video_id"] == video_id].sort_values("interval_idx")
            video_data = video_data.iloc[:max_seq_len]

            features = video_data[feature_cols].values
            target = video_data["target"].values if "target" in video_data.columns else np.zeros(len(video_data))

            if len(features) < max_seq_len:
                pad_len = max_seq_len - len(features)
                pad_features = np.zeros((pad_len, len(feature_cols)))
                features = np.vstack([features, pad_features])
                pad_target = np.zeros(pad_len)
                target = np.concatenate([target, pad_target])

            if fit_scaler and scaler is None:
                all_features.append(features)
            elif scaler is not None:
                features = scaler.transform(features)

            self.sequences.append(torch.FloatTensor(features))
            self.targets.append(torch.FloatTensor(target))

        if fit_scaler and scaler is None and all_features:
            all_features = np.vstack(all_features)
            self.scaler = StandardScaler()
            self.scaler.fit(all_features)
            for i in range(len(self.sequences)):
                self.sequences[i] = torch.FloatTensor(self.scaler.transform(self.sequences[i].numpy()))
        else:
            self.scaler = scaler

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class AugmentedRetentionDataset(Dataset):
    def __init__(self, data, feature_cols, video_ids, scaler, fit_scaler=True, max_seq_len=100, augment=False, augment_prob=0.5, num_augmentations=1):
        self.feature_cols = feature_cols
        self.video_ids = video_ids
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.augment_prob = augment_prob
        self.num_augmentations = num_augmentations

        self.sequences = []
        self.targets = []

        all_features = []

        for video_id in video_ids:
            video_data = data[data["video_id"] == video_id].sort_values("interval_idx")
            video_data = video_data.iloc[:max_seq_len]

            features = video_data[feature_cols].values
            target = video_data["target"].values if "target" in video_data.columns else np.zeros(len(video_data))

            if len(features) < max_seq_len:
                pad_len = max_seq_len - len(features)
                pad_features = np.zeros((pad_len, len(feature_cols)))
                features = np.vstack([features, pad_features])
                pad_target = np.zeros(pad_len)
                target = np.concatenate([target, pad_target])

            if fit_scaler and scaler is None:
                all_features.append(features)
            elif scaler is not None:
                features = scaler.transform(features)

            self.sequences.append(features)
            self.targets.append(target)

        if fit_scaler and scaler is None and all_features:
            all_features = np.vstack(all_features)
            self.scaler = StandardScaler()
            self.scaler.fit(all_features)
            for i in range(len(self.sequences)):
                self.sequences[i] = self.scaler.transform(self.sequences[i])
        else:
            self.scaler = scaler

        if augment and len(self.sequences) >= 1:
            self._augment_data()

    def _augment_data(self):
        original_len = len(self.sequences)
        for i in range(original_len):
            for _ in range(self.num_augmentations):
                aug_features = RetentionAugmentation.apply_random_augmentation(self.sequences[i], augmentation_prob=self.augment_prob)
                aug_target = self.targets[i].copy()
                self.sequences.append(aug_features)
                self.targets.append(aug_target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])
