"""
Emotion fusion: combine voice and text into YouTube-relevant emotions.
  joy — positivity
  excitement — curiosity
  sadness — loss
  neutral
"""

import numpy as np
import pandas as pd


EKMAN_LABELS = ["joy", "excitement", "sadness", "neutral"]

FACE_MAP = {}

VOICE_MAP = {"joy": ["voice_happy"], "sadness": ["voice_sad"], "neutral": ["voice_neutral"]}

TEXT_MAP = {
    "joy": ["sent_joy", "sent_amusement", "sent_love", "sent_optimism", "sent_admiration"],
    "excitement": ["sent_excitement", "sent_curiosity", "sent_surprise", "sent_realization"],
    "sadness": ["sent_sadness", "sent_grief", "sent_disappointment", "sent_remorse"],
    "neutral": ["sent_neutral"],
}

MODALITY_WEIGHT = {"face": 0.0, "voice": 0.5, "text": 0.5}
TOP_K = 4
THRESHOLD = 0.05


def _mean_cols(df: pd.DataFrame, cols: list[str]) -> np.ndarray | None:
    present = [c for c in cols if c in df.columns]
    if not present:
        return None
    return df[present].to_numpy(dtype=np.float64).mean(axis=1)


def _modality_available_mask(df: pd.DataFrame, cols_map: dict, indicator_col: str | None, indicator_thr: float) -> np.ndarray:
    if indicator_col and indicator_col in df.columns:
        return df[indicator_col].to_numpy(dtype=np.float64) >= indicator_thr
    all_cols = [c for cols in cols_map.values() for c in cols if c in df.columns]
    if not all_cols:
        return np.zeros(len(df), dtype=bool)
    return np.any(df[all_cols].to_numpy(dtype=np.float64) > 0, axis=1)


def compute_ekman_fusion(accumulated: pd.DataFrame, speaker_thr: float = 0.3) -> pd.DataFrame:
    ekman_raw = np.zeros((len(accumulated), len(EKMAN_LABELS)), dtype=np.float64)
    weight_sum = np.zeros(len(accumulated), dtype=np.float64)
    modalities = [
        ("face", FACE_MAP, _modality_available_mask(accumulated, FACE_MAP, "speaker_prob", speaker_thr)),
        ("voice", VOICE_MAP, _modality_available_mask(accumulated, VOICE_MAP, None, 0)),
        ("text", TEXT_MAP, _modality_available_mask(accumulated, TEXT_MAP, None, 0)),
    ]

    for name, cols_map, available in modalities:
        if any(col in accumulated.columns for cols in cols_map.values() for col in cols):
            weight_sum += available * MODALITY_WEIGHT[name]

    for i, ekman in enumerate(EKMAN_LABELS):
        for name, cols_map, available in modalities:
            values = _mean_cols(accumulated, cols_map.get(ekman, []))
            if values is None:
                continue
            weight = MODALITY_WEIGHT[name]
            ekman_raw[:, i] += values * available * weight

    ekman_raw /= np.maximum(weight_sum, 1e-9)[:, None]
    ranked = np.argsort(ekman_raw, axis=1)[:, ::-1][:, :TOP_K]
    output = np.where(ekman_raw >= THRESHOLD, ekman_raw, 0.0) * np.isin(np.arange(len(EKMAN_LABELS))[None, :], ranked)

    cols = {f"ekman_{e}": output[:, i] for i, e in enumerate(EKMAN_LABELS)}
    cols["ekman_intensity"] = np.max(output, axis=1)

    return pd.DataFrame(cols, index=accumulated.index)
