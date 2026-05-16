"""
Hybrid of next models: LSTM, Transformer, VideoMAE.
evaluates individual models, and an optimized weighted ensemble.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.retention_multimodal_lstm import MultimodalRetentionLSTM
from src.models.retention_multimodal_transformer import MultimodalRetentionTransformer
from train.common.seq_data_utils import FeatureNormalizer, load_aligned_embeddings_for_videos, load_all_merged, plot_mae_summary, predict_video_multimodal, seq_metrics
from train.lstm.lstm_seq_base import plot_retention_prediction


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_videomae_embeddings(video_dfs, emb_root):
    emb_root = Path(emb_root)
    res = {}
    for vid, _ in video_dfs.items():
        p = emb_root / vid / "videomae_embeddings.npy"
        if p.exists():
            res[vid] = np.load(str(p))
    return res


def load_model(ckpt_path: Path, device: torch.device, is_lstm: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_tab = len(ckpt["feature_cols"])
    if is_lstm:
        model = MultimodalRetentionLSTM(
            hidden_size=ckpt.get("hidden_size", 256),
            n_layers=ckpt.get("n_layers", 3),
            dropout=ckpt.get("dropout", 0.2),
            bidirectional=ckpt.get("bidirectional", True),
            n_tabular_features=n_tab,
        ).to(device)
    else:
        model = MultimodalRetentionTransformer(
            d_model=ckpt.get("d_model", 256),
            n_heads=ckpt.get("n_heads", 4),
            n_layers=ckpt.get("n_layers", 4),
            d_ff=ckpt.get("d_ff", 512),
            dropout=ckpt.get("dropout", 0.2),
            n_tabular_features=n_tab,
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt["feature_cols"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="train/metamodel/output")
    parser.add_argument("--lstm-exp", default="experiments/lstm_exp/v3_multimodal")
    parser.add_argument("--tf-exp", default="experiments/transformer_exp/v4_tuned_multimodal")
    parser.add_argument("--vmae-exp", default="experiments/videomae_exp/v1_hybrid")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    logger.info("Loading embs")
    video_dfs = load_all_merged("output", "data", use_curve_raw=True, emb_pca_components=0)
    aligned_embs = load_aligned_embeddings_for_videos(video_dfs, "embeddings")
    vmae_embs_raw = _load_videomae_embeddings(video_dfs, "embeddings")

    vmae_embs = {}
    for vid, df in video_dfs.items():
        vmae = vmae_embs_raw.get(vid)
        clip = aligned_embs.get(vid)
        if vmae is not None and clip is not None:
            n = min(len(vmae), len(clip), len(df))
            vmae_embs[vid] = np.concatenate([vmae[:n], clip[:n, 768:]], axis=1)

    logger.info("Loading models")
    m_lstm, cols_lstm = load_model(Path(args.lstm_exp) / "multimodal_lstm_model.pt", device, is_lstm=True)
    m_tf, cols_tf = load_model(Path(args.tf_exp) / "multimodal_transformer_model.pt", device, is_lstm=False)
    m_vmae, cols_vmae = load_model(Path(args.vmae_exp) / "videomae_model.pt", device, is_lstm=False)

    train_ids_lstm = json.loads(Path(args.lstm_exp, "metrics.json").read_text())["train_ids"]
    train_ids_tf = json.loads(Path(args.tf_exp, "metrics.json").read_text())["train_ids"]
    train_ids_vmae = json.loads(Path(args.vmae_exp, "metrics.json").read_text())["train_ids"]

    norm_lstm = FeatureNormalizer()
    norm_lstm.fit({v: video_dfs[v] for v in train_ids_lstm}, cols_lstm)

    norm_tf = FeatureNormalizer()
    norm_tf.fit({v: video_dfs[v] for v in train_ids_tf}, cols_tf)

    norm_vmae = FeatureNormalizer()
    norm_vmae.fit({v: video_dfs[v] for v in train_ids_vmae}, cols_vmae)

    def set_baseline_for_model(model, train_ids, normalizer):
        max_len = max(len(video_dfs[v]) for v in train_ids)
        acc = np.zeros(max_len, dtype=np.float64)
        cnt = np.zeros(max_len, dtype=np.float64)
        for v in train_ids:
            ret = pd.to_numeric(video_dfs[v]["retention"], errors="coerce").fillna(0).values
            acc[: len(ret)] += ret
            cnt[: len(ret)] += 1.0
        curve = (acc / cnt).astype(np.float32)
        model.set_baseline(torch.tensor(normalizer.normalize_retention(curve).astype(np.float32)))

    set_baseline_for_model(m_lstm, train_ids_lstm, norm_lstm)
    set_baseline_for_model(m_tf, train_ids_tf, norm_tf)
    set_baseline_for_model(m_vmae, train_ids_vmae, norm_vmae)

    val_ids = json.loads(Path(args.lstm_exp, "metrics.json").read_text())["val_ids"]
    logger.info(f"Evaluating on {len(val_ids)} validation videos")

    all_true = []
    all_preds_lstm = []
    all_preds_tf = []
    all_preds_vmae = []
    vid_data = {}  
    
    for vid in val_ids:
        df = video_dfs[vid]
        y_true, p_lstm = predict_video_multimodal(m_lstm, df, aligned_embs.get(vid), cols_lstm, norm_lstm, device, 128)
        _, p_tf = predict_video_multimodal(m_tf, df, aligned_embs.get(vid), cols_tf, norm_tf, device, 128)
        _, p_vmae = predict_video_multimodal(m_vmae, df, vmae_embs.get(vid), cols_vmae, norm_vmae, device, 128)
        n = min(len(y_true), len(p_lstm), len(p_tf), len(p_vmae))
        y_true = y_true[:n]
        p_lstm = p_lstm[:n]
        p_tf = p_tf[:n]
        p_vmae = p_vmae[:n]

        all_true.append(y_true)
        all_preds_lstm.append(p_lstm)
        all_preds_tf.append(p_tf)
        all_preds_vmae.append(p_vmae)

        vid_data[vid] = {"y_true": y_true, "p_lstm": p_lstm, "p_tf": p_tf, "p_vmae": p_vmae}

    y = np.concatenate(all_true)
    p1 = np.concatenate(all_preds_lstm)
    p2 = np.concatenate(all_preds_tf)
    p3 = np.concatenate(all_preds_vmae)

    logger.info(f"LSTM v3 Multimodal MAE: {np.mean(np.abs(y - p1)):.4f}")
    logger.info(f"Transformer v4 Multimodal MAE: {np.mean(np.abs(y - p2)):.4f}")
    logger.info(f"VideoMAE v1 Hybrid MAE: {np.mean(np.abs(y - p3)):.4f}")

    p_avg = (p1 + p2 + p3) / 3.0
    logger.info(f"Average Ensemble MAE: {np.mean(np.abs(y - p_avg)):.4f}")

    X = np.stack([p1, p2, p3], axis=1)
    weights, _ = nnls(X, y)
    weights = weights / np.sum(weights)

    p_weight = X @ weights
    logger.info("Weighted Metamodel")
    logger.info(f"Optimal Weights -> LSTM: {weights[0]:.3f}, TF: {weights[1]:.3f}, VMAE: {weights[2]:.3f}")
    logger.info(f"Weighted Ensemble MAE: {np.mean(np.abs(y - p_weight)):.4f}")

    all_metrics = {}
    for vid in val_ids:
        v_data = vid_data[vid]
        y_true_v = v_data["y_true"]
        X_v = np.stack([v_data["p_lstm"], v_data["p_tf"], v_data["p_vmae"]], axis=1)
        p_weight_v = X_v @ weights

        metrics = seq_metrics(y_true_v, p_weight_v)
        metrics["video_id"] = vid
        metrics["split"] = "val"
        all_metrics[vid] = metrics

        plot_path = Path(args.output_dir) / "videos" / vid / "prediction.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_retention_prediction(vid, y_true_v, p_weight_v, None, None, metrics, str(plot_path))

    plot_mae_summary(all_metrics, args.output_dir, model_name="Metamodel")

    meta_path = Path(args.output_dir) / "metrics.json"
    meta_path.write_text(
        json.dumps(
            {
                "model": "MetamodelEnsemble",
                "models": ["lstm_v3_multimodal", "transformer_v4_tuned_multimodal", "videomae_v1_hybrid"],
                "weights": weights.tolist(),
                "best_val_loss": float(np.mean(np.abs(y - p_weight))),
            },
            indent=2,
        )
    )
    logger.info(f"Saved metamodel weights and metrics to {meta_path}")


if __name__ == "__main__":
    main()
