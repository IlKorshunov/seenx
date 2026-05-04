"""Default RetentionTransformer training pipeline."""

from __future__ import annotations

import glob
import os
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets import RetentionWindowDataset
from src.models.retention_transformer import RetentionTransformer
from src.retention_analysis import RETENTION_FEATURE_EXCLUDE
from src.retention_analysis import calc_retention_metrics
from src.utils.logger import Logger


logger = Logger(show=True).get_logger()


def _resolve_device():
    from src.utils.config import Config

    return torch.device(Config("configs/local.json").get("device"))


class TransformerTrainer:
    def __init__(self, video_frames, val_ratio=0.2, random_state=42):
        self.video_frames = video_frames
        self.model = None
        self.train_losses, self.val_losses = [], []
        self.window_size = 128
        all_cols = set()
        for df in video_frames.values():
            all_cols.update(column for column in df.columns if column not in RETENTION_FEATURE_EXCLUDE)
        self.feature_names = sorted(all_cols)
        ids = sorted(video_frames.keys())
        np.random.default_rng(random_state).shuffle(ids)
        n_val = max(1, int(len(ids) * val_ratio))
        self.val_ids, self.train_ids = ids[:n_val], ids[n_val:]
        logger.info("Train: %s, Val: %s", self.train_ids, self.val_ids)

    @classmethod
    def from_output_dir(cls, output_dir="output", val_ratio=0.2):
        csv_paths = sorted(glob.glob(os.path.join(output_dir, "*_features.csv")))
        if not csv_paths:
            raise FileNotFoundError(f"No *_features.csv in {output_dir}")
        video_frames = {}
        for csv_path in csv_paths:
            video_id = os.path.basename(csv_path).replace("_features.csv", "")
            df = pd.read_csv(csv_path, index_col=0)
            if "retention" not in df.columns:
                continue
            video_frames[video_id] = df.dropna(subset=["retention"])
            logger.info("Loaded %s: %d rows", video_id, len(video_frames[video_id]))
        return cls(video_frames, val_ratio=val_ratio)

    def _loader(self, ids, window_size, stride, batch_size, shuffle):
        return DataLoader(RetentionWindowDataset(self.video_frames, ids, self.feature_names, window_size, stride), batch_size=batch_size, shuffle=shuffle)

    def train(
        self,
        epochs=200,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-4,
        window_size=128,
        window_stride=64,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.2,
        patience=15,
        grad_clip=1.0,
        save_path="static/weights/retention_transformer.pt",
        log_dir="train/tensorboard_transformer",
    ):
        device = _resolve_device()
        self.window_size = window_size
        n_features = len(self.feature_names)
        model = RetentionTransformer(n_features=n_features, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout).to(device)
        logger.info("Params: %d (%.1fK)", sum(param.numel() for param in model.parameters()), sum(param.numel() for param in model.parameters()) / 1000)

        train_loader = self._loader(self.train_ids, window_size, window_stride, batch_size, True)
        val_loader = self._loader(self.val_ids, window_size, window_stride, batch_size, False)
        logger.info("Train windows: %d, Val windows: %d", len(train_loader.dataset), len(val_loader.dataset))

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        criterion = nn.SmoothL1Loss(reduction="none")

        writer = None
        if log_dir:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)

        best_val_loss, no_improve, best_state = float("inf"), 0, {}
        self.train_losses, self.val_losses = [], []
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_total, train_valid_total = 0.0, 0
            for batch in train_loader:
                features = batch["features"].to(device)
                target = batch["retention"].to(device)
                mask = batch["padding_mask"].to(device)
                loss_per_elem = criterion(model(features, src_key_padding_mask=mask), target)
                loss_per_elem[mask] = 0.0
                n_valid = (~mask).sum().clamp(min=1)
                loss = loss_per_elem.sum() / n_valid
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                train_loss_total += loss.item() * n_valid.item()
                train_valid_total += n_valid.item()
            scheduler.step()
            train_loss = train_loss_total / max(train_valid_total, 1)
            self.train_losses.append(train_loss)

            model.eval()
            val_loss_total, val_valid_total = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].to(device)
                    target = batch["retention"].to(device)
                    mask = batch["padding_mask"].to(device)
                    loss_per_elem = criterion(model(features, src_key_padding_mask=mask), target)
                    loss_per_elem[mask] = 0.0
                    n_valid = (~mask).sum().clamp(min=1)
                    val_loss_total += loss_per_elem.sum().item()
                    val_valid_total += n_valid.item()
            val_loss = val_loss_total / max(val_valid_total, 1)
            self.val_losses.append(val_loss)

            if writer:
                writer.add_scalar("SmoothL1/train", train_loss, epoch)
                writer.add_scalar("SmoothL1/val", val_loss, epoch)
            if epoch % 10 == 0 or epoch == 1:
                logger.info("Epoch %3d/%d  train=%.4f  val=%.4f", epoch, epochs, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss, no_improve = val_loss, 0
                best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stop at epoch %d", epoch)
                    break

        if writer:
            writer.close()
        model.load_state_dict(best_state)
        self.model = model

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "n_features": n_features,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "d_ff": d_ff,
                    "dropout": dropout,
                    "window_size": window_size,
                    "feature_names": self.feature_names,
                },
                save_path,
            )
            logger.info("Saved %s", save_path)

        return {
            "train": self._eval(self.train_ids, "train"),
            "val": self._eval(self.val_ids, "val"),
            "best_val_loss": round(best_val_loss, 4),
            "epochs_trained": epoch,
            "elapsed_sec": round(time.time() - t_start, 1),
        }

    def _eval(self, ids, split):
        y_true, y_pred = zip(*[self.predict_video(video_id) for video_id in ids], strict=True)
        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        metrics = calc_retention_metrics(y_true, y_pred)
        logger.info("Transformer %s — MSE=%.4f MAE=%.4f R2=%.4f", split, metrics["mse"], metrics["mae"], metrics["r2"])
        return metrics

    @torch.no_grad()
    def predict_video(self, video_id):
        assert self.model is not None, "Call .train() first"
        device = next(self.model.parameters()).device
        self.model.eval()
        df = self.video_frames[video_id]
        features = df.reindex(columns=self.feature_names, fill_value=0).astype(float).fillna(0).values
        y_true = df["retention"].values.astype(float)
        n_rows, window_size = len(features), self.window_size

        if n_rows <= window_size:
            pred = self.model(torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()[:n_rows]
            return y_true, pred

        pred_sum, pred_count = np.zeros(n_rows), np.zeros(n_rows)
        for start in range(0, n_rows - window_size + 1):
            pred = self.model(torch.tensor(features[start : start + window_size], dtype=torch.float32).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            pred_sum[start : start + window_size] += pred
            pred_count[start : start + window_size] += 1.0
        return y_true, pred_sum / np.maximum(pred_count, 1.0)

    def plot_training_curves(self, output_path=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.train_losses, label="train SmoothL1", color="#2196F3")
        ax.plot(self.val_losses, label="val SmoothL1", color="#FF5722")
        ax.set(xlabel="epoch", ylabel="SmoothL1", title="Transformer training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_predictions(self, output_dir="my_metrics"):
        os.makedirs(output_dir, exist_ok=True)
        paths, all_metrics = [], {}
        for video_id in sorted(self.video_frames):
            y_true, y_pred = self.predict_video(video_id)
            split = "val" if video_id in self.val_ids else "train"
            metrics = calc_retention_metrics(y_true, y_pred)
            all_metrics[video_id] = {**metrics, "split": split}
            time_axis = np.arange(len(y_true))
            fig, (retention_ax, residual_ax) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)
            retention_ax.plot(time_axis, y_true, color="#2196F3", label="actual")
            retention_ax.plot(time_axis, y_pred, color="#FF5722", label="predicted", alpha=0.8)
            retention_ax.fill_between(time_axis, y_true, y_pred, alpha=0.1, color="#9C27B0")
            retention_ax.set(ylabel="Retention (%)", title=f"{video_id} [{split}] MSE={metrics['mse']:.2f} MAE={metrics['mae']:.2f}")
            retention_ax.legend(fontsize=9)
            retention_ax.grid(True, alpha=0.3)
            residual = y_pred - y_true
            residual_ax.fill_between(time_axis, residual, alpha=0.3, color="#4CAF50", where=residual >= 0)
            residual_ax.fill_between(time_axis, residual, alpha=0.3, color="#F44336", where=residual < 0)
            residual_ax.axhline(0, color="black", linewidth=0.5)
            residual_ax.set(xlabel="sec", ylabel="error (pp)")
            residual_ax.grid(True, alpha=0.3)
            plt.tight_layout()
            output_path = os.path.join(output_dir, "videos", video_id, "prediction", "transformer_pred.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(output_path)
        self._plot_mae_summary(all_metrics, output_dir)
        logger.info("Saved %d plots to %s", len(paths), output_dir)
        return paths

    def _plot_mae_summary(self, all_metrics, output_dir):
        if not all_metrics:
            return
        rows = sorted(all_metrics.items(), key=lambda item: item[1]["mae"])
        video_ids = [video_id for video_id, _ in rows]
        maes = [metrics["mae"] for _, metrics in rows]
        splits = [metrics["split"] for _, metrics in rows]
        overall = float(np.mean(maes))
        val_maes = [mae for mae, split in zip(maes, splits, strict=True) if split == "val"]
        train_maes = [mae for mae, split in zip(maes, splits, strict=True) if split == "train"]
        val_mean = float(np.mean(val_maes)) if val_maes else float("nan")
        train_mean = float(np.mean(train_maes)) if train_maes else float("nan")

        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(video_ids) * 0.28)), gridspec_kw={"width_ratios": [2, 1]})
        ax = axes[0]
        y_pos = np.arange(len(video_ids))
        colors = ["#FF5722" if split == "val" else "#2196F3" for split in splits]
        ax.barh(y_pos, maes, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(video_ids, fontsize=7)
        ax.axvline(overall, color="#F44336", linestyle="--", linewidth=1.5, label=f"mean={overall:.2f}")
        for idx, value in enumerate(maes):
            ax.text(value + 0.05, idx, f"{value:.2f}", va="center", fontsize=6.5)
        ax.set(xlabel="MAE", title=f"Transformer — MAE per video  (overall={overall:.2f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

        ax2 = axes[1]
        names = ["train", "val", "all"]
        values = [train_mean, val_mean, overall]
        colors = ["#2196F3", "#FF5722", "#4CAF50"]
        bars = ax2.bar(names, values, color=colors, edgecolor="white")
        for bar, value in zip(bars, values, strict=True):
            if np.isfinite(value):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax2.set(ylabel="MAE", title="Transformer — MAE by split")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "mae_summary.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        pd.DataFrame(
            [
                {
                    "video_id": video_id,
                    "split": all_metrics[video_id]["split"],
                    "mae": all_metrics[video_id]["mae"],
                    "mse": all_metrics[video_id]["mse"],
                    "r2": all_metrics[video_id]["r2"],
                }
                for video_id in video_ids
            ]
        ).to_csv(os.path.join(output_dir, "mae_summary.csv"), index=False)
        logger.info("Transformer MAE summary: overall=%.3f  train=%.3f  val=%.3f", overall, train_mean, val_mean)
