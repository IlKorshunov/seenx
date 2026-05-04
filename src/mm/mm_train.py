import argparse
import os
import random
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoConfig

from ..aggregator import get_retention
from ..utils.logger import Logger
from .mm_arch import encode_images_or_videos, temporal_aggregator
from .mm_constants import NUM_FRAMES
from .mm_models import build_audio_tower, build_vision_tower
from .mm_processing import process_video
from .mm_projector import build_audio_projector, build_vision_projector


logger = Logger(show=True).get_logger()


def _resolve_device():
    from ..utils.config import Config

    return torch.device(Config("configs/local.json").get("device"))


device = _resolve_device()
working_dir = Path(__file__).resolve().parent.parent.parent.parent
logger.info(f"{__file__} working directory: {working_dir}")


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def build_mm_components(model_name: str = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV", device: torch.device = device):
    logger.info("Initializing multimodal components...")

    config = AutoConfig.from_pretrained(model_name)

    config.num_frames = NUM_FRAMES
    config.mm_vision_tower = "google/siglip-so400m-patch14-384"
    config.mm_projector_type = "stc_connector_v35"
    config.mm_audio_tower = "audio_tower.bin"
    config.mm_projector_a_type = "mlp2x_gelu"

    vision_tower = build_vision_tower(config).half().to(device)
    vision_tower.requires_grad_(False)

    audio_tower, _ = build_audio_tower(config)
    audio_tower = audio_tower.half().to(device)
    audio_tower.requires_grad_(False)

    vision_projector = build_vision_projector(config).to(device)
    audio_projector = build_audio_projector(config).to(device)

    unfreeze_module(vision_projector)
    unfreeze_module(audio_projector)

    processor = partial(process_video, processor=vision_tower.image_processor, aspect_ratio=None, num_frames=config.num_frames if hasattr(config, "num_frames") else NUM_FRAMES)

    logger.info("Multimodal components initialized.")

    return {
        "config": config,
        "vision_tower": vision_tower,
        "audio_tower": audio_tower,
        "vision_projector": vision_projector,
        "audio_projector": audio_projector,
        "processor": processor,
    }


class MultiVideoRetentionDataset(Dataset):
    def __init__(self, videos: list, processor, interval_len: int = 10, stride: int = 5):
        self.processor = processor
        self.samples = []

        logger.info("Building multi-video dataset...")

        for vid in videos:
            video_path = vid["video_path"]
            html_path = vid["html_path"]

            retention_df = get_retention(video_path, html_path)
            retention = retention_df.values.astype("float32")

            T = retention.shape[0]

            for start in range(0, T - interval_len, stride):
                end = start + int(interval_len)

                self.samples.append({"video_path": video_path, "start": start, "end": end, "retention": retention[start:end]})

        logger.info(f"Dataset size: {len(self.samples)} intervals")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        data = self.processor(s["video_path"], s=s["start"], e=s["end"], va=True)

        return {"video": data["video"].half(), "audio": data["audio"].half(), "retention": torch.tensor(s["retention"])}


def run_epoch(
    dataloader: DataLoader,
    vision_tower,
    audio_tower,
    vision_projector,
    audio_projector,
    regressor,
    config,
    criterion,
    optimizer=None,
    train: bool = True,
    device: torch.device = device,
):
    if train:
        vision_projector.train()
        audio_projector.train()
        regressor.train()
        desc = "Train"
    else:
        vision_projector.eval()
        audio_projector.eval()
        regressor.eval()
        desc = "Val"

    n = 0
    epoch_loss = 0.0
    mse_sum = mae_sum = 0.0

    for batch in tqdm(dataloader, desc=desc):
        video_batch = batch["video"].to(device)
        audio_batch = batch["audio"].to(device)
        target = torch.cat([r for r in batch["retention"]], dim=0).to(device)

        with torch.no_grad():
            video_feats = encode_images_or_videos(vision_tower, [(v, "video") for v in video_batch], config)
            audio_batch = audio_batch.flatten(0, 1)
            audio_padding_mask = torch.zeros_like(audio_batch, dtype=torch.bool)
            audio_feats, _, _ = audio_tower.extract_features(audio_batch, padding_mask=audio_padding_mask)

        with torch.amp.autocast(device_type=device.type, enabled=True):
            video_feats = temporal_aggregator(vision_projector, config, video_feats.float())

            audio_features = audio_projector(audio_feats.float())
            audio_features = audio_features.view(audio_batch.size(0), -1, audio_features.size(-1))

            mm_feats = torch.cat([video_feats, audio_features], dim=1)
            pred = regressor(mm_feats).mean(dim=1)

            if torch.isnan(pred).any():
                logger.error("NaN values found in predictions!")
                logger.error(f"Pred shape: {pred.shape}")
                logger.error(f"Target shape: {target.shape}")

            loss = criterion(pred, target)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * pred.size(0)

        diff = pred.detach() - target
        mse_sum += (diff**2).sum().item()
        mae_sum += diff.abs().sum().item()
        n += pred.size(0)

    avg_loss = epoch_loss / n
    mse = mse_sum / n
    mae = mae_sum / n
    r2 = 1.0 - mse / (target.var().item() + 1e-6)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return avg_loss, mse, mae, r2


def sample_epoch_videos(data: list[dict], n_total: int = 4, val_ratio: float = 0.25):
    assert len(data) >= n_total, "Not enough videos to sample from"

    sampled = random.sample(data, n_total)
    n_val = max(1, int(n_total * val_ratio))

    val_videos = sampled[:n_val]
    train_videos = sampled[n_val:]

    return train_videos, val_videos


def train(
    data: list[dict],
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-4,
    save_dir: str = "train/saved_models",
    log_dir: str = "train/tensorboard_logs",
    resume_from: str | None = None,
):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(working_dir, save_dir, run_id)
    log_dir = os.path.join(working_dir, log_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger.info(f"Model checkpoints will be saved to: {save_dir}")
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")

    writer = SummaryWriter(log_dir=log_dir)

    train_videos, val_videos = train_test_split(data, test_size=0.2, random_state=42)
    logger.info(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

    mm = build_mm_components()
    config = mm["config"]
    vision_tower = mm["vision_tower"]
    audio_tower = mm["audio_tower"]
    vision_projector = mm["vision_projector"]
    audio_projector = mm["audio_projector"]
    processor = mm["processor"]

    regressor = nn.Sequential(nn.Linear(3584, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(vision_projector.parameters()) + list(audio_projector.parameters()) + list(regressor.parameters()), lr=lr)

    start_epoch = 0

    if resume_from is not None:
        logger.info(f"Loading checkpoint from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)

        vision_projector.load_state_dict(ckpt["vision_projector"])
        audio_projector.load_state_dict(ckpt["audio_projector"])
        regressor.load_state_dict(ckpt["regressor"])

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

        start_epoch = ckpt.get("epoch", 0)
        logger.info(f"Resumed from epoch {start_epoch}")

    logger.info("Starting training...")
    for epoch in range(start_epoch, epochs):
        train_videos, val_videos = sample_epoch_videos(data, n_total=4, val_ratio=0.25)
        logger.info(f"Epoch {epoch + 1}: Train videos={len(train_videos)}, Val videos={len(val_videos)}")
        train_dataset = MultiVideoRetentionDataset(train_videos, processor=processor, interval_len=1)
        val_dataset = MultiVideoRetentionDataset(val_videos, processor=processor, interval_len=1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        train_loss, train_mse, train_mae, train_r2 = run_epoch(
            train_loader, vision_tower, audio_tower, vision_projector, audio_projector, regressor, config, criterion, optimizer, train=True
        )
        val_loss, val_mse, val_mae, val_r2 = run_epoch(val_loader, vision_tower, audio_tower, vision_projector, audio_projector, regressor, config, criterion, train=False)

        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch + 1)
        writer.add_scalars("MSE", {"Train": train_mse, "Val": val_mse}, epoch + 1)
        writer.add_scalars("MAE", {"Train": train_mae, "Val": val_mae}, epoch + 1)
        writer.add_scalars("R2", {"Train": train_r2, "Val": val_r2}, epoch + 1)

        logger.info(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f} | "
            f"Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "vision_projector": vision_projector.state_dict(),
                "audio_projector": audio_projector.state_dict(),
                "regressor": regressor.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(save_dir, f"epoch_{epoch + 1}.pt"),
        )
        logger.info(f"Saved model weights for epoch {epoch + 1}")

    writer.close()
    logger.info("Training complete.")
    return vision_projector, audio_projector, regressor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--html_path", type=str, required=True)
    args = parser.parse_args()
    data = [{"video_path": args.video_path, "html_path": args.html_path}]
    train(data, epochs=5, batch_size=4, lr=1e-4)
