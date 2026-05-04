from __future__ import annotations

import gc
import math

import cv2
import numpy as np
import torch
from transformers import ViTImageProcessor, ViTModel

from ..feature_extractor import VideoFeature
from .common import get_capture_fps, open_video_capture
from .constants import DINO_INPUT_SIZE, DINO_MODEL_ID, DINO_PATCH_SIZE, SALIENCY_FALLBACK_FPS


_GRID = DINO_INPUT_SIZE // DINO_PATCH_SIZE  # 28


class SaliencyFeature(VideoFeature):
    def produces_keys(self) -> set[str]:
        return {"saliency_mean", "saliency_spread", "saliency_face_overlap"}

    def run(self, video_path: str, context: dict):
        df = context["data"]
        dur = len(df)
        device = torch.device(self.config.get("device", "cpu"))
        use_fp16 = device.type == "cuda"

        for col in self.produces_keys():
            df[col] = 0.0

        model, processor = _load_model(device)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        with open_video_capture(video_path) as cap:
            fps = get_capture_fps(cap, fallback=SALIENCY_FALLBACK_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for sec in range(dur):
                target_frame = int(sec * fps + fps / 2)
                if target_frame >= total_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ok, frame = cap.read()
                if not ok:
                    continue

                h_orig, w_orig = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                attn_map = _compute_saliency(frame_rgb, model, processor, device, use_fp16)

                df.at[sec, "saliency_mean"] = float(attn_map.mean())

                flat = attn_map.flatten()
                flat = flat / (flat.sum() + 1e-12)
                entropy = float(-np.sum(flat * np.log(flat + 1e-12)))
                max_entropy = np.log(flat.shape[0])
                df.at[sec, "saliency_spread"] = entropy / max_entropy if max_entropy > 0 else 0.0

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
                if len(faces) > 0:
                    attn_full = cv2.resize(attn_map, (w_orig, h_orig))
                    face_mask = np.zeros_like(attn_full)
                    for x, y, w, fh in faces:
                        face_mask[y : y + fh, x : x + w] = 1.0
                    total_sal = attn_full.sum() + 1e-12
                    face_sal = (attn_full * face_mask).sum()
                    df.at[sec, "saliency_face_overlap"] = float(face_sal / total_sal)

        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_model(device: torch.device):
    processor = ViTImageProcessor.from_pretrained(DINO_MODEL_ID)
    model = ViTModel.from_pretrained(DINO_MODEL_ID).to(device)
    model.eval()
    return model, processor


def _compute_saliency(frame_rgb: np.ndarray, model, processor, device: torch.device, use_fp16: bool) -> np.ndarray:
    inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_fp16):
        outputs = model(**inputs, output_attentions=True)

    attn_last = outputs.attentions[-1]
    cls_attn = attn_last[0, :, 0, 1:]
    saliency = cls_attn.mean(dim=0)
    grid = math.isqrt(saliency.shape[0])
    attn_map = saliency.reshape(grid, grid).cpu().numpy()

    attn_min, attn_max = attn_map.min(), attn_map.max()
    if attn_max > attn_min:
        attn_map = (attn_map - attn_min) / (attn_max - attn_min)

    return attn_map
