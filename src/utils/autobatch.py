import torch

from .logger import Logger


MEMORY_PER_SAMPLE_MB = {"default": 100, "clip": 70, "emotion": 60, "yolo": 140, "transnet": 50, "ocr": 200}


def _get_gpu_memory_gb() -> float:
    try:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return 0.0


def _get_free_memory_gb() -> float:
    try:
        if not torch.cuda.is_available():
            return 0.0
        torch.cuda.synchronize()
        if hasattr(torch.cuda, "mem_get_info"):
            return torch.cuda.mem_get_info(0)[0] / (1024**3)
        return _get_gpu_memory_gb()
    except Exception:
        return _get_gpu_memory_gb()


def autobatch(fraction: float = 0.6, memory_per_sample_mb: float = 80, min_batch: int = 2, max_batch: int = 128) -> int:
    gb = _get_gpu_memory_gb()
    if gb <= 0:
        return min_batch

    free_gb = _get_free_memory_gb()
    target_gb = min(gb * fraction, free_gb * 0.95) if free_gb > 0 else gb * fraction
    batch = max(min_batch, min(max_batch, int(max(0.5, target_gb - 1.0) * 1024 / memory_per_sample_mb)))
    for candidate in [64, 48, 32, 24, 16, 12, 8, 6, 4, 2]:
        if batch >= candidate:
            batch = candidate
            break
    else:
        batch = max(min_batch, 2)
    _log_autobatch(batch, fraction, gb)
    return batch


def _log_autobatch(batch: int, fraction: float, gb: float) -> None:
    Logger(show=True).get_logger().info("AutoBatch: using batch_size=%d (%.0f%% of %.1fG GPU)", batch, fraction * 100, gb)


def resolve_batch_size(value: int | float | None, default: int = 4, task: str = "default") -> int:
    if value >= 1:
        return int(value)
    if value == -1:
        fraction = 0.6
    elif 0 < value < 1:
        fraction = float(value)
    else:
        return max(1, default)
    result = autobatch(fraction=fraction, memory_per_sample_mb=MEMORY_PER_SAMPLE_MB.get(task, MEMORY_PER_SAMPLE_MB["default"]), min_batch=2, max_batch=128)
    return max(1, result)
