from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID
from train.loo.train_retention_regressor_loo import run_experiment as run_regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("LOO-эксперимент с integration-aware постобработкой: базовый regressor + явный штраф на интервале интеграции со спадом и восстановлением к концу интеграции.")
    )
    parser.add_argument("--env-file", default=".env", help="Путь к .env.")
    parser.add_argument("--snapshot-dir", default="drive_snapshot_90", help="Локальный снапшот с features_llm.json и retention_parsed.json.")
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID, help="ID корневой папки на Google Drive (если snapshot-dir пуст).")
    parser.add_argument("--limit-videos", type=int, default=45)
    parser.add_argument("--train-videos", type=int, default=44)
    parser.add_argument("--curve-points", type=int, default=50)
    parser.add_argument("--eval-video-folder", default="")
    parser.add_argument("--eval-drive-file-id", default="")
    parser.add_argument("--output-dir", default="integration_penalty_experiment")

    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--delta-blend", type=float, default=0.65)
    parser.add_argument("--delta-max-step", type=float, default=0.25)

    parser.add_argument("--integration-max-drop", type=float, default=0.08, help="Максимальная глубина штрафа на активном интервале интеграции.")
    parser.add_argument("--integration-strength-power", type=float, default=1.0, help="Степень влияния integration_strength на амплитуду штрафа.")
    parser.add_argument("--integration-active-threshold", type=float, default=0.5, help="Порог активности integration_present для выбора сегмента штрафа.")
    parser.add_argument("--integration-end-lift", type=float, default=0.35, help="Коэффициент подъема в последних точках интеграции.")
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    return parser.parse_args()


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value).strip())
    return cleaned or "unnamed"


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=float), 0.0, 1.0)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_features_payload(snapshot_dir: Path, video_folder: str, transcript_path: str) -> dict[str, Any]:
    item_dir = snapshot_dir / _safe_name(video_folder) / _safe_name(transcript_path)
    features_path = item_dir / "features_llm.json"
    if not features_path.exists():
        return {}
    try:
        payload = json.loads(features_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _fallback_first_features_payload(snapshot_dir: Path, video_folder: str) -> dict[str, Any]:
    video_dir = snapshot_dir / _safe_name(video_folder)
    if not video_dir.exists():
        return {}
    for child in sorted(video_dir.iterdir()):
        if not child.is_dir():
            continue
        feature_candidate = child / "features_llm.json"
        if not feature_candidate.exists():
            continue
        try:
            payload = json.loads(feature_candidate.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _integration_strength_from_payload(payload: dict[str, Any], curve_points: int) -> np.ndarray:
    if not payload:
        return np.zeros((curve_points,), dtype=float)

    source = payload.get("source", {}) if isinstance(payload.get("source"), dict) else {}
    duration = _to_float(source.get("duration_seconds", 0.0), 0.0)
    text_features = payload.get("text_features", {}) if isinstance(payload.get("text_features"), dict) else {}
    integration = text_features.get("integration_feature", {}) if isinstance(text_features.get("integration_feature"), dict) else {}
    series = integration.get("series", {}) if isinstance(integration.get("series"), dict) else {}
    rows = series.get("integration_present", []) if isinstance(series.get("integration_present"), list) else []
    if not rows:
        return np.zeros((curve_points,), dtype=float)

    strengths = np.zeros((curve_points,), dtype=float)
    if duration > 1e-6:
        for i in range(curve_points):
            frac = i / max(1, curve_points - 1)
            t = frac * duration
            value = 0.0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                start = _to_float(row.get("start", 0.0), 0.0)
                end = _to_float(row.get("end", 0.0), 0.0)
                if end < start:
                    start, end = end, start
                if start <= t <= end:
                    value = _to_float(row.get("value", 0.0), 0.0)
                    break
            strengths[i] = np.clip(value, 0.0, 1.0)
        return strengths

    values: list[float] = []
    for row in rows:
        if isinstance(row, dict):
            values.append(np.clip(_to_float(row.get("value", 0.0), 0.0), 0.0, 1.0))
    if not values:
        return strengths
    if len(values) == 1:
        strengths[:] = values[0]
        return strengths
    x_old = np.linspace(0.0, 1.0, num=len(values))
    x_new = np.linspace(0.0, 1.0, num=curve_points)
    strengths = np.interp(x_new, x_old, np.asarray(values, dtype=float))
    return _clip01(strengths)


def _largest_active_segment(strength: np.ndarray, threshold: float) -> tuple[int, int] | None:
    active = strength >= float(np.clip(threshold, 0.0, 1.0))
    if not np.any(active):
        return None
    best = None
    run_start = None
    for i, flag in enumerate(active):
        if flag and run_start is None:
            run_start = i
        if not flag and run_start is not None:
            seg = (run_start, i - 1)
            if best is None or (seg[1] - seg[0]) > (best[1] - best[0]):
                best = seg
            run_start = None
    if run_start is not None:
        seg = (run_start, len(active) - 1)
        if best is None or (seg[1] - seg[0]) > (best[1] - best[0]):
            best = seg
    return best


def _apply_integration_penalty(
    y_pred: np.ndarray, strength: np.ndarray, max_drop: float, strength_power: float, active_threshold: float, end_lift: float
) -> tuple[np.ndarray, np.ndarray]:
    y_base = _clip01(y_pred)
    penalty = np.zeros_like(y_base, dtype=float)

    seg = _largest_active_segment(strength, threshold=active_threshold)
    if seg is None:
        return y_base.copy(), penalty

    s, e = seg
    if e <= s:
        return y_base.copy(), penalty

    drop_cap = float(max(0.0, max_drop))
    pwr = float(max(0.25, strength_power))
    for i in range(s, e + 1):
        amp = float(np.clip(strength[i], 0.0, 1.0)) ** pwr
        penalty[i] = drop_cap * amp

    y_adj = _clip01(y_base - penalty)
    _ = end_lift
    return y_adj, penalty


def _curve_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    y_pred = _clip01(y_pred)
    y_true = _clip01(y_true)
    abs_err = np.abs(y_pred - y_true)
    d_pred = np.diff(y_pred)
    d_true = np.diff(y_true)
    dd_pred = np.diff(y_pred, n=2)
    dd_true = np.diff(y_true, n=2)
    return {
        "spearman": float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")),
        "pearson": float(pd.Series(y_pred).corr(pd.Series(y_true), method="pearson")),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(abs_err)),
        "spike_rmse": float(np.sqrt(np.mean((d_pred - d_true) ** 2))) if d_pred.size else 0.0,
        "curvature_rmse": float(np.sqrt(np.mean((dd_pred - dd_true) ** 2))) if dd_pred.size else 0.0,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    base_args = SimpleNamespace(
        env_file=args.env_file,
        snapshot_dir=args.snapshot_dir,
        root_folder_id=args.root_folder_id,
        limit_videos=args.limit_videos,
        train_videos=args.train_videos,
        curve_points=args.curve_points,
        eval_video_folder=args.eval_video_folder,
        eval_drive_file_id=args.eval_drive_file_id,
        output_dir=args.output_dir,
        iterations=int(getattr(args, "iterations", 700)),
        learning_rate=float(getattr(args, "learning_rate", 0.05)),
        depth=int(getattr(args, "depth", 6)),
        random_seed=int(getattr(args, "random_seed", 42)),
        delta_blend=float(getattr(args, "delta_blend", 0.65)),
        delta_max_step=float(getattr(args, "delta_max_step", 0.25)),
        task_type=getattr(args, "task_type", "GPU"),
        gpu_ram_part=float(getattr(args, "gpu_ram_part", 0.6)),
    )
    metrics = run_regressor(base_args)

    pred_path = Path(args.output_dir) / "holdout_prediction_vs_true.csv"
    if not pred_path.exists():
        return metrics
    df = pd.read_csv(pred_path)
    if "pred_retention" not in df.columns:
        return metrics

    y_pred_base = df["pred_retention"].to_numpy(dtype=float)
    curve_points = len(y_pred_base)
    video_folder = str(metrics.get("test_video", "")).strip()
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser()

    payload = _load_features_payload(snapshot_dir, video_folder=video_folder, transcript_path="transcripts")
    if not payload:
        payload = _fallback_first_features_payload(snapshot_dir, video_folder=video_folder)

    strength = _integration_strength_from_payload(payload, curve_points=curve_points)
    y_pred_adj, penalty = _apply_integration_penalty(
        y_pred=y_pred_base,
        strength=strength,
        max_drop=float(getattr(args, "integration_max_drop", 0.08)),
        strength_power=float(getattr(args, "integration_strength_power", 1.0)),
        active_threshold=float(getattr(args, "integration_active_threshold", 0.5)),
        end_lift=float(getattr(args, "integration_end_lift", 0.35)),
    )

    df["pred_retention_base"] = y_pred_base
    df["integration_strength"] = strength
    df["integration_penalty"] = penalty
    df["pred_retention"] = y_pred_adj
    df["pred_retention_norm"] = y_pred_adj
    df["pred_score_raw"] = y_pred_adj
    if "true_retention" in df.columns:
        true = df["true_retention"].to_numpy(dtype=float)
        df["abs_error"] = np.abs(y_pred_adj - true)

    df.to_csv(pred_path, index=False)

    if "true_retention" in df.columns:
        y_true = df["true_retention"].to_numpy(dtype=float)
        metrics.update(_curve_metrics(y_pred_adj, y_true))

    metrics["integration_penalty_enabled"] = 1
    metrics["integration_max_drop"] = float(getattr(args, "integration_max_drop", 0.08))
    metrics["integration_strength_power"] = float(getattr(args, "integration_strength_power", 1.0))
    metrics["integration_active_threshold"] = float(getattr(args, "integration_active_threshold", 0.5))
    metrics["integration_end_lift"] = float(getattr(args, "integration_end_lift", 0.35))
    metrics_path = Path(args.output_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Integration-Penalty LOO ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
