from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from tqdm import tqdm

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID, _point_col, _safe_float, build_rows_with_targets_source, make_feature_matrix, select_train_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("LOO-модель: pointwise regressor с усиленным весом пиков/просадок и postprocess-понижением retention на рекламных интервалах."))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--snapshot-dir", default="drive_snapshot_90")
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    parser.add_argument("--limit-videos", type=int, default=45)
    parser.add_argument("--train-videos", type=int, default=44)
    parser.add_argument("--curve-points", type=int, default=50)
    parser.add_argument("--eval-video-folder", default="")
    parser.add_argument("--eval-drive-file-id", default="")
    parser.add_argument("--output-dir", default="ad_peak_weighted_experiment")

    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--weight-slope-alpha", type=float, default=1.4)
    parser.add_argument("--weight-curvature-alpha", type=float, default=0.9)
    parser.add_argument("--weight-ad-alpha", type=float, default=1.2)
    parser.add_argument("--weight-max", type=float, default=6.0)

    parser.add_argument("--integration-max-drop", type=float, default=0.10)
    parser.add_argument("--integration-strength-power", type=float, default=1.0)
    parser.add_argument("--integration-active-threshold", type=float, default=0.5)
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


def _load_features_payload(snapshot_dir: Path, video_folder: str) -> dict[str, Any]:
    video_dir = snapshot_dir / _safe_name(video_folder)
    if not video_dir.exists():
        return {}

    candidate = video_dir / "transcripts" / "features_llm.json"
    if candidate.exists():
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    for child in sorted(video_dir.iterdir()):
        if not child.is_dir():
            continue
        p = child / "features_llm.json"
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
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
        for i in tqdm(range(curve_points), desc="Training points"):
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
    return _clip01(np.interp(x_new, x_old, np.asarray(values, dtype=float)))


def _get_target_matrix(df: pd.DataFrame, curve_points: int) -> np.ndarray:
    mat = np.zeros((len(df), curve_points), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        for p in range(curve_points):
            mat[i, p] = _safe_float(row[_point_col(p)], 0.0)
    return _clip01(mat)


def _point_weights(y_train: np.ndarray, ad_train: np.ndarray, point_idx: int, slope_alpha: float, curvature_alpha: float, ad_alpha: float, w_max: float) -> np.ndarray:
    n, points = y_train.shape
    if n == 0:
        return np.zeros((0,), dtype=float)

    prev_idx = max(0, point_idx - 1)
    prev2_idx = max(0, point_idx - 2)
    slope = np.abs(y_train[:, point_idx] - y_train[:, prev_idx])
    curvature = np.abs(y_train[:, point_idx] - 2.0 * y_train[:, prev_idx] + y_train[:, prev2_idx])
    ad = np.clip(ad_train[:, point_idx], 0.0, 1.0)

    slope_scale = float(np.median(slope[slope > 1e-9])) if np.any(slope > 1e-9) else 1.0
    curv_scale = float(np.median(curvature[curvature > 1e-9])) if np.any(curvature > 1e-9) else 1.0

    slope_norm = slope / max(1e-6, slope_scale)
    curv_norm = curvature / max(1e-6, curv_scale)
    w = 1.0 + float(slope_alpha) * slope_norm + float(curvature_alpha) * curv_norm + float(ad_alpha) * ad
    return np.clip(w, 0.2, max(1.0, float(w_max)))


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


def _apply_ad_drop(y_pred_base: np.ndarray, ad_strength: np.ndarray, max_drop: float, strength_power: float, active_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    y_pred_base = _clip01(y_pred_base)
    ad_strength = _clip01(ad_strength)
    active = ad_strength >= float(np.clip(active_threshold, 0.0, 1.0))
    pwr = float(max(0.25, strength_power))
    penalty = np.zeros_like(y_pred_base, dtype=float)
    penalty[active] = float(max(0.0, max_drop)) * (ad_strength[active] ** pwr)
    return _clip01(y_pred_base - penalty), penalty


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser() if str(args.snapshot_dir).strip() else None
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snapshot_dir)
    all_df, train_df, test_df = select_train_test(rows, args)

    x_train = make_feature_matrix(train_df).reset_index(drop=True)
    x_test = make_feature_matrix(test_df).reset_index(drop=True)
    if x_train.empty:
        raise RuntimeError("Пустая матрица признаков для ad_peak_weighted")

    y_train = _get_target_matrix(train_df, args.curve_points)
    y_true = _get_target_matrix(test_df, args.curve_points)[0]

    train_ad: list[np.ndarray] = []
    for _, row in train_df.iterrows():
        payload = _load_features_payload(Path(args.snapshot_dir), str(row.get("video_folder", "")))
        train_ad.append(_integration_strength_from_payload(payload, curve_points=args.curve_points))
    ad_train_mat = np.vstack(train_ad) if train_ad else np.zeros((len(train_df), args.curve_points), dtype=float)

    test_payload = _load_features_payload(Path(args.snapshot_dir), str(test_df.iloc[0].get("video_folder", "")))
    ad_test = _integration_strength_from_payload(test_payload, curve_points=args.curve_points)

    y_pred_base = np.zeros((args.curve_points,), dtype=float)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "point_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(range(args.curve_points), desc="Training points"):
        target = y_train[:, p]
        w = _point_weights(
            y_train=y_train,
            ad_train=ad_train_mat,
            point_idx=p,
            slope_alpha=float(args.weight_slope_alpha),
            curvature_alpha=float(args.weight_curvature_alpha),
            ad_alpha=float(args.weight_ad_alpha),
            w_max=float(args.weight_max),
        )
        if target.size == 0:
            y_pred_base[p] = 0.0
            continue
        if float(np.var(target)) < 1e-12:
            y_pred_base[p] = float(target[0])
            continue
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            random_seed=args.random_seed + p,
            task_type=getattr(args, "task_type", "GPU"),
            gpu_ram_part=getattr(args, "gpu_ram_part", 0.6),
            verbose=False,
        )
        model.fit(x_train, target, sample_weight=w)
        y_pred_base[p] = float(model.predict(x_test)[0])
        model.save_model(str(models_dir / f"catboost_point_{p:03d}.cbm"))

    y_pred_base = _clip01(y_pred_base)
    y_pred, penalty = _apply_ad_drop(
        y_pred_base=y_pred_base,
        ad_strength=ad_test,
        max_drop=float(args.integration_max_drop),
        strength_power=float(args.integration_strength_power),
        active_threshold=float(args.integration_active_threshold),
    )
    abs_err = np.abs(y_pred - y_true)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(args.curve_points)),
            "point_frac": np.linspace(0.0, 1.0, args.curve_points),
            "integration_strength": ad_test,
            "integration_penalty": penalty,
            "pred_retention_base": y_pred_base,
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    dataset_path = out_dir / "ad_peak_weighted_dataset.csv"
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    metrics_path = out_dir / "metrics.json"
    all_df.to_csv(dataset_path, index=False)
    result_df.to_csv(pred_path, index=False)

    cm = _curve_metrics(y_pred, y_true)
    metrics = {
        "videos_total_with_target": len(rows),
        "videos_used": len(all_df),
        "train_videos": len(train_df),
        "curve_points": int(args.curve_points),
        "test_video": str(test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(test_df.iloc[0]["drive_file_id"]),
        "weight_slope_alpha": float(args.weight_slope_alpha),
        "weight_curvature_alpha": float(args.weight_curvature_alpha),
        "weight_ad_alpha": float(args.weight_ad_alpha),
        "weight_max": float(args.weight_max),
        "integration_max_drop": float(args.integration_max_drop),
        "integration_strength_power": float(args.integration_strength_power),
        "integration_active_threshold": float(args.integration_active_threshold),
        "dataset_path": str(dataset_path),
        "prediction_path": str(pred_path),
        "models_dir": str(models_dir),
        **cm,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retention Ad-Peak-Weighted LOO ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
