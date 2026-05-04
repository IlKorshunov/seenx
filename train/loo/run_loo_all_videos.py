from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from train.common.retention_data_layer import _point_col, _safe_float, build_rows_with_targets_source
from train.loo.train_retention_ad_peak_weighted_loo import run_experiment as run_ad_peak_weighted
from train.loo.train_retention_blended_quantile_loo import run_experiment as run_blended_quantile
from train.loo.train_retention_hybrid_loo import run_experiment as run_hybrid
from train.loo.train_retention_integration_penalty_loo import run_experiment as run_integration_penalty
from train.loo.train_retention_local_knn_loo import run_experiment as run_local_knn
from train.loo.train_retention_meta_ensemble_loo import run_experiment as run_meta_ensemble
from train.loo.train_retention_peak_weighted_loo import run_experiment as run_peak_weighted
from train.loo.train_retention_ranker_loo import run_experiment as run_ranker
from train.loo.train_retention_regressor_loo import run_experiment as run_regressor
from train.loo.train_retention_residual_huber_loo import run_experiment as run_residual_huber
from train.loo.train_retention_shape_only_loo import run_experiment as run_shape_only
from train.loo.train_retention_stacked_loo import run_experiment as run_stacked


SNAPSHOT_DIR = "drive_snapshot_90"
ROOT_FOLDER_ID = "1aIqGRHTsO9kNBrOXRRz9XV8kD0Ru8zSV"
ENV_FILE = ".env"

CURVE_POINTS = 50
RANDOM_SEED = 42
OUTPUT_ROOT = "loo_all_videos"
MAX_WORKERS = 2
RESUME_FROM_EXISTING = True

MODEL_NAMES = [
    "baseline_cached",
    "local_knn",
    "shape_only",
    "regressor",
    "regressor_integration_penalty",
    "ad_peak_weighted",
    "peak_weighted",
    "ranker",
    "hybrid",
    "stacked",
    "residual_huber",
    "meta_ensemble",
    "blended_quantile",
]


def _ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _curve_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    y_true = np.clip(np.asarray(y_true, dtype=float), 0.0, 1.0)
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


def _load_pred_true_from_csv(pred_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(pred_path)
    if "pred_retention" in df.columns:
        y_pred = df["pred_retention"].to_numpy(dtype=float)
    elif "pred_retention_norm" in df.columns:
        y_pred = df["pred_retention_norm"].to_numpy(dtype=float)
    else:
        raise RuntimeError(f"В {pred_path} нет pred_retention/pred_retention_norm")
    if "true_retention" not in df.columns:
        raise RuntimeError(f"В {pred_path} нет true_retention")
    y_true = df["true_retention"].to_numpy(dtype=float)
    return y_pred, y_true


def _runner_and_base_args(model_name: str) -> tuple[Callable[[Any], dict[str, Any]], dict[str, Any]]:
    if model_name == "baseline_cached":
        return (lambda _: {}), {}
    common = {"env_file": ENV_FILE, "snapshot_dir": SNAPSHOT_DIR, "root_folder_id": ROOT_FOLDER_ID, "curve_points": CURVE_POINTS}
    if model_name == "local_knn":
        return run_local_knn, {
            **common,
            "neighbors_k": 12,
            "distance_temperature": 1.0,
            "residual_strength": 0.70,
            "spike_gain": 1.10,
            "smooth_window": 7,
            "auto_tune": True,
            "tune_folds": 5,
        }
    if model_name == "shape_only":
        return run_shape_only, {
            **common,
            "iterations": 700,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": RANDOM_SEED,
            "shape_anchor_points": 1,
            "fixed_anchor_value": 1.0,
            "delta_max_step": 0.24,
            "curvature_max_step": 0.18,
            "curvature_blend": 0.78,
            "shape_max": 1.30,
            "spike_sensitivity": 1.45,
            "spike_threshold_quantile": 0.70,
            "spike_max_amplify": 1.8,
        }
    if model_name == "regressor":
        return run_regressor, {**common, "iterations": 700, "learning_rate": 0.05, "depth": 6, "random_seed": RANDOM_SEED, "delta_blend": 0.65, "delta_max_step": 0.25}
    if model_name == "regressor_integration_penalty":
        return run_integration_penalty, {
            **common,
            "iterations": 700,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": RANDOM_SEED,
            "delta_blend": 0.65,
            "delta_max_step": 0.25,
            "integration_max_drop": 0.08,
            "integration_strength_power": 1.0,
            "integration_active_threshold": 0.5,
            "integration_end_lift": 0.35,
        }
    if model_name == "ad_peak_weighted":
        return run_ad_peak_weighted, {
            **common,
            "iterations": 700,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": RANDOM_SEED,
            "weight_slope_alpha": 1.4,
            "weight_curvature_alpha": 0.9,
            "weight_ad_alpha": 1.2,
            "weight_max": 6.0,
            "integration_max_drop": 0.10,
            "integration_strength_power": 1.0,
            "integration_active_threshold": 0.5,
        }
    if model_name == "peak_weighted":
        return run_peak_weighted, {
            **common,
            "iterations": 700,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": RANDOM_SEED,
            "weight_slope_alpha": 1.8,
            "weight_curvature_alpha": 1.2,
            "weight_max": 7.0,
        }
    if model_name == "ranker":
        return run_ranker, {**common, "iterations": 500, "learning_rate": 0.05, "depth": 6, "random_seed": RANDOM_SEED}
    if model_name == "hybrid":
        return run_hybrid, {
            **common,
            "ranker_iterations": 600,
            "ranker_learning_rate": 0.05,
            "ranker_depth": 6,
            "level_iterations": 600,
            "level_learning_rate": 0.05,
            "level_depth": 6,
            "random_seed": RANDOM_SEED,
            "ensemble_seeds": "42,52,62",
            "shape_ranker_weight": 0.7,
            "shape_anchor_points": 1,
            "level_anchor_points": 2,
            "tail_anchor_points": 2,
            "disable_tail_floor": False,
            "enable_affine_calibration": True,
        }
    if model_name == "stacked":
        return run_stacked, {
            **common,
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": RANDOM_SEED,
            "oof_folds": 5,
            "meta_l2": 0.02,
            "shape_anchor_points": 1,
            "level_anchor_points": 2,
            "tail_anchor_points": 2,
        }
    if model_name == "residual_huber":
        return run_residual_huber, {
            **common,
            "iterations": 700,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": RANDOM_SEED,
            "delta_blend": 0.65,
            "delta_max_step": 0.15,
            "huber_delta": 0.03,
            "huber_delta_delta": 0.015,
        }
    if model_name == "blended_quantile":
        return run_blended_quantile, {**common, "iterations": 600, "learning_rate": 0.05, "depth": 6, "random_seed": RANDOM_SEED, "baseline_weight": 0.35}
    if model_name == "meta_ensemble":
        return run_meta_ensemble, {
            **common,
            "random_seed": RANDOM_SEED,
            "oof_folds": 5,
            "meta_l2": 0.02,
            "rh_iterations": 500,
            "rh_learning_rate": 0.05,
            "rh_depth": 6,
            "rh_huber_delta": 0.03,
            "rh_delta_blend": 0.65,
            "rh_delta_max_step": 0.15,
            "rh_huber_delta_delta": 0.015,
            "knn_neighbors_k": 12,
            "knn_distance_temperature": 1.0,
            "knn_residual_strength": 0.70,
            "knn_spike_gain": 1.10,
            "knn_smooth_window": 7,
        }
    raise RuntimeError(f"Неизвестная MODEL_NAME='{model_name}'")


def _summary_stats(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in metric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append({"metric": col, "mean": float(vals.mean()), "median": float(vals.median()), "std": float(vals.std(ddof=0)), "min": float(vals.min()), "max": float(vals.max())})
    return pd.DataFrame(rows)


def _metric_cols() -> list[str]:
    return ["spearman", "pearson", "rmse", "mae", "spike_rmse", "curvature_rmse"]


def _per_video_cols() -> list[str]:
    return ["model", "video_id", "status", "error", *_metric_cols()]


def _normalize_per_video_df(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = df.copy()
    for col in _per_video_cols():
        if col not in out.columns:
            out[col] = np.nan if col in _metric_cols() else ""
    out["model"] = model_name
    out["video_id"] = out["video_id"].astype(str)
    out["status"] = out["status"].astype(str)
    out["error"] = out["error"].fillna("").astype(str)
    for col in _metric_cols():
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[_per_video_cols()]


def _video_order_map(video_ids: list[str]) -> dict[str, int]:
    return {video_id: idx for idx, video_id in enumerate(video_ids)}


def _sort_per_video_df(df: pd.DataFrame, video_ids: list[str]) -> pd.DataFrame:
    order_map = _video_order_map(video_ids)
    out = df.copy()
    out["_ord"] = out["video_id"].map(order_map).fillna(len(order_map)).astype(int)
    out = out.sort_values(["_ord", "video_id"], ascending=[True, True]).drop(columns=["_ord"])
    return out.reset_index(drop=True)


def _curve_from_row(row: dict[str, Any], curve_points: int) -> np.ndarray:
    return np.clip(np.array([_safe_float(row.get(_point_col(i), 0.0), 0.0) for i in range(curve_points)], dtype=float), 0.0, 1.0)


def _build_global_baseline_curve(rows: list[dict[str, Any]], curve_points: int) -> np.ndarray:
    curves = [_curve_from_row(row, curve_points) for row in rows]
    if not curves:
        raise RuntimeError("Невозможно построить baseline: пустой список rows")
    return np.clip(np.mean(np.vstack(curves), axis=0), 0.0, 1.0)


def _write_baseline_prediction(run_dir: Path, y_pred: np.ndarray, y_true: np.ndarray) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    abs_err = np.abs(y_pred - y_true)
    pd.DataFrame(
        {"point_idx": list(range(len(y_true))), "pred_retention": y_pred, "pred_retention_norm": y_pred, "pred_score_raw": y_pred, "true_retention": y_true, "abs_error": abs_err}
    ).to_csv(run_dir / "holdout_prediction_vs_true.csv", index=False)


def _run_loo_for_model(model_name: str, video_ids: list[str], output_root: Path, row_by_video: dict[str, dict[str, Any]], baseline_curve: np.ndarray) -> dict[str, Any]:
    model_out_dir = output_root / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    model_runner, base_args = _runner_and_base_args(model_name)
    n = len(video_ids)
    per_video_csv = model_out_dir / "loo_per_video_metrics.csv"
    print(f"=== LOO over {n} videos | model={model_name} ===")

    existing_df = pd.DataFrame(columns=_per_video_cols())
    done_ok_ids: set[str] = set()
    if RESUME_FROM_EXISTING and per_video_csv.exists():
        try:
            existing_raw = pd.read_csv(per_video_csv)
            existing_df = _normalize_per_video_df(existing_raw, model_name=model_name)
            existing_df = existing_df.drop_duplicates(subset=["video_id"], keep="last")
            done_ok_ids = set(existing_df.loc[existing_df["status"] == "ok", "video_id"].astype(str).tolist())
            print(f"[{model_name}] resume: found {len(existing_df)} rows, already_ok={len(done_ok_ids)}, pending={len(video_ids) - len(done_ok_ids)}")
        except Exception as exc:
            print(f"[{model_name}] resume skipped: failed to read existing csv: {exc}")
            existing_df = pd.DataFrame(columns=_per_video_cols())
            done_ok_ids = set()

    recovered_rows: list[dict[str, Any]] = []
    for video_id in video_ids:
        if video_id in done_ok_ids:
            continue
        pred_path = model_out_dir / video_id / "holdout_prediction_vs_true.csv"
        if not pred_path.exists():
            continue
        try:
            y_pred, y_true = _load_pred_true_from_csv(pred_path)
            m_curve = _curve_metrics(y_pred, y_true)
            recovered_rows.append({"model": model_name, "video_id": video_id, "status": "ok", "error": "", **m_curve})
        except Exception:
            continue
    if recovered_rows:
        recovered_df = _normalize_per_video_df(pd.DataFrame(recovered_rows), model_name=model_name)
        existing_df = pd.concat([existing_df, recovered_df], ignore_index=True)
        existing_df = existing_df.drop_duplicates(subset=["video_id"], keep="last")
        done_ok_ids = set(existing_df.loc[existing_df["status"] == "ok", "video_id"].astype(str).tolist())
        print(f"[{model_name}] resume: recovered from run dirs={len(recovered_rows)}, already_ok={len(done_ok_ids)}, pending={len(video_ids) - len(done_ok_ids)}")

    pending_ids = [video_id for video_id in video_ids if video_id not in done_ok_ids]
    already_done = len(done_ok_ids)
    pending_total = len(pending_ids)

    if not existing_df.empty:
        existing_df = _sort_per_video_df(existing_df, video_ids=video_ids)
        existing_df.to_csv(per_video_csv, index=False)

    def _run_one(video_id: str, pending_idx: int) -> dict[str, Any]:
        global_idx = already_done + pending_idx
        print(f"[{model_name}] [{global_idx}/{n}] eval={video_id} (pending {pending_idx}/{pending_total})")
        run_dir = model_out_dir / video_id
        try:
            if model_name == "baseline_cached":
                row_obj = row_by_video.get(video_id)
                if row_obj is None:
                    raise RuntimeError(f"Видео не найдено в row_by_video: {video_id}")
                y_true = _curve_from_row(row_obj, CURVE_POINTS)
                y_pred = baseline_curve.copy()
                _write_baseline_prediction(run_dir, y_pred=y_pred, y_true=y_true)
                m_curve = _curve_metrics(y_pred, y_true)
                metrics = {}
            else:
                args = _ns(**base_args, limit_videos=n, train_videos=n - 1, eval_video_folder=video_id, eval_drive_file_id="", output_dir=str(run_dir))
                metrics = model_runner(args)
                pred_path = run_dir / "holdout_prediction_vs_true.csv"
                if pred_path.exists():
                    y_pred, y_true = _load_pred_true_from_csv(pred_path)
                    m_curve = _curve_metrics(y_pred, y_true)
                else:
                    m_curve = {}

            out = {"model": model_name, "video_id": video_id, "status": "ok", "error": ""}
            for key in ["spearman", "pearson", "rmse", "mae", "spike_rmse", "curvature_rmse"]:
                if key in m_curve:
                    out[key] = m_curve[key]
                elif key in metrics:
                    out[key] = metrics[key]
                else:
                    out[key] = np.nan
            return out
        except Exception as exc:
            print(f"[{model_name}] [{global_idx}/{n}] failed on {video_id}: {exc}")
            return {
                "model": model_name,
                "video_id": video_id,
                "status": "error",
                "error": str(exc),
                "spearman": np.nan,
                "pearson": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "spike_rmse": np.nan,
                "curvature_rmse": np.nan,
            }

    per_video_df = existing_df.copy()
    if pending_ids:
        workers = max(1, min(int(MAX_WORKERS), len(pending_ids)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one, video_id, idx): video_id for idx, video_id in enumerate(pending_ids, start=1)}
            for fut in as_completed(futures):
                row = fut.result()
                row_df = _normalize_per_video_df(pd.DataFrame([row]), model_name=model_name)
                per_video_df = per_video_df[per_video_df["video_id"] != str(row["video_id"])]
                per_video_df = pd.concat([per_video_df, row_df], ignore_index=True)
                per_video_df = _sort_per_video_df(per_video_df, video_ids=video_ids)
                per_video_df.to_csv(per_video_csv, index=False)
    else:
        print(f"[{model_name}] nothing to run, all videos already finished with status=ok")

    per_video_df = _sort_per_video_df(per_video_df, video_ids=video_ids)
    per_video_df.to_csv(per_video_csv, index=False)

    ok_df = per_video_df[per_video_df["status"] == "ok"].copy()
    summary_df = _summary_stats(ok_df, metric_cols=_metric_cols())
    summary_csv = model_out_dir / "loo_summary_stats.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_json = model_out_dir / "loo_run_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "model": model_name,
                "videos_total": len(video_ids),
                "runs_ok": int((per_video_df["status"] == "ok").sum()),
                "runs_error": int((per_video_df["status"] == "error").sum()),
                "per_video_csv": str(per_video_csv),
                "summary_csv": str(summary_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[{model_name}] complete")
    print(f"  per-video: {per_video_csv}")
    print(f"  summary:   {summary_csv}")
    return {
        "model": model_name,
        "per_video_df": per_video_df,
        "summary_df": summary_df,
        "runs_ok": int((per_video_df["status"] == "ok").sum()),
        "runs_error": int((per_video_df["status"] == "error").sum()),
    }


def _build_model_leaderboard(per_video_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["spearman", "pearson", "rmse", "mae", "spike_rmse", "curvature_rmse"]
    ok_df = per_video_df[per_video_df["status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame(columns=["model"])

    grouped = ok_df.groupby("model", dropna=False)
    rows: list[dict[str, Any]] = []
    for model_name, g in grouped:
        row: dict[str, Any] = {"model": model_name, "n_videos_ok": len(g)}
        for m in metric_cols:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if not vals.empty else np.nan
            row[f"{m}_median"] = float(vals.median()) if not vals.empty else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=0)) if not vals.empty else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if "rmse_mean" in out.columns and "spike_rmse_mean" in out.columns:
        out = out.sort_values(["rmse_mean", "spike_rmse_mean"], ascending=[True, True]).reset_index(drop=True)
    return out


def main() -> None:
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = build_rows_with_targets_source(root_folder_id=ROOT_FOLDER_ID, env_file=Path(ENV_FILE), curve_points=CURVE_POINTS, snapshot_dir=Path(SNAPSHOT_DIR))
    if not rows:
        raise RuntimeError("Не найдено данных в snapshot")

    video_ids = sorted({str(r.get("video_folder", "")).strip() for r in rows if str(r.get("video_folder", "")).strip()})
    if len(video_ids) < 2:
        raise RuntimeError(f"Недостаточно видео для LOO: {len(video_ids)}")
    row_by_video = {str(r.get("video_folder", "")).strip(): r for r in rows if str(r.get("video_folder", "")).strip()}
    baseline_curve = _build_global_baseline_curve(rows=rows, curve_points=CURVE_POINTS)
    print(f"Global cached baseline built once: points={len(baseline_curve)}")

    model_results: list[dict[str, Any]] = []
    all_per_video: list[pd.DataFrame] = []
    all_summary_rows: list[pd.DataFrame] = []

    for model_name in MODEL_NAMES:
        result = _run_loo_for_model(model_name=model_name, video_ids=video_ids, output_root=output_root, row_by_video=row_by_video, baseline_curve=baseline_curve)
        model_results.append({"model": model_name, "runs_ok": int(result["runs_ok"]), "runs_error": int(result["runs_error"])})
        per_video_df: pd.DataFrame = result["per_video_df"]
        summary_df: pd.DataFrame = result["summary_df"]
        all_per_video.append(per_video_df)
        if not summary_df.empty:
            all_summary_rows.append(summary_df.assign(model=model_name))

    all_per_video_df = pd.concat(all_per_video, ignore_index=True) if all_per_video else pd.DataFrame()
    all_per_video_csv = output_root / "loo_all_models_per_video_metrics.csv"
    all_per_video_df.to_csv(all_per_video_csv, index=False)

    all_summary_df = pd.concat(all_summary_rows, ignore_index=True) if all_summary_rows else pd.DataFrame()
    all_summary_csv = output_root / "loo_all_models_summary_long.csv"
    all_summary_df.to_csv(all_summary_csv, index=False)

    leaderboard_df = _build_model_leaderboard(all_per_video_df)
    leaderboard_csv = output_root / "loo_model_leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_csv, index=False)

    run_summary_json = output_root / "loo_all_models_run_summary.json"
    run_summary_json.write_text(
        json.dumps(
            {
                "models": MODEL_NAMES,
                "videos_total": len(video_ids),
                "results": model_results,
                "all_per_video_csv": str(all_per_video_csv),
                "all_summary_long_csv": str(all_summary_csv),
                "leaderboard_csv": str(leaderboard_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== LOO all models complete ===")
    print(f"All per-video:     {all_per_video_csv}")
    print(f"All summary (long):{all_summary_csv}")
    print(f"Leaderboard:       {leaderboard_csv}")
    print(f"Run summary:       {run_summary_json}")


if __name__ == "__main__":
    main()
