"""
Unified data layer for LOO retention experiments.

Merges two data sources per video:
  data/<video_id>/features_llm.json   — LLM-extracted video-level features (~64 cols)
  data/<video_id>/retention.json      — YouTube retention curve
  data/<video_id>/meta.json           — video metadata
  output/<video_id>_features.csv      — per-second extracted features (~80 cols) → aggregated to video-level

Re-exports helpers from train_utils so that all LOO scripts can do:
    from train.common.retention_data_layer import build_rows_with_targets_source, _point_col, ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train.common.train_utils import point_col as _point_col
from train.common.train_utils import resample_series as _resample_series
from train.common.train_utils import safe_float as _safe_float
from train.tabular_catboost.train_catboost_regressor import fallback_flatten


DEFAULT_PARENT_FOLDER_ID = "1aIqGRHTsO9kNBrOXRRz9XV8kD0Ru8zSV"


def select_train_test(rows: list[dict[str, Any]], args) -> tuple:
    """Split rows into (all_df, train_df, test_df) based on args.limit_videos / train_videos."""
    limit = getattr(args, "limit_videos", len(rows))
    train_n = getattr(args, "train_videos", max(limit - 1, 1))
    eval_video_folder = str(getattr(args, "eval_video_folder", "") or "").strip()
    eval_drive_file_id = str(getattr(args, "eval_drive_file_id", "") or "").strip()

    if len(rows) < limit:
        raise RuntimeError(f"Недостаточно видео с target retention: найдено {len(rows)}, нужно {limit}")
    if train_n >= limit:
        raise RuntimeError("--train-videos должен быть меньше --limit-videos")

    if eval_video_folder or eval_drive_file_id:
        matched = [
            r
            for r in rows
            if (not eval_video_folder or str(r.get("video_folder", "")) == eval_video_folder) and (not eval_drive_file_id or str(r.get("drive_file_id", "")) == eval_drive_file_id)
        ]
        if not matched:
            raise RuntimeError(f"Eval-видео не найдено: folder='{eval_video_folder}', id='{eval_drive_file_id}'")
        eval_row = matched[0]
        remaining = [r for r in rows if str(r.get("drive_file_id", "")) != str(eval_row.get("drive_file_id", ""))]
        train_df = pd.DataFrame(remaining[:train_n])
        test_df = pd.DataFrame([eval_row])
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        return all_df, train_df, test_df

    rows = rows[:limit]
    all_df = pd.DataFrame(rows)
    train_df = all_df.iloc[:train_n].copy()
    test_df = all_df.iloc[train_n : train_n + 1].copy()
    if test_df.empty:
        raise RuntimeError("Не удалось выделить тестовое видео")
    return all_df, train_df, test_df


_OUTPUT_SKIP_COLS = {"time", "frame", "retention"}
_AGG_FUNCS = ["mean", "std", "min", "max", "median"]


def _aggregate_output_csv(csv_path: Path) -> dict[str, float]:
    """Read per-second CSV → aggregate each numeric column to video-level stats."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    numeric_cols = [c for c in df.columns if c not in _OUTPUT_SKIP_COLS and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return {}
    agg = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        vals = series.values
        agg[f"out__{col}__mean"] = float(np.mean(vals))
        agg[f"out__{col}__std"] = float(np.std(vals))
        agg[f"out__{col}__min"] = float(np.min(vals))
        agg[f"out__{col}__max"] = float(np.max(vals))
        agg[f"out__{col}__median"] = float(np.median(vals))
        q25, q75 = np.percentile(vals, [25, 75])
        agg[f"out__{col}__q25"] = float(q25)
        agg[f"out__{col}__q75"] = float(q75)
        if len(vals) >= 2:
            agg[f"out__{col}__trend"] = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])
    return agg


def _parse_retention_list(raw: list, curve_points: int) -> dict | None:
    if not raw or not isinstance(raw, list):
        return None
    values = []
    for entry in raw:
        if isinstance(entry, dict):
            v = entry.get("audience_watch_ratio")
            if v is not None:
                values.append(_safe_float(v, 0.0))
        elif isinstance(entry, (int, float)):
            values.append(_safe_float(entry, 0.0))
    if not values:
        return None
    mean_val = float(np.mean(values))
    mid_idx = len(values) // 2
    tail_start = int(len(values) * 0.8)
    return {
        "status": "ok",
        "curve_raw": values,
        "mean_retention": mean_val,
        "mid_retention": float(values[mid_idx]) if mid_idx < len(values) else mean_val,
        "tail_retention": float(np.mean(values[tail_start:])) if tail_start < len(values) else mean_val,
    }


def _find_output_dir(snapshot_dir: Path) -> Path | None:
    """Locate output/ directory relative to snapshot_dir (data/)."""
    candidates = [snapshot_dir.parent / "output", Path("output"), Path("./output")]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def build_rows_with_targets_source(
    root_folder_id: str = DEFAULT_PARENT_FOLDER_ID, env_file: Path | str = ".env", curve_points: int = 20, snapshot_dir: Path | str | None = None
) -> list[dict[str, Any]]:
    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
    else:
        snapshot_dir = Path("data")
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

    output_dir = _find_output_dir(snapshot_dir)
    output_cache: dict[str, dict[str, float]] = {}

    rows: list[dict[str, Any]] = []

    for item_dir in sorted(snapshot_dir.iterdir()):
        if not item_dir.is_dir():
            continue

        features_path = item_dir / "features_llm.json"
        if not features_path.exists():
            continue

        retention_path = item_dir / "retention.json"
        retention_parsed_path = item_dir / "retention_parsed.json"

        retention = None
        if retention_parsed_path.exists():
            try:
                rdata = json.loads(retention_parsed_path.read_text(encoding="utf-8"))
                if isinstance(rdata, dict) and rdata.get("status") == "ok":
                    retention = rdata
            except Exception:
                pass

        if retention is None and retention_path.exists():
            try:
                rdata = json.loads(retention_path.read_text(encoding="utf-8"))
                if isinstance(rdata, list):
                    retention = _parse_retention_list(rdata, curve_points)
                elif isinstance(rdata, dict) and rdata.get("status") == "ok":
                    retention = rdata
            except Exception:
                pass

        if retention is None:
            continue

        try:
            features_payload = json.loads(features_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(features_payload, dict):
            continue

        flat = features_payload.get("video_features_flat", {})
        if not isinstance(flat, dict) or not flat:
            flat = fallback_flatten(features_payload)

        video_id = item_dir.name
        flat["video_folder"] = video_id
        flat["drive_file_id"] = video_id

        meta_path = item_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    flat["transcript_path"] = str(meta.get("title", ""))
                    if meta.get("id"):
                        flat["drive_file_id"] = str(meta["id"])
            except Exception:
                pass
        if "transcript_path" not in flat:
            flat["transcript_path"] = ""

        if output_dir is not None:
            csv_path = output_dir / f"{video_id}_features.csv"
            if csv_path.exists():
                if video_id not in output_cache:
                    output_cache[video_id] = _aggregate_output_csv(csv_path)
                flat.update(output_cache[video_id])

        curve_raw = retention.get("curve_raw", [])
        if not isinstance(curve_raw, list) or len(curve_raw) == 0:
            curve_raw = retention.get("curve_20", [])
        if not isinstance(curve_raw, list) or len(curve_raw) == 0:
            continue

        curve = _resample_series([_safe_float(x, 0.0) for x in curve_raw], curve_points)
        if len(curve) != curve_points:
            continue

        flat["target__retention_mean"] = _safe_float(retention.get("mean_retention", 0.0))
        flat["target__retention_mid"] = _safe_float(retention.get("mid_retention", 0.0))
        flat["target__retention_tail"] = _safe_float(retention.get("tail_retention", 0.0))
        for i, value in enumerate(curve):
            flat[_point_col(i)] = _safe_float(value, 0.0)

        rows.append(flat)

    uniq: dict[str, dict[str, Any]] = {}
    for row in rows:
        uniq[str(row.get("drive_file_id", ""))] = row
    rows = list(uniq.values())
    rows.sort(key=lambda x: str(x.get("video_folder", "")).lower())
    return rows
