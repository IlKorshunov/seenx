from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Собирает единый датасет по видео и обучает CatBoostRegressor.")
    parser.add_argument("--features-root", default="drive_feature_labels", help="Корень с разметками (поиск **/features_llm.json).")
    parser.add_argument("--dataset-csv", default="video_features_dataset.csv", help="Путь для сохранения объединённой таблицы по видео.")
    parser.add_argument("--labels-csv", default="", help="Опционально: CSV с таргетом для обучения.")
    parser.add_argument("--feature-key", default="drive_file_id", help="Ключ в фичах для merge с labels-csv.")
    parser.add_argument("--label-key", default="drive_file_id", help="Ключ в labels-csv для merge.")
    parser.add_argument("--target-column", default="", help="Название целевой переменной для регрессии.")
    parser.add_argument("--target-prefix", default="", help=("Префикс для multi-target обучения (по одной модели на каждый столбец), например target__retention_curve_20__."))
    parser.add_argument("--model-output", default="catboost_regressor.cbm", help="Путь сохранения обученной модели.")
    parser.add_argument("--eval-fraction", type=float, default=0.2, help="Доля данных под holdout-валидацию (0..0.5).")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed для перемешивания и CatBoost.")
    parser.add_argument("--iterations", type=int, default=600, help="Количество деревьев CatBoost.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate CatBoost.")
    parser.add_argument("--depth", type=int, default=6, help="Глубина деревьев CatBoost.")
    parser.add_argument("--experiment-name", default="CatBoostRegressor", help="Имя модели в metrics.json (для сводки run_all_experiments).")
    parser.add_argument("--report-dir", default="", help="Куда писать графики/CSV (по умолчанию — каталог model-output).")
    parser.add_argument("--video-ids-file", default="", help="Опционально: одна строка = один id видео; оставить только эти строки (по --feature-key).")
    return parser.parse_args()


def discover_feature_files(features_root: Path) -> list[Path]:
    return sorted(features_root.rglob("features_llm.json"))

def safe_float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def mean_series(rows: Any) -> float:
    if not isinstance(rows, list) or not rows:
        return 0.0
    values: list[float] = []
    for row in rows:
        if isinstance(row, dict):
            values.append(safe_float(row.get("value", 0.0), 0.0))
    return float(sum(values) / len(values)) if values else 0.0


def fallback_flatten(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source", {}) if isinstance(payload.get("source"), dict) else {}
    interval_labels = payload.get("interval_labels", {}) if isinstance(payload.get("interval_labels"), dict) else {}
    text_features = payload.get("text_features", {}) if isinstance(payload.get("text_features"), dict) else {}

    row: dict[str, Any] = {
        "video_folder": source.get("video_folder", ""),
        "transcript_path": source.get("transcript_path", ""),
        "drive_file_id": source.get("drive_file_id", ""),
        "duration_seconds": safe_float(source.get("duration_seconds", 0.0)),
        "segments_count": int(safe_float(source.get("segments_count", 0), 0.0)),
    }

    for key, rows in interval_labels.items():
        row[f"interval_label_mean__{key}"] = mean_series(rows)

    linguistic = text_features.get("linguistic_metrics", {})
    if isinstance(linguistic, dict):
        means = linguistic.get("global_mean_from_intervals", {})
        if isinstance(means, dict):
            for k, v in means.items():
                row[f"linguistic__{k}"] = safe_float(v)

    emotion = text_features.get("emotion_timeseries", {})
    if isinstance(emotion, dict):
        means = emotion.get("global_mean_from_intervals", {})
        if isinstance(means, dict):
            for k, v in means.items():
                row[f"emotion__{k}"] = safe_float(v)

    integration = text_features.get("integration_feature", {})
    if isinstance(integration, dict):
        global_from_intervals = integration.get("global_from_intervals", {})
        if isinstance(global_from_intervals, dict):
            row["integration_present"] = int(safe_float(global_from_intervals.get("integration_present", 0), 0))
            row["integration_confidence"] = safe_float(global_from_intervals.get("confidence", 0.0), 0.0)
            row["integration_evidence_count_max"] = int(safe_float(global_from_intervals.get("evidence_count_max", 0), 0))
            row["integration_suggested_next_step_present"] = int(safe_float(global_from_intervals.get("suggested_next_step_present", 0), 0))

    expectation = text_features.get("expectation_alignment", {})
    if isinstance(expectation, dict):
        global_from_video = expectation.get("global_from_video", {})
        if isinstance(global_from_video, dict):
            row["expectation_alignment_score"] = safe_float(global_from_video.get("alignment_score", 0.0), 0.0)
            row["expectation_mismatch_points_count"] = int(safe_float(global_from_video.get("mismatch_points_count", 0), 0))

    targets = payload.get("targets", {})
    if isinstance(targets, dict):
        retention = targets.get("retention", {})
        if isinstance(retention, dict) and retention.get("status") == "ok":
            row["target__retention_mean"] = safe_float(retention.get("mean_retention", 0.0), 0.0)
            row["target__retention_mid"] = safe_float(retention.get("mid_retention", 0.0), 0.0)
            row["target__retention_tail"] = safe_float(retention.get("tail_retention", 0.0), 0.0)
            curve_20 = retention.get("curve_20", [])
            if isinstance(curve_20, list):
                for i, value in enumerate(curve_20):
                    row[f"target__retention_curve_20__{i:02d}"] = safe_float(value, 0.0)

    return row


def build_dataset(features_root: Path):
    rows: list[dict[str, Any]] = []

    files = discover_feature_files(features_root)
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        flat = payload.get("video_features_flat", {})
        if isinstance(flat, dict) and flat:
            rows.append(flat)
        else:
            rows.append(fallback_flatten(payload))

    df = pd.DataFrame(rows)
    return df, len(files)


def _save_learning_curve(model: CatBoostRegressor, out_path: Path) -> None:
    if plt is None:
        return
    try:
        ev = model.get_evals_result()
    except Exception:
        return
    learn = ev.get("learn", {})
    val = ev.get("validation", {})
    if not learn and not val:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, series in learn.items():
        if series:
            ax.plot(series, label=f"train {name}")
    for name, series in val.items():
        if series:
            ax.plot(series, label=f"val {name}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("metric")
    ax.legend()
    ax.set_title("CatBoost learning curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str = "Holdout: pred vs true") -> None:
    if plt is None or len(y_true) < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, s=12)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("true")
    ax.set_ylabel("pred")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_feature_importance_plot(features: list[str], importance: np.ndarray, out_path: Path, top_k: int = 30) -> None:
    if plt is None or not features:
        return
    idx = np.argsort(importance)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.18)))
    y_pos = np.arange(len(idx))
    ax.barh(y_pos, importance[idx])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([features[i] for i in idx], fontsize=7)
    ax.invert_yaxis()
    ax.set_title(f"CatBoost feature importance (top {top_k})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_catboost_report(
    report_dir: Path,
    experiment_name: str,
    target_column: str,
    model: CatBoostRegressor,
    metrics: dict[str, Any],
    feature_cols: list[str],
    y_train: pd.Series,
    train_pred: np.ndarray,
    y_eval: pd.Series | None,
    eval_pred: np.ndarray | None,
) -> None:
    """LSTM-like artifacts: curves, holdout CSV, FI csv/png, unified metrics.json."""
    report_dir.mkdir(parents=True, exist_ok=True)
    imp = model.get_feature_importance()
    fi_df = pd.DataFrame({"feature": feature_cols, "importance_gain": imp.astype(float)}).sort_values("importance_gain", ascending=False)
    fi_df.to_csv(report_dir / "feature_importance.csv", index=False)
    _save_feature_importance_plot(feature_cols, imp, report_dir / "feature_importance.png")

    _save_learning_curve(model, report_dir / "training_curve.png")

    if y_eval is not None and eval_pred is not None and len(y_eval) > 0:
        h = pd.DataFrame({"true": y_eval.to_numpy(), "pred": eval_pred, "error": eval_pred - y_eval.to_numpy()})
        h.to_csv(report_dir / "holdout_prediction_vs_true.csv", index=False)
        _save_pred_vs_true(y_eval.to_numpy(), eval_pred, report_dir / "prediction_vs_true_scatter.png", title=f"Holdout {target_column}")

    # Summary metrics.json (leaderboard + optional per-row holdout)
    out_metrics = {
        "model": experiment_name,
        "target": target_column,
        "train_rmse": metrics.get("train_rmse"),
        "eval_rmse": metrics.get("eval_rmse"),
        "rows_train": metrics.get("rows_train"),
        "rows_eval": metrics.get("rows_eval"),
        "n_features": metrics.get("n_features"),
        "model_path": metrics.get("model_path"),
    }
    if y_eval is not None and eval_pred is not None and len(y_eval) > 0:
        out_metrics["per_row_holdout"] = [{"idx": int(i), "true": float(t), "pred": float(p)} for i, (t, p) in enumerate(zip(y_eval.tolist(), eval_pred.tolist(), strict=True))]
    (report_dir / "metrics.json").write_text(json.dumps(out_metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def write_catboost_multi_summary(
    report_dir: Path, experiment_name: str, target_prefix: str, all_metrics: dict[str, dict[str, Any]], feature_cols: list[str], importance_sum: np.ndarray, n_models: int
) -> None:
    """Aggregated report for multi-target curve training."""
    report_dir.mkdir(parents=True, exist_ok=True)
    rmses = {k: v.get("eval_rmse") for k, v in all_metrics.items() if isinstance(v.get("eval_rmse"), (int, float))}
    mean_rmse = float(np.mean(list(rmses.values()))) if rmses else None

    if plt is not None and rmses:
        keys = sorted(rmses.keys())
        vals = [rmses[k] for k in keys]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(keys)), vals)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k.replace(target_prefix, "") for k in keys], rotation=90, fontsize=6)
        ax.set_ylabel("eval RMSE")
        ax.set_title("CatBoost per-bucket eval RMSE")
        fig.tight_layout()
        fig.savefig(report_dir / "eval_rmse_by_target.png", dpi=120)
        plt.close(fig)

    imp_mean = importance_sum / max(n_models, 1)
    fi_df = pd.DataFrame({"feature": feature_cols, "importance_gain_mean": imp_mean}).sort_values("importance_gain_mean", ascending=False)
    fi_df.to_csv(report_dir / "feature_importance_mean.csv", index=False)
    _save_feature_importance_plot(feature_cols, imp_mean, report_dir / "feature_importance_mean.png", top_k=35)

    summary = {
        "model": experiment_name,
        "mode": "multi_target",
        "target_prefix": target_prefix,
        "mean_eval_rmse": mean_rmse,
        "n_targets": len(all_metrics),
        "per_target_eval_rmse": rmses,
        "note": "Per-target .cbm and __metrics.json live alongside this report.",
    }
    (report_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _catboost_preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Match train_catboost object/numeric column handling."""
    X = X.copy()
    for col in X.columns:
        if str(X[col].dtype) == "object":
            converted = pd.to_numeric(X[col], errors="coerce")
            non_null_ratio = float(converted.notna().mean()) if len(converted) else 0.0
            if non_null_ratio > 0.98:
                X[col] = converted.fillna(0.0)
            else:
                X[col] = X[col].astype(str).fillna("")
    return X


def _video_row_id(row: pd.Series) -> str:
    d = row.get("drive_file_id")
    if pd.notna(d) and str(d).strip():
        return str(d)
    vf = row.get("video_folder")
    if pd.notna(vf) and str(vf).strip():
        return str(vf)
    return "unknown"


def _safe_video_dir_name(vid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", vid) or "unknown"


def save_multitarget_retention_curve_video_plots(
    merged: pd.DataFrame, target_cols: list[str], model_output: Path, eval_fraction: float, random_seed: int, report_dir: Path, n_train_samples: int = 10
) -> None:
    """
    After training one model per curve bucket, plot true vs predicted retention curves
    per video (holdout + random train sample). Uses the same shuffle/split as
    train_catboost when all target columns are non-null (typical for curve_20).
    """
    if plt is None or not target_cols:
        return

    work = merged[merged[target_cols].notna().all(axis=1)].copy()
    if work.empty:
        return

    feature_cols = [c for c in work.columns if c not in {"video_folder", "transcript_path", "drive_file_id"} and not str(c).startswith("target__")]
    if not feature_cols:
        return

    X = work[feature_cols].copy()
    X = _catboost_preprocess_features(X)
    video_ids = work.apply(_video_row_id, axis=1)

    shuffled_idx = X.sample(frac=1.0, random_state=random_seed).index
    X = X.loc[shuffled_idx].reset_index(drop=True)
    video_ids = video_ids.loc[shuffled_idx].reset_index(drop=True)
    y_mat = work.loc[shuffled_idx, target_cols].to_numpy(dtype=float)

    eval_fraction = max(0.0, min(0.5, float(eval_fraction)))
    n_total = len(X)
    n_eval = int(n_total * eval_fraction) if n_total >= 10 else 0
    n_eval = min(n_eval, max(0, n_total - 5))

    if n_eval > 0:
        X_train, X_eval = X.iloc[:-n_eval].copy(), X.iloc[-n_eval:].copy()
        y_train, y_eval = y_mat[:-n_eval], y_mat[-n_eval:]
        ids_train, ids_eval = video_ids.iloc[:-n_eval], video_ids.iloc[-n_eval:]
    else:
        X_train, y_train = X, y_mat
        ids_train = video_ids
        X_eval = y_eval = ids_eval = None

    models: list[CatBoostRegressor] = []
    for col in target_cols:
        safe_col = re.sub(r"[^A-Za-z0-9_.-]+", "_", col)
        path = model_output.with_name(f"{model_output.stem}__{safe_col}.cbm")
        if not path.is_file():
            return
        m = CatBoostRegressor()
        m.load_model(str(path))
        models.append(m)

    pred_eval = np.column_stack([m.predict(X_eval) for m in models]) if X_eval is not None and len(X_eval) > 0 else None
    pred_train = np.column_stack([m.predict(X_train) for m in models])

    videos_root = report_dir / "videos"
    n_points = len(target_cols)
    x_axis = np.arange(n_points)

    def _plot_one(vid: str, y_true: np.ndarray, y_pred: np.ndarray, split_label: str) -> None:
        sub = videos_root / _safe_video_dir_name(vid)
        sub.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_axis, y_true, color="C0")
        ax.plot(x_axis, y_pred, color="C1")
        ax.set_xlabel("Curve point")
        ax.set_ylabel("Retention")
        ax.set_title(f"{vid} [{split_label}]")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(sub / "prediction.png", dpi=120)
        plt.close(fig)

    if pred_eval is not None and ids_eval is not None:
        for i in range(len(ids_eval)):
            _plot_one(str(ids_eval.iloc[i]), y_eval[i], pred_eval[i], "val")

    rng = np.random.RandomState(random_seed)
    n_pick = min(n_train_samples, len(ids_train))
    if n_pick > 0:
        pick_idx = rng.choice(len(ids_train), size=n_pick, replace=False)
        for i in pick_idx:
            _plot_one(str(ids_train.iloc[i]), y_train[i], pred_train[i], "train")


def train_catboost(
    df, target_column: str, model_output: Path, eval_fraction: float, random_seed: int, iterations: int, learning_rate: float, depth: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    work = df.copy()
    if target_column not in work.columns:
        raise RuntimeError(f"Не найден target-column '{target_column}' в собранной таблице")

    work = work[work[target_column].notna()].copy()
    if work.empty:
        raise RuntimeError("После фильтрации по target не осталось строк")

    target = pd.to_numeric(work[target_column], errors="coerce")
    work = work[target.notna()].copy()
    target = pd.to_numeric(work[target_column], errors="coerce")
    if work.empty:
        raise RuntimeError("Target не содержит числовых значений")

    exclude_cols = {target_column, "video_folder", "transcript_path", "drive_file_id"}
    feature_cols = [c for c in work.columns if c not in exclude_cols and not str(c).startswith("target__")]
    if not feature_cols:
        raise RuntimeError("Нет признаков для обучения после исключений")

    X = work[feature_cols].copy()
    y = target.astype(float)

    for col in X.columns:
        if str(X[col].dtype) == "object":
            converted = pd.to_numeric(X[col], errors="coerce")
            non_null_ratio = float(converted.notna().mean()) if len(converted) else 0.0
            if non_null_ratio > 0.98:
                X[col] = converted.fillna(0.0)
            else:
                X[col] = X[col].astype(str).fillna("")

    shuffled_idx = X.sample(frac=1.0, random_state=random_seed).index
    X = X.loc[shuffled_idx].reset_index(drop=True)
    y = y.loc[shuffled_idx].reset_index(drop=True)

    eval_fraction = max(0.0, min(0.5, float(eval_fraction)))
    n_total = len(X)
    n_eval = int(n_total * eval_fraction) if n_total >= 10 else 0
    n_eval = min(n_eval, max(0, n_total - 5))

    if n_eval > 0:
        X_train, X_eval = X.iloc[:-n_eval].copy(), X.iloc[-n_eval:].copy()
        y_train, y_eval = y.iloc[:-n_eval].copy(), y.iloc[-n_eval:].copy()
    else:
        X_train, y_train = X, y
        X_eval = y_eval = None

    cat_feature_names = [c for c in X_train.columns if str(X_train[c].dtype) in {"object", "category"}]

    model = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", random_seed=random_seed, iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=100)

    fit_kwargs: dict[str, Any] = {"cat_features": cat_feature_names}
    if X_eval is not None and y_eval is not None and len(X_eval) > 0:
        fit_kwargs["eval_set"] = (X_eval, y_eval)

    model.fit(X_train, y_train, **fit_kwargs)
    model.save_model(str(model_output))

    train_pred = model.predict(X_train)
    train_rmse = float(np.sqrt(np.mean((train_pred - y_train.to_numpy()) ** 2)))

    metrics: dict[str, Any] = {
        "rows_train": len(X_train),
        "rows_eval": len(X_eval) if X_eval is not None else 0,
        "n_features": len(feature_cols),
        "train_rmse": train_rmse,
        "model_path": str(model_output),
    }
    eval_pred = None
    if X_eval is not None and y_eval is not None and len(X_eval) > 0:
        eval_pred = model.predict(X_eval)
        eval_rmse = float(np.sqrt(np.mean((eval_pred - y_eval.to_numpy()) ** 2)))
        metrics["eval_rmse"] = eval_rmse

    bundle = {"model": model, "feature_cols": feature_cols, "y_train": y_train, "train_pred": train_pred, "y_eval": y_eval, "eval_pred": eval_pred}
    return metrics, bundle


def main() -> None:
    args = parse_args()

    features_root = Path(args.features_root)
    dataset_csv = Path(args.dataset_csv)
    labels_csv = Path(args.labels_csv) if args.labels_csv else None
    model_output = Path(args.model_output)

    df, discovered = build_dataset(features_root)
    if df.empty:
        raise RuntimeError(f"Не удалось собрать датасет: нет валидных features_llm.json в {features_root}")

    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_csv, index=False)
    print(f"[dataset] files_discovered={discovered} rows={len(df)} cols={len(df.columns)}")
    print(f"[dataset] saved={dataset_csv}")

    merged = df
    if labels_csv:
        labels_df = pd.read_csv(labels_csv)
        if args.label_key not in labels_df.columns:
            raise RuntimeError(f"В labels-csv нет колонки '{args.label_key}'")
        if args.feature_key not in df.columns:
            raise RuntimeError(f"В собранных фичах нет колонки '{args.feature_key}'")
        merged = df.merge(labels_df, left_on=args.feature_key, right_on=args.label_key, how="inner")
        print(f"[merge] labels_rows={len(labels_df)} merged_rows={len(merged)} key={args.feature_key}~{args.label_key}")

    if getattr(args, "video_ids_file", ""):
        allow = {ln.strip() for ln in Path(args.video_ids_file).read_text(encoding="utf-8").splitlines() if ln.strip()}
        fk = args.feature_key
        if fk not in merged.columns:
            raise RuntimeError(f"В датасете нет колонки '{fk}' для фильтра --video-ids-file")
        before = len(merged)
        merged = merged[merged[fk].astype(str).isin(allow)]
        print(f"[filter] video_ids_file: rows {before} -> {len(merged)} ({args.video_ids_file})")

    report_dir = Path(args.report_dir) if args.report_dir else model_output.parent
    exp_name = args.experiment_name

    if args.target_column:
        metrics, bundle = train_catboost(
            df=merged,
            target_column=args.target_column,
            model_output=model_output,
            eval_fraction=args.eval_fraction,
            random_seed=args.random_seed,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
        )
        print("[train] done")
        for k, v in metrics.items():
            print(f"[train] {k}={v}")
        write_catboost_report(
            report_dir=report_dir,
            experiment_name=exp_name,
            target_column=args.target_column,
            model=bundle["model"],
            metrics=metrics,
            feature_cols=bundle["feature_cols"],
            y_train=bundle["y_train"],
            train_pred=bundle["train_pred"],
            y_eval=bundle["y_eval"],
            eval_pred=bundle["eval_pred"],
        )
        print(f"[train] report -> {report_dir}")
        return

    if args.target_prefix:
        target_cols = sorted([c for c in merged.columns if str(c).startswith(args.target_prefix)])
        if not target_cols:
            raise RuntimeError(f"Не найдено колонок по target-prefix '{args.target_prefix}'")
        all_metrics: dict[str, dict[str, Any]] = {}
        model_output.parent.mkdir(parents=True, exist_ok=True)
        imp_sum: np.ndarray | None = None
        fcols: list[str] = []
        example_curve_png = False
        for col in target_cols:
            safe_col = re.sub(r"[^A-Za-z0-9_.-]+", "_", col)
            per_target_model = model_output.with_name(f"{model_output.stem}__{safe_col}.cbm")
            metrics, bundle = train_catboost(
                df=merged,
                target_column=col,
                model_output=per_target_model,
                eval_fraction=args.eval_fraction,
                random_seed=args.random_seed,
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
            )
            all_metrics[col] = metrics
            print(f"[train] done target={col} model={per_target_model}")
            imp = bundle["model"].get_feature_importance()
            if imp_sum is None:
                imp_sum = np.zeros_like(imp, dtype=float)
                fcols = list(bundle["feature_cols"])
            imp_sum = imp_sum + imp.astype(float)
            if col.rstrip().endswith("09") and not example_curve_png:
                sub = report_dir / "example_target_training_curve"
                sub.mkdir(parents=True, exist_ok=True)
                _save_learning_curve(bundle["model"], sub / f"training_curve__{safe_col}.png")
                example_curve_png = True

        metrics_path = model_output.with_name(f"{model_output.stem}__metrics.json")
        metrics_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[train] all-target metrics saved={metrics_path}")

        if imp_sum is not None and fcols:
            write_catboost_multi_summary(
                report_dir=report_dir,
                experiment_name=exp_name,
                target_prefix=args.target_prefix,
                all_metrics=all_metrics,
                feature_cols=fcols,
                importance_sum=imp_sum,
                n_models=len(target_cols),
            )
        if "retention_curve" in args.target_prefix.lower():
            save_multitarget_retention_curve_video_plots(
                merged=merged, target_cols=target_cols, model_output=model_output, eval_fraction=args.eval_fraction, random_seed=args.random_seed, report_dir=report_dir
            )
        print(f"[train] summary report -> {report_dir}")
        return

    print("[train] target-column или target-prefix не задан, обучение пропущено")


if __name__ == "__main__":
    main()
