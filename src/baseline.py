import glob
import inspect
import os
import time
from typing import ClassVar, Literal

import catboost as cb
import matplotlib
import numpy as np
import optuna
import pandas as pd
from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter

from .retention_analysis import *
from .utils.logger import Logger


matplotlib.use("Agg")
import matplotlib.pyplot as plt


logger = Logger(show=True).get_logger()

FEATURE_EXCLUDE_COLS = {
    "retention", "frame", "time", "time_pct",
    "audience_watch_ratio", "avd", "avd_sec", "avd_pct", "retention_30", "mean_retention",
    "hook_score_x_time_pct", "hook_score_x_time_pct.1",
    "edit_pace_x_screencast.1", "is_ad_x_viewer_address.1",
}

DEFAULT_OUTPUT_DIR, DEFAULT_DATA_DIR = "output", "data"
DEFAULT_MODEL_PATH = "static/weights/model.cbm"
DEFAULT_AVD_MODEL_PATH = "static/weights/model_avd.cbm"
DEFAULT_TENSORBOARD_LOG_DIR = "train/tensorboard_catboost"
PREDICTION_OUTPUT_DIR, PREDICTION_REL_PATH = "my_metrics", ("prediction", "pred.png")
MODEL_EXTENSION, DELTA_MODEL_SUFFIX = ".cbm", "_delta.cbm"

DEFAULT_VAL_RATIO, DEFAULT_RANDOM_STATE, DEFAULT_ALPHA = 0.2, 42, 0.5
DEFAULT_ITERATIONS, DEFAULT_DEPTH, DEFAULT_LEARNING_RATE, DEFAULT_EARLY_STOPPING_ROUNDS = 1000, 6, 0.05, 50
CATBOOST_LOSS_FUNCTION, CATBOOST_EVAL_METRIC, CATBOOST_VERBOSE = "RMSE", "MAE", 100
BASELINE_STRATEGY = "mean_duration"

OPTUNA_TRIALS, ALPHA_ROUND_DIGITS = 20, 2
OPTUNA_RANGES = {
    "iterations": (200, 1500), "depth": (4, 10),
    "learning_rate": (0.01, 0.2), "early_stopping_rounds": (20, 100), "alpha": (0.0, 1.0),
}

LAG_WINDOWS = [5, 15, 30, 60]
LAG_SOURCE_COLS = ["rms", "speaker_prob", "face_screen_ratio", "motion_speed", "edit_pace", "visual_entropy", "wps", "brightness"]
AVD_AGG_STATS = ["mean", "std", "min", "max"]

COLORS = {
    "video": "#2196F3", "baseline": "#FF5722", "baseline_std": "#FF9800",
    "positive": "#4CAF50", "negative": "#F44336", "fill": "#9C27B0",
    "bar": "#2196F3", "axis": "black", "abs_head": "#4CAF50", "delta_head": "#9C27B0",
}

FIG_SIZE = {"comparison": (14, 8), "training": (14, 10), "prediction": (14, 9)}
HEIGHT_RATIOS = [3, 1]
DPI, GRID_ALPHA, FONT_SIZE_LEGEND, DEFAULT_TOP_N = 150, 0.3, 9, 20
LINE_WIDTH, LINE_WIDTH_THIN, LINE_WIDTH_AXIS = 2, 1.5, 0.5

LOGIT_EPS = 0.5
SplitStrategy = Literal["random", "loo"]
DEFAULT_SPLIT_STRATEGY: SplitStrategy = "loo"


def _save_fig(fig: Figure, path: str | None, *, close: bool = True):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved %s", path)
    if close: plt.close(fig)


def _to_logit(y_pct: np.ndarray, eps: float = LOGIT_EPS) -> np.ndarray:
    y = np.clip(y_pct, eps, 100.0 - eps)
    return np.log(y / (100.0 - y))


def _from_logit(z: np.ndarray) -> np.ndarray:
    return 100.0 / (1.0 + np.exp(-z))


def compare_video(baseline_result: dict, channel_data: list[dict], video_name: str) -> dict:
    baseline = baseline_result["baseline_retention"]
    video_entry = next((e for e in channel_data if e["name"] == video_name), None)
    if video_entry is None: raise KeyError(f"{video_name} not found in channel_data")

    video_curve = _resample_retention(video_entry["retention_series"], baseline_result["target_length"])
    delta = video_curve - baseline

    return {
        "video_name": video_name,
        "duration_sec": video_entry["duration_sec"],
        "video_curve": video_curve,
        "baseline_curve": baseline,
        "delta": delta,
        "mean_delta_pp": float(np.mean(delta)),
        "max_above_pp": float(np.max(delta)),
        "max_below_pp": float(np.min(delta)),
        "correlation": float(np.corrcoef(video_curve, baseline)[0, 1]),
    }


def compare_all(baseline_result: dict, channel_data: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "video_id": r["video_name"], "duration_sec": r["duration_sec"],
            "mean_delta_pp": round(r["mean_delta_pp"], 3),
            "max_above_pp": round(r["max_above_pp"], 3),
            "max_below_pp": round(r["max_below_pp"], 3),
            "correlation": round(r["correlation"], 3),
        }
        for r in (compare_video(baseline_result, channel_data, v["name"]) for v in channel_data)
    ])


def plot_comparison(baseline_result: dict, channel_data: list[dict], video_name: str, output_path: str | None = None) -> Figure:
    comparison = compare_video(baseline_result, channel_data, video_name)
    baseline, baseline_std = baseline_result["baseline_retention"], baseline_result["baseline_std"]
    time_points, delta = np.arange(len(baseline)), comparison["delta"]

    fig, (top, bot) = plt.subplots(2, 1, figsize=FIG_SIZE["comparison"], height_ratios=HEIGHT_RATIOS, sharex=True)

    top.fill_between(time_points, baseline - baseline_std, baseline + baseline_std, alpha=0.15, color=COLORS["baseline_std"], label="baseline +-1 sigma")
    top.plot(time_points, baseline, color=COLORS["baseline"], linewidth=LINE_WIDTH, label="baseline")
    top.plot(time_points, comparison["video_curve"], color=COLORS["video"], linewidth=LINE_WIDTH, label=video_name)
    top.set(ylabel="Retention (%)", title=f"{video_name} vs channel baseline")
    top.legend(fontsize=FONT_SIZE_LEGEND)
    top.grid(True, alpha=GRID_ALPHA)

    bot.fill_between(time_points, delta, alpha=0.3, color=COLORS["positive"], where=delta >= 0)
    bot.fill_between(time_points, delta, alpha=0.3, color=COLORS["negative"], where=delta < 0)
    bot.axhline(0, color=COLORS["axis"], linewidth=LINE_WIDTH_AXIS)
    bot.set(xlabel="Time (resampled)", ylabel="delta (pp)", title=f"Delta: mean={comparison['mean_delta_pp']:+.1f}pp, corr={comparison['correlation']:.3f}")
    bot.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    _save_fig(fig, output_path)
    return fig


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in (col for col in LAG_SOURCE_COLS if col in df.columns):
        values = df[c].astype(float)
        for w in LAG_WINDOWS:
            r = values.rolling(w, min_periods=1)
            new_cols[f"{c}_rmean{w}"] = r.mean()
            new_cols[f"{c}_rstd{w}"] = r.std().fillna(0)
        new_cols[f"{c}_diff1"] = values.diff().fillna(0)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


class BaseCatBoostTrainer:
    LOG_PREFIX = "CatBoost"
    LOO_METRIC_KEYS: ClassVar[list[str]] = []

    def __init__(self, video_frames: dict[str, pd.DataFrame], val_ratio: float = DEFAULT_VAL_RATIO, random_state: int = DEFAULT_RANDOM_STATE, split_strategy: SplitStrategy = DEFAULT_SPLIT_STRATEGY, val_video: str | None = None):
        self.video_frames = video_frames
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.split_strategy: SplitStrategy = split_strategy
        self.val_video = val_video
        self.feature_names: list[str] = []
        self.train_ids: list[str] = []
        self.val_ids: list[str] = []
        self._split()

    def _split(self):
        ids = sorted(self.video_frames)
        if self.split_strategy == "loo":
            held = self.val_video or ids[-1]
            if held not in ids: raise ValueError(f"val_video '{held}' not in video_frames")
            self.val_ids, self.train_ids = [held], [v for v in ids if v != held]
        else:
            np.random.default_rng(self.random_state).shuffle(ids)
            val_count = max(1, int(len(ids) * self.val_ratio))
            self.val_ids, self.train_ids = ids[:val_count], ids[val_count:]
        logger.info("%s [%s] Train: %s, Val: %s", self.LOG_PREFIX, self.split_strategy, self.train_ids, self.val_ids)

    def _reset_for_fold(self):
        self.feature_names = []

    @staticmethod
    def _get_nested(d: dict, dotted_key: str):
        for k in dotted_key.split("."): d = d[k]
        return d

    def train_loo_cv(self, **train_kwargs) -> dict:
        if not self.LOO_METRIC_KEYS: raise NotImplementedError(f"{type(self).__name__} must define LOO_METRIC_KEYS")

        sig = inspect.signature(self.train).parameters
        for k in ("save_path", "log_dir"):
            if k in sig and k not in train_kwargs: train_kwargs[k] = None

        prev_strategy, prev_val_video = self.split_strategy, self.val_video
        self.split_strategy = "loo"

        fold_results = []
        for held_out in sorted(self.video_frames):
            self.val_video = held_out
            self._split()
            self._reset_for_fold()
            logger.info("=== LOO fold: held out '%s' ===", held_out)
            fold_results.append({"held_out": held_out, **self.train(**train_kwargs)})

        self.split_strategy, self.val_video = prev_strategy, prev_val_video
        self._split()
        self._reset_for_fold()

        agg: dict = {"n_folds": len(fold_results), "fold_results": fold_results}
        for key in self.LOO_METRIC_KEYS:
            values = [self._get_nested(f, key) for f in fold_results]
            label = key.replace(".", "_")
            agg[f"mean_{label}"], agg[f"std_{label}"] = float(np.mean(values)), float(np.std(values))
        logger.info("LOO-CV agg: %s", {k: v for k, v in agg.items() if k != "fold_results"})
        return agg

    @classmethod
    def _load_video_frames_from_output_dir(cls, output_dir: str) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for path in sorted(glob.glob(os.path.join(output_dir, "*_features.csv"))):
            df = pd.read_csv(path, index_col=0)
            if "retention" not in df.columns: continue
            video_id = os.path.basename(path).replace("_features.csv", "")
            frames[video_id] = df.dropna(subset=["retention"])
            logger.info("Loaded %s: %d rows, %d features", video_id, len(frames[video_id]), len(frames[video_id].columns) - 1)
        logger.info("Total: %d videos with retention", len(frames))
        return frames

    @classmethod
    def from_output_dir(cls, output_dir: str = DEFAULT_OUTPUT_DIR, **kwargs):
        return cls(cls._load_video_frames_from_output_dir(output_dir), **kwargs)

    def _catboost_params(self, iterations: int, depth: int, learning_rate: float, early_stopping_rounds: int, *, verbose: bool | int) -> dict:
        return dict(
            iterations=iterations, depth=depth, learning_rate=learning_rate,
            loss_function=CATBOOST_LOSS_FUNCTION, eval_metric=CATBOOST_EVAL_METRIC,
            verbose=verbose, early_stopping_rounds=early_stopping_rounds, random_seed=self.random_state,
        )

    def _fit_regressor(self, X_train, y_train, X_val, y_val, params: dict, *, log_label: str | None = None) -> tuple[cb.CatBoostRegressor, float]:
        if log_label: logger.info("Training %s", log_label)
        model = cb.CatBoostRegressor(**params)
        t0 = time.time()
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        return model, time.time() - t0

    @staticmethod
    def _save_model(model: cb.CatBoostRegressor | None, save_path: str | None, *, label: str = "Model"):
        if model is None: raise RuntimeError(f"{label} is not trained yet")
        if not save_path: return
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        model.save_model(save_path)
        logger.info("%s saved to %s", label, save_path)

    @staticmethod
    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_pred - y_true)))

    def _feature_importance_dict(self, model: cb.CatBoostRegressor | None) -> dict[str, float]:
        if model is None: raise RuntimeError("Model not trained yet")
        return dict(zip(self.feature_names, model.get_feature_importance().tolist(), strict=True))


class RetentionTrainer(BaseCatBoostTrainer):
    LOG_PREFIX = "Retention"
    LOO_METRIC_KEYS: ClassVar[list[str]] = ["val.mae", "val.mse", "val.r2"]

    def __init__(self, video_frames: dict[str, pd.DataFrame], val_ratio: float = DEFAULT_VAL_RATIO, random_state: int = DEFAULT_RANDOM_STATE, data_dir: str = DEFAULT_DATA_DIR, split_strategy: SplitStrategy = DEFAULT_SPLIT_STRATEGY, val_video: str | None = None, use_logit: bool = True):
        super().__init__(video_frames=video_frames, val_ratio=val_ratio, random_state=random_state, split_strategy=split_strategy, val_video=val_video)
        self.data_dir = data_dir
        self.use_logit = use_logit
        self.model_abs: cb.CatBoostRegressor | None = None
        self.model_delta: cb.CatBoostRegressor | None = None
        self.alpha = DEFAULT_ALPHA
        self.baseline_result = None
        self.baselines_by_video: dict[str, np.ndarray] | None = None
        self.evals_result_abs: dict = {}
        self.evals_result_delta: dict = {}

    def _reset_for_fold(self):
        super()._reset_for_fold()
        self.model_abs = self.model_delta = None
        self.alpha = DEFAULT_ALPHA
        self.baseline_result = None
        self.baselines_by_video = None
        self.evals_result_abs, self.evals_result_delta = {}, {}

    def _compute_baselines(self):
        if self.baselines_by_video is not None: return

        all_data = load_channel_retentions_csv(self.data_dir)
        data_by_id = {d["name"]: d for d in all_data}
        train_data = [data_by_id[v] for v in self.train_ids if v in data_by_id]
        missing = [v for v in self.train_ids + self.val_ids if v not in data_by_id]

        if missing: logger.warning("Missing retention CSV for videos: %s", missing)
        if not train_data: raise ValueError("No train retention curves found for baseline")

        self.baseline_result = compute_channel_baseline(train_data, strategy=BASELINE_STRATEGY)
        baselines: dict[str, np.ndarray] = {}

        for vid in self.train_ids:
            excluded = [d for d in train_data if d["name"] != vid]
            if not excluded: continue
            loo = compute_channel_baseline(excluded, strategy=BASELINE_STRATEGY)
            if loo is not None: baselines[vid] = loo["baseline_retention"]

        self.baselines_by_video = baselines
        logger.info("Computed train-only LOO baselines for %d videos", len(baselines))

    def _get_video_baseline(self, video_id: str) -> np.ndarray:
        assert self.baseline_result is not None and self.baselines_by_video is not None
        baseline = self.baselines_by_video.get(video_id, self.baseline_result["baseline_retention"])
        return _resample_retention(baseline, len(self.video_frames[video_id]))

    def _baseline_target_space(self, video_id: str) -> np.ndarray:
        b = self._get_video_baseline(video_id)
        return _to_logit(b) if self.use_logit else b

    def _compute_feature_names(self):
        all_cols: set[str] = set()
        for vid in self.train_ids:
            all_cols.update(_add_lag_features(self.video_frames[vid]).columns)
        self.feature_names = sorted(c for c in all_cols if c not in FEATURE_EXCLUDE_COLS)

    def _build_dataset(self, video_ids: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if not self.feature_names: self._compute_feature_names()

        frames, y_abs_parts, y_delta_parts = [], [], []
        for vid in video_ids:
            f = _add_lag_features(self.video_frames[vid])
            r = f["retention"].to_numpy(dtype=float)
            b = self._get_video_baseline(vid)
            r_t = _to_logit(r) if self.use_logit else r
            b_t = _to_logit(b) if self.use_logit else b
            frames.append(f)
            y_abs_parts.append(r_t)
            y_delta_parts.append(r_t - b_t)

        X = pd.concat(frames, ignore_index=True).reindex(columns=self.feature_names, fill_value=0).astype(float).fillna(0)
        return X, np.concatenate(y_abs_parts), np.concatenate(y_delta_parts)

    def train(self, optuna_trials: int = OPTUNA_TRIALS, save_path: str | None = DEFAULT_MODEL_PATH, log_dir: str | None = DEFAULT_TENSORBOARD_LOG_DIR) -> dict:
        self._compute_baselines()
        X_train, y_train_abs, y_train_delta = self._build_dataset(self.train_ids)
        X_val, y_val_abs, y_val_delta = self._build_dataset(self.val_ids)
        val_baseline_target = np.concatenate([self._baseline_target_space(v) for v in self.val_ids])

        logger.info("X_train: %s, X_val: %s, features: %d, use_logit=%s", X_train.shape, X_val.shape, len(self.feature_names), self.use_logit)

        t0 = time.time()

        def objective(trial: optuna.Trial) -> float:
            iters = trial.suggest_int("iterations", *OPTUNA_RANGES["iterations"])
            depth = trial.suggest_int("depth", *OPTUNA_RANGES["depth"])
            lr = trial.suggest_float("learning_rate", *OPTUNA_RANGES["learning_rate"], log=True)
            es = trial.suggest_int("early_stopping_rounds", *OPTUNA_RANGES["early_stopping_rounds"])
            a = trial.suggest_float("alpha", *OPTUNA_RANGES["alpha"])

            params = self._catboost_params(iters, depth, lr, es, verbose=False)
            m_abs, _ = self._fit_regressor(X_train, y_train_abs, X_val, y_val_abs, params)
            m_delta, _ = self._fit_regressor(X_train, y_train_delta, X_val, y_val_delta, params)

            blended = a * m_abs.predict(X_val) + (1 - a) * (val_baseline_target + m_delta.predict(X_val))
            return float(np.mean(np.abs(blended - y_val_abs)))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)

        bp = study.best_params
        self.alpha = round(float(bp["alpha"]), ALPHA_ROUND_DIGITS)
        final_params = self._catboost_params(int(bp["iterations"]), int(bp["depth"]), float(bp["learning_rate"]), int(bp["early_stopping_rounds"]), verbose=CATBOOST_VERBOSE)

        logger.info(
            "Optuna best: iterations=%d depth=%d lr=%.4f early_stop=%d alpha=%.2f (val MAE=%.4f)",
            int(bp["iterations"]), int(bp["depth"]), float(bp["learning_rate"]),
            int(bp["early_stopping_rounds"]), self.alpha, float(study.best_value),
        )

        self.model_abs, t_abs = self._fit_regressor(X_train, y_train_abs, X_val, y_val_abs, final_params, log_label="HEAD 1 abs final")
        self.evals_result_abs = self.model_abs.evals_result_

        self.model_delta, t_delta = self._fit_regressor(X_train, y_train_delta, X_val, y_val_delta, final_params, log_label="HEAD 2 delta final")
        self.evals_result_delta = self.model_delta.evals_result_

        if save_path:
            self._save_model(self.model_abs, save_path, label="Absolute retention model")
            self._save_model(self.model_delta, save_path.replace(MODEL_EXTENSION, DELTA_MODEL_SUFFIX), label="Delta retention model")
        if log_dir: self._write_tensorboard(log_dir)

        return {
            "train": self._evaluate_blended(self.train_ids, "train"),
            "val": self._evaluate_blended(self.val_ids, "val"),
            "alpha": self.alpha,
            "optuna_best_val_mae": round(float(study.best_value), 6),
            "optuna_best_params": bp,
            "optuna_trials": optuna_trials,
            "n_trees_abs": self.model_abs.tree_count_,
            "n_trees_delta": self.model_delta.tree_count_,
            "elapsed_sec": round(time.time() - t0, 2),
            "elapsed_fit_abs_sec": round(t_abs, 2),
            "elapsed_fit_delta_sec": round(t_delta, 2),
            "feature_importance": self._feature_importance_dict(self.model_abs),
        }

    def _predict_components(self, video_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model_abs is None or self.model_delta is None: raise RuntimeError("Models not trained yet")
        f = _add_lag_features(self.video_frames[video_id])
        X = f.reindex(columns=self.feature_names, fill_value=0).astype(float).fillna(0)
        b_target = self._baseline_target_space(video_id)
        return f["retention"].to_numpy(dtype=float), self.model_abs.predict(X), b_target + self.model_delta.predict(X)

    def _predict_blended(self, video_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        true, abs_pred, delta_pred = self._predict_components(video_id)
        blended = self.alpha * abs_pred + (1 - self.alpha) * delta_pred
        if self.use_logit: return true, _from_logit(abs_pred), _from_logit(delta_pred), _from_logit(blended)
        return true, abs_pred, delta_pred, blended

    def predict_video(self, video_id: str) -> tuple[np.ndarray, np.ndarray]:
        true, _, _, blended = self._predict_blended(video_id)
        return true, blended

    def _evaluate_blended(self, video_ids: list[str], split: str) -> dict:
        all_true, all_pred = [], []
        for vid in video_ids:
            true, _, _, blended = self._predict_blended(vid)
            all_true.append(true)
            all_pred.append(blended)
        metrics = calc_retention_metrics(np.concatenate(all_true), np.concatenate(all_pred))
        logger.info("%s [blended α=%.2f] — MSE: %.4f, MAE: %.4f, R2: %.4f", split, self.alpha, metrics["mse"], metrics["mae"], metrics["r2"])
        return metrics

    def _write_tensorboard(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        for label, evals in [("abs", self.evals_result_abs), ("delta", self.evals_result_delta)]:
            for pool, metrics in evals.items():
                split = "train" if pool == "learn" else "val"
                for metric, values in metrics.items():
                    for step, v in enumerate(values):
                        writer.add_scalar(f"{label}/{metric}/{split}", v, step)
        writer.close()
        logger.info("TensorBoard logs saved to %s", log_dir)

    def plot_training_curves(self, output_path: str | None = None) -> Figure:
        fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE["training"])
        for row, (label, evals) in enumerate([("Absolute", self.evals_result_abs), ("Delta", self.evals_result_delta)]):
            for col, metric in enumerate(["RMSE", "MAE"]):
                ax = axes[row][col]
                if (tr := evals.get("learn", {}).get(metric, [])): ax.plot(tr, label="train", color=COLORS["video"], linewidth=LINE_WIDTH_THIN)
                if (vl := evals.get("validation", {}).get(metric, [])): ax.plot(vl, label="val", color=COLORS["baseline"], linewidth=LINE_WIDTH_THIN)
                ax.set(xlabel="iteration", ylabel=metric, title=f"{label} — {metric}")
                ax.legend()
                ax.grid(True, alpha=GRID_ALPHA)
        plt.tight_layout()
        _save_fig(fig, output_path)
        return fig

    def plot_predictions(self, output_dir: str = PREDICTION_OUTPUT_DIR) -> list[str]:
        paths = []
        for vid in sorted(self.video_frames):
            true, abs_pred, delta_pred, blended = self._predict_blended(vid)
            split = "val" if vid in self.val_ids else "train"
            t = np.arange(len(true))

            fig, (top, bot) = plt.subplots(2, 1, figsize=FIG_SIZE["prediction"], height_ratios=[3, 1], sharex=True)

            top.plot(t, true, color=COLORS["video"], linewidth=LINE_WIDTH_THIN, alpha=0.5, label="actual")
            top.plot(t, abs_pred, color=COLORS["abs_head"], linewidth=LINE_WIDTH_THIN, alpha=0.7, label="abs head")
            top.plot(t, delta_pred, color=COLORS["delta_head"], linewidth=LINE_WIDTH_THIN, alpha=0.7, label="baseline+delta head")
            top.fill_between(t, true, blended, alpha=0.1, color=COLORS["fill"])

            mse, mae = float(np.mean((true - blended) ** 2)), float(np.mean(np.abs(true - blended)))
            top.set(ylabel="Retention (%)", title=f"{vid} [{split}]  MSE={mse:.2f}  MAE={mae:.2f}  α={self.alpha:.2f}")
            top.legend(fontsize=FONT_SIZE_LEGEND)
            top.grid(True, alpha=GRID_ALPHA)

            residual = blended - true
            bot.fill_between(t, residual, alpha=0.3, color=COLORS["positive"], where=residual >= 0)
            bot.fill_between(t, residual, alpha=0.3, color=COLORS["negative"], where=residual < 0)
            bot.axhline(0, color=COLORS["axis"], linewidth=LINE_WIDTH_AXIS)
            bot.set(xlabel="sec", ylabel="error (pp)")
            bot.grid(True, alpha=GRID_ALPHA)

            plt.tight_layout()
            path = os.path.join(output_dir, "videos", vid, *PREDICTION_REL_PATH)
            _save_fig(fig, path)
            paths.append(path)
        logger.info("Saved %d prediction plots to %s", len(paths), output_dir)
        return paths

    def plot_feature_importance(self, top_n: int = DEFAULT_TOP_N, output_path: str | None = None) -> Figure:
        if self.model_abs is None or self.model_delta is None: raise RuntimeError("Model not trained yet")
        blended = self.alpha * self.model_abs.get_feature_importance() + (1 - self.alpha) * self.model_delta.get_feature_importance()
        top_idx = np.argsort(blended)[::-1][:top_n]
        names, values = [self.feature_names[i] for i in top_idx], blended[top_idx]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.barh(range(len(names)), values[::-1], color=COLORS["bar"])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1], fontsize=FONT_SIZE_LEGEND)
        ax.set(xlabel="Importance", title=f"Top-{top_n} feature importances (blended α={self.alpha:.2f})")
        ax.grid(True, alpha=GRID_ALPHA, axis="x")

        plt.tight_layout()
        _save_fig(fig, output_path)
        return fig


class AVDTrainer(BaseCatBoostTrainer):
    LOG_PREFIX = "AVD"
    LOO_METRIC_KEYS: ClassVar[list[str]] = ["val_mae_sec"]

    def __init__(self, video_frames: dict[str, pd.DataFrame], val_ratio: float = DEFAULT_VAL_RATIO, random_state: int = DEFAULT_RANDOM_STATE, split_strategy: SplitStrategy = DEFAULT_SPLIT_STRATEGY, val_video: str | None = None):
        super().__init__(video_frames=video_frames, val_ratio=val_ratio, random_state=random_state, split_strategy=split_strategy, val_video=val_video)
        self.model: cb.CatBoostRegressor | None = None

    def _reset_for_fold(self):
        super()._reset_for_fold()
        self.model = None

    def _aggregate_features(self, df: pd.DataFrame) -> pd.Series:
        cols = [c for c in df.columns if c not in FEATURE_EXCLUDE_COLS]
        agg = df[cols].astype(float).agg(AVD_AGG_STATS).T.fillna(0)
        return pd.Series(agg.to_numpy().ravel(), index=[f"{f}_{s}" for f in agg.index for s in agg.columns])

    def _compute_feature_names(self):
        names: set[str] = set()
        for vid in self.train_ids:
            names.update(self._aggregate_features(self.video_frames[vid]).index)
        self.feature_names = sorted(names)

    def _build_dataset(self, video_ids: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
        if not self.feature_names: self._compute_feature_names()
        X = pd.DataFrame([self._aggregate_features(self.video_frames[v]).reindex(self.feature_names, fill_value=0) for v in video_ids]).fillna(0)
        y = np.array([compute_avd_from_retention(self.video_frames[v]["retention"].to_numpy(dtype=float), float(len(self.video_frames[v]))) for v in video_ids], dtype=float)
        return X, y

    def train(self, iterations: int = DEFAULT_ITERATIONS, depth: int = DEFAULT_DEPTH, learning_rate: float = DEFAULT_LEARNING_RATE, early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS, save_path: str | None = DEFAULT_AVD_MODEL_PATH) -> dict:
        X_train, y_train = self._build_dataset(self.train_ids)
        X_val, y_val = self._build_dataset(self.val_ids)

        params = self._catboost_params(iterations, depth, learning_rate, early_stopping_rounds, verbose=CATBOOST_VERBOSE)
        self.model, elapsed = self._fit_regressor(X_train, y_train, X_val, y_val, params, log_label="AVD model")
        self._save_model(self.model, save_path, label="AVD model")

        train_mae = self._mae(y_train, self.model.predict(X_train))
        val_mae = self._mae(y_val, self.model.predict(X_val))
        logger.info("AVD Train MAE: %.2f sec, Val MAE: %.2f sec", train_mae, val_mae)

        return {"train_mae_sec": round(train_mae, 2), "val_mae_sec": round(val_mae, 2), "elapsed_sec": round(elapsed, 1), "feature_importance": self._feature_importance_dict(self.model)}
