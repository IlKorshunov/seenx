"""Cluster-aware SARIMAX trend priors for retention prediction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_ORDER_GRID = ((1, 0, 1), (2, 0, 1), (1, 1, 1), (2, 1, 1), (2, 1, 2))

DEFAULT_EXOG_COLS = ("hook_score", "topic_change_rate", "question_density", "is_ad", "edit_pace", "duration_sec", "video_cluster")


def _safe_cluster_id(df: pd.DataFrame) -> int:
    if "video_cluster" not in df.columns:
        return 0
    ser = pd.to_numeric(df["video_cluster"], errors="coerce").dropna()
    if ser.empty:
        return 0
    return int(round(float(ser.iloc[0])))


def _mean_curve(video_dfs: dict[str, pd.DataFrame], video_ids: Iterable[str]) -> np.ndarray:
    video_ids = list(video_ids)
    if not video_ids:
        return np.zeros(1, dtype=np.float32)
    max_len = max(len(video_dfs[vid]) for vid in video_ids)
    acc = np.zeros(max_len, dtype=np.float64)
    cnt = np.zeros(max_len, dtype=np.float64)
    for vid in video_ids:
        y = pd.to_numeric(video_dfs[vid]["retention"], errors="coerce").fillna(0.0).values.astype(np.float64)
        acc[: len(y)] += y
        cnt[: len(y)] += 1.0
    return (acc / np.maximum(cnt, 1.0)).astype(np.float32)


def _build_exog_matrix(df: pd.DataFrame, exog_cols: tuple[str, ...]) -> np.ndarray:
    n = len(df)
    mats: list[np.ndarray] = []
    for col in exog_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").ffill().fillna(0.0).values.astype(np.float64)
        elif col == "duration_sec":
            vals = np.full(n, float(n), dtype=np.float64)
        elif col == "video_cluster":
            vals = np.full(n, float(_safe_cluster_id(df)), dtype=np.float64)
        else:
            vals = np.zeros(n, dtype=np.float64)
        mats.append(vals.reshape(-1, 1))
    if not mats:
        return np.zeros((n, 0), dtype=np.float64)
    return np.concatenate(mats, axis=1)


def _mean_exog(video_dfs: dict[str, pd.DataFrame], video_ids: Iterable[str], exog_cols: tuple[str, ...]) -> np.ndarray:
    video_ids = list(video_ids)
    if not video_ids:
        return np.zeros((1, len(exog_cols)), dtype=np.float64)
    max_len = max(len(video_dfs[vid]) for vid in video_ids)
    acc = np.zeros((max_len, len(exog_cols)), dtype=np.float64)
    cnt = np.zeros((max_len, len(exog_cols)), dtype=np.float64)
    for vid in video_ids:
        exog = _build_exog_matrix(video_dfs[vid], exog_cols)
        acc[: len(exog)] += exog
        cnt[: len(exog)] += 1.0
    return acc / np.maximum(cnt, 1.0)


def _fit_best_order(series: np.ndarray, exog: np.ndarray, order_grid=DEFAULT_ORDER_GRID):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = np.asarray(series, dtype=np.float64)
    best_aic = float("inf")
    best_model = None
    best_order = None
    for order in order_grid:
        try:
            model = SARIMAX(y, exog=exog if exog.size else None, order=order, trend="c", enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic = float(res.aic)
                best_model = res
                best_order = order
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError("Failed to fit any SARIMA candidate")
    return best_model, best_order, best_aic


@dataclass
class FittedSarimaxTrend:
    cluster_id: int
    order: tuple[int, int, int]
    aic: float
    base_series: np.ndarray
    base_exog: np.ndarray
    train_len: int
    exog_cols: tuple[str, ...]
    model_result: object

    def predict_from_df(self, df: pd.DataFrame) -> np.ndarray:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        exog = _build_exog_matrix(df, self.exog_cols)
        endog = np.full(len(df), np.nan, dtype=np.float64)
        model = SARIMAX(endog, exog=exog if exog.size else None, order=self.order, trend="c", enforce_stationarity=False, enforce_invertibility=False)
        result = model.smooth(self.model_result.params)
        pred = np.asarray(result.get_prediction().predicted_mean, dtype=np.float64)
        return np.clip(pred, 0.0, 100.0).astype(np.float32)


class ClusterSarimaxTrendProvider:
    """Fits one SARIMAX trend per cluster and serves priors for new videos."""

    def __init__(self, order_grid=DEFAULT_ORDER_GRID, exog_cols: tuple[str, ...] = DEFAULT_EXOG_COLS):
        self.order_grid = order_grid
        self.exog_cols = exog_cols
        self.global_model: FittedSarimaxTrend | None = None
        self.by_cluster: dict[int, FittedSarimaxTrend] = {}

    def fit(self, video_dfs: dict[str, pd.DataFrame], train_ids: list[str]) -> ClusterSarimaxTrendProvider:
        cluster_to_ids: dict[int, list[str]] = {}
        for vid in train_ids:
            cid = _safe_cluster_id(video_dfs[vid])
            cluster_to_ids.setdefault(cid, []).append(vid)

        global_series = _mean_curve(video_dfs, train_ids)
        global_exog = _mean_exog(video_dfs, train_ids, self.exog_cols)
        global_res, global_order, global_aic = _fit_best_order(global_series, global_exog, self.order_grid)
        self.global_model = FittedSarimaxTrend(
            cluster_id=-1,
            order=global_order,
            aic=global_aic,
            base_series=global_series,
            base_exog=global_exog,
            train_len=len(global_series),
            exog_cols=self.exog_cols,
            model_result=global_res,
        )

        for cid, vids in cluster_to_ids.items():
            if len(vids) < 2:
                continue
            mean_series = _mean_curve(video_dfs, vids)
            mean_exog = _mean_exog(video_dfs, vids, self.exog_cols)
            res, order, aic = _fit_best_order(mean_series, mean_exog, self.order_grid)
            self.by_cluster[cid] = FittedSarimaxTrend(
                cluster_id=cid, order=order, aic=aic, base_series=mean_series, base_exog=mean_exog, train_len=len(mean_series), exog_cols=self.exog_cols, model_result=res
            )
        return self

    def get_baseline_for_df(self, df: pd.DataFrame) -> np.ndarray:
        if self.global_model is None:
            raise RuntimeError("ClusterSarimaxTrendProvider is not fitted")
        cid = _safe_cluster_id(df)
        model = self.by_cluster.get(cid, self.global_model)
        return model.predict_from_df(df)

    def describe(self) -> dict:
        if self.global_model is None:
            return {}
        return {
            "global": {
                "order": self.global_model.order,
                "aic": round(self.global_model.aic, 4),
                "length": self.global_model.train_len,
                "exog_cols": list(self.global_model.exog_cols),
            },
            "clusters": {
                str(cid): {"order": fitted.order, "aic": round(fitted.aic, 4), "length": fitted.train_len, "exog_cols": list(fitted.exog_cols)}
                for cid, fitted in sorted(self.by_cluster.items())
            },
        }
