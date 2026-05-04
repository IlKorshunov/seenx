from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def train_models(
    X: pd.DataFrame, targets: np.ndarray, param_names: list[str], iterations: int = 300, depth: int = 4, lr: float = 0.05, rng: int = 42
) -> dict[str, CatBoostRegressor]:
    models: dict[str, CatBoostRegressor] = {}
    for j, name in enumerate(param_names):
        y = targets[:, j]
        valid = np.isfinite(y)
        if valid.sum() < 3:
            continue
        m = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=lr, loss_function="RMSE", verbose=0, random_seed=rng, l2_leaf_reg=5.0)
        m.fit(X.values[valid], y[valid])
        models[name] = m
    return models


def predict_params(models: dict[str, CatBoostRegressor], X: pd.DataFrame, param_names: list[str]) -> np.ndarray:
    out = np.full((len(X), len(param_names)), np.nan)
    for j, name in enumerate(param_names):
        if name in models:
            out[:, j] = models[name].predict(X.values)
    return out


def loo_predict(X: pd.DataFrame, targets: np.ndarray, param_names: list[str], iterations: int = 300, depth: int = 4, lr: float = 0.05, rng: int = 42) -> np.ndarray:
    n = len(X)
    preds = np.full_like(targets, np.nan)
    Xv = X.values
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        for j, name in enumerate(param_names):
            y = targets[:, j]
            train_ok = mask & np.isfinite(y)
            if train_ok.sum() < 3 or not np.isfinite(y[i]):
                continue
            m = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=lr, loss_function="RMSE", verbose=0, random_seed=rng, l2_leaf_reg=5.0)
            m.fit(Xv[train_ok], y[train_ok])
            preds[i, j] = m.predict(Xv[i : i + 1])[0]
    return preds


def feature_importance(models: dict[str, CatBoostRegressor], feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for param_name, model in models.items():
        imp = model.get_feature_importance()
        for feat, val in sorted(zip(feature_names, imp, strict=True), key=lambda t: -t[1])[:20]:
            rows.append({"param": param_name, "feature": feat, "importance": round(float(val), 2)})
    return pd.DataFrame(rows)
