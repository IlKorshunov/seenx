from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


N_POINTS = 100
MAX_PARAMS = 5

CURVE_DEFS = {"hill": {"names": ["a", "b", "c", "d"], "n": 4}, "double_exp": {"names": ["a", "b", "c", "d", "e"], "n": 5}, "weibull": {"names": ["d", "lam", "k"], "n": 3}}


def hill(x, a, b, c, d):
    return d + (a - d) / (1.0 + np.power(x / c, b))


def double_exp(x, a, b, c, d, e):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e


def weibull(x, d, lam, k):
    return d * np.exp(-np.power(x / lam, k))


def _fit_hill(t, y):
    n = len(t)
    c_max = 0.15 * n
    popt, _ = curve_fit(hill, t, y, p0=[float(y[-1]), 0.8, min(max(1.0, n * 0.2), c_max), float(y[0])], bounds=([0, 0.01, 1, 0], [100, 20, c_max, 100]), maxfev=8000)
    return popt


def _fit_double_exp(t, y):
    drop = max(float(y[0] - y[-1]), 1.0)
    popt, _ = curve_fit(double_exp, t, y, p0=[drop * 0.6, 0.05, drop * 0.3, 0.005, float(y[-1])], bounds=([0, 1e-4, 0, 1e-5, 0], [100, 1.0, 100, 0.5, 100]), maxfev=8000)
    return popt


def _fit_weibull(t, y):
    n = len(t)
    popt, _ = curve_fit(weibull, t + 1.0, y, p0=[float(y[0]), float(n * 0.3), 0.5], bounds=([0, 1, 0.01], [100, n * 2, 5.0]), maxfev=8000)
    return popt


_FITTERS = {"hill": _fit_hill, "double_exp": _fit_double_exp, "weibull": _fit_weibull}
_FUNCS = {"hill": hill, "double_exp": double_exp, "weibull": weibull}


def fit_curve(curve_type: str, retention: np.ndarray) -> np.ndarray | None:
    t = np.arange(len(retention), dtype=np.float64)
    y = np.clip(retention, 0.0, 100.0)
    try:
        return np.array(_FITTERS[curve_type](t, y), dtype=np.float64)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def reconstruct(curve_type: str, params: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    t = np.arange(n_points, dtype=np.float64)
    if curve_type == "weibull":
        return np.clip(weibull(t + 1.0, *params), 0, 100)
    return np.clip(_FUNCS[curve_type](t, *params), 0, 100)


def resample(curve: np.ndarray, target_len: int) -> np.ndarray:
    if len(curve) == target_len:
        return curve.copy()
    return np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(curve)), curve)
