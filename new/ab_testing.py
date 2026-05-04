"""A/B-style statistical comparison for model metrics (and generic paired samples).

Typical use: two ``metrics.json`` files from training runs with a ``per_video``
dict containing ``mae``, ``rmse``, ``pearson``, etc. The module aligns videos
common to both runs and reports:

- bootstrap confidence intervals for the mean of each metric;
- paired difference (B − A or A − B) with CI;
- paired t-test, Wilcoxon signed-rank, and a paired permutation test on the mean difference;
- Cohen's d on paired differences (effect size).

Requires ``numpy``; ``scipy`` is optional but recommended for p-values.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


@dataclass
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    n: int
    statistic_name: str


def bootstrap_statistic(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 8000,
    alpha: float = 0.05,
    random_state: int | None = None,
    statistic_name: str = "mean",
) -> BootstrapResult:
    """Percentile bootstrap CI for a scalar functional of i.i.d. samples."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    n = int(x.size)
    if n == 0:
        raise ValueError("samples is empty")
    rng = np.random.default_rng(random_state)
    stat_obs = float(statistic(x))
    n_boot = max(100, int(n_bootstrap))
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = statistic(x[idx])
    q_lo, q_hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return BootstrapResult(point=stat_obs, ci_low=float(np.percentile(boot, q_lo)), ci_high=float(np.percentile(boot, q_hi)), n=n, statistic_name=statistic_name)


def paired_difference_ci(
    a: np.ndarray, b: np.ndarray, n_bootstrap: int = 8000, alpha: float = 0.05, random_state: int | None = None, second_minus_first: bool = True
) -> tuple[BootstrapResult, np.ndarray]:
    """Bootstrap CI for mean(b - a) if ``second_minus_first`` else mean(a - b)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    diff = (b - a) if second_minus_first else (a - b)
    br = bootstrap_statistic(diff, statistic=np.mean, n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state, statistic_name="mean_paired_diff")
    return br, diff


def cohens_d_paired(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=np.float64).ravel()
    sd = float(np.std(d, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(d) / sd)


def paired_ttest(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if scipy_stats is None:
        return {"error": "scipy not installed", "statistic": None, "pvalue": None}
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    t, p = scipy_stats.ttest_rel(a, b, nan_policy="omit")
    return {"test": "paired_ttest", "statistic": float(t), "pvalue": float(p)}


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if scipy_stats is None:
        return {"error": "scipy not installed", "statistic": None, "pvalue": None}
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    try:
        w, p = scipy_stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError as e:
        return {"test": "wilcoxon", "error": str(e), "statistic": None, "pvalue": None}
    return {"test": "wilcoxon", "statistic": float(w), "pvalue": float(p)}


def paired_permutation_mean_test(a: np.ndarray, b: np.ndarray, n_perm: int = 10000, random_state: int | None = None, second_minus_first: bool = True) -> dict[str, Any]:
    """H0: mean(b-a)=0 vs two-sided alternative; permutation flips signs of differences."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = (b - a) if second_minus_first else (a - b)
    obs = float(np.mean(d))
    rng = np.random.default_rng(random_state)
    n = d.size
    if n == 0:
        return {"test": "paired_permutation", "statistic": obs, "pvalue": 1.0, "n_perm": 0}
    extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs(np.mean(d * signs)) >= abs(obs) - 1e-15:
            extreme += 1
    p = (extreme + 1) / (n_perm + 1)
    return {"test": "paired_permutation_mean", "statistic": obs, "pvalue": float(p), "n_perm": n_perm}


def load_metrics_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def extract_per_video_metric(metrics_doc: dict[str, Any], metric_key: str, per_video_key: str = "per_video", split_filter: str | None = None) -> tuple[list[str], np.ndarray]:
    """Return (video_ids, values) for videos that have ``metric_key``."""
    pv = metrics_doc.get(per_video_key)
    if not isinstance(pv, dict):
        raise KeyError(f"expected '{per_video_key}' dict in metrics file")
    ids: list[str] = []
    vals: list[float] = []
    for vid, row in pv.items():
        if not isinstance(row, dict) or metric_key not in row:
            continue
        if split_filter is not None and row.get("split") != split_filter:
            continue
        try:
            v = float(row[metric_key])
        except (TypeError, ValueError):
            continue
        if math.isnan(v):
            continue
        ids.append(str(vid))
        vals.append(v)
    return ids, np.asarray(vals, dtype=np.float64)


def align_paired_metrics(ids_a: list[str], vals_a: np.ndarray, ids_b: list[str], vals_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    map_b = {i: v for i, v in zip(ids_b, vals_b, strict=True)}
    out_a, out_b, common = [], [], []
    for i, va in zip(ids_a, vals_a, strict=True):
        if i in map_b:
            common.append(i)
            out_a.append(va)
            out_b.append(map_b[i])
    return np.asarray(out_a, dtype=np.float64), np.asarray(out_b, dtype=np.float64), common


@dataclass
class MetricComparisonReport:
    metric: str
    n_common: int
    mean_a: float
    mean_b: float
    bootstrap_a: BootstrapResult
    bootstrap_b: BootstrapResult
    paired_diff: BootstrapResult
    cohens_d: float
    paired_ttest: dict[str, Any]
    wilcoxon: dict[str, Any]
    permutation: dict[str, Any]


def compare_metric_paired(
    vals_a: np.ndarray, vals_b: np.ndarray, metric_name: str, n_bootstrap: int = 8000, alpha: float = 0.05, random_state: int | None = None, second_minus_first: bool = True
) -> MetricComparisonReport:
    if vals_a.shape != vals_b.shape:
        raise ValueError("paired arrays must match")
    ba = bootstrap_statistic(vals_a, n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state)
    bb = bootstrap_statistic(vals_b, n_bootstrap=n_bootstrap, alpha=alpha, random_state=(random_state + 1) if random_state is not None else None)
    bd, diff = paired_difference_ci(vals_a, vals_b, n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state, second_minus_first=second_minus_first)
    label = "mean(B−A)" if second_minus_first else "mean(A−B)"
    bd = BootstrapResult(point=bd.point, ci_low=bd.ci_low, ci_high=bd.ci_high, n=bd.n, statistic_name=label)
    return MetricComparisonReport(
        metric=metric_name,
        n_common=int(vals_a.size),
        mean_a=float(np.mean(vals_a)),
        mean_b=float(np.mean(vals_b)),
        bootstrap_a=ba,
        bootstrap_b=bb,
        paired_diff=bd,
        cohens_d=cohens_d_paired(diff),
        paired_ttest=paired_ttest(vals_a, vals_b),
        wilcoxon=wilcoxon_signed_rank(vals_a, vals_b),
        permutation=paired_permutation_mean_test(vals_a, vals_b, random_state=random_state, second_minus_first=second_minus_first),
    )


def compare_metrics_json_files(
    path_a: str | Path,
    path_b: str | Path,
    metric_keys: Iterable[str] = ("mae", "rmse", "pearson"),
    per_video_key: str = "per_video",
    split_filter: str | None = None,
    n_bootstrap: int = 8000,
    alpha: float = 0.05,
    random_state: int | None = None,
    second_minus_first: bool = True,
) -> dict[str, Any]:
    doc_a = load_metrics_json(path_a)
    doc_b = load_metrics_json(path_b)
    out: dict[str, Any] = {"path_a": str(path_a), "path_b": str(path_b), "split_filter": split_filter, "metrics": {}}
    for mk in metric_keys:
        ids_a, va = extract_per_video_metric(doc_a, mk, per_video_key, split_filter)
        ids_b, vb = extract_per_video_metric(doc_b, mk, per_video_key, split_filter)
        a_p, b_p, common = align_paired_metrics(ids_a, va, ids_b, vb)
        if a_p.size < 2:
            out["metrics"][mk] = {"error": "not enough common videos", "n_common": int(a_p.size)}
            continue
        rep = compare_metric_paired(a_p, b_p, mk, n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state, second_minus_first=second_minus_first)
        out["metrics"][mk] = {
            "n_common": rep.n_common,
            "mean_a": rep.mean_a,
            "mean_b": rep.mean_b,
            "ci_mean_a": {"low": rep.bootstrap_a.ci_low, "high": rep.bootstrap_a.ci_high},
            "ci_mean_b": {"low": rep.bootstrap_b.ci_low, "high": rep.bootstrap_b.ci_high},
            "paired_mean_diff": {"label": rep.paired_diff.statistic_name, "point": rep.paired_diff.point, "ci_low": rep.paired_diff.ci_low, "ci_high": rep.paired_diff.ci_high},
            "cohens_d_paired": rep.cohens_d,
            "paired_ttest": rep.paired_ttest,
            "wilcoxon": rep.wilcoxon,
            "permutation": rep.permutation,
        }
    return out


def _report_to_text(report: dict[str, Any]) -> str:
    lines = [f"Compare: A = {report['path_a']}", f"         B = {report['path_b']}", f"Split filter: {report.get('split_filter')}", ""]
    for mk, block in report.get("metrics", {}).items():
        lines.append(f"=== {mk} ===")
        if "error" in block:
            lines.append(f"  {block['error']} (n={block.get('n_common', 0)})")
            lines.append("")
            continue
        lines.append(f"  n_common: {block['n_common']}")
        lines.append(f"  mean A: {block['mean_a']:.6g}  CI [{block['ci_mean_a']['low']:.6g}, {block['ci_mean_a']['high']:.6g}]")
        lines.append(f"  mean B: {block['mean_b']:.6g}  CI [{block['ci_mean_b']['low']:.6g}, {block['ci_mean_b']['high']:.6g}]")
        pd = block["paired_mean_diff"]
        lines.append(f"  {pd['label']}: {pd['point']:.6g}  CI [{pd['ci_low']:.6g}, {pd['ci_high']:.6g}]")
        lines.append(f"  Cohen d (paired): {block['cohens_d_paired']:.6g}")
        tt = block["paired_ttest"]
        if tt.get("pvalue") is not None:
            lines.append(f"  paired t-test: t={tt.get('statistic'):.4g} p={tt['pvalue']:.4g}")
        wx = block["wilcoxon"]
        if wx.get("pvalue") is not None:
            lines.append(f"  Wilcoxon: W={wx.get('statistic'):.4g} p={wx['pvalue']:.4g}")
        pm = block["permutation"]
        lines.append(f"  permutation (mean diff): stat={pm.get('statistic'):.6g} p={pm.get('pvalue'):.4g}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two metrics.json (paired per video)")
    p.add_argument("metrics_a", type=Path, help="Baseline / model A metrics.json")
    p.add_argument("metrics_b", type=Path, help="Model B metrics.json")
    p.add_argument("--metrics", nargs="+", default=["mae", "rmse", "pearson"])
    p.add_argument("--split", choices=["train", "val", "any"], default="any", help="Restrict to videos with this split tag; 'any' uses all common videos")
    p.add_argument("--bootstrap", type=int, default=8000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--a-minus-b", action="store_true", help="Report mean(A−B) instead of mean(B−A)")
    args = p.parse_args()

    split_filter = None if args.split == "any" else args.split
    report = compare_metrics_json_files(
        args.metrics_a,
        args.metrics_b,
        metric_keys=args.metrics,
        split_filter=split_filter,
        n_bootstrap=args.bootstrap,
        alpha=args.alpha,
        random_state=args.seed,
        second_minus_first=not args.a_minus_b,
    )
    print(_report_to_text(report))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
