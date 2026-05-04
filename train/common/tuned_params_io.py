"""
Load Optuna-style best_params JSON and merge into training argparse.Namespace.

Used by multimodal and tabular sequence trainers so v4 "tuned" runs can point at
`tune_hp/results/*_best.json` without re-running Optuna.
"""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Literal


logger = logging.getLogger(__name__)

ModelFamily = Literal["multimodal_lstm", "multimodal_transformer", "tabular_transformer"]


def load_tuned_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Tuned params JSON not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid tuned JSON (expected object): {p}")
    return data


def apply_best_params_to_args(args: Namespace, best_params: dict[str, Any], *, model_family: ModelFamily, apply_architecture: bool) -> Namespace:
    """Mutates args in place; returns args."""
    if not best_params:
        logger.warning("apply_best_params_to_args: empty best_params")
        return args

    shared_map = {
        "lr": "lr",
        "weight_decay": "weight_decay",
        "alpha_corr": "alpha_corr",
        "alpha_smooth": "alpha_smooth",
        "alpha_delta": "alpha_delta",
        "alpha_mono": "alpha_mono",
        "ad_penalty": "ad_penalty_weight",
        "start_boost_secs": "start_boost_secs",
        "start_boost_factor": "start_boost_factor",
        "window_size": "window_size",
        "batch_size": "batch_size",
        "feature_mask_prob": "feature_mask_prob",
        "noise_std": "noise_std",
    }
    for src, dst in shared_map.items():
        if src in best_params:
            setattr(args, dst, best_params[src])

    if "window_size" in best_params:
        ws = int(best_params["window_size"])
        args.window_stride = max(1, ws // 2)

    # Feature filtering (stored as optional attributes consumed by trainers)
    fk = {"redundant_corr_threshold": "tuned_corr_threshold", "max_nan_pct": "tuned_nan_pct", "min_nonzero_pct": "tuned_nonzero_pct", "top_k_features": "tuned_top_k"}
    for src, dst in fk.items():
        if src in best_params:
            setattr(args, dst, best_params[src])

    if not apply_architecture:
        logger.info("Tuned JSON: applied loss / optimizer / window / feature-filter params (architecture kept from CLI — use --tuned-apply-architecture to override).")
        return args

    if model_family == "multimodal_transformer":
        arch_map = {"d_model": "d_model", "n_heads": "n_heads", "n_layers": "n_layers", "d_ff": "d_ff", "dropout": "dropout"}
    elif model_family == "multimodal_lstm":
        arch_map = {"hidden_size": "d_model", "n_layers": "n_layers", "dropout": "dropout"}
    else:
        arch_map = {"d_model": "d_model", "n_heads": "n_heads", "n_layers": "n_layers", "d_ff": "d_ff", "dropout": "dropout"}

    for src, dst in arch_map.items():
        if src in best_params:
            setattr(args, dst, best_params[src])

    logger.info("Tuned JSON: also applied architecture fields from study.")
    return args


def merge_tuned_file_into_args(args: Namespace, json_path: str | Path, *, model_family: ModelFamily, apply_architecture: bool, save_copy_to: Path | None = None) -> Namespace:
    data = load_tuned_json(json_path)
    best = data.get("best_params")
    if not isinstance(best, dict):
        raise ValueError(f"No 'best_params' object in {json_path}")
    apply_best_params_to_args(args, best, model_family=model_family, apply_architecture=apply_architecture)
    if save_copy_to is not None:
        save_copy_to.parent.mkdir(parents=True, exist_ok=True)
        save_copy_to.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved tuned payload copy -> %s", save_copy_to)
    logger.info("Merged tuned params from %s", json_path)
    return args
