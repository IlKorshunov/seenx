from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass(frozen=True)
class Experiment:
    module: str
    supports_args: bool = True


EXPERIMENTS: dict[str, Experiment] = {
    "bert.seq": Experiment("train.bert.train_bert_seq"),
    "clustering.content_specialists": Experiment("train.clustering.content_cluster_specialists"),
    "clustering.multimodal_specialists": Experiment("train.clustering.cluster_specialists_multimodal"),
    "loo.ad_peak_weighted": Experiment("train.loo.train_retention_ad_peak_weighted_loo"),
    "loo.blended_quantile": Experiment("train.loo.train_retention_blended_quantile_loo"),
    "loo.conservative_catboost": Experiment("train.loo.train_retention_conservative_catboost_loo"),
    "loo.decomposed": Experiment("train.loo.train_retention_decomposed_loo"),
    "loo.hybrid": Experiment("train.loo.train_retention_hybrid_loo"),
    "loo.integration_penalty": Experiment("train.loo.train_retention_integration_penalty_loo"),
    "loo.kernel_baseline": Experiment("train.loo.train_retention_kernel_baseline_loo"),
    "loo.local_knn": Experiment("train.loo.train_retention_local_knn_loo"),
    "loo.meta_ensemble": Experiment("train.loo.train_retention_meta_ensemble_loo"),
    "loo.peak_weighted": Experiment("train.loo.train_retention_peak_weighted_loo"),
    "loo.ranker": Experiment("train.loo.train_retention_ranker_loo"),
    "loo.regressor": Experiment("train.loo.train_retention_regressor_loo"),
    "loo.report": Experiment("train.loo.loo_report"),
    "loo.residual_huber": Experiment("train.loo.train_retention_residual_huber_loo"),
    "loo.run_all_videos": Experiment("train.loo.run_loo_all_videos", supports_args=False),
    "loo.shape_only": Experiment("train.loo.train_retention_shape_only_loo"),
    "loo.stacked": Experiment("train.loo.train_retention_stacked_loo"),
    "loo.transformer": Experiment("train.loo.train_retention_transformer_video_loo"),
    "loo.transformer_v2": Experiment("train.loo.train_retention_transformer_v2_loo"),
    "loo.transformer_v2_video": Experiment("train.loo.train_retention_transformer_v2_video_loo"),
    "loo.xgb_flat": Experiment("train.loo.train_retention_xgb_flat_loo"),
    "lstm.seq": Experiment("train.lstm.train_lstm_seq"),
    "lstm.seq_v1": Experiment("train.lstm.train_lstm_seq_v1"),
    "metamodel.train": Experiment("train.metamodel.train_metamodel"),
    "tabular.catboost_regressor": Experiment("train.tabular_catboost.train_catboost_regressor"),
    "tools.compare_fixed_inference_models": Experiment("train.tools.compare_fixed_inference_models", supports_args=False),
    "tools.merge_output_partials": Experiment("train.tools.merge_output_partials"),
    "tools.sync_partial_from_main": Experiment("train.tools.sync_partial_from_main"),
    "tools.visualize_ranker_experiment": Experiment("train.tools.visualize_ranker_experiment"),
    "transformer.multimodal_seq": Experiment("train.transformer.train_multimodal_seq"),
    "transformer.seq": Experiment("train.transformer.train_transformer_seq"),
    "transformer.seq_v1": Experiment("train.transformer.train_transformer_seq_v1"),
    "videomae.seq": Experiment("train.videomae.train_videomae_seq"),
}

def dispatch(key: str, forwarded_args: list[str]) -> None:
    if key not in EXPERIMENTS:
        print(f"Unknown experiment: {key}", file=sys.stderr)
        print("Run `python -m train.run --list` to see available experiments", file=sys.stderr)
        raise SystemExit(2)

    entry = EXPERIMENTS[key]
    if forwarded_args and not entry.supports_args:
        print(f"Warning: {key} does not define CLI args")

    selected_main = getattr(importlib.import_module(entry.module), "main", None)
    if selected_main is None:
        raise SystemExit(f"Experiment module has no main(): {entry.module}")

    old_argv = sys.argv
    sys.argv = [f"train.run {key}", *forwarded_args]
    try:
        selected_main()
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "--help":
        print("CLI HELP: python -m train.run <experiment> [args]")
        return
    if args[0] == "--list":
        for key in sorted(EXPERIMENTS):
            print(f"{key:36} {EXPERIMENTS[key].module}")
        return
    dispatch(args[0], args[1:])

if __name__ == "__main__":
    main()
