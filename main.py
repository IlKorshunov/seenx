"""
Central entry point for seenx-ml pipeline.
Usage:
    python main.py retention --html_dir data/html/ --strategy mean_duration
    python main.py aggregate -v video.mp4 -o output.csv
    python main.py predict --model_path model.cbm --video_path video.mp4 --config_path configs/local.json
    python main.py extract_bumpers --data_dir testing --output_dir result
    python main.py extract_stems --video-id 1J5zlq2Vs3Y
    python main.py channel_baseline --data_dir data --all
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import matplotlib.pyplot as plt

from scripts.error_analysis import run_error_analysis
from src.aggregator import aggregate, aggregate_batch
from src.analysis.channel_baseline import run as run_channel_baseline_batch
from src.analysis.retention_clustering import run_retention_clustering
from src.app.predict import Predictor
from src.baseline import AVDTrainer, RetentionTrainer, compare_all, plot_comparison
from src.cutting_shots.pipeline import run_bumper_pipeline
from src.ensembles import EnsemblePredictor
from src.extractors.audio.source_separation import combine, mp4_to_wav, separate
from src.retention_analysis import (
    _format_time,
    channel_metrics_table,
    compute_channel_baseline,
    load_channel_retentions,
    load_channel_retentions_csv,
    plot_all_strategies,
    plot_channel_baseline,
    plot_single_retention,
)
from src.utils.config import Config
from src.utils.logger import Logger
from train.transformer.default_transformer_trainer import TransformerTrainer


logger = Logger(show=True).get_logger()
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
_DEFAULT_CHANNEL_BASELINE_DIR = os.path.join(_REPO_ROOT, "function_result", "baseline")
_DEFAULT_BUMPERS_AUDIO_CFG = os.path.join(_REPO_ROOT, "configs", "bumpers.json")
_DEFAULT_BUMPERS_VIDEO_CFG = os.path.join(_REPO_ROOT, "configs", "bumpers_video.json")
_CHANNEL_BASELINE_STRATEGIES = ("min_duration", "max_duration", "mean_duration", "extrapolate")


def run_retention(args):
    os.makedirs(args.output_dir, exist_ok=True)
    channel_data = load_channel_retentions(args.html_dir, args.video_dir)
    if not channel_data:
        logger.error("No data loaded, exiting")
        return
    metrics_df = channel_metrics_table(channel_data)
    logger.info("Channel Metrics:%s%s", os.linesep, metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(args.output_dir, "channel_metrics.csv"), index=False)
    for d in channel_data:
        plot_single_retention(
            d["retention_series"], d["duration_sec"], title=f"Retention: {d['name']}", output_path=os.path.join(args.output_dir, f"retention_{d['name']}.png"), show=False
        )
        plt.close()
    baseline_result = compute_channel_baseline(channel_data, strategy=args.strategy)
    if baseline_result is None:
        logger.error("Failed to compute baseline, exiting")
        return
    plot_channel_baseline(baseline_result, output_path=os.path.join(args.output_dir, f"baseline_{args.strategy}.png"), show=False)
    plt.close()
    if args.compare_all:
        plot_all_strategies(channel_data, output_path=os.path.join(args.output_dir, "all_strategies_comparison.png"), show=False)
        plt.close()
    logger.info("Results saved to %s", args.output_dir)
    logger.info("Baseline strategy: %s", args.strategy)
    logger.info("Baseline AVD: %s", _format_time(baseline_result["baseline_metrics"]["avd_sec"]))
    if baseline_result["baseline_metrics"]["retention_30"] is not None:
        logger.info("Baseline retention at 30s: %.1f%%", baseline_result["baseline_metrics"]["retention_30"])


def run_aggregate(args):
    config = Config(config_path=args.config_path)
    if args.batch:
        output_dir = args.output_path if args.output_path else "output"
        only = set(args.only.split(",")) if args.only else None
        aggregate_batch(
            data_dir=args.data_dir,
            output_dir=output_dir,
            config=config,
            only=only,
            skip_comment_features=args.skip_comment_features,
            skip_emotion_features=args.skip_emotion_features,
        )
        logger.info("Batch aggregation complete -> %s", output_dir)
        return
    if not args.video_path:
        logger.error("--video_path (-v) is required in single-video mode. Use --batch for batch mode.")
        return
    only = set(args.only.split(",")) if args.only else None
    aggregated_df = aggregate(
        video_path=args.video_path,
        audio_path=args.video_path,
        output_path=args.output_path,
        config=config,
        html_path=args.retention_path,
        data_dir=args.data_dir,
        only=only,
        skip_comment_features=args.skip_comment_features,
        skip_emotion_features=args.skip_emotion_features,
    )
    aggregated_df.to_csv(args.output_path, index=True)
    logger.info("Aggregated features saved to %s", args.output_path)


def run_predict(args):
    predictor = Predictor(model_path=args.model_path, config_path=args.config_path)
    predictor.predict(args.video_path)
    logger.info("Prediction completed")


def run_extract_stems(args):
    data_dir = os.path.join(args.data_root, args.video_id)
    audio_path = os.path.join(data_dir, "audio.mp3")
    if not os.path.exists(audio_path):
        logger.error("Audio not found: %s", audio_path)
        sys.exit(1)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    wav_path = tmp.name
    logger.info("Converting %s -> %s", audio_path, wav_path)
    mp4_to_wav(audio_path, wav_path)
    logger.info("Running Demucs separation")
    ok = separate([wav_path], outp=args.output_dir, device="cuda", segment=args.segment)
    if not ok:
        logger.warning("Demucs failed on CUDA, retrying on CPU")
        ok = separate([wav_path], outp=args.output_dir, device="cpu", segment=args.segment)
    filename = os.path.splitext(os.path.basename(wav_path))[0]
    separated_folder = os.path.join(args.output_dir, "htdemucs", filename)
    if not ok:
        os.unlink(wav_path)
        logger.error("Demucs failed completely.")
        sys.exit(1)
    music_path, vocal_path = combine(separated_folder)
    logger.info("Combined: music=%s, vocal=%s", music_path, vocal_path)
    stems_dir = os.path.join(data_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)
    for name in ("vocals.mp3", "mixed.mp3", "other.mp3", "drums.mp3", "bass.mp3"):
        src = os.path.join(separated_folder, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(stems_dir, name))
            logger.info("Saved %s", name)
    os.unlink(wav_path)
    shutil.rmtree(separated_folder)
    logger.info("Stems under %s (vocals, mixed, other, drums, bass)", stems_dir)


def run_channel_baseline_cli(args):
    run_channel_baseline_batch(data_dir=args.data_dir, results_dir=args.results_dir, strategy=args.strategy, all_strategies=args.all_strategies)


def run_extract_bumpers_cmd(args):
    run_bumper_pipeline(args)


def _run_baseline_comparison(args, output_dir):
    channel_data = load_channel_retentions_csv(args.data_dir)
    if len(channel_data) >= 2:
        baseline_result = compute_channel_baseline(channel_data, strategy="mean_duration")
        if baseline_result is not None:
            df = compare_all(baseline_result, channel_data)
            df.to_csv(os.path.join(output_dir, "baseline_comparison.csv"), index=False)
            logger.info("Baseline comparison:\n%s", df.to_string(index=False))
            for d in channel_data:
                out_path = os.path.join(output_dir, "videos", d["name"], "baseline", "comparison.png")
                plot_comparison(baseline_result, channel_data, d["name"], output_path=out_path)
    else:
        logger.info("Need >= 2 videos for baseline comparison, skipping")


def _train_catboost(args):
    trainer = RetentionTrainer.from_output_dir(args.features_dir, val_ratio=args.val_ratio, data_dir=args.data_dir)
    results = trainer.train(optuna_trials=args.optuna_trials, save_path=args.save_path)
    logger.info("CatBoost Train MSE=%.4f MAE=%.4f R2=%.4f", results["train"]["mse"], results["train"]["mae"], results["train"]["r2"])
    logger.info("CatBoost Val   MSE=%.4f MAE=%.4f R2=%.4f", results["val"]["mse"], results["val"]["mae"], results["val"]["r2"])
    logger.info("CatBoost alpha=%.2f optuna_val_mae=%s trials=%s", results["alpha"], results.get("optuna_best_val_mae"), results.get("optuna_trials"))
    trainer.plot_training_curves(output_path=os.path.join(args.output_dir, "training_curves.png"))
    trainer.plot_predictions(output_dir=args.output_dir)
    trainer.plot_feature_importance(top_n=20, output_path=os.path.join(args.output_dir, "feature_importance.png"))
    return trainer, results


def _train_transformer(args):
    trainer = TransformerTrainer.from_output_dir(args.features_dir, val_ratio=args.val_ratio)
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.transformer_lr,
        window_size=args.window_size,
        window_stride=args.window_stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        save_path=args.save_path.replace(".cbm", "_transformer.pt"),
        log_dir="train/tensorboard_transformer",
    )
    logger.info("Transformer Train MSE=%.4f MAE=%.4f R2=%.4f", results["train"]["mse"], results["train"]["mae"], results["train"]["r2"])
    logger.info("Transformer Val   MSE=%.4f MAE=%.4f R2=%.4f", results["val"]["mse"], results["val"]["mae"], results["val"]["r2"])
    trainer.plot_training_curves(output_path=os.path.join(args.output_dir, "transformer_training_curves.png"))
    trainer.plot_predictions(output_dir=args.output_dir)
    return trainer, results


def run_train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.target == "avd":
        trainer = AVDTrainer.from_output_dir(args.features_dir, val_ratio=args.val_ratio)
        results = trainer.train(iterations=args.iterations, depth=args.depth, learning_rate=args.lr, save_path=args.save_path.replace(".cbm", "_avd.cbm"))
        logger.info("AVD Train MAE=%.2f sec, Val MAE=%.2f sec", results["train_mae_sec"], results["val_mae_sec"])
        return
    if args.target == "transformer":
        _train_transformer(args)
        _run_baseline_comparison(args, args.output_dir)
        return
    if args.target == "ensemble":
        cb_trainer, cb_results = _train_catboost(args)
        tf_trainer, tf_results = _train_transformer(args)
        ensemble = EnsemblePredictor(cb_trainer, tf_trainer)
        ensemble.tune_alpha(tf_trainer.video_frames, tf_trainer.val_ids)
        train_ens = ensemble.evaluate(tf_trainer.video_frames, tf_trainer.train_ids, "train")
        val_ens = ensemble.evaluate(tf_trainer.video_frames, tf_trainer.val_ids, "val")
        logger.info("Ensemble Train MSE=%.4f MAE=%.4f R2=%.4f", train_ens["mse"], train_ens["mae"], train_ens["r2"])
        logger.info("Ensemble Val   MSE=%.4f MAE=%.4f R2=%.4f (alpha=%.2f)", val_ens["mse"], val_ens["mae"], val_ens["r2"], val_ens["alpha"])
        logger.info("--- Comparison ---")
        logger.info("CatBoost   val MAE=%.4f R2=%.4f", cb_results["val"]["mae"], cb_results["val"]["r2"])
        logger.info("Transformer val MAE=%.4f R2=%.4f", tf_results["val"]["mae"], tf_results["val"]["r2"])
        logger.info("Ensemble    val MAE=%.4f R2=%.4f (alpha=%.2f)", val_ens["mae"], val_ens["r2"], val_ens["alpha"])
        _run_baseline_comparison(args, args.output_dir)
        return
    _train_catboost(args)
    _run_baseline_comparison(args, args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="seenx-ml pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ret = subparsers.add_parser("retention", help="Retention analysis: AVD, retention at 30s, channel baseline")
    ret.add_argument("--html_dir", type=str, required=True, help="Directory with YouTube retention HTML files")
    ret.add_argument("--video_dir", type=str, default=None, help="Directory with video files (for duration)")
    ret.add_argument("--strategy", type=str, default="mean_duration", choices=["min_duration", "max_duration", "mean_duration", "extrapolate"])
    ret.add_argument("--output_dir", type=str, default="retention_output")
    ret.add_argument("--compare_all", action="store_true")
    agg = subparsers.add_parser("aggregate", help="Extract and aggregate all features from video")
    agg.add_argument("-v", "--video_path", type=str, required=False)
    agg.add_argument("-o", "--output_path", type=str, required=False, default="output")
    agg.add_argument("-r", "--retention_path", type=str, required=False)
    agg.add_argument("-c", "--config_path", type=str, required=False)
    agg.add_argument("-d", "--data_dir", type=str, default="data")
    agg.add_argument("--only", type=str, default=None, help="Comma-separated feature names to extract (e.g. edit_pace,hook_score). Default: all")
    agg.add_argument("--batch", action="store_true", help="Batch mode: process all videos in data_dir with model-group optimization")
    agg.add_argument("--skip-comment-features", action="store_true", help="Do not run comment_social / extract_comment_features (use when output already has these columns)")
    agg.add_argument("--skip-emotion-features", action="store_true", help="Do not run text_sentiment (sent_*) and skip Ekman fusion + dropping raw emotion cols")
    pred = subparsers.add_parser("predict", help="Predict retention curve for a video")
    pred.add_argument("--model_path", type=str, required=True)
    pred.add_argument("--video_path", type=str, required=True)
    pred.add_argument("--config_path", type=str, required=True)
    tr = subparsers.add_parser("train", help="Train CatBoost model + feature importance + baseline comparison")
    tr.add_argument("--features_dir", type=str, default="output")
    tr.add_argument("--data_dir", type=str, default="data")
    tr.add_argument("--output_dir", type=str, default="my_metrics")
    tr.add_argument("--save_path", type=str, default="static/weights/model.cbm")
    tr.add_argument("--iterations", type=int, default=1000, help="AVD CatBoost iterations (retention head uses Optuna)")
    tr.add_argument("--depth", type=int, default=6, help="AVD CatBoost depth (retention head uses Optuna)")
    tr.add_argument("--lr", type=float, default=0.05, help="AVD CatBoost lr (retention head uses Optuna)")
    tr.add_argument("--optuna_trials", type=int, default=20, help="Retention dual-head: Optuna trials (each retrains abs+delta; default matches src.baseline.OPTUNA_TRIALS)")
    tr.add_argument("--val_ratio", type=float, default=0.2)
    tr.add_argument(
        "--target",
        type=str,
        default="retention",
        choices=["retention", "avd", "transformer", "ensemble"],
        help="retention=CatBoost, transformer=Transformer, ensemble=both blended, avd=AVD model",
    )
    tr.add_argument("--epochs", type=int, default=200, help="Transformer training epochs")
    tr.add_argument("--batch_size", type=int, default=16, help="Transformer batch size")
    tr.add_argument("--transformer_lr", type=float, default=1e-3, help="Transformer learning rate")
    tr.add_argument("--window_size", type=int, default=128, help="Transformer sliding window size (seconds)")
    tr.add_argument("--window_stride", type=int, default=64, help="Transformer window stride for training")
    tr.add_argument("--d_model", type=int, default=128, help="Transformer hidden dimension")
    tr.add_argument("--n_heads", type=int, default=4, help="Transformer attention heads")
    tr.add_argument("--n_layers", type=int, default=4, help="Transformer encoder layers")
    tr.add_argument("--d_ff", type=int, default=256, help="Transformer FFN dimension")
    err = subparsers.add_parser("error_analysis", help="Error analysis: MAE per video, residual curves, clustering")
    err.add_argument("--features_dir", type=str, default="output")
    err.add_argument("--model_path", type=str, default="static/weights/model.cbm")
    err.add_argument("--output_dir", type=str, default="my_metrics", help="Base dir; CSVs in output_dir/error_analysis/, plots in output_dir/videos/{vid}/residual/")
    clust = subparsers.add_parser("retention_clusters", help="Cluster retention curves by shape")
    clust.add_argument("--data_dir", type=str, default="data")
    clust.add_argument("--n_clusters", type=int, default=4)
    clust.add_argument("--output_dir", type=str, default="my_metrics/retention_clusters")
    bump = subparsers.add_parser("extract_bumpers", help="Audio fingerprint + CLIP verify + cut bumper clips")
    bump.add_argument("--data_dir", default="testing")
    bump.add_argument("--output_dir", default="result")
    bump.add_argument("--audio_config", default=_DEFAULT_BUMPERS_AUDIO_CFG)
    bump.add_argument("--video_config", default=_DEFAULT_BUMPERS_VIDEO_CFG)
    bump.add_argument("--max_types", type=int, default=10)
    bump.add_argument("--scan_ratio", type=float, default=None)
    bump.add_argument("--skip_video_verify", action="store_true")
    stems = subparsers.add_parser("extract_stems", help="Demucs separation: data/<video_id>/audio.mp3 -> stems in data/<video_id>/stems")
    stems.add_argument("--video-id", dest="video_id", default="1J5zlq2Vs3Y")
    stems.add_argument("--data-root", dest="data_root", default="data")
    stems.add_argument("--output-dir", dest="output_dir", default="static/separated/htdemucs/output_audio")
    stems.add_argument("--segment", type=int, default=4)
    cba = subparsers.add_parser("channel_baseline", help="Channel baseline CSV/plots for all or one strategy (CSV retention in data_dir)")
    cba.add_argument("--data_dir", default="data")
    cba.add_argument("--results_dir", default=_DEFAULT_CHANNEL_BASELINE_DIR)
    cba.add_argument("--strategy", default="mean_duration", choices=_CHANNEL_BASELINE_STRATEGIES)
    cba.add_argument("--all", dest="all_strategies", action="store_true")
    args = parser.parse_args()
    dispatch = {
        "retention": run_retention,
        "aggregate": run_aggregate,
        "predict": run_predict,
        "train": run_train,
        "extract_bumpers": run_extract_bumpers_cmd,
        "extract_stems": run_extract_stems,
        "channel_baseline": run_channel_baseline_cli,
    }
    if args.command in dispatch:
        dispatch[args.command](args)
    elif args.command == "error_analysis":
        run_error_analysis(args.features_dir, args.model_path, args.output_dir)
    elif args.command == "retention_clusters":
        run_retention_clustering(args.data_dir, args.n_clusters, args.output_dir)


if __name__ == "__main__":
    main()
