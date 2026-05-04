import argparse
import os

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..retention_analysis import *
from ..utils.logger import Logger


logger = Logger(show=True).get_logger()
ALL_STRATEGIES = ["min_duration", "max_duration", "mean_duration", "extrapolate"]


def run(data_dir="data", results_dir="function_result/baseline", strategy="mean_duration", all_strategies=True):
    os.makedirs(results_dir, exist_ok=True)
    channel_data = load_channel_retentions_csv(data_dir)
    metrics_df = channel_metrics_table(channel_data)
    metrics_df.to_csv(os.path.join(results_dir, "channel_metrics.csv"), index=False)
    logger.info("Channel metrics:%s%s", os.linesep, metrics_df.to_string(index=False))

    if all_strategies:
        plot_all_strategies(channel_data, output_path=os.path.join(results_dir, "all_strategies.png"), show=False)
        plt.close("all")

    for strat in ALL_STRATEGIES if all_strategies else [strategy]:
        result = compute_channel_baseline(channel_data, strategy=strat)
        if result is None:
            logger.error("Failed baseline for strategy=%s", strat)
            continue
        out_png = os.path.join(results_dir, f"baseline_{strat}.png")
        plot_channel_baseline(result, output_path=out_png, show=False)
        plt.close("all")
        pd.DataFrame({"time_sec": result["time_axis"], "baseline_retention": result["baseline_retention"].round(2), "baseline_std": result["baseline_std"].round(2)}).to_csv(
            os.path.join(results_dir, f"baseline_{strat}.csv"), index=False
        )
        bm = result["baseline_metrics"]
        logger.info("[%s] AVD=%s (%.1f%%) ret_30s=%.1f%% mean=%.1f%%", strat, format_time(bm["avd_sec"]), bm["avd_pct"], bm["retention_30"] or 0.0, bm["mean_retention"])


def main():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--results_dir", default=os.path.join(ROOT, "function_result", "baseline"))
    p.add_argument("--strategy", default="mean_duration", choices=ALL_STRATEGIES)
    p.add_argument("--all", dest="all_strategies", action="store_true")
    args = p.parse_args()
    run(data_dir=args.data_dir, results_dir=args.results_dir, strategy=args.strategy, all_strategies=args.all_strategies)


if __name__ == "__main__":
    main()
