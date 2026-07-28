"""Grid search KDE configurations using the original 5-minute bar data.

Parses Devcenter/data/since2019_future_data.txt, builds KDE features for each
(timeframes, bandwidth, window) combination, merges with ml_dataset.csv, and
runs the enhanced pipeline.
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generate_kde_features import generate_kde_features
from ml_full_pipeline_enhanced import load_data, run_pipeline_enhanced

logging.basicConfig(level=logging.WARNING, format="%(message)s")
_logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TXT_PATH = ROOT / "Devcenter" / "data" / "since2019_future_data.txt"
BARS_CSV = ROOT / "Devcenter" / "ml" / "ml_data" / "bars_5min.csv"
ML_DATASET = ROOT / "Devcenter" / "ml" / "ml_data" / "ml_dataset.csv"
OUTPUT_DIR = ROOT / "Devcenter" / "ml" / "ml_models"


def parse_bars() -> pd.DataFrame:
    cols = ["idx", "dt_str", "open", "high", "low", "close"]
    df = pd.read_csv(TXT_PATH, sep=r"\s+", header=None, names=cols)
    df["timestamp"] = pd.to_datetime(df["dt_str"].str.replace("_", " "), format="%Y/%m/%d %H%M")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close"]]


def run_grid():
    timeframes_grid = [
        (1, 5),
        (1, 5, 15),
        (5, 15),
        (1, 5, 15, 60),
    ]
    bandwidth_grid = ["scott"]
    window_grid = [2000, 3000]

    bars = parse_bars()
    bars[["timestamp", "close"]].to_csv(BARS_CSV, index=False)

    ml = pd.read_csv(ML_DATASET)
    ml["entry_time"] = pd.to_datetime(ml["entry_time"])

    rows = []
    combinations = list(itertools.product(timeframes_grid, bandwidth_grid, window_grid))
    _logger.warning("Running %d KDE feature combinations on original 5-min bars", len(combinations))

    for i, (tfs, bw, window) in enumerate(combinations, 1):
        _logger.warning("\n[%d/%d] tfs=%s bw=%s window=%d", i, len(combinations), tfs, bw, window)
        kde_csv = ROOT / "Devcenter" / "ml" / "ml_data" / f"bars_5min_kde_{i}.csv"
        try:
            generate_kde_features(
                input_path=BARS_CSV,
                output_path=kde_csv,
                timeframes=tfs,
                window=window,
                min_samples=500,
                bandwidth=bw,
                refit_every=500,
            )
            kde_bars = pd.read_csv(kde_csv)
            kde_bars["timestamp"] = pd.to_datetime(kde_bars["timestamp"])
            kde_cols = [c for c in kde_bars.columns if "_kde_" in c or c.startswith("ret_log_")]
            kde_bars = kde_bars[["timestamp"] + kde_cols]

            merged = ml.merge(kde_bars, left_on="entry_time", right_on="timestamp", how="left")
            if merged[kde_cols[0]].isna().mean() > 0.01:
                _logger.error("Low merge rate %.2f, skipping", merged[kde_cols[0]].notna().mean())
                continue

            # Temporarily overwrite the dataset used by load_data()
            target = ROOT / "Devcenter" / "ml" / "ml_data" / "ml_dataset_with_kde.csv"
            merged.to_csv(target, index=False)

            df = load_data(slippage_ticks=1)
            _, info = run_pipeline_enhanced(
                df,
                use_kde=True,
                variant="KDE-enhanced",
                train_years=(2019, 2020, 2021, 2022, 2023),
                val_year=2024,
                test_year=(2025, 2026),
                stage1_metric="pnl",
                stage2_metric="pnl",
                stage3_metric="sharpe",
                stage3_type="xgb_reg",
                stage3_min_trades=30,
                regime_aware=False,
                meta_filter=False,
                ensemble_exit=False,
                trailing_stop=False,
                fractional_kelly=False,
                drawdown_guard=False,
            )
            final = info["final"].copy()
            final.update({
                "timeframes": str(tfs),
                "bandwidth": str(bw),
                "window": window,
            })
            rows.append(final)
            _logger.warning(
                "Result: trades=%d PF=%.2f Sharpe=%.2f MAR=%.2f PnL=%.0f",
                final["n_trades"], final["profit_factor"], final["sharpe"], final["mar"], final["total_pnl"],
            )
        except Exception as exc:
            _logger.error("Failed for tfs=%s bw=%s window=%d: %s", tfs, bw, window, exc)

    result_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "kde_grid_results_v2_original_bars.csv"
    result_df.to_csv(out_path, index=False)
    _logger.warning("\nSaved %d rows to %s", len(result_df), out_path)

    if not result_df.empty:
        print("\nTop 10 by Sharpe:")
        print(result_df.nlargest(10, "sharpe")[[
            "timeframes", "bandwidth", "window", "n_trades",
            "profit_factor", "sharpe", "mar", "total_pnl",
        ]].to_string(index=False))

        print("\nTop 10 by PF:")
        print(result_df.nlargest(10, "profit_factor")[[
            "timeframes", "bandwidth", "window", "n_trades",
            "profit_factor", "sharpe", "mar", "total_pnl",
        ]].to_string(index=False))


if __name__ == "__main__":
    run_grid()
