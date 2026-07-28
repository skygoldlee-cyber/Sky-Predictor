"""Grid search over KDE timeframes, bandwidth, and window size.

Generates multiple KDE-augmented datasets and runs the enhanced pipeline on
each to find the best feature configuration.
"""

from __future__ import annotations

import itertools
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generate_kde_features import generate_kde_features
from ml_full_pipeline_enhanced import load_data, run_pipeline_enhanced

logging.basicConfig(level=logging.WARNING, format="%(message)s")
_logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "ml_data"
OUTPUT_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_PATH = DATA_DIR / "ml_dataset_close_renamed.csv"
KDE_TARGET = DATA_DIR / "ml_dataset_with_kde.csv"


def run_grid():
    timeframes_grid = [
        (1, 5),
        (1, 5, 15),
        (1, 5, 15, 60),
        (5, 15),
        (5, 15, 60),
    ]
    bandwidth_grid = ["scott", 0.3, 0.5]
    window_grid = [3000, 2000]

    rows = []
    combinations = list(itertools.product(timeframes_grid, bandwidth_grid, window_grid))
    _logger.warning("Running %d KDE feature combinations", len(combinations))

    for i, (tfs, bw, window) in enumerate(combinations, 1):
        _logger.warning("\n[%d/%d] tfs=%s bw=%s window=%d", i, len(combinations), tfs, bw, window)
        temp_output = DATA_DIR / f"ml_dataset_with_kde_grid_{i}.csv"
        try:
            generate_kde_features(
                input_path=INPUT_PATH,
                output_path=temp_output,
                timeframes=tfs,
                window=window,
                min_samples=500,
                bandwidth=bw,
                refit_every=50,
            )
            shutil.copy(temp_output, KDE_TARGET)
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
                "n_cols": df.shape[1],
            })
            rows.append(final)
            _logger.warning(
                "Result: trades=%d PF=%.2f Sharpe=%.2f MAR=%.2f PnL=%.0f",
                final["n_trades"], final["profit_factor"], final["sharpe"], final["mar"], final["total_pnl"],
            )
        except Exception as exc:
            _logger.error("Failed for tfs=%s bw=%s window=%d: %s", tfs, bw, window, exc)

    result_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "kde_grid_results.csv"
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
