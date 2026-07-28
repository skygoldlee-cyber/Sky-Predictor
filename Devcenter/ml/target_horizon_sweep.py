"""Sweep fixed-horizon target labels and summarize final test metrics.

Runs the enhanced pipeline for a set of target horizons (signed directional
returns h bars after entry) using the same pipeline options, then prints a
compact table of final-test PnL, PF, Sharpe, and MAR.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_full_pipeline_enhanced import load_data, run_pipeline_enhanced

logging.basicConfig(level=logging.WARNING, format="%(message)s")
_logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 4, 5, 6, 7, 10, 20])
    parser.add_argument("--slippage-ticks", type=int, default=1)
    parser.add_argument("--feature-selection-k", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("Devcenter/ml/ml_models/target_horizon_sweep.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []

    for h in args.horizons:
        _logger.warning("\n=== target-horizon %d ===", h)
        df = load_data(slippage_ticks=args.slippage_ticks, target_horizon=h)
        for use_kde, variant in [(False, "Baseline"), (True, "KDE-enhanced")]:
            try:
                _, info = run_pipeline_enhanced(
                    df,
                    use_kde=use_kde,
                    variant=variant,
                    train_years=(2019, 2020, 2021, 2022, 2023),
                    val_year=2024,
                    test_year=(2025, 2026),
                    stage1_metric="pnl",
                    stage2_metric="pnl",
                    stage3_metric="sharpe",
                    stage1_threshold=None,
                    stage2_threshold=None,
                    stage3_threshold=None,
                    stage3_type="xgb_reg",
                    stage3_min_trades=30,
                    regime_aware=False,
                    meta_filter=False,
                    ensemble_exit=False,
                    trailing_stop=False,
                    fractional_kelly=False,
                    drawdown_guard=False,
                    feature_selection_k=args.feature_selection_k,
                )
                final = info["final"].copy()
                final["horizon"] = h
                final["variant"] = variant
                rows.append(final)
            except Exception as exc:
                _logger.error("Failed for horizon %s %s: %s", h, variant, exc)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        _logger.warning("No successful runs")
        return

    result_df = result_df[[
        "horizon", "variant", "n_trades", "win_rate", "total_pnl",
        "profit_factor", "sharpe", "cagr", "mdd", "mar",
    ]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    _logger.warning("Saved %d rows to %s", len(result_df), args.output)
    print("\n" + result_df.to_string(index=False))

    print("\nTop 5 by Sharpe:")
    print(result_df.nlargest(5, "sharpe").to_string(index=False))

    print("\nTop 5 by PF:")
    print(result_df.nlargest(5, "profit_factor").to_string(index=False))


if __name__ == "__main__":
    main()
