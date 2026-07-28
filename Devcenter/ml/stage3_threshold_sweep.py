"""Sweep stage3 fixed xgb_reg exit thresholds.

Skips stage1/stage2 (pass-through) and tests a range of stage3 regression
thresholds to find the PF/Sharpe/MAR sweet spot.
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
    parser.add_argument("--slippage-ticks", type=int, default=1)
    parser.add_argument("--stage3-type", type=str, default="xgb_reg")
    parser.add_argument("--stage3-metric", type=str, default="pf")
    parser.add_argument("--stage1-metric", type=str, default="pnl")
    parser.add_argument("--stage2-metric", type=str, default="pnl")
    parser.add_argument("--skip-stage1-stage2", action="store_true", help="Pass stage1/stage2 through (threshold=0)")
    parser.add_argument("--feature-selection-k", type=int, default=0)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[2000, 4000, 6000, 8000, 10000, 12000, 14000])
    parser.add_argument("--output", type=Path, default=Path("Devcenter/ml/ml_models/stage3_threshold_sweep.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data(slippage_ticks=args.slippage_ticks)
    _logger.warning("Loaded %d trades (slippage=%d tick)", len(df), args.slippage_ticks)

    rows = []
    for i, thr in enumerate(args.thresholds, 1):
        _logger.warning("[%d/%d] stage3 threshold = %s", i, len(args.thresholds), thr)
        try:
            _, info = run_pipeline_enhanced(
                df,
                use_kde=True,
                variant="KDE-enhanced",
                train_years=(2019, 2020, 2021, 2022, 2023),
                val_year=2024,
                test_year=(2025, 2026),
                stage1_metric=args.stage1_metric,
                stage2_metric=args.stage2_metric,
                stage3_metric=args.stage3_metric,
                stage1_threshold=0.0 if args.skip_stage1_stage2 else None,
                stage2_threshold=0.0 if args.skip_stage1_stage2 else None,
                stage3_threshold=float(thr),
                stage3_type=args.stage3_type,
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
            final["stage3_threshold"] = thr
            rows.append(final)
        except Exception as exc:
            _logger.error("Failed for threshold %s: %s", thr, exc)

    result_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    _logger.warning("Saved %d rows to %s", len(result_df), args.output)

    if not result_df.empty:
        print("\nAll results:")
        print(result_df[[
            "stage3_threshold", "n_trades", "win_rate", "total_pnl",
            "profit_factor", "sharpe", "cagr", "mdd", "mar",
        ]].to_string(index=False))

        print("\nTop 5 by Sharpe:")
        print(result_df.nlargest(5, "sharpe")[[
            "stage3_threshold", "n_trades", "win_rate", "total_pnl",
            "profit_factor", "sharpe", "cagr", "mdd", "mar",
        ]].to_string(index=False))

        print("\nTop 5 by PF:")
        print(result_df.nlargest(5, "profit_factor")[[
            "stage3_threshold", "n_trades", "win_rate", "total_pnl",
            "profit_factor", "sharpe", "cagr", "mdd", "mar",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
