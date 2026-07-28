"""Grid search over stage1/2/3 metrics and stage3 min_trades.

Runs ``ml_full_pipeline_enhanced.run_pipeline_enhanced`` over a Cartesian
product of candidate metrics and min-trade thresholds.  Saves the resulting
trading metrics to CSV for offline analysis.

Example:
    python Devcenter/ml/grid_search_thresholds.py \
        --slippage-ticks 1 --use-kde --no-regime-aware \
        --stage3-type xgb_reg --output Devcenter/ml/ml_models/threshold_grid.csv
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_full_pipeline_enhanced import load_data, run_pipeline_enhanced

logging.basicConfig(level=logging.WARNING, format="%(message)s")
_logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage metric/threshold grid search")
    parser.add_argument("--slippage-ticks", type=int, default=1)
    parser.add_argument("--use-kde", action="store_true", default=True)
    parser.add_argument("--no-use-kde", dest="use_kde", action="store_false")
    parser.add_argument("--regime-aware", action="store_true", default=False)
    parser.add_argument("--stage3-type", type=str, default="xgb_reg")
    parser.add_argument(
        "--stage1-metrics",
        nargs="+",
        default=["pnl", "sharpe_x_pf"],
        help="Stage 1 threshold-selection metrics to try",
    )
    parser.add_argument(
        "--stage2-metrics",
        nargs="+",
        default=["pnl", "sharpe_x_pf"],
        help="Stage 2 threshold-selection metrics to try",
    )
    parser.add_argument(
        "--stage3-metrics",
        nargs="+",
        default=["sharpe", "pf"],
        help="Stage 3 threshold-selection metrics to try",
    )
    parser.add_argument(
        "--stage3-min-trades",
        nargs="+",
        type=int,
        default=[30, 50, 75, 100],
        help="Stage 3 minimum-trade thresholds to try",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Devcenter/ml/ml_models/threshold_grid.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data(slippage_ticks=args.slippage_ticks)
    _logger.warning("Loaded %d trades (slippage=%d tick)", len(df), args.slippage_ticks)

    rows = []
    combinations = list(itertools.product(
        args.stage1_metrics,
        args.stage2_metrics,
        args.stage3_metrics,
        args.stage3_min_trades,
    ))
    _logger.warning("Running %d combinations", len(combinations))

    for i, (s1_metric, s2_metric, s3_metric, s3_min) in enumerate(combinations, 1):
        _logger.warning("[%d/%d] s1=%s s2=%s s3=%s min_trades=%d", i, len(combinations), s1_metric, s2_metric, s3_metric, s3_min)
        try:
            _, info = run_pipeline_enhanced(
                df,
                use_kde=args.use_kde,
                variant="KDE-enhanced" if args.use_kde else "Baseline",
                train_years=(2019, 2020, 2021, 2022, 2023),
                val_year=2024,
                test_year=(2025, 2026),
                stage1_metric=s1_metric,
                stage2_metric=s2_metric,
                stage3_metric=s3_metric,
                stage3_type=args.stage3_type,
                stage3_min_trades=s3_min,
                regime_aware=args.regime_aware,
                meta_filter=False,
                ensemble_exit=False,
                trailing_stop=False,
                fractional_kelly=False,
                drawdown_guard=False,
            )
            final = info["final"].copy()
            final.update({
                "stage1_metric": s1_metric,
                "stage2_metric": s2_metric,
                "stage3_metric": s3_metric,
                "stage3_min_trades": s3_min,
                "stage1_threshold": info["stage1_metrics"].get("threshold"),
                "stage2_threshold": info["stage2_metrics"].get("threshold"),
                "stage3_threshold": info["stage3_metrics"].get("threshold"),
                "stage3_val_score": info["stage3_metrics"].get("val_score"),
            })
            rows.append(final)
        except Exception as exc:
            _logger.error("Failed for combination %s: %s", (s1_metric, s2_metric, s3_metric, s3_min), exc)

    result_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    _logger.warning("Saved %d rows to %s", len(result_df), args.output)

    if not result_df.empty:
        # Sort by a combined score: Sharpe then MAR then PF
        result_df["_score"] = (
            result_df["sharpe"].fillna(-999)
            + result_df["mar"].fillna(-999)
            + (result_df["profit_factor"].fillna(0) - 1.0).clip(lower=0)
        )
        top = result_df.sort_values("_score", ascending=False).head(10)
        _logger.warning("\nTop 10 by combined score (Sharpe + MAR + excess PF):")
        print(top[[
            "stage1_metric", "stage2_metric", "stage3_metric", "stage3_min_trades",
            "n_trades", "win_rate", "total_pnl", "profit_factor", "sharpe", "cagr", "mdd", "mar",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
