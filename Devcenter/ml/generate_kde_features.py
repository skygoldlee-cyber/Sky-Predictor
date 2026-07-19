"""Generate KDE-based return features for backtest datasets.

Adds rolling PDF/CDF/Z-score/tail-probability columns to an existing CSV.
Intended for Phase 1 prototyping of KDE mean-reversion signals.

Usage:
    python Devcenter/ml/generate_kde_features.py \
        --input Devcenter/ml/ml_data/ml_dataset.csv \
        --output Devcenter/ml/ml_data/ml_dataset_with_kde.csv \
        --timeframes 1 5 \
        --window 3000 \
        --min-samples 500

The script assumes a datetime/price column layout compatible with the
SkyPredictor ml_dataset.csv schema. If no close column is found, it falls
back to ``last``/``Close``/``close``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


# Allow running from repository root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indicators.return_distribution import (
    ReturnDistributionEstimator,
    add_return_features,
    build_kde_features,
)


_logger = logging.getLogger(__name__)


def find_close_column(df: pd.DataFrame) -> str:
    """Locate a close-price column case-insensitively."""
    candidates = ["close", "last", "Close", "LAST", "종가", "현재가"]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(
        f"No close-like column found in {list(df.columns)}"
    )


def generate_kde_features(
    input_path: Path,
    output_path: Path,
    timeframes: Sequence[int] = (1, 5),
    window: int = 3000,
    min_samples: int = 500,
    bandwidth: str | float = "scott",
    decay: Optional[float] = None,
    grid_points: int = 2048,
    outlier_clip: float = 5.0,
    refit_every: int = 1,
    chunksize: Optional[int] = None,
) -> None:
    """Load CSV, add returns and KDE features, save result."""
    _logger.info("Loading %s", input_path)
    df = pd.read_csv(input_path)

    close_col = find_close_column(df)
    _logger.info("Detected close column: %s", close_col)

    # Sort by time if a datetime column exists
    dt_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if dt_cols:
        df[dt_cols[0]] = pd.to_datetime(df[dt_cols[0]], errors="coerce")
        df = df.sort_values(dt_cols[0]).reset_index(drop=True)

    # 1. Add log-returns
    df = add_return_features(df, close_col=close_col, timeframes=timeframes)

    # 2. Add KDE features per timeframe
    for tf in timeframes:
        ret_col = f"ret_log_{tf}m"
        if ret_col not in df.columns:
            continue
        _logger.info("Building KDE features for %s", ret_col)
        df = build_kde_features(
            df,
            return_col=ret_col,
            window=window,
            min_samples=min_samples,
            bandwidth=bandwidth,
            decay=decay,
            grid_points=grid_points,
            outlier_clip=outlier_clip,
            refit_every=refit_every,
        )

    # 3. Basic diagnostics
    for tf in timeframes:
        ready_col = f"ret_log_{tf}m_kde_ready"
        if ready_col in df.columns:
            n_ready = int(df[ready_col].sum())
            _logger.info("%s ready samples: %d / %d", ready_col, n_ready, len(df))

    _logger.info("Saving %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    _logger.info("Done. Rows=%d Cols=%d", len(df), len(df.columns))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KDE return-distribution features for a CSV dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Devcenter/ml/ml_data/ml_dataset.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Devcenter/ml/ml_data/ml_dataset_with_kde.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--timeframes",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Return timeframes in minutes",
    )
    parser.add_argument("--window", type=int, default=3000, help="Rolling window size")
    parser.add_argument("--min-samples", type=int, default=500, help="Min samples before KDE ready")
    parser.add_argument(
        "--bandwidth",
        type=str,
        default="scott",
        help="KDE bandwidth: scott | silverman | float",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=None,
        help="Optional exponential decay factor (0,1)",
    )
    parser.add_argument("--grid-points", type=int, default=2048, help="CDF lookup grid points")
    parser.add_argument("--outlier-clip", type=float, default=5.0, help="Grid std clip factor")
    parser.add_argument(
        "--refit-every",
        type=int,
        default=1,
        help="Refit KDE every N rows for speed (1=exact, slow)",
    )
    parser.add_argument("--chunksize", type=int, default=None, help="Unused (reserved)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        bandwidth: str | float = args.bandwidth
        try:
            bandwidth = float(bandwidth)
        except ValueError:
            pass

        generate_kde_features(
            input_path=args.input,
            output_path=args.output,
            timeframes=args.timeframes,
            window=args.window,
            min_samples=args.min_samples,
            bandwidth=bandwidth,
            decay=args.decay,
            grid_points=args.grid_points,
            outlier_clip=args.outlier_clip,
            refit_every=args.refit_every,
        )
        return 0
    except Exception as exc:
        _logger.exception("Failed to generate KDE features: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
