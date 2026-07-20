"""Generate KDE-based return features from 1-minute parquet files in duckdb.

Reads futures_YYYYMMDD_1min.parquet files, concatenates them in chronological
order, computes log-returns, and adds rolling KDE features.

Usage:
    python Devcenter/ml/generate_kde_features_parquet.py \
        --glob "Devcenter/data/duckdb/futures_*_1min.parquet" \
        --output Devcenter/ml/ml_data/futures_1min_with_kde.csv \
        --timeframes 1 5 \
        --window 3000 \
        --min-samples 500 \
        --refit-every 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indicators.return_distribution import add_return_features, build_kde_features


_logger = logging.getLogger(__name__)


def load_parquet_files(
    glob_pattern: str,
    *,
    timestamp_col: str = "timestamp",
    symbol: str = "FUTURES",
) -> pd.DataFrame:
    """Load and concatenate parquet files using DuckDB.

    Args:
        glob_pattern: Glob path to parquet files.
        timestamp_col: Name of the timestamp column.
        symbol: Symbol label to add.

    Returns:
        Sorted DataFrame with timestamp parsed.
    """
    _logger.info("Loading parquet files: %s", glob_pattern)
    con = duckdb.connect()

    # DuckDB can read a glob of parquet files directly.
    query = f"""
        SELECT *
        FROM read_parquet('{glob_pattern}')
        ORDER BY {timestamp_col}
    """
    df = con.execute(query).fetchdf()
    con.close()

    _logger.info("Loaded %d rows", len(df))

    # Parse timestamp: YYYYMMDD HHMM
    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col],
        format="%Y%m%d %H%M",
        errors="coerce",
    )
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)
    df["symbol"] = symbol
    _logger.info("After parsing/sorting: %d rows, %s ~ %s", len(df), df[timestamp_col].min(), df[timestamp_col].max())
    return df


def generate_kde_features_from_parquet(
    glob_pattern: str,
    output_path: Path,
    *,
    timeframes: Sequence[int] = (1, 5),
    window: int = 3000,
    min_samples: int = 500,
    bandwidth: str | float = "scott",
    decay: Optional[float] = None,
    grid_points: int = 2048,
    outlier_clip: float = 5.0,
    refit_every: int = 50,
    symbol: str = "FUTURES",
) -> None:
    """Load 1-minute parquet data and append KDE return features."""
    df = load_parquet_files(glob_pattern, symbol=symbol)

    # Add session date so overnight gaps do not create spurious 1-min returns.
    session_col = "session_date"
    df[session_col] = df["timestamp"].dt.strftime("%Y%m%d")

    # 1. Add log-returns within each session only (boundary rows become NaN)
    df = add_return_features(df, close_col="close", timeframes=timeframes, session_col=session_col)

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

    # 3. Diagnostics
    for tf in timeframes:
        ready_col = f"ret_log_{tf}m_kde_ready"
        if ready_col in df.columns:
            n_ready = int(df[ready_col].sum())
            _logger.info("%s ready samples: %d / %d", ready_col, n_ready, len(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    _logger.info("Saved %s (rows=%d, cols=%d)", output_path, len(df), len(df.columns))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KDE features from 1-minute parquet files."
    )
    parser.add_argument(
        "--glob",
        type=str,
        default=r"Devcenter\data\duckdb\futures_*_1min.parquet",
        help="Glob pattern for parquet files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Devcenter/ml/ml_data/futures_1min_with_kde.csv"),
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
    parser.add_argument("--decay", type=float, default=None, help="Optional exponential decay factor")
    parser.add_argument(
        "--grid-points",
        type=int,
        default=1024,
        help="CDF lookup grid points (lower=faster)",
    )
    parser.add_argument("--outlier-clip", type=float, default=5.0, help="Grid std clip factor")
    parser.add_argument(
        "--refit-every",
        type=int,
        default=100,
        help="Refit KDE every N rows for speed",
    )
    parser.add_argument("--symbol", type=str, default="FUTURES", help="Symbol label")
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

        generate_kde_features_from_parquet(
            glob_pattern=args.glob,
            output_path=args.output,
            timeframes=args.timeframes,
            window=args.window,
            min_samples=args.min_samples,
            bandwidth=bandwidth,
            decay=args.decay,
            grid_points=args.grid_points,
            outlier_clip=args.outlier_clip,
            refit_every=args.refit_every,
            symbol=args.symbol,
        )
        return 0
    except Exception as exc:
        _logger.exception("Failed to generate KDE features: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
