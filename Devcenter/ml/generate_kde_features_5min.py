"""Generate 5-minute KDE features from the legacy 5m text data file.

The source file `Devcenter/data/since2019_future_data.txt` has columns:
    <date>_<time> open high low close
Example:
    2019/06/03_0900 263.15 263.45 262.20 263.40

This script parses it, computes 5-minute log-returns, adds rolling KDE
features, and saves a CSV that can be merged with `ml_dataset.csv` by entry_time.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indicators.return_distribution import add_return_features, build_kde_features


_logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("Devcenter/data/since2019_future_data.txt")
OUTPUT_DIR = Path("Devcenter/ml/ml_data")


def load_5min_txt(path: Path) -> pd.DataFrame:
    """Load 5-minute OHLC data from the project text file."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["timestamp_str", "open", "high", "low", "close"],
    )
    df["timestamp"] = pd.to_datetime(
        df["timestamp_str"],
        format="%Y/%m/%d_%H%M",
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"]).drop(columns=["timestamp_str"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_regime(
    df: pd.DataFrame,
    fast_span: int = 20,
    slow_span: int = 60,
    vol_span: int = 60,
    vol_quantile: float = 0.5,
) -> pd.DataFrame:
    """Add a trend+volatility regime column.

    Trend is defined by EMA20/EMA60:
        - bull: close > slow_ema and fast_ema > slow_ema
        - bear: close < slow_ema and fast_ema < slow_ema
        - neutral: otherwise

    Volatility is defined by close-to-close rolling standard deviation:
        - high: current vol >= rolling `vol_quantile` of recent `vol_span` bars
        - low:  current vol <  rolling `vol_quantile`

    Final regime is e.g. bull_high, bull_low, bear_high, bear_low, neutral_high, neutral_low.
    """
    df = df.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    returns = close.pct_change()
    fast = close.ewm(span=fast_span, adjust=False).mean()
    slow = close.ewm(span=slow_span, adjust=False).mean()
    vol = returns.rolling(window=vol_span, min_periods=vol_span // 2).std()
    vol_flag = np.where(vol >= vol.rolling(window=vol_span, min_periods=vol_span // 2).quantile(vol_quantile), "high", "low")

    trend = np.select(
        [(close > slow) & (fast > slow), (close < slow) & (fast < slow)],
        ["bull", "bear"],
        default="neutral",
    )
    df["regime"] = trend + "_" + vol_flag
    return df


def build_5min_kde_features(
    df: pd.DataFrame,
    window: int = 600,
    min_samples: int = 100,
    bandwidth: str | float = "scott",
    grid_points: int = 1024,
    refit_every: int = 20,
    use_regime: bool = False,
) -> pd.DataFrame:
    """Add 5-minute log-return and KDE features with session boundary masking."""
    df = df.copy()
    df["session_date"] = df["timestamp"].dt.strftime("%Y%m%d")
    df = add_regime(df)
    df = add_return_features(
        df,
        close_col="close",
        timeframes=(1,),
        session_col="session_date",
    )
    # rename to explicit 5m return column name for downstream consistency
    df = df.rename(columns={"ret_log_1m": "ret_log_5m"})
    df = build_kde_features(
        df,
        return_col="ret_log_5m",
        window=window,
        min_samples=min_samples,
        bandwidth=bandwidth,
        grid_points=grid_points,
        refit_every=refit_every,
        regime_col="regime" if use_regime else None,
    )
    df["feature_time"] = df["timestamp"] + pd.Timedelta(minutes=5)
    return df


def add_multi_timeframe_returns(
    df: pd.DataFrame,
    timeframes_minutes: Tuple[int, ...] = (5, 15, 30, 60),
    close_col: str = "close",
    session_col: str = "session_date",
) -> pd.DataFrame:
    """Add session-aware log-return columns for multiple 5m-bar timeframes."""
    df = df.copy()
    if session_col not in df.columns:
        df[session_col] = df["timestamp"].dt.strftime("%Y%m%d")
    closes = pd.to_numeric(df[close_col], errors="coerce")
    grouper = df[session_col]
    for tf in timeframes_minutes:
        shift_bars = max(1, tf // 5)
        col = f"ret_log_{tf}m"
        df[col] = np.log(closes / closes.groupby(grouper).shift(shift_bars))
    return df


def build_multi_tf_kde_features(
    df: pd.DataFrame,
    timeframes_minutes: Tuple[int, ...] = (5, 15, 30, 60),
    window: int = 1000,
    min_samples: int = 100,
    bandwidth: str | float = "scott",
    grid_points: int = 1024,
    refit_every: int = 50,
    use_regime: bool = False,
) -> pd.DataFrame:
    """Build KDE features for multiple 5m-bar return timeframes."""
    df = add_regime(df)
    df = add_multi_timeframe_returns(df, timeframes_minutes)
    for tf in timeframes_minutes:
        return_col = f"ret_log_{tf}m"
        if return_col not in df.columns:
            continue
        df = build_kde_features(
            df,
            return_col=return_col,
            window=window,
            min_samples=min_samples,
            bandwidth=bandwidth,
            grid_points=grid_points,
            refit_every=refit_every,
            prefix=f"{return_col}_kde",
            regime_col="regime" if use_regime else None,
        )
    df["feature_time"] = df["timestamp"] + pd.Timedelta(minutes=5)
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Generate 5m KDE features")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to 5m OHLC text file",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=600,
        help="Rolling window size in 5m bars (default 600 ~= ~1-2 weeks)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples required before fitting KDE",
    )
    parser.add_argument(
        "--bandwidth",
        type=str,
        default="scott",
        help="KDE bandwidth: 'scott', 'silverman', or float",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=1024,
        help="Number of grid points for CDF/PDF lookup",
    )
    parser.add_argument(
        "--refit-every",
        type=int,
        default=20,
        help="Recompute KDE every N bars for speed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "futures_5min_with_kde.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--use-regime",
        action="store_true",
        help="Build regime-specific (bull/bear/neutral) conditional KDE features",
    )
    args = parser.parse_args()

    bw = args.bandwidth
    try:
        bw = float(bw)
    except ValueError:
        pass

    _logger.info("Loading 5m data from %s", args.input)
    df = load_5min_txt(args.input)
    _logger.info("Loaded %d 5m rows, %s ~ %s", len(df), df["timestamp"].min(), df["timestamp"].max())

    _logger.info(
        "Building 5m KDE features: window=%d, min_samples=%d, bandwidth=%s, use_regime=%s",
        args.window, args.min_samples, bw, args.use_regime,
    )
    df = build_5min_kde_features(
        df,
        window=args.window,
        min_samples=args.min_samples,
        bandwidth=bw,
        grid_points=args.grid_points,
        refit_every=args.refit_every,
        use_regime=args.use_regime,
    )

    ready = df["ret_log_5m_kde_ready"].sum()
    _logger.info("KDE ready rows: %d / %d", int(ready), len(df))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    _logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
