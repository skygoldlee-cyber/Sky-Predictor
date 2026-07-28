"""Rebuild entry-side technical indicators from the original 5-minute bars.

Input: Devcenter/data/since2019_future_data.txt
Output: Devcenter/ml/ml_data/ml_dataset.csv (overwritten)

Recomputes ATR, RSI, MACD, Bollinger Bands, MAs, and SuperTrend from the raw
OHLCV bars and merges them into the existing trade-level dataset by entry_time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TXT_PATH = ROOT / "Devcenter" / "data" / "since2019_future_data.txt"
ML_PATH = ROOT / "Devcenter" / "ml" / "ml_data" / "ml_dataset.csv"


def parse_bars(path: Path) -> pd.DataFrame:
    cols = ["idx", "dt_str", "open", "high", "low", "close"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    df["timestamp"] = pd.to_datetime(df["dt_str"].str.replace("_", " "), format="%Y/%m/%d %H%M")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close"]]


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
    })


def compute_bollinger(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    middle = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
    })


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    atr = compute_atr(df, period=period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df))

    for i in range(1, len(df)):
        if df["close"].iloc[i] > supertrend[i - 1]:
            direction[i] = 1
            supertrend[i] = max(lower_band.iloc[i], supertrend[i - 1])
        else:
            direction[i] = -1
            supertrend[i] = min(upper_band.iloc[i], supertrend[i - 1])

    # Initialize first values
    supertrend[0] = lower_band.iloc[0] if df["close"].iloc[0] > upper_band.iloc[0] else upper_band.iloc[0]
    direction[0] = 1 if df["close"].iloc[0] > supertrend[0] else -1

    return pd.DataFrame({
        "supertrend": supertrend,
        "supertrend_dir": direction,
    }, index=df.index)


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["timestamp", "open", "high", "low", "close"]].copy()
    out["atr_14"] = compute_atr(df, period=14)
    out["rsi_14"] = compute_rsi(df["close"], period=14)
    out = out.join(compute_macd(df["close"]))
    out = out.join(compute_bollinger(df["close"]))
    out["ma5"] = df["close"].rolling(window=5, min_periods=5).mean()
    out["ma10"] = df["close"].rolling(window=10, min_periods=10).mean()
    out["ma20"] = df["close"].rolling(window=20, min_periods=20).mean()
    out["ma60"] = df["close"].rolling(window=60, min_periods=60).mean()
    out = out.join(compute_supertrend(df, period=10, multiplier=3.0))
    return out


def main():
    print("Parsing original 5-min bars...")
    bars = parse_bars(TXT_PATH)
    print(f"Bars: {len(bars)} rows, {bars['timestamp'].min()} ~ {bars['timestamp'].max()}")

    print("Computing indicators...")
    ind = build_indicators(bars)

    # Rename columns to match ml_dataset entry_* convention
    rename_map = {
        "timestamp": "entry_time",
        "open": "entry_open",
        "high": "entry_high",
        "low": "entry_low",
        "close": "entry_close_bar",
        "atr_14": "entry_atr",
        "rsi_14": "entry_rsi",
        "macd": "entry_macd",
        "macd_signal": "entry_macd_signal",
        "macd_hist": "entry_macd_hist",
        "bb_upper": "entry_bb_upper",
        "bb_middle": "entry_bb_middle",
        "bb_lower": "entry_bb_lower",
        "ma5": "entry_ma5",
        "ma10": "entry_ma10",
        "ma20": "entry_ma20",
        "ma60": "entry_ma60",
        "supertrend": "entry_supertrend",
        "supertrend_dir": "entry_supertrend_dir",
    }
    ind = ind.rename(columns=rename_map)

    print("Loading ml_dataset...")
    ml = pd.read_csv(ML_PATH)
    ml["entry_time"] = pd.to_datetime(ml["entry_time"])

    # Remove the old, degenerate indicator columns so the bar-computed versions
    # replace them cleanly after the merge.
    indicator_cols = list(rename_map.values())
    indicator_cols.remove("entry_time")
    drop_cols = [c for c in indicator_cols if c in ml.columns and c != "entry_time"]
    print(f"Dropping old degenerate columns from ml_dataset: {drop_cols}")
    ml = ml.drop(columns=drop_cols)

    print("Merging indicators...")
    merged = ml.merge(ind, on="entry_time", how="left")

    # Overwrite old indicator columns with bar-computed values
    overwrite_cols = [
        "entry_atr", "entry_rsi", "entry_macd", "entry_macd_signal", "entry_macd_hist",
        "entry_ma5", "entry_ma10", "entry_ma20", "entry_ma60",
        "entry_bb_upper", "entry_bb_middle", "entry_bb_lower",
        "entry_supertrend", "entry_supertrend_dir",
    ]
    for col in overwrite_cols:
        if col in merged.columns:
            print(f"  {col}: non-null {merged[col].notna().sum()} / {len(merged)}")
        else:
            print(f"  {col}: MISSING")

    # Sanity check: compare bar close to ml_dataset entry_close
    if "entry_close_bar" in merged.columns:
        diff = (merged["entry_close_bar"] - merged["entry_close"]).abs()
        print(f"  Bar close vs ml_dataset entry_close: max diff = {diff.max():.4f}")

    # Drop the helper close column
    if "entry_close_bar" in merged.columns:
        merged = merged.drop(columns=["entry_close_bar"])

    backup_path = ML_PATH.with_suffix(".csv.bak")
    print(f"Backing up original dataset to {backup_path}")
    ml.to_csv(backup_path, index=False)

    merged.to_csv(ML_PATH, index=False)
    print(f"Saved updated dataset to {ML_PATH}")


if __name__ == "__main__":
    main()
