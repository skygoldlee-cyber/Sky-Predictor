"""Build KDE features from the original 5-minute OHLCV bars and merge them
into ml_dataset.csv by entry_time.

Input: Devcenter/data/since2019_future_data.txt
Output: Devcenter/ml/ml_data/ml_dataset_with_kde_v2.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generate_kde_features import generate_kde_features

ROOT = Path(__file__).resolve().parents[2]
TXT_PATH = ROOT / "Devcenter" / "data" / "since2019_future_data.txt"
BARS_CSV = ROOT / "Devcenter" / "ml" / "ml_data" / "bars_5min.csv"
BARS_KDE_CSV = ROOT / "Devcenter" / "ml" / "ml_data" / "bars_5min_with_kde.csv"
ML_DATASET = ROOT / "Devcenter" / "ml" / "ml_data" / "ml_dataset.csv"
OUTPUT = ROOT / "Devcenter" / "ml" / "ml_data" / "ml_dataset_with_kde_v2.csv"


def parse_bars() -> pd.DataFrame:
    cols = ["idx", "dt_str", "open", "high", "low", "close"]
    df = pd.read_csv(TXT_PATH, sep=r"\s+", header=None, names=cols)
    # Convert 2019/06/03_0900 -> datetime
    df["timestamp"] = pd.to_datetime(
        df["dt_str"].str.replace("_", " "), format="%Y/%m/%d %H%M"
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close"]]


def main():
    print("Parsing original 5-min bars...")
    bars = parse_bars()
    print(f"Bars: {len(bars)} rows, {bars['timestamp'].min()} ~ {bars['timestamp'].max()}")
    bars[["timestamp", "close"]].to_csv(BARS_CSV, index=False)

    print("Generating KDE features on 5-min bars...")
    generate_kde_features(
        input_path=BARS_CSV,
        output_path=BARS_KDE_CSV,
        timeframes=(1, 5),
        window=2000,
        min_samples=500,
        bandwidth="scott",
        refit_every=500,
    )

    kde_bars = pd.read_csv(BARS_KDE_CSV)
    kde_bars["timestamp"] = pd.to_datetime(kde_bars["timestamp"])
    kde_cols = [c for c in kde_bars.columns if "_kde_" in c or c.startswith("ret_log_")]
    kde_bars = kde_bars[["timestamp"] + kde_cols]

    print(f"KDE columns: {kde_cols}")

    ml = pd.read_csv(ML_DATASET)
    ml["entry_time"] = pd.to_datetime(ml["entry_time"])

    merged = ml.merge(kde_bars, left_on="entry_time", right_on="timestamp", how="left")
    match_rate = merged[kde_cols[0]].notna().mean()
    print(f"Match rate with ml_dataset: {match_rate:.2%}")

    merged.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT} ({merged.shape})")


if __name__ == "__main__":
    main()
