"""Add fixed-horizon return labels to the trade-level dataset.

Uses the original 5-minute bars to compute the return N bars after each
trade's entry_time, signed by trade direction. These labels do not depend on
the strategy's own exit decisions, so they are less prone to look-ahead leakage
when training entry/exit models.

Input: Devcenter/data/since2019_future_data.txt
       Devcenter/ml/ml_data/ml_dataset.csv
Output: Devcenter/ml/ml_data/ml_dataset.csv (overwritten)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TXT_PATH = ROOT / "Devcenter" / "data" / "since2019_future_data.txt"
ML_PATH = ROOT / "Devcenter" / "ml" / "ml_data" / "ml_dataset.csv"

HORIZONS = [1, 3, 4, 5, 6, 7, 10, 20]


def parse_bars(path: Path) -> pd.DataFrame:
    cols = ["idx", "dt_str", "open", "high", "low", "close"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    df["timestamp"] = pd.to_datetime(df["dt_str"].str.replace("_", " "), format="%Y/%m/%d %H%M")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "close"]]


def main():
    print("Parsing bars...")
    bars = parse_bars(TXT_PATH)
    bars = bars.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    ts_to_idx = {t: i for i, t in enumerate(bars["timestamp"])}

    print("Loading ml_dataset...")
    ml = pd.read_csv(ML_PATH)
    ml["entry_time"] = pd.to_datetime(ml["entry_time"])

    for h in HORIZONS:
        print(f"Computing horizon {h}...")
        future_close = []
        for _, row in ml.iterrows():
            idx = ts_to_idx.get(row["entry_time"])
            if idx is None or idx + h >= len(bars):
                future_close.append(np.nan)
            else:
                future_close.append(bars["close"].iloc[idx + h])
        ml[f"future_close_{h}"] = future_close
        ml[f"future_return_{h}"] = ml["direction"] * (ml[f"future_close_{h}"] - ml["entry_px"]) / ml["entry_px"]
        ml[f"is_win_h{h}"] = (ml[f"future_return_{h}"] > 0).astype(int)

    ml = ml.drop(columns=[f"future_close_{h}" for h in HORIZONS])

    backup_path = ML_PATH.with_suffix(".csv.bak2")
    print(f"Backing up to {backup_path}")
    ml.to_csv(backup_path, index=False)

    ml.to_csv(ML_PATH, index=False)
    print(f"Saved updated dataset to {ML_PATH}")

    for h in HORIZONS:
        col = f"is_win_h{h}"
        if col in ml.columns:
            print(f"  {col}: mean={ml[col].mean():.3f}, non-null={ml[col].notna().sum()}")


if __name__ == "__main__":
    main()
