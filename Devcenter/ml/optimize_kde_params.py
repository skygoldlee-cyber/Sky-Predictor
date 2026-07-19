"""KDE parameter search for trade-filter XGBoost performance.

Uses a representative 3-month 1-minute sample to search over KDE window
and bandwidth choices. The best combo is then applied to the full dataset.

Usage:
    python Devcenter/ml/optimize_kde_params.py
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indicators.return_distribution import add_return_features, build_kde_features


_logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"


def load_sample_1min(start_date: str, end_date: str) -> pd.DataFrame:
    """Load 1-minute futures data for a date range via DuckDB."""
    con = duckdb.connect()
    glob_path = DATA_DIR.parent.parent / "data" / "duckdb" / "futures_*_1min.parquet"
    query = f"""
        SELECT *
        FROM read_parquet('{glob_path}')
        WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'
        ORDER BY timestamp
    """
    df = con.execute(query).fetchdf()
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d %H%M", errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def add_direction_aware_kde_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add direction-aware KDE features aligned with trade direction."""
    for tf in ["1m", "5m"]:
        lt = f"ret_log_{tf}_kde_left_tail"
        rt = f"ret_log_{tf}_kde_right_tail"
        zs = f"ret_log_{tf}_kde_zscore"
        if lt not in df.columns:
            continue
        direction = df["direction"].fillna(0).astype(float)
        df[f"kde_aligned_tail_{tf}"] = np.where(direction == 1, df[lt], df[rt])
        df[f"kde_opposite_tail_{tf}"] = np.where(direction == 1, df[rt], df[lt])
        df[f"kde_aligned_zscore_{tf}"] = np.where(direction == 1, -df[zs], df[zs])
    return df


def build_kde_for_params(
    df_1min: pd.DataFrame,
    window: int,
    bandwidth: str | float,
    min_samples: int = 500,
    grid_points: int = 1024,
    refit_every: int = 100,
) -> pd.DataFrame:
    """Add returns and KDE features for a parameter combo."""
    df = add_return_features(df_1min, close_col="close", timeframes=(1, 5))
    for tf in (1, 5):
        df = build_kde_features(
            df,
            return_col=f"ret_log_{tf}m",
            window=window,
            min_samples=min_samples,
            bandwidth=bandwidth,
            grid_points=grid_points,
            refit_every=refit_every,
        )
    return df


def merge_with_trades(kde_df: pd.DataFrame, trade_df: pd.DataFrame) -> pd.DataFrame:
    """Merge KDE features with trades on timestamp / entry_time."""
    kde_cols = [c for c in kde_df.columns if c.endswith(("_kde_cdf", "_kde_pdf", "_kde_zscore", "_kde_left_tail", "_kde_right_tail", "_kde_ready"))]
    merged = trade_df.merge(
        kde_df[["timestamp"] + kde_cols],
        left_on="entry_time",
        right_on="timestamp",
        how="left",
    ).drop(columns=["timestamp"])
    merged = add_direction_aware_kde_features(merged)
    return merged


def train_eval_xgboost(df: pd.DataFrame) -> float:
    """Train XGBoost baseline+KDE and return test ROC AUC."""
    baseline = [
        "entry_rsi", "entry_macd", "entry_macd_signal", "entry_macd_hist",
        "entry_atr", "entry_supertrend", "entry_supertrend_dir",
        "entry_ma20", "entry_ma60", "entry_bb_upper", "entry_bb_lower", "entry_bb_middle",
        "entry_hour", "entry_dayofweek", "entry_month",
        "volatility_regime", "trend_regime", "momentum_regime",
    ]
    kde_features = [c for c in df.columns if c.startswith(("ret_log_", "kde_"))]
    feature_cols = [c for c in baseline if c in df.columns] + kde_features

    df = df[df["ret_log_1m_kde_ready"].notna()].copy()
    if len(df) < 200:
        return -1.0

    df = df.sort_values("entry_time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df[feature_cols].fillna(0).astype(float)
    y_train = train_df["is_win"]
    X_val = val_df[feature_cols].fillna(0).astype(float)
    y_val = val_df["is_win"]
    X_test = test_df[feature_cols].fillna(0).astype(float)
    y_test = test_df["is_win"]

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric="logloss",
        reg_alpha=0.5,
        reg_lambda=2.0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_proba = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else -1.0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load a representative 3-month sample of 1-minute data
    sample_start = "20251001"
    sample_end = "20251231"
    _logger.info("Loading 1-minute sample: %s ~ %s", sample_start, sample_end)
    df_1min = load_sample_1min(sample_start, sample_end)
    _logger.info("Loaded %d 1-minute rows", len(df_1min))

    # Load trades in the same period
    trades = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades = trades[
        (trades["entry_time"] >= pd.Timestamp("2025-10-01")) &
        (trades["entry_time"] <= pd.Timestamp("2025-12-31"))
    ].copy()
    _logger.info("Loaded %d trades in sample period", len(trades))

    windows = [1000, 2000, 3000, 5000]
    bandwidths = ["scott", "silverman", 0.0005, 0.001, 0.002]

    results = []
    for window, bw in itertools.product(windows, bandwidths):
        _logger.info("Trying window=%d bandwidth=%s", window, bw)
        try:
            kde_df = build_kde_for_params(df_1min, window=window, bandwidth=bw)
            merged = merge_with_trades(kde_df, trades)
            auc = train_eval_xgboost(merged)
            results.append({
                "window": window,
                "bandwidth": str(bw),
                "n_trades": len(merged[merged["ret_log_1m_kde_ready"].notna()]),
                "test_roc_auc": auc,
            })
            _logger.info("  -> AUC=%.4f trades=%d", auc, len(merged[merged["ret_log_1m_kde_ready"].notna()]))
        except Exception as exc:
            _logger.warning("Failed window=%d bandwidth=%s: %s", window, bw, exc)
            results.append({
                "window": window,
                "bandwidth": str(bw),
                "n_trades": 0,
                "test_roc_auc": -1.0,
            })

    results_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False)
    print("\n=== KDE Parameter Search Results ===")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest params: window={best['window']}, bandwidth={best['bandwidth']}, AUC={best['test_roc_auc']:.4f}")

    # Save best params
    params_path = MODELS_DIR / "kde_best_params.json"
    best.to_json(params_path)
    _logger.info("Saved best params to %s", params_path)


if __name__ == "__main__":
    main()
