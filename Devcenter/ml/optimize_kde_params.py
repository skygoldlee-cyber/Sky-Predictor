"""KDE parameter search for trade-filter XGBoost performance.

Uses a representative 3-month 1-minute sample to search over KDE window
and bandwidth choices. Hyperparameters are selected by validation-set
ROC AUC; the chosen config is evaluated **once** on the held-out test set
to avoid optimistic bias from multiple comparisons.

Usage:
    python Devcenter/ml/optimize_kde_params.py
"""

from __future__ import annotations

import itertools
import json
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
    # Session label for boundary-aware return calculation.
    df["session_date"] = df["timestamp"].dt.strftime("%Y%m%d")
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


def _coerce_bandwidth(bandwidth: str | float) -> str | float:
    """Return float for numeric strings, otherwise leave as scott/silverman."""
    if isinstance(bandwidth, str):
        try:
            return float(bandwidth)
        except ValueError:
            return bandwidth
    return bandwidth


def build_kde_for_params(
    df_1min: pd.DataFrame,
    window: int,
    bandwidth: str | float,
    min_samples: int = 500,
    grid_points: int = 1024,
    refit_every: int = 100,
) -> pd.DataFrame:
    """Add returns and KDE features for a parameter combo."""
    bw = _coerce_bandwidth(bandwidth)
    df = add_return_features(
        df_1min,
        close_col="close",
        timeframes=(1, 5),
        session_col="session_date",
    )
    for tf in (1, 5):
        df = build_kde_features(
            df,
            return_col=f"ret_log_{tf}m",
            window=window,
            min_samples=min_samples,
            bandwidth=bw,
            grid_points=grid_points,
            refit_every=refit_every,
        )
    return df


def merge_with_trades(kde_df: pd.DataFrame, trade_df: pd.DataFrame) -> pd.DataFrame:
    """Merge KDE features with trades on timestamp / entry_time.

    Uses merge_asof with direction='backward' so an entry at e.g. 10:23:17
    aligns with the most recent fully-closed 1-minute bar (10:23:00), avoiding
    exact-timestamp misses and preventing look-ahead leakage.
    """
    kde_cols = [c for c in kde_df.columns if c.endswith(("_kde_cdf", "_kde_pdf", "_kde_zscore", "_kde_left_tail", "_kde_right_tail", "_kde_ready"))]
    kde = kde_df[["timestamp"] + kde_cols].sort_values("timestamp").reset_index(drop=True)
    trades = trade_df.sort_values("entry_time").reset_index(drop=True)
    merged = pd.merge_asof(
        trades,
        kde,
        left_on="entry_time",
        right_on="timestamp",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.drop(columns=["timestamp"])
    merged = add_direction_aware_kde_features(merged)
    return merged


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("entry_time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
) -> xgb.XGBClassifier:
    X_train = train_df[feature_cols].fillna(0).astype(float)
    y_train = train_df["is_win"]
    X_val = val_df[feature_cols].fillna(0).astype(float)
    y_val = val_df["is_win"]

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric="logloss",
        reg_alpha=0.5,
        reg_lambda=2.0,
        early_stopping_rounds=20,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _evaluate_auc(df: pd.DataFrame, feature_cols: list) -> float:
    """Return test ROC AUC on a single held-out split."""
    train_df, val_df, test_df = _split(df)
    if len(train_df) < 50 or len(val_df) < 20 or len(test_df) < 20:
        return -1.0

    model = _train_model(train_df, val_df, feature_cols)
    X_test = test_df[feature_cols].fillna(0).astype(float)
    y_test = test_df["is_win"]
    if len(np.unique(y_test)) < 2:
        return -1.0
    y_proba = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, y_proba))


def _build_feature_cols(df: pd.DataFrame) -> list:
    baseline = [
        "entry_rsi", "entry_macd", "entry_macd_signal", "entry_macd_hist",
        "entry_atr", "entry_supertrend", "entry_supertrend_dir",
        "entry_ma20", "entry_ma60", "entry_bb_upper", "entry_bb_lower", "entry_bb_middle",
        "entry_hour", "entry_dayofweek", "entry_month",
        "volatility_regime", "trend_regime", "momentum_regime",
    ]
    kde_features = [c for c in df.columns if c.startswith(("ret_log_", "kde_"))]
    return [c for c in baseline if c in df.columns] + kde_features


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load a representative 3-month sample of 1-minute data.
    sample_start = "20251001"
    sample_end = "20251231"
    _logger.info("Loading 1-minute sample: %s ~ %s", sample_start, sample_end)
    df_1min = load_sample_1min(sample_start, sample_end)
    _logger.info("Loaded %d 1-minute rows", len(df_1min))

    # Load trades in the same period.
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
            feature_cols = _build_feature_cols(merged)
            merged_ready = merged[merged["ret_log_1m_kde_ready"].notna()].copy()

            # Train/val/test split: select by validation AUC only.
            train_df, val_df, test_df = _split(merged_ready)
            if len(train_df) < 50 or len(val_df) < 20 or len(test_df) < 20:
                _logger.info("  -> too few samples")
                results.append({
                    "window": window,
                    "bandwidth": str(bw),
                    "n_trades": len(merged_ready),
                    "val_roc_auc": -1.0,
                    "test_roc_auc": -1.0,
                })
                continue

            model = _train_model(train_df, val_df, feature_cols)
            val_proba = model.predict_proba(val_df[feature_cols].fillna(0).astype(float))[:, 1]
            val_auc = float(roc_auc_score(val_df["is_win"], val_proba))

            results.append({
                "window": window,
                "bandwidth": str(bw),
                "n_trades": len(merged_ready),
                "val_roc_auc": val_auc,
                "test_roc_auc": -1.0,  # filled only for the selected config
            })
            _logger.info("  -> val AUC=%.4f trades=%d", val_auc, len(merged_ready))
        except Exception as exc:
            _logger.warning("Failed window=%d bandwidth=%s: %s", window, bw, exc)
            results.append({
                "window": window,
                "bandwidth": str(bw),
                "n_trades": 0,
                "val_roc_auc": -1.0,
                "test_roc_auc": -1.0,
            })

    results_df = pd.DataFrame(results)
    if (results_df["val_roc_auc"] > 0).any():
        best_idx = results_df["val_roc_auc"].idxmax()
        best = results_df.loc[best_idx].copy()
    else:
        best = results_df.iloc[0].copy()

    # Evaluate the single selected config on the untouched test set.
    _logger.info("Selected config window=%s bandwidth=%s by val AUC=%.4f", best["window"], best["bandwidth"], best["val_roc_auc"])
    try:
        kde_df = build_kde_for_params(df_1min, window=int(best["window"]), bandwidth=_coerce_bandwidth(best["bandwidth"]))
        merged = merge_with_trades(kde_df, trades)
        feature_cols = _build_feature_cols(merged)
        test_auc = _evaluate_auc(merged[merged["ret_log_1m_kde_ready"].notna()].copy(), feature_cols)
        best["test_roc_auc"] = test_auc
        _logger.info("  -> test AUC=%.4f", test_auc)
    except Exception as exc:
        _logger.warning("Failed final test evaluation: %s", exc)
        best["test_roc_auc"] = -1.0

    print("\n=== KDE Parameter Search Results ===")
    print(results_df.sort_values("val_roc_auc", ascending=False).to_string(index=False))
    print(f"\nBest params (by validation AUC): window={best['window']}, bandwidth={best['bandwidth']}")
    print(f"Validation AUC={best['val_roc_auc']:.4f}, Test AUC={best['test_roc_auc']:.4f}")

    # Save best params
    params_path = MODELS_DIR / "kde_best_params.json"
    best.to_json(params_path)
    _logger.info("Saved best params to %s", params_path)

    # Also save a readable JSON summary
    summary_path = MODELS_DIR / "kde_optimization_summary.json"
    summary = {
        "best": best.to_dict(),
        "all_results": results_df.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    _logger.info("Saved optimization summary to %s", summary_path)


if __name__ == "__main__":
    main()
