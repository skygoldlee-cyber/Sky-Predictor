"""End-to-end 5m ML pipeline with optional KDE features.

Re-implements the three-stage pipeline (XGBoost filter -> RF entry timing ->
LSTM exit timing) on `ml_dataset_with_kde.csv`.  It runs a baseline variant
(no KDE) and a KDE-enhanced variant and compares final test-set PnL.

Time split:
    train: 2019-2023
    val:   2024
    test:  2025-2026
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Tuple

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
random.seed(42)

import joblib
import numpy as np
np.random.seed(42)
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import tensorflow as tf
    tf.random.set_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    from tensorflow import keras
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization, Bidirectional,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
except Exception as exc:  # pragma: no cover
    tf = keras = None
    LSTM = Dense = Dropout = BatchNormalization = Bidirectional = None
    logging.warning("TensorFlow not available: %s", exc)


_logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "ml_data"
OUTPUT_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_FEATURES = [
    "entry_rsi", "entry_macd", "entry_macd_signal", "entry_macd_hist",
    "entry_atr", "entry_supertrend", "entry_supertrend_dir",
    "entry_ma5", "entry_ma10", "entry_ma20", "entry_ma60",
    "entry_bb_upper", "entry_bb_lower", "entry_bb_middle",
    "entry_hour", "entry_dayofweek", "entry_month",
    "atr_normalized", "volatility_regime", "trend_regime", "momentum_regime",
]

# KOSPI200 futures assumption for cost adjustment
CONTRACT_MULTIPLIER = 31500.0  # KRW per point per contract
TICK_SIZE = 0.05                 # index point tick
SLIPPAGE_TICKS = 1               # ticks per side

KDE_FEATURES = [
    "ret_log_5m_kde_cdf", "ret_log_5m_kde_pdf", "ret_log_5m_kde_zscore",
    "ret_log_5m_kde_left_tail", "ret_log_5m_kde_right_tail",
    "kde_aligned_tail_5m", "kde_opposite_tail_5m", "kde_aligned_zscore_5m",
]


def apply_trading_costs(df: pd.DataFrame, slippage_ticks: int = SLIPPAGE_TICKS) -> pd.DataFrame:
    """Adjust net PnL by adding round-trip slippage and recompute is_win.

    `net_krw` already contains commission from the backtest; this function
    subtracts an additional slippage cost so reported PnL is closer to live
    execution.
    """
    df = df.copy()
    round_trip_slippage_pts = 2.0 * TICK_SIZE * slippage_ticks
    size = df["size_factor"].replace(0, np.nan).fillna(1.0)
    slippage_krw = round_trip_slippage_pts * CONTRACT_MULTIPLIER * size
    df["net_krw"] = df["net_krw"] - slippage_krw
    df["net_pts"] = df["net_krw"] / (CONTRACT_MULTIPLIER * size)
    df["is_win"] = (df["net_krw"] > 0).astype(int)
    return df


def load_data(slippage_ticks: int = SLIPPAGE_TICKS) -> pd.DataFrame:
    path = DATA_DIR / "ml_dataset_with_kde.csv"
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["year"] = df["entry_time"].dt.year
    df = apply_trading_costs(df, slippage_ticks=slippage_ticks)
    df = add_derived_features(df)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add entry-timing derived features used by the RF stage."""
    df = df.copy()
    df["rsi_oversold"] = (df["entry_rsi"] < 30).astype(int)
    df["rsi_overbought"] = (df["entry_rsi"] > 70).astype(int)
    df["macd_bullish"] = (df["entry_macd"] > df["entry_macd_signal"]).astype(int)
    df["macd_strength"] = (df["entry_macd"] - df["entry_macd_signal"]).abs()
    df["price_above_ma20"] = (df["entry_px"] > df["entry_ma20"]).astype(int)
    df["price_above_ma60"] = (df["entry_px"] > df["entry_ma60"]).astype(int)
    df["price_above_ma5"] = (df["entry_px"] > df["entry_ma5"]).astype(int)
    df["price_above_ma10"] = (df["entry_px"] > df["entry_ma10"]).astype(int)
    bb_range = df["entry_bb_upper"] - df["entry_bb_lower"]
    df["bb_position"] = (df["entry_px"] - df["entry_bb_lower"]) / bb_range.replace(0, np.nan)
    df["bb_lower_touch"] = (df["entry_px"] <= df["entry_bb_lower"] * 1.01).astype(int)
    df["price_above_st"] = (df["entry_px"] > df["entry_supertrend"]).astype(int)
    df["is_morning"] = ((df["entry_hour"] >= 9) & (df["entry_hour"] < 12)).astype(int)
    df["is_afternoon"] = ((df["entry_hour"] >= 12) & (df["entry_hour"] < 15)).astype(int)
    df["is_bull"] = (df["trend_regime"] == 1).astype(int)
    df["is_neutral"] = (df["trend_regime"] == 0).astype(int)
    df["is_bear"] = (df["trend_regime"] == -1).astype(int)

    # Additional descriptive / interaction features
    df["price_ma20_ratio"] = (df["entry_px"] - df["entry_ma20"]) / df["entry_px"]
    df["price_ma60_ratio"] = (df["entry_px"] - df["entry_ma60"]) / df["entry_px"]
    df["bb_width"] = (df["entry_bb_upper"] - df["entry_bb_lower"]) / df["entry_px"]
    df["atr_ratio"] = df["entry_atr"] / df["entry_px"]
    df["rsi_macd_interaction"] = df["entry_rsi"] * df["entry_macd_hist"]
    df["rsi_bb_interaction"] = df["entry_rsi"] * df["bb_position"]

    # KDE interactions per timeframe (if present)
    for tf in ["5m", "15m", "30m", "60m"]:
        aligned = f"kde_aligned_tail_{tf}"
        opposite = f"kde_opposite_tail_{tf}"
        zscore = f"kde_aligned_zscore_{tf}"
        if aligned in df.columns and opposite in df.columns:
            df[f"kde_tail_spread_{tf}"] = df[aligned] - df[opposite]
        if zscore in df.columns:
            df[f"kde_zscore_abs_{tf}"] = df[zscore].abs()
    return df


def split_by_year(df: pd.DataFrame, train_years: tuple, val_year: int, test_year: int | tuple) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_years = tuple(train_years)
    if isinstance(test_year, int):
        test_years = (test_year,)
    else:
        test_years = tuple(test_year)
    train = df[df["year"].isin(train_years)].copy()
    val = df[df["year"] == val_year].copy()
    test = df[df["year"].isin(test_years)].copy()
    return train, val, test


def _metric_score(df: pd.DataFrame, metric: str) -> float:
    net = df["net_krw"]
    if len(net) < 2:
        return -np.inf
    if metric == "pnl":
        return float(net.sum())
    if metric == "avg_pnl":
        return float(net.mean())
    if metric == "sharpe":
        s = net.std()
        return float(np.sqrt(len(net)) * net.mean() / s) if s and s > 0 else -np.inf
    if metric == "pf":
        wins = net[net > 0].sum()
        losses = abs(net[net < 0].sum())
        return float(wins / losses) if losses > 0 else np.inf
    if metric == "sharpe_x_pf":
        s = net.std()
        sharpe = float(np.sqrt(len(net)) * net.mean() / s) if s and s > 0 else -np.inf
        wins = net[net > 0].sum()
        losses = abs(net[net < 0].sum())
        pf = float(wins / losses) if losses > 0 else 0.0
        return sharpe * pf
    return float(net.sum())


def select_threshold_by_pnl(
    df_val: pd.DataFrame,
    y_proba_val: np.ndarray,
    thresholds: np.ndarray | None = None,
    min_trades: int = 50,
    metric: str = "pnl",
) -> float:
    """Pick threshold maximizing a chosen metric on validation data."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.95, 0.05)
    best_thr = 0.5
    best_score = -np.inf
    for thr in thresholds:
        mask = y_proba_val >= thr
        if mask.sum() < min_trades:
            continue
        subset = df_val.loc[mask].copy()
        score = _metric_score(subset, metric)
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr


def train_xgboost_filter(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
) -> xgb.XGBClassifier:
    X_train = train_df[feature_cols].fillna(0).astype(float)
    y_train = train_df["is_win"]
    X_val = val_df[feature_cols].fillna(0).astype(float)
    y_val = val_df["is_win"]

    counts = y_train.value_counts()
    neg, pos = counts.get(0, 1), counts.get(1, 1)
    scale_pos_weight = float(neg / max(pos, 1))

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
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def stage1_filter(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    min_trades: int = 300,
    metric: str = "pnl",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    model = train_xgboost_filter(train_df, val_df, feature_cols)
    proba_val = model.predict_proba(val_df[feature_cols].fillna(0).astype(float))[:, 1]
    threshold = select_threshold_by_pnl(val_df, proba_val, min_trades=min_trades, metric=metric)

    def apply(df):
        proba = model.predict_proba(df[feature_cols].fillna(0).astype(float))[:, 1]
        return df.assign(filter_score=proba).loc[proba >= threshold].copy()

    filt_train = apply(train_df)
    filt_val = apply(val_df)
    filt_test = apply(test_df)

    metrics = classification_metrics(
        test_df["is_win"],
        (model.predict_proba(test_df[feature_cols].fillna(0).astype(float))[:, 1] >= threshold).astype(int),
        model.predict_proba(test_df[feature_cols].fillna(0).astype(float))[:, 1],
    )
    return filt_train, filt_val, filt_test, metrics


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
    }


def train_random_forest_entry(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
) -> RandomForestClassifier:
    X_train = train_df[feature_cols].fillna(0).astype(float)
    y_train = train_df["is_win"]
    X_val = val_df[feature_cols].fillna(0).astype(float)
    y_val = val_df["is_win"]

    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=4,
        min_samples_split=25,
        min_samples_leaf=12,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def stage2_entry_timing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    min_trades: int = 100,
    metric: str = "pnl",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    model = train_random_forest_entry(train_df, val_df, feature_cols)
    proba_val = model.predict_proba(val_df[feature_cols].fillna(0).astype(float))[:, 1]
    threshold = select_threshold_by_pnl(val_df, proba_val, min_trades=min_trades, metric=metric)

    def apply(df):
        proba = model.predict_proba(df[feature_cols].fillna(0).astype(float))[:, 1]
        return df.assign(entry_score=proba).loc[proba >= threshold].copy()

    opt_train = apply(train_df)
    opt_val = apply(val_df)
    opt_test = apply(test_df)

    metrics = classification_metrics(
        test_df["is_win"],
        (model.predict_proba(test_df[feature_cols].fillna(0).astype(float))[:, 1] >= threshold).astype(int),
        model.predict_proba(test_df[feature_cols].fillna(0).astype(float))[:, 1],
    )
    return opt_train, opt_val, opt_test, metrics


def build_lstm_model(input_shape: Tuple[int, int], config: dict) -> Sequential:
    """Build an LSTM exit-timing model from a config dict."""
    units = config.get("lstm_units", 64)
    dropout = config.get("dropout", 0.5)
    lr = config.get("learning_rate", 0.001)
    bidirectional = config.get("use_bidirectional", False)

    model = Sequential()
    if bidirectional:
        model.add(
            Bidirectional(
                LSTM(units, return_sequences=True,
                     kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
                input_shape=input_shape,
            )
        )
    else:
        model.add(
            LSTM(units, return_sequences=True,
                 input_shape=input_shape,
                 kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))
        )
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(
        LSTM(max(units // 2, 8), return_sequences=False,
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))
    )
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(max(units // 4, 8), activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weight(y: np.ndarray) -> dict:
    counts = np.bincount(y.astype(int))
    if len(counts) < 2 or counts[1] == 0:
        return {0: 1.0, 1: 1.0}
    return {0: 1.0, 1: counts[0] / counts[1]}


def train_lstm_exit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    sequence_length: int = 10,
    config: dict | None = None,
):
    if config is None:
        config = {}
    df_sorted = pd.concat([train_df, val_df]).sort_values("entry_time").reset_index(drop=True)

    X_raw = df_sorted[feature_cols].fillna(0).astype(float).values
    y = df_sorted["is_win"].values

    scaler = MinMaxScaler()
    train_len = len(train_df)
    val_len = len(val_df)
    X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_seq, train_mask_seq, val_mask_seq = [], [], [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i : i + sequence_length])
        y_seq.append(y[i + sequence_length])
        idx = i + sequence_length
        train_mask_seq.append(idx < train_len)
        val_mask_seq.append(train_len <= idx < train_len + val_len)

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    train_mask_seq = np.array(train_mask_seq)
    val_mask_seq = np.array(val_mask_seq)

    if train_mask_seq.sum() < 50 or val_mask_seq.sum() < 10:
        raise ValueError(f"Not enough sequences: train={train_mask_seq.sum()}, val={val_mask_seq.sum()}")

    model = build_lstm_model((X_seq.shape[1], X_seq.shape[2]), config)
    class_weight = compute_class_weight(y_seq[train_mask_seq])

    model.fit(
        X_seq[train_mask_seq],
        y_seq[train_mask_seq],
        epochs=config.get("epochs", 30),
        batch_size=config.get("batch_size", 32),
        validation_data=(X_seq[val_mask_seq], y_seq[val_mask_seq]),
        class_weight=class_weight,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.get("patience", 5),
                restore_best_weights=True,
            )
        ],
    )
    return model, scaler


def stage3_exit_timing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    sequence_length: int = 10,
    lstm_config: dict | None = None,
    metric: str = "sharpe",
) -> Tuple[pd.DataFrame, dict]:
    if lstm_config is None:
        lstm_config = {}

    # Guard against too few samples for a meaningful LSTM sequence
    min_needed = sequence_length + 10
    if len(train_df) < min_needed or len(val_df) < min_needed or len(test_df) < min_needed:
        print(f"  Skipping LSTM: insufficient trades for sequence_length={sequence_length}")
        metrics = classification_metrics(
            test_df["is_win"],
            np.ones(len(test_df), dtype=int),
            np.ones(len(test_df)) * 0.5,
        )
        return test_df.copy(), metrics

    model, scaler = train_lstm_exit(train_df, val_df, feature_cols, sequence_length, lstm_config)

    # Predict on validation to choose threshold (lower min_trades for LSTM sequences)
    df_val_sorted = val_df.sort_values("entry_time").reset_index(drop=True)
    X_val = df_val_sorted[feature_cols].fillna(0).astype(float).values
    X_val_scaled = scaler.transform(X_val)
    X_val_seq = np.array([X_val_scaled[i : i + sequence_length]
                          for i in range(len(X_val_scaled) - sequence_length)])
    proba_val = model.predict(X_val_seq, verbose=0).flatten()
    df_val_for_thr = df_val_sorted.iloc[sequence_length:].copy()
    threshold = select_threshold_by_pnl(df_val_for_thr, proba_val, min_trades=30, metric=metric)

    # Predict on test
    df_test_sorted = test_df.sort_values("entry_time").reset_index(drop=True)
    X_test = df_test_sorted[feature_cols].fillna(0).astype(float).values
    X_test_scaled = scaler.transform(X_test)
    X_test_seq = np.array([X_test_scaled[i : i + sequence_length]
                           for i in range(len(X_test_scaled) - sequence_length)])
    proba_test = model.predict(X_test_seq, verbose=0).flatten()
    df_test_for_pred = df_test_sorted.iloc[sequence_length:].copy()
    df_test_for_pred["exit_score"] = proba_test
    final_test = df_test_for_pred[df_test_for_pred["exit_score"] >= threshold].copy()

    metrics = classification_metrics(
        df_test_for_pred["is_win"],
        (proba_test >= threshold).astype(int),
        proba_test,
    )
    return final_test, metrics


def stage3_exit_filter(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    min_trades: int = 30,
    metric: str = "sharpe",
) -> Tuple[pd.DataFrame, dict]:
    """Deterministic final exit filter using an XGBoost classifier.

    Replaces the LSTM exit stage with a reproducible, seed-fixed XGB model.
    Threshold is chosen on validation data using the specified metric.
    """
    model = train_xgboost_filter(train_df, val_df, feature_cols)
    proba_val = model.predict_proba(val_df[feature_cols].fillna(0).astype(float))[:, 1]
    threshold = select_threshold_by_pnl(val_df, proba_val, min_trades=min_trades, metric=metric)

    def apply(df):
        proba = model.predict_proba(df[feature_cols].fillna(0).astype(float))[:, 1]
        return df.assign(exit_score=proba).loc[proba >= threshold].copy()

    final_train = apply(train_df)
    final_val = apply(val_df)
    final_test = apply(test_df)

    metrics = classification_metrics(
        final_test["is_win"],
        (final_test["exit_score"] >= threshold).astype(int),
        final_test["exit_score"],
    )
    return final_test, metrics


def evaluate_trades(df: pd.DataFrame) -> dict:
    return {
        "n_trades": len(df),
        "win_rate": df["is_win"].mean() * 100 if len(df) else 0.0,
        "total_pnl": df["net_krw"].sum() if "net_krw" in df.columns else 0.0,
        "avg_pnl": df["net_krw"].mean() if "net_krw" in df.columns and len(df) else 0.0,
    }


def get_feature_cols(df: pd.DataFrame, use_kde: bool) -> Tuple[list, list]:
    base_cols = [c for c in BASELINE_FEATURES if c in df.columns]
    if use_kde:
        # Include raw KDE outputs and direction-aware KDE features only.
        # Avoid derived spread/abs columns, which are added below.
        kde_cols = [
            c for c in df.columns
            if ("_kde_" in c and not c.endswith("_ready"))
            or c.startswith("kde_aligned_tail_")
            or c.startswith("kde_opposite_tail_")
            or c.startswith("kde_aligned_zscore_")
        ]
    else:
        kde_cols = []
    feature_cols_stage1 = base_cols + kde_cols

    derived_cols = [
        "rsi_oversold", "rsi_overbought", "macd_bullish", "macd_strength",
        "price_above_ma5", "price_above_ma10", "price_above_ma20", "price_above_ma60",
        "bb_position", "bb_lower_touch", "price_above_st",
        "is_morning", "is_afternoon", "is_bull", "is_neutral", "is_bear",
    ]
    # KDE interaction features (per timeframe)
    if use_kde:
        derived_cols += [c for c in df.columns if c.startswith("kde_tail_spread_") or c.startswith("kde_zscore_abs_")]
    feature_cols_stage2 = feature_cols_stage1 + [c for c in derived_cols if c in df.columns]
    return feature_cols_stage1, feature_cols_stage2


def run_pipeline(
    df: pd.DataFrame,
    use_kde: bool,
    variant: str,
    train_years: tuple,
    val_year: int,
    test_year: int | tuple,
    stage1_metric: str = "pnl",
    stage2_metric: str = "pnl",
    stage3_metric: str = "sharpe",
) -> Tuple[pd.DataFrame, dict]:
    print(f"\n{'='*80}")
    print(f"Running full pipeline: {variant}")
    print(f"  train: {train_years}, val: {val_year}, test: {test_year}")
    print(f"  stage metrics: stage1={stage1_metric}, stage2={stage2_metric}, stage3={stage3_metric}")
    print(f"{'='*80}")

    train_df, val_df, test_df = split_by_year(df, train_years, val_year, test_year)
    print(f"Data split -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    feature_cols_stage1, feature_cols_stage2 = get_feature_cols(df, use_kde)

    # Stage 1: XGBoost filter
    print("\n[Stage 1/3] XGBoost trade filter")
    filt_train, filt_val, filt_test, metrics1 = stage1_filter(
        train_df, val_df, test_df, feature_cols_stage1, metric=stage1_metric
    )
    print(f"  Filter metrics: {metrics1}")
    print(f"  Filtered -> train: {len(filt_train)}, val: {len(filt_val)}, test: {len(filt_test)}")

    # Stage 2: RF entry timing
    print("\n[Stage 2/3] Random Forest entry timing")
    opt_train, opt_val, opt_test, metrics2 = stage2_entry_timing(
        filt_train, filt_val, filt_test, feature_cols_stage2, metric=stage2_metric
    )
    print(f"  Entry metrics: {metrics2}")
    print(f"  Optimized -> train: {len(opt_train)}, val: {len(opt_val)}, test: {len(opt_test)}")

    # Stage 3: LSTM exit timing (or bypass for stage1+stage2 only test)
    print("\n[Stage 3/3] LSTM exit timing")
    use_lstm = os.environ.get("SKIP_LSTM_STAGE", "0") != "1"
    if use_lstm:
        final_test, metrics3 = stage3_exit_timing(
            opt_train, opt_val, opt_test, feature_cols_stage2,
            sequence_length=10, metric=stage3_metric,
        )
    else:
        final_test = opt_test.copy()
        metrics3 = classification_metrics(
            final_test["is_win"],
            np.ones(len(final_test), dtype=int),
            np.ones(len(final_test)) * 0.5,
        )
    print(f"  Exit metrics: {metrics3}")
    print(f"  Final test trades: {len(final_test)}")

    return final_test, {
        "variant": variant,
        "train_years": train_years,
        "val_year": val_year,
        "test_year": test_year,
        "stage1_metrics": metrics1,
        "stage2_metrics": metrics2,
        "stage3_metrics": metrics3,
        "final": evaluate_trades(final_test),
    }


def walk_forward_validation(df: pd.DataFrame, *args, **kwargs) -> list:
    """Run multiple train/val/test year splits and compare baseline vs KDE."""
    folds = [
        ((2019, 2020, 2021, 2022, 2023), 2024, 2025),
        ((2019, 2020, 2021, 2022, 2023, 2024), 2025, 2026),
        ((2020, 2021, 2022, 2023, 2024), 2025, 2026),
    ]
    results = []
    for train_years, val_year, test_year in folds:
        print(f"\n{'#'*80}")
        print(f"# Walk-forward fold: train={train_years}, val={val_year}, test={test_year}")
        print(f"{'#'*80}")
        for use_kde, variant in [(False, "Baseline"), (True, "KDE-enhanced")]:
            final_test, info = run_pipeline(df, use_kde, variant, train_years, val_year, test_year)
            info["fold"] = f"{train_years[-1]}->{test_year}"
            results.append(info)
            suffix = f"{train_years[-1]}_{test_year}"
            final_test.to_csv(OUTPUT_DIR / f"final_trades_{variant.lower()}_{suffix}.csv", index=False)
    return results


def print_results_table(results: list) -> None:
    print("\n" + "=" * 100)
    print("Walk-forward / Single-run results summary")
    print("=" * 100)
    print(f"{'Fold/Run':<20}{'Variant':<15}{'Trades':>8}{'Win%':>10}{'Total PnL':>15}{'Avg PnL':>12}")
    print("-" * 100)
    for info in results:
        f = info["final"]
        fold = info.get("fold", "single")
        print(
            f"{fold:<20}{info['variant']:<15}{f['n_trades']:>8}{f['win_rate']:>10.2f}"
            f"{f['total_pnl']:>15,.0f}{f['avg_pnl']:>12,.0f}"
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if tf is None:
        print("TensorFlow is required for LSTM exit timing.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Full 3-stage ML pipeline with optional KDE")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--slippage-ticks", type=int, default=SLIPPAGE_TICKS)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--lstm-dropout", type=float, default=0.5)
    parser.add_argument("--lstm-lr", type=float, default=0.001)
    parser.add_argument("--lstm-epochs", type=int, default=30)
    parser.add_argument("--lstm-patience", type=int, default=5)
    parser.add_argument("--lstm-bidirectional", action="store_true")
    args = parser.parse_args()

    lstm_config = {
        "lstm_units": args.lstm_units,
        "dropout": args.lstm_dropout,
        "learning_rate": args.lstm_lr,
        "epochs": args.lstm_epochs,
        "patience": args.lstm_patience,
        "use_bidirectional": args.lstm_bidirectional,
    }

    df = load_data(slippage_ticks=args.slippage_ticks)
    print(f"Loaded {len(df)} trades with slippage={args.slippage_ticks} tick(s)/side")

    if args.walk_forward:
        results = walk_forward_validation(df, lstm_config, args.sequence_length)
    else:
        final_baseline, info_baseline = run_pipeline(
            df, False, "Baseline", (2019, 2020, 2021, 2022, 2023), 2024, (2025, 2026),
            lstm_config=lstm_config, sequence_length=args.sequence_length,
        )
        final_kde, info_kde = run_pipeline(
            df, True, "KDE-enhanced", (2019, 2020, 2021, 2022, 2023), 2024, (2025, 2026),
            lstm_config=lstm_config, sequence_length=args.sequence_length,
        )
        final_baseline.to_csv(OUTPUT_DIR / "final_trades_baseline_retrain.csv", index=False)
        final_kde.to_csv(OUTPUT_DIR / "final_trades_kde.csv", index=False)
        results = [info_baseline, info_kde]

    print_results_table(results)
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
