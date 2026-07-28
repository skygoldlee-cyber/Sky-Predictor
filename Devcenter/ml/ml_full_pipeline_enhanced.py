"""Enhanced 5m ML pipeline with sequential improvements (Stages 1-5).

Builds on `ml_full_pipeline_kde.py` and adds:
    1. High-confidence meta-filter (KDE-aware when available)
    2. Ensemble exit (LSTM + xgb_cls + xgb_reg + optional trailing stop)
    3. Regime-aware stage thresholds
    4. Fractional Kelly sizing + drawdown guard
    5. Multi-timeframe KDE + walk-forward harness

Designed to run even when `ml_dataset_with_kde.csv` is missing; KDE-dependent
stages gracefully degrade to baseline features.

Time split default:
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
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTHONHASHSEED", "42")
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

# Re-use the core training/evaluation helpers from the original KDE pipeline.
from ml_full_pipeline_kde import (
    BASELINE_FEATURES,
    CONTRACT_MULTIPLIER,
    KDE_FEATURES,
    TICK_SIZE,
    add_derived_features,
    apply_trading_costs,
    build_lstm_model,
    classification_metrics,
    compute_class_weight,
    evaluate_trades,
    get_feature_cols,
    select_threshold_by_pnl,
    split_by_year,
    stage1_filter,
    stage2_entry_timing,
    stage3_exit_filter,
    stage3_exit_regression,
    stage3_exit_timing,
    train_lstm_exit,
    train_random_forest_entry,
    train_xgboost_filter,
    train_xgboost_regressor,
)

try:
    import tensorflow as tf
    tf.random.set_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    from tensorflow import keras
except Exception as exc:  # pragma: no cover
    tf = keras = None
    logging.warning("TensorFlow not available: %s", exc)


_logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "ml_data"
OUTPUT_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)

SLIPPAGE_TICKS = 1


def load_data(
    slippage_ticks: int = SLIPPAGE_TICKS,
    kde_path: Optional[Path] = None,
    target_horizon: int | None = None,
) -> pd.DataFrame:
    """Load dataset, preferring KDE version if available.

    Falls back to `ml_dataset.csv` if KDE CSV is missing.
    If ``target_horizon`` is given, use fixed-horizon directional returns as the
    supervised labels instead of the strategy's own exit PnL.
    """
    if kde_path is None:
        kde_path = DATA_DIR / "ml_dataset_with_kde.csv"
    base_path = DATA_DIR / "ml_dataset.csv"

    if kde_path.exists():
        _logger.info("Loading KDE dataset: %s", kde_path)
        df = pd.read_csv(kde_path)
        has_kde = True
    else:
        _logger.warning("KDE dataset not found, loading base dataset: %s", base_path)
        df = pd.read_csv(base_path)
        has_kde = False

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["year"] = df["entry_time"].dt.year
    df = apply_trading_costs(df, slippage_ticks=slippage_ticks)
    df = add_derived_features(df)

    if target_horizon is not None:
        ret_col = f"future_return_{target_horizon}"
        win_col = f"is_win_h{target_horizon}"
        if ret_col not in df.columns or win_col not in df.columns:
            raise KeyError(
                f"Fixed-horizon target columns {ret_col}/{win_col} not found. "
                "Run Devcenter/ml/add_fixed_horizon_labels.py first."
            )
        size = df["size_factor"].replace(0, np.nan).fillna(1.0)
        df["is_win"] = df[win_col]
        df["pnl_per_contract"] = df[ret_col] * CONTRACT_MULTIPLIER
    else:
        size = df["size_factor"].replace(0, np.nan).fillna(1.0)
        df["pnl_per_contract"] = df["net_krw"] / size

    df["_has_kde"] = has_kde
    return df


# ---------------------------------------------------------------------------
# Enhanced evaluation metrics
# ---------------------------------------------------------------------------

def _daily_equity(df: pd.DataFrame, init_capital: float = 10_000_000.0) -> pd.Series:
    """Build daily equity curve from net_krw trades."""
    df = df.copy()
    df["exit_date"] = pd.to_datetime(df.get("exit_date", df["entry_time"]))
    daily = df.groupby("exit_date")["net_krw"].sum().sort_index()
    equity = init_capital + daily.cumsum()
    return equity


def evaluate_trades_enhanced(
    df: pd.DataFrame,
    init_capital: float = 10_000_000.0,
) -> Dict[str, float]:
    """Compute metrics aligned with MODEL_EVALUATION_FRAMEWORK.md."""
    if len(df) == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
            "profit_factor": 0.0, "sharpe": 0.0, "cagr": 0.0, "mdd": 0.0,
            "mar": 0.0, "profit_loss_ratio": 0.0,
        }

    net = df["net_krw"]
    wins = net[net > 0]
    losses = net[net < 0]
    pf = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else float("inf")
    win_rate = float((net > 0).mean() * 100)
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    pl_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float("inf")

    equity = _daily_equity(df, init_capital)
    daily_pnl = equity.diff().dropna()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = float(np.sqrt(252) * daily_pnl.mean() / daily_pnl.std())
    else:
        sharpe = 0.0

    # Drawdown on daily equity
    running_max = equity.cummax()
    drawdown = equity - running_max
    mdd = float(-drawdown.min())

    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-6)
    final_equity = float(equity.iloc[-1])
    if final_equity <= 0:
        cagr = -1.0
    else:
        cagr = float((final_equity / init_capital) ** (1.0 / years) - 1.0)
    mar = cagr / (mdd / init_capital) if mdd > 0 else 0.0

    return {
        "n_trades": len(df),
        "win_rate": win_rate,
        "total_pnl": float(net.sum()),
        "avg_pnl": float(net.mean()),
        "profit_factor": pf,
        "sharpe": sharpe,
        "cagr": cagr * 100,  # percent
        "mdd": mdd,
        "mar": mar,
        "profit_loss_ratio": pl_ratio,
    }


# ---------------------------------------------------------------------------
# Stage 3: Ensemble exit model
# ---------------------------------------------------------------------------

def stage3_exit_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    sequence_length: int = 10,
    weights: Optional[Dict[str, float]] = None,
    metric: str = "sharpe",
    min_trades: int = 30,
    lstm_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Combine LSTM, XGB classifier, and XGB regressor exit scores.

    Each sub-model is trained independently and its score is normalized to
    [0, 1] on validation. The final ensemble score is a weighted average.
    Threshold is chosen on the ensemble score using the specified metric.
    """
    if weights is None:
        weights = {"lstm": 0.4, "xgb_cls": 0.3, "xgb_reg": 0.3}

    scores_val = pd.Series(0.0, index=val_df.index)
    scores_test = pd.Series(0.0, index=test_df.index)
    total_weight = 0.0

    # XGB classifier exit
    try:
        cls_model = train_xgboost_filter(train_df, val_df, feature_cols)
        val_proba = cls_model.predict_proba(val_df[feature_cols].fillna(0).astype(float))[:, 1]
        test_proba = cls_model.predict_proba(test_df[feature_cols].fillna(0).astype(float))[:, 1]
        scores_val += weights["xgb_cls"] * val_proba
        scores_test += weights["xgb_cls"] * test_proba
        total_weight += weights["xgb_cls"]
    except Exception as exc:
        _logger.warning("xgb_cls exit failed: %s", exc)

    # XGB regressor exit
    try:
        reg_model = train_xgboost_regressor(train_df, val_df, feature_cols)
        val_pred = reg_model.predict(val_df[feature_cols].fillna(0).astype(float))
        test_pred = reg_model.predict(test_df[feature_cols].fillna(0).astype(float))
        # Normalize regressor output to [0, 1] using validation min/max
        v_min, v_max = val_pred.min(), val_pred.max()
        if v_max > v_min:
            val_norm = (val_pred - v_min) / (v_max - v_min)
            test_norm = (test_pred - v_min) / (v_max - v_min)
        else:
            val_norm = np.ones_like(val_pred) * 0.5
            test_norm = np.ones_like(test_pred) * 0.5
        scores_val += weights["xgb_reg"] * val_norm
        scores_test += weights["xgb_reg"] * test_norm
        total_weight += weights["xgb_reg"]
    except Exception as exc:
        _logger.warning("xgb_reg exit failed: %s", exc)

    # LSTM exit (only if TF available)
    if tf is not None and len(train_df) >= sequence_length + 10 and len(val_df) >= sequence_length + 10 and len(test_df) >= sequence_length + 10:
        try:
            model, scaler = train_lstm_exit(train_df, val_df, feature_cols, sequence_length, lstm_config or {})

            df_val_sorted = val_df.sort_values("entry_time")
            X_val = scaler.transform(df_val_sorted[feature_cols].fillna(0).astype(float).values)
            X_val_seq = np.array([X_val[i : i + sequence_length] for i in range(len(X_val) - sequence_length)])
            val_proba = model.predict(X_val_seq, verbose=0).flatten()
            val_index = df_val_sorted.index[sequence_length:]

            df_test_sorted = test_df.sort_values("entry_time")
            X_test = scaler.transform(df_test_sorted[feature_cols].fillna(0).astype(float).values)
            X_test_seq = np.array([X_test[i : i + sequence_length] for i in range(len(X_test) - sequence_length)])
            test_proba = model.predict(X_test_seq, verbose=0).flatten()
            test_index = df_test_sorted.index[sequence_length:]

            scores_val.loc[val_index] += weights["lstm"] * val_proba
            scores_test.loc[test_index] += weights["lstm"] * test_proba
            total_weight += weights["lstm"]
        except Exception as exc:
            _logger.warning("LSTM exit failed: %s", exc)

    if total_weight == 0:
        _logger.warning("No ensemble exit models succeeded; returning unfiltered test set")
        return test_df.copy(), classification_metrics(test_df["is_win"], np.ones(len(test_df), dtype=int), np.ones(len(test_df)) * 0.5)

    scores_val /= total_weight
    scores_test /= total_weight

    threshold, val_score, _ = select_threshold_by_pnl(val_df, scores_val.values, min_trades=min_trades, metric=metric)

    val_scored = val_df.assign(exit_score=scores_val)
    test_scored = test_df.assign(exit_score=scores_test)
    final_test = test_scored[test_scored["exit_score"] >= threshold].copy()

    metrics = classification_metrics(
        test_df["is_win"],
        (test_scored["exit_score"] >= threshold).astype(int),
        test_scored["exit_score"],
    )
    metrics["threshold"] = round(threshold, 6)
    metrics["val_score"] = round(val_score, 2)
    metrics["weights"] = weights
    return final_test, metrics


# ---------------------------------------------------------------------------
# Stage 3.5: Trailing stop overlay (rule-based guard)
# ---------------------------------------------------------------------------

def apply_trailing_stop(
    df: pd.DataFrame,
    atr_col: str = "entry_atr",
    atr_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Simulate an ATR-based trailing stop overlay on completed trades.

    This is a post-hoc approximation: if a trade's unrealized drawdown
    exceeded `atr_multiplier * entry_atr`, assume it was stopped at that
    level and cap `net_krw`.
    """
    df = df.copy()
    if atr_col not in df.columns or "gross_pts" not in df.columns:
        return df

    # Approximate max favorable / adverse excursion from gross_pts
    # Positive gross_pts -> long wins; negative -> long losses
    direction = df["direction"].replace(0, 1)
    stop_pts = df[atr_col].abs() * atr_multiplier
    # Cap loss at stop distance on losing side
    gross = df["gross_pts"] * direction
    capped_gross = np.where(
        gross < -stop_pts,
        -stop_pts,
        gross,
    )
    df["gross_pts_capped"] = capped_gross * direction
    cost = df["cost_pts"]
    df["net_pts"] = df["gross_pts_capped"] - cost
    size = df["size_factor"].replace(0, np.nan).fillna(1.0)
    df["net_krw"] = df["net_pts"] * CONTRACT_MULTIPLIER * size
    df["is_win"] = (df["net_krw"] > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Stage 4: Fractional Kelly sizing + drawdown guard
# ---------------------------------------------------------------------------

def apply_fractional_kelly(
    df: pd.DataFrame,
    fraction: float = 0.5,
    lookback: int = 252,
    min_contracts: float = 0.0,
    max_contracts: float = 10.0,
    drawdown_guard: bool = True,
    dd_threshold: float = 0.10,
    dd_reduction: float = 0.5,
) -> pd.DataFrame:
    """Apply fractional Kelly sizing with optional drawdown guard.

    Uses a rolling window of per-contract PnL to estimate edge (b) and
    variance. Kelly fraction = edge / variance. This is then scaled by the
    user-supplied fraction and bounded by [min_contracts, max_contracts].

    Drawdown guard: when the running drawdown exceeds `dd_threshold` of
    initial capital, multiply the target contract size by `dd_reduction`.
    """
    df = df.copy().sort_values("entry_time").reset_index(drop=True)
    pnl_per_contract = df["pnl_per_contract"].astype(float)

    contracts = []
    equity = 0.0
    peak_equity = 0.0
    for i in range(len(df)):
        if i < lookback:
            # Warm-up: fixed 1 contract
            c = 1.0
        else:
            window = pnl_per_contract.iloc[max(0, i - lookback) : i]
            mean_p = float(window.mean())
            var_p = float(window.var())
            if var_p > 0 and mean_p > 0:
                kelly = mean_p / var_p
            else:
                kelly = 0.0
            c = kelly * fraction

        c = max(min_contracts, min(max_contracts, c))

        if drawdown_guard:
            if peak_equity > 0 and (peak_equity - equity) / peak_equity > dd_threshold:
                c *= dd_reduction
            elif equity > peak_equity:
                peak_equity = equity

        contracts.append(c)
        equity += pnl_per_contract.iloc[i] * c

    df["size_factor"] = np.array(contracts)
    df["net_krw"] = df["pnl_per_contract"] * df["size_factor"]
    df["is_win"] = (df["net_krw"] > 0).astype(int)
    return df


def _select_top_features(
    train_df: pd.DataFrame,
    feature_cols: list,
    k: int = 15,
    target_col: str = "is_win",
    seed: int = 42,
) -> list:
    """Select the top-k features by XGBClassifier gain importance.

    Drops constant / zero-variance columns first, then trains a small XGB
    classifier on the training set and returns the k most important features.
    """
    if k <= 0 or len(feature_cols) <= k:
        return feature_cols

    X = train_df[feature_cols].fillna(0).astype(float)
    std = X.std(numeric_only=True)
    usable = std[std > 0].index.tolist()
    if len(usable) <= k:
        return usable

    X = X[usable]
    y = train_df[target_col]
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="logloss",
        reg_alpha=0.5,
        reg_lambda=2.0,
    )
    model.fit(X, y, verbose=False)
    importance = pd.Series(model.feature_importances_, index=usable).sort_values(ascending=False)
    selected = set(importance.head(k).index)
    return [c for c in feature_cols if c in selected]


# ---------------------------------------------------------------------------
# Stage 3 extension: Regime-aware thresholds
# ---------------------------------------------------------------------------

def _select_regime_threshold(
    df_val: pd.DataFrame,
    scores: np.ndarray,
    regime_col: str = "trend_regime",
    min_trades: int = 30,
    metric: str = "sharpe_x_pf",
) -> Tuple[Dict[int, float], float]:
    """Select score threshold per regime on validation data.

    Regimes with too few validation samples fall back to a global threshold
    computed on the full validation set, avoiding a pass-through threshold of
    0.0 that can pollute later stages.
    """
    global_thr, global_score, _ = select_threshold_by_pnl(
        df_val, scores, min_trades=min_trades, metric=metric
    )

    regimes = sorted(df_val[regime_col].dropna().unique())
    thresholds = {}
    overall_score = global_score
    for r in regimes:
        mask = df_val[regime_col] == r
        if mask.sum() < min_trades:
            thresholds[int(r)] = float(global_thr)
            continue
        thr, score, _ = select_threshold_by_pnl(
            df_val.loc[mask],
            scores[mask.values],
            min_trades=max(10, min_trades // 3),
            metric=metric,
        )
        thresholds[int(r)] = float(thr)
        overall_score += score
    return thresholds, overall_score


def _add_balanced_regime(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_col: str = "entry_hour",
    n_regimes: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign a balanced regime column using training-set quantiles.

    The default feature is ``entry_hour`` because many pre-computed technical
    indicators in the current dataset (ATR, RSI, volume, etc.) are constant
    zeros. Hour-of-day is a stable intraday seasonality proxy. The regime
    labels 0..n_regimes-1 split the training set into roughly equal buckets by
    the chosen feature.
    """

    if base_col not in train_df.columns:
        raise KeyError(f"Regime base column '{base_col}' not found in training data")

    train = train_df.copy()
    train["__regime_feat__"] = pd.to_numeric(train[base_col], errors="coerce")
    train = train.dropna(subset=["__regime_feat__"])
    if len(train) < n_regimes * 10:
        raise ValueError(f"Not enough training samples to define {n_regimes} regimes")

    quantile_edges = np.linspace(0, 1, n_regimes + 1)
    thresholds = train["__regime_feat__"].quantile(quantile_edges[1:-1]).values

    def assign(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        feat = pd.to_numeric(df[base_col], errors="coerce").values
        regime = np.searchsorted(thresholds, feat, side="right")
        regime = np.clip(regime, 0, n_regimes - 1)
        df["balanced_regime"] = regime.astype(int)
        return df

    return assign(train_df), assign(val_df), assign(test_df)


def stage1_filter_regime_aware(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    regime_col: str = "trend_regime",
    min_trades: int = 300,
    metric: str = "sharpe_x_pf",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Stage 1 XGBoost filter with regime-specific thresholds."""
    model = train_xgboost_filter(train_df, val_df, feature_cols)

    def score(df):
        return model.predict_proba(df[feature_cols].fillna(0).astype(float))[:, 1]

    val_scores = score(val_df)
    thresholds, val_score = _select_regime_threshold(val_df, val_scores, regime_col=regime_col, min_trades=min_trades, metric=metric)

    def apply(df):
        proba = score(df)
        thr = df[regime_col].map(thresholds).fillna(0.0)
        return df.assign(filter_score=proba).loc[proba >= thr].copy()

    filt_train = apply(train_df)
    filt_val = apply(val_df)
    filt_test = apply(test_df)

    proba_test = score(test_df)
    thr_test = test_df[regime_col].map(thresholds).fillna(0.0)
    metrics = classification_metrics(
        test_df["is_win"],
        (proba_test >= thr_test.values).astype(int),
        proba_test,
    )
    metrics["threshold"] = thresholds
    metrics["val_score"] = round(val_score, 2)
    return filt_train, filt_val, filt_test, metrics


def stage2_entry_timing_regime_aware(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    regime_col: str = "trend_regime",
    min_trades: int = 100,
    metric: str = "sharpe_x_pf",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Stage 2 Random Forest entry timing with regime-specific thresholds."""
    model = train_random_forest_entry(train_df, val_df, feature_cols)

    def score(df):
        return model.predict_proba(df[feature_cols].fillna(0).astype(float))[:, 1]

    val_scores = score(val_df)
    thresholds, val_score = _select_regime_threshold(val_df, val_scores, regime_col=regime_col, min_trades=min_trades, metric=metric)

    def apply(df):
        proba = score(df)
        thr = df[regime_col].map(thresholds).fillna(0.0)
        return df.assign(entry_score=proba).loc[proba >= thr].copy()

    opt_train = apply(train_df)
    opt_val = apply(val_df)
    opt_test = apply(test_df)

    proba_test = score(test_df)
    thr_test = test_df[regime_col].map(thresholds).fillna(0.0)
    metrics = classification_metrics(
        test_df["is_win"],
        (proba_test >= thr_test.values).astype(int),
        proba_test,
    )
    metrics["threshold"] = thresholds
    metrics["val_score"] = round(val_score, 2)
    return opt_train, opt_val, opt_test, metrics


# ---------------------------------------------------------------------------
# Stage 1: High-confidence meta-filter
# ---------------------------------------------------------------------------

def stage1_meta_filter(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    regime_col: str = "trend_regime",
    min_trades: int = 100,
    metric: str = "sharpe_x_pf",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Stage 1 meta-filter combining filter score, entry timing, and regime.

    Trains a small XGB classifier to predict `is_win` using:
        - stage1 filter score
        - stage2 entry score
        - regime indicators
        - KDE features (if present)
    """
    # Build stage scores as features
    filter_model = train_xgboost_filter(train_df, val_df, feature_cols)
    entry_model = train_random_forest_entry(train_df, val_df, feature_cols)

    def add_scores(df):
        df = df.copy()
        df["meta_filter_score"] = filter_model.predict_proba(df[feature_cols].fillna(0).astype(float))[:, 1]
        df["meta_entry_score"] = entry_model.predict_proba(df[feature_cols].fillna(0).astype(float))[:, 1]
        for r in sorted(df[regime_col].dropna().unique()):
            df[f"regime_{int(r)}"] = (df[regime_col] == r).astype(int)
        # Add KDE tail features if available
        for col in KDE_FEATURES:
            if col in df.columns:
                df[f"meta_{col}"] = df[col]
        return df

    train_scored = add_scores(train_df)
    val_scored = add_scores(val_df)
    test_scored = add_scores(test_df)

    meta_cols = [c for c in train_scored.columns if c.startswith("meta_") or c.startswith("regime_")]
    meta_model = train_xgboost_filter(train_scored, val_scored, meta_cols)

    val_proba = meta_model.predict_proba(val_scored[meta_cols].fillna(0).astype(float))[:, 1]
    threshold, val_score, _ = select_threshold_by_pnl(val_scored, val_proba, min_trades=min_trades, metric=metric)

    def apply(df):
        proba = meta_model.predict_proba(df[meta_cols].fillna(0).astype(float))[:, 1]
        return df.assign(meta_score=proba).loc[proba >= threshold].copy()

    filt_train = apply(train_scored)
    filt_val = apply(val_scored)
    filt_test = apply(test_scored)

    test_proba = meta_model.predict_proba(test_scored[meta_cols].fillna(0).astype(float))[:, 1]
    metrics = classification_metrics(
        test_scored["is_win"],
        (test_proba >= threshold).astype(int),
        test_proba,
    )
    metrics["threshold"] = round(threshold, 6)
    metrics["val_score"] = round(val_score, 2)
    return filt_train, filt_val, filt_test, metrics


# ---------------------------------------------------------------------------
# Full enhanced pipeline
# ---------------------------------------------------------------------------

def run_pipeline_enhanced(
    df: pd.DataFrame,
    use_kde: bool,
    variant: str,
    train_years: tuple,
    val_year: int,
    test_year: int | tuple,
    stage1_metric: str = "sharpe_x_pf",
    stage2_metric: str = "sharpe_x_pf",
    stage3_metric: str = "sharpe_x_pf",
    stage1_threshold: Optional[float] = None,
    stage2_threshold: Optional[float] = None,
    stage3_threshold: Optional[float] = None,
    stage3_min_trades: int = 30,
    stage3_type: str = "ensemble",
    regime_aware: bool = True,
    meta_filter: bool = False,
    ensemble_exit: bool = True,
    trailing_stop: bool = False,
    fractional_kelly: bool = True,
    kelly_fraction: float = 0.5,
    drawdown_guard: bool = True,
    sequence_length: int = 10,
    feature_selection_k: int = 0,
    lstm_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Run the enhanced 3-stage pipeline with optional improvements.

    Set ``stage1_threshold=0`` or ``stage2_threshold=0`` to skip that stage
    and pass all trades through; this is useful when the input dataset has
    already been filtered/optimized upstream.
    """
    print(f"\n{'='*80}")
    print(f"Running enhanced full pipeline: {variant}")
    print(f"  train: {train_years}, val: {val_year}, test: {test_year}")
    print(f"  stage metrics: stage1={stage1_metric}, stage2={stage2_metric}, stage3={stage3_metric}")
    print(f"  options: regime_aware={regime_aware}, meta_filter={meta_filter}, ensemble_exit={ensemble_exit}")
    print(f"           trailing_stop={trailing_stop}, fractional_kelly={fractional_kelly}, drawdown_guard={drawdown_guard}")
    print(f"{'='*80}")

    train_df, val_df, test_df = split_by_year(df, train_years, val_year, test_year)
    print(f"Data split -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    if regime_aware:
        train_df, val_df, test_df = _add_balanced_regime(train_df, val_df, test_df)

    feature_cols_stage1, feature_cols_stage2 = get_feature_cols(df, use_kde)

    if feature_selection_k > 0:
        print(f"  Selecting top-{feature_selection_k} features per stage")
        feature_cols_stage1 = _select_top_features(
            train_df, feature_cols_stage1, k=feature_selection_k, target_col="is_win",
        )
        feature_cols_stage2 = _select_top_features(
            train_df, feature_cols_stage2, k=feature_selection_k, target_col="is_win",
        )
        print(f"    stage1 features ({len(feature_cols_stage1)}): {feature_cols_stage1}")
        print(f"    stage2 features ({len(feature_cols_stage2)}): {feature_cols_stage2}")

    # Stage 1
    print("\n[Stage 1/3] Trade filter")
    skip_stage1 = stage1_threshold is not None and stage1_threshold <= 0.0
    if skip_stage1:
        filt_train, filt_val, filt_test = train_df.copy(), val_df.copy(), test_df.copy()
        metrics1 = classification_metrics(
            test_df["is_win"], np.ones(len(test_df), dtype=int), np.ones(len(test_df)) * 0.5
        )
        metrics1["threshold"] = 0.0
        metrics1["val_score"] = 0.0
        print("  Skipped (pass-through)")
    elif meta_filter:
        filt_train, filt_val, filt_test, metrics1 = stage1_meta_filter(
            train_df, val_df, test_df, feature_cols_stage1, min_trades=100, metric=stage1_metric,
        )
    elif regime_aware:
        filt_train, filt_val, filt_test, metrics1 = stage1_filter_regime_aware(
            train_df, val_df, test_df, feature_cols_stage1, regime_col="balanced_regime", min_trades=300, metric=stage1_metric,
        )
    else:
        filt_train, filt_val, filt_test, metrics1 = stage1_filter(
            train_df, val_df, test_df, feature_cols_stage1, metric=stage1_metric,
            fixed_threshold=stage1_threshold,
        )
    print(f"  Filter metrics: {metrics1}")
    print(f"  Filtered -> train: {len(filt_train)}, val: {len(filt_val)}, test: {len(filt_test)}")

    # Stage 2
    print("\n[Stage 2/3] Entry timing")
    skip_stage2 = stage2_threshold is not None and stage2_threshold <= 0.0
    if skip_stage2:
        opt_train, opt_val, opt_test = filt_train.copy(), filt_val.copy(), filt_test.copy()
        metrics2 = classification_metrics(
            filt_test["is_win"], np.ones(len(filt_test), dtype=int), np.ones(len(filt_test)) * 0.5
        )
        metrics2["threshold"] = 0.0
        metrics2["val_score"] = 0.0
        print("  Skipped (pass-through)")
    elif regime_aware:
        opt_train, opt_val, opt_test, metrics2 = stage2_entry_timing_regime_aware(
            filt_train, filt_val, filt_test, feature_cols_stage2, regime_col="balanced_regime", min_trades=100, metric=stage2_metric,
        )
    else:
        opt_train, opt_val, opt_test, metrics2 = stage2_entry_timing(
            filt_train, filt_val, filt_test, feature_cols_stage2, metric=stage2_metric,
            fixed_threshold=stage2_threshold,
        )
    print(f"  Entry metrics: {metrics2}")
    print(f"  Optimized -> train: {len(opt_train)}, val: {len(opt_val)}, test: {len(opt_test)}")

    # Stage 3
    print(f"\n[Stage 3/3] Exit stage ({stage3_type})")
    if stage3_type == "ensemble" and ensemble_exit:
        final_test, metrics3 = stage3_exit_ensemble(
            opt_train, opt_val, opt_test, feature_cols_stage2,
            sequence_length=sequence_length, metric=stage3_metric,
            lstm_config=lstm_config,
        )
    elif stage3_type == "lstm":
        final_test, metrics3 = stage3_exit_timing(
            opt_train, opt_val, opt_test, feature_cols_stage2,
            sequence_length=sequence_length, metric=stage3_metric,
            lstm_config=lstm_config,
        )
    elif stage3_type == "xgb_cls":
        final_test, metrics3 = stage3_exit_filter(
            opt_train, opt_val, opt_test, feature_cols_stage2, metric=stage3_metric,
            fixed_threshold=stage3_threshold, min_trades=stage3_min_trades,
        )
    elif stage3_type == "xgb_reg":
        final_test, metrics3 = stage3_exit_regression(
            opt_train, opt_val, opt_test, feature_cols_stage2, metric=stage3_metric,
            fixed_threshold=stage3_threshold, min_trades=stage3_min_trades,
        )
    else:
        final_test = opt_test.copy()
        metrics3 = classification_metrics(
            final_test["is_win"],
            np.ones(len(final_test), dtype=int),
            np.ones(len(final_test)) * 0.5,
        )
    print(f"  Exit metrics: {metrics3}")
    print(f"  Final test trades before sizing: {len(final_test)}")

    # Stage 3.5: trailing stop overlay
    if trailing_stop:
        final_test = apply_trailing_stop(final_test)
        print(f"  After trailing stop: {len(final_test)} trades")

    # Stage 4: fractional Kelly sizing + drawdown guard
    if fractional_kelly:
        final_test = apply_fractional_kelly(
            final_test,
            fraction=kelly_fraction,
            drawdown_guard=drawdown_guard,
        )
        print(f"  After fractional Kelly sizing: {len(final_test)} trades")

    final_metrics = evaluate_trades_enhanced(final_test)
    print(f"\n[Final metrics] {final_metrics}")

    return final_test, {
        "variant": variant,
        "train_years": train_years,
        "val_year": val_year,
        "test_year": test_year,
        "stage1_metrics": metrics1,
        "stage2_metrics": metrics2,
        "stage3_metrics": metrics3,
        "final": final_metrics,
    }


# ---------------------------------------------------------------------------
# Walk-forward harness
# ---------------------------------------------------------------------------

def walk_forward_validation_enhanced(
    df: pd.DataFrame,
    *args,
    folds: Optional[List[Tuple]] = None,
    **kwargs,
) -> List[dict]:
    """Run multiple train/val/test splits and compare variants."""
    if folds is None:
        folds = [
            ((2019, 2020, 2021), 2022, (2023, 2024)),
            ((2020, 2021, 2022), 2023, (2024, 2025)),
            ((2021, 2022, 2023), 2024, (2025, 2026)),
            ((2019, 2020, 2021, 2022, 2023), 2024, (2025, 2026)),
        ]
    results = []
    for train_years, val_year, test_year in folds:
        print(f"\n{'#'*80}")
        print(f"# Walk-forward fold: train={train_years}, val={val_year}, test={test_year}")
        print(f"{'#'*80}")
        for use_kde, variant in [(False, "Baseline"), (True, "KDE-enhanced")]:
            final_test, info = run_pipeline_enhanced(
                df, use_kde, variant, train_years, val_year, test_year,
                *args, **kwargs,
            )
            info["fold"] = f"{train_years[-1]}->{test_year}"
            results.append(info)
            suffix = f"{variant.lower()}_{train_years[-1]}_{str(test_year).replace(', ', '_')}"
            final_test.to_csv(OUTPUT_DIR / f"final_trades_enhanced_{suffix}.csv", index=False)
    return results


def print_results_table(results: List[dict]) -> None:
    print("\n" + "=" * 130)
    print("Walk-forward / Single-run results summary")
    print("=" * 130)
    header = (
        f"{'Fold/Run':<20}{'Variant':<18}{'Trades':>8}{'Win%':>10}"
        f"{'Total PnL':>15}{'PF':>8}{'Sharpe':>10}{'CAGR%':>10}{'MDD':>12}{'MAR':>10}"
    )
    print(header)
    print("-" * 130)
    for info in results:
        f = info["final"]
        fold = info.get("fold", "single")
        print(
            f"{fold:<20}{info['variant']:<18}{f['n_trades']:>8}{f['win_rate']:>10.2f}"
            f"{f['total_pnl']:>15,.0f}{f['profit_factor']:>8.2f}{f['sharpe']:>10.2f}"
            f"{f['cagr']:>10.2f}{f['mdd']:>12,.0f}{f['mar']:>10.2f}"
        )


# ---------------------------------------------------------------------------
# Slippage robustness sweep
# ---------------------------------------------------------------------------

def slippage_sweep(
    df: pd.DataFrame,
    ticks_list: List[int] = [0, 1, 2, 3],
    use_kde: bool = False,
    variant: str = "KDE-enhanced",
    train_years: tuple = (2019, 2020, 2021, 2022, 2023),
    val_year: int = 2024,
    test_year: int | tuple = (2025, 2026),
    target_horizon: int | None = None,
    **pipeline_kwargs,
) -> pd.DataFrame:
    """Run the enhanced pipeline under multiple slippage assumptions."""
    rows = []
    for ticks in ticks_list:
        df_slip = load_data(slippage_ticks=ticks, target_horizon=target_horizon)
        final_test, info = run_pipeline_enhanced(
            df_slip, use_kde, variant, train_years, val_year, test_year,
            **pipeline_kwargs,
        )
        row = {"slippage_ticks": ticks, **info["final"]}
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Enhanced 3-stage ML pipeline with sequential improvements")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--slippage-ticks", type=int, default=SLIPPAGE_TICKS)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--lstm-dropout", type=float, default=0.5)
    parser.add_argument("--lstm-lr", type=float, default=0.001)
    parser.add_argument("--lstm-epochs", type=int, default=30)
    parser.add_argument("--lstm-patience", type=int, default=5)
    parser.add_argument("--lstm-bidirectional", action="store_true")
    parser.add_argument("--stage3-type", type=str, default="ensemble", choices=["ensemble", "lstm", "xgb_cls", "xgb_reg", "none"])
    parser.add_argument("--stage1-metric", type=str, default="sharpe_x_pf")
    parser.add_argument("--stage2-metric", type=str, default="sharpe_x_pf")
    parser.add_argument("--stage3-metric", type=str, default="sharpe_x_pf")
    parser.add_argument("--stage1-threshold", type=float, default=None, help="Fixed stage1 threshold; <=0 skips stage1")
    parser.add_argument("--stage2-threshold", type=float, default=None, help="Fixed stage2 threshold; <=0 skips stage2")
    parser.add_argument("--stage3-threshold", type=float, default=None, help="Fixed stage3 threshold (where supported)")
    parser.add_argument("--stage3-min-trades", type=int, default=30, help="Minimum trades for stage3 threshold selection")
    parser.add_argument("--no-regime-aware", action="store_true", help="Disable regime-aware thresholds")
    parser.add_argument("--meta-filter", action="store_true", help="Use stage1 meta-filter")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable ensemble exit")
    parser.add_argument("--trailing-stop", action="store_true", help="Apply ATR trailing stop overlay")
    parser.add_argument("--no-fractional-kelly", action="store_true", help="Disable fractional Kelly sizing")
    parser.add_argument("--kelly-fraction", type=float, default=0.5)
    parser.add_argument("--no-drawdown-guard", action="store_true", help="Disable drawdown guard")
    parser.add_argument("--feature-selection-k", type=int, default=0, help="Select top-k features per stage by XGB gain (0 = use all features)")
    parser.add_argument("--slippage-sweep", action="store_true", help="Run 0/1/2/3 tick slippage sweep")
    parser.add_argument("--slippage-ticks-list", nargs="+", type=int, default=[0, 1, 2, 3], help="Slippage ticks to sweep")
    parser.add_argument("--target-horizon", type=int, default=None, help="Use fixed-horizon return label (e.g. 5) instead of strategy exit PnL")
    parser.add_argument("--use-kde", action="store_true", help="Attempt to use KDE features (requires ml_dataset_with_kde.csv)")
    args = parser.parse_args()

    lstm_config = {
        "lstm_units": args.lstm_units,
        "dropout": args.lstm_dropout,
        "learning_rate": args.lstm_lr,
        "epochs": args.lstm_epochs,
        "patience": args.lstm_patience,
        "use_bidirectional": args.lstm_bidirectional,
    }

    df = load_data(slippage_ticks=args.slippage_ticks, target_horizon=args.target_horizon)
    print(f"Loaded {len(df)} trades with slippage={args.slippage_ticks} tick(s)/side")
    if args.target_horizon is not None:
        print(f"Using fixed-horizon target: {args.target_horizon} bars")
    print(f"KDE features available: {df['_has_kde'].iloc[0]}")

    pipeline_kwargs = {
        "use_kde": args.use_kde,
        "variant": "KDE-enhanced" if args.use_kde else "Baseline",
        "train_years": (2019, 2020, 2021, 2022, 2023),
        "val_year": 2024,
        "test_year": (2025, 2026),
        "stage1_metric": args.stage1_metric,
        "stage2_metric": args.stage2_metric,
        "stage3_metric": args.stage3_metric,
        "stage1_threshold": args.stage1_threshold,
        "stage2_threshold": args.stage2_threshold,
        "stage3_threshold": args.stage3_threshold,
        "stage3_min_trades": args.stage3_min_trades,
        "stage3_type": args.stage3_type,
        "regime_aware": not args.no_regime_aware,
        "meta_filter": args.meta_filter,
        "ensemble_exit": not args.no_ensemble,
        "trailing_stop": args.trailing_stop,
        "fractional_kelly": not args.no_fractional_kelly,
        "kelly_fraction": args.kelly_fraction,
        "drawdown_guard": not args.no_drawdown_guard,
        "sequence_length": args.sequence_length,
        "feature_selection_k": args.feature_selection_k,
        "lstm_config": lstm_config,
    }

    # ``pipeline_kwargs`` contains run-level keys (use_kde/variant/train/val/test)
    # that must not be forwarded as duplicate keyword arguments.
    run_kwargs = {
        k: v for k, v in pipeline_kwargs.items()
        if k not in ("use_kde", "variant", "train_years", "val_year", "test_year")
    }

    if args.slippage_sweep:
        sweep = slippage_sweep(
            df, ticks_list=args.slippage_ticks_list, target_horizon=args.target_horizon, **pipeline_kwargs,
        )
        print("\nSlippage sweep results:")
        print(sweep.to_string(index=False))
        sweep.to_csv(OUTPUT_DIR / "slippage_sweep_enhanced.csv", index=False)
    elif args.walk_forward:
        wf_kwargs = {
            k: v for k, v in pipeline_kwargs.items()
            if k not in ("use_kde", "variant", "train_years", "val_year", "test_year")
        }
        results = walk_forward_validation_enhanced(df, **wf_kwargs)
        print_results_table(results)
    else:
        final_baseline, info_baseline = run_pipeline_enhanced(
            df, False, "Baseline", (2019, 2020, 2021, 2022, 2023), 2024, (2025, 2026),
            **run_kwargs,
        )
        final_baseline.to_csv(OUTPUT_DIR / "final_trades_baseline_enhanced.csv", index=False)
        results = [info_baseline]
        if args.use_kde or df["_has_kde"].iloc[0]:
            final_kde, info_kde = run_pipeline_enhanced(
                df, True, "KDE-enhanced", (2019, 2020, 2021, 2022, 2023), 2024, (2025, 2026),
                **run_kwargs,
            )
            final_kde.to_csv(OUTPUT_DIR / "final_trades_kde_enhanced.csv", index=False)
            results.append(info_kde)
        print_results_table(results)

    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
