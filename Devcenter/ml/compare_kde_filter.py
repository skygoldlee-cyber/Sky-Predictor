"""KDE feature 추가 전후 XGBoost 거래 필터 성능 비교.

- ml_dataset.csv 와 futures_1min_with_kde.csv 를 entry_time 기준으로 병합
- KDE 피처를 추가한 모델 vs 기존 피처만 사용한 모델 비교
- 시간 기반 훈련/검증/테스트 분할 (데이터 누설 방지)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_DIR = Path(__file__).parent / "ml_data"
OUTPUT_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_merged_dataset() -> pd.DataFrame:
    """Load ml_dataset and merge KDE features by entry_time."""
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df["entry_time"] = pd.to_datetime(df["entry_time"])

    kde_path = DATA_DIR / "futures_1min_with_kde.csv"
    if not kde_path.exists():
        raise FileNotFoundError(f"KDE features not found: {kde_path}")

    kde = pd.read_csv(kde_path)
    kde["timestamp"] = pd.to_datetime(kde["timestamp"])

    # Select only KDE metric columns + timestamp
    kde_cols = [c for c in kde.columns if c.endswith(("_kde_cdf", "_kde_pdf", "_kde_zscore", "_kde_left_tail", "_kde_right_tail", "_kde_ready"))]
    kde = kde[["timestamp"] + kde_cols]

    merged = df.merge(
        kde,
        left_on="entry_time",
        right_on="timestamp",
        how="left",
    )
    merged = merged.drop(columns=["timestamp"])
    return merged


def get_feature_sets() -> tuple:
    """Return baseline feature list and KDE-enhanced feature list."""
    baseline = [
        "entry_rsi", "entry_macd", "entry_macd_signal", "entry_macd_hist",
        "entry_atr", "entry_supertrend", "entry_supertrend_dir",
        "entry_ma20", "entry_ma60", "entry_bb_upper", "entry_bb_lower", "entry_bb_middle",
        "entry_hour", "entry_dayofweek", "entry_month",
        "volatility_regime", "trend_regime", "momentum_regime",
    ]
    kde_features = [
        "ret_log_1m_kde_cdf", "ret_log_1m_kde_pdf", "ret_log_1m_kde_zscore",
        "ret_log_1m_kde_left_tail", "ret_log_1m_kde_right_tail",
        "ret_log_5m_kde_cdf", "ret_log_5m_kde_pdf", "ret_log_5m_kde_zscore",
        "ret_log_5m_kde_left_tail", "ret_log_5m_kde_right_tail",
        # direction-aware KDE signals
        "kde_aligned_tail_1m", "kde_opposite_tail_1m", "kde_aligned_zscore_1m",
        "kde_aligned_tail_5m", "kde_opposite_tail_5m", "kde_aligned_zscore_5m",
    ]
    return baseline, baseline + kde_features


def add_direction_aware_kde_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add direction-aware KDE features aligned with trade direction.

    For long trades (direction=1), a low left tail / negative zscore is favorable.
    For short trades (direction=-1), a low right tail / positive zscore is favorable.
    """
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


def time_based_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Sort by entry_time and split into train/val/test."""
    df = df.sort_values("entry_time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def prepare_xy(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Extract X, y and fill NaN with 0."""
    X = df[feature_cols].copy()
    # Deduplicate columns defensively (XGBoost rejects duplicate names)
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]
    X = X.fillna(0).astype(float)
    y = df["is_win"].copy()
    return X, y


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBClassifier:
    """Train XGBoost classifier."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        reg_alpha=0.5,
        reg_lambda=2.0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
    }


def evaluate_pnl(df: pd.DataFrame, y_proba: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Evaluate total PnL and win rate at various probability thresholds."""
    rows = []
    for thr in thresholds:
        mask = y_proba >= thr
        filt = df[mask]
        if len(filt) == 0:
            continue
        rows.append({
            "threshold": thr,
            "n_trades": len(filt),
            "win_rate": filt["is_win"].mean() * 100,
            "total_pnl": filt["net_krw"].sum(),
            "avg_pnl": filt["net_krw"].mean(),
        })
    return pd.DataFrame(rows)


def print_feature_importance(model: xgb.XGBClassifier, feature_cols: list, top_n: int = 15) -> None:
    """Print top feature importances."""
    n_imp = len(model.feature_importances_)
    importance = pd.DataFrame({
        "feature": feature_cols[:n_imp],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)
    print(importance.to_string(index=False))


def main() -> None:
    print("=" * 80)
    print("KDE Feature Impact Comparison")
    print("=" * 80)

    df = load_merged_dataset()
    print(f"\nml_dataset rows: {len(df)}")
    print(f"entry_time range: {df['entry_time'].min()} ~ {df['entry_time'].max()}")

    df = add_direction_aware_kde_features(df)

    # Only use rows where KDE features exist
    df_kde = df[df["ret_log_1m_kde_ready"].notna()].copy()
    print(f"Rows with KDE features: {len(df_kde)} ({len(df_kde)/len(df)*100:.1f}%)")
    print(f"KDE period: {df_kde['entry_time'].min()} ~ {df_kde['entry_time'].max()}")

    baseline_features, kde_features = get_feature_sets()

    # Check availability
    available_baseline = [c for c in baseline_features if c in df_kde.columns]
    missing_baseline = set(baseline_features) - set(available_baseline)
    if missing_baseline:
        print(f"Warning: missing baseline features {missing_baseline}")
    available_kde = [c for c in kde_features if c in df_kde.columns]
    missing_kde = set(kde_features) - set(available_kde)
    if missing_kde:
        print(f"Warning: missing KDE features {missing_kde}")

    baseline_features = available_baseline
    kde_features = baseline_features + available_kde

    train_df, val_df, test_df = time_based_split(df_kde)
    print(f"\nSplit sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    X_train_b, y_train = prepare_xy(train_df, baseline_features)
    X_val_b, y_val = prepare_xy(val_df, baseline_features)
    X_test_b, y_test = prepare_xy(test_df, baseline_features)

    X_train_k, _ = prepare_xy(train_df, kde_features)
    X_val_k, _ = prepare_xy(val_df, kde_features)
    X_test_k, _ = prepare_xy(test_df, kde_features)

    # Baseline model
    print("\n[1/4] Training baseline XGBoost...")
    model_b = train_xgboost(X_train_b, y_train, X_val_b, y_val)
    y_pred_b = model_b.predict(X_test_b)
    y_proba_b = model_b.predict_proba(X_test_b)[:, 1]
    metrics_b = evaluate_classification(y_test, y_pred_b, y_proba_b)

    # KDE model
    print("[2/4] Training KDE-enhanced XGBoost...")
    model_k = train_xgboost(X_train_k, y_train, X_val_k, y_val)
    y_pred_k = model_k.predict(X_test_k)
    y_proba_k = model_k.predict_proba(X_test_k)[:, 1]
    metrics_k = evaluate_classification(y_test, y_pred_k, y_proba_k)

    # Print classification metrics
    print("\n[3/4] Classification Metrics on Test Set")
    print(f"{'Metric':<15}{'Baseline':>12}{'KDE':>12}{'Diff':>12}")
    print("-" * 52)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        diff = metrics_k[k] - metrics_b[k]
        print(f"{k:<15}{metrics_b[k]:>12.4f}{metrics_k[k]:>12.4f}{diff:>+12.4f}")

    # PnL evaluation
    print("\n[4/4] PnL-based Evaluation on Test Set")
    thresholds = np.arange(0.4, 0.9, 0.05)
    pnl_b = evaluate_pnl(test_df, y_proba_b, thresholds)
    pnl_k = evaluate_pnl(test_df, y_proba_k, thresholds)

    print("\nBaseline PnL by threshold:")
    print(pnl_b.to_string(index=False))
    print("\nKDE-enhanced PnL by threshold:")
    print(pnl_k.to_string(index=False))

    # Feature importance
    print("\nBaseline Top Features:")
    print_feature_importance(model_b, baseline_features)
    print("\nKDE Top Features:")
    print_feature_importance(model_k, kde_features)

    # Save models
    model_b.save_model(str(OUTPUT_DIR / "trade_filter_xgboost_baseline.json"))
    model_k.save_model(str(OUTPUT_DIR / "trade_filter_xgboost_kde.json"))
    print("\nModels saved:")
    print(f"  {OUTPUT_DIR / 'trade_filter_xgboost_baseline.json'}")
    print(f"  {OUTPUT_DIR / 'trade_filter_xgboost_kde.json'}")

    # Save merged dataset with KDE features
    merged_path = DATA_DIR / "ml_dataset_with_kde.csv"
    df.to_csv(merged_path, index=False)
    print(f"\nMerged dataset saved: {merged_path}")

    return metrics_b, metrics_k, pnl_b, pnl_k


if __name__ == "__main__":
    main()
