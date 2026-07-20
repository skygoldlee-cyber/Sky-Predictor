"""Generate multi-timeframe KDE features with the best single-tf params and run the full 3-stage pipeline.

This script reuses the best 5m KDE configuration saved by `optimize_kde_params_5min.py`
and adds 15m/30m/60m KDE features on top of it. It then compares Baseline vs KDE on the
held-out test period.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Devcenter.ml.generate_kde_features_5min import (
    DEFAULT_DATA_PATH,
    build_multi_tf_kde_features,
    load_5min_txt,
)
from Devcenter.ml.ml_full_pipeline_kde import (
    DATA_DIR,
    OUTPUT_DIR,
    SLIPPAGE_TICKS,
    apply_trading_costs,
    load_data,
    run_pipeline,
)

_logger = logging.getLogger(__name__)


def merge_kde_with_trades(kde_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Merge 5m KDE features with trade entries using merge_asof (no lookahead).

    KDE features are computed at the close of bar t and tagged with feature_time=t+5m,
    so an entry at the open of bar t+1 uses only information available at bar t close.
    """
    trades = trades_df.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])

    kde_cols = [c for c in kde_df.columns if "_kde_" in c]
    # If feature_time already exists, use it; otherwise compute t+5m from timestamp.
    if "feature_time" in kde_df.columns:
        kde = kde_df[["feature_time"] + kde_cols].copy()
        kde["feature_time"] = pd.to_datetime(kde["feature_time"])
    else:
        kde = kde_df[["timestamp"] + kde_cols].copy()
        kde["timestamp"] = pd.to_datetime(kde["timestamp"])
        kde["feature_time"] = kde["timestamp"] + pd.Timedelta(minutes=5)
        kde = kde.drop(columns=["timestamp"])

    trades = trades.sort_values("entry_time").reset_index(drop=True)
    kde = kde.sort_values("feature_time").reset_index(drop=True)

    merged = pd.merge_asof(
        trades,
        kde,
        left_on="entry_time",
        right_on="feature_time",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.drop(columns=["feature_time"])
    return merged


def add_direction_aware_kde_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add direction-aware KDE tail/zscore features for every available timeframe."""
    df = df.copy()
    direction = df["direction"].fillna(0).astype(float)
    for tf in ["5m", "15m", "30m", "60m"]:
        lt = f"ret_log_{tf}_kde_left_tail"
        rt = f"ret_log_{tf}_kde_right_tail"
        zs = f"ret_log_{tf}_kde_zscore"
        if lt not in df.columns or rt not in df.columns:
            continue
        df[f"kde_aligned_tail_{tf}"] = np.where(direction == 1, df[lt], df[rt])
        df[f"kde_opposite_tail_{tf}"] = np.where(direction == 1, df[rt], df[lt])
        if zs in df.columns:
            df[f"kde_aligned_zscore_{tf}"] = np.where(direction == 1, -df[zs], df[zs])
    return df


def _coerce_bandwidth(bw: str | float) -> str | float:
    if isinstance(bw, (int, float)):
        return float(bw)
    try:
        return float(bw)
    except ValueError:
        return bw


def _test_sharpe(final_df: pd.DataFrame) -> float:
    if "net_krw" not in final_df.columns or len(final_df) < 2:
        return 0.0
    s = final_df["net_krw"].std()
    if s and s > 0:
        return float(np.sqrt(len(final_df)) * final_df["net_krw"].mean() / s)
    return 0.0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    params_path = OUTPUT_DIR / "kde_best_params_5min_multi.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Best params not found: {params_path}")

    best = json.loads(params_path.read_text())
    _logger.info("Using best single-tf params: %s", best)

    _logger.info("Loading 5m OHLC data")
    df_5m = load_5min_txt(DEFAULT_DATA_PATH)

    _logger.info("Building multi-timeframe KDE features (5m, 15m, 30m, 60m)")
    kde_df = build_multi_tf_kde_features(
        df_5m,
        timeframes_minutes=(5, 15, 30, 60),
        window=int(best["window"]),
        min_samples=max(50, int(best["window"]) // 6),
        bandwidth=_coerce_bandwidth(best["bandwidth"]),
        grid_points=int(best["grid_points"]),
        refit_every=int(best["refit_every"]),
        use_regime=True,
    )

    trades = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    trades = apply_trading_costs(trades, slippage_ticks=SLIPPAGE_TICKS)

    merged = merge_kde_with_trades(kde_df, trades)
    merged = add_direction_aware_kde_features(merged)
    merged = merged.drop_duplicates(subset=["entry_time"], keep="first")

    out_path = DATA_DIR / "ml_dataset_with_kde.csv"
    merged.to_csv(out_path, index=False)
    _logger.info("Saved %s", out_path)

    df = load_data(slippage_ticks=SLIPPAGE_TICKS)
    final_b, info_b = run_pipeline(
        df, False, "Baseline_MultiTF",
        (2019, 2020, 2021, 2022, 2023), 2024, (2025, 2026),
    )
    final_k, info_k = run_pipeline(
        df, True, "KDE_MultiTF",
        (2019, 2020, 2021, 2022, 2023), 2024, (2025, 2026),
    )

    final_b.to_csv(OUTPUT_DIR / "final_trades_baseline_multitf.csv", index=False)
    final_k.to_csv(OUTPUT_DIR / "final_trades_kde_multitf.csv", index=False)

    print("\n=== Multi-timeframe KDE full pipeline comparison (test 2025-2026) ===")
    final_dfs = {"Baseline_MultiTF": final_b, "KDE_MultiTF": final_k}
    for info in [info_b, info_k]:
        f = info["final"]
        df = final_dfs[info["variant"]]
        print(
            f"{info['variant']:<20} trades={f['n_trades']:<5} win%={f['win_rate']:.2f} "
            f"total_pnl={f['total_pnl']:>15,.0f} avg_pnl={f['avg_pnl']:>12,.0f} "
            f"test_sharpe={_test_sharpe(df):.2f}"
        )


if __name__ == "__main__":
    main()
