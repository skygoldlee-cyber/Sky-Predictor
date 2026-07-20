"""Optimize KDE window/bandwidth for 5-minute return features.

The grid search uses the XGBoost trade-filter stage only (fast proxy) and
selects the configuration by validation ROC AUC.  The best configuration is
also evaluated on the held-out test set and saved back to
``ml_data/ml_dataset_with_kde.csv`` so the full 3-stage pipeline can use it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indicators.return_distribution import build_kde_features

# Import helpers from the full-pipeline module; it loads TensorFlow at import
# time, which is acceptable here because we optionally run the full pipeline.
from Devcenter.ml.ml_full_pipeline_kde import (
    BASELINE_FEATURES,
    DATA_DIR,
    OUTPUT_DIR,
    SLIPPAGE_TICKS,
    TICK_SIZE,
    apply_trading_costs,
    classification_metrics,
    get_feature_cols,
    load_data,
    run_pipeline,
    select_threshold_by_pnl,
    train_xgboost_filter,
)
from Devcenter.ml.generate_kde_features_5min import (
    DEFAULT_DATA_PATH as DEFAULT_5MIN_PATH,
    build_5min_kde_features,
    build_multi_tf_kde_features,
    load_5min_txt,
)

_logger = logging.getLogger(__name__)


def _coerce_bandwidth(bw: str | float) -> str | float:
    if isinstance(bw, (int, float)):
        return float(bw)
    try:
        return float(bw)
    except ValueError:
        return bw


def merge_kde_with_trades(kde_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Merge 5m KDE features with trade entries using merge_asof (no lookahead)."""
    trades = trades_df.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])

    kde_cols = [c for c in kde_df.columns if "_kde_" in c]
    kde = kde_df[["timestamp"] + kde_cols].copy()
    kde["timestamp"] = pd.to_datetime(kde["timestamp"])

    trades = trades.sort_values("entry_time").reset_index(drop=True)
    kde = kde.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        trades,
        kde,
        left_on="entry_time",
        right_on="timestamp",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.drop(columns=["timestamp"])
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


def build_kde_for_params(
    df_5m: pd.DataFrame,
    window: int,
    bandwidth: str | float,
    grid_points: int,
    refit_every: int,
) -> pd.DataFrame:
    """Generate 5m KDE features for a parameter set."""
    bw = _coerce_bandwidth(bandwidth)
    return build_5min_kde_features(
        df_5m,
        window=window,
        min_samples=max(50, window // 6),
        bandwidth=bw,
        grid_points=grid_points,
        refit_every=refit_every,
    )


def evaluate_params(
    kde_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    train_years: Tuple[int, ...] = (2019, 2020, 2021, 2022, 2023),
    val_year: int = 2024,
    min_val_trades: int = 20,
) -> dict:
    """Train XGB filter with baseline+KDE features and report validation metrics."""
    merged = merge_kde_with_trades(kde_df, trades_df)
    merged = add_direction_aware_kde_features(merged)

    ready_col = "ret_log_5m_kde_ready"
    if ready_col in merged.columns:
        merged = merged[merged[ready_col].notna() & (merged[ready_col] == True)].copy()

    if len(merged) < 100:
        return {"val_roc_auc": -1.0, "val_pnl": -np.inf, "val_sharpe": -np.inf, "val_win_rate": 0.0, "n_trades": len(merged)}

    merged["year"] = pd.to_datetime(merged["entry_time"]).dt.year
    train = merged[merged["year"].isin(train_years)].copy()
    val = merged[merged["year"] == val_year].copy()

    if len(train) < 50 or len(val) < 20:
        return {"val_roc_auc": -1.0, "val_pnl": -np.inf, "val_sharpe": -np.inf, "val_win_rate": 0.0, "n_trades": len(merged)}

    base_cols = [c for c in BASELINE_FEATURES if c in merged.columns]
    kde_cols = [c for c in merged.columns if c.startswith("ret_log_5m_kde_") or c.startswith("kde_")]
    feature_cols = base_cols + kde_cols

    model = train_xgboost_filter(train, val, feature_cols)
    X_val = val[feature_cols].fillna(0).astype(float)
    y_val = val["is_win"]
    proba_val = model.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, proba_val)) if len(np.unique(y_val)) > 1 else -1.0

    # Pick threshold on validation PnL and record validation PnL / Sharpe
    threshold = select_threshold_by_pnl(val, proba_val, min_trades=min_val_trades)
    val_filt = val[proba_val >= threshold]
    n_val = len(val_filt)
    if n_val >= min_val_trades:
        val_pnl = float(val_filt["net_krw"].sum())
        val_win_rate = float(val_filt["is_win"].mean())
        std = val_filt["net_krw"].std()
        val_sharpe = float(np.sqrt(n_val) * val_filt["net_krw"].mean() / std) if std and std > 0 else -np.inf
    else:
        val_pnl = -np.inf
        val_win_rate = 0.0
        val_sharpe = -np.inf

    return {
        "val_roc_auc": val_auc,
        "val_pnl": val_pnl,
        "val_sharpe": val_sharpe,
        "val_win_rate": val_win_rate,
        "threshold": float(threshold),
        "n_trades": len(merged),
    }


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series to [0, 1]."""
    finite = series.replace([-np.inf, np.inf], np.nan)
    min_v = finite.min()
    max_v = finite.max()
    if max_v == min_v or pd.isna(min_v) or pd.isna(max_v):
        return pd.Series(0.0, index=series.index)
    return (finite - min_v) / (max_v - min_v)


def _pareto_front(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return non-dominated rows for maximizing all listed columns."""
    vals = df[cols].to_numpy(dtype=float)
    n = len(vals)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                dominated[i] = True
                break
    return df[~dominated].copy()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Optimize 5m KDE parameters")
    parser.add_argument("--data-5m", type=Path, default=DEFAULT_5MIN_PATH)
    parser.add_argument("--slippage-ticks", type=int, default=SLIPPAGE_TICKS)
    parser.add_argument(
        "--windows", type=int, nargs="+", default=[900, 1000, 1100, 1200]
    )
    parser.add_argument(
        "--bandwidths", nargs="+", default=["scott", "silverman"]
    )
    parser.add_argument(
        "--grid-points-list", type=int, nargs="+", default=[1024, 2048]
    )
    parser.add_argument(
        "--refit-every-list", type=int, nargs="+", default=[50, 100]
    )
    parser.add_argument(
        "--train-years", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023]
    )
    parser.add_argument("--val-year", type=int, default=2024)
    parser.add_argument("--test-years", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument(
        "--criterion", choices=["auc", "pnl", "sharpe", "multi"], default="multi",
        help="Metric used to select the best KDE parameter set."
    )
    parser.add_argument(
        "--weights", type=float, nargs=3, default=[0.2, 0.4, 0.4],
        help="Weights for [AUC, PnL, Sharpe] when criterion=multi."
    )
    parser.add_argument("--min-val-trades", type=int, default=20)
    parser.add_argument(
        "--multi-tf", action="store_true",
        help="Use multi-timeframe (5m, 15m, 30m, 60m) KDE features for the final dataset."
    )
    parser.add_argument("--run-full-pipeline", action="store_true")
    args = parser.parse_args()

    _logger.info("Loading 5m OHLC data from %s", args.data_5m)
    df_5m = load_5min_txt(args.data_5m)
    _logger.info("Loaded %d 5m rows", len(df_5m))

    _logger.info("Loading base trade dataset")
    trades = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    trades = apply_trading_costs(trades, slippage_ticks=args.slippage_ticks)

    bandwidths = [_coerce_bandwidth(bw) for bw in args.bandwidths]
    results = []

    total = len(args.windows) * len(bandwidths) * len(args.grid_points_list) * len(args.refit_every_list)
    _logger.info("Starting grid search with %d candidates", total)

    for window, bandwidth, grid_points, refit_every in itertools.product(
        args.windows, bandwidths, args.grid_points_list, args.refit_every_list
    ):
        _logger.info(
            "Evaluating window=%d bandwidth=%s grid=%d refit=%d",
            window, bandwidth, grid_points, refit_every,
        )
        try:
            kde_df = build_kde_for_params(
                df_5m,
                window=window,
                bandwidth=bandwidth,
                grid_points=grid_points,
                refit_every=refit_every,
            )
            metrics = evaluate_params(
                kde_df, trades, tuple(args.train_years), args.val_year, args.min_val_trades
            )
            _logger.info(
                "  -> val_auc=%.4f val_pnl=%.0f val_sharpe=%.3f trades=%d",
                metrics["val_roc_auc"],
                metrics["val_pnl"],
                metrics["val_sharpe"],
                metrics["n_trades"],
            )
            results.append({
                "window": window,
                "bandwidth": str(bandwidth),
                "grid_points": grid_points,
                "refit_every": refit_every,
                "val_roc_auc": metrics["val_roc_auc"],
                "val_pnl": metrics["val_pnl"],
                "val_sharpe": metrics["val_sharpe"],
                "val_win_rate": metrics["val_win_rate"],
                "threshold": metrics["threshold"],
                "n_trades": metrics["n_trades"],
            })
        except Exception as exc:
            _logger.warning("Failed window=%d bandwidth=%s grid=%d refit=%d: %s", window, bandwidth, grid_points, refit_every, exc)
            results.append({
                "window": window,
                "bandwidth": str(bandwidth),
                "grid_points": grid_points,
                "refit_every": refit_every,
                "val_roc_auc": -1.0,
                "val_pnl": -np.inf,
                "val_sharpe": -np.inf,
                "val_win_rate": 0.0,
                "threshold": 0.5,
                "n_trades": 0,
            })

    results_df = pd.DataFrame(results)
    eligible = results_df[results_df["n_trades"] >= args.min_val_trades].copy()

    # Per-criterion bests for reporting
    best_auc = results_df.loc[results_df["val_roc_auc"].idxmax()].copy()
    best_pnl = eligible.loc[eligible["val_pnl"].idxmax()].copy() if len(eligible) else results_df.loc[results_df["val_pnl"].idxmax()].copy()
    best_sharpe = eligible.loc[eligible["val_sharpe"].idxmax()].copy() if len(eligible) else results_df.loc[results_df["val_sharpe"].idxmax()].copy()

    # Multi-objective weighted score
    if args.criterion == "multi" and len(eligible):
        norm_auc = _normalize(eligible["val_roc_auc"])
        norm_pnl = _normalize(eligible["val_pnl"])
        norm_sharpe = _normalize(eligible["val_sharpe"])
        w_auc, w_pnl, w_sharpe = args.weights
        eligible["multi_score"] = w_auc * norm_auc + w_pnl * norm_pnl + w_sharpe * norm_sharpe
        best = eligible.loc[eligible["multi_score"].idxmax()].copy()
    elif args.criterion == "auc":
        best = best_auc
    elif args.criterion == "pnl":
        best = best_pnl
    else:  # sharpe
        best = best_sharpe

    pareto = _pareto_front(eligible if len(eligible) else results_df, ["val_roc_auc", "val_pnl", "val_sharpe"])

    print("\n=== Best by each validation criterion ===")
    print(f"AUC:    {best_auc.to_dict()}")
    print(f"PnL:    {best_pnl.to_dict()}")
    print(f"Sharpe: {best_sharpe.to_dict()}")

    if args.criterion == "multi":
        print(f"\nSelected multi-objective best (weights={args.weights}): {best.to_dict()}")
        print("\n=== Top candidates by multi-objective score ===")
        print(eligible.sort_values("multi_score", ascending=False).head(10).to_string(index=False))
    else:
        sort_col = f"val_{args.criterion}"
        print(f"\n=== Grid search results sorted by {args.criterion} ===")
        print(results_df.sort_values(sort_col, ascending=False).to_string(index=False))
        print(f"\nBest config ({args.criterion}): {best.to_dict()}")

    print("\n=== Pareto frontier (AUC, PnL, Sharpe) ===")
    print(pareto.sort_values("val_pnl", ascending=False).to_string(index=False))

    # Generate best KDE features and overwrite ml_dataset_with_kde.csv
    if args.multi_tf:
        _logger.info("Building best multi-timeframe KDE features for full dataset")
        best_kde_df = build_multi_tf_kde_features(
            df_5m,
            timeframes_minutes=(5, 15, 30, 60),
            window=int(best["window"]),
            min_samples=max(50, int(best["window"]) // 6),
            bandwidth=_coerce_bandwidth(best["bandwidth"]),
            grid_points=int(best["grid_points"]),
            refit_every=int(best["refit_every"]),
        )
    else:
        _logger.info("Building best single-timeframe KDE features for full dataset")
        best_kde_df = build_kde_for_params(
            df_5m,
            window=int(best["window"]),
            bandwidth=_coerce_bandwidth(best["bandwidth"]),
            grid_points=int(best["grid_points"]),
            refit_every=int(best["refit_every"]),
        )
    merged_best = merge_kde_with_trades(best_kde_df, trades)
    merged_best = add_direction_aware_kde_features(merged_best)
    merged_best = merged_best.drop_duplicates(subset=["entry_time"], keep="first")
    out_path = DATA_DIR / "ml_dataset_with_kde.csv"
    merged_best.to_csv(out_path, index=False)
    _logger.info("Saved %s", out_path)

    # Save best params and full results
    params_path = OUTPUT_DIR / f"kde_best_params_5min_{args.criterion}.json"
    params_path.write_text(json.dumps(best.to_dict(), indent=2, default=str))
    _logger.info("Saved best params to %s", params_path)
    results_path = OUTPUT_DIR / f"kde_grid_results_5min_{args.criterion}.csv"
    results_df.to_csv(results_path, index=False)
    _logger.info("Saved grid results to %s", results_path)

    # Optional full pipeline comparison with best KDE config
    if args.run_full_pipeline:
        print("\nRunning full 3-stage pipeline with optimized KDE features...")
        df = load_data(slippage_ticks=args.slippage_ticks)
        final_b, info_b = run_pipeline(
            df, False, "Baseline_Opt",
            tuple(args.train_years), args.val_year, tuple(args.test_years),
        )
        final_k, info_k = run_pipeline(
            df, True, "KDE_Opt",
            tuple(args.train_years), args.val_year, tuple(args.test_years),
        )
        final_b.to_csv(OUTPUT_DIR / "final_trades_baseline_opt.csv", index=False)
        final_k.to_csv(OUTPUT_DIR / "final_trades_kde_opt.csv", index=False)

        print(f"\n=== Full pipeline comparison (optimized by {args.criterion}) ===")
        for info in [info_b, info_k]:
            f = info["final"]
            print(
                f"{info['variant']:<20} trades={f['n_trades']:<5} win%={f['win_rate']:.2f} "
                f"total_pnl={f['total_pnl']:>15,.0f} avg_pnl={f['avg_pnl']:>12,.0f}"
            )


if __name__ == "__main__":
    main()
