"""Generate multi-timeframe KDE features with the best single-tf params and run the full 3-stage pipeline.

This script reuses the best 5m KDE configuration saved by `optimize_kde_params_5min.py`
and adds 15m/30m/60m KDE features on top of it. It then compares Baseline vs KDE on the
held-out test period.
"""

from __future__ import annotations

import argparse
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


def _pipeline_metrics(df: pd.DataFrame, info: dict) -> dict:
    f = info["final"]
    return {
        "variant": info["variant"],
        "trades": int(f["n_trades"]),
        "win_pct": float(f["win_rate"]),
        "total_pnl": float(f["total_pnl"]),
        "avg_pnl": float(f["avg_pnl"]),
        "test_sharpe": float(_test_sharpe(df)),
    }


def _run_single_experiment(skip_kde_generation: bool = False) -> Tuple[dict, dict]:
    if not skip_kde_generation:
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
    return (final_b, info_b), (final_k, info_k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-timeframe KDE pipeline once or repeatedly."
    )
    parser.add_argument(
        "--skip-kde-generation",
        action="store_true",
        help="Reuse existing ml_dataset_with_kde.csv instead of regenerating KDE features.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of repetitions (useful for seed-stability checks).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results_b: list[dict] = []
    results_k: list[dict] = []
    for run in range(1, args.n_runs + 1):
        print(f"\n========== Run {run}/{args.n_runs} ==========")
        (final_b, info_b), (final_k, info_k) = _run_single_experiment(
            skip_kde_generation=args.skip_kde_generation
        )
        results_b.append(_pipeline_metrics(final_b, info_b))
        results_k.append(_pipeline_metrics(final_k, info_k))

        final_b.to_csv(OUTPUT_DIR / f"final_trades_baseline_multitf_run{run}.csv", index=False)
        final_k.to_csv(OUTPUT_DIR / f"final_trades_kde_multitf_run{run}.csv", index=False)

    print("\n=== Multi-timeframe KDE full pipeline comparison (test 2025-2026) ===")
    for metrics_b, metrics_k in zip(results_b, results_k):
        for m in [metrics_b, metrics_k]:
            print(
                f"{m['variant']:<20} trades={m['trades']:<5} win%={m['win_pct']:.2f} "
                f"total_pnl={m['total_pnl']:>15,.0f} avg_pnl={m['avg_pnl']:>12,.0f} "
                f"test_sharpe={m['test_sharpe']:.2f}"
            )

    if args.n_runs > 1:
        import statistics
        for variant, results in [("Baseline_MultiTF", results_b), ("KDE_MultiTF", results_k)]:
            sharpes = [r["test_sharpe"] for r in results]
            pnls = [r["total_pnl"] for r in results]
            trades = [r["trades"] for r in results]
            print(
                f"\n{variant} stability over {args.n_runs} runs: "
                f"Sharpe mean={statistics.mean(sharpes):.2f} std={statistics.stdev(sharpes):.2f} | "
                f"PnL mean={statistics.mean(pnls):,.0f} std={statistics.stdev(pnls):,.0f} | "
                f"trades mean={statistics.mean(trades):.0f}"
            )


if __name__ == "__main__":
    main()
