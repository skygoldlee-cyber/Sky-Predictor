# -*- coding: utf-8 -*-
"""
최고 피봇 파라미터의 구간별 성과 분석 (diagnose_windows)
"""
import sys
import os
import contextlib
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

import pandas as pd
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg

DB_PATH = "c:/Project/SkyPredictor/Devcenter/data/duckdb/market_data.duckdb"
OUTPUT_LOG = Path(__file__).parent / "pivot_diagnose_windows.log"

BT = pv.BacktestConfig(
    multiplier=250_000,
    commission_pct_per_side=0.00015,
    slippage_ticks_per_side=1.0,
    tick_size=0.05,
    entry_on="next_open",
    annualization=252.0,
    intraday_only=True,
    session_boundary_hour=8,
    direction_mode="both",
)


def load_data():
    df_1min = pv.load_data_by_date(
        DB_PATH, "futures_1min", start="2025-06-24", end="2026-06-18"
    )
    df_1min = pv.filter_day_session(df_1min, start="08:45", end="15:45")
    df_5min = df_1min.resample("5min").agg({
        "OPEN": "first", "HIGH": "max", "LOW": "min",
        "CLOSE": "last", "VOLUME": "sum",
    }).dropna()
    df_5min = pv.compute_indicators(df_5min)
    return df_5min


def long_or_flat_windowed(df: pd.DataFrame, n_windows: int = 6):
    bt = pv.BacktestConfig(
        multiplier=250_000,
        commission_pct_per_side=0.00015,
        slippage_ticks_per_side=1.0,
        tick_size=0.05,
        entry_on="next_open",
        annualization=252.0,
        intraday_only=True,
        session_boundary_hour=8,
        direction_mode="long_only",
    )
    daily_full = rg.to_daily(df, bt.session_boundary_hour)
    signal = rg.daily_regime_signal(
        daily_full, regime_method="adx", ma_short=20, ma_long=60,
        adx_threshold=25.0, allow_short=False
    )
    n = len(df)
    w = n // n_windows
    rows = []
    for i in range(n_windows):
        s = i * w
        e = (i + 1) * w if i < n_windows - 1 else n
        seg = df.iloc[s:e]
        res = rg.regime_intraday_daily(seg, bt, signal=signal)
        rows.append({
            "window": i + 1,
            "start": seg.index[0], "end": seg.index[-1],
            "n_trades": res.n_trades,
            "sharpe": round(res.sharpe_daily, 3),
            "pnl_krw": round(res.total_pnl_krw),
            "maxdd_krw": round(res.max_drawdown_krw),
        })
    return pd.DataFrame(rows)


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        df = load_data()
        cases = [
            ("In-sample 최고 (both)", "both", pv.HybridAdaptivePivotConfig(
                base_pct=0.5, base_multiplier=2.0, atr_weight=0.3, confirmation_bars=2
            ), pv.FilterConfig(
                enabled=True, min_wave_pct=0.3, min_pivot_interval_bars=20,
                st_distance_threshold=0.1, adx_hold_threshold=15.0
            )),
            ("Train 최고 (long_only)", "long_only", pv.HybridAdaptivePivotConfig(
                base_pct=0.5, base_multiplier=2.5, atr_weight=0.3, confirmation_bars=2
            ), pv.FilterConfig(
                enabled=True, min_wave_pct=0.3, min_pivot_interval_bars=30,
                st_distance_threshold=0.1, adx_hold_threshold=15.0
            )),
        ]
        for name, mode, pcfg, fcfg in cases:
            log.write(f"\n=== {name} ===\n")
            bt = pv.BacktestConfig(**BT.__dict__)
            bt.direction_mode = mode
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                out = pv.diagnose_windows(df, pcfg, fcfg, bt, n_windows=6, daily_reset=True, verbose=False)
            log.write(out.to_string(index=False) + "\n")
            log.write(f"수익 구간: {(out['pnl_krw'] > 0).sum()}/{len(out)}\n")
            log.write(f"합계 PnL: {int(out['pnl_krw'].sum()):,}\n")
            log.flush()

        log.write("\n=== 롱-또는-플랫 윈도우 ===\n")
        out_lf = long_or_flat_windowed(df, n_windows=6)
        log.write(out_lf.to_string(index=False) + "\n")
        log.write(f"수익 구간: {(out_lf['pnl_krw'] > 0).sum()}/{len(out_lf)}\n")
        log.write(f"합계 PnL: {int(out_lf['pnl_krw'].sum()):,}\n")
        log.flush()
        print(f"Saved to {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
