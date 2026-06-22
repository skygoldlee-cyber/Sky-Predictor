# -*- coding: utf-8 -*-
"""
현재 DB(2025-06-25 ~ 2026-06-19)에서 하락장 구간을 추출하여
피봇 반전 로직을 검증하는 스크립트.

- 하락장 정의: 일봉 종가 < 20일 이동평균 and 일봉 수익률 < 0
- 피봇 반전 전략: in-sample 최적 파라미터로 장중 피봇 반전 매매
- 베이스라인: 롱-또는-플랫 (동일 기간)
- 결과: 전체 / 하락장 / 비하락장 구간으로 분리 비교
"""
import sys
import os
import math
import contextlib
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg

DB_PATH = "c:/Project/SkyPredictor/Devcenter/data/duckdb/market_data.duckdb"
OUTPUT_LOG = Path(__file__).parent / "pivot_bear_market_test.log"

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

# In-sample 최고 파라미터 (pivot_viability_traintest.py 기준)
PCFG = pv.HybridAdaptivePivotConfig(
    base_pct=0.5,
    base_multiplier=2.0,
    atr_weight=0.3,
    confirmation_bars=2,
)
FCFG = pv.FilterConfig(
    enabled=True,
    min_wave_pct=0.3,
    min_pivot_interval_bars=20,
    st_distance_threshold=0.1,
    adx_hold_threshold=15.0,
)


def log_and_print(msg: str, log_file):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    log_file.write(line + "\n")
    log_file.flush()
    print(line)


def load_data():
    df_1min = pv.load_data_by_date(
        DB_PATH, "futures_1min", start="2025-06-24", end="2026-06-19"
    )
    df_1min = pv.filter_day_session(df_1min, start="08:45", end="15:45")
    df_5min = df_1min.resample("5min").agg({
        "OPEN": "first", "HIGH": "max", "LOW": "min",
        "CLOSE": "last", "VOLUME": "sum",
    }).dropna()
    df_5min = pv.compute_indicators(df_5min)
    return df_5min


def identify_bearish_days(daily: pd.DataFrame, ma_window: int = 20):
    """하락장/조정 날짜 식별 - 여러 정의를 동시에 제공."""
    daily = daily.copy()
    daily["MA20"] = daily["CLOSE"].rolling(ma_window).mean()
    daily["RETURN"] = daily["CLOSE"].pct_change()
    daily["MA_BEAR"] = daily["CLOSE"] < daily["MA20"]
    daily["DOWN_DAY"] = daily["RETURN"] < 0
    daily["STRONG_DOWN"] = daily["RETURN"] < -0.01

    sets = {
        "MA20_bear": set(daily[daily["MA_BEAR"]].index.date),
        "down_day": set(daily[daily["DOWN_DAY"]].index.date),
        "MA20_bear_and_down": set(daily[daily["MA_BEAR"] & daily["DOWN_DAY"]].index.date),
        "strong_down": set(daily[daily["STRONG_DOWN"]].index.date),
    }
    return sets, daily


def run_pivot(df: pd.DataFrame, direction_mode: str):
    bt = pv.BacktestConfig(**BT.__dict__)
    bt.direction_mode = direction_mode
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        pivots = pv.detect_pivots_daily(df, PCFG, FCFG, bt.session_boundary_hour)
    return pv.backtest(df, pivots, bt)


def run_long_or_flat(df: pd.DataFrame):
    return rg.regime_intraday_daily(
        df, BT, regime_method="adx", ma_short=20, ma_long=60, adx_threshold=25.0
    )


def filter_trades_by_exit_dates(trades: pd.DataFrame, dates: set):
    if trades is None or trades.empty:
        return trades
    d = pd.to_datetime(trades["exit_time"]).dt.date
    return trades[d.isin(dates)].copy()


def compute_metrics(tdf: pd.DataFrame, cfg: pv.BacktestConfig):
    """trades DataFrame로부터 성과 지표를 재계산."""
    if tdf is None or tdf.empty:
        return pv.BacktestResult(trades=pd.DataFrame())
    net = tdf["net_pts"]
    wins = net[net > 0]
    losses = net[net < 0]
    total_pts = float(net.sum())
    n_trades = int(len(tdf))
    win_rate = float((net > 0).mean() * 100)
    expectancy = float(net.mean())
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = (
        float(gross_win / gross_loss) if gross_loss > 0
        else (float("inf") if gross_win > 0 else 0.0)
    )
    tdf["exit_date"] = pd.to_datetime(tdf["exit_time"]).dt.date
    daily = tdf.groupby("exit_date")["net_krw"].sum()
    if len(daily) >= 2 and daily.std(ddof=1) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(cfg.annualization))
    else:
        sharpe = 0.0
    equity = tdf["net_krw"].cumsum()
    max_dd = float((equity - equity.cummax()).min())
    return pv.BacktestResult(
        n_trades=n_trades,
        win_rate=win_rate,
        total_pnl_pts=total_pts,
        total_pnl_krw=total_pts * cfg.multiplier,
        expectancy_pts=expectancy,
        profit_factor=profit_factor,
        sharpe_daily=sharpe,
        max_drawdown_krw=max_dd,
        trades=tdf,
    )


def fmt_result(res: pv.BacktestResult):
    return (f"거래={res.n_trades:>4} | 승률={res.win_rate:>6.2f}% | "
            f"PnL={res.total_pnl_krw:>13,.0f} | Sharpe={res.sharpe_daily:>7.3f} | "
            f"MaxDD={res.max_drawdown_krw:>13,.0f}")


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        log_and_print("=" * 100, log)
        log_and_print("현재 DB에서 하락장 구간 추출 - 피봇 반전 검증", log)
        log_and_print("=" * 100, log)

        df_5min = load_data()
        log_and_print(f"전체 5분봉: {len(df_5min)}봉, {df_5min.index[0]} ~ {df_5min.index[-1]}", log)

        daily = rg.to_daily(df_5min, BT.session_boundary_hour)
        log_and_print(f"전체 거래일: {len(daily)}일, {daily.index[0].date()} ~ {daily.index[-1].date()}", log)

        bearish_sets, daily_df = identify_bearish_days(daily)
        log_and_print("\n[하락장 정의별 거래일 수]", log)
        for name, dates in bearish_sets.items():
            log_and_print(f"  {name}: {len(dates)}일", log)

        # 전략 실행
        strategies = {
            "롱-또는-플랫": run_long_or_flat(df_5min),
            "피봇 반전(both)": run_pivot(df_5min, "both"),
            "피봇 반전(long_only)": run_pivot(df_5min, "long_only"),
            "피봇 반전(short_only)": run_pivot(df_5min, "short_only"),
        }

        for name, res in strategies.items():
            log_and_print(f"\n[{name}]", log)
            if res.trades is None or res.trades.empty:
                log_and_print("  거래 없음", log)
                continue

            log_and_print("  전체               " + fmt_result(res), log)

            for set_name, dates in bearish_sets.items():
                bearish_trades = filter_trades_by_exit_dates(res.trades, dates)
                bear_res = compute_metrics(bearish_trades, BT)
                log_and_print(f"  {set_name:<20} {fmt_result(bear_res)}", log)

            non_bear_dates = set(daily.index.date) - bearish_sets["MA20_bear_and_down"]
            non_bear_trades = filter_trades_by_exit_dates(res.trades, non_bear_dates)
            non_bear_res = compute_metrics(non_bear_trades, BT)
            log_and_print(f"  {'비하락장(MA20_bear_and_down)':<20} {fmt_result(non_bear_res)}", log)

        log_and_print("\n" + "=" * 100, log)
        log_and_print("분석 완료", log)
        log_and_print("=" * 100, log)


if __name__ == "__main__":
    main()
