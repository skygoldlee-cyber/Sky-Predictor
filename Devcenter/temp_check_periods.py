# -*- coding: utf-8 -*-
import sys
sys.path.append('c:/Project/SkyPredictor/Devcenter')
import pandas as pd
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg
from datetime import date
from pivot_short_bear_strategy import load_data, detect_short_pivots, compute_bearish_trade_dates, filter_trades_by_dates, compute_metrics

BT = pv.BacktestConfig(
    multiplier=250_000,
    commission_pct_per_side=0.00015,
    slippage_ticks_per_side=1.0,
    tick_size=0.05,
    entry_on='next_open',
    annualization=252.0,
    intraday_only=True,
    session_boundary_hour=8,
    direction_mode='short_only',
)

df_5min = load_data()
daily = rg.to_daily(df_5min, BT.session_boundary_hour)
all_dates = set(daily.index.date)
print('all dates', min(all_dates), max(all_dates))
pivots = detect_short_pivots(df_5min)
bearish_dates = compute_bearish_trade_dates(daily)
print('bearish dates', len(bearish_dates))

base_res = pv.backtest(df_5min, pivots, BT)
print('base trades', len(base_res.trades))
for start, end in [
    (min(all_dates), date(2025, 9, 30)),
    (date(2025, 10, 1), date(2026, 1, 31)),
    (date(2026, 1, 1), max(all_dates)),
]:
    tdf = filter_trades_by_dates(
        base_res.trades,
        bearish_dates & {d for d in all_dates if start <= d <= end},
    )
    res = compute_metrics(tdf, BT)
    print(
        f'Period {start}~{end}: trades={res.n_trades}, PnL={res.total_pnl_krw:,.0f}, '
        f'Sharpe={res.sharpe_daily:.3f}, MaxDD={res.max_drawdown_krw:,.0f}'
    )
