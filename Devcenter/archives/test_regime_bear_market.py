# -*- coding: utf-8 -*-
"""
하락장 기간 식별 및 MA20/60 + ADX 전략 테스트

일봉 데이터를 확인하여 하락장 기간을 식별하고 해당 기간으로 백테스트
"""
import sys
sys.path.append(r"c:\Project\SkyPredictor\Devcenter")

import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg
import pandas as pd

# DB 경로
DB_PATH = "c:/Project/SkyPredictor/Devcenter/data/duckdb/market_data.duckdb"

# 백테스트 설정
bt = pv.BacktestConfig(
    multiplier=250_000,
    commission_pct_per_side=0.00003,
    slippage_ticks_per_side=1.0,
    tick_size=0.05,
    entry_on="next_open",
    annualization=252.0,
    intraday_only=True,
    session_boundary_hour=8,
    direction_mode="both",
)

print("=" * 80)
print("하락장 기간 식별 및 테스트")
print("=" * 80)

# 1. 전체 데이터 로드
print("\n1. 전체 데이터 로드 중...")
df = pv.load_data_by_date(DB_PATH, "futures_1min", start="2025-06-24", end="2026-06-18")
df = pv.filter_day_session(df, start="08:45", end="15:45")
print(f"   데이터 로드 완료: {len(df)} 봉")

# 2. 일봉 변환
print("\n2. 일봉 변환 중...")
daily = rg.to_daily(df, bt.session_boundary_hour)
print(f"   일봉 변환 완료: {len(daily)} 일")

# 3. 일봉 추세 분석
print("\n3. 일봉 추세 분석 중...")
daily['MA20'] = daily['CLOSE'].rolling(20).mean()
daily['MA60'] = daily['CLOSE'].rolling(60).mean()
daily['MA20_gt_MA60'] = daily['MA20'] > daily['MA60']

# 월별 수익률 계산
daily['monthly_return'] = daily['CLOSE'].pct_change(20)  # 20일(약 1개월) 수익률

# 하락장 기간 식별 (MA20 < MA60인 기간)
bear_market_periods = []
in_bear = False
start_date = None

for i in range(len(daily)):
    if not daily['MA20_gt_MA60'].iloc[i] and not in_bear:
        in_bear = True
        start_date = daily.index[i]
    elif daily['MA20_gt_MA60'].iloc[i] and in_bear:
        in_bear = False
        end_date = daily.index[i]
        bear_market_periods.append((start_date, end_date))

# 마지막 기간 처리
if in_bear:
    bear_market_periods.append((start_date, daily.index[-1]))

print(f"   하락장 기간 수: {len(bear_market_periods)}")
for i, (start, end) in enumerate(bear_market_periods):
    print(f"   기간 {i+1}: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")

# 4. 전체 기간 백테스트 (롱-또는-플랫)
print("\n4. 전체 기간 백테스트 (롱-또는-플랫) 중...")
signal_all = rg.daily_regime_signal(
    daily,
    regime_method="ma",
    ma_short=20,
    ma_long=60,
    adx_threshold=25.0,
    allow_short=False
)
result_all = rg._bt_from_daily_signal(daily, signal_all, bt)
print(f"   전체 기간 백테스트 완료")

# 4-1. 전체 기간 백테스트 (롱-숏)
print("\n4-1. 전체 기간 백테스트 (롱-숏) 중...")
signal_all_ls = rg.daily_regime_signal(
    daily,
    regime_method="ma",
    ma_short=20,
    ma_long=60,
    adx_threshold=25.0,
    allow_short=True
)
result_all_ls = rg._bt_from_daily_signal(daily, signal_all_ls, bt)
print(f"   전체 기간 백테스트 완료 (롱-숏)")

# 5. 하락장 기간 백테스트 (롱-또는-플랫)
print("\n5. 하락장 기간 백테스트 (롱-또는-플랫) 중...")
bear_results = []

for i, (start, end) in enumerate(bear_market_periods):
    # 해당 기간 데이터 필터링
    mask = (daily.index >= start) & (daily.index <= end)
    daily_bear = daily[mask].copy()
    signal_bear = signal_all[mask].copy()
    
    if len(daily_bear) < 20:  # 최소 20일 필요
        print(f"   기간 {i+1}: 데이터 부족 ({len(daily_bear)} 일) - 스킵")
        continue
    
    result_bear = rg._bt_from_daily_signal(daily_bear, signal_bear, bt)
    bear_results.append({
        'period': i+1,
        'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'),
        'n_days': len(daily_bear),
        'n_trades': result_bear.n_trades,
        'win_rate': result_bear.win_rate,
        'total_pnl_krw': result_bear.total_pnl_krw,
        'expectancy_pts': result_bear.expectancy_pts,
        'sharpe_daily': result_bear.sharpe_daily,
        'max_drawdown_krw': result_bear.max_drawdown_krw
    })
    print(f"   기간 {i+1}: 백테스트 완료 (거래수: {result_bear.n_trades}, 손익: {result_bear.total_pnl_krw:,.0f} 원)")

# 5-1. 하락장 기간 백테스트 (롱-숏)
print("\n5-1. 하락장 기간 백테스트 (롱-숏) 중...")
bear_results_ls = []

for i, (start, end) in enumerate(bear_market_periods):
    # 해당 기간 데이터 필터링
    mask = (daily.index >= start) & (daily.index <= end)
    daily_bear = daily[mask].copy()
    signal_bear_ls = signal_all_ls[mask].copy()
    
    if len(daily_bear) < 20:  # 최소 20일 필요
        print(f"   기간 {i+1}: 데이터 부족 ({len(daily_bear)} 일) - 스킵")
        continue
    
    result_bear_ls = rg._bt_from_daily_signal(daily_bear, signal_bear_ls, bt)
    bear_results_ls.append({
        'period': i+1,
        'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'),
        'n_days': len(daily_bear),
        'n_trades': result_bear_ls.n_trades,
        'win_rate': result_bear_ls.win_rate,
        'total_pnl_krw': result_bear_ls.total_pnl_krw,
        'expectancy_pts': result_bear_ls.expectancy_pts,
        'sharpe_daily': result_bear_ls.sharpe_daily,
        'max_drawdown_krw': result_bear_ls.max_drawdown_krw
    })
    print(f"   기간 {i+1}: 백테스트 완료 (거래수: {result_bear_ls.n_trades}, 손익: {result_bear_ls.total_pnl_krw:,.0f} 원)")

# 6. 결과 출력
print("\n" + "=" * 80)
print("백테스트 결과 요약")
print("=" * 80)

print("\n전체 기간:")
print("  롱-또는-플랫:")
print(f"    거래수: {result_all.n_trades}")
print(f"    승률: {result_all.win_rate:.2f}%")
print(f"    총 손익 (원): {result_all.total_pnl_krw:,.0f} 원")
print(f"    기대값 (pt/거래): {result_all.expectancy_pts:.2f} pt")
print(f"    Sharpe (일): {result_all.sharpe_daily:.3f}")
print(f"    Max Drawdown (원): {result_all.max_drawdown_krw:,.0f} 원")
print("  롱-숏:")
print(f"    거래수: {result_all_ls.n_trades}")
print(f"    승률: {result_all_ls.win_rate:.2f}%")
print(f"    총 손익 (원): {result_all_ls.total_pnl_krw:,.0f} 원")
print(f"    기대값 (pt/거래): {result_all_ls.expectancy_pts:.2f} pt")
print(f"    Sharpe (일): {result_all_ls.sharpe_daily:.3f}")
print(f"    Max Drawdown (원): {result_all_ls.max_drawdown_krw:,.0f} 원")

if bear_results:
    print("\n하락장 기간 (롱-또는-플랫):")
    avg_n_trades = sum(r['n_trades'] for r in bear_results) / len(bear_results)
    avg_win_rate = sum(r['win_rate'] for r in bear_results) / len(bear_results)
    avg_pnl = sum(r['total_pnl_krw'] for r in bear_results) / len(bear_results)
    avg_expectancy = sum(r['expectancy_pts'] for r in bear_results) / len(bear_results)
    avg_sharpe = sum(r['sharpe_daily'] for r in bear_results) / len(bear_results)
    avg_max_dd = sum(r['max_drawdown_krw'] for r in bear_results) / len(bear_results)
    
    print(f"  거래수: {avg_n_trades:.0f}")
    print(f"  승률: {avg_win_rate:.2f}%")
    print(f"  총 손익 (원): {avg_pnl:,.0f} 원")
    print(f"  기대값 (pt/거래): {avg_expectancy:.2f} pt")
    print(f"  Sharpe (일): {avg_sharpe:.3f}")
    print(f"  Max Drawdown (원): {avg_max_dd:,.0f} 원")
    
    print("\n하락장 기간별 상세 (롱-또는-플랫):")
    for r in bear_results:
        print(f"\n  기간 {r['period']} ({r['start']} ~ {r['end']}, {r['n_days']}일):")
        print(f"    거래수: {r['n_trades']}")
        print(f"    승률: {r['win_rate']:.2f}%")
        print(f"    총 손익 (원): {r['total_pnl_krw']:,.0f} 원")
        print(f"    기대값 (pt/거래): {r['expectancy_pts']:.2f} pt")
        print(f"    Sharpe (일): {r['sharpe_daily']:.3f}")
        print(f"    Max Drawdown (원): {r['max_drawdown_krw']:,.0f} 원")

if bear_results_ls:
    print("\n하락장 기간 (롱-숏):")
    avg_n_trades_ls = sum(r['n_trades'] for r in bear_results_ls) / len(bear_results_ls)
    avg_win_rate_ls = sum(r['win_rate'] for r in bear_results_ls) / len(bear_results_ls)
    avg_pnl_ls = sum(r['total_pnl_krw'] for r in bear_results_ls) / len(bear_results_ls)
    avg_expectancy_ls = sum(r['expectancy_pts'] for r in bear_results_ls) / len(bear_results_ls)
    avg_sharpe_ls = sum(r['sharpe_daily'] for r in bear_results_ls) / len(bear_results_ls)
    avg_max_dd_ls = sum(r['max_drawdown_krw'] for r in bear_results_ls) / len(bear_results_ls)
    
    print(f"  거래수: {avg_n_trades_ls:.0f}")
    print(f"  승률: {avg_win_rate_ls:.2f}%")
    print(f"  총 손익 (원): {avg_pnl_ls:,.0f} 원")
    print(f"  기대값 (pt/거래): {avg_expectancy_ls:.2f} pt")
    print(f"  Sharpe (일): {avg_sharpe_ls:.3f}")
    print(f"  Max Drawdown (원): {avg_max_dd_ls:,.0f} 원")
    
    print("\n하락장 기간별 상세 (롱-숏):")
    for r in bear_results_ls:
        print(f"\n  기간 {r['period']} ({r['start']} ~ {r['end']}, {r['n_days']}일):")
        print(f"    거래수: {r['n_trades']}")
        print(f"    승률: {r['win_rate']:.2f}%")
        print(f"    총 손익 (원): {r['total_pnl_krw']:,.0f} 원")
        print(f"    기대값 (pt/거래): {r['expectancy_pts']:.2f} pt")
        print(f"    Sharpe (일): {r['sharpe_daily']:.3f}")
        print(f"    Max Drawdown (원): {r['max_drawdown_krw']:,.0f} 원")

print("=" * 80)
