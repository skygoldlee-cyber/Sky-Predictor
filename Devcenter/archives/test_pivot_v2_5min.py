# -*- coding: utf-8 -*-
"""
pivot_optuna_v2.py 테스트 스크립트 (5분봉)

1분봉 데이터를 5분봉으로 변환하여 피봇 반전 로직 테스트
"""
import sys
sys.path.append(r"c:\Project\SkyPredictor\Devcenter")

import pivot_optuna_v2 as pv
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

# 피봇 설정 (기본값)
pcfg = pv.HybridAdaptivePivotConfig(
    base_pct=0.5,
    base_multiplier=1.5,
    atr_weight=0.3,
    confirmation_bars=3,
)

# 필터 설정 (기본값)
fcfg = pv.FilterConfig(
    enabled=True,
    min_wave_pct=0.3,
    min_pivot_interval_bars=10,
    st_distance_threshold=0.1,
    adx_hold_threshold=15.0,
)

print("=" * 80)
print("pivot_optuna_v2.py 테스트 (5분봉)")
print("=" * 80)

# 1. 데이터 로드 (1분봉)
print("\n1. 1분봉 데이터 로드 중...")
df_1min = pv.load_data_by_date(DB_PATH, "futures_1min", start="2025-06-24", end="2026-06-18")
print(f"   1분봉 데이터 로드 완료: {len(df_1min)} 봉")
print(f"   기간: {df_1min.index[0]} ~ {df_1min.index[-1]}")

# 2. 5분봉으로 리샘플링
print("\n2. 5분봉으로 변환 중...")
df_5min = df_1min.resample('5T').agg({
    'OPEN': 'first',
    'HIGH': 'max',
    'LOW': 'min',
    'CLOSE': 'last',
    'VOLUME': 'sum'
}).dropna()
print(f"   5분봉 변환 완료: {len(df_5min)} 봉")
print(f"   기간: {df_5min.index[0]} ~ {df_5min.index[-1]}")

# 3. 지표 계산
print("\n3. 지표 계산 중...")
df_5min_i = pv.compute_indicators(df_5min)
print(f"   지표 계산 완료: {len(df_5min_i)} 봉")

# 4. 피봇 검출
print("\n4. 피봇 검출 중...")
pivots = pv.detect_pivots(df_5min_i, pcfg, fcfg)
print(f"   피봇 검출 완료: {len(pivots)} 개")

# 5. 백테스트
print("\n5. 백테스트 중...")
result = pv.backtest(df_5min_i, pivots, bt)
print(f"   백테스트 완료")

# 6. 결과 출력
print("\n" + "=" * 80)
print("백테스트 결과 (5분봉)")
print("=" * 80)
print(f"거래수: {result.n_trades}")
print(f"승률: {result.win_rate:.2%}")
print(f"총 손익 (pt): {result.total_pnl_pts:.2f}")
print(f"총 손익 (원): {result.total_pnl_krw:,.0f} 원")
print(f"기대값 (pt/거래): {result.expectancy_pts:.2f} pt")
print(f"Profit Factor: {result.profit_factor:.2f}")
print(f"Sharpe (일): {result.sharpe_daily:.3f}")
print(f"Max Drawdown (원): {result.max_drawdown_krw:,.0f} 원")
print("=" * 80)

# 7. 1분봉과 비교
print("\n" + "=" * 80)
print("1분봉 vs 5분봉 비교")
print("=" * 80)
print("1분봉:")
print("  거래수: 1,844")
print("  총 손익 (원): -5,751,068 원")
print("  기대값 (pt/거래): -0.01 pt")
print("  Profit Factor: 0.99")
print("  Sharpe (일): -0.091")
print("  Max Drawdown (원): -107,837,005 원")
print("\n5분봉:")
print(f"  거래수: {result.n_trades}")
print(f"  총 손익 (원): {result.total_pnl_krw:,.0f} 원")
print(f"  기대값 (pt/거래): {result.expectancy_pts:.2f} pt")
print(f"  Profit Factor: {result.profit_factor:.2f}")
print(f"  Sharpe (일): {result.sharpe_daily:.3f}")
print(f"  Max Drawdown (원): {result.max_drawdown_krw:,.0f} 원")
print("=" * 80)
