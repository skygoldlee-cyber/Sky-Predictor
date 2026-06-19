# -*- coding: utf-8 -*-
"""
pivot_optuna_v2.py 테스트 스크립트 (1분봉)

개선된 피봇 반전 로직을 1분봉 데이터로 테스트
"""
import sys
sys.path.append(r"c:\Project\SkyPredictor\Devcenter")

import pivot_optuna_v2 as pv

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
print("pivot_optuna_v2.py 테스트 (1분봉)")
print("=" * 80)

# 1. 데이터 로드
print("\n1. 데이터 로드 중...")
df = pv.load_data_by_date(DB_PATH, "futures_1min", start="2025-06-24", end="2026-06-18")
print(f"   데이터 로드 완료: {len(df)} 봉")
print(f"   기간: {df.index[0]} ~ {df.index[-1]}")

# 2. 지표 계산
print("\n2. 지표 계산 중...")
df_i = pv.compute_indicators(df)
print(f"   지표 계산 완료: {len(df_i)} 봉")

# 3. 피봇 검출
print("\n3. 피봇 검출 중...")
pivots = pv.detect_pivots(df_i, pcfg, fcfg)
print(f"   피봇 검출 완료: {len(pivots)} 개")

# 4. 백테스트
print("\n4. 백테스트 중...")
result = pv.backtest(df_i, pivots, bt)
print(f"   백테스트 완료")

# 5. 결과 출력
print("\n" + "=" * 80)
print("백테스트 결과")
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
