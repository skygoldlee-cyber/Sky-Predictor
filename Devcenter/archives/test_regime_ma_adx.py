# -*- coding: utf-8 -*-
"""
MA20/60 + ADX 기반 레짐 감지 전략 백테스트

regime_intraday_v2.py를 사용하여 MA20/60 + ADX 기반 레짐 감지 전략 테스트
"""
import sys
sys.path.append(r"c:\Project\SkyPredictor\Devcenter")

import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg

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
print("MA20/60 + ADX 기반 레짐 감지 전략 백테스트")
print("=" * 80)

# 1. 데이터 로드
print("\n1. 데이터 로드 중...")
df = pv.load_data_by_date(DB_PATH, "futures_1min", start="2025-06-24", end="2026-06-18")
print(f"   데이터 로드 완료: {len(df)} 봉")
print(f"   기간: {df.index[0]} ~ {df.index[-1]}")

# 2. 주간세션 필터
print("\n2. 주간세션 필터 중...")
df = pv.filter_day_session(df, start="08:45", end="15:45")
print(f"   주간세션 필터 완료: {len(df)} 봉")

# 3. 일봉 변환
print("\n3. 일봉 변환 중...")
daily = rg.to_daily(df, bt.session_boundary_hour)
print(f"   일봉 변환 완료: {len(daily)} 일")

# 4. 레짐 신호 계산 (MA20/60 + ADX)
print("\n4. 레짐 신호 계산 중 (MA20/60 + ADX)...")
signal = rg.daily_regime_signal(
    daily,
    regime_method="ma",
    ma_short=20,
    ma_long=60,
    adx_threshold=25.0,
    allow_short=False  # 롱-또는-플랫
)
print(f"   레짐 신호 계산 완료: {len(signal)} 일")

# 5. 백테스트
print("\n5. 백테스트 중...")
result = rg._bt_from_daily_signal(daily, signal, bt)
print(f"   백테스트 완료")

# 거래 데이터 확인
if result.trades is not None and len(result.trades) > 0:
    print(f"\n거래 데이터 샘플 (처음 5개):")
    print(result.trades.head())
    print(f"\nnet_pts 통계:")
    print(f"  최소: {result.trades['net_pts'].min()}")
    print(f"  최대: {result.trades['net_pts'].max()}")
    print(f"  평균: {result.trades['net_pts'].mean()}")
    print(f"  양수 개수: {(result.trades['net_pts'] > 0).sum()}")
    print(f"  음수 개수: {(result.trades['net_pts'] < 0).sum()}")
    print(f"  0 개수: {(result.trades['net_pts'] == 0).sum()}")

# 6. 결과 출력
print("\n" + "=" * 80)
print("백테스트 결과 (MA20/60 + ADX)")
print("=" * 80)
print(f"거래수: {result.n_trades}")

# 직접 승률 계산
if result.trades is not None and len(result.trades) > 0:
    wins = (result.trades['net_pts'] > 0).sum()
    total = len(result.trades)
    actual_win_rate = (wins / total) * 100
    print(f"승률 (계산): {actual_win_rate:.2f}%")
    print(f"승률 (result): {result.win_rate:.2f}%")
else:
    print(f"승률: {result.win_rate:.2f}%")

print(f"총 손익 (pt): {result.total_pnl_pts:.2f}")
print(f"총 손익 (원): {result.total_pnl_krw:,.0f} 원")
print(f"기대값 (pt/거래): {result.expectancy_pts:.2f} pt")
print(f"Profit Factor: {result.profit_factor:.2f}")
print(f"Sharpe (일): {result.sharpe_daily:.3f}")
print(f"Max Drawdown (원): {result.max_drawdown_krw:,.0f} 원")
print("=" * 80)

# 7. 피봇 반전 로직과 비교
print("\n" + "=" * 80)
print("전략 비교")
print("=" * 80)
print("피봇 반전 로직 (1분봉):")
print("  거래수: 1,844")
print("  총 손익 (원): -5,751,068 원")
print("  기대값 (pt/거래): -0.01 pt")
print("  Profit Factor: 0.99")
print("  Sharpe (일): -0.091")
print("  Max Drawdown (원): -107,837,005 원")
print("\n피봇 반전 로직 (5분봉):")
print("  거래수: 762")
print("  총 손익 (원): -103,023,314 원")
print("  기대값 (pt/거래): -0.54 pt")
print("  Profit Factor: 0.82")
print("  Sharpe (일): -1.872")
print("  Max Drawdown (원): -149,315,332 원")
print("\nMA20/60 + ADX 레짐 감지:")
print(f"  거래수: {result.n_trades}")
print(f"  총 손익 (원): {result.total_pnl_krw:,.0f} 원")
print(f"  기대값 (pt/거래): {result.expectancy_pts:.2f} pt")
print(f"  Profit Factor: {result.profit_factor:.2f}")
print(f"  Sharpe (일): {result.sharpe_daily:.3f}")
print(f"  Max Drawdown (원): {result.max_drawdown_krw:,.0f} 원")
print("=" * 80)
