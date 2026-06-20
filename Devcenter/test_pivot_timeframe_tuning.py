# -*- coding: utf-8 -*-
"""
피봇반전 시간봉 튜닝 백테스트

1분봉 / 5분봉 / 15분봉 피봇반전 전략 비교
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


def run_pivot_backtest(df: pd.DataFrame, timeframe_name: str) -> dict:
    """주어진 시간봉 데이터로 피봇반전 백테스트 실행"""
    print(f"\n[{timeframe_name}] 백테스트 시작")
    
    # 지표 계산
    df_i = pv.compute_indicators(df)
    print(f"   지표 계산 완료: {len(df_i)} 봉")
    
    # 피봇 검출
    pivots = pv.detect_pivots(df_i, pcfg, fcfg)
    print(f"   피봇 검출 완료: {len(pivots)} 개")
    
    # 백테스트
    result = pv.backtest(df_i, pivots, bt)
    
    return {
        'timeframe': timeframe_name,
        'n_trades': result.n_trades,
        'win_rate': result.win_rate,
        'total_pnl_pts': result.total_pnl_pts,
        'total_pnl_krw': result.total_pnl_krw,
        'expectancy_pts': result.expectancy_pts,
        'profit_factor': result.profit_factor,
        'sharpe_daily': result.sharpe_daily,
        'max_drawdown_krw': result.max_drawdown_krw,
    }


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLCV 데이터 리샘플링"""
    df_resampled = df.resample(rule).agg({
        'OPEN': 'first',
        'HIGH': 'max',
        'LOW': 'min',
        'CLOSE': 'last',
        'VOLUME': 'sum'
    }).dropna()
    return df_resampled


def main():
    print("=" * 80)
    print("피봇반전 시간봉 튜닝 백테스트")
    print("=" * 80)
    
    # 1분봉 데이터 로드
    print("\n1. 1분봉 데이터 로드 중...")
    df_1min = pv.load_data_by_date(DB_PATH, "futures_1min", start="2025-06-24", end="2026-06-18")
    print(f"   1분봉 데이터 로드 완료: {len(df_1min)} 봉")
    print(f"   기간: {df_1min.index[0]} ~ {df_1min.index[-1]}")
    
    # 5분봉 변환
    print("\n2. 5분봉으로 변환 중...")
    df_5min = resample_ohlcv(df_1min, '5min')
    print(f"   5분봉 변환 완료: {len(df_5min)} 봉")
    
    # 15분봉 변환
    print("\n3. 15분봉으로 변환 중...")
    df_15min = resample_ohlcv(df_1min, '15min')
    print(f"   15분봉 변환 완료: {len(df_15min)} 봉")
    
    # 각 시간봉 백테스트
    results = []
    results.append(run_pivot_backtest(df_1min, "1분봉"))
    results.append(run_pivot_backtest(df_5min, "5분봉"))
    results.append(run_pivot_backtest(df_15min, "15분봉"))
    
    # 결과 비교
    print("\n" + "=" * 80)
    print("시간봉별 백테스트 결과 비교")
    print("=" * 80)
    print(f"{'시간봉':<10}{'거래수':>10}{'승률(%)':>10}{'손익(원)':>15}{'기대값(pt)':>12}{'PF':>8}{'Sharpe':>10}{'MaxDD(원)':>15}")
    print("-" * 80)
    for r in results:
        print(f"{r['timeframe']:<10}{r['n_trades']:>10}{r['win_rate']:>10.2f}{r['total_pnl_krw']:>15,.0f}{r['expectancy_pts']:>12.2f}{r['profit_factor']:>8.2f}{r['sharpe_daily']:>10.3f}{r['max_drawdown_krw']:>15,.0f}")
    print("=" * 80)
    
    # 최적 시간봉 찾기
    best_sharpe = max(results, key=lambda x: x['sharpe_daily'])
    best_pnl = max(results, key=lambda x: x['total_pnl_krw'])
    best_maxdd = max(results, key=lambda x: x['max_drawdown_krw'])  # 클수록 좋음
    
    print("\n최적 시간봉")
    print("-" * 80)
    print(f"Sharpe 기준: {best_sharpe['timeframe']} (Sharpe: {best_sharpe['sharpe_daily']:.3f})")
    print(f"손익 기준: {best_pnl['timeframe']} (손익: {best_pnl['total_pnl_krw']:,.0f} 원)")
    print(f"MaxDD 기준: {best_maxdd['timeframe']} (MaxDD: {best_maxdd['max_drawdown_krw']:,.0f} 원)")
    print("=" * 80)


if __name__ == '__main__':
    main()
