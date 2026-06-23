# -*- coding: utf-8 -*-
"""
피봇반전 파라미터 최적화 백테스트

base_pct, confirmation_bars, min_wave_pct 등을 그리드 서치로 테스트
"""
import sys
sys.path.append(r"c:\Project\SkyPredictor\Devcenter")

import pivot_optuna_v2 as pv
import pandas as pd
from itertools import product

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

# 파라미터 그리드
param_grid = {
    'base_pct': [0.2, 0.3, 0.5, 0.7, 1.0],
    'base_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0],
    'atr_weight': [0.0, 0.3, 0.5, 0.7, 1.0],
    'confirmation_bars': [0, 1, 2, 3, 5],
    'min_wave_pct': [0.1, 0.2, 0.3, 0.5, 0.7],
    'min_pivot_interval_bars': [5, 10, 20, 30],
}


def run_pivot_backtest(df: pd.DataFrame, params: dict) -> dict:
    """주어진 파라미터로 피봇반전 백테스트 실행"""
    # 피봇 설정
    pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=params['base_pct'],
        base_multiplier=params['base_multiplier'],
        atr_weight=params['atr_weight'],
        confirmation_bars=params['confirmation_bars'],
    )
    
    # 필터 설정
    fcfg = pv.FilterConfig(
        enabled=True,
        min_wave_pct=params['min_wave_pct'],
        min_pivot_interval_bars=params['min_pivot_interval_bars'],
        st_distance_threshold=0.1,
        adx_hold_threshold=15.0,
    )
    
    # 지표 계산
    df_i = pv.compute_indicators(df)
    
    # 피봇 검출
    pivots = pv.detect_pivots(df_i, pcfg, fcfg)
    
    # 백테스트
    result = pv.backtest(df_i, pivots, bt)
    
    return {
        'params': params,
        'n_trades': result.n_trades,
        'win_rate': result.win_rate,
        'total_pnl_pts': result.total_pnl_pts,
        'total_pnl_krw': result.total_pnl_krw,
        'expectancy_pts': result.expectancy_pts,
        'profit_factor': result.profit_factor,
        'sharpe_daily': result.sharpe_daily,
        'max_drawdown_krw': result.max_drawdown_krw,
    }


def main():
    print("=" * 80)
    print("피봇반전 파라미터 최적화")
    print("=" * 80)
    
    # 1분봉 데이터 로드
    print("\n1. 1분봉 데이터 로드 중...")
    df_1min = pv.load_data_by_date(DB_PATH, "futures_1min", start="2025-06-24", end="2026-06-18")
    print(f"   1분봉 데이터 로드 완료: {len(df_1min)} 봉")
    print(f"   기간: {df_1min.index[0]} ~ {df_1min.index[-1]}")
    
    # 5분봉 변환 (시간봉 튜닝 결과 5분봉이 상대적으로 안정적)
    print("\n2. 5분봉으로 변환 중...")
    df_5min = df_1min.resample('5min').agg({
        'OPEN': 'first',
        'HIGH': 'max',
        'LOW': 'min',
        'CLOSE': 'last',
        'VOLUME': 'sum'
    }).dropna()
    print(f"   5분봉 변환 완료: {len(df_5min)} 봉")
    
    # 그리드 생성 (조합 수 제한)
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    all_combinations = list(product(*values))
    
    print(f"\n3. 총 {len(all_combinations)} 개 파라미터 조합 테스트")
    
    # 백테스트 실행
    results = []
    for i, combo in enumerate(all_combinations):
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(all_combinations)}] {params}")
        
        try:
            result = run_pivot_backtest(df_5min, params)
            results.append(result)
            print(f"   거래수: {result['n_trades']}, 손익: {result['total_pnl_krw']:,.0f}, Sharpe: {result['sharpe_daily']:.3f}, MaxDD: {result['max_drawdown_krw']:,.0f}")
        except Exception as e:
            print(f"   오류: {e}")
    
    if not results:
        print("\n테스트 결과 없음")
        return
    
    # 결과 정렬
    best_sharpe = max(results, key=lambda x: x['sharpe_daily'])
    best_pnl = max(results, key=lambda x: x['total_pnl_krw'])
    best_pf = max(results, key=lambda x: x['profit_factor'])
    
    # 상위 10개 Sharpe 출력
    top_sharpe = sorted(results, key=lambda x: x['sharpe_daily'], reverse=True)[:10]
    top_pnl = sorted(results, key=lambda x: x['total_pnl_krw'], reverse=True)[:10]
    
    print("\n" + "=" * 80)
    print("파라미터 최적화 결과")
    print("=" * 80)
    
    print("\n[최고 Sharpe]")
    print(f"   파라미터: {best_sharpe['params']}")
    print(f"   거래수: {best_sharpe['n_trades']}, 승률: {best_sharpe['win_rate']:.2f}%")
    print(f"   손익: {best_sharpe['total_pnl_krw']:,.0f} 원, Sharpe: {best_sharpe['sharpe_daily']:.3f}, MaxDD: {best_sharpe['max_drawdown_krw']:,.0f} 원")
    
    print("\n[최고 손익]")
    print(f"   파라미터: {best_pnl['params']}")
    print(f"   거래수: {best_pnl['n_trades']}, 승률: {best_pnl['win_rate']:.2f}%")
    print(f"   손익: {best_pnl['total_pnl_krw']:,.0f} 원, Sharpe: {best_pnl['sharpe_daily']:.3f}, MaxDD: {best_pnl['max_drawdown_krw']:,.0f} 원")
    
    print("\n[최고 Profit Factor]")
    print(f"   파라미터: {best_pf['params']}")
    print(f"   거래수: {best_pf['n_trades']}, 승률: {best_pf['win_rate']:.2f}%")
    print(f"   손익: {best_pf['total_pnl_krw']:,.0f} 원, PF: {best_pf['profit_factor']:.2f}, Sharpe: {best_pf['sharpe_daily']:.3f}")
    
    print("\n[상위 10 Sharpe 파라미터]")
    print("-" * 80)
    print(f"{'순위':<5}{'base_pct':<10}{'base_mult':<10}{'atr_w':<8}{'conf':<6}{'min_wave':<10}{'interval':<10}{'거래수':<10}{'승률':<8}{'손익':<15}{'Sharpe':<10}")
    print("-" * 80)
    for i, r in enumerate(top_sharpe, 1):
        p = r['params']
        print(f"{i:<5}{p['base_pct']:<10}{p['base_multiplier']:<10}{p['atr_weight']:<8}{p['confirmation_bars']:<6}{p['min_wave_pct']:<10}{p['min_pivot_interval_bars']:<10}{r['n_trades']:<10}{r['win_rate']:<8.2f}{r['total_pnl_krw']:<15,.0f}{r['sharpe_daily']:<10.3f}")
    print("=" * 80)
    
    print("\n[상위 10 손익 파라미터]")
    print("-" * 80)
    print(f"{'순위':<5}{'base_pct':<10}{'base_mult':<10}{'atr_w':<8}{'conf':<6}{'min_wave':<10}{'interval':<10}{'거래수':<10}{'승률':<8}{'손익':<15}{'Sharpe':<10}")
    print("-" * 80)
    for i, r in enumerate(top_pnl, 1):
        p = r['params']
        print(f"{i:<5}{p['base_pct']:<10}{p['base_multiplier']:<10}{p['atr_weight']:<8}{p['confirmation_bars']:<6}{p['min_wave_pct']:<10}{p['min_pivot_interval_bars']:<10}{r['n_trades']:<10}{r['win_rate']:<8.2f}{r['total_pnl_krw']:<15,.0f}{r['sharpe_daily']:<10.3f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
