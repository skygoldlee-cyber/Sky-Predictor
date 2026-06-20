# -*- coding: utf-8 -*-
"""
롱-숏 전략 백테스트

MA20/60 + ADX 기반 롱-숏 전략 백테스트
- allow_short=True: 하락장에서 숏 진입
- allow_short=False: 하락장에서 플랫 (현재 전략)
"""
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime
import sys

# Devcenter 경로 추가
sys.path.append(str(Path(__file__).parent))

import regime_intraday_v2 as rg
import pivot_optuna_v2 as pv


def load_config():
    """설정 파일 로드"""
    import json
    config_path = Path(__file__).parent / 'config' / 'long_or_flat_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def backtest_long_short(config: dict, allow_short: bool):
    """롱-숏 전략 백테스트
    
    Args:
        config: 설정 정보
        allow_short: 숏 허용 여부
    """
    # 백테스트 설정
    bt_config = config['backtest']
    bt = pv.BacktestConfig(
        multiplier=bt_config['multiplier'],
        commission_pct_per_side=bt_config['commission_pct_per_side'],
        slippage_ticks_per_side=bt_config['slippage_ticks_per_side'],
        tick_size=bt_config['tick_size'],
        entry_on=bt_config['entry_on'],
        annualization=bt_config['annualization'],
        intraday_only=bt_config['intraday_only'],
        session_boundary_hour=bt_config['session_boundary_hour'],
        direction_mode=bt_config['direction_mode'],
    )
    
    # 데이터 로드
    db_path = config['data']['db_path']
    table_name = config['data']['table_name']
    
    # 전체 기간 데이터 로드
    con = duckdb.connect(str(db_path))
    df = con.execute(f'''
        SELECT * FROM {table_name}
        ORDER BY timestamp
    ''').df()
    con.close()
    
    # 주간세션 필터
    session_start = config['data']['session_start']
    session_end = config['data']['session_end']
    
    # timestamp 컬럼을 DatetimeIndex로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = pv.filter_day_session(df, start=session_start, end=session_end)
    # 인덱스 유지 (to_daily에서 필요)
    
    # 컬럼명 대문자로 변환 (to_daily에서 OPEN, HIGH, LOW, CLOSE 필요)
    df.columns = [col.upper() for col in df.columns]
    
    print(f"데이터 로드 완료: {len(df)} 봉")
    print(f"기간: {df.index[0]} ~ {df.index[-1]}")
    
    # 일봉 변환
    try:
        daily = rg.to_daily(df, bt.session_boundary_hour)
        print(f"일봉 변환 완료: {len(daily)}일")
    except Exception as e:
        print(f"일봉 변환 오류: {e}")
        print(f"df 컬럼: {df.columns.tolist()}")
        print(f"df head:\n{df.head()}")
        raise
    
    # 파라미터
    ma_short = config['parameters']['ma_short']
    ma_long = config['parameters']['ma_long']
    adx_threshold = config['parameters']['adx_threshold']
    
    # 레짐 신호 계산
    signal = rg.daily_regime_signal(
        daily,
        regime_method="adx",
        ma_short=ma_short,
        ma_long=ma_long,
        adx_threshold=adx_threshold,
        allow_short=allow_short
    )
    
    # 백테스트
    result = rg._bt_from_daily_signal(daily, signal, bt)
    
    return result, signal


def print_results(result, allow_short: bool):
    """백테스트 결과 출력"""
    strategy_name = "롱-숏" if allow_short else "롱-또는-플랫"
    
    print("=" * 80)
    print(f"{strategy_name} 전략 백테스트 결과")
    print("=" * 80)
    print(f"거래수: {result.n_trades}")
    print(f"승률: {result.win_rate:.2f}%")
    print(f"총 손익 (pt): {result.total_pnl_pts:.2f}")
    print(f"총 손익 (원): {result.total_pnl_krw:,.0f} 원")
    print(f"기대값 (pt/거래): {result.expectancy_pts:.2f}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Sharpe (일): {result.sharpe_daily:.3f}")
    print(f"Max Drawdown (원): {result.max_drawdown_krw:,.0f} 원")
    print("=" * 80)


def compare_strategies():
    """롱-숏 vs 롱-또는-플랫 비교"""
    config = load_config()
    
    print("\n" + "=" * 80)
    print("롱-숏 전략 백테스트 비교")
    print("=" * 80)
    
    # 롱-또는-플랫 (현재 전략)
    print("\n[1] 롱-또는-플랫 전략 (allow_short=False)")
    result_flat, signal_flat = backtest_long_short(config, allow_short=False)
    print_results(result_flat, allow_short=False)
    
    # 롱-숏 (개선 전략)
    print("\n[2] 롱-숏 전략 (allow_short=True)")
    result_long_short, signal_long_short = backtest_long_short(config, allow_short=True)
    print_results(result_long_short, allow_short=True)
    
    # 비교
    print("\n" + "=" * 80)
    print("전략 비교")
    print("=" * 80)
    print(f"{'지표':<20}{'롱-또는-플랫':>20}{'롱-숏':>20}{'차이':>20}")
    print("-" * 80)
    print(f"{'거래수':<20}{result_flat.n_trades:>20}{result_long_short.n_trades:>20}{result_long_short.n_trades - result_flat.n_trades:>20}")
    print(f"{'승률 (%)':<20}{result_flat.win_rate:>20.2f}{result_long_short.win_rate:>20.2f}{result_long_short.win_rate - result_flat.win_rate:>20.2f}")
    print(f"{'총 손익 (원)':<20}{result_flat.total_pnl_krw:>20,.0f}{result_long_short.total_pnl_krw:>20,.0f}{result_long_short.total_pnl_krw - result_flat.total_pnl_krw:>20,.0f}")
    print(f"{'기대값 (pt)':<20}{result_flat.expectancy_pts:>20.2f}{result_long_short.expectancy_pts:>20.2f}{result_long_short.expectancy_pts - result_flat.expectancy_pts:>20.2f}")
    print(f"{'Sharpe (일)':<20}{result_flat.sharpe_daily:>20.3f}{result_long_short.sharpe_daily:>20.3f}{result_long_short.sharpe_daily - result_flat.sharpe_daily:>20.3f}")
    print(f"{'MaxDD (원)':<20}{result_flat.max_drawdown_krw:>20,.0f}{result_long_short.max_drawdown_krw:>20,.0f}{result_long_short.max_drawdown_krw - result_flat.max_drawdown_krw:>20,.0f}")
    print("=" * 80)
    
    # 신호 분석
    print("\n신호 분석")
    print("-" * 80)
    print(f"{'신호':<10}{'롱-또는-플랫':>15}{'롱-숏':>15}")
    print("-" * 80)
    print(f"{'롱 (+1)':<10}{(signal_flat == 1).sum():>15}{(signal_long_short == 1).sum():>15}")
    print(f"{'플랫 (0)':<10}{(signal_flat == 0).sum():>15}{(signal_long_short == 0).sum():>15}")
    print(f"{'숏 (-1)':<10}{(signal_flat == -1).sum():>15}{(signal_long_short == -1).sum():>15}")
    print("=" * 80)
    
    # 결론
    print("\n결론")
    print("-" * 80)
    if result_long_short.total_pnl_krw > result_flat.total_pnl_krw:
        print("[O] 롱-숏 전략이 더 높은 수익")
    else:
        print("[X] 롱-또는-플랫 전략이 더 높은 수익")
    
    if result_long_short.sharpe_daily > result_flat.sharpe_daily:
        print("[O] 롱-숏 전략이 더 높은 Sharpe")
    else:
        print("[X] 롱-또는-플랫 전략이 더 높은 Sharpe")
    
    if result_long_short.max_drawdown_krw < result_flat.max_drawdown_krw:
        print("[O] 롱-숏 전략이 더 낮은 Max Drawdown")
    else:
        print("[X] 롱-또는-플랫 전략이 더 낮은 Max Drawdown")
    print("=" * 80)


def test_adx_thresholds():
    """ADX 임계값 조정 테스트"""
    config = load_config()
    
    # 테스트할 ADX 임계값
    adx_thresholds = [15.0, 20.0, 25.0, 30.0, 35.0]
    
    print("\n" + "=" * 80)
    print("ADX 임계값 조정 테스트 (롱-숏 전략)")
    print("=" * 80)
    
    results = []
    for adx_threshold in adx_thresholds:
        print(f"\n[ADX 임계값: {adx_threshold}]")
        
        # 설정 업데이트
        config['parameters']['adx_threshold'] = adx_threshold
        
        # 백테스트
        result, signal = backtest_long_short(config, allow_short=True)
        
        # 신호 분석
        long_count = (signal == 1).sum()
        flat_count = (signal == 0).sum()
        short_count = (signal == -1).sum()
        
        print(f"거래수: {result.n_trades}")
        print(f"승률: {result.win_rate:.2f}%")
        print(f"총 손익 (원): {result.total_pnl_krw:,.0f}")
        print(f"Sharpe (일): {result.sharpe_daily:.3f}")
        print(f"Max Drawdown (원): {result.max_drawdown_krw:,.0f}")
        print(f"신호: 롱={long_count}, 플랫={flat_count}, 숏={short_count}")
        
        results.append({
            'adx_threshold': adx_threshold,
            'n_trades': result.n_trades,
            'win_rate': result.win_rate,
            'total_pnl_krw': result.total_pnl_krw,
            'sharpe_daily': result.sharpe_daily,
            'max_drawdown_krw': result.max_drawdown_krw,
            'long_count': long_count,
            'flat_count': flat_count,
            'short_count': short_count
        })
    
    # 결과 비교
    print("\n" + "=" * 80)
    print("ADX 임계값 비교")
    print("=" * 80)
    print(f"{'ADX':<10}{'거래수':>10}{'승률(%)':>10}{'손익(원)':>15}{'Sharpe':>10}{'MaxDD(원)':>15}{'롱':>8}{'플랫':>8}{'숏':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['adx_threshold']:<10.1f}{r['n_trades']:>10}{r['win_rate']:>10.2f}{r['total_pnl_krw']:>15,.0f}{r['sharpe_daily']:>10.3f}{r['max_drawdown_krw']:>15,.0f}{r['long_count']:>8}{r['flat_count']:>8}{r['short_count']:>8}")
    print("=" * 80)
    
    # 최적 ADX 임계값 찾기
    best_sharpe = max(results, key=lambda x: x['sharpe_daily'])
    best_pnl = max(results, key=lambda x: x['total_pnl_krw'])
    
    print("\n최적 ADX 임계값")
    print("-" * 80)
    print(f"Sharpe 기준: ADX {best_sharpe['adx_threshold']} (Sharpe: {best_sharpe['sharpe_daily']:.3f})")
    print(f"손익 기준: ADX {best_pnl['adx_threshold']} (손익: {best_pnl['total_pnl_krw']:,.0f} 원)")
    print("=" * 80)


if __name__ == '__main__':
    # ADX 임계값 조정 테스트
    test_adx_thresholds()
