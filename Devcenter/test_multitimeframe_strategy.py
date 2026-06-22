# -*- coding: utf-8 -*-
"""
다중 시간봉 분석 전략 백테스트

1분봉 + 5분봉 + 일봉 조합 전략
- 1분봉: 진입/청산 타이밍
- 5분봉: 단기 트렌드 확인
- 일봉: 장기 트렌드 확인
"""
import pandas as pd
import duckdb
from pathlib import Path
import sys
import json

# Devcenter 경로 추가
sys.path.append(str(Path(__file__).parent))

import regime_intraday_v2 as rg
import pivot_optuna_v2 as pv


def load_config():
    """설정 파일 로드"""
    config_path = Path(__file__).parent / 'config' / 'long_or_flat_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """1분봉을 5분봉으로 리샘플링
    
    Args:
        df: 1분봉 데이터 (DatetimeIndex)
    
    Returns:
        pd.DataFrame: 5분봉 데이터
    """
    # 5분봉 OHLCV 생성
    df_5min = df.resample('5min').agg({
        'OPEN': 'first',
        'HIGH': 'max',
        'LOW': 'min',
        'CLOSE': 'last',
        'VOLUME': 'sum'
    }).dropna()
    
    return df_5min


def calculate_multitimeframe_signal(df_1min: pd.DataFrame, df_5min: pd.DataFrame, daily: pd.DataFrame, 
                                     ma_short: int = 20, ma_long: int = 60, adx_threshold: float = 25.0,
                                     allow_short: bool = False) -> pd.Series:
    """다중 시간봉 신호 계산
    
    Args:
        df_1min: 1분봉 데이터
        df_5min: 5분봉 데이터
        daily: 일봉 데이터
        ma_short: 단기 MA 기간
        ma_long: 장기 MA 기간
        adx_threshold: ADX 임계값
        allow_short: 숏 허용 여부
    
    Returns:
        pd.Series: 일별 신호 (1=롱, 0=플랫, -1=숏)
    """
    # 일봉 신호 (장기 트렌드)
    daily_signal = rg.daily_regime_signal(
        daily,
        regime_method="adx",
        ma_short=ma_short,
        ma_long=ma_long,
        adx_threshold=adx_threshold,
        allow_short=allow_short
    )
    
    # 5분봉 MA 크로스 계산 (일봉 기간의 1/5로 조정)
    # 일봉 MA20/60 -> 5분봉 MA4/12 (대략적으로 비슷한 시간 범위)
    ma_short_5min = df_5min['CLOSE'].rolling(window=max(4, ma_short // 5)).mean()
    ma_long_5min = df_5min['CLOSE'].rolling(window=max(12, ma_long // 5)).mean()
    
    # 5분봉 ADX 계산
    adx_5min = pv._adx(df_5min, period=14)
    
    # 5분봉 신호 (MA 크로스 + ADX 필터)
    signal_5min = pd.Series(0, index=df_5min.index)
    signal_5min[ma_short_5min > ma_long_5min] = 1  # 상승
    if allow_short:
        signal_5min[ma_short_5min < ma_long_5min] = -1  # 하락
    
    # ADX 필터 (임계값을 낮추어 신호 생성 증가)
    signal_5min[adx_5min < adx_threshold * 0.8] = 0
    
    # 5분봉 신호를 일별로 집계 (하루 중 5분봉 신호의 최빈값)
    # 일자별로 5분봉 신호를 집계
    df_5min_with_signal = df_5min.copy()
    df_5min_with_signal['signal'] = signal_5min
    df_5min_with_signal['date'] = df_5min_with_signal.index.date
    
    # 일자별 신호 집계 (롱 신호가 많으면 롱, 숏 신호가 많으면 숏)
    daily_signal_5min = df_5min_with_signal.groupby('date')['signal'].agg(
        lambda x: 1 if (x == 1).sum() > (x == -1).sum() else (-1 if (x == -1).sum() > (x == 1).sum() else 0)
    )
    
    # 인덱스를 일봉과 맞추기
    daily_signal_5min.index = pd.to_datetime(daily_signal_5min.index)
    
    # 다중 시간봉 필터: 일봉 신호를 기준으로 5분봉 신호로 확인
    # 일봉 신호가 롱일 때 5분봉이 숏이 아니면 롱 진입 (더 유연)
    final_signal = daily_signal.copy()
    
    for i in range(len(final_signal)):
        if i >= len(daily_signal_5min):
            final_signal.iloc[i] = 0
            continue
            
        daily_date = daily.index[i].date()
        if daily_date in daily_signal_5min.index:
            signal_5min_daily = daily_signal_5min.loc[daily_date]
        else:
            signal_5min_daily = 0
        
        # 일봉 신호가 플랫이면 플랫 유지
        if daily_signal.iloc[i] == 0:
            final_signal.iloc[i] = 0
        # 일봉 신호가 롱이고 5분봉이 숏이 아니면 롱 (5분봉이 플랫이거나 롱이면 OK)
        elif daily_signal.iloc[i] == 1 and signal_5min_daily != -1:
            final_signal.iloc[i] = 1
        # 일봉 신호가 숏이고 5분봉이 롱이 아니면 숏 (5분봉이 플랫이거나 숏이면 OK)
        elif daily_signal.iloc[i] == -1 and signal_5min_daily != 1:
            final_signal.iloc[i] = -1
        # 일봉과 5분봉 신호가 완전히 반대면 플랫
        else:
            final_signal.iloc[i] = 0
    
    return final_signal


def backtest_multitimeframe(config: dict, use_multitimeframe: bool = False):
    """다중 시간봉 전략 백테스트
    
    Args:
        config: 설정 정보
        use_multitimeframe: 다중 시간봉 사용 여부
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
    
    con = duckdb.connect(str(db_path))
    df = con.execute(f'''
        SELECT * FROM {table_name}
        ORDER BY timestamp
    ''').df()
    con.close()
    
    # 주간세션 필터
    session_start = config['data']['session_start']
    session_end = config['data']['session_end']
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = pv.filter_day_session(df, start=session_start, end=session_end)
    
    # 컬럼명 대문자로 변환
    df.columns = [col.upper() for col in df.columns]
    
    print(f"데이터 로드 완료: {len(df)} 봉")
    print(f"기간: {df.index[0]} ~ {df.index[-1]}")
    
    # 일봉 변환
    daily = rg.to_daily(df, bt.session_boundary_hour)
    print(f"일봉 변환 완료: {len(daily)}일")
    
    # 파라미터
    ma_short = config['parameters']['ma_short']
    ma_long = config['parameters']['ma_long']
    adx_threshold = config['parameters']['adx_threshold']
    allow_short = config['parameters']['allow_short']
    
    # 신호 계산
    if use_multitimeframe:
        print("\n다중 시간봉 신호 계산 (1분봉 + 5분봉 + 일봉)")
        
        # 5분봉 변환
        df_5min = resample_to_5min(df)
        print(f"5분봉 변환 완료: {len(df_5min)} 봉")
        
        # 다중 시간봉 신호
        signal = calculate_multitimeframe_signal(
            df, df_5min, daily,
            ma_short=ma_short,
            ma_long=ma_long,
            adx_threshold=adx_threshold,
            allow_short=allow_short
        )
    else:
        print("\n단일 시간봉 신호 계산 (일봉만)")
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


def print_results(result, strategy_name: str):
    """백테스트 결과 출력"""
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
    """단일 시간봉 vs 다중 시간봉 비교"""
    config = load_config()
    
    print("\n" + "=" * 80)
    print("다중 시간봉 전략 백테스트 비교")
    print("=" * 80)
    
    # 단일 시간봉 (현재 전략)
    print("\n[1] 단일 시간봉 전략 (일봉만)")
    result_single, signal_single = backtest_multitimeframe(config, use_multitimeframe=False)
    print_results(result_single, "단일 시간봉")
    
    # 다중 시간봉 (개선 전략)
    print("\n[2] 다중 시간봉 전략 (1분봉 + 5분봉 + 일봉)")
    result_multi, signal_multi = backtest_multitimeframe(config, use_multitimeframe=True)
    print_results(result_multi, "다중 시간봉")
    
    # 비교
    print("\n" + "=" * 80)
    print("전략 비교")
    print("=" * 80)
    print(f"{'지표':<20}{'단일 시간봉':>20}{'다중 시간봉':>20}{'차이':>20}")
    print("-" * 80)
    print(f"{'거래수':<20}{result_single.n_trades:>20}{result_multi.n_trades:>20}{result_multi.n_trades - result_single.n_trades:>20}")
    print(f"{'승률 (%)':<20}{result_single.win_rate:>20.2f}{result_multi.win_rate:>20.2f}{result_multi.win_rate - result_single.win_rate:>20.2f}")
    print(f"{'총 손익 (원)':<20}{result_single.total_pnl_krw:>20,.0f}{result_multi.total_pnl_krw:>20,.0f}{result_multi.total_pnl_krw - result_single.total_pnl_krw:>20,.0f}")
    print(f"{'기대값 (pt)':<20}{result_single.expectancy_pts:>20.2f}{result_multi.expectancy_pts:>20.2f}{result_multi.expectancy_pts - result_single.expectancy_pts:>20.2f}")
    print(f"{'Sharpe (일)':<20}{result_single.sharpe_daily:>20.3f}{result_multi.sharpe_daily:>20.3f}{result_multi.sharpe_daily - result_single.sharpe_daily:>20.3f}")
    print(f"{'MaxDD (원)':<20}{result_single.max_drawdown_krw:>20,.0f}{result_multi.max_drawdown_krw:>20,.0f}{result_multi.max_drawdown_krw - result_single.max_drawdown_krw:>20,.0f}")
    print("=" * 80)
    
    # 신호 분석
    print("\n신호 분석")
    print("-" * 80)
    print(f"{'신호':<10}{'단일 시간봉':>15}{'다중 시간봉':>15}")
    print("-" * 80)
    print(f"{'롱 (+1)':<10}{(signal_single == 1).sum():>15}{(signal_multi == 1).sum():>15}")
    print(f"{'플랫 (0)':<10}{(signal_single == 0).sum():>15}{(signal_multi == 0).sum():>15}")
    print(f"{'숏 (-1)':<10}{(signal_single == -1).sum():>15}{(signal_multi == -1).sum():>15}")
    print("=" * 80)
    
    # 결론
    print("\n결론")
    print("-" * 80)
    if result_multi.total_pnl_krw > result_single.total_pnl_krw:
        print("[O] 다중 시간봉 전략이 더 높은 수익")
    else:
        print("[X] 단일 시간봉 전략이 더 높은 수익")
    
    if result_multi.sharpe_daily > result_single.sharpe_daily:
        print("[O] 다중 시간봉 전략이 더 높은 Sharpe")
    else:
        print("[X] 단일 시간봉 전략이 더 높은 Sharpe")
    
    if result_multi.max_drawdown_krw < result_single.max_drawdown_krw:
        print("[O] 다중 시간봉 전략이 더 낮은 Max Drawdown")
    else:
        print("[X] 단일 시간봉 전략이 더 낮은 Max Drawdown")
    print("=" * 80)


if __name__ == '__main__':
    compare_strategies()
