"""
피봇 발생 빈도 분석 스크립트

하루 1건 이상 피봇이 발생하도록 min_wave_pct 파라미터를 조정하기 위한 분석
"""

import duckdb
import pandas as pd
from indicators.hybrid_adaptive_pivot import HybridAdaptivePivot, HybridAdaptivePivotConfig
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DuckDB 데이터베이스 경로
DB_PATH = "Devcenter/data/duckdb/market_data.duckdb"

def load_data_from_duckdb(db_path: str, start_date: str = "2019-01-01", end_date: str = "2026-06-26") -> pd.DataFrame:
    """DuckDB에서 1분봉 데이터 로드 후 5분봉으로 집계"""
    conn = duckdb.connect(db_path)
    
    # 1분봉 데이터 로드
    query = f"""
    SELECT timestamp, open, high, low, close, volume
    FROM futures_1min
    WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'
    ORDER BY timestamp
    """
    
    df = conn.execute(query).df()
    conn.close()
    
    if len(df) == 0:
        logger.warning(f"No data loaded from {start_date} to {end_date}")
        return df
    
    # 컬럼 이름 변경 (timestamp -> datetime)
    df.rename(columns={'timestamp': 'datetime'}, inplace=True)
    
    # 5분봉으로 집계
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # 5분봉 OHLCV 집계
    df_5min = df.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_5min.reset_index(inplace=True)
    
    logger.info(f"Loaded {len(df)} 1-minute bars, aggregated to {len(df_5min)} 5-minute bars from {start_date} to {end_date}")
    return df_5min

def analyze_pivot_frequency(df: pd.DataFrame, min_wave_pct: float) -> dict:
    """특정 min_wave_pct 파라미터로 피봇 발생 빈도 분석"""
    
    # HybridAdaptivePivot 설정
    config = HybridAdaptivePivotConfig(
        base_pct=0.3,
        base_multiplier=2.0,
        atr_weight=0.5,
        min_wave_pct=min_wave_pct,
        min_wave_atr_ratio=0.2,
        confirmation_bars=0,  # 즉시 확정
        warmup_bars=20
    )
    
    pivot = HybridAdaptivePivot(config)
    
    # 피봇 신호 수집
    pivot_signals = []
    
    for idx, row in df.iterrows():
        state = pivot.update(
            high=row['high'],
            low=row['low'],
            close=row['close'],
            bar_time=row['datetime']
        )
        
        if state.new_pivot_signal in ("new_high", "new_low"):
            pivot_signals.append({
                'datetime': row['datetime'],
                'signal': state.new_pivot_signal,
                'price': state.last_high if state.new_pivot_signal == "new_high" else state.last_low
            })
    
    # 일별 피봇 발생 건수 계산
    df_signals = pd.DataFrame(pivot_signals)
    if len(df_signals) == 0:
        return {
            'min_wave_pct': min_wave_pct,
            'total_pivots': 0,
            'daily_avg': 0.0,
            'max_daily': 0,
            'min_daily': 0
        }
    
    df_signals['date'] = pd.to_datetime(df_signals['datetime']).dt.date
    daily_counts = df_signals.groupby('date').size()
    
    return {
        'min_wave_pct': min_wave_pct,
        'total_pivots': len(pivot_signals),
        'daily_avg': daily_counts.mean(),
        'max_daily': daily_counts.max(),
        'min_daily': daily_counts.min(),
        'days_with_pivots': len(daily_counts),
        'total_days': df['datetime'].dt.date.nunique()
    }

def main():
    """메인 분석 함수"""
    
    # 데이터 로드 (전체 기간: 2019-2026)
    start_date = "2019-01-01"
    end_date = "2026-06-26"
    
    logger.info(f"Loading data from {start_date} to {end_date}")
    df = load_data_from_duckdb(DB_PATH, start_date, end_date)
    
    if len(df) == 0:
        logger.error("No data loaded from DuckDB")
        return
    
    # 다양한 min_wave_pct 값 테스트
    min_wave_pct_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    
    results = []
    
    for min_wave_pct in min_wave_pct_values:
        logger.info(f"Testing min_wave_pct={min_wave_pct}")
        result = analyze_pivot_frequency(df, min_wave_pct)
        results.append(result)
        logger.info(f"  Total pivots: {result['total_pivots']}, Daily avg: {result['daily_avg']:.2f}")
    
    # 결과 정리
    results_df = pd.DataFrame(results)
    
    # 하루 1건 이상 피봇이 발생하는 파라미터 추천
    valid_results = results_df[results_df['daily_avg'] >= 1.0]
    
    print("\n" + "="*80)
    print("피봇 발생 빈도 분석 결과")
    print("="*80)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("하루 1건 이상 피봇 발생 파라미터 추천")
    print("="*80)
    
    if len(valid_results) > 0:
        # 가장 낮은 min_wave_pct로 1건 이상 달성한 파라미터 추천
        best_result = valid_results.loc[valid_results['min_wave_pct'].idxmin()]
        print(f"추천 min_wave_pct: {best_result['min_wave_pct']}")
        print(f"  일일 평균 피봇: {best_result['daily_avg']:.2f}건")
        print(f"  총 피봇 수: {best_result['total_pivots']}건")
        print(f"  최대 일일 피봇: {best_result['max_daily']}건")
        print(f"  최소 일일 피봇: {best_result['min_daily']}건")
        print(f"  피봇 발생 일수: {best_result['days_with_pivots']}/{best_result['total_days']}일")
    else:
        print("하루 1건 이상 피봇이 발생하는 파라미터가 없습니다.")
        print("min_wave_pct를 더 낮추거나 다른 파라미터 조정이 필요합니다.")
    
    # 현재 설정 (min_wave_pct=0.05) 결과 확인
    current_result = results_df[results_df['min_wave_pct'] == 0.05]
    if len(current_result) > 0:
        print("\n" + "="*80)
        print("현재 설정 (min_wave_pct=0.05) 결과")
        print("="*80)
        print(f"일일 평균 피봇: {current_result.iloc[0]['daily_avg']:.2f}건")
        print(f"총 피봇 수: {current_result.iloc[0]['total_pivots']}건")

if __name__ == "__main__":
    main()
