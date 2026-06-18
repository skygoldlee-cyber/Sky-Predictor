"""
피봇 탐색 로직 성능 검증 스크립트

선물 240일, 현물 91일 데이터를 사용하여 피봇 탐색 성능을 검증합니다.
"""
import pandas as pd
import duckdb
from pathlib import Path
import sys
import logging
import traceback
import gc
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))

from indicators.hybrid_adaptive_pivot import HybridAdaptivePivot, HybridAdaptivePivotConfig

# 로깅 설정
class FlushFileHandler(logging.FileHandler):
    """즉시 파일에 쓰는 핸들러"""
    def emit(self, record):
        super().emit(record)
        self.flush()

# 로그 파일 경로 설정 (Devcenter 내부로 변경)
log_dir = Path(__file__).parent
log_file = log_dir / 'pivot_backtest.log'

# 로그 디렉토리 생성
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        FlushFileHandler(str(log_file), encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# 로그 파일 생성 확인
logger.info(f"로그 파일 경로: {log_file}")
logger.info("로깅 시스템 초기화 완료")


def calculate_atr(df, period=14):
    """ATR 계산"""
    high = df['HIGH']
    low = df['LOW']
    close = df['CLOSE'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_adx(df, period=14):
    """ADX 계산"""
    high = df['HIGH']
    low = df['LOW']
    close = df['CLOSE']
    
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # +DM, -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Smoothed TR, +DM, -DM
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # DX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    
    # ADX
    adx = dx.rolling(window=period).mean()
    
    return adx


def calculate_supertrend(df, period=10, multiplier=1.5):
    """SuperTrend 계산"""
    atr = calculate_atr(df, period)
    hl2 = (df['HIGH'] + df['LOW']) / 2
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(1, len(df)):
        prev_close = df['CLOSE'].iloc[i-1]
        prev_supertrend = supertrend.iloc[i-1]
        prev_direction = direction.iloc[i-1]
        
        if prev_direction == 1:  # Uptrend
            if df['CLOSE'].iloc[i] <= lower_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = max(upper_band.iloc[i], prev_supertrend)
                direction.iloc[i] = 1
        else:  # Downtrend
            if df['CLOSE'].iloc[i] >= upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = min(lower_band.iloc[i], prev_supertrend)
                direction.iloc[i] = -1
    
    # 첫 번째 값 설정
    supertrend.iloc[0] = lower_band.iloc[0]
    direction.iloc[0] = 1
    
    return supertrend, direction


def load_data_from_duckdb(db_path: str, table_name: str, days: int = None):
    """
    DuckDB에서 데이터 로드
    
    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 테이블명 (futures_1min, kospi_1min)
        days: 로드할 일수 (None이면 전체)
    
    Returns:
        DataFrame: timestamp, open, high, low, close, volume
    """
    logger.info(f"데이터 로드 시작: {table_name}, days={days}")
    try:
        con = duckdb.connect(db_path)
        
        if days:
            limit_count = days * 500  # 하루 최대 500건 가정
            query = f"""
            SELECT * FROM {table_name}
            ORDER BY timestamp
            LIMIT {limit_count}
            """
        else:
            query = f"""
            SELECT * FROM {table_name}
            ORDER BY timestamp
            """
        
        df = con.execute(query).df()
        con.close()
        
        # timestamp를 datetime으로 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M')
        df = df.set_index('timestamp')
        
        # 컬럼명을 대문자로 변경 (백테스팅 코드 호환성)
        df.columns = df.columns.str.upper()
        
        logger.info(f"데이터 로드 완료: {len(df)}건 ({df.index[0]} ~ {df.index[-1]})")
        return df
    except Exception as e:
        logger.error(f"데이터 로드 실패: {table_name}, 오류: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def run_pivot_detection(df, pivot_config=None, apply_filters=True):
    """
    피봇 탐색 실행 (P1, P2, P5, P10 필터 적용)
    
    Args:
        df: OHLCV 데이터
        pivot_config: 피봇 설정 (None이면 기본값)
        apply_filters: 필터 적용 여부
    
    Returns:
        DataFrame: 피봇 포인트
    """
    logger.info(f"피봇 탐색 시작: config={pivot_config}, apply_filters={apply_filters}")
    try:
        if pivot_config is None:
            pivot_config = HybridAdaptivePivotConfig(
                base_pct=0.3,
                base_multiplier=2.0,
                atr_weight=0.5,
                confirmation_bars=3,
            )
        
        pivot_detector = HybridAdaptivePivot(pivot_config)
        
        # 필터용 지표 계산
        if apply_filters:
            df = df.copy()
            df['ATR'] = calculate_atr(df, period=14)
            df['ADX'] = calculate_adx(df, period=14)
            df['ST'], df['ST_DIR'] = calculate_supertrend(df, period=10, multiplier=1.5)
            logger.info("필터용 지표 계산 완료 (ATR, ADX, SuperTrend)")
        
        # 필터 설정
        min_wave_pct = 0.3  # P1: wave_size_pct 하한
        min_pivot_interval_bars = 10  # P2: 최소 피봇 간격
        adx_hold_threshold = 15.0  # P10: ADX HOLD 임계값
        
        # 데이터 순회하며 피봇 탐지
        pivots = []
        bar_count = 0
        last_pivot_bar_idx = -999
        last_pivot_price = None
        
        for idx, row in df.iterrows():
            state = pivot_detector.update(
                high=row['HIGH'],
                low=row['LOW'],
                close=row['CLOSE'],
                bar_time=idx
            )
            
            if state.new_pivot_signal in ('new_high', 'new_low'):
                # 필터링 로직
                filtered = False
                filter_reason = ""
                
                if apply_filters:
                    # P1: wave_size_pct 필터
                    if last_pivot_price is not None:
                        wave_size_pct = abs(row['CLOSE'] - last_pivot_price) / last_pivot_price * 100
                        if wave_size_pct < min_wave_pct:
                            filtered = True
                            filter_reason = f"wave_size_too_small({wave_size_pct:.2f}%)"
                    
                    # P2: min_pivot_interval_bars 필터
                    current_bar_idx = bar_count
                    if current_bar_idx - last_pivot_bar_idx < min_pivot_interval_bars:
                        filtered = True
                        filter_reason = f"too_soon_after_last_pivot({current_bar_idx - last_pivot_bar_idx}bars)"
                    
                    # P5: SuperTrend 거리 필터 (간단 구현)
                    if not filtered and pd.notna(row['ST']):
                        st_distance_pct = abs(row['CLOSE'] - row['ST']) / row['CLOSE'] * 100
                        if st_distance_pct < 0.1:  # 너무 가까우면 필터
                            filtered = True
                            filter_reason = f"too_close_to_supertrend({st_distance_pct:.2f}%)"
                    
                    # P10: ADX 기반 필터
                    if not filtered and pd.notna(row['ADX']):
                        adx_value = row['ADX']
                        if adx_value < adx_hold_threshold:
                            filtered = True
                            filter_reason = f"ADX_too_weak({adx_value:.1f})"
                
                if not filtered:
                    pivots.append({
                        'timestamp': idx,
                        'price': row['CLOSE'],
                        'is_high': state.new_pivot_signal == 'new_high',
                        'pivot_score': pivot_detector.pivot_score,
                        'filter_reason': None
                    })
                    last_pivot_bar_idx = bar_count
                    last_pivot_price = row['CLOSE']
                else:
                    logger.debug(f"피봇 필터됨: {filter_reason}")
            
            bar_count += 1
            if bar_count % 10000 == 0:
                logger.info(f"피봇 탐지 진행 중: {bar_count}/{len(df)} 봉 처리, 피봇 수: {len(pivots)}")
        
        logger.info(f"피봇 탐지 완료: 총 {len(pivots)}개 피봇 발견 (필터 적용: {apply_filters})")
        return pd.DataFrame(pivots)
    except Exception as e:
        logger.error(f"피봇 탐지 실패: 오류: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def run_backtest(df, pivots, spot_pivots=None, config=None):
    """
    백테스팅 실행 (선물 매매 시뮬레이션, 현물 보조 지표 활용)
    
    Args:
        df: 선물 OHLCV 데이터
        pivots: 선물 피봇 포인트
        spot_pivots: 현물 피봇 포인트 (보조 지표)
        config: 백테스팅 설정 (사용하지 않음)
    
    Returns:
        백테스팅 결과 (딕셔너리)
    """
    # 기본 통계 계산
    total_pivots = len(pivots)
    high_pivots = len(pivots[pivots['is_high']])
    low_pivots = len(pivots[~pivots['is_high']])
    
    # 현물-선물 피봇 동시성 분석
    if spot_pivots is not None and len(spot_pivots) > 0:
        # 현물 피봇 타임스탬프 집합
        spot_pivot_times = set(spot_pivots['timestamp'])
        
        # 선물 피봇 중 현물 피봇과 동시에 발생한 것 확인 (±5분 이내)
        confirmed_pivots = []
        for idx, pivot in pivots.iterrows():
            pivot_time = pivot['timestamp']
            # 현물 피봇과 ±5분 이내인지 확인
            is_confirmed = any(abs((pivot_time - sp).total_seconds()) <= 300 for sp in spot_pivot_times)
            confirmed_pivots.append(is_confirmed)
        
        pivots['spot_confirmed'] = confirmed_pivots
        confirmed_count = sum(confirmed_pivots)
    else:
        pivots['spot_confirmed'] = False
        confirmed_count = 0
    
    # 피봇 기반 매매 시뮬레이션 (선물만)
    trades = []
    position = None  # None, 'long', 'short'
    entry_price = 0
    entry_time = None
    use_spot_filter = spot_pivots is not None and len(spot_pivots) > 0
    
    for idx, row in df.iterrows():
        # 해당 시간에 피봇이 있는지 확인
        pivot_at_time = pivots[pivots['timestamp'] == idx]
        
        if len(pivot_at_time) > 0:
            pivot = pivot_at_time.iloc[0]
            
            # 현물 필터 적용: 현물 피봇과 동시에 발생한 경우만 진입
            if use_spot_filter and not pivot['spot_confirmed']:
                continue
            
            # 기존 포지션 청산
            if position is not None:
                exit_price = row['CLOSE']
                if position == 'long':
                    profit = (exit_price - entry_price)
                else:  # short
                    profit = (entry_price - exit_price)
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': idx,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'spot_confirmed': pivot['spot_confirmed'],
                })
                
                position = None
                entry_price = 0
                entry_time = None
            
            # 새 포지션 진입
            if pivot['is_high']:
                position = 'short'  # 고점에서 숏 진입
            else:
                position = 'long'   # 저점에서 롱 진입
            
            entry_price = row['CLOSE']
            entry_time = idx
    
    # 마지막 포지션 청산
    if position is not None:
        exit_price = df.iloc[-1]['CLOSE']
        if position == 'long':
            profit = (exit_price - entry_price)
        else:  # short
            profit = (entry_price - exit_price)
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': df.index[-1],
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit,
            'spot_confirmed': False,
        })
    
    # 일별 승률 계산
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_time']).dt.date
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time']).dt.date
        trades_df['is_win'] = trades_df['profit'] > 0
        
        # 일별 승률 (진입일 기준)
        daily_stats = trades_df.groupby('entry_date').agg({
            'is_win': ['sum', 'count'],
            'profit': 'sum'
        }).reset_index()
        daily_stats.columns = ['date', 'wins', 'total_trades', 'total_profit']
        daily_stats['win_rate'] = (daily_stats['wins'] / daily_stats['total_trades'] * 100).round(2)
        
        # 전체 승률
        total_wins = trades_df['is_win'].sum()
        total_trades = len(trades_df)
        overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        # 시간 경과에 따른 승률 변화 분석
        if len(daily_stats) >= 2:
            # 초기 기간 (전체 기간의 30%)
            initial_period = int(len(daily_stats) * 0.3)
            if initial_period < 1:
                initial_period = 1
            
            initial_stats = daily_stats.head(initial_period)
            initial_wins = initial_stats['wins'].sum()
            initial_trades = initial_stats['total_trades'].sum()
            initial_win_rate = (initial_wins / initial_trades * 100) if initial_trades > 0 else 0
            
            # 최근 기간 (전체 기간의 30%)
            recent_period = int(len(daily_stats) * 0.3)
            if recent_period < 1:
                recent_period = 1
            
            recent_stats = daily_stats.tail(recent_period)
            recent_wins = recent_stats['wins'].sum()
            recent_trades = recent_stats['total_trades'].sum()
            recent_win_rate = (recent_wins / recent_trades * 100) if recent_trades > 0 else 0
            
            # 승률 변화
            win_rate_change = recent_win_rate - initial_win_rate
            win_rate_improvement = (win_rate_change / initial_win_rate * 100) if initial_win_rate > 0 else 0
        else:
            initial_win_rate = None
            recent_win_rate = None
            win_rate_change = None
            win_rate_improvement = None
        
        # 현물 확인 필터 적용 시 승률
        if use_spot_filter:
            confirmed_trades = trades_df[trades_df['spot_confirmed']]
            if len(confirmed_trades) > 0:
                confirmed_wins = confirmed_trades['is_win'].sum()
                confirmed_win_rate = (confirmed_wins / len(confirmed_trades) * 100)
            else:
                confirmed_win_rate = 0
        else:
            confirmed_win_rate = None
    else:
        daily_stats = pd.DataFrame()
        overall_win_rate = 0
        total_wins = 0
        total_trades = 0
        confirmed_win_rate = None
        initial_win_rate = None
        recent_win_rate = None
        win_rate_change = None
        win_rate_improvement = None
    
    # 피봇 간 거리 계산
    if total_pivots > 1:
        pivots_sorted = pivots.sort_values('timestamp')
        pivot_distances = pivots_sorted['timestamp'].diff().dt.total_seconds() / 60  # 분 단위
        avg_distance = pivot_distances.mean()
        min_distance = pivot_distances.min()
        max_distance = pivot_distances.max()
    else:
        avg_distance = 0
        min_distance = 0
        max_distance = 0
    
    # 피봇 스코어 통계
    if 'pivot_score' in pivots.columns:
        avg_score = pivots['pivot_score'].mean()
        max_score = pivots['pivot_score'].max()
        min_score = pivots['pivot_score'].min()
    else:
        avg_score = 0
        max_score = 0
        min_score = 0
    
    return {
        'total_pivots': total_pivots,
        'high_pivots': high_pivots,
        'low_pivots': low_pivots,
        'avg_distance_minutes': avg_distance,
        'min_distance_minutes': min_distance,
        'max_distance_minutes': max_distance,
        'avg_score': avg_score,
        'max_score': max_score,
        'min_score': min_score,
        'spot_confirmed_count': confirmed_count,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'overall_win_rate': overall_win_rate,
        'confirmed_win_rate': confirmed_win_rate,
        'initial_win_rate': initial_win_rate,
        'recent_win_rate': recent_win_rate,
        'win_rate_change': win_rate_change,
        'win_rate_improvement': win_rate_improvement,
        'daily_stats': daily_stats,
        'trades': trades_df,
    }


def main():
    logger.info("="*60)
    logger.info("피봇 탐색 성능 검증 시작")
    logger.info("="*60)
    
    try:
        # 데이터 경로
        db_path = str(Path(__file__).parent / 'data' / 'duckdb' / 'market_data.duckdb')
        logger.info(f"데이터베이스 경로: {db_path}")
        
        # 선물 데이터 로드 (240일 전체 로드)
        print("선물 데이터 로드 중...")
        logger.info("선물 데이터 로드 시작 (240일)")
        futures_df = load_data_from_duckdb(db_path, 'futures_1min', days=240)
        print(f"선물 데이터: {len(futures_df)}건 ({futures_df.index[0]} ~ {futures_df.index[-1]})")

        # 현물 데이터 로드 (240일 전체 로드)
        print("\n현물 데이터 로드 중...")
        logger.info("현물 데이터 로드 시작 (240일)")
        kospi_df = load_data_from_duckdb(db_path, 'kospi_1min', days=240)
        print(f"현물 데이터: {len(kospi_df)}건 ({kospi_df.index[0]} ~ {kospi_df.index[-1]})")
    
        # 기간 맞추기 (두 데이터의 겹치는 기간 사용)
        print("\n기간 맞추기 중...")
        logger.info("기간 맞추기 시작")
        start_date = max(futures_df.index[0], kospi_df.index[0])
        end_date = min(futures_df.index[-1], kospi_df.index[-1])

        if start_date > end_date:
            logger.warning(f"기간이 겹치지 않음: 선물({futures_df.index[0]}~{futures_df.index[-1]}), 현물({kospi_df.index[0]}~{kospi_df.index[-1]})")
            # 겹치지 않으면 선물 데이터만 사용
            kospi_df = None  # 현물 데이터 사용 안함
            logger.info("현물 데이터 기간이 맞지 않아 선물 데이터만 사용")
        else:
            futures_df_filtered = futures_df[(futures_df.index >= start_date) & (futures_df.index <= end_date)]
            kospi_df_filtered = kospi_df[(kospi_df.index >= start_date) & (kospi_df.index <= end_date)]

            print(f"필터링된 선물 데이터: {len(futures_df_filtered)}건 ({futures_df_filtered.index[0]} ~ {futures_df_filtered.index[-1]})")
            print(f"필터링된 현물 데이터: {len(kospi_df_filtered)}건 ({kospi_df_filtered.index[0]} ~ {kospi_df_filtered.index[-1]})")
            logger.info(f"필터링된 선물 데이터: {len(futures_df_filtered)}건")
            logger.info(f"필터링된 현물 데이터: {len(kospi_df_filtered)}건")

            futures_df = futures_df_filtered
            kospi_df = kospi_df_filtered
        
        # 점진적 청크 기반 파라미터 최적화
        print("\n" + "="*60)
        print("점진적 청크 기반 파라미터 최적화")
        print("="*60)
        logger.info("점진적 청크 기반 파라미터 최적화 시작")
        
        # 랜덤 서치 파라미터 범위
        import random
        random.seed(42)
        
        # 랜덤 서치 횟수 (승률 60% 달성을 위해 100회로 증가)
        n_random_search = 100
        
        # 청크 크기 (30일)
        chunk_days = 30
        
        # 전체 데이터 기간
        total_days = len(futures_df.resample('D').size())
        target_pivots_per_day = 10
        target_win_rate = 60.0
        
        print(f"총 데이터 기간: {total_days}일")
        print(f"청크 크기: {chunk_days}일")
        print(f"랜덤 서치 횟수: {n_random_search}")
        logger.info(f"총 데이터 기간: {total_days}일, 청크 크기: {chunk_days}일, 랜덤 서치 횟수: {n_random_search}")
        
        best_config = None
        best_win_rate = 0
        results = []
        
        for search_iter in range(n_random_search):
            # 랜덤 파라미터 생성 (승률 60% 달성을 위해 넓은 범위)
            base_pct = random.uniform(0.1, 1.0)  # 넓은 범위
            base_multiplier = random.uniform(1.0, 6.0)  # 넓은 범위
            confirmation_bars = random.randint(0, 6)  # 넓은 범위
            atr_weight = random.uniform(0.0, 0.8)  # 넓은 범위
            
            logger.info(f"랜덤 서치 {search_iter+1}/{n_random_search}: base_pct={base_pct:.3f}, base_multiplier={base_multiplier:.2f}, confirmation_bars={confirmation_bars}, atr_weight={atr_weight:.2f}")
            
            # 청크별 결과 집계
            chunk_win_rates = []
            chunk_pivot_counts = []
            chunk_count = 0
            
            # 날짜별로 청크 나누기
            unique_dates = futures_df.index.normalize().unique()
            
            for chunk_start in range(0, len(unique_dates), chunk_days):
                chunk_end = min(chunk_start + chunk_days, len(unique_dates))
                chunk_dates = unique_dates[chunk_start:chunk_end]
                
                if len(chunk_dates) < 5:  # 너무 작은 청크는 건너뜀
                    continue
                
                chunk_start_date = chunk_dates[0]
                chunk_end_date = chunk_dates[-1]
                
                # 청크 데이터 추출
                chunk_futures = futures_df[(futures_df.index.normalize() >= chunk_start_date) & 
                                            (futures_df.index.normalize() <= chunk_end_date)].copy()
                
                if len(chunk_futures) == 0:
                    continue
                
                chunk_count += 1
                logger.info(f"  청크 {chunk_count}: {chunk_start_date.date()} ~ {chunk_end_date.date()} ({len(chunk_futures)} 봉)")
                
                try:
                    pivot_config = HybridAdaptivePivotConfig(
                        base_pct=base_pct,
                        base_multiplier=base_multiplier,
                        atr_weight=atr_weight,
                        confirmation_bars=confirmation_bars,
                    )
                    
                    pivots = run_pivot_detection(chunk_futures, pivot_config, apply_filters=True)
                    pivot_count = len(pivots)
                    chunk_days_count = len(chunk_futures.resample('D').size())
                    pivots_per_day = pivot_count / chunk_days_count if chunk_days_count > 0 else 0
                    
                    # 백테스팅으로 승률 계산
                    if pivot_count > 0:
                        backtest_result = run_backtest(chunk_futures, pivots, spot_pivots=None)
                        win_rate = backtest_result['overall_win_rate']
                    else:
                        win_rate = 0
                    
                    chunk_win_rates.append(win_rate)
                    chunk_pivot_counts.append(pivots_per_day)
                    
                    logger.info(f"    청크 결과: 피봇 {pivot_count}개 ({pivots_per_day:.1f}/일), 승률 {win_rate:.2f}%")
                    
                    # 메모리 해제
                    del pivots
                    del pivot_config
                    del chunk_futures
                    del backtest_result
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"    청크 처리 실패: {str(e)}")
                    logger.error(traceback.format_exc())
                    gc.collect()
                    continue
            
            # 전체 청크의 평균 승률 계산
            if len(chunk_win_rates) > 0:
                avg_win_rate = sum(chunk_win_rates) / len(chunk_win_rates)
                avg_pivots_per_day = sum(chunk_pivot_counts) / len(chunk_pivot_counts)
                
                # 승률 60% 이상인 파라미터만 고려
                if avg_win_rate >= 60.0:
                    result = {
                        'base_pct': base_pct,
                        'base_multiplier': base_multiplier,
                        'confirmation_bars': confirmation_bars,
                        'atr_weight': atr_weight,
                        'avg_win_rate': avg_win_rate,
                        'avg_pivots_per_day': avg_pivots_per_day,
                        'chunk_count': chunk_count,
                    }
                    results.append(result)
                    
                    # 최적 파라미터 업데이트 (승률 기준)
                    if avg_win_rate > best_win_rate:
                        best_win_rate = avg_win_rate
                        best_config = result.copy()
                        logger.info(f"최적 파라미터 업데이트: 승률 {avg_win_rate:.2f}%, 피봇 {avg_pivots_per_day:.1f}/일")
                    
                    logger.info(f"랜덤 서치 {search_iter+1} 완료: 평균 승률 {avg_win_rate:.2f}%, 평균 피봇 {avg_pivots_per_day:.1f}/일 (60% 이상 통과)")
                else:
                    logger.info(f"랜덤 서치 {search_iter+1} 완료: 평균 승률 {avg_win_rate:.2f}%, 평균 피봇 {avg_pivots_per_day:.1f}/일 (60% 미만 제외)")
            else:
                logger.warning(f"랜덤 서치 {search_iter+1}: 유효한 청크 없음")
            
            # 메모리 해제
            gc.collect()
        
        # 결과 정렬
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('avg_win_rate', ascending=False)
        else:
            results_df = pd.DataFrame()
        
        if len(results_df) > 0:
            print(f"\n최적 파라미터 (승률 60% 이상 기준):")
            print(f"base_pct: {best_config['base_pct']:.3f}")
            print(f"base_multiplier: {best_config['base_multiplier']:.2f}")
            print(f"confirmation_bars: {best_config['confirmation_bars']}")
            print(f"atr_weight: {best_config['atr_weight']:.2f}")
            print(f"평균 승률: {best_config['avg_win_rate']:.2f}%")
            print(f"평균 피봇/일: {best_config['avg_pivots_per_day']:.2f}개")
            print(f"처리 청크 수: {best_config['chunk_count']}")
            logger.info(f"최적 파라미터 찾음: {best_config}")
        else:
            print(f"\n승률 60% 이상인 파라미터를 찾지 못했습니다.")
            logger.warning("승률 60% 이상인 파라미터 없음")
            print("랜덤 서치 범위를 조정하거나 승률 기준을 낮춰주세요.")
            return
        
        print(f"\n상위 10개 파라미터 조합 (승률 순):")
        print(results_df.head(10).to_string(index=False))
        
        # 승률 60% 이상이면서 하루 10개 이하인 파라미터 필터링
        valid_results = results_df[
            (results_df['avg_win_rate'] >= target_win_rate) &
            (results_df['avg_pivots_per_day'] <= target_pivots_per_day)
        ]
        
        if len(valid_results) > 0:
            # 유효한 파라미터 중에서 최고 승률 선택
            selected_config = valid_results.iloc[0].to_dict()
            print(f"\n승률 {target_win_rate}% 이상이면서 하루 {target_pivots_per_day}개 이하인 파라미터 찾음!")
            print(f"선택된 파라미터: 승률 {selected_config['avg_win_rate']:.2f}%, 하루 {selected_config['avg_pivots_per_day']:.2f}개")
        else:
            # 조건을 만족하는 파라미터가 없으면 최고 승률 파라미터 사용
            selected_config = best_config
            print(f"\n승률 {target_win_rate}% 이상이면서 하루 {target_pivots_per_day}개 이하인 파라미터 없음.")
            print(f"최고 승률 파라미터 사용: 승률 {selected_config['avg_win_rate']:.2f}%, 하루 {selected_config['avg_pivots_per_day']:.2f}개")
        
        # 최적 파라미터로 설정
        pivot_config = HybridAdaptivePivotConfig(
            base_pct=selected_config['base_pct'],
            base_multiplier=selected_config['base_multiplier'],
            atr_weight=selected_config['atr_weight'],
            confirmation_bars=int(selected_config['confirmation_bars']),
        )
        
        # 최종 전체 데이터 청크 기반 백테스팅
        print("\n" + "="*60)
        print("최종 전체 데이터 청크 기반 백테스팅")
        print("="*60)
        logger.info("최종 전체 데이터 청크 기반 백테스팅 시작")
        
        # 청크별 결과 집계
        all_trades = []
        all_daily_stats = []
        total_pivots = 0
        high_pivots = 0
        low_pivots = 0
        
        unique_dates = futures_df.index.normalize().unique()
        final_chunk_count = 0
        
        for chunk_start in range(0, len(unique_dates), chunk_days):
            chunk_end = min(chunk_start + chunk_days, len(unique_dates))
            chunk_dates = unique_dates[chunk_start:chunk_end]
            
            if len(chunk_dates) < 5:
                continue
            
            chunk_start_date = chunk_dates[0]
            chunk_end_date = chunk_dates[-1]
            
            chunk_futures = futures_df[(futures_df.index.normalize() >= chunk_start_date) & 
                                        (futures_df.index.normalize() <= chunk_end_date)].copy()
            
            if len(chunk_futures) == 0:
                continue
            
            final_chunk_count += 1
            logger.info(f"최종 청크 {final_chunk_count}: {chunk_start_date.date()} ~ {chunk_end_date.date()} ({len(chunk_futures)} 봉)")
            
            try:
                chunk_pivots = run_pivot_detection(chunk_futures, pivot_config, apply_filters=True)
                pivot_count = len(chunk_pivots)
                total_pivots += pivot_count
                high_pivots += len(chunk_pivots[chunk_pivots['is_high']])
                low_pivots += len(chunk_pivots[~chunk_pivots['is_high']])
                
                if pivot_count > 0:
                    chunk_result = run_backtest(chunk_futures, chunk_pivots, spot_pivots=None)
                    if 'trades' in chunk_result and len(chunk_result['trades']) > 0:
                        all_trades.append(chunk_result['trades'])
                    if 'daily_stats' in chunk_result and len(chunk_result['daily_stats']) > 0:
                        all_daily_stats.append(chunk_result['daily_stats'])
                
                logger.info(f"  청크 결과: 피봇 {pivot_count}개, 승률 {chunk_result['overall_win_rate']:.2f}%")
                
                del chunk_pivots
                del chunk_futures
                del chunk_result
                gc.collect()
                
            except Exception as e:
                logger.error(f"최종 청크 처리 실패: {str(e)}")
                logger.error(traceback.format_exc())
                gc.collect()
                continue
        
        # 전체 결과 집계
        if len(all_trades) > 0:
            all_trades_df = pd.concat(all_trades, ignore_index=True)
            total_trades = len(all_trades_df)
            total_wins = all_trades_df['is_win'].sum()
            overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            
            if len(all_daily_stats) > 0:
                all_daily_stats_df = pd.concat(all_daily_stats, ignore_index=True)
            else:
                all_daily_stats_df = pd.DataFrame()
            
            avg_pivots_per_day = total_pivots / total_days if total_days > 0 else 0
            
            print(f"\n최종 결과 (전체 {total_days}일, {final_chunk_count} 청크):")
            print(f"총 피봇: {total_pivots}개")
            print(f"고점 피봇: {high_pivots}개")
            print(f"저점 피봇: {low_pivots}개")
            print(f"평균 피봇/일: {avg_pivots_per_day:.2f}개")
            print(f"총 거래: {total_trades}건")
            print(f"총 승리: {total_wins}건")
            print(f"전체 승률: {overall_win_rate:.2f}%")
            
            if len(all_daily_stats_df) > 0:
                print(f"\n일별 승률 (최근 10일):")
                print(all_daily_stats_df.tail(10).to_string(index=False))
            
            # 결과 저장
            output_dir = Path(__file__).parent / 'data' / 'backtest_results'
            output_dir.mkdir(exist_ok=True)
            
            import json
            save_data = {
                'total_pivots': total_pivots,
                'high_pivots': high_pivots,
                'low_pivots': low_pivots,
                'avg_pivots_per_day': avg_pivots_per_day,
                'total_trades': total_trades,
                'total_wins': total_wins,
                'overall_win_rate': overall_win_rate,
                'daily_stats': all_daily_stats_df.to_dict('records') if len(all_daily_stats_df) > 0 else [],
                'trades': all_trades_df.to_dict('records') if len(all_trades_df) > 0 else [],
            }
            with open(output_dir / 'chunk_based_backtest.json', 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            if len(all_daily_stats_df) > 0:
                all_daily_stats_df.to_csv(output_dir / 'chunk_based_daily_stats.csv', index=False, encoding='utf-8-sig')
            
            if len(all_trades_df) > 0:
                all_trades_df.to_csv(output_dir / 'chunk_based_trades.csv', index=False, encoding='utf-8-sig')
            
            logger.info("최종 결과 저장 완료")
        else:
            print("유효한 거래가 없습니다")
            logger.warning("유효한 거래 없음")
        
        print("\n검증 완료")
        logger.info("피봇 탐색 성능 검증 완료")
        
    except Exception as e:
        logger.error(f"메인 함수 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"오류 발생: {str(e)}")
        print("자세한 내용은 로그 파일(pivot_backtest.log)을 확인하세요")


if __name__ == '__main__':
    main()
