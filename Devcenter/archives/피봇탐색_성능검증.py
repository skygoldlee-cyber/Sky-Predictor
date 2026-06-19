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
import optuna
import argparse
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
        
        # 필터 설정 (전역 변수 또는 기본값 사용)
        global _min_wave_pct, _min_pivot_interval_bars, _st_distance_threshold, _adx_hold_threshold
        min_wave_pct = _min_wave_pct if apply_filters else 0.3
        min_pivot_interval_bars = _min_pivot_interval_bars if apply_filters else 10
        st_distance_threshold = _st_distance_threshold if apply_filters else 0.1
        adx_hold_threshold = _adx_hold_threshold if apply_filters else 15.0
        
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
                        if st_distance_pct < st_distance_threshold:  # 너무 가까우면 필터
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
    parser = argparse.ArgumentParser(description='피봇 탐색 성능 검증')
    parser.add_argument('--mode', type=str, choices=['backtest', 'optuna', 'analyze', 'walkforward'], default='backtest',
                        help='실행 모드: backtest (기본 백테스트), optuna (파라미터 최적화), analyze (최적 파라미터로 상세 분석), walkforward (Walk-Forward Validation)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Optuna 시도 횟수 (optuna/walkforward 모드에서만 사용)')
    parser.add_argument('--n-chunks', type=int, default=5,
                        help='청크 수 (optuna/walkforward 모드에서만 사용)')
    parser.add_argument('--train-days', type=int, default=30,
                        help='Walk-Forward 트레이닝 기간 (일)')
    parser.add_argument('--test-days', type=int, default=7,
                        help='Walk-Forward 테스트 기간 (일)')
    parser.add_argument('--step-days', type=int, default=7,
                        help='Walk-Forward 윈도우 이동 간격 (일)')
    args = parser.parse_args()
    
    print(f"실행 모드: {args.mode}")
    print(f"시도 횟수: {args.n_trials}")
    print(f"청크 수: {args.n_chunks}")
    
    logger.info("="*60)
    logger.info(f"피봇 탐색 성능 검증 시작 (모드: {args.mode})")
    logger.info("="*60)
    
    try:
        # 데이터 경로
        db_path = str(Path(__file__).parent / 'data' / 'duckdb' / 'market_data.duckdb')
        logger.info(f"데이터베이스 경로: {db_path}")
        
        # Optuna 모드인 경우 데이터 크기 줄임 (빠른 최적화)
        if args.mode == 'optuna':
            print("선물 데이터 로드 중 (Optuna 모드: 60일)...")
            logger.info("선물 데이터 로드 시작 (60일)")
            futures_df = load_data_from_duckdb(db_path, 'futures_1min', days=60)
            print(f"선물 데이터: {len(futures_df)}건 ({futures_df.index[0]} ~ {futures_df.index[-1]})")
            
            # 현물 데이터는 Optuna 모드에서 필요 없음
            kospi_df = None
        else:
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
        if kospi_df is not None:
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
        
        # Optuna 모드인 경우 최적화 실행
        if args.mode == 'optuna':
            print("\n" + "="*60)
            print("Optuna 파라미터 최적화 (트레이닝/테스트 분리)")
            print("="*60)
            logger.info("Optuna 파라미터 최적화 시작 (트레이닝/테스트 분리)")
            
            # 전체 데이터 로드 (240일)
            logger.info("전체 데이터(240일) 로드 중...")
            futures_df_full = load_data_from_duckdb(db_path, 'futures_1min', days=240)
            print(f"전체 선물 데이터: {len(futures_df_full)}건 ({futures_df_full.index[0]} ~ {futures_df_full.index[-1]})")
            
            # 트레이닝/테스트 분리 (75% / 25%)
            total_days = len(futures_df_full.resample('D').size())
            train_days = int(total_days * 0.75)
            test_days = total_days - train_days
            
            print(f"\n데이터 분리:")
            print(f"전체 기간: {total_days}일")
            print(f"트레이닝 세트: {train_days}일 (75%)")
            print(f"테스트 세트: {test_days}일 (25%)")
            
            # 트레이닝 세트 (첫 75%)
            train_df = futures_df_full.iloc[:int(len(futures_df_full) * 0.75)]
            print(f"트레이닝 데이터: {len(train_df)}건 ({train_df.index[0]} ~ {train_df.index[-1]})")
            
            # 테스트 세트 (마지막 25%)
            test_df = futures_df_full.iloc[int(len(futures_df_full) * 0.75):]
            print(f"테스트 데이터: {len(test_df)}건 ({test_df.index[0]} ~ {test_df.index[-1]})")
            
            # 트레이닝 세트로 Optuna 최적화
            print("\n" + "="*60)
            print("트레이닝 세트로 파라미터 최적화")
            print("="*60)
            best_params = run_optuna_optimization(train_df, n_trials=args.n_trials, n_chunks=args.n_chunks)
            
            # 최적 파라미터로 테스트 세트 백테스트
            print("\n" + "="*60)
            print("테스트 세트로 성능 검증")
            print("="*60)
            logger.info("최적 파라미터로 테스트 세트 백테스트 시작")
            
            pivot_config = HybridAdaptivePivotConfig(
                base_pct=best_params['base_pct'],
                base_multiplier=best_params['base_multiplier'],
                atr_weight=best_params['atr_weight'],
                confirmation_bars=best_params['confirmation_bars'],
            )
            
            # 필터 파라미터 설정
            _min_wave_pct = best_params['min_wave_pct']
            _min_pivot_interval_bars = best_params['min_pivot_interval_bars']
            _st_distance_threshold = best_params['st_distance_threshold']
            _adx_hold_threshold = best_params['adx_hold_threshold']
            
            # 테스트 세트 피봇 탐색
            test_pivots = run_pivot_detection(test_df, pivot_config, apply_filters=True)
            print(f"테스트 세트 피봇: {len(test_pivots)}개")
            
            # 테스트 세트 백테스트
            test_backtest_result = run_backtest(test_df, test_pivots, spot_pivots=None)
            
            print(f"\n테스트 세트 백테스트 결과:")
            print(f"총 피봇: {test_backtest_result['total_pivots']}개")
            print(f"총 거래: {test_backtest_result['total_trades']}건")
            print(f"총 승리: {test_backtest_result['total_wins']}건")
            print(f"테스트 승률: {test_backtest_result['overall_win_rate']:.2f}%")
            
            # 트레이닝 세트도 백테스트하여 비교
            train_pivots = run_pivot_detection(train_df, pivot_config, apply_filters=True)
            train_backtest_result = run_backtest(train_df, train_pivots, spot_pivots=None)
            
            print(f"\n트레이닝 세트 백테스트 결과 (동일 파라미터):")
            print(f"총 피봇: {train_backtest_result['total_pivots']}개")
            print(f"총 거래: {train_backtest_result['total_trades']}건")
            print(f"총 승리: {train_backtest_result['total_wins']}건")
            print(f"트레이닝 승률: {train_backtest_result['overall_win_rate']:.2f}%")
            
            # 일반화 성능 분석
            print(f"\n일반화 성능 분석:")
            win_rate_diff = test_backtest_result['overall_win_rate'] - train_backtest_result['overall_win_rate']
            print(f"승률 차이 (테스트 - 트레이닝): {win_rate_diff:+.2f}%")
            
            if abs(win_rate_diff) < 5:
                print("✓ 일반화 성능 양호 (승률 차이 < 5%)")
            elif abs(win_rate_diff) < 10:
                print("⚠ 일반화 성능 보통 (승률 차이 5-10%)")
            else:
                print("✗ 과적합 가능성 (승률 차이 > 10%)")
            
            # 결과 저장
            output_dir = Path(__file__).parent / 'data' / 'backtest_results'
            output_dir.mkdir(exist_ok=True)
            
            import json
            save_data = {
                'data_split': {
                    'total_days': total_days,
                    'train_days': train_days,
                    'test_days': test_days,
                    'train_period': f"{train_df.index[0]} ~ {train_df.index[-1]}",
                    'test_period': f"{test_df.index[0]} ~ {test_df.index[-1]}",
                },
                'best_params': best_params,
                'train_result': {
                    'total_pivots': train_backtest_result['total_pivots'],
                    'total_trades': train_backtest_result['total_trades'],
                    'total_wins': train_backtest_result['total_wins'],
                    'overall_win_rate': train_backtest_result['overall_win_rate'],
                },
                'test_result': {
                    'total_pivots': test_backtest_result['total_pivots'],
                    'total_trades': test_backtest_result['total_trades'],
                    'total_wins': test_backtest_result['total_wins'],
                    'overall_win_rate': test_backtest_result['overall_win_rate'],
                },
                'generalization': {
                    'win_rate_diff': win_rate_diff,
                    'is_well_generalized': abs(win_rate_diff) < 5,
                }
            }
            with open(output_dir / 'optuna_train_test_split.json', 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            print("\n최적화 완료")
            logger.info("Optuna 파라미터 최적화 완료 (트레이닝/테스트 분리)")
            return
        
        # Analyze 모드: 최적 파라미터로 상세 거래 분석
        if args.mode == 'analyze':
            print("\n" + "="*60)
            print("최적 파라미터로 상세 거래 분석")
            print("="*60)
            logger.info("최적 파라미터로 상세 거래 분석 시작")
            
            # 최적 파라미터 로드
            output_dir = Path(__file__).parent / 'data' / 'backtest_results'
            optuna_result_path = output_dir / 'optuna_optimization.json'
            
            if not optuna_result_path.exists():
                print(f"오류: 최적 파라미터 파일이 없습니다: {optuna_result_path}")
                logger.error(f"최적 파라미터 파일 없음: {optuna_result_path}")
                return
            
            import json
            with open(optuna_result_path, 'r', encoding='utf-8') as f:
                optuna_result = json.load(f)
            
            best_params = optuna_result['best_params']
            print(f"\n최적 파라미터 (승률: {optuna_result['best_win_rate']:.2f}%):")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
            
            # 전체 데이터 로드 (240일)
            print("\n전체 데이터 로드 중 (240일)...")
            logger.info("전체 데이터(240일) 로드 중...")
            futures_df = load_data_from_duckdb(db_path, 'futures_1min', days=240)
            print(f"선물 데이터: {len(futures_df)}건 ({futures_df.index[0]} ~ {futures_df.index[-1]})")
            
            # 최적 파라미터로 피봇 설정
            pivot_config = HybridAdaptivePivotConfig(
                base_pct=best_params['base_pct'],
                base_multiplier=best_params['base_multiplier'],
                atr_weight=best_params['atr_weight'],
                confirmation_bars=best_params['confirmation_bars'],
            )
            
            # 필터 파라미터 설정
            _min_wave_pct = best_params['min_wave_pct']
            _min_pivot_interval_bars = best_params['min_pivot_interval_bars']
            _st_distance_threshold = best_params['st_distance_threshold']
            _adx_hold_threshold = best_params['adx_hold_threshold']
            
            # 피봇 탐색
            print("\n피봇 탐색 중...")
            logger.info("피봇 탐색 시작")
            pivots = run_pivot_detection(futures_df, pivot_config, apply_filters=True)
            print(f"탐색된 피봇: {len(pivots)}개")
            logger.info(f"탐색된 피봇: {len(pivots)}개")
            
            # 백테스트 실행
            print("\n백테스트 실행 중...")
            logger.info("백테스트 시작")
            backtest_result = run_backtest(futures_df, pivots, spot_pivots=None)
            
            # 결과 출력
            print("\n" + "="*60)
            print("백테스트 결과")
            print("="*60)
            print(f"총 피봇: {backtest_result['total_pivots']}개")
            print(f"총 거래: {backtest_result['total_trades']}건")
            print(f"총 승리: {backtest_result['total_wins']}건")
            print(f"전체 승률: {backtest_result['overall_win_rate']:.2f}%")
            
            # 총 수익 계산
            total_profit = 0
            if 'trades' in backtest_result and len(backtest_result['trades']) > 0:
                total_profit = backtest_result['trades']['profit'].sum()
            print(f"총 수익: {total_profit:.2f}")
            print(f"평균 일일 피봇: {backtest_result['total_pivots'] / max(1, len(futures_df.resample('D').size())):.2f}개")
            
            # 일별 통계
            if 'daily_stats' in backtest_result and len(backtest_result['daily_stats']) > 0:
                daily_stats = backtest_result['daily_stats']
                print(f"\n일별 통계 (총 {len(daily_stats)}일):")
                print(f"평균 일일 승률: {daily_stats['win_rate'].mean():.2f}%")
                print(f"최고 일일 승률: {daily_stats['win_rate'].max():.2f}%")
                print(f"최저 일일 승률: {daily_stats['win_rate'].min():.2f}%")
                print(f"평균 일일 거래: {daily_stats['total_trades'].mean():.1f}건")
                print(f"총 일일 수익: {daily_stats['total_profit'].sum():.2f}")
                
                # 승률 60% 이상인 날
                high_win_days = daily_stats[daily_stats['win_rate'] >= 60]
                print(f"승률 60% 이상인 날: {len(high_win_days)}일 ({len(high_win_days)/len(daily_stats)*100:.1f}%)")
                
                # 수익 상위 10일
                top_profit_days = daily_stats.nlargest(10, 'total_profit')
                print(f"\n수익 상위 10일:")
                for idx, row in top_profit_days.iterrows():
                    print(f"  {row['date']}: 승률 {row['win_rate']:.1f}%, 수익 {row['total_profit']:.2f}, 거래 {row['total_trades']}건")
            
            # 거래 상세 분석
            if 'trades' in backtest_result and len(backtest_result['trades']) > 0:
                trades_df = backtest_result['trades']
                print(f"\n거래 상세 분석:")
                print(f"평균 수익: {trades_df['profit'].mean():.2f}")
                print(f"최대 수익: {trades_df['profit'].max():.2f}")
                print(f"최대 손실: {trades_df['profit'].min():.2f}")
                print(f"수익 표준편차: {trades_df['profit'].std():.2f}")
                
                # 롱/숏 별 분석
                long_trades = trades_df[trades_df['position'] == 'long']
                short_trades = trades_df[trades_df['position'] == 'short']
                
                if len(long_trades) > 0:
                    long_win_rate = (long_trades['profit'] > 0).sum() / len(long_trades) * 100
                    print(f"\n롱 거래: {len(long_trades)}건, 승률 {long_win_rate:.2f}%, 평균 수익 {long_trades['profit'].mean():.2f}")
                
                if len(short_trades) > 0:
                    short_win_rate = (short_trades['profit'] > 0).sum() / len(short_trades) * 100
                    print(f"숏 거래: {len(short_trades)}건, 승률 {short_win_rate:.2f}%, 평균 수익 {short_trades['profit'].mean():.2f}")
                
                # 거래 보유 시간 분석
                trades_df['hold_hours'] = (pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])).dt.total_seconds() / 3600
                print(f"\n평균 보유 시간: {trades_df['hold_hours'].mean():.1f}시간")
                print(f"최단 보유 시간: {trades_df['hold_hours'].min():.1f}시간")
                print(f"최장 보유 시간: {trades_df['hold_hours'].max():.1f}시간")
            
            # 결과 저장
            analyze_result = {
                'best_params': best_params,
                'best_win_rate_from_optuna': optuna_result['best_win_rate'],
                'backtest_result': {
                    'total_pivots': backtest_result['total_pivots'],
                    'total_trades': backtest_result['total_trades'],
                    'total_wins': backtest_result['total_wins'],
                    'overall_win_rate': backtest_result['overall_win_rate'],
                    'total_profit': total_profit,
                },
                'data_period': f"{futures_df.index[0]} ~ {futures_df.index[-1]}",
                'data_days': len(futures_df.resample('D').size()),
            }
            
            if 'daily_stats' in backtest_result:
                analyze_result['daily_stats'] = backtest_result['daily_stats'].to_dict('records')
            
            if 'trades' in backtest_result:
                analyze_result['trades'] = backtest_result['trades'].to_dict('records')
            
            with open(output_dir / 'optuna_detailed_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(analyze_result, f, indent=2, default=str)
            
            print(f"\n상세 분석 결과 저장: {output_dir / 'optuna_detailed_analysis.json'}")
            logger.info("최적 파라미터 상세 분석 완료")
            return
        
        # Walk-Forward Validation 모드
        if args.mode == 'walkforward':
            print("\n" + "="*60)
            print("Walk-Forward Validation")
            print("="*60)
            logger.info("Walk-Forward Validation 시작")
            
            print(f"트레이닝 기간: {args.train_days}일")
            print(f"테스트 기간: {args.test_days}일")
            print(f"이동 간격: {args.step_days}일")
            
            # 전체 데이터 로드 (최대 240일)
            print("\n전체 데이터 로드 중...")
            logger.info("전체 데이터 로드 중...")
            futures_df = load_data_from_duckdb(db_path, 'futures_1min', days=240)
            print(f"선물 데이터: {len(futures_df)}건 ({futures_df.index[0]} ~ {futures_df.index[-1]})")
            
            # Walk-Forward 윈도우 생성
            total_days = len(futures_df.resample('D').size())
            windows = []
            
            # 날짜 기반으로 윈도우 생성
            dates = futures_df.resample('D').size().index.tolist()
            
            start_idx = 0
            while True:
                train_end_idx = start_idx + args.train_days
                test_end_idx = train_end_idx + args.test_days
                
                if test_end_idx >= len(dates):
                    break
                
                train_start_date = dates[start_idx]
                train_end_date = dates[train_end_idx - 1] + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
                test_start_date = dates[train_end_idx]
                test_end_date = dates[test_end_idx - 1] + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
                
                windows.append({
                    'train_start': train_start_date,
                    'train_end': train_end_date,
                    'test_start': test_start_date,
                    'test_end': test_end_date,
                })
                
                start_idx += args.step_days
            
            print(f"\n생성된 윈도우: {len(windows)}개")
            logger.info(f"Walk-Forward 윈도우 수: {len(windows)}")
            
            # 각 윈도우에 대해 최적화 및 테스트
            all_results = []
            
            for i, window in enumerate(windows):
                print(f"\n{'='*60}")
                print(f"윈도우 {i+1}/{len(windows)}")
                print(f"{'='*60}")
                print(f"트레이닝: {window['train_start']} ~ {window['train_end']}")
                print(f"테스트: {window['test_start']} ~ {window['test_end']}")
                
                # 트레이닝 데이터 추출
                train_df = futures_df.loc[window['train_start']:window['train_end']]
                test_df = futures_df.loc[window['test_start']:window['test_end']]
                
                print(f"트레이닝 데이터: {len(train_df)}건")
                print(f"테스트 데이터: {len(test_df)}건")
                
                # 트레이닝 데이터로 Optuna 최적화
                print("\n트레이닝 데이터로 파라미터 최적화 중...")
                logger.info(f"윈도우 {i+1} 트레이닝 최적화 시작")
                
                best_params = run_optuna_optimization(train_df, n_trials=args.n_trials, n_chunks=args.n_chunks)
                
                # 최적 파라미터로 테스트 데이터 백테스트
                print("\n테스트 데이터로 백테스트 중...")
                logger.info(f"윈도우 {i+1} 테스트 백테스트 시작")
                
                pivot_config = HybridAdaptivePivotConfig(
                    base_pct=best_params['base_pct'],
                    base_multiplier=best_params['base_multiplier'],
                    atr_weight=best_params['atr_weight'],
                    confirmation_bars=best_params['confirmation_bars'],
                )
                
                _min_wave_pct = best_params['min_wave_pct']
                _min_pivot_interval_bars = best_params['min_pivot_interval_bars']
                _st_distance_threshold = best_params['st_distance_threshold']
                _adx_hold_threshold = best_params['adx_hold_threshold']
                
                test_pivots = run_pivot_detection(test_df, pivot_config, apply_filters=True)
                
                if len(test_pivots) == 0:
                    print("경고: 테스트 데이터에서 피봇이 감지되지 않음")
                    window_result = {
                        'window': i + 1,
                        'train_period': f"{window['train_start']} ~ {window['train_end']}",
                        'test_period': f"{window['test_start']} ~ {window['test_end']}",
                        'best_params': best_params,
                        'test_pivots': 0,
                        'test_trades': 0,
                        'test_wins': 0,
                        'test_win_rate': 0,
                        'test_profit': 0,
                    }
                    all_results.append(window_result)
                    print(f"\n윈도우 {i+1} 결과:")
                    print(f"  테스트 승률: 0.00%")
                    print(f"  테스트 수익: 0.00")
                    print(f"  테스트 거래: 0건")
                    continue
                
                test_backtest = run_backtest(test_df, test_pivots, spot_pivots=None)
                
                # 결과 저장
                window_result = {
                    'window': i + 1,
                    'train_period': f"{window['train_start']} ~ {window['train_end']}",
                    'test_period': f"{window['test_start']} ~ {window['test_end']}",
                    'best_params': best_params,
                    'test_pivots': test_backtest['total_pivots'],
                    'test_trades': test_backtest['total_trades'],
                    'test_wins': test_backtest['total_wins'],
                    'test_win_rate': test_backtest['overall_win_rate'],
                }
                
                # 총 수익 계산
                if 'trades' in test_backtest and len(test_backtest['trades']) > 0:
                    window_result['test_profit'] = test_backtest['trades']['profit'].sum()
                else:
                    window_result['test_profit'] = 0
                
                all_results.append(window_result)
                
                print(f"\n윈도우 {i+1} 결과:")
                print(f"  테스트 승률: {window_result['test_win_rate']:.2f}%")
                print(f"  테스트 수익: {window_result['test_profit']:.2f}")
                print(f"  테스트 거래: {window_result['test_trades']}건")
            
            # 전체 결과 요약
            print("\n" + "="*60)
            print("Walk-Forward Validation 결과 요약")
            print("="*60)
            
            win_rates = [r['test_win_rate'] for r in all_results]
            profits = [r['test_profit'] for r in all_results]
            
            print(f"평균 테스트 승률: {sum(win_rates)/len(win_rates):.2f}%")
            print(f"최고 테스트 승률: {max(win_rates):.2f}%")
            print(f"최저 테스트 승률: {min(win_rates):.2f}%")
            print(f"승률 표준편차: {pd.Series(win_rates).std():.2f}%")
            print(f"\n평균 테스트 수익: {sum(profits)/len(profits):.2f}")
            print(f"총 테스트 수익: {sum(profits):.2f}")
            print(f"수익 표준편차: {pd.Series(profits).std():.2f}")
            
            # 결과 저장
            walkforward_result = {
                'config': {
                    'train_days': args.train_days,
                    'test_days': args.test_days,
                    'step_days': args.step_days,
                    'n_trials': args.n_trials,
                    'n_chunks': args.n_chunks,
                },
                'summary': {
                    'avg_win_rate': sum(win_rates)/len(win_rates),
                    'max_win_rate': max(win_rates),
                    'min_win_rate': min(win_rates),
                    'win_rate_std': pd.Series(win_rates).std(),
                    'avg_profit': sum(profits)/len(profits),
                    'total_profit': sum(profits),
                    'profit_std': pd.Series(profits).std(),
                },
                'windows': all_results,
            }
            
            with open(output_dir / 'walkforward_validation.json', 'w', encoding='utf-8') as f:
                json.dump(walkforward_result, f, indent=2, default=str)
            
            print(f"\nWalk-Forward 결과 저장: {output_dir / 'walkforward_validation.json'}")
            logger.info("Walk-Forward Validation 완료")
            return
        
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


def optuna_objective(trial, df, n_chunks=5):
    """Optuna objective 함수 - 승률 최대화
    
    Args:
        trial: Optuna trial 객체
        df: OHLCV 데이터
        n_chunks: 청크 수
    
    Returns:
        승률 (0~100)
    """
    # 피봇 파라미터 최적화
    base_pct = trial.suggest_float('base_pct', 0.05, 2.0)
    base_multiplier = trial.suggest_float('base_multiplier', 0.5, 10.0)
    atr_weight = trial.suggest_float('atr_weight', 0.0, 1.0)
    confirmation_bars = trial.suggest_int('confirmation_bars', 0, 10)
    
    # 필터 파라미터 최적화 (범위 조정 - 더 넓게)
    min_wave_pct = trial.suggest_float('min_wave_pct', 0.05, 2.0)  # 더 넓은 범위
    min_pivot_interval_bars = trial.suggest_int('min_pivot_interval_bars', 1, 30)  # 더 넓은 범위
    st_distance_threshold = trial.suggest_float('st_distance_threshold', 0.01, 1.0)  # 더 넓은 범위
    adx_hold_threshold = trial.suggest_float('adx_hold_threshold', 5.0, 50.0)  # 더 넓은 범위
    
    try:
        # 청크 기반 백테스팅
        total_win_rate = 0.0
        valid_chunks = 0
        
        chunk_size = len(df) // n_chunks
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else len(df)
            chunk_df = df.iloc[start_idx:end_idx].copy()
            
            if len(chunk_df) < 100:  # 너무 작은 청크는 건너뜀
                continue
            
            # 피봇 탐색 (필터 적용)
            pivot_config = HybridAdaptivePivotConfig(
                base_pct=base_pct,
                base_multiplier=base_multiplier,
                atr_weight=atr_weight,
                confirmation_bars=confirmation_bars,
            )
            
            # 필터 파라미터를 전역 변수로 설정 (임시 해결책)
            global _min_wave_pct, _min_pivot_interval_bars, _st_distance_threshold, _adx_hold_threshold
            _min_wave_pct = min_wave_pct
            _min_pivot_interval_bars = min_pivot_interval_bars
            _st_distance_threshold = st_distance_threshold
            _adx_hold_threshold = adx_hold_threshold
            
            pivots = run_pivot_detection(chunk_df, pivot_config, apply_filters=True)
            
            if len(pivots) == 0:
                continue
            
            # 백테스팅
            backtest_result = run_backtest(chunk_df, pivots, spot_pivots=None)
            win_rate = backtest_result['overall_win_rate']
            
            # 피봇 수가 너무 적으면 페널티 (완화)
            pivots_per_day = len(pivots) / max(1, len(chunk_df.resample('D').size()))
            if pivots_per_day < 2:  # 너무 적은 피봇 (기준 낮춤)
                win_rate *= 0.7  # 페널티 완화
            elif pivots_per_day > 100:  # 너무 많은 피봇 (기준 높임)
                win_rate *= 0.9  # 페널티 완화
            
            total_win_rate += win_rate
            valid_chunks += 1
        
        if valid_chunks == 0:
            return 0.0
        
        avg_win_rate = total_win_rate / valid_chunks
        return avg_win_rate
        
    except Exception as e:
        logger.error(f"Optuna trial 실패: {str(e)}")
        return 0.0


# 전역 필터 파라미터 (Optuna에서 사용)
_min_wave_pct = 0.3
_min_pivot_interval_bars = 10
_st_distance_threshold = 0.1
_adx_hold_threshold = 15.0


def run_optuna_optimization(df, n_trials=50, n_chunks=5):
    """Optuna를 사용한 파라미터 최적화
    
    Args:
        df: OHLCV 데이터
        n_trials: 시도 횟수
        n_chunks: 청크 수
    """
    logger.info(f"Optuna 최적화 시작: n_trials={n_trials}, n_chunks={n_chunks}")
    
    # Objective 함수 래핑
    def objective(trial):
        return optuna_objective(trial, df, n_chunks)
    
    # Study 생성
    study = optuna.create_study(direction='maximize')
    
    # 최적화 실행
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("Optuna 최적화 결과")
    print("=" * 80)
    print(f"최적 파라미터: {study.best_params}")
    print(f"최적 승률: {study.best_value:.2f}%")
    print(f"시도 횟수: {len(study.trials)}")
    print("=" * 80)
    
    # 상위 5개 결과
    print("\n상위 5개 결과:")
    print("-" * 80)
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
    for i, trial in enumerate(sorted_trials, 1):
        print(f"#{i}: {trial.params} → 승률: {trial.value:.2f}%")
    print("-" * 80)
    
    # 결과 저장
    output_dir = Path(__file__).parent / 'data' / 'backtest_results'
    output_dir.mkdir(exist_ok=True)
    
    import json
    result = {
        'best_params': study.best_params,
        'best_win_rate': study.best_value,
        'n_trials': len(study.trials),
        'top_trials': [
            {'params': t.params, 'win_rate': t.value}
            for t in sorted_trials
        ]
    }
    
    with open(output_dir / 'optuna_optimization.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Optuna 최적화 완료: 최적 승률 {study.best_value:.2f}%")
    return study.best_params


if __name__ == '__main__':
    main()
