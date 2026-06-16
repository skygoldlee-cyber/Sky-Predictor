import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from datetime import datetime
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from indicators.adaptive_zigzag import AdaptiveZigZag, AdaptiveZigZagConfig

# ──────────────────────────────────────────────
# 데이터 로드 (기존 유지)
# ──────────────────────────────────────────────
def load_saved_minute_data(symbol: str) -> pd.DataFrame:
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/minute_bars')
    prefix = f'minute_bars_{symbol}_'
    files = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix)])
    if not files:
        raise FileNotFoundError(f"No saved data found for {symbol}")
    latest_file = os.path.join(data_dir, files[-1])
    return pd.read_csv(latest_file, index_col=0, parse_dates=True)


# ──────────────────────────────────────────────
# 데이터 요약 출력 (Phase 5: 2026-05-10)
# ──────────────────────────────────────────────
def print_data_summary(df: pd.DataFrame, symbol: str, atr_period: int = 14):
    """데이터프레임 요약 정보 출력"""
    logger.info(f"{'='*80}")
    logger.info(f"[데이터 요약] {symbol.upper()}")
    logger.info(f"{'='*80}")

    # 기본 정보
    logger.info("[기본 정보]")
    logger.info(f"  데이터 크기: {len(df):,} 봉")
    logger.info(f"  기간: {df.index[0]} ~ {df.index[-1]}")
    logger.info(f"  거래일 수: {len(df.index.date)}일")

    # 가격 통계
    logger.info("[가격 통계]")
    logger.info(f"  종가 범위: {df['Close'].min():.2f} ~ {df['Close'].max():.2f}")
    logger.info(f"  종가 평균: {df['Close'].mean():.2f}")
    logger.info(f"  종가 표준편차: {df['Close'].std():.2f}")
    logger.info(f"  총 변동률: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:.2f}%")

    # ATR 계산
    try:
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values

        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        atr = pd.Series(tr).rolling(window=atr_period).mean().iloc[-1]
        atr_pct = atr / df['Close'].iloc[-1] * 100

        logger.info(f"[ATR 통계 (기간: {atr_period})]")
        logger.info(f"  ATR: {atr:.4f}")
        logger.info(f"  ATR (%): {atr_pct:.2f}%")
        logger.info(f"  평균 ATR (전체): {tr[atr_period:].mean():.4f}")
        logger.info(f"  ATR 표준편차: {tr[atr_period:].std():.4f}")
    except Exception as e:
        logger.warning(f"[ATR 통계] 계산 실패: {e}")

    # 거래량 통계
    if 'Volume' in df.columns:
        logger.info("[거래량 통계]")
        logger.info(f"  평균 거래량: {df['Volume'].mean():,.0f}")
        logger.info(f"  최대 거래량: {df['Volume'].max():,.0f}")
        logger.info(f"  최소 거래량: {df['Volume'].min():,.0f}")

    # 결측치 확인
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info("[결측치]")
        for col, count in missing[missing > 0].items():
            logger.info(f"  {col}: {count}개 ({count/len(df)*100:.2f}%)")
    else:
        logger.info("[결측치] 없음")

    logger.info(f"{'='*80}")

# ──────────────────────────────────────────────
# Phase 1: 향상된 평가지표
# ──────────────────────────────────────────────

# 스코어 가중치 상수 (DESIGN-1 수정)
SCORE_WEIGHTS = {
    'lag_weight': 1.0,
    'quality_weight': 10.0,
    'alternation_weight': 5.0,
    'pivot_weight': 0.5
}

def calculate_pivot_quality(zz: AdaptiveZigZag, df: pd.DataFrame) -> float:
    """피봇 품질 계산 (실제 추세 전환점 탐지율)"""
    confirmed_swings = [s for s in zz._all_swings if s.confirmed]
    if len(confirmed_swings) < 2:
        return 0.0
    
    # BUG-2 수정: 피봇 이후 N봉 동안 추세 지속 여부 확인
    correct_count = 0
    lookforward_bars = 10  # 피봇 후 10봉 확인
    
    for i, s in enumerate(confirmed_swings[:-1]):
        pivot_idx = s.index
        
        # 피봇 후 N봉 범위
        end_idx = min(pivot_idx + lookforward_bars, len(df))
        if end_idx <= pivot_idx + 1:
            continue
        
        future_df = df.iloc[pivot_idx + 1:end_idx]
        
        if s.swing_type.value == 'high':
            # HIGH 피봇 후 하락 추세 확인
            if future_df['Close'].min() < s.price:
                correct_count += 1
        elif s.swing_type.value == 'low':
            # LOW 피봇 후 상승 추세 확인
            if future_df['Close'].max() > s.price:
                correct_count += 1
    
    return correct_count / len(confirmed_swings) if confirmed_swings else 0.0

def calculate_alternation_rate(swings: list) -> float:
    """H/L 교번 준수율 계산"""
    if len(swings) < 2:
        return 1.0
    
    violations = 0
    for i in range(1, len(swings)):
        if swings[i].swing_type == swings[i-1].swing_type:
            violations += 1
    
    return 1.0 - (violations / len(swings))

def calculate_lag_metrics(lag_details: list) -> dict:
    """지연시간 통계 메트릭 계산"""
    avg_lag = float(np.mean(lag_details))
    median_lag = float(np.median(lag_details))
    max_lag = int(max(lag_details))
    lag_std = float(np.std(lag_details))
    lag_p95 = float(np.percentile(lag_details, 95)) if len(lag_details) > 0 else 0.0
    
    # 복합 지연시간 스코어
    lag_score = avg_lag * 0.5 + lag_p95 * 0.3 + lag_std * 0.2
    
    return {
        'avg_lag': avg_lag,
        'median_lag': median_lag,
        'max_lag': max_lag,
        'lag_std': lag_std,
        'lag_p95': lag_p95,
        'lag_score': lag_score
    }

def calculate_metrics_enhanced(zz: AdaptiveZigZag, df: pd.DataFrame, 
                               target_pivots: float) -> dict:
    """향상된 평가지표 (Phase 1)"""
    confirmed_swings = [s for s in zz._all_swings if s.confirmed]
    pivot_count = len(confirmed_swings)
    
    if pivot_count < 2:
        return {'score': 9999.0, 'avg_lag': 999.0, 'max_lag': 999, 
                'pivot_count': pivot_count, 'pivot_quality': 0.0, 
                'alternation_rate': 0.0}
    
    # 1. 지연시간 메트릭
    lag_details = []
    for s in sorted(confirmed_swings, key=lambda s: s.index):
        pivot_index = s.index
        confirm_index = getattr(s, 'confirmed_at_idx', pivot_index)
        lag_details.append(confirm_index - pivot_index)
    
    lag_metrics = calculate_lag_metrics(lag_details)
    
    # 2. 피봇 품질 메트릭 (진짜 추세 전환점 탐지)
    pivot_quality = calculate_pivot_quality(zz, df)
    
    # 3. H/L 교번 준수율
    alternation_rate = calculate_alternation_rate(confirmed_swings)
    
    # 4. 복합 스코어 (DESIGN-1 수정: 상수 사용)
    # 지연시간 (낮을수록 좋음) + 피봇 품질 (높을수록 좋음) + 교번 준수율 (높을수록 좋음)
    lag_score = lag_metrics['lag_score'] * SCORE_WEIGHTS['lag_weight']
    quality_penalty = (1.0 - pivot_quality) * SCORE_WEIGHTS['quality_weight']
    alternation_penalty = (1.0 - alternation_rate) * SCORE_WEIGHTS['alternation_weight']
    pivot_penalty = abs(pivot_count - target_pivots) * SCORE_WEIGHTS['pivot_weight']
    
    score = lag_score + quality_penalty + alternation_penalty + pivot_penalty
    
    return {
        'avg_lag': lag_metrics['avg_lag'],
        'median_lag': lag_metrics['median_lag'],
        'max_lag': lag_metrics['max_lag'],
        'lag_std': lag_metrics['lag_std'],
        'lag_p95': lag_metrics['lag_p95'],
        'pivot_count': pivot_count,
        'pivot_quality': pivot_quality,
        'alternation_rate': alternation_rate,
        'score': round(score, 4)
    }

# 기존 함수 호환성 유지
def calculate_metrics(zz: AdaptiveZigZag, target_pivots: float) -> dict:
    """기존 평가지표 (하위 호환성 유지)"""
    confirmed_swings = [s for s in zz._all_swings if s.confirmed]
    pivot_count = len(confirmed_swings)
    
    if pivot_count < 2:
        return {'avg_lag': 999.0, 'max_lag': 999, 'pivot_count': pivot_count, 'score': 9999.0}

    lag_details = []
    for s in sorted(confirmed_swings, key=lambda s: s.index):
        pivot_index = s.index
        confirm_index = getattr(s, 'confirmed_at_idx', pivot_index)
        lag_details.append(confirm_index - pivot_index)

    avg_lag = float(np.mean(lag_details))
    
    pivot_penalty = abs(pivot_count - target_pivots) * 0.5 
    score = avg_lag + pivot_penalty

    return {
        'avg_lag': avg_lag,
        'max_lag': int(max(lag_details)),
        'pivot_count': pivot_count,
        'score': round(score, 4)
    }

# ──────────────────────────────────────────────
# 단일 파라미터 조합 평가 (Phase 2: 확장된 파라미터 그리드 지원)
# ──────────────────────────────────────────────
def evaluate_single(params: tuple, df: pd.DataFrame, zigzag_cfg: dict,
                    pivot_count_range: tuple, target_pivots: float,
                    use_enhanced_metrics: bool = True) -> dict | None:
    # Phase 2: 파라미터 그리드 확장 (atr_period, freeze_on_confirm 지원)
    # Phase 5: min_wave_pct 파라미터 추가 (2026-05-10)
    if len(params) == 4:
        # 기존 4개 파라미터 (하위 호환성)
        atr_mult, thresh_min, conf_bars, min_wave_atr = params
        atr_period = zigzag_cfg.get('atr_period', 14)
        freeze_on_confirm = zigzag_cfg.get('freeze_on_confirm', True)
        min_wave_pct = zigzag_cfg.get('min_wave_pct', 0.0)
    elif len(params) == 5:
        # Phase 5: 5개 파라미터 (min_wave_pct 추가)
        atr_mult, thresh_min, conf_bars, min_wave_atr, min_wave_pct = params
        atr_period = zigzag_cfg.get('atr_period', 14)
        freeze_on_confirm = zigzag_cfg.get('freeze_on_confirm', True)
    elif len(params) == 7:
        # Phase 2: 7개 파라미터 (atr_period, freeze_on_confirm, min_wave_pct 추가)
        atr_mult, thresh_min, conf_bars, min_wave_atr, min_wave_pct, atr_period, freeze_on_confirm = params
    elif len(params) == 6:
        # Phase 2: 6개 파라미터 (atr_period, freeze_on_confirm 추가) - 하위 호환성
        atr_mult, thresh_min, conf_bars, min_wave_atr, atr_period, freeze_on_confirm = params
        min_wave_pct = zigzag_cfg.get('min_wave_pct', 0.0)
    else:
        return None
    
    try:
        # Config 빌드
        g = zigzag_cfg.get
        thresh_max = g('pivot_threshold_max_pct', 0.3)
        actual_thresh_min = min(thresh_min, thresh_max)

        cfg = AdaptiveZigZagConfig(
            atr_period=int(atr_period),
            atr_multiplier=atr_mult,
            pivot_threshold_min_pct=actual_thresh_min,
            pivot_threshold_max_pct=thresh_max,
            confirmation_bars=int(conf_bars),
            atr_multiplier_min=g('atr_multiplier_min', 0.3),
            atr_multiplier_max=g('atr_multiplier_max', 1.0),
            major_swing_ratio=g('major_swing_ratio', 1.2),
            max_swings=g('max_swings', 20),
            freeze_on_confirm=bool(freeze_on_confirm),
            min_wave_bars=g('min_wave_bars', 1),
            min_wave_pct=min_wave_pct,  # Phase 5: 파라미터에서 사용 (2026-05-10)
            cluster_tolerance_pct=g('cluster_tolerance_pct', 0.3),
            structure_lookback_swings=g('structure_lookback_swings', 8),
            structure_points=g('structure_points', 3),
        )
        
        zz = AdaptiveZigZag(cfg)
        zz.compute_from_df(df)
        
        # Phase 1: 향상된 평가지표 사용
        if use_enhanced_metrics:
            metrics = calculate_metrics_enhanced(zz, df, target_pivots)
        else:
            metrics = calculate_metrics(zz, target_pivots)

        # 피봇 수 범위 필터링
        if not (pivot_count_range[0] <= metrics['pivot_count'] <= pivot_count_range[1]):
            return None

        result = {
            'atr_multiplier': round(float(atr_mult), 4),
            'pivot_threshold_min_pct': round(float(thresh_min), 4),
            'confirmation_bars': int(conf_bars),
            'min_wave_atr_ratio': round(float(min_wave_atr), 4),
            'min_wave_pct': round(float(min_wave_pct), 4),  # Phase 5: min_wave_pct 추가 (2026-05-10)
            **metrics
        }

        # Phase 2: 추가 파라미터 결과에 포함
        if len(params) == 7:
            result['atr_period'] = int(atr_period)
            result['freeze_on_confirm'] = bool(freeze_on_confirm)
        elif len(params) == 6:
            result['atr_period'] = int(atr_period)
            result['freeze_on_confirm'] = bool(freeze_on_confirm)
        
        return result
    except Exception:
        return None

# ──────────────────────────────────────────────
# 병렬 그리드 탐색 (Phase 4: 배치 처리 및 메모리 최적화)
# ──────────────────────────────────────────────
def run_grid(param_grid: list[tuple], df: pd.DataFrame, zigzag_cfg: dict,
             n_jobs: int, label: str, pivot_count_range: tuple,
             use_enhanced_metrics: bool = True, batch_size: int = 100) -> pd.DataFrame:
    
    target_pivots = sum(pivot_count_range) / 2
    
    # BUG-6 수정: 사용하지 않는 df_values 코드 삭제
    
    results = []
    
    # Phase 4: 배치 처리로 메모리 효율화
    with tqdm(total=len(param_grid), desc=f"  {label}", ncols=90) as pbar:
        for i in range(0, len(param_grid), batch_size):
            batch = param_grid[i:i+batch_size]
            
            parallel_pool = Parallel(n_jobs=n_jobs, backend='loky')
            batch_results = parallel_pool(
                delayed(evaluate_single)(p, df, zigzag_cfg, pivot_count_range, 
                                        target_pivots, use_enhanced_metrics) 
                for p in batch
            )
            
            for result in batch_results:
                if result is not None:
                    results.append(result)
            pbar.update(len(batch))

    if not results: return pd.DataFrame()

    # score(낮을수록 좋음) -> avg_lag 순으로 정렬
    return pd.DataFrame(results).sort_values(['score', 'avg_lag']).reset_index(drop=True)

# ──────────────────────────────────────────────
# Phase 3: 시기별 균형 샘플링
# ──────────────────────────────────────────────
def load_balanced_sample(df: pd.DataFrame, sample_ratio: float = 0.3) -> pd.DataFrame:
    """시기별 균형 샘플링 (Phase 3, BUG-3 수정)"""
    # 전체 데이터를 시기별로 분할
    total_len = len(df)
    period_size = total_len // 4  # 4개 시기로 분할
    
    samples = []
    for i in range(4):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < 3 else total_len
        period_df = df.iloc[start_idx:end_idx]
        
        # BUG-3 수정: 각 시기에서 균등하게 샘플링 (마지막 N봉이 아닌 전체에서 균등 샘플링)
        period_sample_size = int(len(period_df) * sample_ratio)
        if period_sample_size > 0:
            # stride로 균등 샘플링
            stride = max(1, len(period_df) // period_sample_size)
            sampled_df = period_df.iloc[::stride].head(period_sample_size)
            samples.append(sampled_df)
    
    if not samples:
        return df
    
    return pd.concat(samples)

def time_series_cv_optimize(symbol: str, config_type: str, n_jobs: int = -1,
                           pivot_count_range: tuple = (5, 9), max_avg_lag: float = 5.0,
                           n_splits: int = 5, use_enhanced_metrics: bool = True,
                           use_extended_params: bool = False):
    """시계열 교차 검증 기반 최적화 (Phase 3)"""
    logger.info(f"Phase 3 Time Series CV Optimization Start: {symbol} ({config_type})")
    logger.info(f"CV splits: {n_splits}, Extended params: {use_extended_params}")
    
    df = load_saved_minute_data(symbol)
    fold_size = len(df) // n_splits
    
    results = []
    for i in range(n_splits):
        # Train: 0~(i+1)*fold_size, Test: (i+1)*fold_size~(i+2)*fold_size
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = min((i + 2) * fold_size, len(df))
        
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]
        
        logger.info(f"Fold {i+1}/{n_splits}: Train={len(train_df)}, Test={len(test_df)}")
        
        # Train으로 최적화
        train_results = optimize_parameters_on_df(train_df, symbol, config_type, n_jobs,
                                                  pivot_count_range, max_avg_lag,
                                                  use_enhanced_metrics, use_extended_params)
        
        if train_results is None or train_results.empty:
            continue
        
        best_train = train_results.iloc[0]
        
        # Test로 검증
        test_metrics = validate_on_test(test_df, best_train.to_dict(), config_type,
                                       pivot_count_range, use_enhanced_metrics)
        
        if test_metrics:
            results.append({
                'fold': i + 1,
                'params': best_train.to_dict(),
                'test_score': test_metrics['test_score'],
                'test_avg_lag': test_metrics['test_avg_lag'],
                'test_pivot_count': test_metrics['test_pivot_count']
            })
    
    if not results:
        logger.warning("No valid CV results, falling back to standard optimization")
        return train_test_split_optimize(symbol, config_type, n_jobs,
                                         pivot_count_range, max_avg_lag,
                                         0.2, use_enhanced_metrics, use_extended_params)
    
    # 테스트 스코어 기준 최적 파라미터 선택
    best_result = min(results, key=lambda x: x['test_score'])
    logger.info(f"Best fold: {best_result['fold']}, Test score: {best_result['test_score']:.4f}")
    
    return pd.DataFrame([best_result['params']])

def optimize_parameters_on_df(df: pd.DataFrame, symbol: str, config_type: str, n_jobs: int = -1,
                                pivot_count_range: tuple = (5, 9), max_avg_lag: float = 5.0,
                                use_enhanced_metrics: bool = True, use_extended_params: bool = False):
    """특정 데이터프레임으로 최적화 (Phase 3 헬퍼 함수)"""
    # Config 로드
    config_path = os.path.join(os.path.dirname(__file__), '../../config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    base_cfg = config.get('adaptive_indicator', {}).get('zigzag', {})
    zigzag_cfg = config.get('adaptive_indicator', {}).get(config_type, base_cfg).copy()

    # 파라미터 범위 설정
    if symbol == 'kospi':
        min_wave_range_coarse = np.arange(0.3, 2.5, 0.7)
        coarse_conf_bars = [1]
    else:
        min_wave_range_coarse = np.arange(0.5, 3.5, 1.0)
        coarse_conf_bars = [1, 2, 3]

    thresh_max = zigzag_cfg.get('pivot_threshold_max_pct', 0.3)
    coarse_thresh_range = np.arange(0.01, min(0.5, thresh_max), 0.1)

    # Phase 5: min_wave_pct 범위 추가 (2026-05-10)
    # 현재 config.json 설정(0.1)을 중심으로 범위 설정
    current_min_wave_pct = zigzag_cfg.get('min_wave_pct', 0.1)
    min_wave_pct_range = np.arange(max(0.0, current_min_wave_pct - 0.1), min(0.5, current_min_wave_pct + 0.2), 0.1)

    if use_extended_params:
        atr_period_range = [10, 14, 21]
        freeze_confirm_range = [True, False]
        coarse_grid = list(product(
            np.arange(0.3, 2.1, 0.6),
            coarse_thresh_range,
            coarse_conf_bars,
            min_wave_range_coarse,
            min_wave_pct_range,
            atr_period_range,
            freeze_confirm_range
        ))
    else:
        coarse_grid = list(product(
            np.arange(0.3, 2.1, 0.6),
            coarse_thresh_range,
            coarse_conf_bars,
            min_wave_range_coarse,
            min_wave_pct_range
        ))

    coarse_df = run_grid(coarse_grid, df, zigzag_cfg, n_jobs, "Coarse",
                        pivot_count_range, use_enhanced_metrics)
    if coarse_df.empty:
        return None

    return coarse_df.head(10)

def validate_on_test(df: pd.DataFrame, params: dict, config_type: str = None,
                    pivot_count_range: tuple = (5, 9), use_enhanced_metrics: bool = True,
                    zigzag_cfg: dict = None) -> dict:
    """테스트 데이터로 파라미터 검증 (Phase 3 헬퍼 함수)"""
    try:
        # DESIGN-2 수정: zigzag_cfg가 제공되면 사용, 아니면 config에서 로드
        if zigzag_cfg is None:
            config_path = os.path.join(os.path.dirname(__file__), '../../config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            base_cfg = config.get('adaptive_indicator', {}).get('zigzag', {})
            zigzag_cfg = config.get('adaptive_indicator', {}).get(config_type, base_cfg).copy()
        
        g = zigzag_cfg.get
        thresh_max = g('pivot_threshold_max_pct', 0.3)
        actual_thresh_min = min(params['pivot_threshold_min_pct'], thresh_max)

        cfg = AdaptiveZigZagConfig(
            # DESIGN-2 수정: params에서 확장 파라미터 우선 사용
            atr_period=int(params.get('atr_period', g('atr_period', 14))),
            atr_multiplier=params['atr_multiplier'],
            pivot_threshold_min_pct=actual_thresh_min,
            pivot_threshold_max_pct=thresh_max,
            confirmation_bars=int(params['confirmation_bars']),
            atr_multiplier_min=g('atr_multiplier_min', 0.3),
            atr_multiplier_max=g('atr_multiplier_max', 1.0),
            major_swing_ratio=g('major_swing_ratio', 1.2),
            max_swings=g('max_swings', 20),
            freeze_on_confirm=bool(params.get('freeze_on_confirm', g('freeze_on_confirm', True))),
            min_wave_bars=g('min_wave_bars', 1),
            min_wave_pct=params.get('min_wave_pct', g('min_wave_pct', 0.0)),  # Phase 5: params에서 우선 사용 (2026-05-10)
            cluster_tolerance_pct=g('cluster_tolerance_pct', 0.3),
            structure_lookback_swings=g('structure_lookback_swings', 8),
            structure_points=g('structure_points', 3),
        )
        
        zz = AdaptiveZigZag(cfg)
        zz.compute_from_df(df)
        
        target_pivots = sum(pivot_count_range) / 2
        if use_enhanced_metrics:
            metrics = calculate_metrics_enhanced(zz, df, target_pivots)
        else:
            metrics = calculate_metrics(zz, target_pivots)
        
        if not (pivot_count_range[0] <= metrics['pivot_count'] <= pivot_count_range[1]):
            return None
        
        return {
            'test_avg_lag': metrics['avg_lag'],
            'test_max_lag': metrics['max_lag'],
            'test_pivot_count': metrics['pivot_count'],
            'test_score': metrics['score']
        }
    except Exception:
        return None

# ──────────────────────────────────────────────
# Phase 2: 시장 레짐별 최적화
# ──────────────────────────────────────────────
def optimize_parameters_with_regime(symbol: str, config_type: str, n_jobs: int = -1,
                                   pivot_count_range: tuple = (5, 9), max_avg_lag: float = 5.0,
                                   test_ratio: float = 0.2, use_enhanced_metrics: bool = True,
                                   use_extended_params: bool = True):
    """시장 레짐별 파라미터 최적화 (Phase 2)"""
    logger.info(f"Phase 2 Optimization Start: {symbol} ({config_type})")
    logger.info(f"Regime-based optimization, Extended params: {use_extended_params}")
    
    df = load_saved_minute_data(symbol)
    
    # 시장 레짐 분류 시도
    try:
        from services.market_regime_classifier import MarketRegimeClassifier
        regime_classifier = MarketRegimeClassifier()
        use_regime = True
    except Exception as e:
        logger.warning(f"MarketRegimeClassifier not available: {e}")
        logger.info("Falling back to standard optimization")
        use_regime = False
    
    if not use_regime:
        # 레짐 분류 불가능하면 기존 방식으로 폴백
        return train_test_split_optimize(symbol, config_type, n_jobs,
                                         pivot_count_range, max_avg_lag,
                                         test_ratio, use_enhanced_metrics)
    
    # 레짐별 데이터 분할 (BUG-4 수정: 연속 시계열 구간 추출)
    regime_data = {}
    window_size = 30  # 레짐 분석 윈도우 크기
    
    # 레짐 시계열 구간 추출
    current_regime = None
    regime_start_idx = 0
    min_segment_length = 50  # 최소 구간 길이
    
    for i in range(window_size, len(df)):
        window_df = df.iloc[i-window_size:i]
        try:
            regime = regime_classifier.classify(window_df, current_idx=i-1)
            if regime is None:
                new_regime_name = "UNKNOWN"
            else:
                new_regime_name = regime.regime.value if hasattr(regime, 'regime') else "UNKNOWN"
        except Exception:
            new_regime_name = "UNKNOWN"
        
        # 레짐이 변경되면 이전 구간 저장
        if new_regime_name != current_regime:
            if current_regime is not None:
                segment_length = i - regime_start_idx
                if segment_length >= min_segment_length:
                    if current_regime not in regime_data:
                        regime_data[current_regime] = []
                    regime_data[current_regime].append(df.iloc[regime_start_idx:i])
            
            current_regime = new_regime_name
            regime_start_idx = i
    
    # 마지막 구간 추가
    if current_regime is not None:
        segment_length = len(df) - regime_start_idx
        if segment_length >= min_segment_length:
            if current_regime not in regime_data:
                regime_data[current_regime] = []
            regime_data[current_regime].append(df.iloc[regime_start_idx:])
    
    # 레짐별 최적화
    regime_params = {}
    for regime_name, regime_segments in regime_data.items():
        # 연속 구간들을 합치되, 구간 사이에 구분을 위해 NaN 행 삽입
        # 또는 각 구간을 독립적으로 처리
        total_bars = sum(len(seg) for seg in regime_segments)
        if total_bars < 100:  # 데이터 부족하면 스킵
            logger.info(f"Skipping regime {regime_name}: insufficient data ({total_bars} bars)")
            continue
        
        # 가장 긴 구간 사용 (연속 시계열 유지)
        regime_df = max(regime_segments, key=len)
        logger.info(f"Optimizing for regime: {regime_name} (data points: {len(regime_df)})")
        
        # 레짐별 파라미터 범위 조정
        if "UP" in regime_name:
            # 추세형: 민감한 파라미터
            atr_mult_range = np.arange(1.0, 1.8, 0.2)
            conf_bars_range = [1]
            min_wave_range = np.arange(0.3, 1.0, 0.2)
        elif "NO_DIRECTION" in regime_name:
            # 횡보형: 보수적인 파라미터
            atr_mult_range = np.arange(1.8, 3.0, 0.3)
            conf_bars_range = [2, 3]
            min_wave_range = np.arange(0.8, 2.0, 0.3)
        elif "DOWN" in regime_name:
            # 하락형: 중간 파라미터
            atr_mult_range = np.arange(1.3, 2.2, 0.3)
            conf_bars_range = [2]
            min_wave_range = np.arange(0.5, 1.5, 0.3)
        else:
            # 기본값
            atr_mult_range = np.arange(1.2, 2.5, 0.3)
            conf_bars_range = [1, 2, 3]
            min_wave_range = np.arange(0.5, 1.5, 0.3)
        
        # Config 로드
        config_path = os.path.join(os.path.dirname(__file__), '../../config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        base_cfg = config.get('adaptive_indicator', {}).get('zigzag', {})
        zigzag_cfg = config.get('adaptive_indicator', {}).get(config_type, base_cfg).copy()
        
        thresh_max = zigzag_cfg.get('pivot_threshold_max_pct', 0.3)
        coarse_thresh_range = np.arange(0.01, min(0.5, thresh_max), 0.1)

        # Phase 5: min_wave_pct 범위 추가 (2026-05-10)
        current_min_wave_pct = zigzag_cfg.get('min_wave_pct', 0.1)
        min_wave_pct_range = np.arange(max(0.0, current_min_wave_pct - 0.1), min(0.5, current_min_wave_pct + 0.2), 0.1)

        # Phase 2: 확장된 파라미터 그리드
        if use_extended_params:
            atr_period_range = [10, 14, 21]  # 단기/중기/장기
            freeze_confirm_range = [True, False]

            coarse_grid = list(product(
                atr_mult_range,
                coarse_thresh_range,
                conf_bars_range,
                min_wave_range,
                min_wave_pct_range,
                atr_period_range,
                freeze_confirm_range
            ))
        else:
            coarse_grid = list(product(
                atr_mult_range,
                coarse_thresh_range,
                conf_bars_range,
                min_wave_range,
                min_wave_pct_range
            ))
        
        # 레짐별 최적화 실행
        coarse_df = run_grid(coarse_grid, regime_df, zigzag_cfg, n_jobs, 
                            f"Coarse({regime_name})", pivot_count_range, use_enhanced_metrics)
        if coarse_df.empty:
            logger.warning(f"No valid combinations for regime {regime_name}")
            continue
        
        top_candidates = coarse_df.head(5)
        regime_params[regime_name] = top_candidates.to_dict('records')
        logger.info(f"Regime {regime_name}: {len(top_candidates)} candidates found")
    
    if not regime_params:
        logger.warning("No regime optimization results, falling back to standard")
        return train_test_split_optimize(symbol, config_type, n_jobs,
                                         pivot_count_range, max_avg_lag,
                                         test_ratio, use_enhanced_metrics)
    
    # 레짐별 결과 통합
    logger.info(f"Regime-based optimization completed for {len(regime_params)} regimes")
    return regime_params

# ──────────────────────────────────────────────
# Phase 1: 학습/테스트 분리 및 검증 (수정: 확장 파라미터 지원)
# ──────────────────────────────────────────────
def train_test_split_optimize(symbol: str, config_type: str, n_jobs: int = -1,
                              pivot_count_range: tuple = (5, 9), max_avg_lag: float = 5.0,
                              test_ratio: float = 0.2, use_enhanced_metrics: bool = True,
                              use_extended_params: bool = False):
    """학습/테스트 분리 후 최적화 (Phase 1)"""
    logger.info(f"Phase 1 Optimization Start: {symbol} ({config_type})")
    logger.info(f"Test ratio: {test_ratio}, Enhanced metrics: {use_enhanced_metrics}")
    
    df = load_saved_minute_data(symbol)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    logger.info(f"Train: {len(train_df)} bars, Test: {len(test_df)} bars")
    
    # Config 로드
    config_path = os.path.join(os.path.dirname(__file__), '../../config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    base_cfg = config.get('adaptive_indicator', {}).get('zigzag', {})
    zigzag_cfg = config.get('adaptive_indicator', {}).get(config_type, base_cfg).copy()

    logger.info(f"Extended params: {use_extended_params}")
    
    # Phase 3: 시기별 균형 샘플링 옵션
    use_balanced_sample = False  # 기본값은 비활성화
    
    logger.info(f"Extended params: {use_extended_params}")
    logger.info(f"Balanced sample: {use_balanced_sample}")
    
    # 파라미터 범위 설정
    if symbol == 'kospi':
        min_wave_range_coarse = np.arange(0.3, 2.5, 0.7)
        min_wave_range_fine = np.arange(0.3, 2.5, 0.15)
        coarse_conf_bars = [1]
    else:
        min_wave_range_coarse = np.arange(0.5, 3.5, 1.0)
        min_wave_range_fine = np.arange(0.5, 3.5, 0.2)
        coarse_conf_bars = [1, 2, 3]
    
    thresh_max = zigzag_cfg.get('pivot_threshold_max_pct', 0.3)
    coarse_thresh_range = np.arange(0.01, min(0.5, thresh_max), 0.1)

    # Phase 5: min_wave_pct 범위 추가 (2026-05-10)
    current_min_wave_pct = zigzag_cfg.get('min_wave_pct', 0.1)
    min_wave_pct_range = np.arange(max(0.0, current_min_wave_pct - 0.1), min(0.5, current_min_wave_pct + 0.2), 0.1)

    # Phase 2: 확장된 파라미터 그리드
    if use_extended_params:
        atr_period_range = [10, 14, 21]  # 단기/중기/장기
        freeze_confirm_range = [True, False]

        coarse_grid = list(product(
            np.arange(0.3, 2.1, 0.6),
            coarse_thresh_range,
            coarse_conf_bars,
            min_wave_range_coarse,
            min_wave_pct_range,
            atr_period_range,
            freeze_confirm_range
        ))
    else:
        coarse_grid = list(product(
            np.arange(0.3, 2.1, 0.6),
            coarse_thresh_range,
            coarse_conf_bars,
            min_wave_range_coarse,
            min_wave_pct_range
        ))
    
    # Phase 3: 시기별 균형 샘플링 적용
    if use_balanced_sample:
        train_df = load_balanced_sample(train_df, sample_ratio=0.3)
        logger.info(f"Balanced sample: {len(train_df)} bars")

    # STEP 1: Coarse (Train 데이터)
    coarse_df = run_grid(coarse_grid, train_df, zigzag_cfg, n_jobs, "Coarse(Train)", 
                        pivot_count_range, use_enhanced_metrics)
    if coarse_df.empty:
        logger.warning("No valid combinations found in Coarse search.")
        return None, None

    top_candidates = coarse_df.head(10)

    # STEP 2: Fine (Train 데이터)
    fine_params = set()
    for _, row in top_candidates.iterrows():
        atr_space = np.linspace(max(0.1, row.atr_multiplier - 0.3), row.atr_multiplier + 0.3, 5)
        thresh_space = np.linspace(max(0.005, row.pivot_threshold_min_pct - 0.05),
                                   min(row.pivot_threshold_min_pct + 0.05, thresh_max), 5)

        # Phase 5: min_wave_pct 범위 추가 (2026-05-10)
        min_wave_pct_val = row.get('min_wave_pct', 0.1)
        min_wave_pct_space = np.linspace(max(0.0, min_wave_pct_val - 0.05), min(0.5, min_wave_pct_val + 0.1), 3)

        if use_extended_params and 'atr_period' in row:
            # 확장 파라미터: atr_period, freeze_on_confirm 유지
            atr_period = row['atr_period']
            freeze_confirm = row['freeze_on_confirm']

            for combo in product(atr_space, thresh_space, [int(row.confirmation_bars)],
                                min_wave_range_fine, min_wave_pct_space, [atr_period], [freeze_confirm]):
                fine_params.add(tuple(round(v, 6) for v in combo))
        else:
            # 기존 파라미터 + min_wave_pct
            for combo in product(atr_space, thresh_space, [int(row.confirmation_bars)], min_wave_range_fine, min_wave_pct_space):
                fine_params.add(tuple(round(v, 6) for v in combo))

    fine_df = run_grid(list(fine_params), train_df, zigzag_cfg, n_jobs, "Fine(Train)", 
                      pivot_count_range, use_enhanced_metrics)
    
    # STEP 3: Train 데이터 최적 파라미터 추출
    final_df = fine_df if not fine_df.empty else coarse_df
    result_filtered = final_df[final_df['avg_lag'] <= max_avg_lag]

    if result_filtered.empty:
        logger.info("No results within max_avg_lag, returning best by score.")
        return final_df.head(10), None
    
    train_best_df = result_filtered.head(10) if not result_filtered.empty else final_df.head(10)
    
    # STEP 4: Test 데이터 검증
    test_results = []
    for _, row in train_best_df.iterrows():
        test_metrics = validate_on_test(test_df, row.to_dict(), zigzag_cfg, pivot_count_range, 
                                       use_enhanced_metrics)
        if test_metrics:
            test_results.append({**row.to_dict(), **test_metrics})
    
    if not test_results:
        logger.warning("No valid test results.")
        return train_best_df.head(10), None
    
    test_df_result = pd.DataFrame(test_results)
    
    # 테스트 성능 기준 필터링
    test_filtered = test_df_result[test_df_result['test_avg_lag'] <= max_avg_lag * 1.5]
    
    if test_filtered.empty:
        logger.info("Test performance below threshold, returning best by train score.")
        return train_best_df.head(10), test_df_result.head(10)
    
    return train_best_df.head(10), test_filtered.head(10)

# 중복된 validate_on_test 함수 삭제 (BUG-1 수정)

# ──────────────────────────────────────────────
# 메인 최적화 함수 (기존 유지 - 하위 호환성)
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Phase 3: 시간대별 최적화
# ──────────────────────────────────────────────
def optimize_time_based_parameters(symbol: str, config_type: str, n_jobs: int = -1,
                                   pivot_count_range: tuple = (5, 9), max_avg_lag: float = 5.0,
                                   use_enhanced_metrics: bool = True,
                                   use_extended_params: bool = False):
    """시간대별 파라미터 최적화 (Phase 3)"""
    logger.info(f"Phase 3 Time-based Optimization Start: {symbol} ({config_type})")
    
    df = load_saved_minute_data(symbol)
    
    # 시간대별 데이터 분할
    time_ranges = {
        'early': ('09:00', '09:30'),
        'morning': ('09:30', '11:30'),
        'lunch': ('11:30', '13:00'),
        'afternoon': ('13:00', '15:00'),
        'close': ('15:00', '15:30')
    }
    
    time_params = {}
    for period, (start, end) in time_ranges.items():
        # DESIGN-3 수정: 날짜별로 그룹핑 후 시간대 필터 적용
        period_segments = []
        for date, date_df in df.groupby(df.index.date):
            # 시간대 필터링
            mask = (date_df.index.time >= pd.to_datetime(start).time()) & \
                   (date_df.index.time <= pd.to_datetime(end).time())
            time_df = date_df[mask]
            if len(time_df) > 0:
                period_segments.append(time_df)
        
        if not period_segments:
            logger.info(f"Skipping {period}: no data")
            continue
        
        # 각 세그먼트를 독립적으로 처리하여 시계열 연속성 유지
        # 가장 긴 세그먼트 사용 (데이터가 충분한 경우)
        period_segments.sort(key=len, reverse=True)
        period_df = period_segments[0]  # 가장 긴 세그먼트 사용
        
        if len(period_df) < 100:
            logger.info(f"Skipping {period}: insufficient data ({len(period_df)} bars)")
            continue
        
        logger.info(f"Optimizing for time period: {period} ({start}~{end}, {len(period_df)} bars)")
        
        # 시간대별 최적화
        period_results = optimize_parameters_on_df(period_df, symbol, config_type, n_jobs,
                                                   pivot_count_range, max_avg_lag,
                                                   use_enhanced_metrics, use_extended_params)
        
        if period_results is not None and not period_results.empty:
            time_params[period] = period_results.head(5).to_dict('records')
            logger.info(f"Period {period}: {len(period_results)} candidates found")
    
    if not time_params:
        logger.warning("No time-based results, falling back to standard")
        return train_test_split_optimize(symbol, config_type, n_jobs,
                                         pivot_count_range, max_avg_lag,
                                         0.2, use_enhanced_metrics, use_extended_params)
    
    # 시간대별 결과 통합
    logger.info(f"Time-based optimization completed for {len(time_params)} periods")
    return time_params

# ──────────────────────────────────────────────
# Phase 4: 결과 분석 기능
# ──────────────────────────────────────────────
def generate_optimization_report(results: pd.DataFrame, best_params: dict, 
                                  symbol: str, config_type: str) -> str:
    """최적화 결과 리포트 생성 (Phase 4)"""
    report = []
    report.append("="*80)
    report.append(f"최적화 결과 리포트 - {symbol.upper()} ({config_type})")
    report.append("="*80)
    
    # 상위 10개 비교
    report.append("\n[상위 10개 파라미터 조합]")
    for i, row in results.head(10).iterrows():
        report.append(f"\n#{i+1}")
        report.append(f"  ATR 배수: {row['atr_multiplier']:.4f}")
        report.append(f"  임계값: {row['pivot_threshold_min_pct']:.4f}")
        report.append(f"  확인봉: {row['confirmation_bars']}")
        report.append(f"  최소파동: {row['min_wave_atr_ratio']:.4f}")
        report.append(f"  피봇수: {row['pivot_count']}")
        report.append(f"  평균지연: {row['avg_lag']:.2f}")
        report.append(f"  스코어: {row['score']:.4f}")
        
        if 'pivot_quality' in row:
            report.append(f"  피봇품질: {row['pivot_quality']:.4f}")
            report.append(f"  교번준수율: {row['alternation_rate']:.4f}")
        
        if 'atr_period' in row:
            report.append(f"  ATR기간: {row['atr_period']}")
            report.append(f"  동결확정: {row['freeze_on_confirm']}")
    
    # 파라미터 민감도 분석
    report.append("\n[파라미터 민감도 분석]")
    param_cols = ['atr_multiplier', 'pivot_threshold_min_pct', 
                  'confirmation_bars', 'min_wave_atr_ratio']
    for param in param_cols:
        if param in results.columns:
            corr = results[param].corr(results['score'])
            report.append(f"  {param}: 상관계수 {corr:.3f}")
    
    # 최적 파라미터 특성
    report.append("\n[최적 파라미터 특성]")
    report.append(f"  ATR 배수: {best_params['atr_multiplier']:.4f}")
    report.append(f"  → 민감도: {'높음' if best_params['atr_multiplier'] < 1.5 else '중간' if best_params['atr_multiplier'] < 2.0 else '낮음'}")
    report.append(f"  확인봉: {best_params['confirmation_bars']}")
    report.append(f"  → 확정속도: {'빠름' if best_params['confirmation_bars'] == 1 else '중간' if best_params['confirmation_bars'] == 2 else '느림'}")
    
    if 'pivot_quality' in best_params:
        report.append(f"  피봇 품질: {best_params['pivot_quality']:.4f}")
        report.append(f"  → 품질: {'우수' if best_params['pivot_quality'] > 0.8 else '보통' if best_params['pivot_quality'] > 0.6 else '낮음'}")
    
    # 통계 요약
    report.append("\n[통계 요약]")
    report.append(f"  총 조합 수: {len(results)}")
    report.append(f"  평균 스코어: {results['score'].mean():.4f}")
    report.append(f"  스코어 표준편차: {results['score'].std():.4f}")
    report.append(f"  최소 스코어: {results['score'].min():.4f}")
    report.append(f"  최대 스코어: {results['score'].max():.4f}")
    
    return "\n".join(report)

def save_optimization_report(report: str, symbol: str, config_type: str):
    """최적화 리포트 저장 (Phase 4)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_report_{symbol}_{config_type}_{timestamp}.txt"
    report_dir = os.path.join(os.path.dirname(__file__), '../../reports')
    
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Optimization report saved: {report_path}")
    return report_path

# ──────────────────────────────────────────────
# 메인 최적화 함수 (기존 유지 - 하위 호환성)
# ──────────────────────────────────────────────
def optimize_parameters(symbol: str, config_type: str, n_jobs: int = -1,
                         pivot_count_range: tuple = (5, 9), max_avg_lag: float = 5.0,
                         use_enhanced_metrics: bool = False):
    
    logger.info(f"Optimization Start: {symbol} ({config_type})")
    logger.info(f"Enhanced metrics: {use_enhanced_metrics}")
    
    df = load_saved_minute_data(symbol)
    sample_size = max(500, int(len(df) * 0.2))
    df_sample = df.iloc[-sample_size:].copy()

    # Config 로드
    config_path = os.path.join(os.path.dirname(__file__), '../../config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    base_cfg = config.get('adaptive_indicator', {}).get('zigzag', {})
    zigzag_cfg = config.get('adaptive_indicator', {}).get(config_type, base_cfg).copy()

    # 파라미터 범위 설정 (심볼별 최적화)
    if symbol == 'kospi':
        min_wave_range_coarse = np.arange(0.3, 2.5, 0.7)
        min_wave_range_fine = np.arange(0.3, 2.5, 0.15)
        coarse_conf_bars = [1]
    else:
        min_wave_range_coarse = np.arange(0.5, 3.5, 1.0)
        min_wave_range_fine = np.arange(0.5, 3.5, 0.2)
        coarse_conf_bars = [1, 2, 3]

    thresh_max = zigzag_cfg.get('pivot_threshold_max_pct', 0.3)
    coarse_thresh_range = np.arange(0.01, min(0.5, thresh_max), 0.1)

    # Phase 5: min_wave_pct 범위 추가 (2026-05-10)
    current_min_wave_pct = zigzag_cfg.get('min_wave_pct', 0.1)
    min_wave_pct_range = np.arange(max(0.0, current_min_wave_pct - 0.1), min(0.5, current_min_wave_pct + 0.2), 0.1)

    coarse_grid = list(product(
        np.arange(0.3, 2.1, 0.6),
        coarse_thresh_range,
        coarse_conf_bars,
        min_wave_range_coarse,
        min_wave_pct_range
    ))

    # STEP 1: Coarse
    coarse_df = run_grid(coarse_grid, df_sample, zigzag_cfg, n_jobs, "Coarse", 
                        pivot_count_range, use_enhanced_metrics)
    if coarse_df.empty:
        logger.warning("No valid combinations found in Coarse search.")
        return None

    top_candidates = coarse_df.head(10)

    # STEP 2: Fine
    fine_params = set()
    for _, row in top_candidates.iterrows():
        atr_space = np.linspace(max(0.1, row.atr_multiplier - 0.3), row.atr_multiplier + 0.3, 5)
        thresh_space = np.linspace(max(0.005, row.pivot_threshold_min_pct - 0.05),
                                   min(row.pivot_threshold_min_pct + 0.05, thresh_max), 5)

        # Phase 5: min_wave_pct 범위 추가 (2026-05-10)
        min_wave_pct_val = row.get('min_wave_pct', 0.1)
        min_wave_pct_space = np.linspace(max(0.0, min_wave_pct_val - 0.05), min(0.5, min_wave_pct_val + 0.1), 3)

        for combo in product(atr_space, thresh_space, [int(row.confirmation_bars)], min_wave_range_fine, min_wave_pct_space):
            fine_params.add(tuple(round(v, 6) for v in combo))

    fine_df = run_grid(list(fine_params), df, zigzag_cfg, n_jobs, "Fine", 
                      pivot_count_range, use_enhanced_metrics)
    
    # STEP 3: Final Filter
    final_df = fine_df if not fine_df.empty else coarse_df
    result_filtered = final_df[final_df['avg_lag'] <= max_avg_lag]

    if result_filtered.empty:
        logger.info("No results within max_avg_lag, returning best by score.")
        return final_df.head(10)

    return result_filtered.head(10)


# ──────────────────────────────────────────────
# 최적 파라미터 추출 및 저장 (Phase 1: 향상된 메트릭 지원)
# ──────────────────────────────────────────────
def extract_and_save_best_params(results: pd.DataFrame, symbol: str, config_type: str,
                                max_avg_lag: float = 5.0, test_results: pd.DataFrame = None):
    """최적 파라미터 추출 및 config.json 업데이트"""
    if results is None or results.empty:
        print(f"  {symbol.upper()}: 유효 결과 없음 - 건너뜀")
        return None

    # 평균 지연시간 및 최대 지연시간 필터링
    filtered = results[(results['avg_lag'] <= max_avg_lag) & (results['max_lag'] <= max_avg_lag * 2)].sort_values('avg_lag')
    best = (filtered if not filtered.empty else results).iloc[0]

    best_params = {
        'atr_multiplier': round(float(best.atr_multiplier), 4),
        'pivot_threshold_min_pct': round(float(best.pivot_threshold_min_pct), 4),
        'confirmation_bars': int(best.confirmation_bars),
        'min_wave_atr_ratio': round(float(best.min_wave_atr_ratio), 4),
        'pivot_count': int(best.pivot_count),
        'avg_lag': round(float(best.avg_lag), 2),
        'max_lag': int(best.max_lag),
    }

    # Phase 5: min_wave_pct 추가 (2026-05-10)
    if 'min_wave_pct' in best:
        best_params['min_wave_pct'] = round(float(best.min_wave_pct), 4)

    # Phase 1: 향상된 메트릭이 있으면 추가
    if 'pivot_quality' in best:
        best_params['pivot_quality'] = round(float(best.pivot_quality), 4)
        best_params['alternation_rate'] = round(float(best.alternation_rate), 4)
        best_params['lag_p95'] = round(float(best.lag_p95), 2)

    print(f"\n{'='*80}")
    print(f"[최적 파라미터] {symbol.upper()} ({config_type})")
    print(f"{'='*80}")
    print(f"  atr_multiplier          : {best_params['atr_multiplier']:.4f}")
    print(f"  pivot_threshold_min_pct : {best_params['pivot_threshold_min_pct']:.4f}")
    print(f"  confirmation_bars       : {best_params['confirmation_bars']}")
    print(f"  min_wave_atr_ratio      : {best_params['min_wave_atr_ratio']:.4f}")
    if 'min_wave_pct' in best_params:
        print(f"  min_wave_pct            : {best_params['min_wave_pct']:.4f}")
    print(f"  피봇 수                 : {best_params['pivot_count']}")
    print(f"  평균 지연시간           : {best_params['avg_lag']:.2f} 봉")
    print(f"  최대 지연시간           : {best_params['max_lag']} 봉")

    if 'pivot_quality' in best_params:
        print(f"  피봇 품질               : {best_params['pivot_quality']:.4f}")
        print(f"  교번 준수율           : {best_params['alternation_rate']:.4f}")
        print(f"  95% 지연시간           : {best_params['lag_p95']:.2f} 봉")
    
    # Phase 2: 확장 파라미터 출력
    if 'atr_period' in best_params:
        print(f"  atr_period             : {best_params['atr_period']}")
        print(f"  freeze_on_confirm      : {best_params['freeze_on_confirm']}")
    
    # 테스트 결과가 있으면 출력
    if test_results is not None and not test_results.empty:
        test_best = test_results.iloc[0]
        print("\n[테스트 검증 결과]")
        print(f"  테스트 평균 지연시간   : {test_best['test_avg_lag']:.2f} 봉")
        print(f"  테스트 피봇 수         : {test_best['test_pivot_count']}")
        print(f"  테스트 스코어         : {test_best['test_score']:.4f}")
    
    print(f"{'='*80}")

    return best_params


def update_config_json(best_params: dict, config_type: str):
    """config.json에 최적 파라미터 업데이트 (기존 설정 유지)"""
    config_path = os.path.join(os.path.dirname(__file__), '../../config.json')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 해당 config_type 섹션 업데이트 (기존 설정 유지)
    if 'adaptive_indicator' not in config:
        config['adaptive_indicator'] = {}

    if config_type not in config['adaptive_indicator']:
        config['adaptive_indicator'][config_type] = {}

    # 최적 파라미터만 업데이트 (기존 설정 유지)
    config['adaptive_indicator'][config_type]['atr_multiplier'] = best_params['atr_multiplier']
    config['adaptive_indicator'][config_type]['pivot_threshold_min_pct'] = best_params['pivot_threshold_min_pct']
    config['adaptive_indicator'][config_type]['confirmation_bars'] = best_params['confirmation_bars']
    config['adaptive_indicator'][config_type]['min_wave_atr_ratio'] = best_params['min_wave_atr_ratio']
    # Phase 5: min_wave_pct 업데이트 (2026-05-10)
    if 'min_wave_pct' in best_params:
        config['adaptive_indicator'][config_type]['min_wave_pct'] = best_params['min_wave_pct']

    # 백업 생성
    backup_path = config_path + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 원본 업데이트
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  config.json 업데이트 완료 ({config_type})")
    print(f"  백업 파일: {backup_path}")


# ──────────────────────────────────────────────
# Phase 5: 시간대별 최적화 결과를 session_min_wave_bars_table로 변환 (2026-05-10)
# ──────────────────────────────────────────────
def convert_time_based_to_session_table(time_params: dict) -> list:
    """시간대별 최적화 결과를 session_min_wave_bars_table 형식으로 변환"""
    # 기본 시간대 정의 (09:00 ~ 15:30)
    session_table = []

    # 시간대별 최적 min_wave_bars 값 추출
    time_to_min_bars = {
        'early': 1,    # 09:00 ~ 09:30
        'morning': 2,  # 09:30 ~ 11:30
        'lunch': 2,    # 11:30 ~ 13:00
        'afternoon': 2, # 13:00 ~ 15:00
        'close': 1     # 15:00 ~ 15:30
    }

    # 최적화 결과가 있으면 해당 값 사용
    for period, results in time_params.items():
        if results and len(results) > 0:
            best_result = results[0] if isinstance(results, list) else results
            # confirmation_bars를 min_wave_bars로 사용
            min_bars = int(best_result.get('confirmation_bars', time_to_min_bars.get(period, 2)))
            time_to_min_bars[period] = min_bars

    # session_min_wave_bars_table 형식 변환
    # [[시작시간, 종료시간, min_wave_bars], ...]
    session_table = [
        ["09:00", "09:30", time_to_min_bars.get('early', 1)],
        ["09:30", "10:30", time_to_min_bars.get('morning', 2)],
        ["10:30", "15:30", time_to_min_bars.get('afternoon', 2)]  # 점심, 오후, 마감 통합
    ]

    return session_table


def update_config_with_session_table(session_table: list, config_type: str):
    """config.json에 session_min_wave_bars_table 업데이트"""
    config_path = os.path.join(os.path.dirname(__file__), '../../config.json')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 해당 config_type 섹션 업데이트
    if 'adaptive_indicator' not in config:
        config['adaptive_indicator'] = {}

    if config_type not in config['adaptive_indicator']:
        config['adaptive_indicator'][config_type] = {}

    # session_min_wave_bars_table 업데이트
    config['adaptive_indicator'][config_type]['session_min_wave_bars_table'] = session_table

    # 백업 생성
    backup_path = config_path + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 원본 업데이트
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  config.json session_min_wave_bars_table 업데이트 완료 ({config_type})")
    print(f"  백업 파일: {backup_path}")
    print(f"  session_min_wave_bars_table: {session_table}")


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────
if __name__ == '__main__':
    t0 = datetime.now()

    # 사용자 입력 받기
    print(f"\n{'='*80}")
    print("최적화 파라미터 설정")
    print(f"{'='*80}")

    # 피봇 수 범위 입력
    print("\n피봇 수 범위 (기본값: 5~9개)")
    pivot_min_input = input("  최소 피봇 수 (기본값 5): ").strip()
    pivot_min = int(pivot_min_input) if pivot_min_input else 5

    pivot_max_input = input("  최대 피봇 수 (기본값 9): ").strip()
    pivot_max = int(pivot_max_input) if pivot_max_input else 9

    if pivot_min > pivot_max:
        print("  경고: 최소 피봇 수가 최대보다 큽니다. 교환합니다.")
        pivot_min, pivot_max = pivot_max, pivot_min

    pivot_count_range = (pivot_min, pivot_max)

    # 최대 평균 지연시간 입력
    print("\n최대 평균 지연시간 (기본값: 5.0 봉)")
    max_lag_input = input("  최대 평균 지연시간 (기본값 5.0): ").strip()
    max_avg_lag = float(max_lag_input) if max_lag_input else 5.0

    print("\n설정된 파라미터:")
    print(f"  피봇 수 범위: {pivot_count_range[0]}~{pivot_count_range[1]}개")
    print(f"  최대 평균 지연: {max_avg_lag} 봉")
    print(f"{'='*80}")

    # Phase 4 모드 선택
    print("\n최적화 모드 선택:")
    print("  1. 기존 모드 (단일 데이터, 기본 평가지표)")
    print("  2. Phase 1 모드 (학습/테스트 분리, 향상된 평가지표)")
    print("  3. Phase 2 모드 (시장 레짐별 최적화, 확장 파라미터)")
    print("  4. Phase 3 모드 (시계열 CV, 시간대별 최적화)")
    print("  5. Phase 4 모드 (결과 분석, 병렬 최적화)")
    mode_input = input("  모드 선택 (기본값: 5): ").strip()
    mode = int(mode_input) if mode_input else 5

    # Phase 5: 데이터 요약 출력 옵션 (2026-05-10)
    summary_input = input("\n  데이터 요약 출력? (y/n, 기본값 y): ").strip().lower()
    print_summary = summary_input != 'n'

    if print_summary:
        try:
            kospi_df = load_saved_minute_data('kospi')
            print_data_summary(kospi_df, 'kospi')
            kp200_df = load_saved_minute_data('kp200')
            print_data_summary(kp200_df, 'kp200')
        except Exception as e:
            print(f"  데이터 요약 출력 실패: {e}")
    
    use_enhanced_metrics = (mode >= 2)
    use_extended_params = (mode >= 3)
    use_batch_processing = (mode >= 4)
    use_report = (mode == 5)
    
    if mode == 1:
        # 기존 모드
        kospi_results = optimize_parameters('kospi', 'kospi_zigzag',
                                           pivot_count_range=pivot_count_range,
                                           max_avg_lag=max_avg_lag,
                                           use_enhanced_metrics=False)
        kospi_best = extract_and_save_best_params(kospi_results, 'kospi', 'kospi_zigzag',
                                                    max_avg_lag=max_avg_lag)

        kp200_results = optimize_parameters('kp200', 'futures_zigzag',
                                           pivot_count_range=pivot_count_range,
                                           max_avg_lag=max_avg_lag,
                                           use_enhanced_metrics=False)
        kp200_best = extract_and_save_best_params(kp200_results, 'kp200', 'futures_zigzag',
                                                    max_avg_lag=max_avg_lag)
    elif mode == 2:
        # Phase 1 모드
        test_ratio_input = input("  테스트 비율 (기본값 0.2): ").strip()
        test_ratio = float(test_ratio_input) if test_ratio_input else 0.2
        
        kospi_train, kospi_test = train_test_split_optimize('kospi', 'kospi_zigzag',
                                                              pivot_count_range=pivot_count_range,
                                                              max_avg_lag=max_avg_lag,
                                                              test_ratio=test_ratio,
                                                              use_enhanced_metrics=True,
                                                              use_extended_params=False)
        kospi_best = extract_and_save_best_params(kospi_train, 'kospi', 'kospi_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kospi_test)

        kp200_train, kp200_test = train_test_split_optimize('kp200', 'futures_zigzag',
                                                              pivot_count_range=pivot_count_range,
                                                              max_avg_lag=max_avg_lag,
                                                              test_ratio=test_ratio,
                              use_enhanced_metrics=True,
                              use_extended_params=False)
        kp200_best = extract_and_save_best_params(kp200_train, 'kp200', 'futures_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kp200_test)
    elif mode == 3:
        # Phase 2 모드
        test_ratio_input = input("  테스트 비율 (기본값 0.2): ").strip()
        test_ratio = float(test_ratio_input) if test_ratio_input else 0.2
        
        print("\nPhase 2 옵션:")
        use_regime_input = input("  시장 레짐별 최적화 사용? (y/n, 기본값 y): ").strip().lower()
        use_regime = use_regime_input != 'n'
        
        if use_regime:
            # 시장 레짐별 최적화
            kospi_regime_results = optimize_parameters_with_regime('kospi', 'kospi_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=test_ratio,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            
            kp200_regime_results = optimize_parameters_with_regime('kp200', 'futures_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=test_ratio,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            
            # 레짐별 결과 출력
            print(f"\n{'='*80}")
            print("[KOSPI 레짐별 최적화 결과]")
            print(f"{'='*80}")
            for regime_name, candidates in kospi_regime_results.items():
                if candidates:
                    best = candidates[0]
                    print(f"\n{regime_name}:")
                    print(f"  atr_multiplier: {best['atr_multiplier']:.4f}")
                    print(f"  confirmation_bars: {best['confirmation_bars']}")
                    print(f"  avg_lag: {best['avg_lag']:.2f}")
                    if 'atr_period' in best:
                        print(f"  atr_period: {best['atr_period']}")
                        print(f"  freeze_on_confirm: {best['freeze_on_confirm']}")
        
            # 전체 데이터 최적화로 폴백
            kospi_train, kospi_test = train_test_split_optimize('kospi', 'kospi_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=test_ratio,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kospi_best = extract_and_save_best_params(kospi_train, 'kospi', 'kospi_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kospi_test)

            kp200_train, kp200_test = train_test_split_optimize('kp200', 'futures_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=test_ratio,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kp200_best = extract_and_save_best_params(kp200_train, 'kp200', 'futures_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kp200_test)
        else:
            # 확장 파라미터만 사용
            kospi_train, kospi_test = train_test_split_optimize('kospi', 'kospi_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=test_ratio,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kospi_best = extract_and_save_best_params(kospi_train, 'kospi', 'kospi_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kospi_test)

            kp200_train, kp200_test = train_test_split_optimize('kp200', 'futures_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=test_ratio,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kp200_best = extract_and_save_best_params(kp200_train, 'kp200', 'futures_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kp200_test)
    elif mode == 4:
        # Phase 3 모드 (시계열 CV, 시간대별 최적화)
        print("\nPhase 3 옵션:")
        cv_input = input("  시계열 교차 검증 사용? (y/n, 기본값 n): ").strip().lower()
        use_cv = cv_input == 'y'
        
        time_based_input = input("  시간대별 최적화 사용? (y/n, 기본값 n): ").strip().lower()
        use_time_based = time_based_input == 'y'
        
        if use_cv:
            # 시계열 교차 검증
            n_splits_input = input("  CV 분할 수 (기본값 5): ").strip()
            n_splits = int(n_splits_input) if n_splits_input else 5
            
            kospi_cv_results = time_series_cv_optimize('kospi', 'kospi_zigzag', n_jobs,
                                                         pivot_count_range, max_avg_lag,
                                                         n_splits, use_enhanced_metrics=True,
                                                         use_extended_params=True)
            kospi_best = extract_and_save_best_params(kospi_cv_results, 'kospi', 'kospi_zigzag',
                                                    max_avg_lag=max_avg_lag)

            kp200_cv_results = time_series_cv_optimize('kp200', 'futures_zigzag', n_jobs,
                                                         pivot_count_range, max_avg_lag,
                                                         n_splits, use_enhanced_metrics=True,
                                                         use_extended_params=True)
            kp200_best = extract_and_save_best_params(kp200_cv_results, 'kp200', 'futures_zigzag',
                                                    max_avg_lag=max_avg_lag)
        elif use_time_based:
            # 시간대별 최적화
            kospi_time_results = optimize_time_based_parameters('kospi', 'kospi_zigzag', n_jobs,
                                                                   pivot_count_range, max_avg_lag,
                                                                   use_enhanced_metrics=True,
                                                                   use_extended_params=True)

            # 시간대별 결과 출력
            print(f"\n{'='*80}")
            print("[KOSPI 시간대별 최적화 결과]")
            print(f"{'='*80}")
            for period, candidates in kospi_time_results.items():
                if candidates:
                    best = candidates[0]
                    print(f"\n{period}:")
                    print(f"  atr_multiplier: {best['atr_multiplier']:.4f}")
                    print(f"  confirmation_bars: {best['confirmation_bars']}")
                    print(f"  avg_lag: {best['avg_lag']:.2f}")

            # Phase 5: session_min_wave_bars_table 변환 (2026-05-10)
            kospi_session_table = convert_time_based_to_session_table(kospi_time_results)
            print(f"\n{'='*80}")
            print("[KOSPI session_min_wave_bars_table]")
            print(f"{'='*80}")
            print(f"  {kospi_session_table}")

            # KP200 시간대별 최적화
            kp200_time_results = optimize_time_based_parameters('kp200', 'futures_zigzag', n_jobs,
                                                                   pivot_count_range, max_avg_lag,
                                                                   use_enhanced_metrics=True,
                                                                   use_extended_params=True)

            # Phase 5: session_min_wave_bars_table 변환 (2026-05-10)
            kp200_session_table = convert_time_based_to_session_table(kp200_time_results)
            print(f"\n{'='*80}")
            print("[KP200 session_min_wave_bars_table]")
            print(f"{'='*80}")
            print(f"  {kp200_session_table}")

            # 전체 데이터 최적화로 폴백
            kospi_train, kospi_test = train_test_split_optimize('kospi', 'kospi_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=0.2,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kospi_best = extract_and_save_best_params(kospi_train, 'kospi', 'kospi_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kospi_test)

            kp200_train, kp200_test = train_test_split_optimize('kp200', 'futures_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=0.2,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kp200_best = extract_and_save_best_params(kp200_train, 'kp200', 'futures_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kp200_test)

            # Phase 5: session_min_wave_bars_table 업데이트 옵션 (2026-05-10)
            update_session_input = input("\n  session_min_wave_bars_table을 config.json에 업데이트하시겠습니까? (y/n, 기본값 n): ").strip().lower()
            if update_session_input == 'y':
                update_config_with_session_table(kospi_session_table, 'kospi_zigzag')
                update_config_with_session_table(kp200_session_table, 'futures_zigzag')
        else:
            # Phase 3 기본 (확장 파라미터 + 균형 샘플링)
            test_ratio_input = input("  테스트 비율 (기본값 0.2): ").strip()
            test_ratio = float(test_ratio_input) if test_ratio_input else 0.2
            
            kospi_train, kospi_test = train_test_split_optimize('kospi', 'kospi_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=test_ratio,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kospi_best = extract_and_save_best_params(kospi_train, 'kospi', 'kospi_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kospi_test)

            kp200_train, kp200_test = train_test_split_optimize('kp200', 'futures_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=test_ratio,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kp200_best = extract_and_save_best_params(kp200_train, 'kp200', 'futures_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kp200_test)
    elif mode == 5:
        # Phase 4 모드 (결과 분석 + 병렬 최적화)
        print("\nPhase 4 옵션:")
        cv_input = input("  시계열 교차 검증 사용? (y/n, 기본값 n): ").strip().lower()
        use_cv = cv_input == 'y'
        
        time_based_input = input("  시간대별 최적화 사용? (y/n, 기본값 n): ").strip().lower()
        use_time_based = time_based_input == 'y'
        
        report_input = input("  최적화 리포트 생성? (y/n, 기본값 y): ").strip().lower()
        use_report = report_input != 'n'
        
        batch_size_input = input("  배치 크기 (기본값 100): ").strip()
        batch_size = int(batch_size_input) if batch_size_input else 100
        
        if use_cv:
            # 시계열 교차 검증
            n_splits_input = input("  CV 분할 수 (기본값 5): ").strip()
            n_splits = int(n_splits_input) if n_splits_input else 5
            
            kospi_cv_results = time_series_cv_optimize('kospi', 'kospi_zigzag', n_jobs,
                                                         pivot_count_range, max_avg_lag,
                                                         n_splits, use_enhanced_metrics=True,
                                                         use_extended_params=True)
            kospi_best = extract_and_save_best_params(kospi_cv_results, 'kospi', 'kospi_zigzag',
                                                    max_avg_lag=max_avg_lag)

            kp200_cv_results = time_series_cv_optimize('kp200', 'futures_zigzag', n_jobs,
                                                         pivot_count_range, max_avg_lag,
                                                         n_splits, use_enhanced_metrics=True,
                                                         use_extended_params=True)
            kp200_best = extract_and_save_best_params(kp200_cv_results, 'kp200', 'futures_zigzag',
                                                    max_avg_lag=max_avg_lag)
            
            # Phase 4: 리포트 생성
            if use_report and kospi_cv_results is not None:
                kospi_report = generate_optimization_report(kospi_cv_results, kospi_best, 'kospi', 'kospi_zigzag')
                print(kospi_report)
                save_optimization_report(kospi_report, 'kospi', 'kospi_zigzag')
            
            if use_report and kp200_cv_results is not None:
                kp200_report = generate_optimization_report(kp200_cv_results, kp200_best, 'kp200', 'futures_zigzag')
                print(kp200_report)
                save_optimization_report(kp200_report, 'kp200', 'futures_zigzag')
        elif use_time_based:
            # 시간대별 최적화
            kospi_time_results = optimize_time_based_parameters('kospi', 'kospi_zigzag', n_jobs,
                                                                   pivot_count_range, max_avg_lag,
                                                                   use_enhanced_metrics=True,
                                                                   use_extended_params=True)
            
            # 시간대별 결과 출력
            print(f"\n{'='*80}")
            print("[KOSPI 시간대별 최적화 결과]")
            print(f"{'='*80}")
            for period, candidates in kospi_time_results.items():
                if candidates:
                    best = candidates[0]
                    print(f"\n{period}:")
                    print(f"  atr_multiplier: {best['atr_multiplier']:.4f}")
                    print(f"  confirmation_bars: {best['confirmation_bars']}")
                    print(f"  avg_lag: {best['avg_lag']:.2f}")
            
            # 전체 데이터 최적화로 폴백
            kospi_train, kospi_test = train_test_split_optimize('kospi', 'kospi_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=0.2,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kospi_best = extract_and_save_best_params(kospi_train, 'kospi', 'kospi_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kospi_test)

            kp200_train, kp200_test = train_test_split_optimize('kp200', 'futures_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=0.2,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kp200_best = extract_and_save_best_params(kp200_train, 'kp200', 'futures_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kp200_test)
            
            # Phase 4: 리포트 생성
            if use_report and kospi_train is not None:
                kospi_report = generate_optimization_report(kospi_train, kospi_best, 'kospi', 'kospi_zigzag')
                print(kospi_report)
                save_optimization_report(kospi_report, 'kospi', 'kospi_zigzag')
            
            if use_report and kp200_train is not None:
                kp200_report = generate_optimization_report(kp200_train, kp200_best, 'kp200', 'futures_zigzag')
                print(kp200_report)
                save_optimization_report(kp200_report, 'kp200', 'futures_zigzag')
        else:
            # Phase 4 기본 (확장 파라미터 + 배치 처리 + 리포트)
            test_ratio_input = input("  테스트 비율 (기본값 0.2): ").strip()
            test_ratio = float(test_ratio_input) if test_ratio_input else 0.2
            
            kospi_train, kospi_test = train_test_split_optimize('kospi', 'kospi_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=test_ratio,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kospi_best = extract_and_save_best_params(kospi_train, 'kospi', 'kospi_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kospi_test)

            kp200_train, kp200_test = train_test_split_optimize('kp200', 'futures_zigzag',
                                                                  pivot_count_range=pivot_count_range,
                                                                  max_avg_lag=max_avg_lag,
                                                                  test_ratio=test_ratio,
                                                                  use_enhanced_metrics=True,
                                                                  use_extended_params=True)
            kp200_best = extract_and_save_best_params(kp200_train, 'kp200', 'futures_zigzag',
                                                    max_avg_lag=max_avg_lag, test_results=kp200_test)
            
            # Phase 4: 리포트 생성
            if use_report and kospi_train is not None:
                kospi_report = generate_optimization_report(kospi_train, kospi_best, 'kospi', 'kospi_zigzag')
                print(kospi_report)
                save_optimization_report(kospi_report, 'kospi', 'kospi_zigzag')
            
            if use_report and kp200_train is not None:
                kp200_report = generate_optimization_report(kp200_train, kp200_best, 'kp200', 'futures_zigzag')
                print(kp200_report)
                save_optimization_report(kp200_report, 'kp200', 'futures_zigzag')

    # config.json 업데이트 (사용자 확인 후)
    print(f"\n{'='*80}")
    print("최적 파라미터 요약")
    print(f"{'='*80}")

    if kospi_best:
        print("\n[KOSPI]")
        print(f"  atr_multiplier          : {kospi_best['atr_multiplier']:.4f}")
        print(f"  pivot_threshold_min_pct : {kospi_best['pivot_threshold_min_pct']:.4f}")
        print(f"  confirmation_bars       : {kospi_best['confirmation_bars']}")
        print(f"  min_wave_atr_ratio      : {kospi_best['min_wave_atr_ratio']:.4f}")
        print(f"  피봇 수                 : {kospi_best['pivot_count']}")
        print(f"  평균 지연시간           : {kospi_best['avg_lag']:.2f} 봉")
        print(f"  최대 지연시간           : {kospi_best['max_lag']} 봉")
        if 'pivot_quality' in kospi_best:
            print(f"  피봇 품질               : {kospi_best['pivot_quality']:.4f}")
            print(f"  교번 준수율           : {kospi_best['alternation_rate']:.4f}")
        if 'atr_period' in kospi_best:
            print(f"  atr_period             : {kospi_best['atr_period']}")
            print(f"  freeze_on_confirm      : {kospi_best['freeze_on_confirm']}")
    else:
        print("\n[KOSPI]")
        print("  ❌ 최적화 실패: 유효한 조합을 찾지 못했습니다.")
        print(f"  피봇 수 범위: {pivot_count_range[0]}~{pivot_count_range[1]}개")
        print(f"  최대 평균 지연: {max_avg_lag} 봉")
        print("  제안: 피봇 수 범위를 넓히거나 최대 평균 지연시간을 늘려보세요.")

    if kp200_best:
        print("\n[KP200]")
        print(f"  atr_multiplier          : {kp200_best['atr_multiplier']:.4f}")
        print(f"  pivot_threshold_min_pct : {kp200_best['pivot_threshold_min_pct']:.4f}")
        print(f"  confirmation_bars       : {kp200_best['confirmation_bars']}")
        print(f"  min_wave_atr_ratio      : {kp200_best['min_wave_atr_ratio']:.4f}")
        print(f"  피봇 수                 : {kp200_best['pivot_count']}")
        print(f"  평균 지연시간           : {kp200_best['avg_lag']:.2f} 봉")
        print(f"  최대 지연시간           : {kp200_best['max_lag']} 봉")
        if 'pivot_quality' in kp200_best:
            print(f"  피봇 품질               : {kp200_best['pivot_quality']:.4f}")
            print(f"  교번 준수율           : {kp200_best['alternation_rate']:.4f}")
        if 'atr_period' in kp200_best:
            print(f"  atr_period             : {kp200_best['atr_period']}")
            print(f"  freeze_on_confirm      : {kp200_best['freeze_on_confirm']}")
    else:
        print("\n[KP200]")
        print("  ❌ 최적화 실패: 유효한 조합을 찾지 못했습니다.")
        print(f"  피봇 수 범위: {pivot_count_range[0]}~{pivot_count_range[1]}개")
        print(f"  최대 평균 지연: {max_avg_lag} 봉")
        print("  제안: 피봇 수 범위를 넓히거나 최대 평균 지연시간을 늘려보세요.")

    print(f"\n{'='*80}")
    print("config.json 업데이트 여부 확인")
    print(f"{'='*80}")

    if kospi_best:
        print("\n[KOSPI] 최적 파라미터 적용하시겠습니까? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            update_config_json(kospi_best, 'kospi_zigzag')

    if kp200_best:
        print("\n[KP200] 최적 파라미터 적용하시겠습니까? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            update_config_json(kp200_best, 'futures_zigzag')

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n{'='*80}")
    print(f"시뮬레이션 완료  |  총 소요시간: {elapsed:.1f}초")
    print(f"{'='*80}")
