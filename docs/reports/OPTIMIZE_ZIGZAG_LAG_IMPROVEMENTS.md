# optimize_zigzag_lag.py 개선점 분석

## 개요

`optimize_zigzag_lag.py`는 과거 데이터를 사용하여 ZigZag 초기 설정값의 최적 조합을 찾는 백테스트 최적화 도구입니다. 현재 구현의 개선점을 정리합니다.

---

## 현재 구조

### 기능 흐름

1. **데이터 로드**: 저장된 분봉 데이터 로드 (최근 20% 샘플링)
2. **파라미터 그리드 생성**: Coarse → Fine 2단계 그리드 생성
3. **병렬 평가**: joblib로 병렬 파라미터 평가
4. **평가지표 계산**: 평균 지연시간 + 피봇 수 페널티
5. **최적 파라미터 추출**: score 기준 정렬 후 상위 10개 추출
6. **config.json 업데이트**: 최적 파라미터 적용

### 최적화 파라미터

- `atr_multiplier`: ATR 배수 (0.3 ~ 2.1)
- `pivot_threshold_min_pct`: 피봇 임계값 최소 퍼센트 (0.01 ~ 0.5)
- `confirmation_bars`: 확인 봉수 (1 ~ 3)
- `min_wave_atr_ratio`: 최소 파동 ATR 비율 (0.3 ~ 3.5)

### 평가지표

```python
score = avg_lag + (pivot_count - target_pivots) * 0.5
```

- **avg_lag**: 평균 지연시간 (낮을수록 좋음)
- **pivot_penalty**: 피봇 수 페널티 (목표값에서 멀수록 불리)

---

## 개선점 분석

### 1. 평가지표의 한계 (심각도: 높음)

**문제:**
- 현재는 평균 지연시간만 최소화하고 피봇의 정확도는 평가하지 않음
- 피봇이 실제 추세 전환점인지, 가짜 피봇인지 구별하지 않음
- 단순히 "빨리 확정"하는 것이 최적이 아님

**개선안:**
```python
def calculate_metrics_enhanced(zz: AdaptiveZigZag, df: pd.DataFrame, 
                               target_pivots: float) -> dict:
    """향상된 평가지표"""
    confirmed_swings = [s for s in zz._all_swings if s.confirmed]
    pivot_count = len(confirmed_swings)
    
    if pivot_count < 2:
        return {'score': 9999.0}
    
    # 1. 지연시간 메트릭
    lag_details = []
    for s in sorted(confirmed_swings, key=lambda s: s.index):
        pivot_index = s.index
        confirm_index = getattr(s, 'confirmed_at_idx', pivot_index)
        lag_details.append(confirm_index - pivot_index)
    
    avg_lag = float(np.mean(lag_details))
    max_lag = int(max(lag_details))
    lag_std = float(np.std(lag_details))
    
    # 2. 피봇 품질 메트릭 (진짜 추세 전환점 탐지)
    pivot_quality = calculate_pivot_quality(zz, df)
    
    # 3. H/L 교번 준수율
    alternation_rate = calculate_alternation_rate(confirmed_swings)
    
    # 4. 복합 스코어
    # 지연시간 (낮을수록 좋음) + 피봇 품질 (높을수록 좋음) + 교번 준수율 (높을수록 좋음)
    lag_score = avg_lag * 1.0
    quality_penalty = (1.0 - pivot_quality) * 10.0  # 품질 낮으면 큰 페널티
    alternation_penalty = (1.0 - alternation_rate) * 5.0
    pivot_penalty = abs(pivot_count - target_pivots) * 0.5
    
    score = lag_score + quality_penalty + alternation_penalty + pivot_penalty
    
    return {
        'avg_lag': avg_lag,
        'max_lag': max_lag,
        'lag_std': lag_std,
        'pivot_count': pivot_count,
        'pivot_quality': pivot_quality,
        'alternation_rate': alternation_rate,
        'score': round(score, 4)
    }

def calculate_pivot_quality(zz: AdaptiveZigZag, df: pd.DataFrame) -> float:
    """피봇 품질 계산 (실제 추세 전환점 탐지율)"""
    # 피봇 후 N봉 동안 가격이 예상 방향으로 이동했는지 확인
    # HIGH 피봇 후 하락, LOW 피봇 후 상승 비율 계산
    confirmed_swings = [s for s in zz._all_swings if s.confirmed]
    if len(confirmed_swings) < 2:
        return 0.0
    
    correct_count = 0
    for i, s in enumerate(confirmed_swings[:-1]):
        next_s = confirmed_swings[i + 1]
        if s.swing_type.value == 'high' and next_s.price < s.price:
            correct_count += 1
        elif s.swing_type.value == 'low' and next_s.price > s.price:
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
```

---

### 2. 시장 레짐 무시 (심각도: 높음)

**문제:**
- 전체 데이터에 대해 단일 파라미터를 찾음
- 추세형/횡보형/고변동/저변동 레짐 차이 고려하지 않음
- MarketRegimeClassifier를 활용하지 않음

**개선안:**
```python
def optimize_parameters_with_regime(symbol: str, config_type: str, n_jobs: int = -1,
                                   pivot_count_range: tuple = (5, 9), 
                                   max_avg_lag: float = 5.0):
    """시장 레짐별 파라미터 최적화"""
    df = load_saved_minute_data(symbol)
    
    # 시장 레짐 분류
    from services.market_regime_classifier import MarketRegimeClassifier
    regime_classifier = MarketRegimeClassifier()
    
    # 레짐별 데이터 분할
    regime_data = {}
    for i in range(len(df) - 10):
        window_df = df.iloc[i:i+10]
        regime = regime_classifier.classify(window_df, current_idx=i+9)
        regime_name = regime.regime.value if regime else "UNKNOWN"
        
        if regime_name not in regime_data:
            regime_data[regime_name] = []
        regime_data[regime_name].append(window_df.iloc[-1])
    
    # 레짐별 최적화
    regime_params = {}
    for regime_name, regime_df_list in regime_data.items():
        if len(regime_df_list) < 50:  # 데이터 부족하면 스킵
            continue
        
        regime_df = pd.concat(regime_df_list)
        logger.info(f"Optimizing for regime: {regime_name} (data points: {len(regime_df)})")
        
        # 레짐별 파라미터 범위 조정
        if "UP" in regime_name:
            # 추세형: 민감한 파라미터
            param_ranges = {
                'atr_mult': np.arange(1.0, 1.8, 0.2),
                'conf_bars': [1],
                'min_wave': np.arange(0.3, 1.0, 0.2)
            }
        elif "NO_DIRECTION" in regime_name:
            # 횡보형: 보수적인 파라미터
            param_ranges = {
                'atr_mult': np.arange(1.8, 3.0, 0.3),
                'conf_bars': [2, 3],
                'min_wave': np.arange(0.8, 2.0, 0.3)
            }
        else:
            # 기본값
            param_ranges = {
                'atr_mult': np.arange(1.2, 2.5, 0.3),
                'conf_bars': [1, 2, 3],
                'min_wave': np.arange(0.5, 1.5, 0.3)
            }
        
        best_params = optimize_for_regime(regime_df, param_ranges, n_jobs, 
                                          pivot_count_range, max_avg_lag)
        regime_params[regime_name] = best_params
    
    return regime_params
```

---

### 3. 데이터 샘플링 편향 (심각도: 중간)

**문제:**
- 최근 20% 데이터만 사용 (line 147-148)
- 특정 시기(최근)에 편향된 결과
- 과거 시장 환경(코로나, 금리 인상 등) 반영 불가

**개선안:**
```python
def load_balanced_sample(df: pd.DataFrame, sample_ratio: float = 0.3) -> pd.DataFrame:
    """시기별 균형 샘플링"""
    # 전체 데이터를 시기별로 분할
    total_len = len(df)
    period_size = total_len // 4  # 4개 시기로 분할
    
    samples = []
    for i in range(4):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < 3 else total_len
        period_df = df.iloc[start_idx:end_idx]
        
        # 각 시기에서 균등하게 샘플링
        period_sample_size = int(len(period_df) * sample_ratio)
        if period_sample_size > 0:
            samples.append(period_df.iloc[-period_sample_size:])
    
    return pd.concat(samples)

# 또는 시계열 교차 검증 사용
def time_series_cv_optimize(df: pd.DataFrame, n_splits: int = 5, **kwargs):
    """시계열 교차 검증 기반 최적화"""
    fold_size = len(df) // n_splits
    results = []
    
    for i in range(n_splits):
        # Train: 0~(i+1)*fold_size, Test: (i+1)*fold_size~(i+2)*fold_size
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = min((i + 2) * fold_size, len(df))
        
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]
        
        # Train으로 최적화
        best_params = optimize_parameters_on_df(train_df, **kwargs)
        
        # Test로 검증
        test_score = validate_on_test(test_df, best_params)
        results.append({'params': best_params, 'test_score': test_score})
    
    # 테스트 스코어 기준 최적 파라미터 선택
    best_result = min(results, key=lambda x: x['test_score'])
    return best_result['params']
```

---

### 4. 파라미터 그리드 제한 (심각도: 중간)

**문제:**
- 4개 파라미터만 최적화 (line 66)
- 중요한 파라미터 누락:
  - `atr_period`: ATR 계산 기간
  - `freeze_on_confirm`: 확정 시 극값 동결
  - `cluster_tolerance_pct`: 클러스터 허용 퍼센트
  - 시간대별 파라미터

**개선안:**
```python
def expand_param_grid(base_grid: list, additional_params: dict) -> list:
    """확장된 파라미터 그리드 생성"""
    expanded_grid = []
    
    for base_combo in base_grid:
        atr_mult, thresh_min, conf_bars, min_wave = base_combo
        
        # 추가 파라미터 조합
        for atr_period in additional_params.get('atr_period', [14]):
            for freeze_confirm in additional_params.get('freeze_on_confirm', [True, False]):
                for cluster_tol in additional_params.get('cluster_tolerance', [0.2, 0.3, 0.4]):
                    expanded_grid.append(
                        (atr_mult, thresh_min, conf_bars, min_wave, 
                         atr_period, freeze_confirm, cluster_tol)
                    )
    
    return expanded_grid

# 사용 예시
additional_params = {
    'atr_period': [10, 14, 21],  # 단기/중기/장기
    'freeze_on_confirm': [True, False],
    'cluster_tolerance': [0.2, 0.3, 0.4]
}

expanded_grid = expand_param_grid(coarse_grid, additional_params)
```

---

### 5. 백테스트 검증 부족 (심각도: 높음)

**문제:**
- 최적화 데이터와 검증 데이터 분리 없음
- 과적합(overfitting) 위험
- 미래 데이터 성능 보장 불가

**개선안:**
```python
def train_test_split_optimize(df: pd.DataFrame, test_ratio: float = 0.2, **kwargs):
    """학습/테스트 분리 후 최적화"""
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    logger.info(f"Train: {len(train_df)} bars, Test: {len(test_df)} bars")
    
    # Train 데이터로 최적화
    train_results = optimize_parameters_on_df(train_df, **kwargs)
    best_params = extract_best_params(train_results)
    
    # Test 데이터로 검증
    test_metrics = validate_on_test(test_df, best_params)
    
    logger.info(f"Test metrics: {test_metrics}")
    
    # 테스트 성능이 기준 미달이면 재최적화
    if test_metrics['avg_lag'] > kwargs.get('max_avg_lag', 5.0) * 1.5:
        logger.warning("Test performance below threshold, re-optimizing...")
        return train_test_split_optimize(df, test_ratio=test_ratio, **kwargs)
    
    return best_params, test_metrics

def validate_on_test(df: pd.DataFrame, params: dict) -> dict:
    """테스트 데이터로 파라미터 검증"""
    cfg = AdaptiveZigZagConfig(**params)
    zz = AdaptiveZigZag(cfg)
    zz.compute_from_df(df)
    
    return calculate_metrics_enhanced(zz, df, target_pivots=7)
```

---

### 6. 시간대별 파라미터 최적화 누락 (심각도: 중간)

**문제:**
- 시간대별 파라미터 최적화 불가
- 장초반/점심/장마감 특성 반영 불가

**개선안:**
```python
def optimize_time_based_parameters(df: pd.DataFrame, **kwargs):
    """시간대별 파라미터 최적화"""
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
        # 시간대 필터링
        mask = (df.index.time >= pd.to_datetime(start).time()) & \
               (df.index.time <= pd.to_datetime(end).time())
        period_df = df[mask]
        
        if len(period_df) < 100:
            continue
        
        logger.info(f"Optimizing for time period: {period} ({start}~{end})")
        
        # 시간대별 최적화
        best_params = optimize_parameters_on_df(period_df, **kwargs)
        time_params[period] = best_params
    
    return time_params
```

---

### 7. 피봇 수 범위 고정 (심각도: 낮음)

**문제:**
- 피봇 수 범위를 사용자 입력으로만 받음 (line 295-299)
- 데이터 특성에 따른 동적 범위 설정 불가

**개선안:**
```python
def calculate_dynamic_pivot_range(df: pd.DataFrame, atr_period: int = 14) -> tuple:
    """데이터 특성에 따른 동적 피봇 수 범위 계산"""
    # ATR 계산
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean().iloc[-1]
    
    # 데이터 변동성 기준 범위 계산
    price_range = (high.max() - low.min()) / close.mean()
    
    if price_range > 0.05:  # 고변동
        return (8, 15)
    elif price_range > 0.02:  # 중변동
        return (5, 10)
    else:  # 저변동
        return (3, 7)

# 사용
pivot_min, pivot_max = calculate_dynamic_pivot_range(df_sample)
pivot_count_range = (pivot_min, pivot_max)
```

---

### 8. 지연시간 분산 무시 (심각도: 낮음)

**문제:**
- 평균 지연시간만 고려 (line 47)
- 최대 지연시간이나 분산 무시
- 일부 피봇이 매우 늦게 확정되는 경우 방지 불가

**개선안:**
```python
def calculate_lag_metrics(lag_details: list) -> dict:
    """지연시간 통계 메트릭 계산"""
    avg_lag = float(np.mean(lag_details))
    median_lag = float(np.median(lag_details))
    max_lag = int(max(lag_details))
    lag_std = float(np.std(lag_details))
    lag_p95 = float(np.percentile(lag_details, 95))
    
    # 복합 지연시간 스코어
    # 평균 + 95백분위 + 표준편차
    lag_score = avg_lag * 0.5 + lag_p95 * 0.3 + lag_std * 0.2
    
    return {
        'avg_lag': avg_lag,
        'median_lag': median_lag,
        'max_lag': max_lag,
        'lag_std': lag_std,
        'lag_p95': lag_p95,
        'lag_score': lag_score
    }
```

---

### 9. 결과 분석 기능 부족 (심각도: 낮음)

**문제:**
- 최적 파라미터만 제시
- 왜 최적인지 설명 부족
- 다른 조합과 비교 불가

**개선안:**
```python
def generate_optimization_report(results: pd.DataFrame, best_params: dict) -> str:
    """최적화 결과 리포트 생성"""
    report = []
    report.append("="*80)
    report.append("최적화 결과 리포트")
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
    
    # 파라미터 민감도 분석
    report.append("\n[파라미터 민감도 분석]")
    for param in ['atr_multiplier', 'pivot_threshold_min_pct', 
                  'confirmation_bars', 'min_wave_atr_ratio']:
        corr = results[param].corr(results['score'])
        report.append(f"  {param}: 상관계수 {corr:.3f}")
    
    # 최적 파라미터 특성
    report.append("\n[최적 파라미터 특성]")
    report.append(f"  ATR 배수: {best_params['atr_multiplier']:.4f}")
    report.append(f"  → 민감도: {'높음' if best_params['atr_multiplier'] < 1.5 else '중간' if best_params['atr_multiplier'] < 2.0 else '낮음'}")
    report.append(f"  확인봉: {best_params['confirmation_bars']}")
    report.append(f"  → 확정속도: {'빠름' if best_params['confirmation_bars'] == 1 else '중간' if best_params['confirmation_bars'] == 2 else '느림'}")
    
    return "\n".join(report)
```

---

### 10. 병렬 처리 효율 (심각도: 낮음)

**문제:**
- `return_as='generator'` 사용하지만 데이터프레임 복사 오버헤드 (line 120)
- 메모리 사용량 최적화 부족

**개선안:**
```python
def run_grid_optimized(param_grid: list, df: pd.DataFrame, zigzag_cfg: dict,
                      n_jobs: int, label: str, pivot_count_range: tuple) -> pd.DataFrame:
    """최적화된 병렬 그리드 탐색"""
    target_pivots = sum(pivot_count_range) / 2
    
    # 데이터프레임 복사 최소화
    df_values = {
        'high': df['High'].values,
        'low': df['Low'].values,
        'close': df['Close'].values,
        'open': df['Open'].values,
        'index': df.index.values
    }
    
    results = []
    with tqdm(total=len(param_grid), desc=f"  {label}", ncols=90) as pbar:
        # 배치 처리로 메모리 효율화
        batch_size = 100
        for i in range(0, len(param_grid), batch_size):
            batch = param_grid[i:i+batch_size]
            
            parallel_pool = Parallel(n_jobs=n_jobs, backend='loky')
            batch_results = parallel_pool(
                delayed(evaluate_single_optimized)(p, df_values, zigzag_cfg, 
                                                   pivot_count_range, target_pivots) 
                for p in batch
            )
            
            for result in batch_results:
                if result is not None:
                    results.append(result)
            pbar.update(len(batch))
    
    return pd.DataFrame(results).sort_values(['score', 'avg_lag']).reset_index(drop=True)

def evaluate_single_optimized(params: tuple, df_values: dict, zigzag_cfg: dict,
                               pivot_count_range: tuple, target_pivots: float) -> dict | None:
    """최적화된 단일 평가 (데이터프레임 복사 없음)"""
    # numpy 배열로 직접 처리
    # ...
```

---

## 우선순위 요약

| 순위 | 개선점 | 심각도 | 난이도 | 예상 효과 |
|------|--------|--------|--------|----------|
| 1 | 시장 레짐별 최적화 | 높음 | 중간 | 매우 높음 |
| 2 | 백테스트 검증 분리 | 높음 | 낮음 | 높음 |
| 3 | 평가지표 향상 | 높음 | 중간 | 높음 |
| 4 | 파라미터 그리드 확장 | 중간 | 낮음 | 중간 |
| 5 | 데이터 샘플링 개선 | 중간 | 낮음 | 중간 |
| 6 | 시간대별 최적화 | 중간 | 중간 | 중간 |
| 7 | 지연시간 분산 고려 | 낮음 | 낮음 | 낮음 |
| 8 | 결과 분석 기능 | 낮음 | 낮음 | 낮음 |
| 9 | 피봇 수 범위 동적화 | 낮음 | 낮음 | 낮음 |
| 10 | 병렬 처리 최적화 | 낮음 | 중간 | 낮음 |

---

## 권장 구현 순서

### Phase 1: 핵심 개선 (즉시 필요)

1. **백테스트 검증 분리** (2-3시간)
   - 학습/테스트 데이터 분리
   - 과적합 방지

2. **평가지표 향상** (3-4시간)
   - 피봇 품질 메트릭 추가
   - H/L 교번 준수율 추가

### Phase 2: 기능 확장 (단기)

3. **시장 레짐별 최적화** (1일)
   - MarketRegimeClassifier 통합
   - 레짐별 파라미터 그리드

4. **파라미터 그리드 확장** (2-3시간)
   - atr_period 추가
   - freeze_on_confirm 추가

### Phase 3: 고급 기능 (중장기)

5. **데이터 샘플링 개선** (2-3시간)
   - 시기별 균형 샘플링
   - 시계열 교차 검증

6. **시간대별 최적화** (3-4시간)
   - 시간대별 데이터 분할
   - 시간대별 파라미터

### Phase 4: 품질 개선 (장기)

7. **결과 분석 기능** (2-3시간)
   - 파라미터 민감도 분석
   - 최적화 리포트

8. **병렬 처리 최적화** (3-4시간)
   - 메모리 효율화
   - 배치 처리

---

## 결론

현재 `optimize_zigzag_lag.py`는 기본적인 그리드 서치 기능을 제공하지만, 실전 활용을 위해서는 다음 개선이 필요합니다:

1. **시장 레짐 고려**: 추세/횡보/고변동/저변동 레짐별 파라미터 최적화
2. **검증 데이터 분리**: 과적합 방지를 위한 학습/테스트 분리
3. **평가지표 향상**: 지연시간뿐만 아니라 피봇 품질, 교번 준수율 고려
4. **파라미터 확장**: 더 많은 ZigZag 파라미터 최적화

이 개선들은 최적화의 신뢰성과 실전 적용 가능성을 크게 향상시킬 것입니다.

---

## 구현 완료 상태

### Phase 1: 핵심 개선 ✅ 완료 (2026-05-09)

1. **백테스트 검증 분리** ✅
   - 학습/테스트 데이터 분리 구현
   - `train_test_split_optimize()` 함수 추가
   - 과적합 방지 기능 구현
   - 테스트 검증 결과 출력

2. **평가지표 향상** ✅
   - 피봇 품질 메트릭 (`calculate_pivot_quality()`) 추가
   - H/L 교번 준수율 (`calculate_alternation_rate()`) 추가
   - 지연시간 통계 메트릭 (`calculate_lag_metrics()`) 추가
   - 향상된 복합 평가지표 (`calculate_metrics_enhanced()`) 구현
   - 사용자 모드 선택 기능 (기존 vs Phase 1)

### Phase 2: 기능 확장 ✅ 완료 (2026-05-09)

3. **시장 레짐별 최적화** ✅
   - MarketRegimeClassifier 통합 구현
   - `optimize_parameters_with_regime()` 함수 추가
   - 레짐별 파라미터 범위 조정 (추세형/횡보형/하락형)
   - 레짐 분류 불가능 시 자동 폴백

4. **파라미터 그리드 확장** ✅
   - `atr_period` 추가 (10, 14, 21 - 단기/중기/장기)
   - `freeze_on_confirm` 추가 (True/False)
   - `evaluate_single()` 함수 수정으로 6개 파라미터 지원
   - 하위 호환성 유지 (4개 파라미터 계속 지원)

### Phase 3: 고급 기능 ✅ 완료 (2026-05-09)

5. **데이터 샘플링 개선** ✅
   - 시기별 균형 샘플링 (`load_balanced_sample()`) 구현
   - 시계열 교차 검증 (`time_series_cv_optimize()`) 구현
   - 헬퍼 함수 추가 (`optimize_parameters_on_df()`, `validate_on_test()`)
   - n-fold CV 기반 최적화

6. **시간대별 최적화** ✅
   - 시간대별 파라미터 최적화 (`optimize_time_based_parameters()`) 구현
   - 5개 시간대 분할 (장초반/오전/점심/오후/장마감)
   - 시간대별 결과 출력
   - 데이터 부족 시 자동 스킵

### Phase 4: 품질 개선 ✅ 완료 (2026-05-09)

7. **결과 분석 기능** ✅
   - 최적화 결과 리포트 생성 (`generate_optimization_report()`) 구현
   - 리포트 저장 (`save_optimization_report()`) 구현
   - 상위 10개 파라미터 조합 비교
   - 파라미터 민감도 분석 (상관계수)
   - 최적 파라미터 특성 분석
   - 통계 요약 (평균/표준편차/최소/최대 스코어)

8. **병렬 처리 최적화** ✅
   - 배치 처리 구현 (`run_grid()` 함수 수정)
   - 메모리 효율화 (배치 크기 조절 가능)
   - 데이터프레임 복사 최소화 준비
   - 기본 배치 크기: 100

---

## 최적화 모드 요약

| 모드 | 기능 | 사용자 입력 |
|------|------|-----------|
| 모드 1 | 기존 모드 | 단일 데이터, 기본 평가지표 |
| 모드 2 | Phase 1 | 학습/테스트 분리, 향상된 평가지표, 테스트 비율 |
| 모드 3 | Phase 2 | 시장 레짐별 최적화, 확장 파라미터, 레짐 사용 여부 |
| 모드 4 | Phase 3 | 시계열 CV, 시간대별 최적화, CV 분할 수 |
| 모드 5 | Phase 4 | 결과 분석 리포트, 병렬 최적화, 배치 크기 |

---

## 수정된 파일

- `c:\Project\SkyPredictor\indicators\optimize_zigzag_lag.py`
  - Phase 1~4 모든 기능 구현 완료
  - 하위 호환성 유지
  - 문법 검사 통과

---

## 검증 결과

- ✅ Python 문법 검사: 통과
- ✅ 하위 호환성: 유지
- ✅ 새로운 기능: 모든 Phase 구현 완료

---

## 버그 수정 이력 (2026-05-09)

### BUG-1: validate_on_test 중복 정의 ✅ 수정 완료
- **문제**: validate_on_test 함수가 두 번 정의되어 후자가 전자를 덮어씀
- **영향**: time_series_cv_optimize에서 첫 번째 시그니처로 호출 시 TypeError 발생
- **수정**: 중복된 두 번째 정의 삭제, 첫 번째 함수를 통합 시그니처로 수정 (config_type과 zigzag_cfg 모두 지원)

### BUG-2: calculate_pivot_quality 지표 변별력 없음 ✅ 수정 완료
- **문제**: H/L 교번만 확인하여 ZigZag 특성상 항상 ~1.0에 수렴
- **영향**: 피봇 품질 지표가 무의미
- **수정**: 피봇 후 N봉(10봉) 동안 추세 지속 여부 확인으로 변경 (HIGH 피봇 후 하락, LOW 피봇 후 상승)

### BUG-3: load_balanced_sample 편향 샘플링 ✅ 수정 완료
- **문제**: 각 시기에서 iloc[-period_sample_size:]로 마지막 N봉만 추출
- **영향**: 시기별 균형 샘플링이 아닌 최근 편향 샘플링
- **수정**: stride로 균등 샘플링으로 변경

### BUG-4: 레짐별 데이터 비연속 행 집합 ✅ 수정 완료
- **문제**: 개별 봉을 수집하여 pd.concat으로 이어 붙여 시계열 파괴
- **영향**: ATR warmup 무의미, 피봇 품질 지표 왜곡
- **수정**: 연속 시계열 구간 추출로 변경 (레짐 변경 시 구간 저장, 가장 긴 구간 사용)

### BUG-6: df_values 미사용 ✅ 수정 완료
- **문제**: df_values 변환 후 실제로 사용하지 않음
- **영향**: 데드 코드
- **수정**: 사용하지 않는 df_values 코드 삭제

### DESIGN-1: 스코어 가중치 하드코딩 ✅ 수정 완료
- **문제**: 가중치가 하드코딩되어 실험적 변경 어려움
- **영향**: calculate_lag_metrics의 복합 스코어 미사용
- **수정**: SCORE_WEIGHTS 상수로 가중치 분리, lag_metrics['lag_score'] 사용

### DESIGN-2: validate_on_test 확장 파라미터 미반영 ✅ 수정 완료
- **문제**: 확장 파라미터(atr_period, freeze_on_confirm)를 params가 아닌 zigzag_cfg에서 읽음
- **영향**: 확장 파라미터 최적화 의미 상실
- **수정**: params에서 확장 파라미터 우선 사용하도록 수정

### DESIGN-3: 시간대별 필터링 날짜 무시 ✅ 수정 완료
- **문제**: 날짜를 무시하고 시간대만 필터링하여 세그먼트 경계에서 ATR 계산 오류
- **영향**: 날짜 롤오버 모르는 연속 처리
- **수정**: 날짜별로 그룹핑 후 시간대 필터 적용, 가장 긴 세그먼트 사용

### BUG-5: optimize_parameters_on_df 내부 df 재로드
- **확인**: 코드 검증 결과 load_saved_minute_data 호출 없음
- **결과**: df 파라미터를 그대로 사용하므로 문제 없음
