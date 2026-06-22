# Optuna 파라미터 최적화 가이드

## 개요

Optuna는 하이퍼파라미터 최적화를 위한 프레임워크로, 본 프로젝트에서는 피봇 탐색 및 필터 파라미터를 최적화하여 백테스트 승률을 높이는 데 사용됩니다.

## 주요 컴포넌트

### 1. 최적화 대상 파라미터

#### 피봇 탐색 파라미터 (HybridAdaptivePivotConfig)

- `base_pct`: 기본 피봇 감지 비율 (0.05 ~ 2.0)
- `base_multiplier`: 기본 배수 (1.0 ~ 10.0)
- `atr_weight`: ATR 가중치 (0.0 ~ 1.0)
- `atr_period`: ATR 기간 (10 ~ 20)
- `multiplier_min`: 최소 배수 (0.5 ~ 1.0)
- `multiplier_max`: 최대 배수 (1.5 ~ 3.0)
- `er_period`: 효율 비율 기간 (5 ~ 15)
- `confirmation_bars`: 확인 바 수 (1 ~ 5)
- `min_wave_pct`: 최소 웨이브 퍼센트 (0.1 ~ 1.0)
- `min_wave_atr_ratio`: 최소 웨이브 ATR 비율 (0.1 ~ 0.5)
- `max_pivots`: 최대 피봇 수 (20 ~ 50)

#### 필터 파라미터

- `min_wave_pct`: 최소 웨이브 퍼센트 필터 (0.1 ~ 1.0)
- `min_pivot_interval_bars`: 피봇 간 최소 간격 바 수 (5 ~ 30)
- `st_distance_threshold`: SuperTrend 거리 임계값 (0.1 ~ 1.0)
- `adx_hold_threshold`: ADX 유지 임계값 (20.0 ~ 50.0)

### 2. Optuna 목적 함수 (optuna_objective)

```python
def optuna_objective(trial):
    # 파라미터 제안
    base_pct = trial.suggest_float('base_pct', 0.05, 2.0)
    base_multiplier = trial.suggest_float('base_multiplier', 1.0, 10.0)
    atr_weight = trial.suggest_float('atr_weight', 0.0, 1.0)
    # ... (나머지 파라미터)
    
    # 피봇 설정 생성
    pivot_config = HybridAdaptivePivotConfig(...)
    
    # 글로벌 필터 파라미터 설정
    global _min_wave_pct, _min_pivot_interval_bars, _st_distance_threshold, _adx_hold_threshold
    _min_wave_pct = trial.suggest_float('min_wave_pct', 0.1, 1.0)
    _min_pivot_interval_bars = trial.suggest_int('min_pivot_interval_bars', 5, 30)
    _st_distance_threshold = trial.suggest_float('st_distance_threshold', 0.1, 1.0)
    _adx_hold_threshold = trial.suggest_float('adx_hold_threshold', 20.0, 50.0)
    
    # 청크 기반 백테스팅
    chunk_size = len(df) // n_chunks
    win_rates = []
    
    for i in range(n_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        pivots = run_pivot_detection(chunk_df, pivot_config, apply_filters=True)
        backtest_result = run_backtest(chunk_df, pivots)
        win_rates.append(backtest_result['overall_win_rate'])
    
    # 평균 승률 계산
    avg_win_rate = sum(win_rates) / len(win_rates)
    
    # 페널티 적용
    pivots_per_day = len(pivots) / max(1, len(chunk_df.resample('D').size()))
    if pivots_per_day < 5:
        avg_win_rate *= 0.8  # 피봇이 너무 적으면 페널티
    elif pivots_per_day > 50:
        avg_win_rate *= 0.9  # 피봇이 너무 많으면 페널티
    
    return avg_win_rate
```

### 3. 청크 기반 백테스팅

데이터를 여러 청크로 나누어 각각 백테스트를 수행하고 평균 승률을 계산합니다.

```python
n_chunks = 3  # 또는 5
chunk_size = len(df) // n_chunks

for i in range(n_chunks):
    chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
    # 피봇 탐지 및 백테스트
```

**장점:**
- 대용량 데이터 처리 가능
- 과적합 방지 (다양한 기간의 성능 확인)
- 최적화 속도 향상

### 4. 필터 적용 로직

피봇 탐지 시 다음 필터를 적용합니다:

```python
def run_pivot_detection(df, pivot_config, apply_filters=True):
    global _min_wave_pct, _min_pivot_interval_bars, _st_distance_threshold, _adx_hold_threshold
    
    min_wave_pct = _min_wave_pct if apply_filters else 0.3
    min_pivot_interval_bars = _min_pivot_interval_bars if apply_filters else 10
    st_distance_threshold = _st_distance_threshold if apply_filters else 0.1
    adx_hold_threshold = _adx_hold_threshold if apply_filters else 15.0
    
    # 피봇 탐지 및 필터링
    # ...
```

**필터 종류:**
- **P1 필터**: 최소 웨이브 퍼센트 (피봇 간 최소 가격 변동)
- **P2 필터**: 피봇 간 최소 간격 (너무 가까운 피봇 제거)
- **P5 필터**: SuperTrend 거리 (트렌드 확인)
- **P10 필터**: ADX 임계값 (트렌드 강도 확인)

### 5. 페널티 로직

피봇 수가 적절하지 않을 경우 페널티를 적용합니다:

```python
pivots_per_day = len(pivots) / max(1, len(chunk_df.resample('D').size()))

if pivots_per_day < 5:
    avg_win_rate *= 0.8  # 너무 적은 피봇
elif pivots_per_day > 50:
    avg_win_rate *= 0.9  # 너무 많은 피봇
```

**목적:**
- 너무 적은 피봇: 거래 기회 부족
- 너무 많은 피봇: 노이즈 많은 신호, 과매매

### 6. 최적화 실행 함수

```python
def run_optuna_optimization(df, n_trials=50, n_chunks=3, output_dir=None):
    # Optuna 스터디 생성
    study = optuna.create_study(direction='maximize')
    
    # 목적 함수 래핑
    def objective(trial):
        return optuna_objective(trial, df, n_chunks)
    
    # 최적화 실행
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 결과 출력
    print(f"최적 파라미터: {study.best_params}")
    print(f"최적 승률: {study.best_value:.2f}%")
    
    # 결과 저장
    result = {
        'best_params': study.best_params,
        'best_win_rate': study.best_value,
        'n_trials': n_trials,
        'n_chunks': n_chunks
    }
    
    with open(output_dir / 'optuna_optimization.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return study.best_params
```

## 실행 모드

### 1. optuna 모드

```bash
python "48. 피봇탐색_성능검증.py" --mode optuna --trials 50 --chunks 3
```

- 훈련 데이터 (60일)로 최적화
- 테스트 데이터로 검증
- 결과를 `optuna_optimization.json`에 저장

### 2. analyze 모드

```bash
python "48. 피봇탐색_성능검증.py" --mode analyze
```

- 최적 파라미터 로드
- 전체 데이터 (240일)로 상세 백테스트
- 일별 통계, 롱/숏 분석, 보유 시간 등 상세 분석
- 결과를 `optuna_detailed_analysis.json`에 저장

## 데이터 흐름

### optuna 모드

```
데이터 로드 (60일 훈련 데이터)
    ↓
데이터 분할 (훈련/테스트)
    ↓
Optuna 최적화 (훈련 데이터)
    ↓
최적 파라미터 테스트 (테스트 데이터)
    ↓
결과 저장 (optuna_optimization.json)
```

### analyze 모드

```
optuna_optimization.json 로드
    ↓
최적 파라미터 적용
    ↓
전체 데이터 로드 (240일)
    ↓
피봇 탐지 및 필터링
    ↓
백테스트 실행
    ↓
상세 통계 계산
    ↓
결과 저장 (optuna_detailed_analysis.json)
```

## 결과 분석

### optuna_optimization.json

```json
{
  "best_params": {
    "base_pct": 0.5187,
    "base_multiplier": 8.977,
    "atr_weight": 0.467,
    "confirmation_bars": 9,
    "min_wave_pct": 1.945,
    "min_pivot_interval_bars": 19,
    "st_distance_threshold": 0.687,
    "adx_hold_threshold": 43.89
  },
  "best_win_rate": 70.0
}
```

### optuna_detailed_analysis.json

```json
{
  "best_params": { ... },
  "best_win_rate_from_optuna": 70.0,
  "backtest_result": {
    "total_pivots": 544,
    "total_trades": 544,
    "total_wins": "237",
    "overall_win_rate": 43.57,
    "total_profit": -141.05
  },
  "data_period": "2025-06-24 ~ 2026-06-18",
  "daily_stats": [ ... ],
  "trade_stats": { ... },
  "long_short_analysis": { ... },
  "hold_time_stats": { ... }
}
```

## 과적합 문제

### 문제 현상

- Optuna 최적화 승률: 70%
- 전체 데이터 백테스트 승률: 43.57%
- 총 손실: -141.05

### 원인

1. **훈련/테스트 데이터 분리 부족**: 최적화에 사용한 데이터와 테스트 데이터의 기간이 명확히 분리되지 않음
2. **파라미터 범위 과도한 최적화**: 특정 기간에 과도하게 최적화된 파라미터
3. **승률만 최적화**: 리스크 지표(최대 낙폭, 샤프 비율 등) 고려 부족

### 해결 방안

1. **훈련/테스트 분리 강화**
   - 최적화: 과거 데이터 (예: 60일)
   - 검증: 최근 데이터 (예: 30일)
   - 테스트: 미래 데이터 (예: 30일)

2. **파라미터 범위 조정**
   - 더 넓은 범위 탐색
   - 로그 스케일 사용
   - 범주형 파라미터 고려

3. **다목적 최적화**
   - 승률 + 최대 낙폭
   - 승률 + 샤프 비율
   - 승률 + profit factor

4. **교차 검증**
   - 시계열 교차 검증 (Time Series Cross-Validation)
   - Walk-forward validation

5. **데이터 기간 확대**
   - 더 긴 기간의 데이터로 최적화
   - 다양한 시장 조건 포함

## 향후 개선 사항

1. **다목적 최적화 구현**
   ```python
   def multi_objective(trial):
       win_rate = calculate_win_rate(...)
       max_drawdown = calculate_max_drawdown(...)
       return win_rate, -max_drawdown  # 최대화, 최소화
   ```

2. **Pruner 사용**
   ```python
   study = optuna.create_study(
       direction='maximize',
       pruner=optuna.pruners.MedianPruner()
   )
   ```

3. **Sampler 최적화**
   ```python
   sampler = optuna.samplers.TPESampler(
       multivariate=True,
       seed=42
   )
   study = optuna.create_study(
       direction='maximize',
       sampler=sampler
   )
   ```

4. **하이퍼파라미터 중요도 분석**
   ```python
   importance = optuna.importance.get_param_importances(study)
   ```

## 참고 파일

- `Devcenter/48. 피봇탐색_성능검증.py` - Optuna 최적화 및 백테스트 메인 스크립트
- `indicators/hybrid_adaptive_pivot.py` - 피봇 탐지 로직
- `Devcenter/data/backtest_results/optuna_optimization.json` - Optuna 최적화 결과
- `Devcenter/data/backtest_results/optuna_detailed_analysis.json` - 상세 분석 결과

## 추가 리소스

- [Optuna 공식 문서](https://optuna.org/)
- [Optuna 튜토리얼](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [하이퍼파라미터 최적화 베스트 프랙티스](https://optuna.readthedocs.io/en/stable/faq.html)
