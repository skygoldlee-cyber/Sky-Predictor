---
description: ZigZag 파라미터 최적화 가이드
---

# ZigZag 파라미터 최적화 가이드

## 개요

AdaptiveZigZag 인디케이터는 ATR (Average True Range) 기반 필터링을 사용하여 피봇 감지 민감도를 동적으로 조정할 수 있습니다. 이 가이드는 ZigZag 파라미터를 최적화하여 피봇 감지 성능을 향상시키는 방법을 설명합니다.

**현재 설정 (2026-05-01)**:
- `use_atr_based_filtering`: 상위 `zigzag` 섹션에서 관리 (true)
- KOSPI: `min_wave_atr_ratio` = 1.5 (피봇 5개)
- KP200 (Futures): `min_wave_atr_ratio` = 2.7 (피봇 3개)

---

## 1. 기본 파라미터

### 1.1 주요 파라미터

#### `use_atr_based_filtering`
- **타입**: boolean
- **설명**: ATR 기반 필터링 사용 여부
- **기본값**: false
- **효과**: true로 설정하면 ATR 기반 필터가 활성화되어 변동성에 따른 동적 임계값이 적용됨

#### `min_wave_atr_ratio`
- **타입**: float
- **설명**: 최소 파동 크기를 ATR의 비율로 지정
- **범위**: 0.0 ~ 5.0+
- **기본값**: 0.5 (상위 zigzag 섹션)
- **현재 설정값**:
  - KOSPI: 1.5
  - KP200 (Futures): 2.7
- **효과**:
  - 값이 낮을수록 더 작은 파동도 피봇으로 인식 → 피봇 갯수 증가
  - 값이 높을수록 큰 파동만 피봇으로 인식 → 피봇 갯수 감소
  - 예: ATR=5.0, ratio=1.5 → 최소 파동 = 7.5

#### `cluster_atr_ratio`
- **타입**: float
- **설명**: 피봇 클러스터링(인접 피봇 병합)에 사용되는 ATR 비율
- **범위**: 0.0 ~ 5.0+
- **기본값**: 0.5 (상위 zigzag 섹션), 1.0 (futures 오버라이드)
- **현재 설정값**:
  - KOSPI: 1.0
  - KP200 (Futures): 1.0
- **효과**:
  - 인접한 피봇들이 이 값 이내의 거리에 있으면 하나로 병합
  - 값이 낮을수록 더 가까운 피봇들도 분리 → 피봇 갯수 증가
  - 값이 높을수록 더 멀리 있는 피봇들도 병합 → 피봇 갯수 감소

#### `min_wave_pct`
- **타입**: float
- **설명**: 최소 파동 백분율 (ATR 필터와 함께 사용)
- **범위**: 0.0 ~ 1.0
- **기본값**: 0.25
- **효과**: ATR 필터와 별도로 가격 변동 백분율 기준으로도 필터링

#### `confirmation_bars_ranging`
- **타입**: int
- **설명**: 횡보장에서 피봇 확정에 필요한 봉 수
- **기본값**: 2
- **효과**: 값이 낮을수록 피봇이 더 빨리 확정됨

#### `pivot_threshold_min_pct`
- **타입**: float
- **설명**: 피봇 임계값 최소 백분율
- **기본값**: 0.3
- **효과**: 값이 낮을수록 더 작은 변동도 피봇으로 인식

### 1.2 데이터 소스별 설정

#### config.json 구조

```json
{
  "adaptive_indicator": {
    "zigzag": {
      // 상위 설정 (KOSPI와 Futures 모두에 적용되는 기본값)
      "use_atr_based_filtering": true,
      "min_wave_atr_ratio": 0.5,
      "cluster_atr_ratio": 0.5,
      "min_wave_pct": 0.25,
      "confirmation_bars_ranging": 2,
      "pivot_threshold_min_pct": 0.3
    },
    "kospi_zigzag": {
      // KOSPI 전용 설정 (상위 설정을 오버라이드)
      "pivot_lifecycle_log": true,
      "pivot_lifecycle_log_prefix": "[KOSPI]",
      "min_wave_atr_ratio": 1.5,      // 상위 0.5 대신 1.5 사용
      "cluster_atr_ratio": 1.0,
      "min_wave_pct": 0.15,
      "confirmation_bars_ranging": 1,
      "pivot_threshold_min_pct": 0.2
    },
    "futures_zigzag": {
      // Futures 전용 설정 (상위 설정을 오버라이드)
      "pivot_lifecycle_log": true,
      "pivot_lifecycle_log_prefix": "[KP200]",
      "min_wave_atr_ratio": 2.7,      // 상위 0.5 대신 2.7 사용
      "cluster_atr_ratio": 1.0
    }
  }
}
```

**참고**: `use_atr_based_filtering`은 상위 `zigzag` 섹션에서 관리하며, KOSPI와 Futures 모두 동일하게 적용됩니다. 데이터 소스별로 다른 `min_wave_atr_ratio`와 `cluster_atr_ratio`를 설정하여 피봇 감지 민감도를 조절할 수 있습니다.

---

## 2. 파라미터 상호작용

각 파라미터는 독립적으로 작동하지만, 실제로는 상호작용이 존재합니다. 파라미터 조합 시 주의사항:

| 파라미터 조합 | 예상 효과 | 주의사항 |
|-------------|----------|----------|
| `min_wave_atr_ratio` ↓ + `cluster_atr_ratio` ↓ | 피봇 급증 | 노이즈 폭발 위험 |
| `min_wave_atr_ratio` ↑ + `cluster_atr_ratio` ↓ | 효과 상충 | 예측 불가, 피해야 할 조합 |
| `confirmation_bars` ↑ + `min_wave_pct` ↓ | 지연 + 민감 | 실용성 저하 |
| `min_wave_atr_ratio` ↓ + `min_wave_pct` ↓ | 과도 민감도 | 거짓 피봇 증가 |
| `cluster_atr_ratio` ↑ + `min_wave_atr_ratio` ↑ | 과도 보수적 | 중요 피봇 놓침 위험 |

**튜닝 시 권장 순서:**
1. `min_wave_atr_ratio` 먼저 조정 (가장 영향력 큼)
2. `cluster_atr_ratio` 조정
3. `min_wave_pct` 조정
4. `confirmation_bars` 조정
5. `pivot_threshold_min_pct` 조정

---

## 3. 수동 튜닝 가이드

### 3.1 피봇 갯수 조정 가이드

#### 피봇 갯수를 늘리려면

1. **`min_wave_atr_ratio` 낮추기** (가장 효과적)
   - 1.0 → 0.05: 크게 감소시켜 피봇 갯수 크게 증가
   - 예: KOSPI에서 2개 → 7개 증가

2. **`cluster_atr_ratio` 낮추기**
   - 1.0 → 0.05: 인접 피봇 병합 범위 축소
   - 더 많은 개별 피봇 유지

3. **`min_wave_pct` 낮추기**
   - 0.25 → 0.15: 백분율 기준 완화
   - 더 작은 가격 변동도 피봇으로 인식

4. **`confirmation_bars_ranging` 낮추기**
   - 2 → 1: 피봇 확정 속도 향상
   - 더 빠른 피봇 감지

5. **`pivot_threshold_min_pct` 낮추기**
   - 0.3 → 0.2: 임계값 완화
   - 더 낮은 변동성에서도 피봇 감지

#### 피봇 갯수를 줄이려면

위 파라미터들을 반대로 조정:
- `min_wave_atr_ratio`: 0.05 → 0.1 ~ 0.2
- `cluster_atr_ratio`: 0.05 → 0.5 ~ 1.0
- `min_wave_pct`: 0.15 → 0.25 ~ 0.3
- `confirmation_bars_ranging`: 1 → 2 ~ 3
- `pivot_threshold_min_pct`: 0.2 → 0.3 ~ 0.5

### 3.2 추천 시작점

#### 보수적 설정 (피봇 적음)
```json
{
  "min_wave_atr_ratio": 2.0 ~ 3.0,
  "cluster_atr_ratio": 1.0,
  "min_wave_pct": 0.25,
  "confirmation_bars_ranging": 2,
  "pivot_threshold_min_pct": 0.3
}
```
**적용 예**: KP200에 적용 시 피봇 7개 → 3개 감소

#### 중간 설정 (균형)
```json
{
  "min_wave_atr_ratio": 1.0 ~ 1.5,
  "cluster_atr_ratio": 0.5 ~ 1.0,
  "min_wave_pct": 0.2,
  "confirmation_bars_ranging": 2,
  "pivot_threshold_min_pct": 0.25
}
```
**적용 예**: KOSPI에 적용 시 피봇 5개 유지 (현재 설정: 1.5)

#### 공격적 설정 (피봇 많음)
```json
{
  "min_wave_atr_ratio": 0.05 ~ 0.5,
  "cluster_atr_ratio": 0.05 ~ 0.5,
  "min_wave_pct": 0.15,
  "confirmation_bars_ranging": 1,
  "pivot_threshold_min_pct": 0.2
}
```

---

## 4. 빠른 튜닝 의사결정 트리

파라미터 조정 시 어디서 시작해야 할지 불명확할 때 참고하세요.

```
피봇이 너무 많다
    └→ min_wave_atr_ratio 먼저 올리기 (+0.5 단위)
        └→ 여전히 많다 → cluster_atr_ratio 올리기 (+0.3 단위)
            └→ 여전히 많다 → confirmation_bars 올리기 (+1)
                └→ 여전히 많다 → min_wave_pct 올리기 (+0.05)

피봇이 너무 적다
    └→ min_wave_atr_ratio 먼저 내리기 (-0.3 단위)
        └→ 여전히 적다 → min_wave_pct 낮추기 (-0.05)
            └→ 여전히 적다 → pivot_threshold_min_pct 낮추기 (-0.05)
                └→ 여전히 적다 → cluster_atr_ratio 낮추기 (-0.2)

피봇 확정이 너무 늦다
    └→ confirmation_bars 낮추기 (-1)
        └→ 여전히 늦다 → pivot_threshold_min_pct 낮추기 (-0.05)

노이즈 피봇이 너무 많다
    └→ min_wave_atr_ratio 올리기 (+0.3)
        └→ 여전히 많다 → cluster_atr_ratio 올리기 (+0.2)
            └→ 여전히 많다 → min_wave_pct 올리기 (+0.05)
```

---

## 5. 시간대별 동적 튜닝

### 5.1 개요

시장의 변동성은 시간대에 따라 크게 달라집니다. 장 시작 시간은 변동성이 높고, 점심시간은 낮으며, 장 마감 시간은 다시 변동성이 증가하는 패턴을 보입니다. 이러한 패턴을 고려하여 시간대별로 `min_wave_atr_ratio`를 동적으로 조정하면 더 정교한 피봇 감지가 가능합니다.

### 5.2 제안 구조: session_min_wave_atr_ratio_table

기존 `session_min_wave_bars_table`과 유사한 구조로 시간대별 ATR 비율을 설정:

```json
{
  "adaptive_indicator": {
    "zigzag": {
      // 상위 설정 (KOSPI에 적용되는 기본값)
      "session_min_wave_atr_ratio_table": [
        ["09:00", "09:30", 0.8],   // 장 시작: 변동성 높음 → 낮은 비율로 더 많은 피봇
        ["09:30", "10:30", 1.2],   // 오전 활동: 중간 변동성
        ["10:30", "11:30", 1.5],   // 점심 전: 변동성 감소
        ["11:30", "13:00", 2.0],   // 점심시간: 변동성 매우 낮음 → 높은 비율로 노이즈 필터링
        ["13:00", "14:30", 1.2],   // 오후 활동: 중간 변동성
        ["14:30", "15:20", 0.8],   // 장 마감 전: 변동성 증가
        ["15:20", "15:30", 0.5]    // 장 마감: 최고 변동성 → 가장 낮은 비율
      ]
    },
    "futures_zigzag": {
      // KP200 전용 설정 (변동성이 KOSPI보다 크므로 전체적으로 더 높은 비율 적용)
      // KP200 장 시작: 08:45, KOSPI 장 시작: 09:00
      "session_min_wave_atr_ratio_table": [
        ["08:45", "09:00", 1.0],   // 선물 선거래: 변동성 높음 → 중간 비율 (KOSPI 0.8 대비 높게)
        ["09:00", "09:30", 1.2],   // 장 시작: KOSPI 장 개시 → 중간 비율 (KOSPI 0.8 대비 높게)
        ["09:30", "10:30", 1.8],   // 오전 활동: 높은 변동성 (KOSPI 1.2 대비 높게)
        ["10:30", "13:00", 2.5],   // 점심시간: 변동성 매우 낮음 → 최고 비율로 노이즈 필터링
        ["13:00", "14:30", 1.8],   // 오후 활동: 높은 변동성
        ["14:30", "15:20", 1.2],   // 장 마감 전: 변동성 증가
        ["15:20", "15:30", 0.8]    // 장 마감: 최고 변동성 → 낮은 비율
      ]
    }
  }
}
```

**참고**: KP200은 KOSPI보다 변동성이 크므로 동일한 파라미터를 사용하면 과도한 노이즈 필터링이 발생할 수 있습니다. 따라서 KP200에는 더 높은 비율을 적용하여 적절한 피봇 감지를 유지합니다. 또한 KP200은 08:45부터 선거래가 시작되므로 시간대별 설정에 이를 반영해야 합니다.

### 3.3 시간대별 전략

#### 보수적 전략 (노이즈 최소화)

```json
{
  "session_min_wave_atr_ratio_table": [
    ["09:00", "09:30", 1.5],
    ["09:30", "10:30", 2.0],
    ["10:30", "13:00", 2.5],
    ["13:00", "14:30", 2.0],
    ["14:30", "15:20", 1.5],
    ["15:20", "15:30", 1.0]
  ]
}
```
- 모든 시간대에서 높은 비율 유지
- 중요한 피봇만 감지

#### 공격적 전략 (모든 기회 포착)

```json
{
  "session_min_wave_atr_ratio_table": [
    ["09:00", "09:30", 0.3],
    ["09:30", "10:30", 0.5],
    ["10:30", "13:00", 0.8],
    ["13:00", "14:30", 0.5],
    ["14:30", "15:20", 0.3],
    ["15:20", "15:30", 0.2]
  ]
}
```
- 변동성이 높은 시간대에 낮은 비율 적용
- 작은 파동도 피봇으로 감지

#### 균형 전략 (추천)

```json
{
  "session_min_wave_atr_ratio_table": [
    ["09:00", "09:30", 0.8],   // 장 시작: 빠른 반응
    ["09:30", "10:30", 1.2],   // 오전: 안정적
    ["10:30", "13:00", 1.8],   // 점심: 노이즈 필터링
    ["13:00", "14:30", 1.2],   // 오후: 안정적
    ["14:30", "15:20", 0.8],   // 마감 전: 빠른 반응
    ["15:20", "15:30", 0.5]    // 마감: 최고 민감도
  ]
}
```
- 변동성 패턴에 따른 유연한 조정
- 균형 잡힌 피봇 감지

### 3.4 튜닝 가이드라인

**시간대별 비율 설정 원칙:**

1. **장 시작 (09:00-09:30)**: 낮은 비율 (0.5~0.8)
   - 빠른 피봇 감지 필요
   - 변동성 높음

2. **오전 활동 (09:30-10:30)**: 중간 비율 (1.0~1.5)
   - 안정적 피봇 감지
   - 변동성 중간

3. **점심시간 (10:30-13:00)**: 높은 비율 (1.5~2.5)
   - 노이즈 필터링 강화
   - 변동성 낮음

4. **오후 활동 (13:00-14:30)**: 중간 비율 (1.0~1.5)
   - 안정적 피봇 감지
   - 변동성 중간

5. **장 마감 전 (14:30-15:20)**: 낮은 비율 (0.5~0.8)
   - 빠른 피봇 감지 필요
   - 변동성 증가

6. **장 마감 (15:20-15:30)**: 매우 낮은 비율 (0.3~0.5)
   - 최고 민감도
   - 변동성 최고

**데이터 소스별 차이:**

- **KOSPI**: 비율 낮게 (변동성 상대적으로 작음)
- **KP200**: 비율 높게 (변동성 상대적으로 큼)
  - 전체 비율에 1.2~1.5배 곱하여 적용

---

## 6. 백테스팅 방법론

### 6.1 데이터 준비

**데이터 기간:**
- 최소 3개월 데이터 (시장 다양성 확보)
- 권장: 6개월 ~ 1년 데이터
- 장르별, 요일별, 시간대별 패턴 포함

**데이터 소스:**
- KOSPI 지수 분봉 데이터
- KP200 선물 분봉 데이터
- 각각 별도로 테스트

**데이터 분할 (⚠️ 시계열 데이터 주의사항):**

**⚠️ 중요: 시계열 데이터는 무작위 분할 금지**
- 무작위 분할 시 미래 데이터 누출(look-ahead bias) 발생
- 반드시 시간 순서 기준으로 분할해야 함

**올바른 분할 방법:**
- 학습 데이터: 1~7개월 (70%)
- 검증 데이터: 8~9개월 (20%)
- 테스트 데이터: 10개월~ (10%)

**Walk-forward validation 권장:**
- 시간 순서대로 슬라이딩 윈도우 방식으로 검증
- 과거 데이터로 학습 → 미래 데이터로 검증 반복

### 6.2 성능 평가 지표

**피봇 품질 지표:**

1. **피봇 갯수 분포**
   - 시간대별 피봇 갯수 분석
   - 목표: 장 시작/마감에 피봇 많음, 점심시간에 피봇 적음
   - 지표: 시간대별 피봇 밀도 (피봇/시간)

2. **피봇 정확도**
   - 확정 피봇 비율 (확정/후보)
   - 목표: 70% 이상 확정율
   - 지표: 확정율 = 확정 피봇 수 / 전체 후보 수

### 6.3 튜닝 프로세스

**Step 1: 기준선 설정**

```json
{
  "session_min_wave_atr_ratio_table": []
}
```
- 빈 테이블로 기존 `min_wave_atr_ratio` 단일값 사용
- 기준 성능 측정

**Step 2: 균형 전략 적용**

```json
{
  "session_min_wave_atr_ratio_table": [
    ["09:00", "09:30", 0.8],
    ["09:30", "10:30", 1.2],
    ["10:30", "13:00", 1.8],
    ["13:00", "14:30", 1.2],
    ["14:30", "15:20", 0.8],
    ["15:20", "15:30", 0.5]
  ]
}
```
- 문서의 균형 전략 적용
- 성능 변화 측정

**Step 3: 시간대별 튜닝**

```python
# 튜닝 알고리즘
for time_slot in time_slots:
    for ratio in test_ratios:  # 예: [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
        # 해당 시간대만 비율 변경
        table = copy.deepcopy(base_table)
        table[time_slot] = ratio
        
        # 백테스트 실행
        result = run_backtest(table)
        
        # 성능 평가
        score = evaluate_performance(result)
        
        # 최적 비율 저장
        if score > best_score[time_slot]:
            best_ratio[time_slot] = ratio
            best_score[time_slot] = score
```

**Step 4: 반복 최적화**

- 전체 테이블을 한 번에 튜닝 (시간대별 상호작용 고려)
- Grid Search 또는 Bayesian Optimization 사용
- 과적합 방지를 위해 교차 검증

### 6.4 비율 테이블 최적화 방법

**방법 1: Grid Search**

```python
# 각 시간대별 후보 비율
candidate_ratios = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

# 모든 조합 탐색
for ratio1 in candidate_ratios:
    for ratio2 in candidate_ratios:
        for ratio3 in candidate_ratios:
            # ... (시간대 수만큼 반복)
            table = [
                ["09:00", "09:30", ratio1],
                ["09:30", "10:30", ratio2],
                ["10:30", "13:00", ratio3],
                # ...
            ]
            result = run_backtest(table)
```
- 장점: 최적 해 보장
- 단점: 계산 비용 높음 (7^6 = 117,649 조합)

**⚠️ 실용적 경고:**
- 백테스트 1회 = 약 10초 가정 시, 117,649회 = 약 326시간 (13.6일)
- 병렬 처리 없이는 현실적으로 불가
- Bayesian Optimization 우선 권장

**방법 2: 순차적 튜닝**

```python
# 시간대별 순차적으로 튜닝
for i, time_slot in enumerate(time_slots):
    best_ratio = 1.0
    best_score = 0
    
    for ratio in candidate_ratios:
        table[i] = ratio
        result = run_backtest(table)
        score = evaluate_performance(result)
        
        if score > best_score:
            best_ratio = ratio
            best_score = score
    
    table[i] = best_ratio
```
- 장점: 계산 비용 낮음
- 단점: 전역 최적 해 보장 안 됨

**방법 3: Bayesian Optimization**

```python
from skopt import gp_minimize

def objective(params):
    table = [
        ["09:00", "09:30", params[0]],
        ["09:30", "10:30", params[1]],
        ["10:30", "13:00", params[2]],
        ["13:00", "14:30", params[3]],
        ["14:30", "15:20", params[4]],
        ["15:20", "15:30", params[5]]
    ]
    result = run_backtest(table)
    score = -evaluate_performance(result)  # 최소화 문제
    return score

# 탐색 공간
space = [(0.5, 2.0)] * 6  # 6개 시간대, 각각 0.5~2.0

# 최적화
result = gp_minimize(objective, space, n_calls=50)
```
- 장점: 효율적인 탐색
- 단점: 라이브러리 필요

### 6.5 A/B 테스트 방법

**테스트 설계:**

```python
# A: 기존 설정 (단일 비율)
config_A = {
    "min_wave_atr_ratio": 1.5,
    "session_min_wave_atr_ratio_table": []
}

# B: 시간대별 동적 비율
config_B = {
    "min_wave_atr_ratio": 1.5,
    "session_min_wave_atr_ratio_table": [
        ["09:00", "09:30", 0.8],
        ["09:30", "10:30", 1.2],
        ["10:30", "13:00", 1.8],
        ["13:00", "14:30", 1.2],
        ["14:30", "15:20", 0.8],
        ["15:20", "15:30", 0.5]
    ]
}

# 동일 데이터로 테스트
result_A = run_backtest(config_A, test_data)
result_B = run_backtest(config_B, test_data)

# 통계적 유의성 검정
from scipy import stats
t_stat, p_value = stats.ttest_ind(result_A['scores'], result_B['scores'])
```

**비교 지표:**

| 지표 | Config A | Config B | 개선 |
|------|----------|----------|------|
| 전체 피봇 수 | 120 | 115 | -4.2% |
| 확정율 | 65% | 72% | +10.8% |
| 방향 정확도 | 58% | 64% | +10.3% |
| 취소율 | 35% | 28% | -20.0% |
| 시간대별 편차 | 높음 | 낮음 | 개선 |

**참고:** 위 수치는 예시이며, 실제 테스트 시에는 과거 데이터로 백테스팅을 통해 실제 수치를 측정해야 합니다.

---

## 7. 강화학습 기반 튜닝 (고급)

### 7.1 개요

강화학습(RL)을 사용하여 시간대별 동적 ATR 필터링 파라미터를 자동으로 최적화합니다.

### 7.2 환경 설계

#### State Space (상태 공간)

**시장 상태:**
- 현재 시간 (0-1440분, 장 시작 기준)
- 현재 가격 (정규화된 가격)
- 최근 N분봉 OHLCV (N=60)
- 현재 ATR 값 (정규화)
- 거래량 (정규화)
- 시간대 구분 (전장/후장/점심시간)

**피봇 상태:**
- 최근 확정 피봇 갯수
- 현재 후보 피봇 갯수
- 피봇 확정율 (최근 M분)

**파라미터 상태:**
- 현재 파라미터 값 (min_wave_atr_ratio, cluster_atr_ratio)
- 파라미터 변경 이력

#### Action Space (행동 공간)

**연속 행동 공간:**
- `min_wave_atr_ratio`: [0.5, 5.0] 범위 내 연속값
- `cluster_atr_ratio`: [0.5, 5.0] 범위 내 연속값

**이산 행동 공간 (대안):**
- 파라미터 증가/감소/유지
- 증감 폭: ±0.1, ±0.5, ±1.0

#### Reward Function (보상 함수)

**단계 보상 (Step Reward):**

```python
def calculate_reward(state, next_state):
    # 1. 피봇 품질 보상
    pivot_quality_reward = calculate_pivot_quality_reward(next_state)

    # 2. 시간대별 적합성 보상
    time_suitability_reward = calculate_time_suitability_reward(next_state)

    # 3. 파라미터 안정성 보너스/패널티
    stability_reward = calculate_stability_reward(state, next_state)

    # 4. 과도한 변경 패널티
    change_penalty = calculate_change_penalty(state, next_state)

    total_reward = (
        pivot_quality_reward * 0.5 +
        time_suitability_reward * 0.3 +
        stability_reward * 0.15 -
        change_penalty * 0.05
    )

    return total_reward
```

**참고:** 가중치(0.5, 0.3, 0.15, 0.05)는 초기값이며 튜닝 필요

**서브 함수 스펙:**

```python
def calculate_pivot_quality_reward(state: dict) -> float:
    """
    피봇 품질 보상 계산

    Args:
        state: {'confirmation_rate': float, 'confirmed_pivots': int, ...}

    Returns:
        float: 보상 값 (범위: -1.0 ~ 1.0)
    """
    confirmation_rate = state['confirmation_rate']
    pivot_count = state['confirmed_pivots']

    # 확정율 보상
    if confirmation_rate >= 0.7:
        quality_score = 1.0
    elif confirmation_rate >= 0.5:
        quality_score = 0.5
    else:
        quality_score = -0.5

    # 피봇 갯수 보상 (너무 많거나 적으면 패널티)
    if 3 <= pivot_count <= 8:
        count_score = 0.5
    else:
        count_score = -0.3

    return (quality_score + count_score) / 2


def calculate_time_suitability_reward(state: dict) -> float:
    """
    시간대별 적합성 보상 계산

    Args:
        state: {'time': int, 'confirmed_pivots': int, ...}

    Returns:
        float: 보상 값 (범위: -1.0 ~ 1.0)
    """
    time = state['time']  # 분 단위 (0-1440)
    pivot_count = state['confirmed_pivots']

    # 장 시작/마감: 피봇 많으면 보상
    if (0 <= time < 60) or (480 <= time < 540):
        if pivot_count >= 5:
            return 1.0
        else:
            return -0.5

    # 점심시간: 피봇 적으면 보상
    elif 180 <= time < 240:
        if pivot_count <= 3:
            return 1.0
        else:
            return -0.5

    # 그외 시간대: 중간값 선호
    else:
        if 3 <= pivot_count <= 6:
            return 0.5
        else:
            return -0.3


def calculate_stability_reward(state: dict, next_state: dict) -> float:
    """
    파라미터 안정성 보상 계산

    Args:
        state: {'parameter_change_count': int, ...}
        next_state: {'parameter_change_count': int, ...}

    Returns:
        float: 보상 값 (범위: -1.0 ~ 1.0)
    """
    change_count = next_state['parameter_change_count']
    if change_count > 10:  # 하루 10회 이상 변경
        return -0.5
    else:
        return 0.2


def calculate_change_penalty(state: dict, next_state: dict) -> float:
    """
    과도한 변경 패널티 계산

    Args:
        state: {'min_wave_atr_ratio': float, 'cluster_atr_ratio': float, ...}
        next_state: {'min_wave_atr_ratio': float, 'cluster_atr_ratio': float, ...}

    Returns:
        float: 패널티 값 (범위: 0.0 ~ 1.0)
    """
    delta_ratio = abs(next_state['min_wave_atr_ratio'] - state['min_wave_atr_ratio'])
    delta_cluster = abs(next_state['cluster_atr_ratio'] - state['cluster_atr_ratio'])

    # 변경 폭이 크면 패널티
    if delta_ratio > 0.5 or delta_cluster > 0.5:
        return 1.0
    elif delta_ratio > 0.2 or delta_cluster > 0.2:
        return 0.5
    else:
        return 0.0
```

**보상 스케일 정규화:**
- 각 서브 함수의 출력을 [-1.0, 1.0] 범위로 정규화
- 가중치 합계 = 1.0 (0.5 + 0.3 + 0.15 + 0.05 = 1.0)
- 최종 보상 범위: [-1.0, 1.0]

### 7.3 알고리즘 선택

#### PPO (Proximal Policy Optimization)

**장점:**
- 안정적이고 효율적인 policy gradient
- 연속/이산 행동 공간 모두 지원
- 샘플 효율성 높음

**구현:**
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = PivotTuningEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

#### SAC (Soft Actor-Critic)

**장점:**
- 연속 행동 공간에 최적화
- 샘플 효율성 높음
- 안정적인 학습

**구현:**
```python
from stable_baselines3 import SAC

env = PivotTuningEnv()
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### 7.4 실제 시스템 통합

#### 온라인 튜닝

```python
class OnlineRLTuner:
    """온라인 강화학습 튜너."""
    
    def __init__(self, model_path: str):
        self.model = PPO.load(model_path)
        self.env = PivotTuningEnv()
        
    def update_parameters(self, current_state: dict) -> dict:
        """현재 상태에 따라 파라미터 업데이트."""
        obs = self._state_to_observation(current_state)
        action, _ = self.model.predict(obs, deterministic=True)
        
        # 파라미터 업데이트
        new_params = self._apply_action(current_state, action)
        
        return new_params
```

#### 오프라인 튜닝

```python
# 매일 장마감 후 학습
def daily_retraining():
    # 당일 데이터 로드
    data = load_today_data()
    
    # 환경 업데이트
    env.update_data(data)
    
    # 모델 재학습
    model.learn(total_timesteps=10000)
    
    # 모델 저장
    model.save(f"models/rl_pivot_tuner_{date.today()}")
```

### 7.5 주의사항

1. **과적합 방지**
   - 충분한 학습 데이터 확보 (최소 3개월)
   - 검증 데이터 분리
   - 정규화 기법 적용

2. **안정성 확보**
   - 파라미터 변경 범위 제한
   - 너무 빈번한 변경 방지
   - 안전 가드레일 설정

3. **실제 시장 적용 전 테스트**
   - 백테스팅 결과 검증
   - 시뮬레이션 테스트
   - 점진적 롤아웃

### 7.6 구현 우선순위

1. **단계 1:** 기본 환경 구현 (State/Action/Reward)
2. **단계 2:** 단순 알고리즘 테스트 (Random, DQN)
3. **단계 3:** PPO/SAC 구현 및 튜닝
4. **단계 4:** 백테스팅 데이터로 학습
5. **단계 5:** 온라인 통합 테스트
6. **단계 6:** 실제 시장 적용

---

## 8. ATR 모니터링

### 8.1 기능

ATRMonitor 클래스는 ATR 값의 변화를 실시간으로 추적하여 시장 변동성 변화를 감지합니다.

1. **ATR 변화율 계산**: 이전 ATR 대비 현재 ATR의 변화율(%) 계산
2. **추세 분석**: 
   - `rising`: ATR이 5% 이상 상승
   - `falling`: ATR이 5% 이상 하락
   - `stable`: 변화가 5% 미만
3. **급격 변동 감지**: 30% 이상 변동 시 경고 로그 출력 (WARNING 레벨) 및 텔레그램 송출
4. **이동평균 계산**: 최근 14봉 ATR 이동평균 계산

### 8.2 로그 예시

```
[ATR-MONITOR] ATR=5.16, change=+2.3%, trend=rising, spike=False, MA=5.02
[ATR-MONITOR] ATR 급증: 5.16 → 7.83 (+51.7%)
[ATR-MONITOR] ATR=7.83, change=+51.7%, trend=rising, spike=True, MA=5.45
```

### 8.3 텔레그램 알림

급격 변동 감지 시 (30% 이상 변동) 텔레그램으로 알림이 송출됩니다:

```
🚨 ATR 급증 알림

이전 ATR: 5.16
현재 ATR: 7.83
변화율: +51.7%
이동평균: 5.45
```

- KOSPI 지수와 KP200 선물 모두에서 작동
- `ebestapi/live.py`의 `_setup_zigzag_candidate_telegram_hooks()` 함수에서 콜백 설정

### 8.4 ZigZagState 필드

- `atr`: 현재 ATR 값
- `atr_change_pct`: 이전 ATR 대비 변화율 (%)
- `atr_trend`: 추세 ('rising', 'falling', 'stable')
- `atr_spike_detected`: 급격 변동 감지 여부
- `atr_ma`: ATR 이동평균

### 8.5 활용 방법

ATR 급격 변동 시 피봇 감지 민감도를 동적으로 조정하는 등 변동성 변화에 대응하는 로직에 활용할 수 있습니다.

### 8.6 급격 변동 시 동적 비율 조정

ATR이 급격히 변동할 때 노이즈 필터링이 과도하거나 부족해지는 문제를 해결하기 위해 동적으로 비율을 조정합니다:

- **ATR 급증 시 (30% 이상 상승)**: `min_wave_atr_ratio`를 70%로 낮춰 과도한 필터링 방지
  - 예: 기본 1.5 → 조정 후 1.05 (최소 0.5 보장)
- **ATR 급락 시 (30% 이상 하락)**: `min_wave_atr_ratio`를 130%로 높여 부족한 필터링 보완
  - 예: 기본 1.5 → 조정 후 1.95

#### 파라미터 합성 우선순위

세 가지 값이 결합될 때 다음 우선순위로 최종 비율을 계산합니다:

1. **시간대 테이블 값** (base): 현재 시간에 해당하는 `session_min_wave_atr_ratio_table` 값
2. **ATR 급변 배율 적용**: ATR 급증 시 ×0.7, 급락 시 ×1.3
3. **절대 하한/상한 클램프**: 최종 값을 [0.5, 5.0] 범위로 제한

**예시:**
- 시간대 테이블 값: 1.5
- ATR 급증 시: 1.5 × 0.7 = 1.05
- 클램프 후: max(1.05, 0.5) = 1.05

**폴백 동작:**
1. 현재 시간이 테이블에 포함된 구간에 있으면 해당 값 사용
2. 매칭되는 구간이 없으면 상위 `min_wave_atr_ratio` 사용
3. 상위 값도 없으면 전역 기본값 0.5 사용

**로그 예시:**
```
[ATR-MONITOR] ATR 급증으로 비율 조정: 1.50 → 1.05
[ATR-MONITOR] ATR 급락으로 비율 조정: 1.50 → 1.95
```

---

## 9. 실제 적용 사례

### 9.1 KP200 피봇 감소 사례 (2026-05-01)

**문제**: ATR 필터링이 활성화되었으나 피봇 갯수 차이가 없음 (KOSPI 5개, KP200 7개)

**해결 과정**:
1. 초기 설정: `min_wave_atr_ratio` = 0.5 (KP200)
2. 1차 조정: `min_wave_atr_ratio` = 2.0 → 피봇 7개 유지 (필터링 미약)
3. 2차 조정: `min_wave_atr_ratio` = 3.0 → 피봇 7개 → 3개 감소 (과도한 필터링)
4. 최종 조정: `min_wave_atr_ratio` = 2.7 → 적절한 균형점

**최종 설정**:
```json
"adaptive_indicator": {
  "zigzag": {
    "use_atr_based_filtering": true,
    "min_wave_atr_ratio": 0.5,
    "cluster_atr_ratio": 0.5
  },
  "futures_zigzag": {
    "min_wave_atr_ratio": 2.7,      // 상위 0.5 대신 2.7 사용
    "cluster_atr_ratio": 1.0
  }
}
```

**결과**: KP200 피봇 7개 → 3개 감소 (ATR 필터링 효과적 작동)

### 9.2 로그 확인

ATR 필터가 작동하는지 확인하려면 다음 로그를 확인:

```
[ATR-FILTER] use_atr_filter=True, _atr_values=26, close=1017.25
[ATR-FILTER] ATR=1.77, ratio=2.7, min=4.78
[ATR-FILTER] actual_wave=3.30 vs min=4.78
[ATR-FILTER] ATR 필터 차단 (actual_wave < min)
```

- `ratio`: 적용된 `min_wave_atr_ratio`
- `min`: ATR * ratio로 계산된 최소 파동 크기
- `actual_wave`: 실제 파동 크기
- `ATR 필터 통과`: actual_wave >= min일 때
- `ATR 필터 차단`: actual_wave < min일 때

---

## 10. 주의사항

1. **과도한 민감도**: 파라미터를 너무 낮게 설정하면 노이즈가 많은 피봇이 생성될 수 있음
2. **데이터 소스별 차이**: KOSPI와 Futures(KP200)는 변동성 패턴이 다르므로 별도 설정 필요
   - KP200은 KOSPI보다 변동성이 크므로 더 높은 `min_wave_atr_ratio` 적용
   - KP200은 08:45부터 선거래 시작, KOSPI는 09:00부터 장 개시
   - `futures_zigzag` 섹션에서 별도 `session_min_wave_atr_ratio_table` 설정 권장
3. **시간대별 변동성**: 장 시작/종료 시간대는 변동성이 다를 수 있음
4. **백테스팅**: 파라미터 변경 후 반드시 과거 데이터로 백테스팅 권장
5. **ATR 급격 변동**: ATR이 급격히 변동할 때는 노이즈 필터링이 과도/부족할 수 있으므로 주의 필요
   - ATRMonitor 클래스가 ATR 변화를 추적하고 30% 이상 변동 시 경고 로그 출력

---

## 11. 향후 보완 사항

### 11.1 완료된 작업

1. **피봇 마커 위치 정확성 개선** (2026-05-01)
   - 문제: 피봇 마커가 봉의 실제 고가/저가가 아닌 곳에 표시됨
   - 원인: `swing_price`가 확정 시점의 가격이 아닌 다른 값으로 설정됨
   - 해결: `AdaptiveZigZag` 지표에서 확정 시점에 봉의 실제 고가/저가를 `swing_price`로 설정하도록 수정
   - 파일: `indicators/adaptive_zigzag.py` (라인 911-968, 1024-1105)

2. **주변 봉 검증 로직 제거** (2026-05-01)
   - 문제: 주변 봉 검증(`is_valid_H/L`)이 ZigZag 알고리즘 특성을 고려하지 않아 실패
   - 원인: ZigZag는 "주변 극값"이 아니라 "방향 전환 지점"을 식별하며, 확정 후 더 높은/낮은 값이 나올 수 있음
   - 해결: 주변 봉 검증 로직 제거 (ZigZag 특성에 맞게 동작)
   - 파일: `gui/chart_viewer.py` (라인 441-461 제거)

3. **피봇 마커 색상 및 스타일 개선** (2026-05-01)
   - 변경 내용:
     - 확정 피봇: 주황색 → 마젠타(`#FF00FF`)
     - 피봇 후보: 노란색 → 시안(`#00FFFF`)
     - 후보 마커 스타일: 다이아몬드 → 고가/저가별 삼각형 스타일("v"/"^")
   - 파일: `gui/chart_viewer.py` (라인 583-585, 1257-1269)

4. **디버깅 로그 출력 제거** (2026-05-01)
   - 제거된 로그:
     - `[PIVOT-MARKER]`: 피봇 마커 관련 로그
     - `[ATR-MONITOR]`: ATR 모니터링 로그
     - `[ATR-FILTER]`: ATR 필터링 로그
   - 파일: `gui/chart_viewer.py`, `indicators/adaptive_zigzag.py`

5. **피봇 후보 깜빡임 애니메이션** (2026-05-02)
   - 기능: 후보 마커 500ms 주기로 표시/숨김 반복
   - 구현:
     - QTimer를 사용하여 주기적 토글
     - setVisible() 메서드로 표시/숨김 제어
     - 후보 마커 목록 관리
   - 파일: `gui/chart_viewer.py` (라인 634-641, 1110-1119, 1242-1243, 1286-1295, 1028-1030)

6. **ZigZag 특성에 맞는 피봇 검증 로직** (2026-05-02)
   - 구현된 검증 로직:
     - 확정 시점 기준 극값 검증 (±5봉 내 극값 확인)
     - 방향 전환 유효성 검증 (10봉 후 방향성 60% 기준)
     - 파동 크기 검증 (ATR 기준 min_wave_atr_ratio)
   - 검증 결과: 딕셔너리로 반환 (extreme_valid, direction_valid, wave_size_valid)
   - 로그 출력: 주석 처리 (필요시 활성화)
   - 에러 처리: 검증 로직 에러 시 예외 처리로 무시
   - 파일: `indicators/adaptive_zigzag.py` (라인 2091-2250)

7. **파라미터 실시간 조정 기능** (2026-05-02)
   - 구현된 기능:
     - 파라미터 조정 다이얼로그 (ParameterDialog)
     - ATR 기반 필터링 파라미터 조정 (use_atr_based_filtering, min_wave_atr_ratio, cluster_atr_ratio)
     - ZigZag 기본 파라미터 조정 (atr_multiplier, atr_period, confirmation_bars, freeze_on_confirm)
     - 클러스터링 파라미터 조정 (cluster_tolerance_pct)
     - config.json 자동 저장
     - 기본값으로 초기화 기능
   - UI: 차트 뷰어 컨트롤 바에 "⚙ 파라미터" 버튼 추가
   - 파일: `gui/parameter_dialog.py` (새 파일), `gui/chart_viewer.py` (라인 1829-1833, 2009-2024)

8. **시간대별 동적 ATR 필터링** (2026-05-02)
   - 현재 상태: 모든 Phase 구현 완료
   - 완료된 작업:
     - Phase 1: `AdaptiveZigZagConfig`에 `session_min_wave_atr_ratio_table` 추가
     - Phase 2: `_get_time_based_atr_ratio()` 함수 구현
     - Phase 3: `_apply_atr_filter()` 메서드에 통합
     - Phase 4: 백테스팅 및 튜닝 방법론 설계

9. **백테스팅 데이터 자동 저장** (2026-05-02)
   - 장마감 시 자동 데이터 저장
   - KOSPI 지수 1분봉 데이터 저장
   - KP200 선물 1분봉 데이터 저장
   - 데이터 저장 폴더 구조: data/backtesting/{kospi|futures}/{YYYY}/{YYYY-MM-DD}_{source}_1m.csv
   - 중복 저장 방지 (하루에 한 번만 저장)
   - config.json 설정: backtest_data_saver 섹션
   - 파일: `data/backtest_data_saver.py` (새 파일), `data/tick_processor.py` (라인 105-109, 876-880, 1843-1904)

10. **피봇 정보 Crosshair 표시** (2026-05-02)
    - 기능: 십자선이 피봇 근처에 있을 때 피봇 정보를 별도 패널에 표시
    - 구현:
      - QLabel로 피봇 정보 패널 생성
      - 녹색 텍스트, 검은 배경의 스타일 적용
      - crosshair 이벤트 연결
      - 인덱스를 시간으로 변환하여 표시
    - 파일: `gui/chart_viewer.py` (라인 1827, 2000-2027, 2039-2121)

### 11.2 구현 필요 사항

1. **피봇 후보 시각화 기능 확장** (우선순위: 중간)
   - 현재 상태: 기본 구현 완료 (시안색 마커)
   - 완료된 작업 (2026-05-02):
     - 후보 마커 깜빡임 애니메이션 (500ms 주기 표시/숨김)
   - 추가 가능 기능:
     - 후보 마커 크기 조절 (확정보다 작게)
     - 후보 마커 투명도 조절
     - 후보 마커 호버 시 추가 정보 표시

### 11.3 추가 고려 사항

1. **데이터 소스별 파라미터 최적화**
   - KOSPI와 KP200의 변동성 패턴 차이 반영
   - 과거 데이터 백테스팅을 통한 파라미터 튜닝
   - 시장 상황(불장/하락장)별 파라미터 조정 고려

2. **성능 최적화**
   - 대량 데이터 처리 시 성능 저하 방지
   - 피봇 마커 렌더링 최적화
   - ATR 계산 최적화

3. **사용자 인터페이스 개선** (우선순위: 높음)
   - 파라미터 실시간 조정 기능 ✓ 완료 (2026-05-02)
   - 피봇 감지 결과 시각화 개선
     - 피봇 정보 툴팁 (마우스 호버 시 상세 정보) ✓ 완료 (2026-05-02)
     - 피봇 통계 패널 (피봇 갯수, 확정율 등)
     - 피봇 필터링 기능 (특정 피봇만 표시)
     - 피봇 시간대별 분석 차트

---

## 12. 문서 변경 이력

| 날짜 | 변경 내용 | 작성자 |
|------|-----------|--------|
| 2026-05-01 | 최초 작성, KP200 min_wave_atr_ratio=2.7 확정 | - |
| 2026-05-02 | 시간대별 동적 필터링 Phase 완료, 피봇 정보 Crosshair 표시 추가 | - |
| 2026-05-02 | 문서 구조 재구성 (파라미터 최적화 목적) | - |
| 2026-05-02 | 리뷰 반영: 파라미터 합성 우선순위 명시, 시계열 분할 경고 추가, KP200 시간대 비율 일관성 수정, 폴백 동작 명시, cluster_atr_ratio 기본값 명확화, Grid Search 실용 경고 추가, RL 서브 함수 스펙 추가, 파라미터 상호작용 섹션 추가, 의사결정 트리 추가, A/B 테스트 데이터 출처 명시 | - |
| 2026-05-02 | 코드 수정: 파라미터 합성 로직을 문서와 일치하도록 수정 (시간대 테이블 → ATR 급변 배율 → 클램프), 폴백 동작 수정 (1.0 → 상위 min_wave_atr_ratio), KP200 config.json 수정 (09:00-09:30 비율 1.5 → 1.2) | - |
| 2026-05-02 | 머신러닝 기법 추천 섹션 추가 (Bayesian Optimization, LightGBM, Optuna, Isolation Forest, HMM) | - |
| 2026-05-03 | Optuna 튜너 구현 및 사용법 추가 (prediction/zigzag_backtester.py, scripts/optuna_zigzag_tuner.py) | - |

---

## 13. 머신러닝 기법 추천 (ZigZag 파라미터 최적화)

ZigZag 파라미터 최적화 목적에 맞게, 강화학습 제외 기법들을 실용성 기준으로 정리합니다.

### 13.1 1순위: Bayesian Optimization (즉시 적용 가능)

문서에도 언급되어 있지만 가장 먼저 도입할 기법입니다.

**왜 적합한가:**
- 백테스트 1회 실행 비용이 높을 때 최적 (문서 기준 ~10초/회)
- Grid Search 117,649회 → 50~100회로 동등 수준 탐색
- `min_wave_atr_ratio`, `cluster_atr_ratio` 등 연속 파라미터에 직접 적용

```python
from skopt import gp_minimize
from skopt.space import Real

def objective(params):
    config = {
        "min_wave_atr_ratio": params[0],
        "cluster_atr_ratio":  params[1],
        "min_wave_pct":       params[2],
    }
    result = run_backtest(config, data)
    # 확정율 + 피봇 수 균형 점수
    score = result["confirmation_rate"] * 0.6 - abs(result["pivot_count"] - 5) * 0.1
    return -score  # 최소화 문제

space = [
    Real(0.5, 5.0, name="min_wave_atr_ratio"),
    Real(0.5, 3.0, name="cluster_atr_ratio"),
    Real(0.1, 0.4, name="min_wave_pct"),
]

result = gp_minimize(objective, space, n_calls=60, random_state=42)
```

**주의:** `n_calls=60` 기준 약 10분 소요 (백테스트 10초 가정).

### 13.2 2순위: XGBoost / LightGBM (피봇 확정 예측)

파라미터 최적화가 아닌 "이 후보 피봇이 확정될 것인가"를 예측하는 분류 모델입니다.

**입력 피처 설계:**

| 피처 | 설명 |
|------|------|
| atr_ratio | 파동 크기 / ATR |
| wave_bars | 파동 봉 수 |
| time_slot | 시간대 인코딩 (0~5) |
| prev_pivot_distance | 직전 피봇과의 거리 |
| volume_ratio | 파동 구간 평균 거래량 / 전체 평균 |
| body_ratio | 캔들 몸통 비율 |
| atr_trend | ATR 추세 (rising/falling/stable → 0/1/2) |

**학습 데이터 구성:**

```python
# 과거 백테스팅 데이터에서 추출
# label: 후보 피봇이 최종 확정되면 1, 취소되면 0
X = features_df  # 위 피처들
y = (swings_df["confirmed"] == True).astype(int)

# 시계열 분할 (무작위 분할 금지)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
model.fit(X_train, y_train)
```

**실시간 활용:**

```python
# 후보 피봇 등록 시 확정 확률 예측
prob = model.predict_proba(candidate_features)[0][1]
# prob > 0.7이면 강한 피봇 후보로 표시
```

현재 코드의 `PivotProbabilityCalculator`를 이 모델로 교체하면 됩니다.

### 13.3 3순위: Optuna (파라미터 탐색 프레임워크)

Bayesian Optimization의 실용적 대안으로, 시간대별 테이블 6개 파라미터를 동시 최적화할 때 특히 유용합니다.

```python
import optuna

def objective(trial):
    # 시간대별 비율 탐색
    table = [
        ["09:00", "09:30", trial.suggest_float("t1", 0.5, 2.0)],
        ["09:30", "10:30", trial.suggest_float("t2", 0.8, 2.5)],
        ["10:30", "13:00", trial.suggest_float("t3", 1.0, 3.0)],
        ["13:00", "14:30", trial.suggest_float("t4", 0.8, 2.5)],
        ["14:30", "15:20", trial.suggest_float("t5", 0.5, 2.0)],
        ["15:20", "15:30", trial.suggest_float("t6", 0.3, 1.5)],
    ]
    result = run_backtest({"session_table": table})
    return result["confirmation_rate"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=4)  # 병렬 실행 가능
```

**Bayesian Optimization 대비 장점:**
- 조기 종료(pruning) 지원 → 나쁜 파라미터 조합 빠르게 제거
- 병렬 실행 (`n_jobs=4`)으로 실제 소요 시간 1/4 단축
- 탐색 이력 시각화 내장

### 13.4 4순위: Isolation Forest (이상 피봇 필터링)

노이즈 피봇을 사전에 걸러내는 비지도 학습 기법입니다.

```python
from sklearn.ensemble import IsolationForest

# 정상 피봇의 피처 분포 학습
iso = IsolationForest(contamination=0.15, random_state=42)
iso.fit(normal_pivot_features)

# 새 후보 피봇 평가
score = iso.decision_function(candidate_features)
# score < -0.1이면 이상 피봇으로 판단 → 필터링
```

현재 ATR 필터와 조합:
- ATR 필터: 파동 크기 기준 1차 필터
- Isolation Forest: 다변량 피처 기준 2차 필터

### 13.5 5순위: Hidden Markov Model (시장 레짐 감지)

시간대별 고정 테이블 대신 시장 상태(변동성 레짐)를 자동 감지하여 파라미터를 동적으로 전환합니다.

```python
from hmmlearn import hmm

# 상태: [저변동성, 중변동성, 고변동성] 3가지
model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)

# 입력: ATR, 거래량, 가격 변동률
X = np.column_stack([atr_series, volume_series, return_series])
model.fit(X)

# 현재 레짐 예측
current_regime = model.predict(current_features[-1:])
# regime 0 → min_wave_atr_ratio = 0.8 (고변동)
# regime 1 → min_wave_atr_ratio = 1.5 (중변동)
# regime 2 → min_wave_atr_ratio = 2.7 (저변동)
```

### 13.6 도입 우선순위 요약

```
즉시 적용 (1~2주)
  └→ Bayesian Optimization (Optuna) — 시간대 테이블 자동 튜닝

단기 (1개월)
  └→ LightGBM 분류 — PivotProbabilityCalculator 대체

중기 (2~3개월)
  └→ Isolation Forest — 노이즈 피봇 2차 필터
  └→ HMM — 시간대 고정 테이블 → 동적 레짐 전환
```

현재 코드베이스 통합 난이도 기준으로는 **Optuna → LightGBM → Isolation Forest** 순서가 가장 현실적입니다.

### 13.7 Optuna 튜너 사용법

현재 코드베이스에 Optuna 기반 ZigZag 파라미터 튜너가 구현되어 있습니다.

**파일 구조:**
- `prediction/zigzag_backtester.py`: ZigZag 백테스트 실행기
- `scripts/optuna_zigzag_tuner.py`: Optuna 튜너 스크립트

**사용법:**

```bash
# 단순 버전 (min_wave_atr_ratio만)
python scripts/optuna_zigzag_tuner.py --data data/backtesting/futures/2026/2026-05-03_futures_1m.csv --mode simple --n-trials 30

# 고급 버전 (여러 파라미터)
python scripts/optuna_zigzag_tuner.py --data data/backtesting/futures/2026/2026-05-03_futures_1m.csv --mode advanced --n-trials 60

# 결과 저장
python scripts/optuna_zigzag_tuner.py --data data/backtesting/futures/2026/2026-05-03_futures_1m.csv --mode simple --n-trials 30 --output optuna_results.json
```

**데이터 준비:**
- `backtest_data_saver.py`를 사용하여 장마감 시 데이터 자동 저장
- 저장 위치: `data/backtesting/futures/{YYYY}/{YYYY-MM-DD}_futures_1m.csv`

**코드 예시:**

```python
from prediction.zigzag_backtester import ZigZagBacktester
from pathlib import Path

# 백테스터 초기화
backtester = ZigZagBacktester(data_path=Path("data/backtesting/futures/2026/2026-05-03_futures_1m.csv"))

# 단순 objective 함수 사용
import optuna
study = optuna.create_study(direction="maximize")
study.optimize(backtester.objective_simple, n_trials=30)

print(f"최적 파라미터: {study.best_params}")
```

**주의사항:**
- 데이터가 적을수록 결과 신뢰도가 낮음 (최소 1주일치 이상 권장)
- 실시간 데이터로도 테스트 가능하지만 신뢰도 낮음
- 시간대별 테이블 최적화는 아직 구현되지 않음

---

## 14. 참고 자료

- AdaptiveZigZag 인디케이터 구현: `indicators/adaptive_zigzag.py`
- 차트 뷰어 구현: `gui/chart_viewer.py`
- 파라미터 다이얼로그: `gui/parameter_dialog.py`
- 설정 파일: `config.json`
- ZigZag 백테스트 실행기: `prediction/zigzag_backtester.py`
- Optuna 튜너 스크립트: `scripts/optuna_zigzag_tuner.py`
- 백테스트 데이터 저장: `data/backtest_data_saver.py`
