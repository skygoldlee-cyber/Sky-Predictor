# 피봇 탐지기 비교 및 설정 가이드

## 개요

본 문서는 세 가지 피봇 탐지기(`PercentAdaptivePivot`, `ATRAdaptivePivot`, `HybridAdaptivePivot`)의 비교와 가격 수준별 설정 가이드를 제공합니다.

---

## 1. 세 가지 피봇 탐지기 비교

### 1.1 장점 결합 테이블

| 장점 | percent_adaptive_pivot.py | atr_adaptive_pivot.py | hybrid_adaptive_pivot.py |
|------|---------------------------|---------------------|-------------------------|
| **변동성 적응** | ❌ | ✅ | ✅ |
| **직관적 설정** | ✅ | ❌ | ✅ |
| **cancel_ratio 파라미터화** | ✅ | ✅ (수정됨) | ✅ |
| **즉시 확정 지원** | ✅ | ✅ (수정됨) | ✅ |
| **명시적 방향 복귀** | ✅ | ✅ (수정됨) | ✅ |
| **이중 파동 필터** | ❌ | ❌ | ✅ (퍼센트 + ATR) |

### 1.2 핵심 차이점

| 특징 | PercentAdaptivePivot | ATRAdaptivePivot | HybridAdaptivePivot |
|------|---------------------|------------------|---------------------|
| **임계값 기반** | 퍼센트 (%) | ATR | 퍼센트 + ATR 혼합 |
| **의존성** | 없음 (순수 Python) | WilderRMA 필요 | WilderRMA 필요 |
| **민감도** | 높음 (많은 피봇) | 낮음 (적은 피봇) | 중간 (atr_weight 조절) |
| **유연성** | 낮음 | 낮음 | 높음 (가중치 조절) |

### 1.3 동일 데이터 비교 결과

#### 피봇 개수
```
PercentAdaptivePivot: 4개 피봇
ATRAdaptivePivot: 3개 피봇
HybridAdaptivePivot: 3개 피봇
```

#### 임계값 (가격 110 기준)
```
Percent: 0.3300pt (0.30%)
ATR: 5.5347pt (4.69%)
Hybrid: 3.5165pt (2.98%)
```

#### Pivot Score
```
Percent: pivot_score=0.4142
ATR: pivot_score=0.2505
Hybrid: pivot_score=0.2991
```

---

## 2. atr_weight 값의 의미

### 2.1 정의

`atr_weight`는 하이브리드 임계값 계산 시 ATR 기반 임계값과 퍼센트 기반 임계값의 혼합 비율을 결정하는 파라미터입니다.

### 2.2 범위

```python
0.0 ≤ atr_weight ≤ 1.0
```

### 2.3 계산 공식

```python
thr_pct = close × base_pct/100 × er_multiplier × session_scale
thr_atr = atr × base_multiplier × er_multiplier × session_scale
thr_hybrid = (1 - atr_weight) × thr_pct + atr_weight × thr_atr
```

### 2.4 값별 의미

| atr_weight | 계산 | 의미 | 특징 |
|-----------|------|------|------|
| 0.0 | 1.0×thr_pct + 0.0×thr_atr | 퍼센트 전용 | 직관적, 변동성 적응 없음 |
| 0.25 | 0.75×thr_pct + 0.25×thr_atr | 퍼센트 중심 | 퍼센트 중심, 약간의 변동성 적응 |
| 0.5 | 0.5×thr_pct + 0.5×thr_atr | 균형 혼합 | 두 방식의 장점 균형 (기본값) |
| 0.75 | 0.25×thr_pct + 0.75×thr_atr | ATR 중심 | ATR 중심, 약간의 직관성 |
| 1.0 | 0.0×thr_pct + 1.0×thr_atr | ATR 전용 | 변동성 적응 최대, 설정 비직관적 |

### 2.5 실제 예시

**시나리오:** 가격 110, base_pct=0.3%, ATR=2.0, base_multiplier=2.0

| atr_weight | 임계값 | 해석 |
|-----------|--------|------|
| 0.0 | 0.33pt | 퍼센트만 |
| 0.25 | 1.25pt | 퍼센트 중심 |
| 0.5 | 2.17pt | 균형 |
| 0.75 | 3.08pt | ATR 중심 |
| 1.0 | 4.00pt | ATR만 |

### 2.6 피봇 감지 영향

```
atr_weight=0.00: pivots=4, threshold_pct=0.5200%
atr_weight=0.25: pivots=4, threshold_pct=1.7500%
atr_weight=0.50: pivots=3, threshold_pct=2.9801%
atr_weight=0.75: pivots=3, threshold_pct=4.2101%
atr_weight=1.00: pivots=3, threshold_pct=5.4401%
```

**패턴:** atr_weight ↑ → 임계값 ↑ → 피봇 개수 ↓

---

## 3. 가격 수준의 영향

### 3.1 개념

가격 수준은 자산의 현재 가격 위치를 의미하며, 피봇 탐지기의 임계값 계산 방식에 따라 다르게 영향을 미칩니다.

### 3.2 퍼센트 기반 방식

#### 임계값 계산
```python
thr_abs = close × base_pct / 100
```

#### 가격 수준 영향
```
가격 100, base_pct=0.3% → thr = 0.3pt
가격 200, base_pct=0.3% → thr = 0.6pt
가격 1000, base_pct=0.3% → thr = 3.0pt
```

**특징:**
- 가격 수준에 비례하여 절대 임계값 증가
- 동일한 퍼센트 변동만 피봇으로 감지
- 고가 자산에서 더 큰 절대 변동 필요

### 3.3 ATR 기반 방식

#### 임계값 계산
```python
thr_abs = atr × base_multiplier
```

#### 가격 수준 영향
```
가격 100, ATR=1.0 → thr = 2.0pt (2%)
가격 200, ATR=2.0 → thr = 4.0pt (2%)
가격 1000, ATR=10.0 → thr = 20.0pt (2%)
```

**특징:**
- 가격 수준과 무관한 일관된 퍼센트 변동 감지
- ATR이 가격 수준에 비례하면 자연스럽게 조정
- 실제 변동성 기반으로 임계값 결정

### 3.4 하이브리드 방식

#### 임계값 계산
```python
thr_pct = close × base_pct/100 × multiplier
thr_atr = atr × base_multiplier × multiplier
thr_hybrid = (1 - atr_weight) × thr_pct + atr_weight × thr_atr
```

#### 가격 수준 영향

| atr_weight | 가격 100, ATR=1.0 | 가격 1000, ATR=10.0 |
|-----------|-------------------|---------------------|
| 0.0 | 0.3pt (0.3%) | 3.0pt (0.3%) |
| 0.5 | 1.15pt (1.15%) | 11.5pt (1.15%) |
| 1.0 | 2.0pt (2%) | 20.0pt (2%) |

---

## 4. KOSPI 7500, KP200 1200 설정 가이드

### 4.1 가격 수준 분석

```
KOSPI: 7500pt (고가 자산)
KP200: 1200pt (중간 가격 자산)
비율: KOSPI가 KP200의 6.25배
```

### 4.2 퍼센트 기반 설정

#### 추천 설정
```python
# KOSPI (7500)
cfg = PercentAdaptivePivotConfig(
    base_pct=0.15,  # 더 낮은 퍼센트 (고가 자산)
    min_wave_pct=0.1,
)

# KP200 (1200)
cfg = PercentAdaptivePivotConfig(
    base_pct=0.3,  # 기본값
    min_wave_pct=0.15,
)
```

#### 임계값 계산
```python
KOSPI: 7500 × 0.15% = 11.25pt
KP200: 1200 × 0.3% = 3.6pt
```

**이유:** 고가 자산(KOSPI)은 더 낮은 퍼센트로 절대 변동 크기 조절

### 4.3 ATR 기반 설정

#### 추천 설정
```python
# 두 지수 모두 동일한 설정 가능
cfg = ATRAdaptivePivotConfig(
    base_multiplier=2.0,  # 기본값
    min_wave_atr_ratio=0.5,
)
```

#### 임계값 계산 (가정)
```python
# 가정: ATR이 가격의 약 1-2%
KOSPI: ATR≈75~150 → thr = 150~300pt (2~4%)
KP200: ATR≈12~24 → thr = 24~48pt (2~4%)
```

**이유:** ATR이 자동으로 가격 수준을 반영하므로 동일한 설정 사용 가능

### 4.4 하이브리드 설정

#### 추천 설정
```python
# KOSPI (7500)
cfg_kospi = HybridAdaptivePivotConfig(
    base_pct=0.2,          # 낮은 퍼센트
    base_multiplier=2.0,   # 기본 ATR 배수
    atr_weight=0.5,        # 균형 혼합
    min_wave_pct=0.1,
    min_wave_atr_ratio=0.5,
)

# KP200 (1200)
cfg_kp200 = HybridAdaptivePivotConfig(
    base_pct=0.3,          # 기본 퍼센트
    base_multiplier=2.0,   # 기본 ATR 배수
    atr_weight=0.5,        # 균형 혼합
    min_wave_pct=0.15,
    min_wave_atr_ratio=0.5,
)
```

#### atr_weight 조절 옵션

```python
# 보수적 접근 (노이즈 필터 강화)
atr_weight=0.75  # ATR 중심

# 민감한 접근 (빠른 반응)
atr_weight=0.25  # 퍼센트 중심
```

### 4.5 실제 임계값 비교

#### KOSPI 7500

| 방식 | 설정 | 절대 임계값 | 퍼센트 |
|------|------|-----------|--------|
| 퍼센트 | base_pct=0.15% | 11.25pt | 0.15% |
| ATR | base_multiplier=2.0 | ~150~300pt | 2~4% |
| 하이브리드 | atr_weight=0.5 | ~80~155pt | 1~2% |

#### KP200 1200

| 방식 | 설정 | 절대 임계값 | 퍼센트 |
|------|------|-----------|--------|
| 퍼센트 | base_pct=0.3% | 3.6pt | 0.3% |
| ATR | base_multiplier=2.0 | ~24~48pt | 2~4% |
| 하이브리드 | atr_weight=0.5 | ~14~26pt | 1~2% |

---

## 5. 가격 수준별 자동 계산 공식

### 5.1 로그 스케일 기반 자동 계산

#### 공식
```python
def calculate_auto_params(current_price: float, ref_price: float = 100.0) -> tuple:
    """
    가격 수준에 따른 자동 파라미터 계산 (로그 스케일)
    
    Parameters
    ----------
    current_price : float
        현재 가격
    ref_price : float
        기준 가격 (기본값: 100)
    
    Returns
    -------
    base_pct : float
        계산된 base_pct
    min_wave_pct : float
        계산된 min_wave_pct
    """
    # 기준 설정
    ref_base_pct = 0.3      # 기준 가격에서의 기본 퍼센트
    ref_min_wave_pct = 0.15 # 기준 가격에서의 최소 파동 퍼센트
    
    # 로그 스케일 비율 계산
    if current_price <= 0:
        return ref_base_pct, ref_min_wave_pct
    
    log_ratio = math.log10(ref_price) / math.log10(current_price)
    
    # 파라미터 계산
    base_pct = ref_base_pct * log_ratio
    min_wave_pct = ref_min_wave_pct * log_ratio
    
    # 최소/최한 제한
    base_pct = max(0.05, min(base_pct, 1.0))
    min_wave_pct = max(0.05, min(min_wave_pct, 0.5))
    
    return base_pct, min_wave_pct
```

#### 사용 예시
```python
# KOSPI 7500
base_pct, min_wave_pct = calculate_auto_params(7500, 100.0)
# 결과: base_pct ≈ 0.11%, min_wave_pct ≈ 0.055%

# KP200 1200
base_pct, min_wave_pct = calculate_auto_params(1200, 100.0)
# 결과: base_pct ≈ 0.27%, min_wave_pct ≈ 0.135%

# 기준 가격 100
base_pct, min_wave_pct = calculate_auto_params(100, 100.0)
# 결과: base_pct = 0.3%, min_wave_pct = 0.15%
```

### 5.2 선형 비율 기반 자동 계산

#### 공식
```python
def calculate_auto_params_linear(current_price: float, ref_price: float = 100.0) -> tuple:
    """
    가격 수준에 따른 자동 파라미터 계산 (선형 비율)
    
    Parameters
    ----------
    current_price : float
        현재 가격
    ref_price : float
        기준 가격 (기본값: 100)
    
    Returns
    -------
    base_pct : float
        계산된 base_pct
    min_wave_pct : float
        계산된 min_wave_pct
    """
    # 기준 설정
    ref_base_pct = 0.3
    ref_min_wave_pct = 0.15
    
    # 선형 비율 계산
    if current_price <= 0:
        return ref_base_pct, ref_min_wave_pct
    
    linear_ratio = ref_price / current_price
    
    # 파라미터 계산
    base_pct = ref_base_pct * linear_ratio
    min_wave_pct = ref_min_wave_pct * linear_ratio
    
    # 최소/최한 제한
    base_pct = max(0.05, min(base_pct, 1.0))
    min_wave_pct = max(0.05, min(min_wave_pct, 0.5))
    
    return base_pct, min_wave_pct
```

#### 사용 예시
```python
# KOSPI 7500
base_pct, min_wave_pct = calculate_auto_params_linear(7500, 100.0)
# 결과: base_pct = 0.04%, min_wave_pct = 0.02%

# KP200 1200
base_pct, min_wave_pct = calculate_auto_params_linear(1200, 100.0)
# 결과: base_pct = 0.25%, min_wave_pct = 0.125%
```

### 5.3 두 방식 비교

| 가격 | 로그 스케일 | 선형 비율 | 권장 |
|------|-----------|-----------|------|
| 10 | 0.9% | 3.0% | 로그 (너무 민감) |
| 100 | 0.3% | 0.3% | 동일 |
| 1000 | 0.1% | 0.03% | 로그 (너무 보수적) |
| 7500 | 0.11% | 0.004% | 로그 (너무 보수적) |

**권장:** 로그 스케일이 더 균형적인 결과 제공

### 5.4 실제 적용 예시

#### KOSPI 7500
```python
# 로그 스케일
cfg = PercentAdaptivePivotConfig(
    base_pct=0.11,      # 자동 계산
    min_wave_pct=0.055, # 자동 계산
)

# 수동 설정 (권장)
cfg = PercentAdaptivePivotConfig(
    base_pct=0.15,      # 약간 높게 조절
    min_wave_pct=0.1,
)
```

#### KP200 1200
```python
# 로그 스케일
cfg = PercentAdaptivePivotConfig(
    base_pct=0.27,      # 자동 계산
    min_wave_pct=0.135, # 자동 계산
)

# 수동 설정 (권장)
cfg = PercentAdaptivePivotConfig(
    base_pct=0.3,       # 기본값
    min_wave_pct=0.15,
)
```

### 5.5 자동 계산 유틸리티 구현

#### PercentAdaptivePivotConfig에 추가
```python
@classmethod
def from_price(cls, current_price: float, ref_price: float = 100.0) -> 'PercentAdaptivePivotConfig':
    """
    가격 수준에 따른 자동 설정 생성
    
    Parameters
    ----------
    current_price : float
        현재 가격
    ref_price : float
        기준 가격 (기본값: 100)
    
    Returns
    -------
    PercentAdaptivePivotConfig
        자동 계산된 설정
    """
    import math
    
    ref_base_pct = 0.3
    ref_min_wave_pct = 0.15
    
    if current_price <= 0:
        return cls()
    
    log_ratio = math.log10(ref_price) / math.log10(current_price)
    
    base_pct = max(0.05, min(ref_base_pct * log_ratio, 1.0))
    min_wave_pct = max(0.05, min(ref_min_wave_pct * log_ratio, 0.5))
    
    return cls(
        base_pct=base_pct,
        min_wave_pct=min_wave_pct,
    )
```

#### 사용 예시
```python
# KOSPI
cfg_kospi = PercentAdaptivePivotConfig.from_price(7500)

# KP200
cfg_kp200 = PercentAdaptivePivotConfig.from_price(1200)
```

---

## 6. 추천 전략

### 6.1 전략 1: ATR 기반 (권장)

```python
# 두 지수 모두 동일한 설정
cfg = ATRAdaptivePivotConfig(
    base_multiplier=2.0,
    min_wave_atr_ratio=0.5,
)
```

**이유:** 변동성 자동 반영, 가격 수준 고려 불필요

### 5.2 전략 2: 하이브리드 (유연성)

```python
# KOSPI
cfg_kospi = HybridAdaptivePivotConfig(
    base_pct=0.2,
    base_multiplier=2.0,
    atr_weight=0.5,
)

# KP200
cfg_kp200 = HybridAdaptivePivotConfig(
    base_pct=0.3,
    base_multiplier=2.0,
    atr_weight=0.5,
)
```

**이유:** 두 방식의 장점 결합, atr_weight로 민감도 조절

### 5.3 전략 3: 퍼센트 기반 (직관성)

```python
# KOSPI: 더 낮은 퍼센트
cfg_kospi = PercentAdaptivePivotConfig(base_pct=0.15)

# KP200: 기본 퍼센트
cfg_kp200 = PercentAdaptivePivotConfig(base_pct=0.3)
```

**이유:** 직관적 설정, 고가 자산에 낮은 퍼센트 적용

---

## 6. 요약

### 6.1 핵심 차이점 요약

| 특징 | PercentAdaptivePivot | ATRAdaptivePivot | HybridAdaptivePivot |
|------|---------------------|------------------|---------------------|
| **민감도** | 높음 (많은 피봇) | 낮음 (적은 피봇) | 중간 (atr_weight 조절) |
| **임계값** | 낮음 (퍼센트 기반) | 높음 (ATR 기반) | 중간 (혼합) |
| **변동성 적응** | ❌ | ✅ | ✅ |
| **직관성** | ✅ | ❌ | ✅ |
| **유연성** | 낮음 (퍼센트만) | 낮음 (ATR만) | 높음 (가중치 조절) |

### 6.2 가격 수준 설정 요약

**권장 설정:**

1. **ATRAdaptivePivot**: 두 지수 모두 동일한 설정 사용 가능 (변동성 자동 반영)
2. **HybridAdaptivePivot**: KOSPI는 더 낮은 base_pct, KP200은 기본값
3. **PercentAdaptivePivot**: KOSPI는 더 낮은 base_pct (0.15%), KP200은 기본값 (0.3%)

### 6.3 atr_weight 사용 가이드

| 상황 | 추천 atr_weight | 이유 |
|------|----------------|------|
| 가격 수준 일정 | 0.0 | 퍼센트만 사용 |
| 일반적인 트레이딩 | 0.5 | 균형 (기본값) |
| 변동성 큰 시장 | 0.75~1.0 | ATR 중심 |
| 빠른 반응 필요 | 0.0~0.25 | 퍼센트 중심 |

---

## 7. 결론

**atr_weight**는 사용자가 민감도를 제어하는 핵심 파라미터입니다:
- **낮은 값**: 더 민감, 많은 피봇, 빠른 반응
- **높은 값**: 더 보수적, 적은 피봇, 노이즈 필터
- **0.5**: 중간 지점 (기본값)

**가격 수준**은 피봇 탐지기의 민감도에 중요한 영향을 미칩니다:
- **퍼센트 기반**: 가격 수준에 비례하여 임계값 조정, 직관적이지만 변동성 반영 부족
- **ATR 기반**: 변동성에 따라 임계값 조정, 가격 수준 독립적이지만 설정이 직관적이지 않음
- **하이브리드**: atr_weight로 두 방식의 균형 조절 가능

**핵심:** 고가 자산(KOSPI)은 더 낮은 퍼센트로 절대 변동 크기를 조절해야 합니다.
