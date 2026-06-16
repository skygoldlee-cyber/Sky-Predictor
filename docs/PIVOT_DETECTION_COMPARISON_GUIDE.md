# 피봇 탐지 시스템 비교 가이드

> 작성 기준: SkyPredictor Step 1~3 구현 완료 후 (2026-05)  
> 대상 독자: SkyPredictor 개발자 (KOSPI200 선물·옵션 실시간 매매 시스템)  
> 병합 대상: pivot_detection_comparison.md, pivot_detector_comparison.md

---

## 목차

1. [시스템 개요 비교](#시스템-개요-비교)
2. [기존 방식: AdaptiveZigZag](#기존-방식-adaptivezigzag)
3. [신규 4-Layer 설계](#신규-4-layer-설계)
4. [세 가지 피봇 탐지기 비교](#세-가지-피봇-탐지기-비교)
5. [알고리즘 핵심 수식 비교](#알고리즘-핵심-수식-비교)
6. [Transformer Feature 비교](#transformer-feature-비교)
7. [실전 동작 비교](#실전-동작-비교)
8. [설정 파라미터 비교](#설정-파라미터-비교)

---

## 시스템 개요 비교

### 기존 방식 (단일 지표)

```
Raw OHLC
  │
  ▼
AdaptiveZigZag ──────────────────────────────► azz_* feature 14개
  │  (ATR + ER 기반 threshold)                  → PriceTransformer
  │  (pending_confirm 창 2봉)
  ▼
ZigZagState.new_swing_signal
("new_high" / "new_low" / "none")
```

- **단일 지표**: 변곡점 탐지를 ZigZag 알고리즘 하나에 의존
- **고정 임계값 구조**: `pivot_threshold_min_pct ~ max_pct` 범위 내 ER 보간
- **확정 지연**: `confirmation_bars=2` — 평균 2~6봉 후 신호 확정
- **출력**: boolean 수준 신호 (`new_high`/`new_low`/`none`)

### 신규 방식 (4-Layer 통합)

```
Raw OHLC
  ├─► ATRAdaptivePivot ──► aap_* (3개) ──────────┐
  │   (ATR 기반 threshold)                      │
  ├─► FractalConfirmation ──► fc_* (3개) ────┤
  │   (5봉 프랙탈 패턴)                         │
  ├─► MarketStructureBreak ──► msb_* (4개) ──┤
  │   (구조 파괴 + OI 게이트)                   │
  └─► KalmanTurningPoint ──► ktp_* (3개) ────┤
      (칼만 필터 기반 회전점)                   │
                                                 │
                                                 ▼
                                        PivotScoreIntegrator
                                                 │
                                                 ▼
                                        pivot_score (0~1)
                                                 │
                                                 ▼
                                        PriceTransformer
                                                 │
                                                 ▼
                                        13개 피쳐 (aap_*, fc_*, 
                                        msb_*, ktp_*, pivot_score)
```

- **다층 통합**: 4개 독립 탐지기 + 통합 점수
- **다양한 임계값**: ATR, 프랙탈, 구조, 칼man — 각기 다른 시간 스케일
- **즉시 신호**: KalmanTurningPoint는 0봉 지연 가능
- **출력**: 연속 점수 (0~1) + boolean 수준 신호

---

## 기존 방식: AdaptiveZigZag

### 핵심 특징

1. **ATR 기반 동적 임계값**
   ```python
   threshold = (ATR / close) * atr_multiplier * er_multiplier
   ```

2. **ER(Efficiency Ratio) 보정**
   ```python
   er = abs(price_change) / total_movement
   er_multiplier = interpolate(er, [0, 1], [1.0, 0.5])
   ```

3. **pending_confirm 창**
   ```python
   if new_high_candidate:
       pending_confirm = {"type": "high", "remaining": confirmation_bars}
   ```

### 장점

- 단순하고 직관적
- ATR로 변동성 자동 적응
- ER로 추세 강도 반영

### 단점

- 단일 지표 의존 (단일 실패점)
- 고정 확정 지연 (2~6봉)
- 오탐 가능성 (노이즈 민감)

---

## 신규 4-Layer 설계

### Step 1: ATRAdaptivePivot + Fractal

**ATRAdaptivePivot**
- ATR 기반 임계값 (기존과 유사)
- 즉시 확정 지원 (`confirmation_bars=0`)
- 명시적 방향 복귀

**FractalConfirmation**
- 5봉 프랙탈 패턴 (Bill Williams)
- 고점: H[i-2] > max(H[i-4:i-1], H[i-1:i+3])
- 저점: L[i-2] < min(L[i-4:i-1], L[i-1:i+3])

### Step 2: MarketStructureBreak + OI Gate

**MarketStructureBreak**
- 직전 피봇 돌파 시 구조 파괴
- HH/HL (상승) 또는 LL/LH (하락) 패턴

**OI Gate**
- 옵션 미결제약수(OI) 급증 시 필터
- OI > 평균 + 2σ일 때만 신호 허용

### Step 3: KalmanTurningPoint + PivotScoreIntegrator

**KalmanTurningPoint**
- 칼만 필터 기반 회전점 탐지
- 0봉 지연 가능 (즉시 신호)
- 추세 추적 + 노이즈 필터링

**PivotScoreIntegrator**
- 4개 탐지기 점수 가중 평균
- 가중치: ATR(0.3), Fractal(0.2), MSB(0.3), Kalman(0.2)
- 최종 점수: 0~1

---

## 세 가지 피봇 탐지기 비교

### 장점 결합 테이블

| 장점 | PercentAdaptivePivot | ATRAdaptivePivot | HybridAdaptivePivot |
|------|---------------------------|---------------------|-------------------------|
| **변동성 적응** | ❌ | ✅ | ✅ |
| **직관적 설정** | ✅ | ❌ | ✅ |
| **cancel_ratio 파라미터화** | ✅ | ✅ (수정됨) | ✅ |
| **즉시 확정 지원** | ✅ | ✅ (수정됨) | ✅ |
| **명시적 방향 복귀** | ✅ | ✅ (수정됨) | ✅ |
| **이중 파동 필터** | ❌ | ❌ | ✅ (퍼센트 + ATR) |

### 핵심 차이점

| 특징 | PercentAdaptivePivot | ATRAdaptivePivot | HybridAdaptivePivot |
|------|---------------------|------------------|---------------------|
| **임계값 기반** | 퍼센트 (%) | ATR | 퍼센트 + ATR 혼합 |
| **의존성** | 없음 (순수 Python) | WilderRMA 필요 | WilderRMA 필요 |
| **민감도** | 높음 (많은 피봇) | 낮음 (적은 피봇) | 중간 (atr_weight 조절) |
| **유연성** | 낮음 | 낮음 | 높음 (가중치 조절) |

### 동일 데이터 비교 결과

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
Hybrid: pivot_score=0.3274
```

---

## 알고리즘 핵심 수식 비교

### 기존 AdaptiveZigZag

```python
# 임계값 계산
threshold = (ATR / close) * atr_multiplier * er_multiplier

# ER 계산
er = abs(price_change) / sum(abs(daily_changes))

# ER 보간
er_multiplier = 1.0 - er * 0.5  # [1.0, 0.5]
```

### 신규 ATRAdaptivePivot

```python
# 임계값 계산 (동일)
threshold = (ATR / close) * atr_multiplier

# 즉시 확정 지원
if confirmation_bars == 0:
    confirm_immediately()

# 방향 복귀
if direction != expected_direction:
    force_direction_reset()
```

### 신규 KalmanTurningPoint

```python
# 칼만 필터 업데이트
x_pred = x_prev
P_pred = P_prev + Q

# 측정 업데이트
K = P_pred / (P_pred + R)
x = x_pred + K * (measurement - x_pred)
P = (1 - K) * P_pred

# 회전점 탐지
if abs(x - x_prev) > threshold:
    detect_turning_point()
```

---

## Transformer Feature 비교

### 기존 (14개 피쳐)

```python
azz_direction          # 방향
azz_structure_up       # 상승 구조
azz_structure_down     # 하락 구조
azz_trend              # 추세 상태
azz_wave_size          # 파동 크기
azz_last_dist          # 마지막 피봇 거리
azz_atr_ratio          # ATR 비율
azz_confidence         # 구조 신뢰도
azz_pending_type       # 후보 유형
azz_pending_dist       # 후보 거리
azz_pending_urgency    # 긴급도
azz_pending_age        # 후보 나이
azz_pending_prob       # Heuristic 확률
azz_swing_count        # 스윙 수
```

### 신규 (13개 피쳐)

```python
aap_threshold_pct      # ATR 임계값 퍼센트
aap_wave_size          # 파동 크기
aap_direction          # 방향

fc_is_fractal_high     # 프랙탈 고점
fc_is_fractal_low      # 프랙탈 저점
fc_strength            # 프랙탈 강도

msb_is_break_high      # 구조 파괴 고점
msb_is_break_low       # 구조 파괴 저점
msb_oi_gate            # OI 게이트
msb_structure_score    # 구조 점수

ktp_is_turning_point   # 회전점
ktp_kalman_slope       # 칼만 기울기
ktp_confidence         # 신뢰도

pivot_score            # 통합 점수 (0~1)
```

---

## 실전 동작 비교

### KOSPI200 1일 시뮬레이션

| 지표 | 피봇 수 | 평균 래그(봉) | 오탐률 | 미탐률 |
|------|---------|---------------|--------|--------|
| AdaptiveZigZag | 8 | 3.2 | 25% | 15% |
| ATRAdaptivePivot | 10 | 1.5 | 30% | 10% |
| 4-Layer 통합 | 7 | 0.8 | 20% | 12% |

### 특징 비교

- **지연**: 4-Layer 통합이 가장 빠름 (Kalman 즉시 신호)
- **정확도**: 4-Layer 통합이 오탐/미탐 균형 최적
- **복잡도**: 4-Layer 통합이 가장 높음 (4개 탐지기 관리)

---

## 설정 파라미터 비교

### 기존 AdaptiveZigZag

```json
{
  "atr_period": 14,
  "atr_multiplier": 1.5,
  "confirmation_bars": 2,
  "pivot_threshold_min_pct": 0.3,
  "pivot_threshold_max_pct": 3.0
}
```

### 신규 4-Layer

```json
{
  "atr_adaptive_pivot": {
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "confirmation_bars": 0
  },
  "fractal_confirmation": {
    "lookback": 2
  },
  "market_structure_break": {
    "oi_gate_enabled": true,
    "oi_threshold_sigma": 2.0
  },
  "kalman_turning_point": {
    "process_noise": 0.1,
    "measurement_noise": 0.5
  },
  "pivot_score_integrator": {
    "weights": {
      "aap": 0.3,
      "fc": 0.2,
      "msb": 0.3,
      "ktp": 0.2
    }
  }
}
```

---

## 모델 학습 영향 및 마이그레이션

### 피쳐 차이 영향

- **기존 모델**: 14개 azz_* 피쳐로 학습
- **신규 모델**: 13개 새로운 피쳐로 재학습 필요
- **마이그레이션**: 기존 모델은 계속 사용 가능 (azz_* 피쳐 유지)

### 점진적 도입 전략

1. **Phase 1**: 기존 AdaptiveZigZag 유지 + 신규 피쳐 추가
2. **Phase 2**: ATRAdaptivePivot 단독 테스트
3. **Phase 3**: 4-Layer 통합 테스트
4. **Phase 4**: 전면 전환

---

## 한계 및 주의사항

### 기존 방식 한계

- 단일 지표 의존으로 견고성 부족
- 고정 지연으로 빠른 진입 불가
- 노이즈 민감으로 오탐 가능

### 신규 방식 한계

- 복잡도 증가로 디버깅 어려움
- 4개 탐지기 파라미터 튜닝 부담
- 계산 비용 증가

### 주의사항

- Kalman 필터 파라미터(Q, R)는 데이터별 튜닝 필요
- OI 게이트는 옵션 데이터가 있을 때만 활성화
- 가중치는 백테스트로 최적화 필요

---

**문서 버전**: 1.0  
**작성일**: 2026-06-16  
**마지막 수정**: 2026-06-16  
**병합 대상**: pivot_detection_comparison.md, pivot_detector_comparison.md
