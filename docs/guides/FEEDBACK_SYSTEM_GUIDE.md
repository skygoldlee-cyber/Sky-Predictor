# 피드백 시스템 가이드

트레이딩 결과를 기반으로 예측 모델의 가중치를 동적으로 조절하는 피드백 시스템 가이드.

## 개요

피드백 시스템은 실제 트레이딩 결과를 분석하여 예측 모델의 가중치를 동적으로 조절합니다. Transformer와 TFT 가중치를 실시간으로 최적화하여 예측 정확도를 높입니다.

### 목적

- 실시간 가중치 조절
- 레짐별 가중치 최적화
- 예측 정확도 향상
- 적응형 모델 성능

### 대상 독자

- 시스템 운영자
- ML 엔지니어
- 트레이딩 전략 개발자

## 핵심 개념

### 피드백 시스템 흐름

```
┌─────────────────────────────────────────────────────────┐
│                  피드백 시스템 프로세스                     │
│                                                          │
│  1. 트레이딩 결과 수집                                 │
│     ↓                                                   │
│  2. 예측 정확도 계산                                   │
│     ↓                                                   │
│  3. 가중치 조절 로직                                   │
│     ↓                                                   │
│  4. Transformer 가중치 업데이트                        │
│     ↓                                                   │
│  5. 다음 예측에 적용                                   │
└─────────────────────────────────────────────────────────┘
```

### 가중치 조절 원리

```python
# 기본 가중치
transformer_weight = 0.5
tft_weight = 0.5

# 피드백 기반 조절
if transformer_accuracy > tft_accuracy:
    transformer_weight += adjustment
    tft_weight -= adjustment
else:
    transformer_weight -= adjustment
    tft_weight += adjustment
```

## 설정

### config.json 설정

```json
{
  "prediction": {
    "feedback_threshold_ticks": 10,
    "feedback_skip_hold_ticks": 2,
    "feedback_weight_high": 1.0,
    "feedback_weight_mid": 0.5,
    "feedback_weight_low": 0.25,
    "feedback_use_price_snapshot": true,
    "feedback_snapshot_tolerance_sec": 30.0,
    "feedback_snapshot_required": false
  }
}
```

### 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| feedback_threshold_ticks | int | 10 | 피드백 임계값 (틱) | 5 ~ 20 |
| feedback_skip_hold_ticks | int | 2 | HOLD 스킵 (틱) | 1 ~ 5 |
| feedback_weight_high | float | 1.0 | 높은 신뢰도 가중치 | 0.8 ~ 1.2 |
| feedback_weight_mid | float | 0.5 | 중간 신뢰도 가중치 | 0.3 ~ 0.7 |
| feedback_weight_low | float | 0.25 | 낮은 신뢰도 가중치 | 0.1 ~ 0.4 |
| feedback_use_price_snapshot | bool | true | 가격 스냅샷 사용 | true/false |
| feedback_snapshot_tolerance_sec | float | 30.0 | 스냅샷 허용 오차 (초) | 15 ~ 60 |
| feedback_snapshot_required | bool | false | 스냅샷 필수 | true/false |

## 사용 방법

### 기본 사용법

```python
from prediction.pipeline import PredictionPipeline

# 파이프라인 초기화 (피드백 활성화)
pipeline = PredictionPipeline(
    feedback_threshold_ticks=10,
    feedback_weight_high=1.0,
    feedback_weight_mid=0.5,
    feedback_weight_low=0.25,
)

# 예측 실행
result = pipeline.predict(
    df=minute_df,
    now_dt=current_time,
    current_price=current_price,
)

# 가중치 확인
transformer_weight = pipeline.get_transformer_weight()
tft_weight = 1.0 - transformer_weight
```

### 가중치 조절 로직

```python
from prediction.feedback_mixin import FeedbackMixin

# 신뢰도 기반 가중치 조절
confidence = result.get("confidence")

if confidence == "HIGH":
    adjustment = feedback_weight_high
elif confidence == "MEDIUM":
    adjustment = feedback_weight_mid
else:  # LOW
    adjustment = feedback_weight_low

# 가중치 업데이트
update_adaptive_weights(adjustment)
```

### 레짐별 가중치 조절

```python
# 레짐별 가중치 설정
regime_weights = {
    "bullish": {"transformer": 0.6, "tft": 0.4},
    "bearish": {"transformer": 0.4, "tft": 0.6},
    "ranging": {"transformer": 0.5, "tft": 0.5},
}

# 현재 레짐 확인
regime = detect_regime(market_data)

# 레짐별 가중치 적용
set_regime(regime)
```

### Confidence High Margin 역할

```python
# confidence_high_margin: 신뢰도 높음 판정 기준
confidence_high_margin = 0.15

# 확률이 0.5에서 얼마나 떨어져 있는지 계산
margin = abs(prob - 0.5)

if margin >= confidence_high_margin:
    confidence = "HIGH"
elif margin >= confidence_mid_margin:
    confidence = "MEDIUM"
else:
    confidence = "LOW"
```

## 피드백 수집 방식

### 가격 스냅샷 방식

```python
# 예측 시점의 가격 스냅샷 저장
price_snapshot = {
    "timestamp": prediction_time,
    "price": current_price,
    "predicted_direction": predicted_direction,
}

# 트레이딩 결과와 비교
if abs(trade_price - price_snapshot["price"]) <= tolerance:
    # 피드백 수집
    collect_feedback(price_snapshot, trade_result)
```

### 틱 기반 방식

```python
# 틱 기반 피드백
if abs(price_change) >= feedback_threshold_ticks * tick_size:
    # 피드백 수집
    collect_feedback(prediction, actual)
```

## 주의사항

### 일반적인 주의사항

1. **가중치 범위**: 가중치는 항상 0 ~ 1 사이여야 합니다.
2. **조절 속도**: 가중치 조절이 너무 빠르면 불안정해질 수 있습니다.
3. **데이터 품질**: 피드백 데이터의 품질이 중요합니다.
4. **과적합 방지**: 과도한 피드백은 과적합을 유발할 수 있습니다.

### 에러 처리

```python
try:
    update_adaptive_weights(adjustment)
except WeightRangeError:
    # 가중치 범위 초과 시 클램핑
    clamp_weight()
except DataInsufficientError:
    # 데이터 부족 시 스킵
    pass
```

## 관련 문서

- [머신러닝 엔진 개요](./ML_ENGINE_OVERVIEW.md)
- [트레이딩 시그널 생성 가이드](./TRADING_SIGNAL_GENERATION_GUIDE.md)
- [예측 알고리즘](./Prediction_Algorithm.md)

---

**문서 버전**: 1.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-25
