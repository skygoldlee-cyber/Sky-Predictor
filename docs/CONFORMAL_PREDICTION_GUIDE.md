# Conformal Prediction 가이드

예측 불확실성을 정량화하는 Conformal Prediction 가이드.

## 개요

Conformal Prediction은 머신러닝 모델의 예측 불확실성을 정량화하여 예측 구간을 제공합니다. 이를 통해 예측의 신뢰도를 더 정확하게 파악할 수 있습니다.

### 목적

- 예측 불확실성 정량화
- 예측 구간 제공
- 신뢰도 해석
- 리스크 관리

### 대상 독자

- ML 엔지니어
- 트레이더
- 리스크 관리자

## 핵심 개념

### Conformal Prediction 원리

```
┌─────────────────────────────────────────────────────────┐
│              Conformal Prediction 프로세스                │
│                                                          │
│  1. 학습 데이터 분할 (학습/보정)                         │
│     ↓                                                   │
│  2. 모델 학습                                          │
│     ↓                                                   │
│  3. 보정 데이터에서 비잔차 계산                        │
│     ↓                                                   │
│  4. 비잔차 분위수 계산 (conformal score)               │
│     ↓                                                   │
│  5. 예측 구간 생성 (예측 ± conformal score)            │
│     ↓                                                   │
│  6. 신뢰도 해석                                       │
└─────────────────────────────────────────────────────────┘
```

### 예측 구간

```
예측: 0.65
conformal_score: 0.10
구간: [0.55, 0.75]

해석:
- 1 - alpha = 90% 신뢰도로 실제 값이 [0.55, 0.75] 내에 있을 것임
```

## 설정

### config.json 설정

```json
{
  "prediction": {
    "conformal_alpha": 0.12,
    "conformal_path": "prediction/weights/conformal_model.pkl"
  }
}
```

### 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| conformal_alpha | float | 0.12 | 신뢰도 수준 (1 - alpha) | 0.05 ~ 0.2 |
| conformal_path | string | "" | Conformal 모델 경로 | prediction/weights/*.pkl |

## 사용 방법

### 기본 사용법

```python
from prediction.pipeline import PredictionPipeline

# 파이프라인 초기화 (Conformal 활성화)
pipeline = PredictionPipeline(
    conformal_alpha=0.12,
    conformal_path="prediction/weights/conformal_model.pkl",
)

# 예측 실행 (Conformal 구간 포함)
result = pipeline.predict(
    df=minute_df,
    now_dt=current_time,
    current_price=current_price,
)

# Conformal 구간 확인
prediction = result.get("prediction")
conformal_lower = result.get("conformal_lower")
conformal_upper = result.get("conformal_upper")

print(f"예측: {prediction:.2f}")
print(f"구간: [{conformal_lower:.2f}, {conformal_upper:.2f}]")
```

### Conformal Score 계산

```python
from prediction.conformal import compute_conformal_score

# 보정 데이터에서 비잔차 계산
calibration_residuals = []
for x_cal, y_cal in calibration_data:
    y_pred = model.predict(x_cal)
    residual = abs(y_cal - y_pred)
    calibration_residuals.append(residual)

# conformal score = 비잔차 분위수
conformal_score = np.quantile(
    calibration_residuals,
    1 - conformal_alpha
)
```

### 예측 구간 생성

```python
# 새로운 예측
y_pred = model.predict(x_new)

# 예측 구간
lower = y_pred - conformal_score
upper = y_pred + conformal_score

# 구간 클램핑
lower = max(0.0, lower)
upper = min(1.0, upper)
```

### 신뢰도 해석

```python
# 구간 너비 기반 신뢰도
interval_width = upper - lower

if interval_width < 0.1:
    confidence = "HIGH"
elif interval_width < 0.2:
    confidence = "MEDIUM"
else:
    confidence = "LOW"
```

## Confidence Conformal Width

### 설정

```json
{
  "prediction": {
    "confidence_conformal_width_max_for_high": 0.35,
    "confidence_conformal_width_max_for_medium": 0.55
  }
}
```

### 해석

| 구간 너비 | 신뢰도 | 설명 |
|-----------|--------|------|
| < 0.35 | HIGH | 높은 신뢰도 |
| 0.35 ~ 0.55 | MEDIUM | 중간 신뢰도 |
| > 0.55 | LOW | 낮은 신뢰도 |

## 학습 방법

### Conformal 모델 학습

```python
from prediction.conformal import train_conformal

# 학습 데이터 분할
train_data, cal_data = split_data(data, calibration_size=0.2)

# 모델 학습
model = train_model(train_data)

# Conformal score 계산
conformal_score = compute_conformal_score(
    model=model,
    calibration_data=cal_data,
    alpha=0.12,
)

# 저장
save_conformal_model(conformal_score, "conformal_model.pkl")
```

## 주의사항

### 일반적인 주의사항

1. **보정 데이터 크기**: 보정 데이터가 충분히 커야 정확한 conformal score 계산 가능 (최소 100개 이상 권장)
2. **alpha 설정**: alpha가 낮을수록 구간이 넓어집니다 (신뢰도 높음, 정확도 낮음)
3. **데이터 분포**: 학습 데이터와 실제 데이터 분포가 유사해야 합니다
4. **비용**: Conformal Prediction은 추가 계산 비용이 발생합니다

### 에러 처리

```python
try:
    conformal_score = compute_conformal_score(
        model=model,
        calibration_data=cal_data,
        alpha=0.12,
    )
except DataInsufficientError:
    # 데이터 부족 시 기본값 사용
    conformal_score = 0.1
except CalculationError:
    # 계산 오류 시 conformal 비활성화
    conformal_enabled = False
```

## 관련 문서

- [머신러닝 엔진 개요](./ML_ENGINE_OVERVIEW.md)
- [config.json 참조 가이드](./CONFIG_REFERENCE_GUIDE.md)

---

**문서 버전**: 1.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-25
