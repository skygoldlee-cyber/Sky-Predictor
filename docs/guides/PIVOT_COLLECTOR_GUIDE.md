# Pivot Candidate Collector Guide

피봇 후보 수집기는 ZigZag 후보 피봇의 등록부터 확정/취소까지의 전체 히스토리를 수집하여 머신러닝 학습 데이터셋을 생성합니다.

## 개요

### 목적

- 후보 피봇 확정/취소 예측 모델 학습용 데이터 수집
- 시계열 피처 추적 (후보 수명 예측용)
- Heuristic 확률과 ML 모델 비교 벤치마킹

### 수집 데이터 구조

```python
CandidateRecord:
    candidate_id: str           # 고유 ID
    candidate_type: str         # "high" or "low"
    candidate_price: float      # 후보 가격
    
    # 등록 정보
    registered_bar: int
    registered_time: str        # "HH:MM"
    registered_features: Dict   # 등록 시점 피처
    registered_close: float
    
    # 확정/취소 정보
    label: int                  # 1=확정, 0=취소
    confirmed_bar: Optional[int]
    cancelled_bar: Optional[int]
    reason: Optional[str]       # 취소 사유
    
    # 수명
    lifespan_bars: int
    
    # 시계열 히스토리
    sequence: List[CandidateSnapshot]  # 매 봉마다의 피처 변화
    
    # 메타데이터
    symbol: str
    date: str
```

## 설정

### config.json 설정

```json
{
  "adaptive_indicator": {
    "zigzag": {
      "pivot_lifecycle_log": false,
      "pivot_lifecycle_log_prefix": "",
      "enable_pivot_collector": false,      // 수집 활성화
      "pivot_collector_max_sequence": 120   // 시계열 최대 길이
    }
  }
}
```

### 설정 설명

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `enable_pivot_collector` | bool | false | 수집 활성화 여부 |
| `pivot_collector_max_sequence` | int | 120 | 시계열 최대 길이 (봉수) |

## 사용 방법

### 1. 수집 활성화

config.json에서 `enable_pivot_collector`를 `true`로 설정:

```json
{
  "adaptive_indicator": {
    "zigzag": {
      "enable_pivot_collector": true
    }
  }
}
```

### 2. 데이터 수집

시스템을 실행하면 자동으로 후보 피봇 히스토리가 수집됩니다.

```python
from kospi_indicators import AdaptiveZigZag, AdaptiveZigZagConfig

# 설정
config = AdaptiveZigZagConfig(
    enable_pivot_collector=True,
    pivot_collector_max_sequence=120,
)

# 지표 초기화
zigzag = AdaptiveZigZag(config)

# 데이터 업데이트 (자동 수집)
for h, l, c, t in bars:
    state = zigzag.update(h, l, c, bar_time=t)
```

### 3. 데이터셋 저장

```python
# collector 접근
collector = zigzag.pivot_collector

if collector is not None:
    # 통계 확인
    stats = collector.get_statistics()
    print(f"총 후보: {stats['total_candidates']}")
    print(f"확정: {stats['confirmed']}")
    print(f"취소: {stats['cancelled']}")
    print(f"확정률: {stats['confirmation_rate']:.2%}")
    
    # 데이터셋 저장
    collector.save_dataset("data/pivot_candidates.pkl")
```

### 4. 데이터셋 로드

```python
from kospi_indicators import PivotCandidateCollector

collector = PivotCandidateCollector()
collector.load_dataset("data/pivot_candidates.pkl")

stats = collector.get_statistics()
```

## 수집 이벤트

### 후보 등록

조건: ZigZag 임계값 초과 시

```python
collector.on_candidate_registered(
    candidate_id="high_142_abc123",
    candidate_type="high",
    candidate_price=370.25,
    bar_idx=142,
    timestamp="09:15",
    features={...},  # get_transformer_features() 결과
    close=369.50,
    symbol="KP200 선물",
)
```

### 봉 업데이트

조건: 매 봉마다 (후보가 있는 경우)

```python
collector.on_bar_update(
    candidate_id="high_142_abc123",
    bar_idx=143,
    timestamp="09:16",
    features={...},
    close=369.80,
)
```

### 확정

조건: pending_confirm 확정 시

```python
collector.on_candidate_confirmed(
    candidate_id="high_142_abc123",
    confirmed_bar=144,
    confirmed_time="09:17",
    confirmed_close=370.00,
)
```

### 취소

조건: 다음 사유 발생 시
- max_wait_bars 초과
- pending_confirm_exception
- 반대후보교체

```python
collector.on_candidate_cancelled(
    candidate_id="high_142_abc123",
    cancelled_bar=143,
    cancelled_time="09:16",
    cancelled_close=369.80,
    reason="max_wait_bars",
)
```

## 학습 데이터 준비

### 분류 모델 (확정/취소)

```python
import pickle
import numpy as np

# 데이터셋 로드
with open("data/pivot_candidates.pkl", "rb") as f:
    data = pickle.load(f)

records = data["completed_candidates"]

# 피처 추출
X = []
y = []

for record in records:
    # 등록 시점 피처
    features = record["registered_features"]
    
    # 피처 벡터 생성 (ADAPT_KEYS 순서)
    feature_vector = [
        features.get(k, 0.0) for k in ADAPT_KEYS
    ]
    
    X.append(feature_vector)
    y.append(record["label"])  # 1=확정, 0=취소

X = np.array(X)
y = np.array(y)

print(f"총 샘플: {len(X)}")
print(f"확정: {y.sum()}")
print(f"취소: {(1-y).sum()}")
```

### 회귀 모델 (확정 확률)

동일 데이터셋 사용, 레이블은 실제 확정 여부 (0 또는 1)

### 시계열 모델 (후보 수명)

```python
# 시계열 데이터 추출
X_seq = []
y_lifespan = []

for record in records:
    if len(record["sequence"]) < 10:  # 최소 시퀀스 길이
        continue
    
    # 시계열 피처
    sequence = []
    for snapshot in record["sequence"]:
        seq_features = [
            snapshot["features"].get(k, 0.0) for k in ADAPT_KEYS
        ]
        sequence.append(seq_features)
    
    # 패딩 (최대 길이 120)
    while len(sequence) < 120:
        sequence.append([0.0] * len(ADAPT_KEYS))
    
    X_seq.append(sequence[:120])
    y_lifespan.append(record["lifespan_bars"])

X_seq = np.array(X_seq)
y_lifespan = np.array(y_lifespan)
```

## 통계 분석

```python
stats = collector.get_statistics()

print(f"=== 피봇 후보 통계 ===")
print(f"총 후보 수: {stats['total_candidates']}")
print(f"확정: {stats['confirmed']} ({stats['confirmation_rate']:.2%})")
print(f"취소: {stats['cancelled']}")
print(f"평균 수명 (확정): {stats['avg_lifespan_confirmed']:.1f}봉")
print(f"평균 수명 (취소): {stats['avg_lifespan_cancelled']:.1f}봉")
print(f"활성 후보: {stats['active_candidates']}")
```

## 취소 사유 분석

```python
from collections import Counter

reasons = Counter()
for record in collector.completed_candidates:
    if record["reason"]:
        reasons[record["reason"]] += 1

print("=== 취소 사유 ===")
for reason, count in reasons.most_common():
    print(f"{reason}: {count}")
```

## 주의사항

1. **메모리 사용**: 장기간 수집 시 메모리 사용량이 증가할 수 있음. 정기적으로 `save_dataset()` 후 `clear_completed()` 호출 권장
2. **09:00 anchor pivot**: 초기화용 피봇은 수집 제외됨
3. **시계열 길이**: `pivot_collector_max_sequence` 설정에 따라 시퀀스 길이 제한됨
4. **피처 일관성**: ADAPT_KEYS가 변경되면 기존 데이터셋과 호환되지 않음

## 다음 단계

Phase 2: 분류 모델 학습 ✅ 완료
- `PivotConfirmationClassifier` 구현
- 학습 스크립트 작성
- 추론 스크립트 작성

### 학습 실행

```bash
python prediction/train_pivot_classifier.py \
    --data_path data/pivot_candidates.pkl \
    --output_dir prediction/weights \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

### 추론 실행

```python
from prediction.pivot_inference import PivotPredictor

predictor = PivotPredictor("prediction/weights/pivot_classifier_best.pt")

# 피처 준비 (ADAPT_KEYS 기준)
features = zigzag.get_transformer_features(close)

# 예측
result = predictor.predict(features)
print(f"확정 확률: {result['confirmation_probability']:.2%}")
print(f"예측: {'확정' if result['prediction'] == 1 else '취소'}")
```

Phase 3: 회귀 모델 학습 ✅ 완료
- `PivotProbabilityRegressor` 학습 스크립트
- 분류/회귀 모델 비교 스크립트

### 회귀 모델 학습 실행

```bash
python prediction/train_pivot_regressor.py \
    --data_path data/pivot_candidates.pkl \
    --output_dir prediction/weights \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

### 모델 비교 실행

```bash
python prediction/compare_pivot_models.py \
    --classifier_path prediction/weights/pivot_classifier_best.pt \
    --regressor_path prediction/weights/pivot_regressor_best.pt \
    --data_path data/pivot_candidates.pkl \
    --output_dir prediction/results
```

비교 결과:
- Accuracy, Precision, Recall, F1 Score
- AUC (ROC 곡선)
- ROC 곡선 시각화

Phase 4: 시계열 모델 ✅ 완료
- `PivotLifespanPredictor` 학습 스크립트
- 시계열 추론 스크립트

### 시계열 모델 학습 실행

```bash
python prediction/train_pivot_lifespan.py \
    --data_path data/pivot_candidates.pkl \
    --output_dir prediction/weights \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001 \
    --max_seq_len 120
```

### 시계열 추론 실행

```python
from prediction.pivot_lifespan_inference import LifespanPredictor

predictor = LifespanPredictor("prediction/weights/pivot_lifespan_best.pt")

# 시계열 피처 준비 (Collector에서 가져옴)
sequence = collector_record["sequence"]

# 예측
result = predictor.predict(sequence)
print(f"예상 수명: {result['predicted_lifespan_bars']:.1f}봉")
```

Phase 5: 통합 및 배포 ✅ 완료
- `PivotPredictionPipeline` 구현
- `AdaptiveIndicatorManager` 통합
- 실시간 예측 지원

### 파이프라인 초기화

```python
from kospi_indicators import AdaptiveIndicatorManager

manager = AdaptiveIndicatorManager()

# 피봇 예측 파이프라인 초기화
manager.init_pivot_pipeline(
    classifier_path="prediction/weights/pivot_classifier_best.pt",
    regressor_path="prediction/weights/pivot_regressor_best.pt",
    lifespan_path="prediction/weights/pivot_lifespan_best.pt",
    device="cuda",
)
```

### 실시간 예측

```python
# 매 봉 업데이트 시 자동으로 피봇 예측 포함
result = manager.update(high, low, close)

# 예측 결과 확인
pivot_pred = result.get("pivot_prediction")
if pivot_pred and pivot_pred.get("has_candidate"):
    print(f"후보 유형: {pivot_pred['candidate_type']}")
    print(f"앙상블 확률: {pivot_pred['ensemble_prob']:.2%}")
    print(f"Heuristic 확률: {pivot_pred.get('heuristic_prob', 0):.2%}")
    
    if "predicted_lifespan_bars" in pivot_pred:
        print(f"예상 수명: {pivot_pred['predicted_lifespan_bars']:.1f}봉")
```

### 모델 상태 확인

```python
status = manager.pivot_pipeline.get_model_status()
print(f"분류 모델: {status['classifier_loaded']}")
print(f"회귀 모델: {status['regressor_loaded']}")
print(f"시계열 모델: {status['lifespan_loaded']}")
```

## 전체 구현 완료

### Phase 1: 데이터 수집 ✅
- PivotCandidateCollector 구현
- AdaptiveZigZag 통합
- config.json 설정

### Phase 2: 분류 모델 ✅
- PivotConfirmationClassifier 구현
- 학습/추론 스크립트

### Phase 3: 회귀 모델 ✅
- PivotProbabilityRegressor 학습
- 모델 비교 스크립트

### Phase 4: 시계열 모델 ✅
- PivotLifespanPredictor 학습
- 시계열 추론

### Phase 5: 통합 및 배포 ✅
- PivotPredictionPipeline 구현
- AdaptiveIndicatorManager 통합
- 실시간 예측 지원
