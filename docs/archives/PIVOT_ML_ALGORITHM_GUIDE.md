# Pivot Prediction Machine Learning Algorithm Guide

피봇 후보 확정/취소 예측을 위한 머신러닝 알고리즘 상세 설명서.

## 목차

1. [개요](#개요)
2. [문제 정의](#문제-정의)
3. [데이터 구조](#데이터-구조)
4. [피처 엔지니어링](#피처-엔지니어링)
5. [모델 아키텍처](#모델-아키텍처)
6. [학습 전략](#학습-전략)
7. [평가 메트릭](#평가-메트릭)
8. [추론 파이프라인](#추론-파이프라인)
9. [실시간 통합](#실시간-통합)
10. [성능 최적화](#성능-최적화)

---

## 개요

### 목표

ZigZag 지표의 후보 피봇이 확정 피봇이 될 확률을 예측하여 트레이딩 의사결정의 신뢰도를 높이고, 허위 신호를 필터링합니다.

### 세부 목표

1. **분류 모델**: 후보가 확정될지 취소될지 이진 분류
2. **회귀 모델**: 확정 확률을 직접 예측 (0~1)
3. **시계열 모델**: 후보 수명(봉수) 예측

### 기존 방식과의 차이

| 구분 | 기존 Heuristic | ML 모델 |
|------|----------------|---------|
| 확률 계산 | 규칙 기반 (dist, urgency, age) | 데이터 기반 학습 |
| 신뢰도 | 고정 규칙 | 학습된 가중치 |
| 적응성 | 수동 파라미터 조절 | 자동 학습 |
| 해석 가능성 | 높음 | 중간 (SHAP 등으로 가능) |

---

## 문제 정의

### 문제 1: 확정/취소 분류

**입력**: 후보 등록 시점의 피처 벡터 (32차원)
**출력**: 확정(1) 또는 취소(0)
**유형**: 이진 분류

**학습 데이터**:
```python
X = [feature_vector_1, feature_vector_2, ...]  # (N, 32)
y = [1, 0, 1, 1, 0, ...]  # (N,) - 1=확정, 0=취소
```

### 문제 2: 확정 확률 회귀

**입력**: 후보 등록 시점의 피처 벡터 (32차원)
**출력**: 확정 확률 (0~1)
**유형**: 회귀 (출력 범위 제한)

**학습 데이터**:
```python
X = [feature_vector_1, feature_vector_2, ...]  # (N, 32)
y = [1.0, 0.0, 1.0, 1.0, 0.0, ...]  # (N,) - 실제 확정 여부
```

### 문제 3: 후보 잔여 수명 시계열 예측

**입력**: 후보 등록 후 '지금까지' 관측된 부분 시퀀스 (seq_len, 32)
**출력**: 지금 시점부터 확정/취소까지의 '잔여' 봉수 (log1p 스케일)
**유형**: 시계열 회귀

**학습 데이터**:
```python
X = [
    [f_t0, f_t1, f_t2, ...],  # 시퀀스 1 (부분 시퀀스)
    [f_t0, f_t1, f_t2, ...],  # 시퀀스 2 (부분 시퀀스)
    ...
]  # (N, seq_len, 32)
y = [2, 3, 1, 4, 2, ...]  # (N,) - 잔여 수명(봉수, log1p 스케일)
```

**중요 변경사항 (v2.0)**:
- 타겟을 '전체 수명'에서 '잔여 수명'으로 변경하여 데이터 누수 방지
- 추론 시 진행 중인 부분 시퀀스와 학습 분포 일치
- 시퀀스 길이가 곧 정답이 되는 누수 방지
- `pack_padded_sequence` 사용으로 패딩 무시

---

## 데이터 구조

### CandidateRecord

```python
@dataclass
class CandidateRecord:
    candidate_id: str           # 고유 ID
    candidate_type: str         # "high" or "low"
    candidate_price: float      # 후보 가격
    
    # 등록 정보
    registered_bar: int
    registered_time: str        # "HH:MM"
    registered_features: Dict   # 등록 시점 피처 (32개)
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

### CandidateSnapshot

```python
@dataclass
class CandidateSnapshot:
    bar_idx: int
    timestamp: str
    features: Dict[str, float]  # 32개 피처
    close: float
```

---

## 피처 엔지니어링

### ADAPT_KEYS (32개 피처)

**SuperTrend 관련 (9개)**
- `ast_direction`: 추세 방향 (1=상승, -1=하락, 0=횡보)
- `ast_trend_strength`: 추세 강도
- `ast_atr_ratio`: 현재 ATR / 평균 ATR
- `ast_bb_position`: 볼린저 밴드 내 위치
- `ast_er`: 효율 비율
- `ast_adx`: ADX 값
- `ast_bars_in_trend`: 현재 추세 지속 봉 수
- `ast_distance`: 현재 가격과 추세선 거리
- `ast_signal`: SuperTrend 신호

**ZigZag 관련 (19개)**
- `azz_direction`: 현재 방향
- `azz_structure_up`: 상승 구조 점수
- `azz_structure_down`: 하락 구조 점수
- `azz_trend`: 추세 상태
- `azz_wave_size`: 파동 크기
- `azz_last_dist`: 마지막 피봇 거리
- `azz_atr_ratio`: ATR 비율
- `azz_confidence`: 구조 신뢰도
- `azz_pending_type`: 후보 유형 (1=high, -1=low, 0=none)
- `azz_pending_dist`: 후보 거리 (%)
- `azz_pending_urgency`: 긴급도 (0~1)
- `azz_pending_age`: 후보 나이 (0~1)
- `azz_pending_prob`: Heuristic 확정 확률
- `azz_swing_count`: 확정 피봇 수
- `azz_avg_swing_size`: 평균 피봇 크기
- `azz_last_swing_type`: 마지막 피봇 유형
- `azz_fib_level`: 피보나치 레벨
- `azz_cluster_count`: 클러스터 내 피봇 수
- `azz_der_strength**: DER 강도

**크로스 피처 (4개)**
- `cross_trend_alignment`: 두 지표 추세 정렬
- `cross_signal_agreement`: 신호 일치
- `cross_volatility`: 변동성 크로스
- `cross_momentum**: 모멘텀 크로스

### 피처 정규화

모든 피처는 [0, 1] 범위로 정규화됩니다:

```python
# 예시
normalized = (value - min) / (max - min)
```

---

## 모델 아키텍처

### 1. 분류 모델 (PivotConfirmationClassifier)

```
Input (32) 
    ↓
Linear(128) → ReLU → Dropout(0.2)
    ↓
Linear(64) → ReLU → Dropout(0.2)
    ↓
Linear(1) → Sigmoid
    ↓
Output (0~1)
```

**특징**:
- 3층 MLP (Multi-Layer Perceptron)
- Dropout으로 과적합 방지
- Sigmoid로 확률 출력
- BCELoss (Binary Cross Entropy)

**코드**:
```python
class PivotConfirmationClassifier(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
```

### 2. 회귀 모델 (PivotProbabilityRegressor)

구조는 분류 모델과 동일하지만 손실 함수가 다릅니다:

**손실 함수**: MSELoss (Mean Squared Error)

```python
criterion = nn.MSELoss()
```

### 3. 시계열 모델 (PivotLifespanPredictor)

```
Input (seq_len, 32) + lengths (batch,)
    ↓
pack_padded_sequence (패딩 무시)
    ↓
LSTM(64, 2 layers)
    ↓
Last real hidden state (실제 마지막 스텝)
    ↓
Linear(1)
    ↓
Output (remaining_lifespan, log1p scale)
```

**특징**:
- LSTM으로 시계열 패턴 학습
- 2층 LSTM으로 깊은 패턴 학습
- `pack_padded_sequence`로 패딩 무시 및 실제 마지막 스텝 은닉 상태 사용
- 잔여 수명 예측 (log1p 스케일)
- lengths 파라미터로 실제 시퀀스 길이 전달

**코드**:
```python
class PivotLifespanPredictor(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        if lengths is not None:
            lengths_cpu = lengths.detach().to("cpu").long().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)
        last_output = h_n[-1]  # 최종 레이어의 실제 마지막 스텝 은닉 상태
        output = self.fc(last_output)
        return output
```

**중요 변경사항 (v2.0)**:
- `lengths` 파라미터 추가로 패딩 처리 개선
- `pack_padded_sequence` 사용으로 패딩 0벡터에서 출력 읽는 문제 해결
- 타겟을 잔여 수명으로 변경 (전체 수명이 아님)

### 4. 앙상블 모델 (PivotEnsemble)

분류 모델과 회귀 모델의 가중 평균:

```python
ensemble_prob = w * cls_prob + (1 - w) * reg_prob
```

기본 가중치: `w = 0.5`

---

## 학습 전략

### 데이터 분할 (v2.0 변경)

**이전 방식 (v1.0)**:
```python
Train: 70% (무작위)
Validation: 15% (무작위)
Test: 15% (무작위)
```

**현재 방식 (v2.0)**:
```python
# 레코드 단위 시간순 분할 (데이터 누수 방지)
Train: 70% (과거 레코드)
Validation: 15% (중간 레코드)
Test: 15% (최근 레코드)
```

**중요 변경사항**:
- 무작위 분할에서 시간순 분할로 변경
- 레코드 단위 분할로 같은 후보의 prefix들이 train/val/test에 흩어지는 것 방지
- 시계열 평가 원칙(과거 학습/미래 검증) 준수
- prefix 확장: 한 완결 후보에서 여러 부분 시퀀스 샘플 생성

### 학습 하이퍼파라미터

| 파라미터 | 분류 모델 | 회귀 모델 | 시계열 모델 |
|----------|----------|----------|-------------|
| Epoch | 50 | 50 | 50 |
| Batch Size | 32 | 32 | 16 |
| Learning Rate | 0.001 | 0.001 | 0.001 |
| Optimizer | Adam | Adam | Adam |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau |
| Patience | 10 | 10 | 10 |
| Dropout | 0.2 | 0.2 | 0.1 |

### Early Stopping

검증 손실이 10 epoch 동안 개선되지 않으면 학습 중단:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        break
```

### Learning Rate Scheduler

검증 손실이 개선되지 않으면 학습률 감소:

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    patience=5, 
    factor=0.5
)
```

### 시계열 데이터 전처리 (v2.0 변경)

**패딩/트리밍**:
```python
# 트리밍 (최근 max_seq_len) + 실제 길이 기록
if len(seq_features) > self.max_seq_len:
    seq_features = seq_features[-self.max_seq_len:]
true_len = len(seq_features)

# 후방 패딩 (packing으로 무시됨)
while len(seq_features) < self.max_seq_len:
    seq_features.append([0.0] * len(ADAPT_KEYS))
```

**로그 정규화**:
```python
# 잔여 수명 정규화 (로그 스케일)
remaining = terminal_bar - last_bar
y = np.log1p(remaining)  # log(1 + x)
```

**Prefix 확장 (v2.0)**:
```python
# 한 완결 후보에서 여러 부분 시퀀스 샘플 생성
for t in range(min_prefix, len(real_snaps) + 1):
    prefix = real_snaps[:t]
    remaining = terminal_bar - int(prefix[-1]["bar_idx"])
    # prefix → 학습 샘플, remaining → 타겟
```

**중요 변경사항**:
- 전체 수명 대신 잔여 수명을 타겟으로 사용
- 실제 길이 기록으로 `pack_padded_sequence`에 전달
- prefix 확장으로 학습 샘플 수 증가

---

## 평가 메트릭

### 분류 모델 메트릭

**Accuracy**: 정확도
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**: 정밀도
```python
precision = TP / (TP + FP)
```

**Recall**: 재현율
```python
recall = TP / (TP + FN)
```

**F1 Score**: 정밀도와 재현율의 조화 평균
```python
f1 = 2 * (precision * recall) / (precision + recall)
```

**AUC-ROC**: ROC 곡선 아래 면적
```python
auc = roc_auc_score(y_true, y_prob)
```

### 회귀 모델 메트릭

**MSE**: 평균 제곱 오차
```python
mse = mean_squared_error(y_true, y_pred)
```

**MAE**: 평균 절대 오차
```python
mae = mean_absolute_error(y_true, y_pred)
```

**R² Score**: 결정 계수
```python
r2 = r2_score(y_true, y_pred)
```

### 시계열 모델 메트릭

**MAE (bars)**: 봉수 기준 평균 절대 오차
```python
mae_bars = mean_absolute_error(
    np.expm1(y_true),  # 역정규화
    np.expm1(y_pred)
)
```

---

## 추론 파이프라인

### PivotPredictionPipeline

```python
class PivotPredictionPipeline:
    def __init__(
        self,
        classifier_path: str,
        regressor_path: str,
        lifespan_path: str,
        zigzag: AdaptiveZigZag,
        device: str = "cuda",
        ensemble_weight: float = 0.5,
    )
```

### 예측 흐름

```
1. 후보 확인
   ↓
2. 피처 추출 (get_transformer_features)
   ↓
3. 분류 모델 예측 (선택적)
   ↓
4. 회귀 모델 예측 (선택적)
   ↓
5. 앙상블 (분류 + 회귀)
   ↓
6. Heuristic 확률 계산
   ↓
7. 시계열 모델 예측 (수명)
   ↓
8. 결과 반환
```

### 예측 결과 (v2.0 변경)

```python
{
    "has_candidate": True,
    "candidate_type": "high",
    "candidate_price": 370.25,
    "classification_prob": 0.78,
    "regression_prob": 0.75,
    "ensemble_prob": 0.765,
    "ensemble_prediction": 1,
    "ensemble_confidence": 0.53,
    "heuristic_prob": 0.72,
    "predicted_remaining_bars": 1.5,  # v2.0 추가: 잔여 수명
    "predicted_lifespan_bars": 2.3,   # 전체 수명 = 경과 + 잔여
    "lifespan_confidence": 0.8,
}
```

**중요 변경사항 (v2.0)**:
- `predicted_remaining_bars` 필드 추가 (지금부터 확정/취소까지 남은 봉수)
- `predicted_lifespan_bars`는 전체 수명 추정 (경과 봉수 + 잔여 봉수)
- 하위 호환성 유지를 위해 `predicted_lifespan_bars` 유지

---

## 실시간 통합

### AdaptiveIndicatorManager 통합

```python
from kospi_indicators import AdaptiveIndicatorManager

manager = AdaptiveIndicatorManager()

# 파이프라인 초기화
manager.init_pivot_pipeline(
    classifier_path="prediction/weights/pivot_classifier_best.pt",
    regressor_path="prediction/weights/pivot_regressor_best.pt",
    lifespan_path="prediction/weights/pivot_lifespan_best.pt",
    device="cuda",
)

# 매 봉 업데이트
result = manager.update(high, low, close)

# 예측 결과 확인
pivot_pred = result.get("pivot_prediction")
```

### 실시간 예측 타이밍

```
봉 수신 → ZigZag 업데이트 → 후보 등록/확정/취소
         → 피처 추출 → ML 모델 예측 → 결과 반환
```

### 지연 시간

- 피처 추출: < 1ms
- 분류 모델 추론: < 1ms
- 회귀 모델 추론: < 1ms
- 시계열 모델 추론: < 2ms
- **총 지연**: < 5ms (CPU), < 1ms (GPU)

---

## 성능 최적화

### 1. 배치 추론

여러 후보를 동시에 예측:

```python
results = pipeline.predict_batch(features_list)
```

### 2. 모델 양자화

모델 크기 감소 및 추론 속도 향상:

```python
# FP16 양자화
model = model.half()
```

### 3. ONNX 변환

프로덕션 배포를 위한 ONNX 변환:

```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

### 4. 캐싱

동일한 피처에 대한 결과 캐싱:

```python
@lru_cache(maxsize=1000)
def predict_cached(features_tuple):
    return pipeline.predict(features_dict)
```

### 5. GPU 가속

CUDA 사용 시 5~10배 속도 향상:

```python
pipeline = PivotPredictionPipeline(..., device="cuda")
```

---

## 모니터링 및 로깅

### 학습 로그

```
Epoch 1/50 - Train Loss: 0.6521 - Val Loss: 0.6234 - Val Acc: 0.7234 - Val F1: 0.6891
Epoch 2/50 - Train Loss: 0.5834 - Val Loss: 0.5912 - Val Acc: 0.7512 - Val F1: 0.7234
...
Epoch 15/50 - Train Loss: 0.4123 - Val Loss: 0.4234 - Val Acc: 0.8234 - Val F1: 0.8123
체크포인트 저장: prediction/weights/pivot_classifier_best.pt
```

### 추론 로그

```
[INFO] PivotPredictionPipeline: 분류 모델 예측: prob=0.78
[INFO] PivotPredictionPipeline: 회귀 모델 예측: prob=0.75
[INFO] PivotPredictionPipeline: 앙상블 확률: 0.765
[INFO] PivotPredictionPipeline: 시계열 예측: lifespan=2.3봉
```

---

## WalkForward 평가 (v2.0 추가)

### WalkForwardEvaluator

DB 파라미터 추천 시스템의 실증 평가를 위한 WalkForward 평가기입니다.

**평가 모드**:

1. **Estimate 모드 (기본)**:
   - 검출을 재실행하지 않고 surrogate 계산 사용
   - 동일 test_date 지표에 파라미터 선형 보정만 적용
   - 빠르지만 실증 아님

2. **Real 모드**:
   - `evaluator_fn`으로 test_date 봉에 검출 재실행
   - 실제 파라미터 성능 측정
   - 느리지만 실증 가능

**사용 예시**:
```python
from prediction.pivot_parameter_db import WalkForwardEvaluator

# Estimate 모드 (빠름, 실증 아님)
evaluator = WalkForwardEvaluator(symbol="KP200")
results = evaluator.evaluate(
    start_date="2024-01-01",
    end_date="2024-03-31",
    lookback_days=30,
)

# Real 모드 (느림, 실증)
def my_evaluator_fn(test_date: str, params: dict):
    # test_date 봉에 params로 검출 재실행
    # metrics 반환: pivot_confirmation_rate, avg_lag_bars, ...
    return metrics

evaluator = WalkForwardEvaluator(
    symbol="KP200",
    evaluator_fn=my_evaluator_fn
)
results = evaluator.evaluate(
    start_date="2024-01-01",
    end_date="2024-03-31",
    lookback_days=30,
)
```

**판정 기준**:
- `PASS — DB 추천 유효`: 개선률 > 3%, 커버리지 ≥ 80%, 안정성 < 20%
- `PASS — 순환 강화`: 개선률 > 0%, 커버리지 ≥ 60%
- `FAIL — 순환 강화 의심`: 개선률 < 0%
- `ESTIMATE_ONLY`: surrogate 추정값 (real 모드에서 실증 필요)

**스크립트**:
```bash
python prediction/pivot_walkforward_eval.py \
    --symbol KP200 \
    --start_date 2024-01-01 \
    --end_date 2024-03-31 \
    --lookback_days 30
```

---

## 향후 개선 방향

### 1. Attention 메커니즘

시계열 모델에 Transformer 적용:

```python
class TransformerLifespanPredictor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4):
        self.transformer = nn.TransformerEncoder(...)
```

### 2. 멀티태스크 학습

확정/취소 + 수명을 동시에 예측:

```python
class MultiTaskModel(nn.Module):
    def forward(self, x):
        cls_output = self.classifier_head(x)
        lifespan_output = self.lifespan_head(x)
        return cls_output, lifespan_output
```

### 3. 강화 학습

실시간 트레이딩 환경에서 RL 적용:

```python
# State: 현재 피처
# Action: BUY/SELL/HOLD
# Reward: 수익률
```

### 4. 온라인 학습

새로운 데이터로 지속적 학습:

```python
# 매일 장 종료 후 재학습
pipeline.retrain(new_data)
```

### 5. 모델 해석

SHAP 값으로 피처 중요도 분석:

```python
import shap
explainer = shap.Explainer(model)
shap_values = explainer(X)
```

---

## 참고 문헌

1. **ZigZag Indicator**: 기술적 분석 지표 표준
2. **ATR (Average True Range)**: 변동성 측정
3. **LSTM**: Hochreiter & Schmidhuber (1997)
4. **Transformer**: Vaswani et al. (2017)
5. **Ensemble Methods**: Zhou (2012)

---

## 부록

### A. 학습 스크립트 실행 예시

```bash
# 분류 모델
python prediction/train_pivot_classifier.py \
    --data_path data/pivot_candidates.pkl \
    --output_dir prediction/weights \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001

# 회귀 모델
python prediction/train_pivot_regressor.py \
    --data_path data/pivot_candidates.pkl \
    --output_dir prediction/weights \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001

# 시계열 모델
python prediction/train_pivot_lifespan.py \
    --data_path data/pivot_candidates.pkl \
    --output_dir prediction/weights \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001 \
    --max_seq_len 120
```

### B. 모델 비교 실행 예시

```bash
python prediction/compare_pivot_models.py \
    --classifier_path prediction/weights/pivot_classifier_best.pt \
    --regressor_path prediction/weights/pivot_regressor_best.pt \
    --data_path data/pivot_candidates.pkl \
    --output_dir prediction/results
```

### C. 추론 실행 예시

```bash
# 단일 예측
python prediction/pivot_inference.py \
    --model_path prediction/weights/pivot_classifier_best.pt

# 시계열 예측
python prediction/pivot_lifespan_inference.py \
    --model_path prediction/weights/pivot_lifespan_best.pt
```

---

**문서 버전**: 2.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-06-16  
**변경사항**:
- v2.0: PivotLifespanPredictor 타겟을 '전체 수명'에서 '잔여 수명'으로 변경
- v2.0: 데이터 분할을 무작위에서 시간순 레코드 단위로 변경
- v2.0: pack_padded_sequence 지원 추가
- v2.0: WalkForwardEvaluator 및 pivot_walkforward_eval.py 추가
