# 머신러닝 엔진 개요

SkyPredictor 시스템의 머신러닝 예측 엔진 구조 정리.

## 개요

SkyPredictor는 다양한 머신러닝 모델을 조합하여 시장 방향을 예측합니다. 사용자는 `numeric_predictor`와 `model_class` 설정을 통해 예측 엔진을 구성할 수 있습니다.

## 예측 엔진 유형 (numeric_predictor)

### 1. Transformer (transformer)
단일 Transformer 모델을 사용하여 예측합니다.

**특징**:
- 시계열 Transformer 기반
- Attention 메커니즘으로 장기 의존성 학습
- 빠른 추론 속도

**사용 방법**:
```json
{
  "numeric_predictor": "transformer",
  "model_class": "transformer"
}
```

### 2. TFT (tft)
단일 Temporal Fusion Transformer 모델을 사용합니다.

**특징**:
- Google의 TFT 아키텍처
- 다변량 시계열 예측 최적화
- 변수 중요도 해석 가능

**사용 방법**:
```json
{
  "numeric_predictor": "tft",
  "tft_horizon": 300
}
```

### 3. Ensemble (ensemble)
Transformer와 TFT를 앙상블하여 예측합니다.

**특징**:
- 두 모델의 가중 평균
- `transformer_weight`로 가중치 조절 (기본 0.5)
- disagreement_hold: 모델 간 의견 불일치 시 HOLD
- ensemble_agreement_confidence_boost: 일치 시 신뢰도 부스트

**사용 방법**:
```json
{
  "numeric_predictor": "ensemble",
  "model_class": "patch_tst",
  "transformer_weight": 0.5,
  "disagreement_hold": true,
  "ensemble_agreement_confidence_boost": true
}
```

### 4. Rule-Based (rule_based)
머신러닝 없이 규칙 기반으로 예측합니다.

**특징**:
- 기술적 지표 기반 규칙
- 빠른 실행 속도
- 학습 불필요

**사용 방법**:
```json
{
  "numeric_predictor": "rule_based",
  "buy_threshold": 0.62,
  "sell_threshold": 0.38
}
```

## 모델 클래스 (model_class)

### 1. Transformer (transformer)
기본 Transformer 아키텍처입니다.

**특징**:
- Multi-head Attention
- Positional Encoding
- Feed-forward Network

### 2. PatchTST (patch_tst)
Patch Time Series Transformer 아키텍처입니다.

**특징**:
- 시계열을 패치로 분할하여 처리
- `patch_len`: 패치 길이 (기본 8)
- `stride`: 패치 간격 (기본 4)
- 더 나은 장기 패턴 학습

**사용 방법**:
```json
{
  "model_class": "patch_tst",
  "patch_len": 8,
  "stride": 4
}
```

## Mamba 모델 (선택적)

Mamba 상태 공간 모델을 앙상블에 추가할 수 있습니다.

**특징**:
- 효율적인 시계열 모델링
- Transformer보다 빠른 추론
- 긴 시퀀스 처리에 적합

**사용 방법**:
```json
{
  "mamba_enabled": true,
  "mamba_weights_path": "prediction/weights/mamba_model.pt",
  "mamba_weight": 0.33
}
```

**참고**: Mamba는 ensemble 모드에서만 작동하며, transformer와 tft와 함께 3-way 앙상블을 형성합니다.

## 앙상블 구조

### 2-Way Ensemble (Transformer + TFT)
```
Transformer (weight: 0.5)
    ↓
TFT (weight: 0.5)
    ↓
Ensemble Output
```

### 3-Way Ensemble (Transformer + TFT + Mamba)
```
Transformer (weight: 0.5)
    ↓
TFT (weight: 0.5)
    ↓
Mamba (weight: 0.33)
    ↓
Ensemble Output (정규화된 가중합)
```

## 현재 설정 (config.json)

```json
{
  "numeric_predictor": "ensemble",
  "model_class": "patch_tst",
  "patch_len": 8,
  "stride": 4,
  "mamba_enabled": false,
  "mamba_weights_path": "",
  "mamba_weight": 0.33,
  "transformer_weight": 0.5,
  "tft_weights_path": "",
  "tft_horizon": 300
}
```

**해석**:
- 예측 엔진: Ensemble (Transformer + TFT)
- 모델 클래스: PatchTST
- Mamba: 비활성화
- Transformer 가중치: 0.5 (TFT도 0.5)

## 모델 파일 위치

| 모델 | 기본 경로 | 설명 |
|------|----------|------|
| Transformer | `prediction/weights/patch_tst_5m.pt` | PatchTST 5분봉 모델 |
| TFT | `prediction/weights/tft_model.pt` | TFT 모델 (선택적) |
| Mamba | `prediction/weights/mamba_model.pt` | Mamba 모델 (선택적) |

## Conformal Prediction

Conformal Prediction을 사용하여 예측 불확실성을 정량화할 수 있습니다.

**사용 방법**:
```json
{
  "conformal_alpha": 0.12,
  "conformal_path": "prediction/weights/conformal_model.pkl"
}
```

## 피봇 예측 모델 (최신 추가)

피봇 후보 확정/취소 예측을 위한 별도 모델 세트입니다.

### 모델 종류

1. **PivotConfirmationClassifier**: 확정/취소 분류
2. **PivotProbabilityRegressor**: 확정 확률 회귀
3. **PivotLifespanPredictor**: 후보 수명 시계열 예측

### 학습 스크립트

- `prediction/train_pivot_classifier.py`
- `prediction/train_pivot_regressor.py`
- `prediction/train_pivot_lifespan.py`

### 추론 스크립트

- `prediction/pivot_inference.py`
- `prediction/pivot_lifespan_inference.py`

### 통합 파이프라인

- `prediction/pivot_pipeline.py`: 세 모델 통합

## 설정 요약

| 설정 | 옵션 | 기본값 | 설명 |
|------|------|--------|------|
| numeric_predictor | transformet, tft, ensemble, rule_based | ensemble | 예측 엔진 유형 |
| model_class | transformer, patch_tst | patch_tst | 모델 아키텍처 |
| mamba_enabled | true, false | false | Mamba 사용 여부 |
| transformer_weight | 0.0 ~ 1.0 | 0.5 | Transformer 가중치 |
| mamba_weight | 0.0 ~ 1.0 | 0.33 | Mamba 가중치 |
| patch_len | 정수 | 8 | PatchTST 패치 길이 |
| stride | 정수 | 4 | PatchTST 패치 간격 |
| tft_horizon | 정수 | 300 | TFT 예측 지평 |

## 추천 설정

### 고성능 (앙상블)
```json
{
  "numeric_predictor": "ensemble",
  "model_class": "patch_tst",
  "mamba_enabled": true,
  "transformer_weight": 0.5,
  "mamba_weight": 0.33
}
```

### 빠른 추론 (단일 모델)
```json
{
  "numeric_predictor": "transformer",
  "model_class": "patch_tst"
}
```

### 해석 가능성 (TFT)
```json
{
  "numeric_predictor": "tft",
  "tft_horizon": 300
}
```

## 관련 문서

- [Transformer 가이드](./Transformer_GUIDE.md)
- [TFT 설계 가이드](./TFT_DUAL_MODEL_DESIGN_GUIDE.md)
- [AI 예측 알고리즘 기술 문서](./AI_Prediction_Algorithm_Technical_Doc.md)
- [피봇 예측 ML 알고리즘 가이드](./PIVOT_ML_ALGORITHM_GUIDE.md)

---

**문서 버전**: 1.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-25
