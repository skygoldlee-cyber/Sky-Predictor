# 모델 가이드

이 문서는 SkyPredictor 시스템에서 사용 가능한 4개의 딥러닝 모델 (PriceTransformer, PatchTST, Mamba, TFT)을 상세히 설명합니다.

---

## 목차

1. [모델 개요](#1-모델-개요)
2. [PriceTransformer](#2-pricetransformer)
3. [PatchTST](#3-patchtst)
4. [Mamba](#4-mamba)
5. [TFT (Temporal Fusion Transformer)](#5-tft-temporal-fusion-transformer)
6. [모델 비교](#6-모델-비교)
7. [데이터셋 공유](#7-데이터셋-공유)
8. [모델 선택 가이드](#8-모델-선택-가이드)

---

## 1. 모델 개요

### 사용 가능한 모델

| 모델 | 파일 | 아키텍처 | 특징 |
|------|------|----------|------|
| **PriceTransformer** | `model.py` | Transformer Encoder | 기본 Transformer, 안정적 |
| **PatchTST** | `patch_tst_model.py` | Patch-based Transformer | 연산 효율, 국소 패턴 포착 |
| **Mamba** | `mamba_model.py` | Mamba SSM | State Space Model, 장기 의존성 |
| **TFT** | `tft_model.py` | Temporal Fusion Transformer | 시간 기반 가변 선택, 해석 가능 |

### 공통 특징

- **입력:** (B, seq_len, feature_dim) - 시계열 피처
- **출력:** (B,) - 방향 확률 [0, 1]
- **feature_dim:** 동일하게 사용 가능 (OB + CD + OPT + [MS5] + [ADAPT] + TIME)
- **seq_len:** 60 (기본값)
- **가중치 호환성:** 모델별로 호환되지 않음

---

## 2. PriceTransformer

### 개요

기본 Transformer Encoder 기반 방향 예측 모델입니다. 가장 안정적이고 검증된 아키텍처입니다.

### 아키텍처

```
입력 (B, 60, feature_dim)
    ↓
Linear Projection (feature_dim → d_model)
    ↓
CLS Token + Positional Encoding
    ↓
Transformer Encoder (n_layers × n_heads)
    ↓
CLS Pooling
    ↓
Head (d_model → 1)
    ↓
Sigmoid → 방향 확률 [0, 1]
```

### 특징

- **안정성:** 가장 검증된 아키텍처
- **단순성:** 구조가 단순하여 디버깅 용이
- **성능:** 균형 잡힌 성능

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `feature_dim` | 95 | 입력 피처 차원 |
| `d_model` | 64 | Transformer 내부 차원 |
| `n_heads` | 4 | Multi-head Attention 헤드 수 |
| `n_layers` | 2 | Transformer Encoder 레이어 수 |
| `d_ff` | 128 | Feed-forward 차원 |
| `dropout` | 0.1 | 드롭아웃 비율 |
| `pooling` | "cls" | 풀링 방식 (cls/mean/recency_weighted) |

### 학습 방법

```bash
python training/train.py \
  --data dataset_shared.npz \
  --out prediction/weights/transformer_5m.pt \
  --multiscale-5m \
  --epochs 50 --batch-size 256 --lr 1e-3
```

### 장단점

| 장점 | 단점 |
|------|------|
| 안정적 | 연산량 상대적으로 높음 |
| 단순한 구조 | 국소 패턴 포착력 상대적으로 낮음 |
| 검증됨 | - |

---

## 3. PatchTST

### 개요

Patch-based Transformer 기반 방향 예측 모델입니다. 시계열을 겹치는 패치로 분할하여 연산 효율을 높이고 국소 패턴 포착력을 향상시킵니다.

### 아키텍처

```
입력 (B, 60, feature_dim)
    ↓
Patch Embedding (patch_len=8, stride=4)
    → 14개 패치 (60-8)//4 + 1 = 14
    ↓
Linear Projection (patch_len × feature_dim → d_model)
    ↓
Positional Encoding
    ↓
Transformer Encoder (n_layers × n_heads)
    ↓
CLS Pooling
    ↓
Head (d_model → 1)
    ↓
Sigmoid → 방향 확률 [0, 1]
```

### 특징

- **연산 효율:** 60개 토큰 → 14개 패치 (연산량 감소)
- **국소 패턴:** 캔들 군집 패턴 포착력 향상
- **최적화:** KP200 1분봉에 최적화된 기본값

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `feature_dim` | 95 | 입력 피처 차원 |
| `patch_len` | 8 | 패치 길이 (타임스텝 수) |
| `stride` | 4 | 패치 슬라이딩 간격 |
| `d_model` | 64 | Transformer 내부 차원 |
| `n_heads` | 4 | Multi-head Attention 헤드 수 |
| `n_layers` | 3 | Transformer Encoder 레이어 수 |
| `d_ff` | 128 | Feed-forward 차원 |
| `dropout` | 0.1 | 드롭아웃 비율 |
| `pooling` | "cls" | 풀링 방식 (cls/recency_weighted) |

### 학습 방법

```bash
python training/train_patch_tst.py \
  --data dataset_shared.npz \
  --out prediction/weights/patch_tst_multiscale.pt \
  --multiscale-5m \
  --epochs 50 --batch-size 256 --lr 1e-3
```

### 장단점

| 장점 | 단점 |
|------|------|
| 연산 효율 높음 | patch_len/stride 튜닝 필요 |
| 국소 패턴 포착력 우수 | - |
| KP200 최적화 | - |

---

## 4. Mamba

### 개요

Mamba SSM (State Space Model) 기반 방향 예측 모델입니다. 순환 구조를 가진 State Space Model로 장기 의존성 포착에 우수합니다.

### 아키텍처

```
입력 (B, 60, feature_dim)
    ↓
Linear Projection (feature_dim → d_model)
    ↓
LayerNorm
    ↓
Mamba Blocks (n_layers × d_state)
    ↓
Pooling (last/mean/recency_weighted)
    ↓
Head (d_model → 1)
    ↓
Sigmoid → 방향 확률 [0, 1]
```

### 특징

- **장기 의존성:** SSM으로 장기 패턴 포착
- **선형 복잡도:** Transformer의 제곱 복잡도 해결
- **순환 구조:** 시계열의 순차적 특성 활용

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `feature_dim` | 95 | 입력 피처 차원 |
| `d_model` | 64 | SSM 내부 차원 |
| `d_state` | 16 | SSM 상태 차원 (클수록 장기 의존성 강화) |
| `n_layers` | 4 | MambaBlock 스택 수 |
| `seq_len` | 60 | 입력 시퀀스 길이 |
| `dropout` | 0.1 | 드롭아웃 비율 |
| `pooling` | "last" | 풀링 방식 (last/mean/recency_weighted) |

### 학습 방법

```bash
python training/train_mamba.py \
  --data dataset_shared.npz \
  --out prediction/weights/mamba_multiscale.pt \
  --multiscale-5m \
  --epochs 50 --batch-size 256 --lr 1e-3
```

### 장단점

| 장점 | 단점 |
|------|------|
| 장기 의존성 포착 우수 | 상대적으로 새로운 아키텍처 |
| 선형 복잡도 | 검증 데이터 부족 |
| 순차적 특성 활용 | - |

---

## 5. TFT (Temporal Fusion Transformer)

### 개요

Temporal Fusion Transformer 기반 방향 예측 모델입니다. 시간 기반 가변 선택 네트워크(VSN)으로 해석 가능성을 제공합니다.

### 아키텍처

```
입력 (B, 60, feature_dim)
    ↓
Variable Selection Network (VSN)
    - past_unknown (feature_dim)
    - past_known (FUTURE_KNOWN_DIM)
    - future_known (FUTURE_KNOWN_DIM)
    ↓
LSTM Encoder/Decoder
    ↓
Multi-head Attention
    ↓
Gated Residual Network
    ↓
Head → Sigmoid → 방향 확률 [0, 1]
```

### 특징

- **해석 가능성:** VSN으로 피처 중요도 분석 가능
- **시간 구분:** past_unknown/past_known/future_known 구분
- **가변 선택:** 중요한 피처 자동 선택

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `past_unknown_dim` | 95 | 과거 미지 피처 차원 |
| `future_known_dim` | 11 | 미래 알려진 피처 차원 (시간) |
| `static_dim` | 0 | 정적 피처 차원 |
| `d_model` | 64 | 내부 차원 |
| `n_heads` | 4 | Multi-head Attention 헤드 수 |
| `n_layers` | 2 | Transformer 레이어 수 |
| `d_ff` | 128 | Feed-forward 차원 |
| `seq_len` | 60 | 입력 시퀀스 길이 |
| `horizon` | 300 | 예측 horizion (초) |
| `dropout` | 0.1 | 드롭아웃 비율 |

### 학습 방법

```bash
# Step 1: TFT 데이터셋 구축 (past_known/future_known 포함)
python -m prediction.data_builder \
  --files ticks_replay_*.jsonl.gz \
  --out dataset_tft.npz \
  --seq-len 60 --horizon 5 \
  --tft --tft-horizon-sec 300 \
  --multiscale-5m

# Step 2: TFT 학습
python training/train_tft.py \
  --data dataset_tft.npz \
  --out prediction/weights/tft_multiscale.pt \
  --multiscale-5m
```

### 장단점

| 장점 | 단점 |
|------|------|
| 해석 가능성 우수 | 데이터셋 구축 복잡 (past_known/future_known 필요) |
| 피처 중요도 분석 | 학습 시간 상대적으로 김 |
| 시간 구분 | - |

---

## 6. 모델 비교

### 성능 비교

| 모델 | 연산 효율 | 국소 패턴 | 장기 의존성 | 해석 가능성 | 안정성 |
|------|----------|----------|-----------|-----------|--------|
| **PriceTransformer** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **PatchTST** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Mamba** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **TFT** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 추천 사용 사례

| 상황 | 추천 모델 | 이유 |
|------|----------|------|
| **초기 학습** | PriceTransformer | 안정적, 검증됨 |
| **연산 효율 중요** | PatchTST | 연산량 감소, 국소 패턴 우수 |
| **장기 패턴 중요** | Mamba | 장기 의존성 포착 우수 |
| **해석 가능성 중요** | TFT | 피처 중요도 분석 가능 |

### 파라미터 수 비교

| 모델 | 기본 파라미터 수 | 학습 가능 파라미터 |
|------|-----------------|------------------|
| PriceTransformer | ~50K | ~50K |
| PatchTST | ~40K | ~40K |
| Mamba | ~60K | ~60K |
| TFT | ~100K | ~100K |

---

## 7. 데이터셋 공유

### Feature Dimension 일치

모든 모델은 동일한 feature_dim을 사용합니다:

| 설정 | Feature Dimension |
|------|------------------|
| multiscale_5m=False | 87 |
| multiscale_5m=True | 95 |

**계산:** OB(11) + CD(9) + OPT(13) + [MS5(8)] + ADAPT(43) + TIME(11)

### 데이터셋 구축 (공통)

**1단계: 데이터셋 구축 (한 번만)**
```bash
python -m prediction.data_builder \
  --files ticks_replay_*.jsonl.gz \
  --out dataset_shared.npz \
  --seq-len 60 --horizon 5 \
  --multiscale-5m
```

**2단계: 각 모델 학습 (동일 데이터셋 사용)**
```bash
# PriceTransformer
python training/train.py --data dataset_shared.npz --multiscale-5m

# PatchTST
python training/train_patch_tst.py --data dataset_shared.npz --multiscale-5m

# Mamba
python training/train_mamba.py --data dataset_shared.npz --multiscale-5m

# TFT (별도 데이터셋 필요)
python -m prediction.data_builder \
  --files ticks_replay_*.jsonl.gz \
  --out dataset_tft.npz \
  --tft --tft-horizon-sec 300 \
  --multiscale-5m
python training/train_tft.py --data dataset_tft.npz --multiscale-5m
```

### TFT 예외

TFT는 past_known/future_known이 필요하므로 별도 데이터셋 구축 필요:

```bash
python -m prediction.data_builder \
  --files ticks_replay_*.jsonl.gz \
  --out dataset_tft.npz \
  --tft --tft-horizon-sec 300 \
  --multiscale-5m
```

---

## 8. 모델 선택 가이드

### 상황별 추천

**1. 처음 학습 시:**
- 추천: **PriceTransformer**
- 이유: 가장 안정적이고 검증된 아키텍처

**2. 연산 효율 중요 시:**
- 추천: **PatchTST**
- 이유: 연산량 감소, 국소 패턴 포착력 우수

**3. 장기 패턴 중요 시:**
- 추천: **Mamba**
- 이유: 장기 의존성 포착 우수

**4. 해석 가능성 중요 시:**
- 추천: **TFT**
- 이유: 피처 중요도 분석 가능

### 현재 설정

**config.json:**
```json
{
  "prediction": {
    "model_class": "patch_tst",
    "mamba_enabled": false
  }
}
```

**현재 사용 모델:** PatchTST

### 모델 변경 방법

**PatchTST → PriceTransformer:**
```json
{
  "prediction": {
    "model_class": "transformer"
  }
}
```

**PatchTST → Mamba:**
```json
{
  "prediction": {
    "model_class": "mamba",
    "mamba_enabled": true
  }
}
```

**변경 후 재학습 필수**

---

## 요약

| 모델 | 추천 상황 | 특징 |
|------|----------|------|
| **PriceTransformer** | 초기 학습 | 안정적, 검증됨 |
| **PatchTST** | 연산 효율 | 연산량 감소, 국소 패턴 우수 |
| **Mamba** | 장기 패턴 | 장기 의존성 포착 우수 |
| **TFT** | 해석 가능성 | 피처 중요도 분석 가능 |

**데이터셋 공유:** 3개 모델 (PriceTransformer, PatchTST, Mamba)은 동일 데이터셋 사용 가능. TFT는 별도 데이터셋 필요.

**현재 사용:** PatchTST (config.json 설정)

---

**문서 버전**: 1.0  
**작성일**: 2026-04-26  
**마지막 수정**: 2026-04-26

---

## 관련 문서

- [모델 학습 가이드](./MODEL_TRAINING_GUIDE.md)
- [멀티스케일 피처 가이드](./MULTISCALE_FEATURES_GUIDE.md)
- [Adaptive Indicator 가이드](./ADAPTIVE_INDICATOR_GUIDE.md)
