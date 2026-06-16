# 멀티스케일 피처 가이드

다양한 타임스케일의 데이터를 결합하여 예측 성능을 높이는 멀티스케일 피처 가이드.

## 개요

멀티스케일 피처는 1분, 5분, 15분 등 다양한 타임스케일의 데이터를 결합하여 예측 모델의 성능을 높입니다. 단일 타임스케일의 한계를 극복하고 장기/단기 패턴을 모두 포착할 수 있습니다.

### 목적

- 다중 타임스케일 패턴 포착
- 예측 성능 향상
- 장기/단기 추세 동시 분석
- 노이즈 감소
- Rule-based 매매 신호 품질 개선

### 대상 독자

- ML 엔지니어
- 트레이더
- 시스템 운영자

## 핵심 개념

### 멀티스케일 피처 구조

```
┌─────────────────────────────────────────────────────────┐
│              멀티스케일 피처 구조                         │
│                                                          │
│  1분 타임스케일 ──┐                                    │
│                   ├─→ 피처 결합 → 예측 모델           │
│  5분 타임스케일 ──┤                                    │
│                   │                                    │
│ 15분 타임스케일 ──┘                                    │
│                                                          │
│  5분/15분 타임스케일 ──→ Adaptive Indicator (Rule-based)│
└─────────────────────────────────────────────────────────┘
```

### 지원되는 타임스케일

| 타임스케일 | 설명 | 특징 |
|-----------|------|------|
| 1분 | 단기 패턴 | 빠른 반응, 노이즈 많음 |
| 5분 | 중기 패턴 | 균형 잡힌 패턴, 노이즈 감소 |
| 15분 | 장기 패턴 | 추세 포착, 느린 반응, 거짓 신호 필터링 |

## 설정

### config.json 설정

```json
{
  "prediction": {
    "multiscale_5m": true,
    "multiscale_enabled": true,
    "multiscale_time_scales": [1, 5]
  }
}
```

### 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| multiscale_enabled | bool | true | 멀티스케일 활성화 | true/false |
| multiscale_time_scales | list[int] | [1, 5] | 타임스케일 목록 | [1, 5] 또는 [1, 5, 15] |
| multiscale_5m | bool | true | 5분 멀티스케일 (레거시 호환) | true/false |

### GUI 설정

GUI의 **Adaptive indicators** 섹션에서 체크박스로 제어:

- **Multiscale 5m**: 5분봉 기반 멀티스케일 피처 활성화 (중기 추세)
- **Multiscale 15m**: 15분봉 기반 멀티스케일 피처 활성화 (장기 추세)

**기본값**: 5분봉만 활성화 (승률 65% → 68~70% 기대)

## 사용 방법

### 기본 사용법

```python
from prediction.pipeline import PredictionPipeline

# 파이프라인 초기화 (멀티스케일 활성화)
pipeline = PredictionPipeline(
    multiscale_enabled=True,
    multiscale_time_scales=[1, 5],
)

# 예측 실행 (멀티스케일 피처 포함)
result = pipeline.predict(
    df=minute_df,
    now_dt=current_time,
    current_price=current_price,
)
```

### GUI에서 설정

1. GUI 실행
2. **Adaptive indicators** 섹션에서 체크박스 활성화:
   - `5m`: 5분봉 피처 (기본 체크)
   - `15m`: 15분봉 피처 (기본 체크 해제)
3. **Apply** 버튼 클릭
4. 파이프라인 재시작

## 기대 효과

### Transformer 모델 사용 시

- **5분봉**: 노이즈 감소, 중기 추세 파악
- **15분봉**: 장기 추세 확인, 거짓 신호 감소
- **결합**: 단기/중기/장기 추세 통합으로 승률 향상

### Rule-based Adaptive Indicator 사용 시

**SuperTrend:**
- 1분봉 ATR + 5분봉 ATR + 15분봉 ATR 혼합
- 가중 평균: 1분봉 50%, 5분봉 30%, 15분봉 20%
- 더 안정적인 추선, 노이즈 감소

**ZigZag:**
- 상위 타임프레임 피봇 방향 필터링
- 5분봉 피봇 방향과 불일치하는 신호 제거
- 15분봉 피봇 방향과 불일치하는 신호 제거 (더 강력한 필터)

## 승률 분석

### 현재 65% 승률 기준

| 필터링 | 예상 승률 | 향상 폭 | 신호 감소 | 하루 신호 |
|--------|----------|---------|-----------|-----------|
| 5분봉만 | 68~70% | +3~5% | 30~40% | 9~12회 |
| 15분봉 추가 | 70~72% | +5~7% | 추가 20~30% | 4~8회 |
| 둘 다 | 70~75% | +5~10% | 50~60% | 4~8회 |

### 추천 설정

**현재 65% 승률에 추천:**
- **5분봉만 활성화**
- 승률: 68~70%
- 신호: 9~12회/일
- 기회 손실 최소화, 승률 개선

## Feature Dimension 변경

**기존 (1분봉만):**
- OB: 11, CD: 9, OPT: 13, Adaptive: 43, Time: 11
- **Total: 87**

**변경 후 (1분 + 5분):**
- OB: 11, CD: 9, OPT: 13, Multiscale: 8 (MS5), Adaptive: 43, Time: 11
- **Total: 95 (+8 증가)**

**변경 후 (1분 + 5분 + 15분):**
- OB: 11, CD: 9, OPT: 13, Multiscale: 16 (MS5 + MS15), Adaptive: 43, Time: 11
- **Total: 103 (+16 증가)**

## 주의사항

### 일반적인 주의사항

1. **데이터 요구량**: 멀티스케일은 더 많은 데이터를 요구합니다 (최소 15분 데이터 필요)
2. **계산 비용**: 여러 타임스케일 처리로 계산 비용 증가
3. **지연**: 장기 타임스케일은 더 많은 지연 발생
4. **과적합**: 너무 많은 타임스케일은 과적합 유발 가능
5. **모델 호환성**: 기존 가중치 파일 호환 불가, 재학습 필수

### 타임스케일 선택

**권장 조합**:
- 보수적 (현재 65% 승률): [1, 5]
- 균형: [1, 5, 15]
- 공격적 (승률 최우선): [1, 5, 15]

**비권장**:
- 너무 많은 타임스케일: [1, 5, 15, 30, 60] (과적합 위험)

### 모델 재학습

**현재 상태:** `prediction/weights/` 디렉토리가 비어있음

**재학습 필수:**
- 새로운 feature dimension에 맞는 모델 학습
- 학습된 가중치를 `prediction/weights/`에 배치:
  - `patch_tst_multiscale.pt` (Transformer)
  - `tft_multiscale.pt` (TFT, 사용 시)

**재학습 전 작동:**
- 가중치 없으면 자동으로 rule-based 모드로 동작
- 멀티스케일 피처는 계산되지만 Transformer 모델은 사용되지 않음
- Rule-based 신호 + Adaptive Indicator만 작동

## 로그 확인

시작 시 다음 로그로 멀티스케일 활성화 확인:

```
[FEATURE_DIM] option_feature_set=v4 adaptive=True multiscale_5m=True multiscale_enabled=True scales=[1, 5] multiscale_dim=8 time_dim=11 -> feature_dim=95
[MULTISCALE] ATR 5m=0.1234 15m=0.0000 | Pivot 5m=1 15m=0
```

## 관련 문서

- [Multi-timeframe Features 상세](./MULTITIMEFRAME_FEATURES.md)
- [머신러닝 엔진 개요](./ML_ENGINE_OVERVIEW.md)
- [Adaptive Indicator 가이드](./ADAPTIVE_INDICATOR_GUIDE.md)

---

**문서 버전**: 2.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-26 (Rule-based Adaptive Indicator 적용, GUI 설정 추가)
