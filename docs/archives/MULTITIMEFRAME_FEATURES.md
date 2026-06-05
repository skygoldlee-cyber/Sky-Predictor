# Multi-timeframe Features

## 개요

이 프로젝트는 기존 1분봉만 사용하던 예측 알고리즘에 5분봉, 15분봉 멀티타임프레임 기능을 추가하여 승률 향상을 목표로 합니다.

## 기능 설명

### 1. Transformer 모델용 멀티스케일 피처

- **5분봉 피처 (MS5_KEYS)**: 중기 추석 파악, 노이즈 필터링
  - 수익률, 기울기, 거래량 비율, 캔들 실체 비율 등 8개 피처
  
- **15분봉 피처 (MS15_KEYS)**: 장기 추세 확인, 거짓 신호 감소
  - 수익률, 기울기, 거래량 비율, 캔들 실체 비율 등 8개 피처

### 2. Rule-based Adaptive Indicator용 멀티스케일

- **SuperTrend**: 1분봉 ATR + 5분봉 ATR + 15분봉 ATR 혼합
  - 가중 평균: 1분봉 50%, 5분봉 30%, 15분봉 20%
  - 더 안정적인 추세 판단
  
- **ZigZag**: 상위 타임프레임 피봇 방향 필터링
  - 5분봉 피봇 방향과 불일치하는 신호 제거
  - 15분봉 피봇 방향과 불일치하는 신호 제거 (더 강력한 필터)

## 구현 내용

### 1. Config 설정 (`config.json`)

```json
{
  "prediction": {
    "multiscale_5m": true,
    "multiscale_enabled": true,
    "multiscale_time_scales": [1, 5, 15]
  }
}
```

### 2. Feature Engineering (`prediction/features.py`)

- `calc_multiscale_features()`: 1분봉 → 5분봉 리샘플링 및 피처 계산
- `calc_multiscale_features_15m()`: 1분봉 → 15분봉 리샘플링 및 피처 계산
- `build_sequence()`: 멀티스케일 피처를 시퀀스에 통합

### 3. Pipeline Integration

**prediction_mixin.py**:
- `_build_and_predict_numeric()`: 5분봉/15분봉 피처 계산 후 build_sequence에 전달
- `_update_adaptive_multiscale_data()`: Adaptive Indicator에 멀티스케일 데이터 전달

**pipeline.py**:
- Feature dimension 계산에 멀티스케일 블록 추가
- `multiscale_block_dim = 16` (MS5: 8 + MS15: 8)

### 4. Adaptive Indicator Integration (`indicators/indicator_integration.py`)

**AdaptiveIndicatorManager**:
- `update_multiscale_data()`: 5분봉/15분봉 ATR과 피봇 방향 업데이트
- `get_multiscale_atr()`: 가중 평균 ATR 계산
- `should_filter_zigzag_signal()`: 상위 타임프레임 피봇 방향 기반 신호 필터링
- SuperTrend update 후 멀티스케일 ATR 자동 적용

### 5. GUI 설정 (`gui/controller.py`)

- Adaptive indicators 그룹에 체크박스 추가:
  - `Multiscale 5m`: 5분봉 피처 활성화
  - `Multiscale 15m`: 15분봉 피처 활성화
- Config 로드/저장/적용 기능 구현

## 사용 방법

### GUI에서 설정

1. GUI 실행
2. **Adaptive indicators** 섹션에서 체크박스 활성화:
   - `Multiscale 5m`: 중기 추세 (5분봉)
   - `Multiscale 15m`: 장기 추세 (15분봉)
3. **Apply** 버튼 클릭
4. 파이프라인 재시작

### Config 직접 수정

```json
{
  "prediction": {
    "multiscale_5m": true,
    "multiscale_enabled": true,
    "multiscale_time_scales": [1, 5, 15]
  }
}
```

## Feature Dimension 변경

**기존 (1분봉만):**
- OB: 11, CD: 9, OPT: 13, Adaptive: 43, Time: 11
- **Total: 87**

**변경 후 (1분 + 5분 + 15분):**
- OB: 11, CD: 9, OPT: 13, Multiscale: 16, Adaptive: 43, Time: 11
- **Total: 103 (+16 증가)**

## 모델 재학습 필요

**현재 상태:** `prediction/weights/` 디렉토리가 비어있음

**재학습 필수:**
- 새로운 feature dimension (103)에 맞는 모델 학습
- 학습된 가중치를 `prediction/weights/`에 배치:
  - `patch_tst_multiscale.pt` (Transformer)
  - `tft_multiscale.pt` (TFT, 사용 시)

**재학습 전 작동:**
- 가중치 없으면 자동으로 rule-based 모드로 동작
- 멀티스케일 피처는 계산되지만 Transformer 모델은 사용되지 않음
- Rule-based 신호 + Adaptive Indicator만 작동
- 로그에 "모델 가중치 미로드" 경고 표시

## 기대 효과

### Transformer 모델 사용 시

- 5분봉: 노이즈 감소, 중기 추석 파악
- 15분봉: 장기 추세 확인, 거짓 신호 감소
- 결합: 단기/중기/장기 추세 통합으로 승률 향상

### Rule-based Adaptive Indicator 사용 시

**SuperTrend:**
- 더 안정적인 ATR 사용으로 추선 개선
- 노이즈 감소

**ZigZag:**
- 상위 타임프레임 피봇 필터링으로 거짓 신호 제거
- 더 신뢰할 수 있는 피봇 신호만 통과

## 승률 분석

### 이론적 추정 (현재 65% 승률 기준)

| 필터링 | 예상 승률 | 향상 폭 | 신호 감소 |
|--------|----------|---------|-----------|
| 5분봉만 | 68~70% | +3~5% | 30~40% |
| 15분봉 추가 | 70~72% | +5~7% | 추가 20~30% |
| 둘 다 | 70~75% | +5~10% | 50~60% |

### 신호 횟수 변화

**현재 (1분봉만):**
- 하루 10~20회 신호 (평균 15회)

**멀티스케일 필터링 적용 시:**
- 5분봉 필터만: 6~12회 (30~40% 감소)
- 15분봉 필터 추가: 4~8회 (50~60% 감소)

### 추천 전략

**옵션 1: 보수적 접근**
- 5분봉 필터만 적용
- 승률: 68~70%, 신호: 9~12회
- 기회 손실 최소화

**옵션 2: 공격적 접근**
- 15분봉 필터까지 적용
- 승률: 70~75%, 신호: 4~8회
- 고품질 신호만 추려

**현재 65% 승률이라면 5분봉 필터만 적용 추천**

## 로그 확인

시작 시 다음 로그로 멀티스케일 활성화 확인:

```
[FEATURE_DIM] option_feature_set=v4 adaptive=True multiscale_5m=True multiscale_enabled=True scales=[1, 5, 15] multiscale_dim=16 time_dim=11 -> feature_dim=103
[MULTISCALE] ATR 5m=0.1234 15m=0.1456 | Pivot 5m=1 15m=1
```

## 주의사항

1. **모델 호환성**: 기존 가중치 파일 호환 불가, 재학습 필수
2. **데이터 요구량**: 5분봉/15분봉 계산을 위해 최소 15분 데이터 필요
3. **지연 시간**: 상위 타임프레임 피쳐에는 리샘플링 지연 발생
4. **신호 감소**: 필터링 적용 시 매매 기회 감소 고려

## 백테스트

정확한 효과 확인을 위해 백테스트 필요:

1. 과거 데이터 확보 (`data/` 디렉토리의 ticks_replay 파일)
2. 멀티스케일 ON/OFF 비교
3. 동일 기간 신호 횟수/승률 비교
4. 결과 분석

백테스트 스크립트는 별도 요청 시 작성 가능합니다.
