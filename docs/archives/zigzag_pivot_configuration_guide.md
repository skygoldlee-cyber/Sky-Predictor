# ZigZag 피봇 결정 설정 가이드

## 개요

이 문서는 AdaptiveZigZag 인디케이터의 피봇 결정 관련 기본 설정과 시장 레짐을 반영한 적응형 파라미터 설정을 상세히 설명합니다.

## 목차

1. [기본 설정 개요](#기본-설정-개요)
2. [ZigZag 파라미터 상세](#zigzag-파라미터-상세)
3. [시장 레질 기반 적응형 파라미터](#시장-레짐-기반-적응형-파라미터)
4. [config.json 설정 예시](#configjson-설정-예시)
5. [실전 활용 가이드](#실전-활용-가이드)
6. [파라미터 튜닝 팁](#파라미터-튜닝-팁)

---

## 기본 설정 개요

AdaptiveZigZag는 ATR(Average True Range) 기반 동적 임계값을 사용하여 피봇을 결정합니다. 기본적으로 다음 3단계 생애주기를 따릅니다:

1. **후보 등록**: 신고점/신저점 형성 시 후보 등록
2. **pending_confirm**: 확인 봉수 동안 추세 유지 확인
3. **확정**: 확인 봉수 경과 후 피봇 확정

### 핵심 설계 원칙

- **동적 임계값**: ATR 비율로 임계값 결정 (고정 퍼센트 미사용)
- **다층 적응**: ER(추세 강도) + DER(방향 불일치) + 시간대 + 감쇄 4층 적용
- **H/L 교번 강제**: 연속된 동일 타입 피봇 자동 병합
- **캐싱**: 동일 데이터 서명 시 ZigZag 상태 재사용

---

## ZigZag 파라미터 상세

### 기본 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `atr_period` | int | 14 | ATR 계산 기간 (봉) |
| `atr_multiplier` | float | 1.5 | ATR 배수 기본값 (임계값 = ATR × 배수) |
| `atr_multiplier_min` | float | 1.0 | ATR 배수 최소값 |
| `atr_multiplier_max` | float | 4.0 | ATR 배수 최대값 |
| `confirmation_bars` | int | 2 | 피봇 확정 확인 봉수 |
| `pivot_threshold_min_pct` | float | 0.3 | 피봇 임계값 최소 퍼센트 |
| `pivot_threshold_max_pct` | float | 3.0 | 피봇 임계값 최대 퍼센트 |
| `max_swings` | int | 20 | 유지할 최대 스윙 수 |
| `freeze_on_confirm` | bool | true | 확정 시 극값 동결 여부 |

### 시간대별 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `early_session_start_time` | str | "09:00" | 장초반 시작 시간 |
| `early_session_end_time` | str | "09:30" | 장초반 종료 시간 |
| `early_session_atr_multiplier_max` | float | 8.0 | 장초반 ATR 배수 최대값 |
| `session_min_wave_bars_table` | list | [] | 시간대별 최소 파동 봉수 테이블 |

### 피봇 필터링 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `enable_pivot_filtering` | bool | true | 피봇 필터링 활성화 여부 |
| `pivot_filter_replace_with_extreme` | bool | true | 필터 시 극값으로 대체 여부 |
| `pivot_filter_min_bar_gap` | int | 0 | 피봇 필터 최소 봉 간격 |
| `use_atr_based_filtering` | bool | true | ATR 기반 필터링 사용 여부 |
| `min_wave_atr_ratio` | float | 0.5 | 최소 파동 ATR 비율 |
| `session_min_wave_atr_ratio_table` | list | [] | 시간대별 최소 파동 ATR 비율 테이블 |

### 감쇄 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `decay_start_bars` | int | 30 | 감쇠 시작 봉수 |
| `decay_rate_per_bar` | float | 0.005 | 봉당 감쇠율 |
| `decay_max_pct` | float | 0.3 | 최대 감쇠 퍼센트 |

### 피봇 수집기 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `enable_pivot_collector` | bool | false | 피봇 수집기 활성화 여부 |
| `pivot_collector_max_sequence` | int | 120 | 피봇 수집기 최대 시퀀스 길이 |

---

## 시장 레짐 기반 적응형 파라미터

시장 레짐(MarketRegimeClassifier)을 활용하여 현재 시장 상태에 따라 ZigZag 파라미터를 동적으로 조정할 수 있습니다.

### 레짐별 파라미터 매핑

시장 레짐은 다음 7가지로 분류됩니다:

1. **HIGH_VOL_NO_DIRECTION**: 고변동 + 무방향 (흔들기)
2. **HIGH_VOL_UP**: 고변동 + 상승 (강한 상승 추세)
3. **HIGH_VOL_DOWN**: 고변동 + 하락 (강한 하락 추세)
4. **NORMAL_VOL_UP**: 정상 변동 + 상승 (상승 추세)
5. **NORMAL_VOL_DOWN**: 정상 변동 + 하락 (하락 추세)
6. **NORMAL_VOL_NO_DIRECTION**: 정상 변동 + 무방향 (횡보)
7. **LOW_VOL**: 저변동 (박스권)

### 레짐별 권장 파라미터

#### 추세형 레짐 (HIGH_VOL_UP, NORMAL_VOL_UP)

```json
{
  "atr_multiplier": 1.2,           // 낮은 배수로 민감도 증가
  "confirmation_bars": 1,         // 빠른 확정
  "pivot_threshold_min_pct": 0.2,  // 낮은 임계값
  "enable_pivot_filtering": true,  // 필터링 활성화
  "min_wave_atr_ratio": 0.4       // 낮은 최소 파동 비율
}
```

**특징:**
- 추세 추종을 위해 민감한 파라미터
- 빠른 피봇 확정으로 추세 변화 신속 탐지
- 노이즈 필터링으로 가짜 피봇 방지

#### 하락 추세 레짐 (HIGH_VOL_DOWN, NORMAL_VOL_DOWN)

```json
{
  "atr_multiplier": 1.3,           // 상승보다 약간 높은 배수
  "confirmation_bars": 2,         // 안정적인 확정
  "pivot_threshold_min_pct": 0.25, // 중간 임계값
  "enable_pivot_filtering": true,
  "min_wave_atr_ratio": 0.5
}
```

**특징:**
- 하락 추세는 급격한 반등 가능성 고려
- 상승보다 약간 보수적인 파라미터

#### 횡보 레짐 (NORMAL_VOL_NO_DIRECTION, HIGH_VOL_NO_DIRECTION)

```json
{
  "atr_multiplier": 2.0,           // 높은 배수로 민감도 감소
  "confirmation_bars": 3,         // 느린 확정
  "pivot_threshold_min_pct": 0.5,  // 높은 임계값
  "enable_pivot_filtering": true,
  "min_wave_atr_ratio": 0.8       // 높은 최소 파동 비율
}
```

**특징:**
- 횡보장에서 가짜 피봇 방지
- 높은 임계값으로 의미 있는 파동만 탐지
- 느린 확정으로 잡음 필터링

#### 저변동 레짐 (LOW_VOL)

```json
{
  "atr_multiplier": 2.5,           // 매우 높은 배수
  "confirmation_bars": 3,         // 느린 확정
  "pivot_threshold_min_pct": 0.8,  // 매우 높은 임계값
  "enable_pivot_filtering": true,
  "min_wave_atr_ratio": 1.0       // 매우 높은 최소 파동 비율
}
```

**특징:**
- 박스권에서 미세한 변동 무시
- 의미 있는 돌파만 피봇으로 인식
- 과도한 피봇 생성 방지

---

## config.json 설정 예시

### 기본 설정

```json
{
  "adaptive_indicator": {
    "enabled": true,
    "dual_mode": true,
    "kospi_symbol": "KOSPI 지수",
    "futures_symbol": "KP200 선물",
    "warmup_bars": 15,
    "min_swings_for_ready": 0,
    
    "supertrend": {
      "atr_min_period": 7,
      "atr_max_period": 21,
      "multiplier_min": 1.5,
      "multiplier_max": 4.0,
      "er_period": 10,
      "adx_period": 14,
      "use_bb_correction": true,
      "adx_mult_norm_cap": 60.0,
      "bb_correction_floor": 0.7,
      "bb_correction_ref_pct": 0.05,
      "bb_period": 20,
      "bb_std": 2.0,
      "smooth_period": 3
    },
    
    "zigzag": {
      "atr_multiplier": 1.5,
      "atr_period": 14,
      "pivot_threshold_min_pct": 0.3,
      "pivot_threshold_max_pct": 3.0,
      "major_swing_ratio": 2.0,
      "max_swings": 20,
      "confirmation_bars": 2,
      "cluster_tolerance_pct": 0.3,
      "structure_lookback_swings": 30,
      "structure_points": 4,
      "freeze_on_confirm": true,
      "er_period": 10,
      "atr_multiplier_min": 1.0,
      "atr_multiplier_max": 4.0,
      "min_wave_bars": 1,
      "early_session_start_time": "09:00",
      "early_session_end_time": "09:30",
      "early_session_atr_multiplier_max": 8.0,
      "min_wave_pct": 0.25,
      "confirmation_bars_ranging": 2,
      "confirmation_bars_unknown": 3,
      "structure_majority_threshold": 0.7,
      "decay_start_bars": 30,
      "decay_rate_per_bar": 0.005,
      "decay_max_pct": 0.3,
      "major_wave_ratio": 1.5,
      "major_wave_lookback": 3,
      "der_mismatch_threshold": 0.3,
      "der_mismatch_mult_ratio": 0.7,
      "pivot_lifecycle_log": false,
      "pivot_lifecycle_log_prefix": "",
      "enable_pivot_collector": false,
      "pivot_collector_max_sequence": 120,
      "enable_pivot_filtering": true,
      "pivot_filter_replace_with_extreme": true,
      "pivot_filter_min_bar_gap": 0,
      "use_atr_based_filtering": true,
      "min_wave_atr_ratio": 0.5,
      "cluster_atr_ratio": 0.5,
      "session_min_wave_bars_table": [
        ["09:00", "09:30", 10],
        ["09:30", "10:30", 7],
        ["10:30", "14:30", 4],
        ["14:30", "15:20", 7],
        ["15:20", "15:30", 10]
      ],
      "session_min_wave_atr_ratio_table": [],
      "min_wave_atr_ratio": 0.5
    },
    
    "ranging_filter": {
      "adx_min": 18.0,
      "er_min": 0.2
    },
    
    "adx_confidence_filter": {
      "enabled": true,
      "hold_threshold": 15.0,
      "weak_threshold": 20.0,
      "strong_threshold": 35.0
    }
  }
}
```

### 시장 레질 기반 적응형 설정

```json
{
  "adaptive_indicator": {
    "enabled": true,
    "regime_based_adaptive": true,  // 레짐 기반 적응 활성화
    
    "regime_based_params": {
      "HIGH_VOL_UP": {
        "atr_multiplier": 1.2,
        "confirmation_bars": 1,
        "pivot_threshold_min_pct": 0.2,
        "min_wave_atr_ratio": 0.4
      },
      "NORMAL_VOL_UP": {
        "atr_multiplier": 1.3,
        "confirmation_bars": 1,
        "pivot_threshold_min_pct": 0.25,
        "min_wave_atr_ratio": 0.45
      },
      "HIGH_VOL_DOWN": {
        "atr_multiplier": 1.3,
        "confirmation_bars": 2,
        "pivot_threshold_min_pct": 0.25,
        "min_wave_atr_ratio": 0.5
      },
      "NORMAL_VOL_DOWN": {
        "atr_multiplier": 1.4,
        "confirmation_bars": 2,
        "pivot_threshold_min_pct": 0.3,
        "min_wave_atr_ratio": 0.55
      },
      "NORMAL_VOL_NO_DIRECTION": {
        "atr_multiplier": 2.0,
        "confirmation_bars": 3,
        "pivot_threshold_min_pct": 0.5,
        "min_wave_atr_ratio": 0.8
      },
      "HIGH_VOL_NO_DIRECTION": {
        "atr_multiplier": 2.5,
        "confirmation_bars": 3,
        "pivot_threshold_min_pct": 0.6,
        "min_wave_atr_ratio": 0.9
      },
      "LOW_VOL": {
        "atr_multiplier": 2.5,
        "confirmation_bars": 3,
        "pivot_threshold_min_pct": 0.8,
        "min_wave_atr_ratio": 1.0
      }
    },
    
    // 기본 zigzag 설정 (레짐 인식 실패 시 폴백)
    "zigzag": {
      "atr_multiplier": 1.5,
      "atr_period": 14,
      "confirmation_bars": 2,
      // ... 기타 파라미터
    }
  }
}
```

---

## 실전 활용 가이드

### 추세 추종 전략

**상황:** 명확한 상승/하락 추세가 형성된 시장

**권장 설정:**
```json
{
  "atr_multiplier": 1.2,
  "confirmation_bars": 1,
  "pivot_threshold_min_pct": 0.2,
  "freeze_on_confirm": false
}
```

**특징:**
- 민감한 파라미터로 추세 변화 신속 탐지
- freeze_on_confirm=false로 추세 연장 시 피봇 갱신
- 빠른 진입/탈출 기회 포착

### 횡보장 대응 전략

**상황:** 박스권 또는 횡보 구간

**권장 설정:**
```json
{
  "atr_multiplier": 2.0,
  "confirmation_bars": 3,
  "pivot_threshold_min_pct": 0.5,
  "pivot_filter_replace_with_extreme": true
}
```

**특징:**
- 보수적인 파라미터로 가짜 피봇 방지
- 필터링으로 의미 있는 파동만 유지
- 횡보 돌파 시 명확한 신호 생성

### 장초반 대응 전략

**상황:** 장 시작 30분 (09:00~09:30)

**권장 설정:**
```json
{
  "early_session_start_time": "09:00",
  "early_session_end_time": "09:30",
  "early_session_atr_multiplier_max": 8.0,
  "session_min_wave_bars_table": [
    ["09:00", "09:30", 10]
  ]
}
```

**특징:**
- 장초반 변동성 대응
- 높은 ATR 배수로 과도한 피봇 방지
- 최소 파동 봉수로 안정화

### 장 마감 전 대응 전략

**상황:** 장 마감 10분 (15:20~15:30)

**권장 설정:**
```json
{
  "session_min_wave_bars_table": [
    ["15:20", "15:30", 10]
  ],
  "confirmation_bars_unknown": 3
}
```

**특징:**
- 장 마감 전 안정화
- 높은 최소 파동 봉수로 잡음 필터링
- 느린 확정으로 최종 피봇 신뢰도 확보

---

## 파라미터 튜닝 팁

### 1. ATR 배수 튜닝

**목표:** 시장 변동성에 맞는 민감도 조절

**방법:**
- 낮은 배수 (1.0~1.5): 민감한 피봇 탐지, 추세 추종에 적합
- 중간 배수 (1.5~2.5): 일반적인 시장 환경
- 높은 배수 (2.5~4.0): 횡보장, 저변동 환경

**튜닝 프로세스:**
1. 백테스트로 각 배수별 피봇 개수 확인
2. 수익성과 노이즈 비율 분석
3. 최적 배수 범위 도출
4. 시장 레짐별 세분 튜닝

### 2. 확인 봉수 튜닝

**목표:** 피봇 확정 속도와 신뢰도 균형

**방법:**
- 짧은 확인 (1봉): 빠른 확정, 가짜 피봇 가능성 증가
- 중간 확인 (2~3봉): 균형
- 긴 확인 (4~5봉): 느린 확정, 높은 신뢰도

**튜닝 프로세스:**
1. 추세 환경에서는 짧은 확인 사용
2. 횡보 환경에서는 긴 확인 사용
3. 시간대별 차등 적용

### 3. 최소 파동 비율 튜닝

**목표:** 의미 있는 파동만 탐지

**방법:**
- 낮은 비율 (0.3~0.5): 민감한 탐지
- 중간 비율 (0.5~0.8): 일반적인 환경
- 높은 비율 (0.8~1.5): 보수적인 탐지

**튜닝 프로세스:**
1. ATR 기준 최소 파동 계산
2. 실제 가격 움직임과 비교
3. 거래 전략에 맞는 비율 설정

### 4. 시간대별 튜닝

**목표:** 장 중 시간대 특성 반영

**권장 테이블:**
```json
{
  "session_min_wave_bars_table": [
    ["09:00", "09:30", 10],   // 장초반: 안정화
    ["09:30", "10:30", 7],    // 아침: 중간
    ["10:30", "14:30", 4],    // 점심: 민감
    ["14:30", "15:20", 7],    // 오후: 중간
    ["15:20", "15:30", 10]    // 장마감: 안정화
  ]
}
```

### 5. 감쇠 파라미터 튜닝

**목표:** 장기 횡보 시 민감도 회복

**방법:**
- decay_start_bars: 감쇠 시작 시점 (기본 30봉)
- decay_rate_per_bar: 봉당 감쇠율 (기본 0.5%)
- decay_max_pct: 최대 감쇠 (기본 30%)

**튜닝 프로세스:**
1. 장기 횡구 시 피봇 생성 빈도 확인
2. 감쇠 시작 시점 조절
3. 감쇠율로 민감도 조절

---

## 참고 문서

- [AdaptiveZigZag 구현 문서](./adaptive_zigzag_fixes.md)
- [ZigZag 피봇 원칙](./ZigZag_Pivot_Principles_BugFix.md)
- [시장 레짐 분류기 가이드](./market_regime_classifier.md)
- [ZigZag 파라미터 통합 보고서](./zigzag_param_unification_report.md)

---

## 변경 이력

- **2026-05-09**: 문서 생성
  - ZigZag 파라미터 상세 설명 추가
  - 시장 레짐 기반 적응형 파라미터 추가
  - config.json 설정 예시 추가
  - 실전 활용 가이드 추가
  - 파라미터 튜닝 팁 추가
