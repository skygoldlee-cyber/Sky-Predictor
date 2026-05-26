# 가드레일 시스템 가이드

트레이딩 리스크를 관리하기 위한 가드레일 시스템 가이드.

## 개요

가드레일 시스템은 트레이딩 신호를 생성하기 전에 다양한 리스크 요소를 검사하여 안전한 트레이딩을 보장합니다. Basis 스프레드, ATM 스프레드, 유동성, IV 등을 검사합니다.

### 목적

- 리스크 관리
- 안전한 트레이딩 보장
- 비정상 상황 감지
- 손실 방지

### 대상 독자

- 트레이더
- 리스크 관리자
- 시스템 운영자

## 핵심 개념

### 가드레일 검사 흐름

```
┌─────────────────────────────────────────────────────────┐
│                   가드레일 검사 프로세스                   │
│                                                          │
│  1. 신호 생성                                           │
│     ↓                                                   │
│  2. Basis 스프레드 검사                                │
│     ↓                                                   │
│  3. ATM 스프레드 검사                                  │
│     ↓                                                   │
│  4. 유동성 검사                                       │
│     ↓                                                   │
│  5. IV 검사                                            │
│     ↓                                                   │
│  6. Gamma 검사 (선택적)                               │
│     ↓                                                   │
│  7. 최종 신호 승인/거부                               │
└─────────────────────────────────────────────────────────┘
```

### 가드레일 유형

| 가드레일 | 설명 | 위반 시 동작 |
|----------|------|--------------|
| guard_basis_hold_thr | Basis 스프레드 임계값 | HOLD |
| guard_basis_downgrade_thr | Basis 다운그레이드 임계값 | 신뢰도 하향 |
| guard_atm_spread_pct_thr | ATM 스프레드 임계값 (%) | HOLD |
| guard_atm_liq_log_thr | ATM 유동성 임계값 (log) | HOLD |
| iv_dynamic_enabled | IV 동적 가드레일 | IV 기반 HOLD |
| iv_target_mult | IV 목표 배수 | IV 목표 계산 |
| iv_stop_mult | IV 손절 배수 | IV 손절 계산 |

## 설정

### config.json 설정

```json
{
  "prediction": {
    "guard_basis_hold_thr": 2.5,
    "guard_basis_downgrade_thr": 1.5,
    "guard_atm_spread_pct_thr": 1.5,
    "guard_atm_liq_log_thr": 2.0
  },
  "trade_gate": {
    "iv_dynamic_enabled": true,
    "iv_target_mult": 0.5,
    "iv_stop_mult": 0.25,
    "iv_target_min": 1.5,
    "iv_target_max": 5.0,
    "iv_stop_min": 0.75,
    "iv_stop_max": 2.5,
    "gamma_gate_enabled": false
  }
}
```

### 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| guard_basis_hold_thr | float | 2.5 | Basis HOLD 임계값 (pt) | 2.0 ~ 3.5 |
| guard_basis_downgrade_thr | float | 1.5 | Basis 다운그레이드 임계값 (pt) | 1.0 ~ 2.5 |
| guard_atm_spread_pct_thr | float | 1.5 | ATM 스프레드 임계값 (%) | 1.0 ~ 2.5 |
| guard_atm_liq_log_thr | float | 2.0 | ATM 유동성 임계값 (log) | 1.5 ~ 3.0 |
| iv_dynamic_enabled | bool | true | IV 동적 가드레일 | true/false |
| iv_target_mult | float | 0.5 | IV 목표 배수 | 0.3 ~ 0.7 |
| iv_stop_mult | float | 0.25 | IV 손절 배수 | 0.15 ~ 0.4 |
| iv_target_min | float | 1.5 | IV 목표 최소 | 1.0 ~ 2.0 |
| iv_target_max | float | 5.0 | IV 목표 최대 | 3.0 ~ 7.0 |
| iv_stop_min | float | 0.75 | IV 손절 최소 | 0.5 ~ 1.0 |
| iv_stop_max | float | 2.5 | IV 손절 최대 | 1.5 ~ 3.5 |
| gamma_gate_enabled | bool | false | Gamma 가드레일 | true/false |

## 사용 방법

### 기본 사용법

```python
from prediction.pipeline import PredictionPipeline

# 파이프라인 초기화 (가드레일 활성화)
pipeline = PredictionPipeline(
    guard_basis_hold_thr=2.5,
    guard_atm_spread_pct_thr=1.5,
    guard_atm_liq_log_thr=2.0,
)

# 예측 실행 (가드레일 검사 포함)
result = pipeline.predict(
    df=minute_df,
    now_dt=current_time,
    current_price=current_price,
)

# 가드레일 결과 확인
guardrail_violation = result.get("guardrail_violation")
if guardrail_violation:
    signal = "HOLD"  # 가드레일 위반 시 HOLD
```

### Basis 스프레드 검사

```python
from prediction.guardrail_mixin import GuardrailMixin

# Basis 스프레드 계산
basis_spread = futures_price - spot_price

# 검사
if abs(basis_spread) > guard_basis_hold_thr:
    # HOLD
    signal = "HOLD"
elif abs(basis_spread) > guard_basis_downgrade_thr:
    # 신뢰도 다운그레이드
    confidence = downgrade_confidence(confidence)
```

### ATM 스프레드 검사

```python
# ATM 스프레드 계산
atm_spread_pct = (ask_price - bid_price) / mid_price * 100

# 검사
if atm_spread_pct > guard_atm_spread_pct_thr:
    # HOLD
    signal = "HOLD"
```

### 유동성 검사

```python
# ATM 유동성 계산
atm_liquidity = log(atm_volume)

# 검사
if atm_liquidity < guard_atm_liq_log_thr:
    # HOLD
    signal = "HOLD"
```

### IV 동적 가드레일

```python
# IV 기반 목표/손절 계산
current_iv = get_current_iv()
iv_target = current_iv * iv_target_mult
iv_stop = current_iv * iv_stop_mult

# 검사
if iv_target < iv_target_min:
    iv_target = iv_target_min
elif iv_target > iv_target_max:
    iv_target = iv_target_max

if iv_stop < iv_stop_min:
    iv_stop = iv_stop_min
elif iv_stop > iv_stop_max:
    iv_stop = iv_stop_max
```

## 가드레일 트리거 조건

### Basis 스프레드

- **HOLD**: `abs(basis_spread) > guard_basis_hold_thr`
- **다운그레이드**: `abs(basis_spread) > guard_basis_downgrade_thr`

### ATM 스프레드

- **HOLD**: `atm_spread_pct > guard_atm_spread_pct_thr`

### 유동성

- **HOLD**: `log(atm_volume) < guard_atm_liq_log_thr`

### IV

- **동적 목표/손절**: IV 수준에 따라 동적으로 계산

## 주의사항

### 일반적인 주의사항

1. **임계값 설정**: 시장 상황에 따라 임계값 조절 필요
2. **과도한 보수성**: 임계값이 너무 낮으면 트레이딩 기회 상실
3. **데이터 지연**: 실시간 데이터 지연 고려
4. **다중 가드레일**: 여러 가드레일이 동시에 위반될 수 있음

### 에러 처리

```python
try:
    guardrail_result = check_guardrails(signal, market_data)
except DataInsufficientError:
    # 데이터 부족 시 HOLD
    signal = "HOLD"
except CalculationError:
    # 계산 오류 시 HOLD
    signal = "HOLD"
```

## 관련 문서

- [트레이딩 시그널 생성 가이드](./TRADING_SIGNAL_GENERATION_GUIDE.md)
- [config.json 참조 가이드](./CONFIG_REFERENCE_GUIDE.md)
- [예측 알고리즘](./Prediction_Algorithm.md)

---

**문서 버전**: 1.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-25
