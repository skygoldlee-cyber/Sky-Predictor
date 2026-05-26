# 포지션 사이징 가이드

> **동적 포지션 사이징 기능 사용 가이드**
> 작성일: 2026-04-26

---

## 개요

본 가이드는 TradeExecutionGate의 포지션 사이징 기능을 사용하여 자본, 리스크, 시장 변동성에 따라 포지션 사이즈를 동적으로 계산하는 방법을 설명합니다.

## 목차

1. 개요
2. 지원 사이징 방법
3. 설정 파라미터
4. 사용 예시
5. 각 방법 상세 설명
6. 권장 설정

---

## 2. 지원 사이징 방법

### 2.1 Fixed Fractional (고정 비율)

자본의 일정 비율을 항상 투자하는 가장 간단한 방법입니다.

**장점**:
- 구현 간단
- 일관된 투자 비율 유지

**단점**:
- 시장 상황 고려 안 함
- 리스크 조절 불가

**적용 상황**:
- 초보자 추천
- 안정적인 시장

### 2.2 Kelly Criterion (켈리 기준)

수학적으로 최적의 성장률을 제공하는 포지션 사이징 방법입니다.

**공식**:
```
Kelly % = (bp - q) / b
```
- b = 평균 승리 / 평균 패배 (승패비)
- p = 승률
- q = 패배율 (1 - p)

**장점**:
- 최적 성장률 제공
- 장기 수익 최적화

**단점**:
- 과거 데이터 필요
- 과도한 레버리지 가능

**적용 상황**:
- 충분한 거래 이력 있는 경우
- 공격적 투자자

### 2.3 Risk Parity (리스크 패리티)

거래당 리스크를 일정하게 유지하는 방법입니다.

**원리**:
- 거래당 리스크 금액 = 자본 × risk_per_trade
- 포지션 사이즈 = 리스크 금액 / (진입가 - 손절가)

**장점**:
- 리스크 일관성 유지
- 큰 손실 방지

**단점**:
- 손절가 필요
- 낮은 변동성 시 과도한 사이즈

**적용 상황**:
- 리스크 관리 중시
- 변동성 큰 시장

### 2.4 Volatility-based (변동성 기반)

시장 변동성에 따라 포지션 사이즈를 조절합니다.

**원리**:
- 변동성 높으면 사이즈 줄이기
- 변동성 낮으면 사이즈 늘리기

**장점**:
- 시장 상황 반영
- 변동성 적응

**단점**:
- ATR 데이터 필요
- 지연 가능성

**적용 상황**:
- 변동성 변화 큰 시장
- ATR 데이터 있는 경우

---

## 3. 설정 파라미터

### 3.1 공통 파라미터

| 파라미터 | 설명 | 기본값 | 범위 |
|----------|------|--------|------|
| `sizing_method` | 사이징 방법 | fixed_fractional | - |
| `sizing_max_position` | 최대 포지션 비율 | 0.3 | 0.0~1.0 |
| `sizing_min_position` | 최소 포지션 비율 | 0.05 | 0.0~1.0 |

### 3.2 Fixed Fractional 파라미터

| 파라미터 | 설명 | 기본값 | 범위 |
|----------|------|--------|------|
| `sizing_fixed_fraction` | 투자 비율 | 0.95 | 0.0~1.0 |

### 3.3 Kelly Criterion 파라미터

| 파라미터 | 설명 | 기본값 | 범위 |
|----------|------|--------|------|
| `sizing_kelly_fraction` | 켈리 비율 (보수적 적용) | 0.5 | 0.0~1.0 |
| `sizing_min_kelly` | 최소 켈리 비율 | 0.1 | 0.0~1.0 |
| `sizing_max_kelly` | 최대 켈리 비율 | 0.25 | 0.0~1.0 |

### 3.4 Risk Parity 파라미터

| 파라미터 | 설명 | 기본값 | 범위 |
|----------|------|--------|------|
| `sizing_risk_per_trade` | 거래당 리스크 비율 | 0.02 | 0.0~1.0 |
| `sizing_stop_loss_pt` | 손절 포인트 | 1.0 | 0.0~ |

### 3.5 Volatility-based 파라미터

| 파라미터 | 설명 | 기본값 | 범위 |
|----------|------|--------|------|
| `sizing_atr_multiplier` | ATR 멀티플라이어 | 2.0 | 0.0~ |
| `sizing_volatility_target` | 목표 변동성 | 0.15 | 0.0~ |

---

## 4. 사용 예시

### 4.1 Fixed Fractional 사용

```json
{
  "trade_gate": {
    "sizing_method": "fixed_fractional",
    "sizing_fixed_fraction": 0.95,
    "sizing_max_position": 0.3,
    "sizing_min_position": 0.05
  }
}
```

### 4.2 Kelly Criterion 사용

```json
{
  "trade_gate": {
    "sizing_method": "kelly_criterion",
    "sizing_kelly_fraction": 0.5,
    "sizing_min_kelly": 0.1,
    "sizing_max_kelly": 0.25
  }
}
```

### 4.3 Risk Parity 사용

```json
{
  "trade_gate": {
    "sizing_method": "risk_parity",
    "sizing_risk_per_trade": 0.02,
    "sizing_stop_loss_pt": 1.0
  }
}
```

### 4.4 Volatility-based 사용

```json
{
  "trade_gate": {
    "sizing_method": "volatility_based",
    "sizing_fixed_fraction": 0.95,
    "sizing_atr_multiplier": 2.0,
    "sizing_volatility_target": 0.15
  }
}
```

---

## 5. 각 방법 상세 설명

### 5.1 Fixed Fractional 상세

**계산 방법**:
```
사용 자본 = 총 자본 × fixed_fraction
포지션 사이즈 = 사용 자본 / 진입 가격
```

**제한**:
- 최대: 총 자본 × max_position_size
- 최소: 총 자본 × min_position_size

**예시**:
```
자본: 1,000,000원
fixed_fraction: 0.95
진입가: 380원

사용 자본: 950,000원
포지션 사이즈: 2,500 계약
```

### 5.2 Kelly Criterion 상세

**계산 방법**:
```
b = 평균 승리 / 평균 패배
p = 승률
q = 1 - p
Kelly % = (b × p - q) / b
적용 Kelly % = Kelly % × kelly_fraction
적용 Kelly % = clamp(적용 Kelly %, min_kelly, max_kelly)
```

**제한**:
- 최대/최소 켈리 비율
- 최대/최소 포지션 사이즈

**예시**:
```
승률: 60%
평균 승리: 2.0pt
평균 패배: 1.0pt

b = 2.0 / 1.0 = 2.0
Kelly % = (2.0 × 0.6 - 0.4) / 2.0 = 0.4
적용 Kelly % = 0.4 × 0.5 = 0.2
최대 제한: 0.25 → 0.2 사용
```

### 5.3 Risk Parity 상세

**계산 방법**:
```
리스크 금액 = 총 자본 × risk_per_trade
1계약당 리스크 = |진입가 - 손절가|
포지션 사이즈 = 리스크 금액 / 1계약당 리스크
```

**제한**:
- 최대/최소 포지션 사이즈

**예시**:
```
자본: 1,000,000원
risk_per_trade: 0.02
진입가: 380원
손절가: 379원

리스크 금액: 20,000원
1계약당 리스크: 1원
포지션 사이즈: 20,000 계약
```

### 5.4 Volatility-based 상세

**계산 방법**:
```
현재 변동성 = ATR / 진입가
사이징 팩터 = 목표 변동성 / 현재 변동성
사이징 팩터 = clamp(사이징 팩터, 0.5, 2.0)
사용 자본 = 총 자본 × fixed_fraction × 사이징 팩터
포지션 사이즈 = 사용 자본 / 진입가
```

**제한**:
- 최대/최소 포지션 사이즈

**예시**:
```
자본: 1,000,000원
fixed_fraction: 0.95
진입가: 380원
ATR: 2.0

현재 변동성: 2.0 / 380 = 0.0053
사이징 팩터: 0.15 / 0.0053 = 28.3 → 2.0 (제한)
사용 자본: 1,000,000 × 0.95 × 2.0 = 1,900,000
포지션 사이즈: 5,000 계약
```

---

## 6. 권장 설정

### 6.1 초보자 추천

```json
{
  "trade_gate": {
    "sizing_method": "fixed_fractional",
    "sizing_fixed_fraction": 0.5,
    "sizing_max_position": 0.2,
    "sizing_min_position": 0.05
  }
}
```

**이유**:
- 보수적 비율 (50%)
- 최대 제한 (20%)
- 단순하고 이해하기 쉬움

### 6.2 중급자 추천

```json
{
  "trade_gate": {
    "sizing_method": "risk_parity",
    "sizing_risk_per_trade": 0.02,
    "sizing_stop_loss_pt": 1.0,
    "sizing_max_position": 0.25,
    "sizing_min_position": 0.05
  }
}
```

**이유**:
- 리스크 일관성 유지
- 거래당 리스크 제한 (2%)
- 적당한 제한

### 6.3 고급자 추천

```json
{
  "trade_gate": {
    "sizing_method": "kelly_criterion",
    "sizing_kelly_fraction": 0.5,
    "sizing_min_kelly": 0.05,
    "sizing_max_kelly": 0.15,
    "sizing_max_position": 0.2,
    "sizing_min_position": 0.05
  }
}
```

**이유**:
- 최적 성장 추구
- 보수적 적용 (50%)
- 엄격한 제한

### 6.4 변동성 적응형 추천

```json
{
  "trade_gate": {
    "sizing_method": "volatility_based",
    "sizing_fixed_fraction": 0.8,
    "sizing_atr_multiplier": 2.0,
    "sizing_volatility_target": 0.15,
    "sizing_max_position": 0.3,
    "sizing_min_position": 0.05
  }
}
```

**이유**:
- 시장 상황 적응
- 변동성 높을 때 사이즈 줄이기
- 적당한 기본 비율

---

## 7. 주의사항

### 7.1 과도한 레버리지

- Kelly Criterion은 과도한 레버리지 가능
- 항상 보수적 적용 (kelly_fraction < 1.0)
- 최대/최소 제한 필수

### 7.2 데이터 충분성

- Kelly Criterion: 최소 50거래 이상 필요
- Risk Parity: 정확한 손절가 필요
- Volatility-based: 정확한 ATR 필요

### 7.3 시장 상황

- 변동성 급증 시 사이즈 줄이기
- 불확실한 시장 보수적 설정
- 백테스트로 검증 필수

### 7.4 리스크 관리

- 거래당 리스크 2% 이하 권장
- 일일 최대 손실 5% 이하 권장
- 총 포지션 30% 이하 권장

---

## 8. 성과 분석

### 8.1 평가 지표

- 총 수익률
- Sharpe Ratio
- 최대 낙폭 (MDD)
- 승률
- 평균 수익/거래

### 8.2 백테스트

- 과거 데이터로 각 방법 테스트
- 파라미터 튜닝 (scripts/parameter_tuner.py)
- 월간/분기별 재평가

### 8.3 모니터링

- 실제 성과 vs 기대 성과
- 리스크 노출 모니터링
- 설정 주기적 검토

---

**문서 버전**: 1.0  
**최종 갱신**: 2026-04-26
