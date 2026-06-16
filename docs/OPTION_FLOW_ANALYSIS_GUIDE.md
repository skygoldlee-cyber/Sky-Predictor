# 옵션 흐름 (Option Flow) 분석 시스템 가이드

옵션 데이터를 분석하여 시장 심리와 방향성을 파악하는 시스템 가이드.

## 개요

옵션 흐름 분석 시스템은 옵션 거래 데이터를 실시간으로 수집하고 분석하여 시장 참여자의 심리와 방향성을 파악합니다. ITM/OTM 구독, OI 분석, PCR 계산 등을 수행합니다.

### 목적

- 옵션 거래 데이터 실시간 수집
- 시장 심리 분석
- 방향성 지표 계산
- 알림 조건 감지

### 대상 독자

- 옵션 트레이더
- 시장 분석가
- 시스템 운영자

## 핵심 개념

### 옵션 흐름 분석 흐름

```
┌─────────────────────────────────────────────────────────┐
│              옵션 흐름 분석 프로세스                       │
│                                                          │
│  1. 옵션 심볼 구독 (ITM/OTM)                            │
│     ↓                                                   │
│  2. 분봉 OHLCV 수집                                     │
│     ↓                                                   │
│  3. 피처 계산 (OI, PCR, Volume 등)                     │
│     ↓                                                   │
│  4. 지표 계산 (프리미엄 블리드, 패리티 이탈)           │
│     ↓                                                   │
│  5. 알림 조건 감지                                     │
│     ↓                                                   │
│  6. 텔레그램 알림 전송                                 │
└─────────────────────────────────────────────────────────┘
```

### 주요 지표

| 지표 | 설명 | 해석 |
|------|------|------|
| PCR (Put-Call Ratio) | 풋/콜 비율 | > 1: 하락 기대, < 1: 상승 기대 |
| OI (Open Interest) | 미결제 약정 | 증가: 포지션 증가 |
| Volume Imbalance | 거래량 불균형 | 콜 우세: 상승, 풋 우세: 하락 |
| Premium Bleed | 프리미엄 붕괴 | 시간 경과에 따른 프리미엄 감소 |
| Parity Divergence | 패리티 이탈 | 콜-풋 가격 불균형 |

## 설정

### config.json 설정

```json
{
  "options_subscription": {
    "itm": 10,
    "otm_open_min": 0.5,
    "max_otm_calls": 30,
    "max_otm_puts": 40,
    "wait_sec": 2,
    "preopen_oh0_window": 10,
    "oi_itm_count": 10,
    "oi_otm_count": 10,
    "oi_rebalance_interval_sec": 60
  },
  "telegram": {
    "enabled": true,
    "option_flow_status_enabled": true,
    "option_flow_status_cooldown_sec": 300,
    "option_flow_interp_sr_warn": 1.5,
    "option_flow_interp_sr_hot": 2.0,
    "option_flow_interp_pt_low": 0.008,
    "option_flow_interp_pt_high": 0.03
  }
}
```

### 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 | 권장 범위 |
|----------|------|--------|------|-----------|
| itm | int | 10 | ITM 옵션 개수 | 5 ~ 20 |
| otm_open_min | float | 0.5 | OTM 오픈 최소 거리 (%) | 0.3 ~ 1.0 |
| max_otm_calls | int | 30 | 최대 OTM 콜 개수 | 20 ~ 50 |
| max_otm_puts | int | 40 | 최대 OTM 풋 개수 | 30 ~ 60 |
| option_flow_status_enabled | bool | true | 옵션 흐름 알림 활성화 | true/false |
| option_flow_status_cooldown_sec | int | 300 | 쿨다운 (초) | 180 ~ 600 |

## 사용 방법

### 기본 사용법

```python
from ebestapi.live import EbestLive

# eBest 연결
ebest = EbestLive()

# 옵션 구독
ebest.subscribe_options(
    itm=10,
    otm_open_min=0.5,
    max_otm_calls=30,
    max_otm_puts=40,
)

# 데이터 수집
while True:
    data = ebest.get_option_minute_data()
    # 분석 수행
```

### 피처 계산

```python
from prediction.option_features import build_option_snapshot

# 옵션 스냅샷 빌드
snapshot = build_option_snapshot(
    option_data=option_data,
    current_price=current_price,
)

# 피처 확인
pcr_v = snapshot.get("pcr_v")  # Volume 기반 PCR
pcr_oi = snapshot.get("pcr_oi")  # OI 기반 PCR
oi_total = snapshot.get("oi_total")  # 총 OI
```

### 프리미엄 블리드 지표

```python
from prediction.option_flow_features import compute_premium_bleed

# 프리미엄 블리드 계산
bleed = compute_premium_bleed(
    option_data=option_data,
    time_to_expiry=days_to_expiry,
)

# 해석
if bleed > threshold:
    # 프리미염 붕괴 감지
    send_alert("Premium Bleed Detected")
```

### 콜-풋 패리티 이탈 지표

```python
from prediction.parity_features import compute_parity_divergence

# 패리티 이탈 계산
divergence = compute_parity_divergence(
    call_price=call_price,
    put_price=put_price,
    strike=strike,
    current_price=current_price,
)

# 해석
if divergence > threshold:
    # 패리티 이탈 감지
    send_alert("Parity Divergence Detected")
```

## 알림 조건

### 옵션 흐름 상태 알림

```json
{
  "telegram": {
    "option_flow_status_enabled": true,
    "option_flow_interp_sr_warn": 1.5,
    "option_flow_interp_sr_hot": 2.0,
    "option_flow_interp_pt_low": 0.008,
    "option_flow_interp_pt_high": 0.03
  }
}
```

**알림 조건**:
- SR (Support/Resistance) 경고: `sr > 1.5`
- SR 핫: `sr > 2.0`
- PT (Premium Time) 낮음: `pt < 0.008`
- PT 높음: `pt > 0.03`

### 알림 메시지 포맷

```
📊 옵션 흐름 상태

PCR-V: 1.2 (하락 기대)
PCR-OI: 1.1 (하락 기대)
OI: 150,000 (증가)
Volume Imbalance: 콜 우세

⚠️ 프리미엄 블리드 감지
```

## 주의사항

### 일반적인 주의사항

1. **데이터 지연**: 옵션 데이터는 실시간이지만 지연이 있을 수 있습니다.
2. **유동성**: 일부 옵션은 유동성이 낮아 신뢰도가 낮을 수 있습니다.
3. **만기 영향**: 만기 주에는 옵션 행위가 비정상적일 수 있습니다.
4. **알림 빈도**: 쿨다운 설정을 통해 알림 빈도를 조절하세요.

### 에러 처리

```python
try:
    snapshot = build_option_snapshot(option_data)
except DataInsufficientError:
    # 데이터 부족 시 스킵
    pass
except CalculationError:
    # 계산 오류 시 로그
    log_error("Option feature calculation failed")
```

## 관련 문서

- [프리미엄 블리드 설계](./premium_bleed_design.md)
- [콜-풋 패리티 이탈 설계](./call_put_parity_divergence_design.md)
- [텔레그램 시스템 가이드](./telegram.md)

---

**문서 버전**: 1.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-25
