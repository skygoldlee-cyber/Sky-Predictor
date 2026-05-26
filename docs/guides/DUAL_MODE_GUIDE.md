# 듀얼 모드 구조 가이드

KOSPI 지수와 KP200 선물의 피봇 분석을 동시에 수행하는 듀얼 모드 구조 가이드.

## 개요

듀얼 모드는 KOSPI 지수와 KP200 선물 각각에 대해 별도의 ZigZag 인스턴스를 운영하여, 두 시장의 피봇 패턴을 동시에 분석하고 비교할 수 있게 합니다.

### 목적

- KOSPI 지수와 KP200 선물의 피봇 패턴 동시 분석
- 두 시장 간의 선후 관계 파악
- GUI에서 플롯 선택을 통해 각 시장의 차트 확인
- 피봇 로그를 구분하여 기록

### 대상 독자

- 시스템 운영자
- 트레이딩 전략 개발자
- 시장 분석가

## 핵심 개념

### 듀얼 모드 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│         AdaptiveIndicatorManager                         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ kospi_zigzag │  │ futures_zigzag│  │   zigzag     │ │
│  │ (KOSPI 지수) │  │ (KP200 선물)  │  │  (기본)      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│         │                  │                  │        │
│         └──────────────────┴──────────────────┘        │
│                            │                           │
│                    update() 메서드                      │
│                            │                           │
│              세 ZigZag 모두 업데이트                    │
└─────────────────────────────────────────────────────────┘
```

### 피봇 근접 감지

KOSPI와 KP200 선물의 피봇 발생 위치가 근접할 때 감지하여 로그와 알림을 제공합니다.

**감지 조건**: 두 피봇의 인덱스 차이가 `max_bars_diff` 이내 (기본 1봉)

**로그 예시**:
```
[PIVOT_PROXIMITY] KOSPI와 KP200 선물 피봇 근접 감지 (차이: 1봉) | KOSPI: H@370.25 (idx:100) | KP200: H@369.50 (idx:101) | ⚠️ 주요 분봉 가능성 높음
```

**텔레그램 알림**: 설정된 경우 텔레그램으로 알림 전송

```json
{
  "adaptive_indicator": {
    "pivot_proximity_alert": {
      "enabled": true,
      "max_bars_diff": 1,
      "telegram_enabled": true
    }
  }
}
```

## 설정

### config.json 설정

```json
{
  "adaptive_indicator": {
    "enabled": true,
    "dual_mode": true,
    "symbol": "KOSPI 지수",
    "kospi_symbol": "KOSPI 지수",
    "futures_symbol": "KP200 선물",
    "zigzag": {
      "atr_multiplier": 1.5,
      "atr_period": 14,
      "pivot_threshold_min_pct": 0.3,
      "pivot_threshold_max_pct": 3.0,
      ...
    },
    "kospi_zigzag": {
      "pivot_lifecycle_log": true,
      "pivot_lifecycle_log_prefix": "[KOSPI]"
    },
    "futures_zigzag": {
      "pivot_lifecycle_log": true,
      "pivot_lifecycle_log_prefix": "[KP200]"
    }
  }
}
```

### 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| dual_mode | bool | false | 듀얼 모드 활성화 여부 |
| kospi_symbol | string | "KOSPI 지수" | KOSPI 지수 심볼 이름 |
| futures_symbol | string | "KP200 선물" | KP200 선물 심볼 이름 |
| kospi_zigzag.pivot_lifecycle_log | bool | true | KOSPI 피봇 로그 활성화 |
| kospi_zigzag.pivot_lifecycle_log_prefix | string | "[KOSPI]" | KOSPI 로그 접두사 |
| futures_zigzag.pivot_lifecycle_log | bool | true | KP200 피봇 로그 활성화 |
| futures_zigzag.pivot_lifecycle_log_prefix | string | "[KP200]" | KP200 로그 접두사 |
| pivot_proximity_alert.enabled | bool | true | 피봇 근접 알림 활성화 |
| pivot_proximity_alert.max_bars_diff | int | 1 | 근접 감지 봉 차이 |
| pivot_proximity_alert.telegram_enabled | bool | true | 텔레그램 알림 활성화 |

## 사용 방법

### 기본 사용법

1. **config.json 설정**: `dual_mode: true`로 설정
2. **시스템 시작**: 듀얼 모드가 자동으로 활성화
3. **피봇 로그 확인**: 두 시장의 피봇 로그가 구분되어 출력

### 피봇 로그 구분

- **KOSPI 로그**: `[KOSPI]` 접두사
- **KP200 로그**: `[KP200]` 접두사

### GUI 플롯 선택

GUI에서 라디오 버튼으로 플롯을 선택할 수 있습니다:

- **KOSPI**: KOSPI 지수 ZigZag 플롯
- **KP200 선물**: KP200 선물 ZigZag 플롯

차트 제목도 선택에 따라 동적으로 변경됩니다:
- KOSPI 선택: "📈 KOSPI 지수 차트"
- KP200 선물 선택: "📈 KP200 선물 차트"

### 프로그래밍 방식 사용

```python
from kospi_indicators import AdaptiveIndicatorManager, IndicatorManagerConfig

# 듀얼 모드 설정
config = IndicatorManagerConfig(
    dual_mode=True,
    kospi_symbol="KOSPI 지수",
    futures_symbol="KP200 선물",
)

# 매니저 초기화
mgr = AdaptiveIndicatorManager(config=config)

# 업데이트 (세 ZigZag 모두 업데이트)
result = mgr.update(high, low, close)

# 각 ZigZag 상태 확인
kospi_state = result.get("kospi_zigzag_state")
futures_state = result.get("futures_zigzag_state")
```

## 로그 예시

### KOSPI 피봇 로그

```
[ZZ_PIVOT][KOSPI] 후보등록 bar=100 type=H price=370.25
[ZZ_PIVOT][KOSPI] 후보상태 bar=102 type=H rem=2 urgency=0.3 age=0.1 prob=0.75
[ZZ_PIVOT][KOSPI] 확정 bar=104 type=H price=370.50
```

### KP200 피봇 로그

```
[ZZ_PIVOT][KP200] 후보등록 bar=100 type=L price=369.50
[ZZ_PIVOT][KP200] 후보상태 bar=102 type=L rem=1 urgency=0.5 age=0.2 prob=0.68
[ZZ_PIVOT][KP200] 취소 bar=103 type=L reason=breakout
```

## 주의사항

### 일반적인 주의사항

1. **메모리 사용**: 세 개의 ZigZag 인스턴스가 실행되므로 메모리 사용량이 증가합니다.
2. **로그 양**: 두 시장의 로그가 모두 출력되므로 로그 양이 2배로 증가합니다.
3. **데이터 동기화**: 두 시장의 데이터는 별도로 수집되므로 타이밍 차이가 있을 수 있습니다.

### 에러 처리

- 듀얼 모드가 비활성화된 경우 `kospi_zigzag`와 `futures_zigzag`는 `None`입니다.
- GUI에서 플롯 선택 시 해당 ZigZag 인스턴스가 없으면 기본 ZigZag를 사용합니다.

```python
if mgr.config.dual_mode:
    if mgr.kospi_zigzag is not None:
        # KOSPI ZigZag 사용
        pass
    elif mgr.zigzag is not None:
        # 기본 ZigZag fallback
        pass
```

## 관련 문서

- [적응형 지표 가이드](./ADAPTIVE_INDICATOR_GUIDE.md)
- [피봇 수집기 가이드](./PIVOT_COLLECTOR_GUIDE.md)
- [피봇 예측 ML 알고리즘 가이드](./PIVOT_ML_ALGORITHM_GUIDE.md)

---

**문서 버전**: 1.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-25
