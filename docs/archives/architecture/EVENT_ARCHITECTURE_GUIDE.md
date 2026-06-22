# 이벤트 기반 아키텍처 가이드

> **느슨한 결합을 위한 이벤트 기반 시스템 구현 가이드**
> 작성일: 2026-04-26

---

## 개요

본 가이드는 이벤트 기반 아키텍처를 사용하여 시스템 컴포넌트 간 느슨한 결합을 구현하는 방법을 설명합니다. 이벤트 버스, 이벤트 정의, 핸들러를 통해 확장 가능하고 유지보수하기 쉬운 시스템을 구축할 수 있습니다.

## 목차

1. 개요
2. 아키텍처 개념
3. 이벤트 버스
4. 이벤트 정의
5. 이벤트 핸들러
6. TradeExecutionGate 통합
7. 사용 예시
8. 권장 사항

---

## 2. 아키텍처 개념

### 2.1 이벤트 기반 아키텍처란?

이벤트 기반 아키텍처는 시스템 컴포넌트 간의 통신을 이벤트 발행(Publish)과 구독(Subscribe) 패턴으로 구현합니다.

**장점**:
- 느슨한 결합: 컴포넌트가 서로 직접 참조하지 않음
- 확장성: 새로운 핸들러를 쉽게 추가 가능
- 테스트 용이성: 각 컴포넌트를 독립적으로 테스트 가능
- 유연성: 런타임에 동적으로 핸들러 추가/제거 가능

**구성 요소**:
- **Event**: 발생하는 사항을 나타내는 데이터 객체
- **EventBus**: 이벤트를 발행하고 구독자에게 전달하는 중앙 허브
- **Handler**: 특정 이벤트를 처리하는 함수/메서드

### 2.2 발행-구독 패턴

```
[발행자] → 이벤트 발행 → [이벤트 버스] → 이벤트 전달 → [구독자 1]
                                                   → [구독자 2]
                                                   → [구독자 3]
```

- 발행자는 이벤트 버스만 알면 됨 (구독자 몰라도 됨)
- 구독자는 이벤트 버스만 알면 됨 (발행자 몰라도 됨)
- 한 이벤트를 여러 구독자가 처리 가능

---

## 3. 이벤트 버스

### 3.1 기능

**주요 기능**:
- 이벤트 구독/해제
- 이벤트 발행 (동기/비동기)
- 이벤트 통계
- 스레드 안전성 보장
- 로깅 제어

### 3.2 사용법

```python
from events import EventBus, TradeEntryEvent
from datetime import datetime

# 이벤트 버스 생성
bus = EventBus()

# 핸들러 등록
@bus.subscribe(TradeEntryEvent)
def handle_entry(event: TradeEntryEvent):
    print(f"진입: {event.side} @ {event.price}")

# 이벤트 발행
event = TradeEntryEvent(
    side="LONG",
    price=380.0,
    size=100.0,
    confidence="HIGH",
    prob=0.75,
    slot="A",
    signal="BUY"
)
bus.publish(event)

# 통계 확인
stats = bus.get_stats()
print(f"TradeEntryEvent 발행 횟수: {stats.get('TradeEntryEvent', 0)}")
```

### 3.3 비동기 발행

```python
# 별도 스레드에서 비동기로 발행
bus.publish_async(event)
```

### 3.4 전역 이벤트 버스

```python
from events import get_event_bus, set_event_bus

# 전역 이벤트 버스 가져오기
bus = get_event_bus()

# 전역 이벤트 버스 설정
custom_bus = EventBus()
set_event_bus(custom_bus)
```

---

## 4. 이벤트 정의

### 4.1 기본 이벤트

모든 이벤트는 `Event` 기본 클래스를 상속받습니다.

```python
from events import Event
from datetime import datetime

class Event:
    def __init__(self, timestamp: datetime = None, event_id: str = ""):
        self.timestamp = timestamp if timestamp is not None else datetime.now()
        self.event_id = event_id if event_id else f"{self.__class__.__name__}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
```

### 4.2 거래 이벤트

#### TradeEntryEvent (진입 이벤트)
```python
TradeEntryEvent(
    side="LONG",           # "LONG" or "SHORT"
    price=380.0,           # 진입 가격
    size=100.0,            # 포지션 사이즈
    confidence="HIGH",     # 신뢰도
    prob=0.75,             # 확률
    slot="A",              # 슬롯
    signal="BUY",          # 신호
    timestamp=datetime.now(),
    event_id=""
)
```

#### TradeExitEvent (청산 이벤트)
```python
TradeExitEvent(
    side="LONG",
    entry_price=380.0,
    exit_price=382.0,
    size=100.0,
    pnl=2.0,              # 포인트
    pnl_pct=0.526,        # 퍼센트
    reason="TARGET_PROFIT", # 청산 사유
    hold_minutes=30.0,
    slot="A",
    timestamp=datetime.now(),
    event_id=""
)
```

#### SignalEvent (신호 이벤트)
```python
SignalEvent(
    signal="BUY",
    confidence="HIGH",
    prob=0.75,
    price=380.0,
    timestamp=datetime.now(),
    event_id=""
)
```

#### RiskLimitEvent (리스크 한도 이벤트)
```python
RiskLimitEvent(
    limit_type="CONSECUTIVE_LOSS",
    current_value=3.0,
    limit_value=3.0,
    action="BLOCK_ENTRY",
    timestamp=datetime.now(),
    event_id=""
)
```

### 4.3 시스템 이벤트

#### ErrorEvent (에러 이벤트)
```python
ErrorEvent(
    error_type="VALIDATION_ERROR",
    message="잘못된 파라미터",
    context={"param": "value"},
    timestamp=datetime.now(),
    event_id=""
)
```

#### SystemEvent (시스템 이벤트)
```python
SystemEvent(
    event_type="STARTUP",
    message="시스템 시작",
    data={"version": "1.0"},
    timestamp=datetime.now(),
    event_id=""
)
```

#### AlertEvent (알림 이벤트)
```python
AlertEvent(
    alert_type="WARNING",
    message="리스크 한도 근접",
    data={"current": 0.95, "limit": 1.0},
    timestamp=datetime.now(),
    event_id=""
)
```

---

## 5. 이벤트 핸들러

### 5.1 LoggingHandler

모든 이벤트를 로깅하는 핸들러입니다.

**기능**:
- 진입/청산/신호 이벤트 로깅
- 리스크 한도 이벤트 로깅
- 에러/시스템 이벤트 로깅
- 성과 이벤트 로깅
- 알림 이벤트 로깅

**사용법**:
```python
from events import EventBus, LoggingHandler

bus = EventBus()
handler = LoggingHandler(bus)
```

### 5.2 MetricsHandler

이벤트 메트릭을 수집하는 핸들러입니다.

**기능**:
- 총 거래 수
- 진입/청산/신호 수
- 슬롯별 진입 수
- 청산 사유별 수
- 신호 타입별 수

**사용법**:
```python
from events import EventBus, MetricsHandler

bus = EventBus()
handler = MetricsHandler(bus)

# 메트릭 확인
metrics = handler.get_metrics()
print(f"총 거래 수: {metrics['total_trades']}")
print(f"슬롯 A 진입: {metrics['entries_by_slot']['A']}")

# 메트릭 초기화
handler.reset_metrics()
```

### 5.3 TelegramNotifierHandler

텔레그램 알림을 전송하는 핸들러입니다.

**기능**:
- 진입/청산 이벤트 알림
- 리스크 한도 이벤트 알림
- WARNING 이상 알림 이벤트 전송

**사용법**:
```python
from events import EventBus, TelegramNotifierHandler

bus = EventBus()
notifier = TelegramNotifier()  # 기존 노티파이어
handler = TelegramNotifierHandler(bus, notifier=notifier)
```

### 5.4 커스텀 핸들러

```python
from events import EventBus, TradeEntryEvent

bus = EventBus()

@bus.subscribe(TradeEntryEvent)
def custom_entry_handler(event: TradeEntryEvent):
    # 커스텀 로직
    if event.confidence == "HIGH":
        print("높은 신뢰도 진입!")
        # 추가 로직 수행
```

---

## 6. TradeExecutionGate 통합

### 6.1 이벤트 버스 전달

```python
from trading.gate import TradeExecutionGate
from events import EventBus

# 이벤트 버스 생성
event_bus = EventBus()

# 핸들러 등록
from events.handlers import LoggingHandler, MetricsHandler
LoggingHandler(event_bus)
MetricsHandler(event_bus)

# TradeExecutionGate에 이벤트 버스 전달
gate = TradeExecutionGate(
    notifier=notifier,
    config=config.trade_gate,
    event_bus=event_bus  # 이벤트 버스 전달
)
```

### 6.2 자동 이벤트 발행

TradeExecutionGate는 다음 상황에서 자동으로 이벤트를 발행합니다:

1. **진입 시**: `TradeEntryEvent`
   - 진입 가격, 사이즈, 신뢰도, 슬롯 등 포함

2. **청산 시**: `TradeExitEvent`
   - 진입/청산 가격, PnL, 청산 사유, 보유 시간 등 포함

3. **리스크 한도 차단 시**: `RiskLimitEvent`
   - 한도 타입, 현재값, 한도값, 조치 포함

### 6.3 이벤트 비활성화

이벤트 시스템은 선택적입니다. 이벤트 버스를 전달하지 않으면 기존과 동일하게 작동합니다.

```python
# 이벤트 버스 없이 실행 (기존 방식)
gate = TradeExecutionGate(notifier, config=config.trade_gate)
```

---

## 7. 사용 예시

### 7.1 기본 사용

```python
from events import EventBus, TradeEntryEvent
from datetime import datetime

# 버스 생성
bus = EventBus()

# 핸들러 등록
@bus.subscribe(TradeEntryEvent)
def log_entry(event):
    print(f"진입: {event.side} @ {event.price}")

# 이벤트 발행
event = TradeEntryEvent(
    side="LONG",
    price=380.0,
    size=100.0,
    confidence="HIGH",
    prob=0.75,
    slot="A",
    signal="BUY"
)
bus.publish(event)
```

### 7.2 여러 핸들러

```python
from events import EventBus, TradeEntryEvent

bus = EventBus()

@bus.subscribe(TradeEntryEvent)
def handler1(event):
    print(f"핸들러 1: {event.side}")

@bus.subscribe(TradeEntryEvent)
def handler2(event):
    print(f"핸들러 2: {event.price}")

# 두 핸들러 모두 호출됨
bus.publish(event)
```

### 7.3 메트릭 수집

```python
from events import EventBus, MetricsHandler, TradeEntryEvent, TradeExitEvent

bus = EventBus()
handler = MetricsHandler(bus)

# 이벤트 발행
bus.publish(TradeEntryEvent(...))
bus.publish(TradeExitEvent(...))

# 메트릭 확인
metrics = handler.get_metrics()
print(f"진입: {metrics['total_entries']}")
print(f"청산: {metrics['total_exits']}")
```

### 7.4 컨디셔널 핸들러

```python
from events import EventBus, TradeEntryEvent

bus = EventBus()

@bus.subscribe(TradeEntryEvent)
def high_confidence_only(event):
    if event.confidence == "HIGH":
        print("높은 신뢰도 진입만 처리")
        # 추가 로직
```

### 7.5 에러 처리

```python
from events import EventBus, TradeEntryEvent

bus = EventBus()

@bus.subscribe(TradeEntryEvent)
def risky_handler(event):
    try:
        # 위험한 작업
        process_trade(event)
    except Exception as e:
        print(f"핸들러 오류: {e}")
        # 에러 이벤트 발행
        from events import ErrorEvent
        bus.publish(ErrorEvent(
            error_type="HANDLER_ERROR",
            message=str(e),
            context={"event": event.event_id}
        ))
```

---

## 8. 권장 사항

### 8.1 핸들러 설계

**권장 사항**:
- 핸들러는 빨리 실행되어야 함 (장기 실행 금지)
- 복잡한 로직은 별도 스레드로 처리
- 에러 처리를 포함해야 함
- 상태 변경은 스레드 안전하게

**비권장**:
- 핸들러에서 다른 이벤트 발행 (순환 위험)
- 핸들러에서 긴 시간 작업 수행
- 핸들러에서 예외 발생 (반드시 캐치)

### 8.2 이벤트 설계

**권장 사항**:
- 이벤트는 불변(immutable)으로 처리
- 필수 필드와 선택적 필드 명확히 구분
- 이벤트 이름은 명확하고 직관적
- 관련 데이터 포함

**비권장**:
- 이벤트에 너무 많은 데이터 포함
- 이벤트 간 상속 깊게
- 이벤트 이름 모호하게

### 8.3 성능 고려사항

**권장 사항**:
- 빈번한 이벤트는 비동기로 발행
- 핸들러 수는 최소화
- 이벤트 로깅은 선택적으로 활성화
- 통계는 주기적으로 초기화

**비권장**:
- 모든 이벤트 동기 발행
- 불필요한 핸들러 등록
- 과도한 로깅
- 통계 누적만 계속

### 8.4 테스트

**권장 사항**:
- 각 핸들러 단위 테스트
- 이벤트 발행/구독 통합 테스트
- 에러 상황 테스트
- 스레드 안전성 테스트

**비권장**:
- 핸들러 로직 테스트 없이 통합만 테스트
- 에러 상황 테스트 생략
- 동시성 테스트 생략

### 8.5 디버깅

**권장 사항**:
- 이벤트 로깅 활성화
- 이벤트 통계 모니터링
- 핸들러 실행 시간 측정
- 이벤트 추적 ID 사용

**비권장**:
- 로깅 없이 디버깅
- 통계 확인 없이 운영
- 이벤트 ID 무시

---

## 9. 주의사항

### 9.1 순환 참조

이벤트 핸들러에서 다시 이벤트를 발행할 때 순환 참조을 피해야 합니다.

```python
# 나쁜 예
@bus.subscribe(TradeEntryEvent)
def handler(event):
    if condition:
        bus.publish(TradeEntryEvent(...))  # 순환 위험
```

### 9.2 스레드 안전성

이벤트 버스는 스레드 안전하지만, 핸들러 내부의 상태 변경은 스레드 안전하게 처리해야 합니다.

```python
# 좋은 예
import threading

class StatefulHandler:
    def __init__(self):
        self.lock = threading.Lock()
        self.counter = 0
    
    def handle(self, event):
        with self.lock:
            self.counter += 1
```

### 9.3 에러 처리

핸들러에서 예외가 발생해도 다른 핸들러는 계속 실행되도록 에러 처리를 포함해야 합니다.

```python
@bus.subscribe(TradeEntryEvent)
def handler(event):
    try:
        process(event)
    except Exception as e:
        logger.error(f"핸들러 오류: {e}")
```

### 9.4 메모리 누수

핸들러가 등록되지 않거나 해제되지 않으면 메모리 누수가 발생할 수 있습니다.

```python
# 사용 후 해제
bus.unsubscribe(TradeEntryEvent, handler)
```

---

**문서 버전**: 1.0  
**최종 갱신**: 2026-04-26
