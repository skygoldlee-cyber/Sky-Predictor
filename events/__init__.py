"""
이벤트 기반 아키텍처 패키지

이벤트 버스, 이벤트 정의, 핸들러를 제공합니다.
"""

from events.event_bus import EventBus, Event, get_event_bus, set_event_bus
from events.events import (
    TradeEntryEvent,
    TradeExitEvent,
    SignalEvent,
    RiskLimitEvent,
    TrailingStopEvent,
    ErrorEvent,
    SystemEvent,
    PerformanceEvent,
    AlertEvent
)
from events.handlers import (
    LoggingHandler,
    MetricsHandler,
    TelegramNotifierHandler
)

__all__ = [
    # Event Bus
    "EventBus",
    "Event",
    "get_event_bus",
    "set_event_bus",
    # Events
    "TradeEntryEvent",
    "TradeExitEvent",
    "SignalEvent",
    "RiskLimitEvent",
    "TrailingStopEvent",
    "ErrorEvent",
    "SystemEvent",
    "PerformanceEvent",
    "AlertEvent",
    # Handlers
    "LoggingHandler",
    "MetricsHandler",
    "TelegramNotifierHandler",
]
