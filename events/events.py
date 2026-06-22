"""
거래 이벤트 정의

시스템에서 발생하는 다양한 거래 관련 이벤트를 정의합니다.
"""

from datetime import datetime
from events.event_bus import Event


class TradeEntryEvent(Event):
    """진입 이벤트."""
    def __init__(self, side: str, price: float, size: float, confidence: str, 
                 prob: float, slot: str, signal: str, timestamp: datetime = None, event_id: str = ""):
        super().__init__(timestamp, event_id)
        self.side = side
        self.price = price
        self.size = size
        self.confidence = confidence
        self.prob = prob
        self.slot = slot
        self.signal = signal


class TradeExitEvent(Event):
    """청산 이벤트."""
    def __init__(self, side: str, entry_price: float, exit_price: float, size: float,
                 pnl: float, pnl_pct: float, reason: str, hold_minutes: float, slot: str,
                 timestamp: datetime = None, event_id: str = ""):
        super().__init__(timestamp, event_id)
        self.side = side
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.size = size
        self.pnl = pnl
        self.pnl_pct = pnl_pct
        self.reason = reason
        self.hold_minutes = hold_minutes
        self.slot = slot


class SignalEvent(Event):
    """신호 이벤트."""
    def __init__(self, signal: str, confidence: str, prob: float, price: float,
                 timestamp: datetime = None, event_id: str = ""):
        super().__init__(timestamp, event_id)
        self.signal = signal
        self.confidence = confidence
        self.prob = prob
        self.price = price


class RiskLimitEvent(Event):
    """리스크 한도 이벤트."""
    def __init__(self, limit_type: str, current_value: float, limit_value: float, action: str,
                 timestamp: datetime = None, event_id: str = ""):
        super().__init__(timestamp, event_id)
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value
        self.action = action


class TrailingStopEvent(Event):
    """트레일링 스탑 이벤트."""
    def __init__(self, side: str, current_price: float, trailing_stop_price: float,
                 activation_price: float, distance: float, timestamp: datetime = None, event_id: str = ""):
        super().__init__(timestamp, event_id)
        self.side = side
        self.current_price = current_price
        self.trailing_stop_price = trailing_stop_price
        self.activation_price = activation_price
        self.distance = distance


class ErrorEvent(Event):
    """에러 이벤트."""
    def __init__(self, error_type: str, message: str, context: dict,
                 timestamp: datetime = None, event_id: str = ""):
        super().__init__(timestamp, event_id)
        self.error_type = error_type
        self.message = message
        self.context = context


class SystemEvent(Event):
    """시스템 이벤트."""
    def __init__(self, event_type: str, message: str, data: dict,
                 timestamp: datetime = None, event_id: str = ""):
        super().__init__(timestamp, event_id)
        self.event_type = event_type
        self.message = message
        self.data = data


class PerformanceEvent(Event):
    """성과 이벤트."""
    def __init__(self, metric_type: str, value: float, period: str,
                 timestamp: datetime = None, event_id: str = ""):
        super().__init__(timestamp, event_id)
        self.metric_type = metric_type
        self.value = value
        self.period = period


class AlertEvent(Event):
    """알림 이벤트."""
    def __init__(self, alert_type: str, message: str, data: dict,
                 timestamp: datetime = None, event_id: str = ""):
        super().__init__(timestamp, event_id)
        self.alert_type = alert_type
        self.message = message
        self.data = data
