"""
이벤트 핸들러 모듈

다양한 이벤트를 처리하는 핸들러들을 정의합니다.
"""

import logging
from typing import Dict, Any
from events.event_bus import EventBus
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

logger = logging.getLogger(__name__)


class LoggingHandler:
    """이벤트 로깅 핸들러."""
    
    def __init__(self, event_bus: EventBus):
        """초기화.
        
        Args:
            event_bus: 이벤트 버스
        """
        self.event_bus = event_bus
        self._register_handlers()
    
    def _register_handlers(self):
        """모든 이벤트 핸들러 등록."""
        self.event_bus.subscribe(TradeEntryEvent)(self._on_trade_entry)
        self.event_bus.subscribe(TradeExitEvent)(self._on_trade_exit)
        self.event_bus.subscribe(SignalEvent)(self._on_signal)
        self.event_bus.subscribe(RiskLimitEvent)(self._on_risk_limit)
        self.event_bus.subscribe(TrailingStopEvent)(self._on_trailing_stop)
        self.event_bus.subscribe(ErrorEvent)(self._on_error)
        self.event_bus.subscribe(SystemEvent)(self._on_system)
        self.event_bus.subscribe(PerformanceEvent)(self._on_performance)
        self.event_bus.subscribe(AlertEvent)(self._on_alert)
    
    def _on_trade_entry(self, event: TradeEntryEvent):
        """진입 이벤트 처리."""
        logger.info(
            "[EVENT-ENTRY] %s %s @ %.2f 사이즈=%.2f 신뢰도=%s 확률=%.2f%% 슬롯=%s",
            event.signal, event.side, event.price, event.size,
            event.confidence, event.prob * 100, event.slot
        )
    
    def _on_trade_exit(self, event: TradeExitEvent):
        """청산 이벤트 처리."""
        logger.info(
            "[EVENT-EXIT] %s 진입=%.2f 청산=%.2f PnL=%.2fpt (%.2f%%) 사유=%s 보유=%.0f분 슬롯=%s",
            event.side, event.entry_price, event.exit_price,
            event.pnl, event.pnl_pct, event.reason,
            event.hold_minutes, event.slot
        )
    
    def _on_signal(self, event: SignalEvent):
        """신호 이벤트 처리."""
        logger.debug(
            "[EVENT-SIGNAL] %s 신뢰도=%s 확률=%.2f%% 가격=%.2f",
            event.signal, event.confidence, event.prob * 100, event.price
        )
    
    def _on_risk_limit(self, event: RiskLimitEvent):
        """리스크 한도 이벤트 처리."""
        logger.warning(
            "[EVENT-RISK] %s 현재=%.2f 한도=%.2f 조치=%s",
            event.limit_type, event.current_value, event.limit_value, event.action
        )
    
    def _on_trailing_stop(self, event: TrailingStopEvent):
        """트레일링 스탑 이벤트 처리."""
        logger.debug(
            "[EVENT-TRAILING] %s 현재=%.2f 트레일링=%.2f 활성화=%.2f 거리=%.2f",
            event.side, event.current_price, event.trailing_stop_price,
            event.activation_price, event.distance
        )
    
    def _on_error(self, event: ErrorEvent):
        """에러 이벤트 처리."""
        logger.error(
            "[EVENT-ERROR] %s: %s (컨텍스트: %s)",
            event.error_type, event.message, event.context
        )
    
    def _on_system(self, event: SystemEvent):
        """시스템 이벤트 처리."""
        logger.info(
            "[EVENT-SYSTEM] %s: %s (데이터: %s)",
            event.event_type, event.message, event.data
        )
    
    def _on_performance(self, event: PerformanceEvent):
        """성과 이벤트 처리."""
        logger.info(
            "[EVENT-PERF] %s %s=%.4f 기간=%s",
            event.period, event.metric_type, event.value, event.timestamp
        )
    
    def _on_alert(self, event: AlertEvent):
        """알림 이벤트 처리."""
        level = logger.info if event.alert_type == "INFO" else logger.warning
        if event.alert_type == "ERROR" or event.alert_type == "CRITICAL":
            level = logger.error
        
        level(
            "[EVENT-ALERT] [%s] %s (데이터: %s)",
            event.alert_type, event.message, event.data
        )


class MetricsHandler:
    """이벤트 메트릭 수집 핸들러."""
    
    def __init__(self, event_bus: EventBus):
        """초기화.
        
        Args:
            event_bus: 이벤트 버스
        """
        self.event_bus = event_bus
        self.metrics: Dict[str, Any] = {
            "total_trades": 0,
            "total_entries": 0,
            "total_exits": 0,
            "total_signals": 0,
            "total_errors": 0,
            "entries_by_slot": {"A": 0, "B": 0, "C": 0},
            "exits_by_reason": {},
            "signals_by_type": {"BUY": 0, "SELL": 0, "HOLD": 0},
        }
        self._register_handlers()
    
    def _register_handlers(self):
        """핸들러 등록."""
        self.event_bus.subscribe(TradeEntryEvent)(self._on_trade_entry)
        self.event_bus.subscribe(TradeExitEvent)(self._on_trade_exit)
        self.event_bus.subscribe(SignalEvent)(self._on_signal)
        self.event_bus.subscribe(ErrorEvent)(self._on_error)
    
    def _on_trade_entry(self, event: TradeEntryEvent):
        """진입 이벤트 처리."""
        self.metrics["total_entries"] += 1
        self.metrics["total_trades"] += 1
        self.metrics["entries_by_slot"][event.slot] += 1
    
    def _on_trade_exit(self, event: TradeExitEvent):
        """청산 이벤트 처리."""
        self.metrics["total_exits"] += 1
        self.metrics["exits_by_reason"][event.reason] = \
            self.metrics["exits_by_reason"].get(event.reason, 0) + 1
    
    def _on_signal(self, event: SignalEvent):
        """신호 이벤트 처리."""
        self.metrics["total_signals"] += 1
        self.metrics["signals_by_type"][event.signal] = \
            self.metrics["signals_by_type"].get(event.signal, 0) + 1
    
    def _on_error(self, event: ErrorEvent):
        """에러 이벤트 처리."""
        self.metrics["total_errors"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 반환.
        
        Returns:
            메트릭 딕셔너리
        """
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """메트릭 초기화."""
        self.metrics = {
            "total_trades": 0,
            "total_entries": 0,
            "total_exits": 0,
            "total_signals": 0,
            "total_errors": 0,
            "entries_by_slot": {"A": 0, "B": 0, "C": 0},
            "exits_by_reason": {},
            "signals_by_type": {"BUY": 0, "SELL": 0, "HOLD": 0},
        }


class TelegramNotifierHandler:
    """텔레그램 알림 핸들러."""
    
    def __init__(self, event_bus: EventBus, notifier=None):
        """초기화.
        
        Args:
            event_bus: 이벤트 버스
            notifier: 텔레그램 노티파이어 (선택적)
        """
        self.event_bus = event_bus
        self.notifier = notifier
        self._register_handlers()
    
    def _register_handlers(self):
        """핸들러 등록."""
        self.event_bus.subscribe(TradeEntryEvent)(self._on_trade_entry)
        self.event_bus.subscribe(TradeExitEvent)(self._on_trade_exit)
        self.event_bus.subscribe(RiskLimitEvent)(self._on_risk_limit)
        self.event_bus.subscribe(AlertEvent)(self._on_alert)
    
    def _on_trade_entry(self, event: TradeEntryEvent):
        """진입 이벤트 처리."""
        if self.notifier:
            message = (
                f"📈 진입: {event.signal} {event.side}\n"
                f"가격: {event.price:.2f}\n"
                f"사이즈: {event.size:.2f}\n"
                f"신뢰도: {event.confidence}\n"
                f"슬롯: {event.slot}"
            )
            try:
                self.notifier.send_text(message)
            except Exception as e:
                logger.error("[EVENT-TELEGRAM] 진입 알림 전송 실패: %s", e)
    
    def _on_trade_exit(self, event: TradeExitEvent):
        """청산 이벤트 처리."""
        if self.notifier:
            emoji = "📉" if event.pnl < 0 else "📈"
            message = (
                f"{emoji} 청산: {event.side}\n"
                f"진입: {event.entry_price:.2f} → 청산: {event.exit_price:.2f}\n"
                f"PnL: {event.pnl:.2f}pt ({event.pnl_pct:.2f}%)\n"
                f"사유: {event.reason}\n"
                f"보유: {event.hold_minutes:.0f}분"
            )
            try:
                self.notifier.send_text(message)
            except Exception as e:
                logger.error("[EVENT-TELEGRAM] 청산 알림 전송 실패: %s", e)
    
    def _on_risk_limit(self, event: RiskLimitEvent):
        """리스크 한도 이벤트 처리."""
        if self.notifier:
            message = (
                f"⚠️ 리스크 한도: {event.limit_type}\n"
                f"현재: {event.current_value:.2f} / 한도: {event.limit_value:.2f}\n"
                f"조치: {event.action}"
            )
            try:
                self.notifier.send_text(message)
            except Exception as e:
                logger.error("[EVENT-TELEGRAM] 리스크 알림 전송 실패: %s", e)
    
    def _on_alert(self, event: AlertEvent):
        """알림 이벤트 처리."""
        if self.notifier and event.alert_type in ["WARNING", "ERROR", "CRITICAL"]:
            emoji = "⚠️" if event.alert_type == "WARNING" else "🚨"
            message = f"{emoji} [{event.alert_type}] {event.message}"
            try:
                self.notifier.send_text(message)
            except Exception as e:
                logger.error("[EVENT-TELEGRAM] 알림 전송 실패: %s", e)
