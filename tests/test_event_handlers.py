"""
이벤트 핸들러 단위 테스트
"""

from datetime import datetime
from events.event_bus import EventBus
from events.events import (
    TradeEntryEvent,
    TradeExitEvent,
    SignalEvent,
    AlertEvent
)
from events.handlers import LoggingHandler, MetricsHandler, TelegramNotifierHandler


class TestLoggingHandler:
    """LoggingHandler 테스트."""
    
    def test_init(self):
        """초기화 테스트."""
        bus = EventBus()
        handler = LoggingHandler(bus)
        assert handler is not None
        assert handler.event_bus == bus
    
    def test_handlers_registered(self):
        """핸들러 등록 테스트."""
        bus = EventBus()
        handler = LoggingHandler(bus)
        
        # 핸들러가 등록되었는지 확인
        assert len(bus._handlers[TradeEntryEvent]) >= 1
        assert len(bus._handlers[TradeExitEvent]) >= 1
    
    def test_trade_entry_logging(self, caplog):
        """진입 이벤트 로깅 테스트."""
        bus = EventBus()
        handler = LoggingHandler(bus)
        
        event = TradeEntryEvent(
            timestamp=datetime.now(),
            side="LONG",
            price=380.0,
            size=100.0,
            confidence="HIGH",
            prob=0.75,
            slot="A",
            signal="BUY"
        )
        
        bus.publish(event)
        
        # 로그가 기록되었는지 확인 (실제 로그 내용은 caplog로 확인 가능)
        assert len(bus.get_stats()) > 0


class TestMetricsHandler:
    """MetricsHandler 테스트."""
    
    def test_init(self):
        """초기화 테스트."""
        bus = EventBus()
        handler = MetricsHandler(bus)
        assert handler is not None
        assert handler.event_bus == bus
    
    def test_initial_metrics(self):
        """초기 메트릭 테스트."""
        bus = EventBus()
        handler = MetricsHandler(bus)
        
        metrics = handler.get_metrics()
        assert metrics["total_trades"] == 0
        assert metrics["total_entries"] == 0
        assert metrics["total_exits"] == 0
        assert metrics["total_signals"] == 0
        assert metrics["total_errors"] == 0
    
    def test_trade_entry_metric(self):
        """진입 메트릭 테스트."""
        bus = EventBus()
        handler = MetricsHandler(bus)
        
        event = TradeEntryEvent(
            timestamp=datetime.now(),
            side="LONG",
            price=380.0,
            size=100.0,
            confidence="HIGH",
            prob=0.75,
            slot="A",
            signal="BUY"
        )
        
        bus.publish(event)
        
        metrics = handler.get_metrics()
        assert metrics["total_entries"] == 1
        assert metrics["total_trades"] == 1
        assert metrics["entries_by_slot"]["A"] == 1
    
    def test_trade_exit_metric(self):
        """청산 메트릭 테스트."""
        bus = EventBus()
        handler = MetricsHandler(bus)
        
        event = TradeExitEvent(
            timestamp=datetime.now(),
            side="LONG",
            entry_price=380.0,
            exit_price=382.0,
            size=100.0,
            pnl=2.0,
            pnl_pct=0.526,
            reason="TARGET_PROFIT",
            hold_minutes=30.0,
            slot="A"
        )
        
        bus.publish(event)
        
        metrics = handler.get_metrics()
        assert metrics["total_exits"] == 1
        assert metrics["exits_by_reason"]["TARGET_PROFIT"] == 1
    
    def test_signal_metric(self):
        """신호 메트릭 테스트."""
        bus = EventBus()
        handler = MetricsHandler(bus)
        
        event = SignalEvent(
            timestamp=datetime.now(),
            signal="BUY",
            confidence="HIGH",
            prob=0.75,
            price=380.0
        )
        
        bus.publish(event)
        
        metrics = handler.get_metrics()
        assert metrics["total_signals"] == 1
        assert metrics["signals_by_type"]["BUY"] == 1
    
    def test_reset_metrics(self):
        """메트릭 초기화 테스트."""
        bus = EventBus()
        handler = MetricsHandler(bus)
        
        event = TradeEntryEvent(
            timestamp=datetime.now(),
            side="LONG",
            price=380.0,
            size=100.0,
            confidence="HIGH",
            prob=0.75,
            slot="A",
            signal="BUY"
        )
        
        bus.publish(event)
        assert handler.get_metrics()["total_entries"] == 1
        
        handler.reset_metrics()
        assert handler.get_metrics()["total_entries"] == 0


class TestTelegramNotifierHandler:
    """TelegramNotifierHandler 테스트."""
    
    def test_init_without_notifier(self):
        """노티파이어 없이 초기화 테스트."""
        bus = EventBus()
        handler = TelegramNotifierHandler(bus, notifier=None)
        assert handler is not None
        assert handler.notifier is None
    
    def test_init_with_notifier(self):
        """노티파이어와 함께 초기화 테스트."""
        bus = EventBus()
        
        # Mock notifier
        class MockNotifier:
            def send_text(self, message):
                pass
        
        notifier = MockNotifier()
        handler = TelegramNotifierHandler(bus, notifier=notifier)
        
        assert handler.notifier == notifier
    
    def test_handlers_registered(self):
        """핸들러 등록 테스트."""
        bus = EventBus()
        handler = TelegramNotifierHandler(bus, notifier=None)
        
        # 핸들러가 등록되었는지 확인
        assert len(bus._handlers[TradeEntryEvent]) >= 1
        assert len(bus._handlers[TradeExitEvent]) >= 1
    
    def test_trade_entry_without_notifier(self):
        """노티파이어 없는 진입 이벤트 테스트."""
        bus = EventBus()
        handler = TelegramNotifierHandler(bus, notifier=None)
        
        event = TradeEntryEvent(
            timestamp=datetime.now(),
            side="LONG",
            price=380.0,
            size=100.0,
            confidence="HIGH",
            prob=0.75,
            slot="A",
            signal="BUY"
        )
        
        # 노티파이어가 없어도 에러 없이 실행되어야 함
        bus.publish(event)
    
    def test_alert_with_notifier(self):
        """노티파이어 있는 알림 이벤트 테스트."""
        bus = EventBus()
        
        # Mock notifier
        messages = []
        class MockNotifier:
            def send_text(self, message):
                messages.append(message)
        
        notifier = MockNotifier()
        handler = TelegramNotifierHandler(bus, notifier=notifier)
        
        # WARNING 이상 레벨만 전송
        event = AlertEvent(
            timestamp=datetime.now(),
            alert_type="WARNING",
            message="Test warning",
            data={}
        )
        
        bus.publish(event)
        
        # 메시지가 전송되었는지 확인
        assert len(messages) > 0
