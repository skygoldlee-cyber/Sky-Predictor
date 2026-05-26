"""
이벤트 버스 단위 테스트
"""

import pytest
from datetime import datetime
from events.event_bus import EventBus, Event, get_event_bus, set_event_bus
from events.events import (
    TradeEntryEvent,
    TradeExitEvent,
    SignalEvent,
    RiskLimitEvent
)


class TestEvent:
    """Event 기본 클래스 테스트."""
    
    def test_event_creation(self):
        """이벤트 생성 테스트."""
        event = Event(timestamp=datetime.now())
        assert event.timestamp is not None
        assert event.event_id != ""
    
    def test_event_id_generation(self):
        """이벤트 ID 자동 생성 테스트."""
        now = datetime(2026, 1, 1, 10, 0, 0)
        event = Event(timestamp=now)
        assert "Event" in event.event_id
        assert "20260101" in event.event_id


class TestEventBus:
    """EventBus 테스트."""
    
    def test_init(self):
        """초기화 테스트."""
        bus = EventBus()
        assert bus is not None
        assert bus._handlers is not None
    
    def test_subscribe_decorator(self):
        """구독 데코레이터 테스트."""
        bus = EventBus()
        
        @bus.subscribe(TradeEntryEvent)
        def handler(event: TradeEntryEvent):
            pass
        
        assert len(bus._handlers[TradeEntryEvent]) == 1
    
    def test_publish(self):
        """이벤트 발행 테스트."""
        bus = EventBus()
        
        received_events = []
        
        @bus.subscribe(TradeEntryEvent)
        def handler(event: TradeEntryEvent):
            received_events.append(event)
        
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
        
        assert len(received_events) == 1
        assert received_events[0] == event
    
    def test_publish_invalid_event(self):
        """잘못된 이벤트 발행 테스트."""
        bus = EventBus()
        
        # 잘못된 타입 발행
        bus.publish("not an event")  # 에러 없이 무시되어야 함
        
        # 핸들러가 호출되지 않아야 함
        assert bus.get_stats().get("TradeEntryEvent", 0) == 0
    
    def test_unsubscribe(self):
        """구독 해제 테스트."""
        bus = EventBus()
        
        @bus.subscribe(TradeEntryEvent)
        def handler(event: TradeEntryEvent):
            pass
        
        assert len(bus._handlers[TradeEntryEvent]) == 1
        
        result = bus.unsubscribe(TradeEntryEvent, handler)
        assert result is True
        assert len(bus._handlers[TradeEntryEvent]) == 0
    
    def test_unsubscribe_nonexistent(self):
        """존재하지 않는 핸들러 해제 테스트."""
        bus = EventBus()
        
        def handler(event: TradeEntryEvent):
            pass
        
        result = bus.unsubscribe(TradeEntryEvent, handler)
        assert result is False
    
    def test_multiple_handlers(self):
        """여러 핸들러 테스트."""
        bus = EventBus()
        
        received = []
        
        @bus.subscribe(TradeEntryEvent)
        def handler1(event: TradeEntryEvent):
            received.append(1)
        
        @bus.subscribe(TradeEntryEvent)
        def handler2(event: TradeEntryEvent):
            received.append(2)
        
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
        
        assert len(received) == 2
        assert 1 in received
        assert 2 in received
    
    def test_stats(self):
        """이벤트 통계 테스트."""
        bus = EventBus()
        
        @bus.subscribe(TradeEntryEvent)
        def handler(event: TradeEntryEvent):
            pass
        
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
        bus.publish(event)
        
        stats = bus.get_stats()
        assert stats.get("TradeEntryEvent") == 2
    
    def test_reset_stats(self):
        """통계 초기화 테스트."""
        bus = EventBus()
        
        @bus.subscribe(TradeEntryEvent)
        def handler(event: TradeEntryEvent):
            pass
        
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
        assert bus.get_stats().get("TradeEntryEvent") == 1
        
        bus.reset_stats()
        assert bus.get_stats().get("TradeEntryEvent", 0) == 0
    
    def test_logging_enabled(self):
        """로깅 활성화 테스트."""
        bus = EventBus()
        assert bus._enable_logging is True
        
        bus.set_logging_enabled(False)
        assert bus._enable_logging is False
        
        bus.set_logging_enabled(True)
        assert bus._enable_logging is True


class TestGlobalEventBus:
    """전역 이벤트 버스 테스트."""
    
    def test_get_event_bus(self):
        """전역 이벤트 버스 가져오기 테스트."""
        bus = get_event_bus()
        assert bus is not None
        assert isinstance(bus, EventBus)
    
    def test_set_event_bus(self):
        """전역 이벤트 버스 설정 테스트."""
        custom_bus = EventBus()
        set_event_bus(custom_bus)
        
        assert get_event_bus() is custom_bus


class TestTradeEvents:
    """거래 이벤트 테스트."""
    
    def test_trade_entry_event(self):
        """진입 이벤트 테스트."""
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
        
        assert event.side == "LONG"
        assert event.price == 380.0
        assert event.size == 100.0
        assert event.confidence == "HIGH"
        assert event.slot == "A"
    
    def test_trade_exit_event(self):
        """청산 이벤트 테스트."""
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
        
        assert event.side == "LONG"
        assert event.entry_price == 380.0
        assert event.exit_price == 382.0
        assert event.pnl == 2.0
        assert event.reason == "TARGET_PROFIT"
    
    def test_signal_event(self):
        """신호 이벤트 테스트."""
        event = SignalEvent(
            timestamp=datetime.now(),
            signal="BUY",
            confidence="HIGH",
            prob=0.75,
            price=380.0
        )
        
        assert event.signal == "BUY"
        assert event.confidence == "HIGH"
        assert event.prob == 0.75
    
    def test_risk_limit_event(self):
        """리스크 한도 이벤트 테스트."""
        event = RiskLimitEvent(
            timestamp=datetime.now(),
            limit_type="CONSECUTIVE_LOSS",
            current_value=3.0,
            limit_value=3.0,
            action="BLOCK_ENTRY"
        )
        
        assert event.limit_type == "CONSECUTIVE_LOSS"
        assert event.current_value == 3.0
        assert event.action == "BLOCK_ENTRY"
