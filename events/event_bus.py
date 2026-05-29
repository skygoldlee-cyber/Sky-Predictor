"""
이벤트 버스 모듈

시스템 컴포넌트 간 느슨한 결합을 위한 이벤트 기반 아키텍처 구현.

Usage:
    from events.event_bus import EventBus, Event
    from events.events import TradeEntryEvent, TradeExitEvent
    
    # 이벤트 버스 초기화
    bus = EventBus()
    
    # 이벤트 핸들러 등록
    @bus.subscribe(TradeEntryEvent)
    def handle_entry(event: TradeEntryEvent):
        print(f"진입 이벤트: {event.side} @ {event.price}")
    
    # 이벤트 발행
    bus.publish(TradeEntryEvent(
        side="LONG",
        price=380.0,
        size=100.0,
        timestamp=datetime.now()
    ))
"""

import logging
import threading
from typing import Callable, Dict, List, Type
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class Event:
    """이벤트 기본 클래스."""
    def __init__(self, timestamp: datetime = None, event_id: str = ""):
        self.timestamp = timestamp if timestamp is not None else datetime.now()
        self.event_id = event_id if event_id else f"{self.__class__.__name__}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"


class EventBus:
    """이벤트 버스.
    
    이벤트 발행(Publish)과 구독(Subscribe) 패턴을 구현하여
    컴포넌트 간 느슨한 결합을 제공합니다.
    """
    
    def __init__(self):
        """초기화."""
        # 이벤트 타입별 핸들러 목록
        self._handlers: Dict[Type[Event], List[Callable]] = defaultdict(list)
        # 스레드 안전성을 위한 락
        self._lock = threading.RLock()
        # 이벤트 로깅
        self._enable_logging = True
        # 이벤트 통계
        self._stats: Dict[str, int] = defaultdict(int)
    
    def subscribe(self, event_type: Type[Event]) -> Callable:
        """이벤트 구독 데코레이터.
        
        Args:
            event_type: 구독할 이벤트 타입
        
        Returns:
            데코레이터 함수
        
        Example:
            @bus.subscribe(TradeEntryEvent)
            def handle_entry(event: TradeEntryEvent):
                print(event)
        """
        def decorator(handler: Callable):
            with self._lock:
                self._handlers[event_type].append(handler)
                logger.debug("[EVENT] 핸들러 등록: %s -> %s", event_type.__name__, handler.__name__)
            return handler
        return decorator
    
    def unsubscribe(self, event_type: Type[Event], handler: Callable) -> bool:
        """이벤트 구독 해제.
        
        Args:
            event_type: 구독 해제할 이벤트 타입
            handler: 해제할 핸들러
        
        Returns:
            성공 여부
        """
        with self._lock:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                logger.debug("[EVENT] 핸들러 해제: %s -> %s", event_type.__name__, handler.__name__)
                return True
            return False
    
    def publish(self, event: Event) -> None:
        """이벤트 발행.
        
        Args:
            event: 발행할 이벤트
        """
        if not isinstance(event, Event):
            logger.warning("[EVENT] 잘못된 이벤트 타입: %s", type(event))
            return
        
        event_type = type(event)
        
        # 로깅
        if self._enable_logging:
            logger.debug("[EVENT] 이벤트 발행: %s (ID: %s)", event_type.__name__, event.event_id)
        
        # 통계
        self._stats[event_type.__name__] += 1
        
        # 핸들러 호출
        handlers = self._get_handlers(event_type)
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error("[EVENT] 핸들러 실행 오류: %s -> %s: %s", 
                           event_type.__name__, handler.__name__, e, exc_info=True)
    
    def publish_async(self, event: Event) -> None:
        """비동기 이벤트 발행 (별도 스레드에서 실행).
        
        Args:
            event: 발행할 이벤트
        """
        def _publish():
            self.publish(event)
        
        thread = threading.Thread(target=_publish, daemon=True)
        thread.start()
    
    def _get_handlers(self, event_type: Type[Event]) -> List[Callable]:
        """이벤트 타입에 대한 핸들러 목록 반환.
        
        Args:
            event_type: 이벤트 타입
        
        Returns:
            핸들러 목록
        """
        with self._lock:
            return self._handlers[event_type].copy()
    
    def get_stats(self) -> Dict[str, int]:
        """이벤트 통계 반환.
        
        Returns:
            이벤트 타입별 발행 횟수
        """
        return dict(self._stats)
    
    def reset_stats(self) -> None:
        """이벤트 통계 초기화."""
        with self._lock:
            self._stats.clear()
    
    def set_logging_enabled(self, enabled: bool) -> None:
        """이벤트 로깅 활성화/비활성화.
        
        Args:
            enabled: 활성화 여부
        """
        self._enable_logging = enabled


# 전역 이벤트 버스 인스턴스
# [FIX] TOCTOU: None 체크와 생성 사이에 다른 스레드가 진입하면
# 두 개의 EventBus 인스턴스가 생성되어 핸들러 등록이 분산될 수 있다.
# Double-Checked Locking 패턴으로 해결한다.
_global_event_bus: EventBus = None
_global_bus_lock: threading.Lock = threading.Lock()


def get_event_bus() -> EventBus:
    """전역 이벤트 버스 인스턴스 반환 (스레드 안전).

    Double-Checked Locking 패턴:
      1차 체크: 락 없이 빠르게 확인 (인스턴스가 이미 있으면 락 불필요)
      2차 체크: 락 획득 후 재확인 (1차 체크 통과한 두 스레드가 동시 진입 방지)

    Returns:
        전역 이벤트 버스
    """
    global _global_event_bus
    if _global_event_bus is None:                    # 1차 체크 (무락, 성능)
        with _global_bus_lock:
            if _global_event_bus is None:            # 2차 체크 (유락, 안전)
                _global_event_bus = EventBus()
    return _global_event_bus


def set_event_bus(bus: EventBus) -> None:
    """전역 이벤트 버스 설정.

    Args:
        bus: 설정할 이벤트 버스
    """
    global _global_event_bus
    with _global_bus_lock:
        _global_event_bus = bus
