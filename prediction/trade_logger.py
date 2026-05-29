"""실시간 거래 이벤트 로거

실시간 매매 진입/청산 이벤트를 기록하고,
백테스팅이 이를 기반으로 실행할 수 있도록 지원합니다.

Usage:
    from prediction.trade_logger import TradeLogger, TradeEvent
    
    logger = TradeLogger()
    logger.log_event(TradeEvent(
        event_type="ENTRY",
        timestamp=datetime.now(),
        action="BUY",
        price=325.50,
        size=1.0,
        confidence="HIGH",
        reason=None,
        signal_reason="zigzag_pivot_low"
    ))
"""

import logging
import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


@dataclass
class TradeEvent:
    """거래 이벤트."""
    event_type: str  # "ENTRY" or "EXIT"
    timestamp: datetime
    action: str  # "BUY" or "SELL"
    price: float
    size: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    reason: Optional[str] = None  # 청산 사유 (세분화됨)
    signal_reason: str = ""  # 신호 이유 (예: "zigzag_pivot_low,ADX_strong")
    stop_loss: Optional[float] = None  # 손절 가격
    take_profit: Optional[float] = None  # 이익실현 가격
    atr: Optional[float] = None  # ATR 값
    
    # 동적 리스크 관리 필드
    position_id: Optional[str] = None  # 포지션 ID
    trailing_stops: List[dict] = field(default_factory=list)  # 트레일링 스탑 기록
    atr_snapshots: List[float] = field(default_factory=list)  # ATR 스냅샷
    partial_exits: List[dict] = field(default_factory=list)  # 부분 청산 기록
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


# 세분화된 청산 사유 상수
class ExitReason:
    """세분화된 청산 사유."""
    
    # stop_loss 관련
    STOP_LOSS_INITIAL = "stop_loss_initial"  # 초기 손절
    STOP_LOSS_TRAILING = "stop_loss_trailing"  # 트레일링 스탑
    STOP_LOSS_ATR_SPIKE = "stop_loss_atr_spike"  # ATR 급증으로 인한 손절
    
    # take_profit 관련
    TAKE_PROFIT_INITIAL = "take_profit_initial"  # 초기 이익실현
    TAKE_PROFIT_PARTIAL = "take_profit_partial"  # 부분 이익실현
    TAKE_PROFIT_TRAILING = "take_profit_trailing"  # 트레일링 이익실현
    
    # 시장 관련
    MARKET_CLOSE = "market_close"  # 장 마감 강제 청산
    LIQUIDITY_ISSUE = "liquidity_issue"  # 유동성 부족
    VOLATILITY_SPIKE = "volatility_spike"  # 변동성 급증
    
    # 시스템 관련
    MANUAL_OVERRIDE = "manual_override"  # 수동 개입
    SYSTEM_ERROR = "system_error"  # 시스템 오류
    TIMEOUT = "timeout"  # 타임아웃
    
    # 신호 관련 (기존 호환)
    SIGNAL_REVERSAL = "signal_reversal"  # 반대 신호
    
    # 기존 호환 (간단 버전)
    STOP_LOSS = "stop_loss"  # 손절 (간단)
    TAKE_PROFIT = "take_profit"  # 이익실현 (간단)


@dataclass
class RiskMetricsEvent:
    """리스크 메트릭 이벤트."""
    timestamp: datetime
    position_id: str
    current_price: float
    atr: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    distance_to_stop: float
    distance_to_take_profit: float
    max_favorable_excursion: float
    max_adverse_excursion: float
    risk_reward_ratio: float
    position_size_pct: float
    confidence: str
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class PositionState:
    """포지션 상태."""
    position_id: str
    action: str  # "BUY" or "SELL"
    entry_price: float
    entry_time: datetime
    size: float
    confidence: str
    signal_reason: str
    stop_loss: float
    take_profit: float
    atr: float
    current_stop: float  # 현재 스탑 가격 (트레일링 스탑 포함)
    trailing_stops: List[dict] = field(default_factory=list)  # 트레일링 스탑 기록
    atr_snapshots: List[float] = field(default_factory=list)  # ATR 스냅샷
    partial_exits: List[dict] = field(default_factory=list)  # 부분 청산 기록
    is_active: bool = True  # 활성 여부


class PositionTracker:
    """포지션 상태 추적."""
    
    def __init__(self):
        """초기화."""
        self.positions: Dict[str, PositionState] = {}
    
    def create_position(
        self,
        action: str,
        entry_price: float,
        size: float,
        confidence: str,
        signal_reason: str,
        stop_loss: float,
        take_profit: float,
        atr: float
    ) -> str:
        """새 포지션 생성.
        
        Args:
            action: 액션 (BUY/SELL)
            entry_price: 진입 가격
            size: 사이즈
            confidence: 신뢰도
            signal_reason: 신호 이유
            stop_loss: 손절 가격
            take_profit: 이익실현 가격
            atr: ATR 값
        
        Returns:
            포지션 ID
        """
        position_id = str(uuid.uuid4())
        
        position = PositionState(
            position_id=position_id,
            action=action,
            entry_price=entry_price,
            entry_time=datetime.now(),
            size=size,
            confidence=confidence,
            signal_reason=signal_reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            current_stop=stop_loss
        )
        
        self.positions[position_id] = position
        _logger.info("[POS_TRACKER] 포지션 생성: %s %s @ %.2f (size=%.2f)", action, position_id, entry_price, size)
        
        # 자동 진입 로그 기록 (전역 TradeLogger 인스턴스 사용)
        try:
            global_logger = get_trade_logger()
            global_logger.log_entry(
                action=action,
                price=entry_price,
                size=size,
                confidence=confidence,
                signal_reason=signal_reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr=atr
            )
            _logger.info("[POS_TRACKER] 진입 로그 자동 기록됨")
        except Exception as e:
            _logger.warning("[POS_TRACKER] 진입 로그 기록 실패: %s", e)
        
        return position_id
    
    def update_position(
        self,
        position_id: str,
        current_price: float,
        atr: float,
        trailing_stop_multiplier: float = 1.5
    ) -> Optional[float]:
        """포지션 상태 업데이트 (트레일링 스탑).
        
        Args:
            position_id: 포지션 ID
            current_price: 현재 가격
            atr: ATR 값
            trailing_stop_multiplier: 트레일링 스탑 멀티플라이어
        
        Returns:
            새로운 스탑 가격 (변경 시), None (변경 없음)
        """
        if position_id not in self.positions:
            _logger.warning("[POS_TRACKER] 포지션 없음: %s", position_id)
            return None
        
        pos = self.positions[position_id]
        
        if not pos.is_active:
            return None
        
        # 트레일링 스탑 계산
        if pos.action == "BUY":
            new_stop = current_price - (atr * trailing_stop_multiplier)
            # 스탑은 이전보다 높아야 함 (익절 보호)
            new_stop = max(new_stop, pos.current_stop)
        else:  # SELL
            new_stop = current_price + (atr * trailing_stop_multiplier)
            # 스탑은 이전보다 낮아야 함 (익절 보호)
            new_stop = min(new_stop, pos.current_stop)
        
        # 스탑 변경 시 기록
        if new_stop != pos.current_stop:
            pos.current_stop = new_stop
            pos.trailing_stops.append({
                "timestamp": datetime.now().isoformat(),
                "new_stop": new_stop,
                "current_price": current_price,
                "atr": atr
            })
            _logger.info(
                "[POS_TRACKER] 트레일링 스탑 업데이트: %s %.2f -> %.2f (price=%.2f)",
                position_id, pos.stop_loss, new_stop, current_price
            )
            return new_stop
        
        return None
    
    def add_atr_snapshot(self, position_id: str, atr: float):
        """ATR 스냅샷 추가.
        
        Args:
            position_id: 포지션 ID
            atr: ATR 값
        """
        if position_id not in self.positions:
            return
        
        pos = self.positions[position_id]
        pos.atr_snapshots.append(atr)
    
    def add_partial_exit(
        self,
        position_id: str,
        exit_price: float,
        exit_size: float,
        reason: str
    ):
        """부분 청산 기록.
        
        Args:
            position_id: 포지션 ID
            exit_price: 청산 가격
            exit_size: 청산 사이즈
            reason: 청산 사유
        """
        if position_id not in self.positions:
            return
        
        pos = self.positions[position_id]
        pos.partial_exits.append({
            "timestamp": datetime.now().isoformat(),
            "exit_price": exit_price,
            "exit_size": exit_size,
            "reason": reason
        })
        
        # 남은 사이즈 업데이트
        pos.size -= exit_size
        
        _logger.info(
            "[POS_TRACKER] 부분 청산: %s %.2f/%.2f @ %.2f (reason=%s)",
            position_id, exit_size, pos.size + exit_size, exit_price, reason
        )
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str
    ) -> Optional[PositionState]:
        """포지션 청산.
        
        Args:
            position_id: 포지션 ID
            exit_price: 청산 가격
            reason: 청산 사유
        
        Returns:
            청산된 포지션 상태
        """
        if position_id not in self.positions:
            _logger.warning("[POS_TRACKER] 포지션 없음: %s", position_id)
            return None
        
        pos = self.positions[position_id]
        pos.is_active = False
        
        # 수익 계산
        if pos.action == "BUY":
            profit = (exit_price - pos.entry_price) * pos.size
        else:
            profit = (pos.entry_price - exit_price) * pos.size
        
        _logger.info(
            "[POS_TRACKER] 포지션 청산: %s @ %.2f (reason=%s, profit=%.2f)",
            position_id, exit_price, reason, profit
        )
        
        # 자동 청산 로그 기록 (전역 TradeLogger 인스턴스 사용)
        try:
            global_logger = get_trade_logger()
            global_logger.log_exit(
                action=pos.action,
                price=exit_price,
                size=pos.size,
                confidence=pos.confidence,
                reason=reason
            )
            _logger.info("[POS_TRACKER] 청산 로그 자동 기록됨")
        except Exception as e:
            _logger.warning("[POS_TRACKER] 청산 로그 기록 실패: %s", e)
        
        return pos
    
    def get_position(self, position_id: str) -> Optional[PositionState]:
        """포지션 상태 조회.
        
        Args:
            position_id: 포지션 ID
        
        Returns:
            포지션 상태
        """
        return self.positions.get(position_id)
    
    def get_active_positions(self) -> List[PositionState]:
        """활성 포지션 리스트 반환.
        
        Returns:
            활성 포지션 리스트
        """
        return [pos for pos in self.positions.values() if pos.is_active]
    
    def should_exit(
        self,
        position_id: str,
        current_price: float
    ) -> tuple[bool, Optional[str]]:
        """청산 여부 판단.
        
        Args:
            position_id: 포지션 ID
            current_price: 현재 가격
        
        Returns:
            (청산 여부, 청산 사유)
        """
        if position_id not in self.positions:
            return False, None
        
        pos = self.positions[position_id]
        
        if not pos.is_active:
            return False, None
        
        # 손절 체크
        if pos.action == "BUY":
            if current_price <= pos.current_stop:
                return True, "stop_loss"
            elif current_price >= pos.take_profit:
                return True, "take_profit"
        else:  # SELL
            if current_price >= pos.current_stop:
                return True, "stop_loss"
            elif current_price <= pos.take_profit:
                return True, "take_profit"
        
        return False, None
    
    def calculate_risk_metrics(
        self,
        position_id: str,
        current_price: float,
        atr: float,
        position_size_pct: float,
        capital: float
    ) -> Optional[dict]:
        """리스크 메트릭 계산.
        
        Args:
            position_id: 포지션 ID
            current_price: 현재 가격
            atr: ATR 값
            position_size_pct: 포지션 사이즈 비율
            capital: 총 자본
        
        Returns:
            리스크 메트릭 딕셔너리
        """
        if position_id not in self.positions:
            return None
        
        pos = self.positions[position_id]
        
        if not pos.is_active:
            return None
        
        # 미실현 손익 계산
        if pos.action == "BUY":
            unrealized_pnl = (current_price - pos.entry_price) * pos.size
        else:  # SELL
            unrealized_pnl = (pos.entry_price - current_price) * pos.size
        
        unrealized_pnl_pct = unrealized_pnl / (pos.entry_price * pos.size) * 100
        
        # 손절/이익실현까지 거리
        distance_to_stop = abs(current_price - pos.current_stop)
        distance_to_take_profit = abs(current_price - pos.take_profit)
        
        # 리스크/리워드 비율
        potential_profit = abs(pos.take_profit - pos.entry_price)
        potential_loss = abs(pos.entry_price - pos.current_stop)
        risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 else 0.0
        
        return {
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "distance_to_stop": distance_to_stop,
            "distance_to_take_profit": distance_to_take_profit,
            "max_favorable_excursion": pos.max_favorable_excursion,
            "max_adverse_excursion": pos.max_adverse_excursion,
            "risk_reward_ratio": risk_reward_ratio,
            "position_size_pct": position_size_pct,
            "confidence": pos.confidence
        }



class TradeLogger:
    """실시간 거래 이벤트 로거."""
    
    def __init__(self, log_dir: Path = Path("logs/trades")):
        """초기화.
        
        Args:
            log_dir: 로그 디렉토리
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self._get_log_file()
        self.backup_dir = log_dir / "backup"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = 3
        self._notifier = None
        _logger.info("[TRADE_LOGGER] 로거 초기화: %s", self.current_log_file)
    
    def _get_log_file(self) -> Path:
        """오늘 날짜의 로그 파일 경로 반환."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"trades_{today}.jsonl"
    
    def log_event(self, event: TradeEvent):
        """이벤트 로그 기록 (강화된 에러 핸들링).
        
        Args:
            event: 거래 이벤트
        """
        log_entry = event.to_dict()
        
        # 재시도 로직
        for attempt in range(self.max_retries):
            try:
                with open(self.current_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
                _logger.info(
                    "[TRADE_LOGGER] 이벤트 기록: %s %s @ %.2f (size=%.2f, conf=%s, reason=%s)",
                    event.event_type, event.action, event.price, event.size, event.confidence, event.reason
                )
                return  # 성공 시 종료
                
            except IOError as e:
                if attempt == self.max_retries - 1:
                    # 최종 실패 시 백업 경로 시도
                    _logger.error("[TRADE_LOGGER] 이벤트 기록 실패 (시도 %d/%d): %s", attempt + 1, self.max_retries, e)
                    self._log_to_backup(log_entry)
                    self._notify_failure(event, e)
                else:
                    _logger.warning("[TRADE_LOGGER] 이벤트 기록 실패 (시도 %d/%d): %s, 재시도...", attempt + 1, self.max_retries, e)
                    # 잠시 대기 후 재시도
                    import time
                    time.sleep(0.1)
                    
            except Exception as e:
                _logger.error("[TRADE_LOGGER] 이벤트 기록 실패 (시도 %d/%d): %s", attempt + 1, self.max_retries, e)
                self._log_error_to_file(event, e)
                if attempt == self.max_retries - 1:
                    self._notify_failure(event, e)
                break
    
    def _log_to_backup(self, log_entry: dict):
        """백업 경로에 로그 기록.
        
        Args:
            log_entry: 로그 엔트리
        """
        try:
            backup_file = self.backup_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            with open(backup_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            _logger.warning("[TRADE_LOGGER] 백업 경로에 기록: %s", backup_file)
        except Exception as e:
            _logger.error("[TRADE_LOGGER] 백업 경로 기록 실패: %s", e)
    
    def _log_error_to_file(self, event: TradeEvent, error: Exception):
        """에러를 별도 파일에 기록.
        
        Args:
            event: 거래 이벤트
            error: 에러 객체
        """
        try:
            error_file = self.log_dir / f"errors_{datetime.now().strftime('%Y-%m-%d')}.log"
            with open(error_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] ERROR: {type(error).__name__}: {error}\n")
                f.write(f"Event: {event.to_dict()}\n")
                f.write("-" * 80 + "\n")
        except Exception as e:
            _logger.error("[TRADE_LOGGER] 에러 로그 기록 실패: %s", e)
    
    def _notify_failure(self, event: TradeEvent, error: Exception):
        """실패 시 알림.
        
        Args:
            event: 거래 이벤트
            error: 에러 객체
        """
        try:
            if self._notifier:
                # 알림 전송 (구현 필요)
                message = "⚠️ 거래 로그 기록 실패\n"
                message += f"이벤트: {event.event_type} {event.action}\n"
                message += f"에러: {type(error).__name__}: {error}"
                # self._notifier.send_alert(message)
                _logger.warning("[TRADE_LOGGER] 알림 전송: %s", message[:100])
        except Exception as e:
            _logger.error("[TRADE_LOGGER] 알림 전송 실패: %s", e)
    
    def set_notifier(self, notifier):
        """알림 인스턴스 설정.
        
        Args:
            notifier: 알림 인스턴스
        """
        self._notifier = notifier
    
    def log_entry(
        self,
        action: str,
        price: float,
        size: float,
        confidence: str,
        signal_reason: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        atr: Optional[float] = None
    ):
        """진입 이벤트 로그 기록 (편의 메서드).
        
        Args:
            action: 액션 (BUY/SELL)
            price: 가격
            size: 사이즈
            confidence: 신뢰도
            signal_reason: 신호 이유
            stop_loss: 손절 가격
            take_profit: 이익실현 가격
            atr: ATR 값
        """
        event = TradeEvent(
            event_type="ENTRY",
            timestamp=datetime.now(),
            action=action,
            price=price,
            size=size,
            confidence=confidence,
            reason=None,
            signal_reason=signal_reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr
        )
        self.log_event(event)
    
    def log_exit(
        self,
        action: str,
        price: float,
        size: float,
        confidence: str,
        reason: str,
        signal_reason: str = ""
    ):
        """청산 이벤트 로그 기록 (편의 메서드).
        
        Args:
            action: 액션 (BUY/SELL)
            price: 가격
            size: 사이즈
            confidence: 신뢰도
            reason: 청산 사유
            signal_reason: 신호 이유
        """
        event = TradeEvent(
            event_type="EXIT",
            timestamp=datetime.now(),
            action=action,
            price=price,
            size=size,
            confidence=confidence,
            reason=reason,
            signal_reason=signal_reason
        )
        self.log_event(event)
    
    def log_trailing_stop_update(
        self,
        position_id: str,
        new_stop: float,
        current_price: float,
        atr: float
    ):
        """트레일링 스탑 업데이트 로그 기록.
        
        Args:
            position_id: 포지션 ID
            new_stop: 새로운 스탑 가격
            current_price: 현재 가격
            atr: ATR 값
        """
        event = TradeEvent(
            event_type="TRAILING_STOP",
            timestamp=datetime.now(),
            action="",
            price=current_price,
            size=0.0,
            confidence="",
            reason=f"trailing_stop:{new_stop}",
            signal_reason="",
            position_id=position_id,
            atr=atr
        )
        
        # 트레일링 스탑 기록
        trailing_stop_record = {
            "timestamp": datetime.now().isoformat(),
            "new_stop": new_stop,
            "current_price": current_price,
            "atr": atr
        }
        event.trailing_stops.append(trailing_stop_record)
        
        self.log_event(event)
    
    def log_atr_snapshot(
        self,
        position_id: str,
        atr: float
    ):
        """ATR 스냅샷 로그 기록.
        
        Args:
            position_id: 포지션 ID
            atr: ATR 값
        """
        event = TradeEvent(
            event_type="ATR_SNAPSHOT",
            timestamp=datetime.now(),
            action="",
            price=0.0,
            size=0.0,
            confidence="",
            reason=f"atr:{atr}",
            signal_reason="",
            position_id=position_id,
            atr=atr
        )
        event.atr_snapshots.append(atr)
        self.log_event(event)
    
    def log_risk_metrics(
        self,
        position_id: str,
        current_price: float,
        atr: float,
        unrealized_pnl: float,
        unrealized_pnl_pct: float,
        distance_to_stop: float,
        distance_to_take_profit: float,
        max_favorable_excursion: float,
        max_adverse_excursion: float,
        risk_reward_ratio: float,
        position_size_pct: float,
        confidence: str
    ):
        """리스크 메트릭 로그 기록.
        
        Args:
            position_id: 포지션 ID
            current_price: 현재 가격
            atr: ATR 값
            unrealized_pnl: 미실현 손익
            unrealized_pnl_pct: 미실현 손익률
            distance_to_stop: 손절까지 거리
            distance_to_take_profit: 이익실현까지 거리
            max_favorable_excursion: 최대 호재
            max_adverse_excursion: 최대 악재
            risk_reward_ratio: 리스크/리워드 비율
            position_size_pct: 포지션 사이즈 비율
            confidence: 신뢰도
        """
        event = RiskMetricsEvent(
            timestamp=datetime.now(),
            position_id=position_id,
            current_price=current_price,
            atr=atr,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            distance_to_stop=distance_to_stop,
            distance_to_take_profit=distance_to_take_profit,
            max_favorable_excursion=max_favorable_excursion,
            max_adverse_excursion=max_adverse_excursion,
            risk_reward_ratio=risk_reward_ratio,
            position_size_pct=position_size_pct,
            confidence=confidence
        )
        
        # 리스크 메트릭은 별도 파일에 저장
        risk_log_file = self.log_dir / f"risk_metrics_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(risk_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        
        _logger.debug(
            "[TRADE_LOGGER] 리스크 메트릭 기록: %s pnl=%.2f (%.2f%%)",
            position_id, unrealized_pnl, unrealized_pnl_pct
        )
    
    def load_events(self, log_file: Optional[Path] = None) -> List[dict]:
        """로그 파일에서 이벤트 로드.
        
        Args:
            log_file: 로그 파일 (None이면 현재 파일)
        
        Returns:
            이벤트 리스트
        """
        target_file = log_file or self.current_log_file
        
        if not target_file.exists():
            _logger.warning("[TRADE_LOGGER] 로그 파일 없음: %s", target_file)
            return []
        
        events = []
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
            
            _logger.info("[TRADE_LOGGER] 이벤트 로드: %d개 (%s)", len(events), target_file)
        except Exception as e:
            _logger.error("[TRADE_LOGGER] 이벤트 로드 실패: %s", e)
        
        return events
    
    def get_daily_summary(self, log_file: Optional[Path] = None) -> dict:
        """일일 거래 요약 반환.
        
        Args:
            log_file: 로그 파일 (None이면 현재 파일)
        
        Returns:
            요약 딕셔너리
        """
        events = self.load_events(log_file)
        
        entries = [e for e in events if e["event_type"] == "ENTRY"]
        exits = [e for e in events if e["event_type"] == "EXIT"]
        
        # 완료된 거래 계산
        completed_trades = []
        for entry in entries:
            # 해당 진입의 청산 찾기
            matching_exits = [e for e in exits if e["timestamp"] > entry["timestamp"]]
            if matching_exits:
                # 가장 빠른 청산 사용
                exit = min(matching_exits, key=lambda x: x["timestamp"])
                
                profit = None
                if entry["action"] == "BUY":
                    profit = (exit["price"] - entry["price"]) * entry["size"]
                else:  # SELL
                    profit = (entry["price"] - exit["price"]) * entry["size"]
                
                completed_trades.append({
                    "entry_time": entry["timestamp"],
                    "exit_time": exit["timestamp"],
                    "action": entry["action"],
                    "entry_price": entry["price"],
                    "exit_price": exit["price"],
                    "profit": profit,
                    "reason": exit["reason"]
                })
        
        # 통계
        total_trades = len(completed_trades)
        win_trades = sum(1 for t in completed_trades if t["profit"] and t["profit"] > 0)
        loss_trades = sum(1 for t in completed_trades if t["profit"] and t["profit"] < 0)
        total_profit = sum(t["profit"] for t in completed_trades if t["profit"])
        
        summary = {
            "total_entries": len(entries),
            "total_exits": len(exits),
            "completed_trades": total_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "win_rate": win_trades / total_trades if total_trades > 0 else 0,
            "total_profit": total_profit,
            "avg_profit_per_trade": total_profit / total_trades if total_trades > 0 else 0
        }
        
        _logger.info("[TRADE_LOGGER] 일일 요약: %s", summary)
        return summary


# 전역 로거 인스턴스
_global_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """전역 거래 로거 인스턴스 반환."""
    global _global_logger
    if _global_logger is None:
        _global_logger = TradeLogger()
    return _global_logger


# 전역 포지션 트래커 인스턴스
_global_position_tracker: Optional[PositionTracker] = None


def get_position_tracker() -> PositionTracker:
    """전역 포지션 트래커 인스턴스 반환."""
    global _global_position_tracker
    if _global_position_tracker is None:
        _global_position_tracker = PositionTracker()
    return _global_position_tracker
