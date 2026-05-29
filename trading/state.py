"""
trade_state.py
==============
장중 진입/청산 로직의 상태 데이터 모델.

포지션 상태, 일일 거래 기록, 시간대 슬롯 관리를 담당한다.
TradeExecutionGate(trade_gate.py)가 이 모델을 사용하며,
상태 자체는 로직을 포함하지 않는다.

변경 이력
---------
- v1.0  : 초기 구현
- v1.1  : TradeRecord.trade_id 추가 (마이크로초 기반 고유 ID)
- v1.2  : Phase 3 — entry_atm_iv / entry_atm_delta / entry_net_gamma /
          entry_above_vol_trigger / entry_target_pt / entry_stop_pt 필드 추가
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import List, Optional, Dict, Any


# ── 상수 ─────────────────────────────────────────────────────────────────────

class PositionSide(str, Enum):
    NONE  = "NONE"
    LONG  = "LONG"   # 매수 포지션
    SHORT = "SHORT"  # 매도 포지션


class CloseReason(str, Enum):
    TARGET_PROFIT  = "목표수익"
    STOP_LOSS      = "손절"
    REVERSE_SIGNAL = "반대신호"
    FORCE_CLOSE    = "강제청산"    # 14:50 시각 도달
    TRAILING_STOP  = "트레일링손절"  # Trailing Stop-loss


class TradeSlot(str, Enum):
    """하루 3개 시간대 슬롯 — 슬롯당 최대 1회 진입."""
    A = "A"  # 09:05 ~ slot_a_end
    B = "B"  # slot_a_end ~ slot_b_end
    C = "C"  # slot_b_end ~ force_close_time


# ── trade_id 생성 유틸 ───────────────────────────────────────────────────────

def _make_trade_id(dt: Optional[datetime] = None) -> str:
    """마이크로초 기반 고유 거래 ID를 생성한다.

    형식: ``YYYYMMDD_HHMMSS_ffffff``
    예시: ``20260322_100512_834291``

    같은 마이크로초에 두 건이 생성될 확률은 극히 낮으나,
    충돌 가능성이 우려되면 suffix 에 int(time.monotonic_ns() % 1_000_000)
    를 추가할 수 있다.
    """
    ts = dt if dt is not None else datetime.now()
    return ts.strftime("%Y%m%d_%H%M%S_") + f"{ts.microsecond:06d}"


# ── 거래 기록 ────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """완결된 단일 거래 기록."""

    slot:              TradeSlot
    side:              PositionSide
    entry_price:       float
    entry_time:        datetime
    entry_signal:      str           # "BUY" | "SELL"
    entry_confidence:  str           # "HIGH" | "MEDIUM" | "LOW"
    entry_prob:        float

    # 마이크로초 기반 고유 거래 ID — 미지정 시 entry_time 기준으로 자동 생성
    trade_id: str = field(default="")

    # Phase 3 — 진입 시점 옵션 컨텍스트 (동적 목표/손절 계산 및 기록용)
    entry_atm_iv:            float = 0.0   # ATM 내재변동성 (소수, 예: 0.20 = 20%)
    entry_atm_delta:         float = 0.0   # ATM 옵션 Delta (절대값, 0~1)
    entry_net_gamma:         float = 0.0   # net_gamma_proxy (양수=딜러 Long Gamma)
    entry_above_vol_trigger: float = 1.0   # 1.0=Vol Trigger 위(안정), 0.0=아래
    entry_target_pt:         float = 0.0   # 실제 적용된 목표 수익 (pt)
    entry_stop_pt:           float = 0.0   # 실제 적용된 손절 (pt)
    
    # 피봇 기반 전략용
    pivot_price: float = 0.0  # 진입 시점 피봇 가격

    # 포지션 사이징
    position_size: float = 0.0  # 포지션 사이즈 (계약/주 수)
    capital_used: float = 0.0  # 사용 자본
    risk_amount: float = 0.0  # 리스크 금액
    risk_pct: float = 0.0  # 리스크 비율
    sizing_method: str = ""  # 사용된 사이징 방법

    close_price:  float           = 0.0
    close_time:   Optional[datetime] = None
    close_reason: Optional[CloseReason] = None

    def __post_init__(self) -> None:
        # trade_id 가 비어 있으면 entry_time 기준으로 자동 채움
        if not self.trade_id:
            self.trade_id = _make_trade_id(self.entry_time)

    # ── 파생 프로퍼티 ─────────────────────────────────────────────────────────

    @property
    def is_closed(self) -> bool:
        return self.close_time is not None

    @property
    def pnl_pt(self) -> float:
        """손익 (선물 포인트). 미청산이면 0.0."""
        if not self.is_closed or self.close_price == 0.0:
            return 0.0
        if self.side == PositionSide.LONG:
            return self.close_price - self.entry_price
        return self.entry_price - self.close_price

    @property
    def hold_minutes(self) -> float:
        """보유 시간 (분). 미청산이면 0.0."""
        if not self.is_closed or self.close_time is None:
            return 0.0
        delta = self.close_time - self.entry_time
        return delta.total_seconds() / 60.0

    # ── 직렬화 ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "trade_id":               self.trade_id,
            "slot":                   self.slot.value,
            "side":                   self.side.value,
            "entry_price":            self.entry_price,
            "entry_time":             self.entry_time.isoformat(),
            "entry_signal":           self.entry_signal,
            "entry_confidence":       self.entry_confidence,
            "entry_prob":             self.entry_prob,
            "entry_atm_iv":           self.entry_atm_iv,
            "entry_atm_delta":        self.entry_atm_delta,
            "entry_net_gamma":        self.entry_net_gamma,
            "entry_above_vol_trigger": self.entry_above_vol_trigger,
            "entry_target_pt":        self.entry_target_pt,
            "entry_stop_pt":          self.entry_stop_pt,
            "position_size":          self.position_size,
            "capital_used":           self.capital_used,
            "risk_amount":            self.risk_amount,
            "risk_pct":               self.risk_pct,
            "sizing_method":          self.sizing_method,
            "close_price":            self.close_price,
            "close_time":             self.close_time.isoformat() if self.close_time else None,
            "close_reason":           self.close_reason.value if self.close_reason else None,
            "pnl_pt":                 round(self.pnl_pt, 2),
            "hold_minutes":           round(self.hold_minutes, 1),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TradeRecord":
        """to_dict() 역직렬화. JSONL 로드 시 사용."""
        def _dt(v: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(v) if v else None

        rec = cls(
            trade_id=          d.get("trade_id", ""),
            slot=              TradeSlot(d["slot"]),
            side=              PositionSide(d["side"]),
            entry_price=       float(d["entry_price"]),
            entry_time=        _dt(d["entry_time"]),          # type: ignore[arg-type]
            entry_signal=      d["entry_signal"],
            entry_confidence=  d["entry_confidence"],
            entry_prob=        float(d["entry_prob"]),
            entry_atm_iv=      float(d.get("entry_atm_iv", 0.0)),
            entry_atm_delta=   float(d.get("entry_atm_delta", 0.0)),
            entry_net_gamma=   float(d.get("entry_net_gamma", 0.0)),
            entry_above_vol_trigger= float(d.get("entry_above_vol_trigger", 1.0)),
            entry_target_pt=   float(d.get("entry_target_pt", 0.0)),
            entry_stop_pt=     float(d.get("entry_stop_pt", 0.0)),
            position_size=     float(d.get("position_size", 0.0)),
            capital_used=      float(d.get("capital_used", 0.0)),
            risk_amount=       float(d.get("risk_amount", 0.0)),
            risk_pct=          float(d.get("risk_pct", 0.0)),
            sizing_method=     d.get("sizing_method", ""),
            close_price=       float(d.get("close_price", 0.0)),
            close_time=        _dt(d.get("close_time")),
            close_reason=      CloseReason(d["close_reason"]) if d.get("close_reason") else None,
        )
        # from_dict 후 trade_id 가 비어있으면 entry_time 으로 채움
        if not rec.trade_id:
            rec.trade_id = _make_trade_id(rec.entry_time)
        return rec


# ── 현재 포지션 ───────────────────────────────────────────────────────────────

@dataclass
class ActivePosition:
    """현재 보유 중인 포지션."""

    record:             TradeRecord
    consecutive_reverse: int = 0   # 반대 신호 연속 카운터 (청산 트리거용)
    trailing_stop_price: float = 0.0  # Trailing Stop 가격 (0=비활성)
    pivot_price: float = 0.0  # 피봇 가격 (피봇 기반 전략용)

    @property
    def side(self) -> PositionSide:
        return self.record.side

    @property
    def entry_price(self) -> float:
        return self.record.entry_price

    @property
    def entry_time(self) -> datetime:
        return self.record.entry_time


# ── 일일 상태 ────────────────────────────────────────────────────────────────

@dataclass
class DailyState:
    """하루 단위로 리셋되는 거래 상태."""

    date:       str = ""                          # "YYYY-MM-DD"
    used_slots: List[TradeSlot] = field(default_factory=list)   # 이미 사용한 슬롯
    trade_log:  List[TradeRecord] = field(default_factory=list)  # 완결 거래 목록
    active:     Optional[ActivePosition] = None  # 현재 포지션 (None = 없음)

    # 신호 연속성 추적 (진입 조건 판단용)
    consecutive_signal: str = ""   # 현재 연속 중인 신호 ("BUY"/"SELL"/"")
    consecutive_count:  int = 0    # 연속 횟수

    # 당일 선물 시가 — ATM IV 기반 동적 목표/손절 계산의 기준가
    # 09:05 이후 첫 번째 유효 현재가가 기록되고, 이후 갱신되지 않는다.
    daily_open_price: float = 0.0

    # 신뢰도별 통계 {"HIGH": {"total": N, "wins": M}, "MEDIUM": ..., "LOW": ...}
    confidence_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # 리스크 관리
    consecutive_losses: int = 0  # 연속 손실 횟수

    # 일일 총 손익 (퍼센트 단위)
    total_pnl_pct: float = 0.0  # 누적 퍼센트 손익

    # 슬롯별 성과 기반 할당 {"A": {"total": N, "wins": M, "pnl": P}, ...}
    slot_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ── 파생 프로퍼티 ─────────────────────────────────────────────────────────

    @property
    def daily_count(self) -> int:
        return len(self.trade_log)

    @property
    def has_position(self) -> bool:
        return self.active is not None

    @property
    def wins(self) -> int:
        return sum(1 for r in self.trade_log if r.is_closed and r.pnl_pt > 0)

    @property
    def losses(self) -> int:
        return sum(1 for r in self.trade_log if r.is_closed and r.pnl_pt < 0)

    @property
    def total_pnl_pt(self) -> float:
        """거래 로그의 총 포인트 손익 (호환성 유지용)."""
        return sum(r.pnl_pt for r in self.trade_log if r.is_closed)

    def slot_used(self, slot: TradeSlot) -> bool:
        return slot in self.used_slots

    def summary_dict(self) -> dict:
        total = self.daily_count
        wins  = self.wins
        return {
            "date":         self.date,
            "count":        total,
            "wins":         wins,
            "losses":       self.losses,
            "win_rate":     round(wins / (wins + self.losses), 4) if (wins + self.losses) > 0 else 0.0,
            "total_pnl_pt": round(self.total_pnl_pt, 2),  # 호환성 유지
            "total_pnl_pct": round(self.total_pnl_pct, 2),  # 퍼센트 단위
            "trades":       [r.to_dict() for r in self.trade_log],
        }


# ── 슬롯 유틸 ────────────────────────────────────────────────────────────────

def get_trade_slot(
    now: datetime,
    *,
    slot_a_end:   str = "10:30",
    slot_b_end:   str = "13:00",
    force_close:  str = "14:50",
    market_open:  str = "09:05",
) -> Optional[TradeSlot]:
    """현재 시각이 어느 시간대 슬롯에 속하는지 반환.

    Args:
        now:         현재 시각.
        slot_a_end:  슬롯 A 종료 시각 문자열 (``"HH:MM"``).
        slot_b_end:  슬롯 B 종료 시각 (``"HH:MM"``).
        force_close: 강제청산/신규진입 금지 시각 (``"HH:MM"``).
        market_open: 장 시작(신규진입 허용 시작) 시각 (``"HH:MM"``).

    Returns:
        TradeSlot.A / B / C, 또는 None (진입 금지 시간대).
    """
    def _t(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    now_t = now.time()
    if now_t < _t(market_open) or now_t >= _t(force_close):
        return None  # 장 시작 전 또는 강제청산 이후 — 신규 진입 금지
    if now_t < _t(slot_a_end):
        return TradeSlot.A
    if now_t < _t(slot_b_end):
        return TradeSlot.B
    return TradeSlot.C


# ── 스레드 안전 컨테이너 ──────────────────────────────────────────────────────

class TradeStateManager:
    """DailyState 를 스레드 안전하게 관리하는 컨테이너.

    TradeExecutionGate 가 내부에서 직접 접근하므로
    외부 호출자는 이 클래스 대신 TradeExecutionGate 의 공개 메서드를 사용한다.
    """

    def __init__(self, lock: Optional[threading.RLock] = None) -> None:
        self._lock = lock or threading.RLock()  # 외부에서 주입된 락 또는 새 RLock 생성
        self._state = DailyState()

    def get_state(self) -> DailyState:
        """현재 상태 스냅샷 반환 (읽기 전용 용도)."""
        with self._lock:
            return self._state

    def _ensure_today(self, today_str: str) -> None:
        """날짜가 바뀌면 DailyState 를 리셋한다 (lock 보유 상태에서 호출)."""
        if self._state.date != today_str:
            self._state = DailyState(date=today_str)

    def update(self, today_str: str, fn) -> None:
        """lock 을 보유한 채 fn(state) 를 실행한다.

        Args:
            today_str: ``"YYYY-MM-DD"`` 형식의 오늘 날짜 문자열.
            fn:        ``DailyState → None`` 형식의 콜백 (상태를 직접 변경).
        """
        with self._lock:
            self._ensure_today(today_str)
            fn(self._state)
