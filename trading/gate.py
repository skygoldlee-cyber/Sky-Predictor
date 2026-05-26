"""
trade_gate.py
=============
장중 진입/청산 판단 게이트.

5분 주기 예측 신호를 받아 실제 진입/청산 결정을 내리고
텔레그램 알림을 전송한다.

설계 원칙:
  - 하루 최대 3회 진입 (시간대 슬롯 A/B/C 각 1회)
  - 신호 연속성 조건 충족 시에만 진입 허용
  - 포지션 보유 중 반대 신호 N회 연속 → 청산
  - 14:50 강제청산
  - enabled=False(기본) 시 완전 비활성 — 기존 예측 흐름에 영향 없음

사용법:
    gate = TradeExecutionGate(notifier, config.trade_gate)
    # PipelineTelegramBridge 의 예측 루프 내에서 매 틱 호출
    gate.on_signal(result, current_price=price)
    # 청산 감시는 별도 루프에서 주기적으로 호출
    gate.check_close(current_price=price)
"""

from __future__ import annotations

import logging
import threading
import time as _time
from datetime import datetime
from typing import Any, Dict, Optional

from .state import (
    ActivePosition,
    CloseReason,
    DailyState,
    PositionSide,
    TradeRecord,
    TradeSlot,
    TradeStateManager,
    get_trade_slot,
)

# 이벤트 시스템 (선택적)
try:
    from events import EventBus, TradeEntryEvent, TradeExitEvent, RiskLimitEvent, SignalEvent
    EVENTS_AVAILABLE = True
except ImportError:
    EVENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── 설정 기본값 상수 ─────────────────────────────────────────────────────────

_DEFAULT_MAX_DAILY = 3
_DEFAULT_MIN_CONSECUTIVE = 2
_DEFAULT_MIN_CONFIDENCE = "MEDIUM"
_DEFAULT_MIN_PROB_BUY = 0.62
_DEFAULT_MAX_PROB_SELL = 0.38
_DEFAULT_REQUIRE_CONSENSUS = True
_DEFAULT_TARGET_PT = 2.0
_DEFAULT_STOP_PT = 1.0
_DEFAULT_FORCE_CLOSE = "14:50"
_DEFAULT_SLOT_A_END = "10:30"
_DEFAULT_SLOT_B_END = "13:00"
_DEFAULT_MARKET_OPEN = "09:05"
_DEFAULT_REVERSE_CLOSE_COUNT = 2

# ── 리스크 관리 ─────────────────────────────────────────────────────────
_DEFAULT_MAX_CONSECUTIVE_LOSSES = 3         # 최대 연속 손실 횟수
_DEFAULT_MAX_DAILY_LOSS_PT = 5.0           # 일일 최대 손실 (pt)
_DEFAULT_SLOT_PERFORMANCE_ENABLED = False  # 슬롯별 성과 기반 할당 활성화

# ── Trailing Stop-loss ───────────────────────────────────────────────────
_DEFAULT_TRAILING_STOP_ENABLED = False      # Trailing Stop 활성화
_DEFAULT_TRAILING_STOP_ACTIVATION_PT = 1.0  # Trailing 시작 이익 (pt)
_DEFAULT_TRAILING_STOP_DISTANCE_PT = 0.5    # Trailing 거리 (pt)

# ── Phase 3 기본값 ────────────────────────────────────────────────────────────
# ATM IV 기반 동적 목표/손절 — IV × 선물가 × multiplier
_DEFAULT_IV_DYNAMIC_ENABLED = True      # 동적 조정 활성화 여부
_DEFAULT_IV_TARGET_MULT = 0.5           # target = ATM_IV × price × mult (예: 0.20×380×0.5=38pt → 클램핑)
_DEFAULT_IV_STOP_MULT = 0.25            # stop  = ATM_IV × price × mult
_DEFAULT_IV_TARGET_MIN = 1.5            # 동적 목표 하한 (pt)
_DEFAULT_IV_TARGET_MAX = 5.0            # 동적 목표 상한 (pt)
_DEFAULT_IV_STOP_MIN = 0.75             # 동적 손절 하한 (pt)
_DEFAULT_IV_STOP_MAX = 2.5              # 동적 손절 상한 (pt)
# Dealer Gamma 게이트 — net_gamma_proxy 방향과 진입 방향 일치 필요
_DEFAULT_GAMMA_GATE_ENABLED = False     # 기본 비활성 (보수적 시작)
# 신뢰도 기반 동적 목표/손절 — confidence별 배수
_DEFAULT_CONFIDENCE_DYNAMIC_ENABLED = True  # 신뢰도 기반 동적 조정 활성화
_DEFAULT_CONFIDENCE_HIGH_TARGET_MULT = 1.5   # HIGH confidence: 목표 × 1.5
_DEFAULT_CONFIDENCE_HIGH_STOP_MULT = 0.8     # HIGH confidence: 손절 × 0.8 (공격적)
_DEFAULT_CONFIDENCE_MEDIUM_TARGET_MULT = 1.0 # MEDIUM confidence: 목표 × 1.0 (기본)
_DEFAULT_CONFIDENCE_MEDIUM_STOP_MULT = 1.0   # MEDIUM confidence: 손절 × 1.0 (기본)
_DEFAULT_CONFIDENCE_LOW_TARGET_MULT = 0.7    # LOW confidence: 목표 × 0.7 (보수적)
_DEFAULT_CONFIDENCE_LOW_STOP_MULT = 1.3      # LOW confidence: 손절 × 1.3 (보수적)
# 이력 JSON 저장
_DEFAULT_HISTORY_SAVE_ENABLED = True
_DEFAULT_HISTORY_DIR = "trade_history"

# ── 포지션 사이징 ───────────────────────────────────────────────────────────
_DEFAULT_SIZING_METHOD = "fixed_fractional"  # 사이징 방법
_DEFAULT_SIZING_FIXED_FRACTION = 0.95        # 고정 비율
_DEFAULT_SIZING_KELLY_FRACTION = 0.5         # 켈리 비율
_DEFAULT_SIZING_MIN_KELLY = 0.1             # 최소 켈리
_DEFAULT_SIZING_MAX_KELLY = 0.25            # 최대 켈리
_DEFAULT_SIZING_RISK_PER_TRADE = 0.02       # 거래당 리스크
_DEFAULT_SIZING_STOP_LOSS_PT = 1.0          # 손절 포인트
_DEFAULT_SIZING_ATR_MULTIPLIER = 2.0         # ATR 멀티플라이어
_DEFAULT_SIZING_VOLATILITY_TARGET = 0.15    # 목표 변동성
_DEFAULT_SIZING_MAX_POSITION = 0.3          # 최대 포지션
_DEFAULT_SIZING_MIN_POSITION = 0.05         # 최소 포지션


# ── 설정 dataclass (config.py 통합 전 독립 사용 가능) ──────────────────────

class TradeGateConfig:
    """TradeExecutionGate 설정.

    config.py 의 AppConfig 에 trade_gate 필드로 통합되기 전에도
    독립적으로 사용할 수 있도록 일반 클래스로 구현.
    """

    __slots__ = (
        "enabled",
        "max_daily_trades",
        "min_consecutive_signals",
        "min_confidence",
        "min_prob_buy",
        "max_prob_sell",
        "require_consensus",
        "target_profit_pt",
        "stop_loss_pt",
        "force_close_time",
        "slot_a_end",
        "slot_b_end",
        "market_open_time",
        "reverse_close_count",
        # 리스크 관리
        "max_consecutive_losses",
        "max_daily_loss_pt",
        "slot_performance_enabled",
        "initial_capital",  # BUG-5 수정: 초기 자본 설정
        # Trailing Stop-loss
        "trailing_stop_enabled",
        "trailing_stop_activation_pt",
        "trailing_stop_distance_pt",
        # Phase 3
        "iv_dynamic_enabled",
        "iv_target_mult",
        "iv_stop_mult",
        "iv_target_min",
        "iv_target_max",
        "iv_stop_min",
        "iv_stop_max",
        "gamma_gate_enabled",
        # 신뢰도 기반 동적 목표/손절
        "confidence_dynamic_enabled",
        "confidence_high_target_mult",
        "confidence_high_stop_mult",
        "confidence_medium_target_mult",
        "confidence_medium_stop_mult",
        "confidence_low_target_mult",
        "confidence_low_stop_mult",
        # 이력 저장
        "history_save_enabled",
        "history_dir",
        # 포지션 사이징
        "sizing_method",
        "sizing_fixed_fraction",
        "sizing_kelly_fraction",
        "sizing_min_kelly",
        "sizing_max_kelly",
        "sizing_risk_per_trade",
        "sizing_stop_loss_pt",
        "sizing_atr_multiplier",
        "sizing_volatility_target",
        "sizing_max_position",
        "sizing_min_position",
    )

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_daily_trades: int = _DEFAULT_MAX_DAILY,
        min_consecutive_signals: int = _DEFAULT_MIN_CONSECUTIVE,
        min_confidence: str = _DEFAULT_MIN_CONFIDENCE,
        min_prob_buy: float = _DEFAULT_MIN_PROB_BUY,
        max_prob_sell: float = _DEFAULT_MAX_PROB_SELL,
        require_consensus: bool = _DEFAULT_REQUIRE_CONSENSUS,
        target_profit_pt: float = _DEFAULT_TARGET_PT,
        stop_loss_pt: float = _DEFAULT_STOP_PT,
        force_close_time: str = _DEFAULT_FORCE_CLOSE,
        slot_a_end: str = _DEFAULT_SLOT_A_END,
        slot_b_end: str = _DEFAULT_SLOT_B_END,
        market_open_time: str = _DEFAULT_MARKET_OPEN,
        reverse_close_count: int = _DEFAULT_REVERSE_CLOSE_COUNT,
        # 리스크 관리
        max_consecutive_losses: int = _DEFAULT_MAX_CONSECUTIVE_LOSSES,
        max_daily_loss_pt: float = _DEFAULT_MAX_DAILY_LOSS_PT,
        slot_performance_enabled: bool = _DEFAULT_SLOT_PERFORMANCE_ENABLED,
        initial_capital: float = 1000000.0,  # BUG-5 수정: 기본 100만원
        # Trailing Stop-loss
        trailing_stop_enabled: bool = _DEFAULT_TRAILING_STOP_ENABLED,
        trailing_stop_activation_pt: float = _DEFAULT_TRAILING_STOP_ACTIVATION_PT,
        trailing_stop_distance_pt: float = _DEFAULT_TRAILING_STOP_DISTANCE_PT,
        # Phase 3
        iv_dynamic_enabled: bool = _DEFAULT_IV_DYNAMIC_ENABLED,
        iv_target_mult: float = _DEFAULT_IV_TARGET_MULT,
        iv_stop_mult: float = _DEFAULT_IV_STOP_MULT,
        iv_target_min: float = _DEFAULT_IV_TARGET_MIN,
        iv_target_max: float = _DEFAULT_IV_TARGET_MAX,
        iv_stop_min: float = _DEFAULT_IV_STOP_MIN,
        iv_stop_max: float = _DEFAULT_IV_STOP_MAX,
        gamma_gate_enabled: bool = _DEFAULT_GAMMA_GATE_ENABLED,
        # 신뢰도 기반 동적 목표/손절
        confidence_dynamic_enabled: bool = _DEFAULT_CONFIDENCE_DYNAMIC_ENABLED,
        confidence_high_target_mult: float = _DEFAULT_CONFIDENCE_HIGH_TARGET_MULT,
        confidence_high_stop_mult: float = _DEFAULT_CONFIDENCE_HIGH_STOP_MULT,
        confidence_medium_target_mult: float = _DEFAULT_CONFIDENCE_MEDIUM_TARGET_MULT,
        confidence_medium_stop_mult: float = _DEFAULT_CONFIDENCE_MEDIUM_STOP_MULT,
        confidence_low_target_mult: float = _DEFAULT_CONFIDENCE_LOW_TARGET_MULT,
        confidence_low_stop_mult: float = _DEFAULT_CONFIDENCE_LOW_STOP_MULT,
        # 이력 저장
        history_save_enabled: bool = _DEFAULT_HISTORY_SAVE_ENABLED,
        history_dir: str = _DEFAULT_HISTORY_DIR,
        # 포지션 사이징
        sizing_method: str = _DEFAULT_SIZING_METHOD,
        sizing_fixed_fraction: float = _DEFAULT_SIZING_FIXED_FRACTION,
        sizing_kelly_fraction: float = _DEFAULT_SIZING_KELLY_FRACTION,
        sizing_min_kelly: float = _DEFAULT_SIZING_MIN_KELLY,
        sizing_max_kelly: float = _DEFAULT_SIZING_MAX_KELLY,
        sizing_risk_per_trade: float = _DEFAULT_SIZING_RISK_PER_TRADE,
        sizing_stop_loss_pt: float = _DEFAULT_SIZING_STOP_LOSS_PT,
        sizing_atr_multiplier: float = _DEFAULT_SIZING_ATR_MULTIPLIER,
        sizing_volatility_target: float = _DEFAULT_SIZING_VOLATILITY_TARGET,
        sizing_max_position: float = _DEFAULT_SIZING_MAX_POSITION,
        sizing_min_position: float = _DEFAULT_SIZING_MIN_POSITION,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_daily_trades = max(1, int(max_daily_trades))
        self.min_consecutive_signals = max(1, int(min_consecutive_signals))
        self.min_confidence = str(min_confidence).upper()
        self.min_prob_buy = float(min_prob_buy)
        self.max_prob_sell = float(max_prob_sell)
        self.require_consensus = bool(require_consensus)
        self.target_profit_pt = float(target_profit_pt)
        self.stop_loss_pt = float(stop_loss_pt)
        self.force_close_time = str(force_close_time)
        self.slot_a_end = str(slot_a_end)
        self.slot_b_end = str(slot_b_end)
        self.market_open_time = str(market_open_time)
        self.reverse_close_count = max(1, int(reverse_close_count))
        # 리스크 관리
        self.max_consecutive_losses = max(0, int(max_consecutive_losses))
        self.max_daily_loss_pt = float(max_daily_loss_pt)
        self.slot_performance_enabled = bool(slot_performance_enabled)
        self.initial_capital = float(initial_capital)  # BUG-5 수정
        # Trailing Stop-loss
        self.trailing_stop_enabled = bool(trailing_stop_enabled)
        self.trailing_stop_activation_pt = float(trailing_stop_activation_pt)
        self.trailing_stop_distance_pt = float(trailing_stop_distance_pt)
        # Phase 3
        self.iv_dynamic_enabled = bool(iv_dynamic_enabled)
        self.iv_target_mult = float(iv_target_mult)
        self.iv_stop_mult = float(iv_stop_mult)
        self.iv_target_min = float(iv_target_min)
        self.iv_target_max = float(iv_target_max)
        self.iv_stop_min = float(iv_stop_min)
        self.iv_stop_max = float(iv_stop_max)
        self.gamma_gate_enabled = bool(gamma_gate_enabled)
        # 신뢰도 기반 동적 목표/손절
        self.confidence_dynamic_enabled = bool(confidence_dynamic_enabled)
        self.confidence_high_target_mult = float(confidence_high_target_mult)
        self.confidence_high_stop_mult = float(confidence_high_stop_mult)
        self.confidence_medium_target_mult = float(confidence_medium_target_mult)
        self.confidence_medium_stop_mult = float(confidence_medium_stop_mult)
        self.confidence_low_target_mult = float(confidence_low_target_mult)
        self.confidence_low_stop_mult = float(confidence_low_stop_mult)
        # 이력 저장
        self.history_save_enabled = bool(history_save_enabled)
        self.history_dir = str(history_dir)
        # 포지션 사이징
        self.sizing_method = str(sizing_method)
        self.sizing_fixed_fraction = float(sizing_fixed_fraction)
        self.sizing_kelly_fraction = float(sizing_kelly_fraction)
        self.sizing_min_kelly = float(sizing_min_kelly)
        self.sizing_max_kelly = float(sizing_max_kelly)
        self.sizing_risk_per_trade = float(sizing_risk_per_trade)
        self.sizing_stop_loss_pt = float(sizing_stop_loss_pt)
        self.sizing_atr_multiplier = float(sizing_atr_multiplier)
        self.sizing_volatility_target = float(sizing_volatility_target)
        self.sizing_max_position = float(sizing_max_position)
        self.sizing_min_position = float(sizing_min_position)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradeGateConfig":
        """config.json 의 trade_gate 섹션 dict → TradeGateConfig."""
        if not isinstance(d, dict):
            return cls()
        def _f(key, default):
            v = d.get(key)
            return v if v is not None else default
        return cls(
            enabled=bool(_f("enabled", False)),
            max_daily_trades=int(_f("max_daily_trades", _DEFAULT_MAX_DAILY)),
            min_consecutive_signals=int(_f("min_consecutive_signals", _DEFAULT_MIN_CONSECUTIVE)),
            min_confidence=str(_f("min_confidence", _DEFAULT_MIN_CONFIDENCE)),
            min_prob_buy=float(_f("min_prob_buy", _DEFAULT_MIN_PROB_BUY)),
            max_prob_sell=float(_f("max_prob_sell", _DEFAULT_MAX_PROB_SELL)),
            require_consensus=bool(_f("require_consensus", _DEFAULT_REQUIRE_CONSENSUS)),
            target_profit_pt=float(_f("target_profit_pt", _DEFAULT_TARGET_PT)),
            stop_loss_pt=float(_f("stop_loss_pt", _DEFAULT_STOP_PT)),
            force_close_time=str(_f("force_close_time", _DEFAULT_FORCE_CLOSE)),
            slot_a_end=str(_f("slot_a_end", _DEFAULT_SLOT_A_END)),
            slot_b_end=str(_f("slot_b_end", _DEFAULT_SLOT_B_END)),
            market_open_time=str(_f("market_open_time", _DEFAULT_MARKET_OPEN)),
            reverse_close_count=int(_f("reverse_close_count", _DEFAULT_REVERSE_CLOSE_COUNT)),
            # 리스크 관리
            max_consecutive_losses=int(_f("max_consecutive_losses", _DEFAULT_MAX_CONSECUTIVE_LOSSES)),
            max_daily_loss_pt=float(_f("max_daily_loss_pt", _DEFAULT_MAX_DAILY_LOSS_PT)),
            slot_performance_enabled=bool(_f("slot_performance_enabled", _DEFAULT_SLOT_PERFORMANCE_ENABLED)),
            # Trailing Stop-loss
            trailing_stop_enabled=bool(_f("trailing_stop_enabled", _DEFAULT_TRAILING_STOP_ENABLED)),
            trailing_stop_activation_pt=float(_f("trailing_stop_activation_pt", _DEFAULT_TRAILING_STOP_ACTIVATION_PT)),
            trailing_stop_distance_pt=float(_f("trailing_stop_distance_pt", _DEFAULT_TRAILING_STOP_DISTANCE_PT)),
            # Phase 3
            iv_dynamic_enabled=bool(_f("iv_dynamic_enabled", _DEFAULT_IV_DYNAMIC_ENABLED)),
            iv_target_mult=float(_f("iv_target_mult", _DEFAULT_IV_TARGET_MULT)),
            iv_stop_mult=float(_f("iv_stop_mult", _DEFAULT_IV_STOP_MULT)),
            iv_target_min=float(_f("iv_target_min", _DEFAULT_IV_TARGET_MIN)),
            iv_target_max=float(_f("iv_target_max", _DEFAULT_IV_TARGET_MAX)),
            iv_stop_min=float(_f("iv_stop_min", _DEFAULT_IV_STOP_MIN)),
            iv_stop_max=float(_f("iv_stop_max", _DEFAULT_IV_STOP_MAX)),
            gamma_gate_enabled=bool(_f("gamma_gate_enabled", _DEFAULT_GAMMA_GATE_ENABLED)),
            # 신뢰도 기반 동적 목표/손절
            confidence_dynamic_enabled=bool(_f("confidence_dynamic_enabled", _DEFAULT_CONFIDENCE_DYNAMIC_ENABLED)),
            confidence_high_target_mult=float(_f("confidence_high_target_mult", _DEFAULT_CONFIDENCE_HIGH_TARGET_MULT)),
            confidence_high_stop_mult=float(_f("confidence_high_stop_mult", _DEFAULT_CONFIDENCE_HIGH_STOP_MULT)),
            confidence_medium_target_mult=float(_f("confidence_medium_target_mult", _DEFAULT_CONFIDENCE_MEDIUM_TARGET_MULT)),
            confidence_medium_stop_mult=float(_f("confidence_medium_stop_mult", _DEFAULT_CONFIDENCE_MEDIUM_STOP_MULT)),
            confidence_low_target_mult=float(_f("confidence_low_target_mult", _DEFAULT_CONFIDENCE_LOW_TARGET_MULT)),
            confidence_low_stop_mult=float(_f("confidence_low_stop_mult", _DEFAULT_CONFIDENCE_LOW_STOP_MULT)),
            # 이력 저장
            history_save_enabled=bool(_f("history_save_enabled", _DEFAULT_HISTORY_SAVE_ENABLED)),
            history_dir=str(_f("history_dir", _DEFAULT_HISTORY_DIR)),
            # 포지션 사이징
            sizing_method=str(_f("sizing_method", _DEFAULT_SIZING_METHOD)),
            sizing_fixed_fraction=float(_f("sizing_fixed_fraction", _DEFAULT_SIZING_FIXED_FRACTION)),
            sizing_kelly_fraction=float(_f("sizing_kelly_fraction", _DEFAULT_SIZING_KELLY_FRACTION)),
            sizing_min_kelly=float(_f("sizing_min_kelly", _DEFAULT_SIZING_MIN_KELLY)),
            sizing_max_kelly=float(_f("sizing_max_kelly", _DEFAULT_SIZING_MAX_KELLY)),
            sizing_risk_per_trade=float(_f("sizing_risk_per_trade", _DEFAULT_SIZING_RISK_PER_TRADE)),
            sizing_stop_loss_pt=float(_f("sizing_stop_loss_pt", _DEFAULT_SIZING_STOP_LOSS_PT)),
            sizing_atr_multiplier=float(_f("sizing_atr_multiplier", _DEFAULT_SIZING_ATR_MULTIPLIER)),
            sizing_volatility_target=float(_f("sizing_volatility_target", _DEFAULT_SIZING_VOLATILITY_TARGET)),
            sizing_max_position=float(_f("sizing_max_position", _DEFAULT_SIZING_MAX_POSITION)),
            sizing_min_position=float(_f("sizing_min_position", _DEFAULT_SIZING_MIN_POSITION)),
        )

    # confidence 레벨 → 정수 (비교용)
    _CONF_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    def confidence_ok(self, confidence: str) -> bool:
        rank = self._CONF_RANK.get(str(confidence).upper(), 0)
        min_rank = self._CONF_RANK.get(self.min_confidence, 1)
        return rank >= min_rank

    def get_confidence_multiplier(self, confidence: str) -> tuple[float, float]:
        """신뢰도별 목표/손절 배수를 반환한다.
        
        Args:
            confidence: "HIGH", "MEDIUM", "LOW"
        
        Returns:
            (target_multiplier, stop_multiplier)
        
        Example:
            HIGH → (1.5, 0.8)  # 공격적: 목표 1.5배, 손절 0.8배
            MEDIUM → (1.0, 1.0) # 기본
            LOW → (0.7, 1.3)    # 보수적: 목표 0.7배, 손절 1.3배
        """
        conf = str(confidence).upper()
        if conf == "HIGH":
            return (self.confidence_high_target_mult, self.confidence_high_stop_mult)
        elif conf == "MEDIUM":
            return (self.confidence_medium_target_mult, self.confidence_medium_stop_mult)
        else:  # LOW 또는 기타
            return (self.confidence_low_target_mult, self.confidence_low_stop_mult)


# ── 메인 게이트 클래스 ────────────────────────────────────────────────────────

class TradeExecutionGate:
    """5분 예측 신호 → 장중 진입/청산 판단기.

    Args:
        notifier:  TelegramNotifier 인스턴스 (send_text 호출용).
        config:    TradeGateConfig 인스턴스. None 이면 기본값(enabled=False) 사용.
    """

    def __init__(
        self,
        notifier: Any,
        config: Optional[TradeGateConfig] = None,
        event_bus: Optional[Any] = None,
    ) -> None:
        self._notifier = notifier
        self._cfg = config if isinstance(config, TradeGateConfig) else TradeGateConfig()
        self._state = TradeStateManager()
        
        # 이벤트 버스 (선택적)
        self._event_bus = event_bus if EVENTS_AVAILABLE else None

        # 일일 결산 전송 여부 (장 종료 후 1회만)
        self._daily_summary_sent: bool = False
        self._last_summary_date: str = ""
        self._summary_lock = threading.Lock()

        # Phase 3 — 이력 저장 디렉토리 초기화
        if self._cfg.history_save_enabled:
            try:
                import os as _os
                _os.makedirs(self._cfg.history_dir, exist_ok=True)
            except Exception as e:
                logger.warning("[GATE] history_dir 생성 실패: %s", e)

        # 포지션 사이저 초기화
        from .position_sizing import PositionSizer, SizingConfig, SizingMethod
        self._sizer = PositionSizer(SizingConfig(
            method=SizingMethod(self._cfg.sizing_method),
            fixed_fraction=self._cfg.sizing_fixed_fraction,
            kelly_fraction=self._cfg.sizing_kelly_fraction,
            min_kelly=self._cfg.sizing_min_kelly,
            max_kelly=self._cfg.sizing_max_kelly,
            risk_per_trade=self._cfg.sizing_risk_per_trade,
            stop_loss_pt=self._cfg.sizing_stop_loss_pt,
            atr_multiplier=self._cfg.sizing_atr_multiplier,
            volatility_target=self._cfg.sizing_volatility_target,
            max_position_size=self._cfg.sizing_max_position,
            min_position_size=self._cfg.sizing_min_position,
        ))

        logger.info(
            "[GATE] TradeExecutionGate 초기화 enabled=%s max_daily=%d "
            "min_consecutive=%d target=%.1fpt stop=%.1fpt "
            "iv_dynamic=%s gamma_gate=%s history=%s "
            "sizing=%s",
            self._cfg.enabled,
            self._cfg.max_daily_trades,
            self._cfg.min_consecutive_signals,
            self._cfg.target_profit_pt,
            self._cfg.stop_loss_pt,
            self._cfg.iv_dynamic_enabled,
            self._cfg.gamma_gate_enabled,
            self._cfg.history_save_enabled,
            self._cfg.sizing_method,
        )

    # ── 공개 API ─────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def on_signal(
        self,
        result: Dict[str, Any],
        *,
        current_price: float = 0.0,
    ) -> None:
        """5분 예측 결과를 받아 진입/청산을 판단한다.

        PipelineTelegramBridge 의 예측 루프에서 매 틱 호출한다.
        enabled=False 면 즉시 반환(no-op).

        Args:
            result:        pipeline.get_prediction() 반환 dict.
            current_price: 현재 선물 가격. 0이면 result 에서 추출 시도.
        """
        if not self._cfg.enabled:
            return
        try:
            self._on_signal_inner(result, current_price=current_price)
        except Exception:
            logger.exception("[GATE] on_signal 오류")

    def check_close(self, *, current_price: float = 0.0) -> None:
        """청산 조건(목표/손절/강제)을 점검한다.

        별도 감시 루프(예: 1분 주기)에서 주기적으로 호출한다.
        포지션이 없으면 no-op.
        """
        if not self._cfg.enabled:
            return
        try:
            self._check_close_inner(current_price=current_price)
        except Exception:
            logger.exception("[GATE] check_close 오류")

    def send_daily_summary(self, *, force: bool = False) -> bool:
        """일일 결산 메시지를 전송한다.

        장 종료 후 1회만 전송한다.
        force=True 면 이미 전송했어도 재전송.
        """
        if not self._cfg.enabled:
            return False
        try:
            return self._send_daily_summary_inner(force=force)
        except Exception:
            logger.exception("[GATE] send_daily_summary 오류")
            return False

    def get_daily_summary_dict(self) -> dict:
        """오늘의 거래 요약 dict 반환 (로깅/테스트용)."""
        state = self._state.get_state()
        return state.summary_dict()

    # ── 내부 로직: 신호 처리 ─────────────────────────────────────────────────

    def _on_signal_inner(
        self,
        result: Dict[str, Any],
        *,
        current_price: float,
    ) -> None:
        signal = str(result.get("signal", "HOLD")).upper()
        confidence = str(result.get("confidence", "LOW")).upper()
        prob = float(result.get("prob", 0.5) or 0.5)
        consensus = bool(result.get("consensus", False))

        price = float(current_price or result.get("current_price") or 0.0)
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        # Phase 3 — opt_context: result["options"] (opt_snap 전체)
        opt_context: Dict[str, Any] = {}
        try:
            opts = result.get("options")
            if isinstance(opts, dict):
                opt_context = opts
        except Exception as _e:
            logger.debug("[_on_signal_inner] 오류 무시: %s", _e)

        # ── 당일 시가(daily_open_price) 기록 — 장 시작 후 첫 유효 가격 ──
        # get_trade_slot()이 None을 반환하기 전(09:05 이전)에도 기록할 수 있도록
        # market_open 체크를 별도로 수행한다.
        if price > 0.0:
            _mo_str = str(self._cfg.market_open_time or _DEFAULT_MARKET_OPEN)
            try:
                _mo_h, _mo_m = int(_mo_str.split(":")[0]), int(_mo_str.split(":")[1])
                _after_open = (now.hour, now.minute) >= (_mo_h, _mo_m)
            except Exception:
                _after_open = True
            if _after_open:
                def _record_open(s: DailyState) -> None:
                    if s.daily_open_price <= 0.0:
                        s.daily_open_price = float(price)
                self._state.update(today_str, _record_open)

        # ── 반대신호 청산 체크 (포지션 보유 중일 때만) ──
        state = self._state.get_state()
        if state.has_position and state.active is not None:
            self._handle_reverse_signal(state, signal, price, now, today_str)
            state = self._state.get_state()

        # ── 신호 연속성 업데이트 ──
        self._update_consecutive(signal, today_str)
        state = self._state.get_state()

        # ── 진입 판단 ──
        if not state.has_position and signal in ("BUY", "SELL"):
            self._try_enter(
                signal=signal,
                confidence=confidence,
                prob=prob,
                consensus=consensus,
                price=price,
                now=now,
                today_str=today_str,
                state=state,
                opt_context=opt_context,
            )

        # ── 장 종료 후 결산 (15:10 이후 최초 1회) ──
        self._maybe_send_daily_summary(now, today_str)

    def _update_consecutive(self, signal: str, today_str: str) -> None:
        """연속 신호 카운터를 업데이트한다."""
        def _fn(state: DailyState) -> None:
            if signal in ("BUY", "SELL"):
                if state.consecutive_signal == signal:
                    state.consecutive_count += 1
                else:
                    state.consecutive_signal = signal
                    state.consecutive_count = 1
            else:
                # HOLD → 연속성 리셋
                state.consecutive_signal = ""
                state.consecutive_count = 0
        self._state.update(today_str, _fn)

    def _handle_reverse_signal(
        self,
        state: DailyState,
        signal: str,
        price: float,
        now: datetime,
        today_str: str,
    ) -> None:
        """보유 포지션과 반대 신호 시 카운터를 올리고 임계값 초과 시 청산."""
        if state.active is None:
            return
        pos = state.active
        is_reverse = (
            (pos.side == PositionSide.LONG and signal == "SELL")
            or (pos.side == PositionSide.SHORT and signal == "BUY")
        )
        if not is_reverse:
            # 같은 방향 신호 → 반대 카운터 리셋
            def _reset(s: DailyState) -> None:
                if s.active:
                    s.active.consecutive_reverse = 0
            self._state.update(today_str, _reset)
            return

        def _inc(s: DailyState) -> None:
            if s.active:
                s.active.consecutive_reverse += 1
        self._state.update(today_str, _inc)

        state = self._state.get_state()
        if state.active and state.active.consecutive_reverse >= self._cfg.reverse_close_count:
            self._execute_close(
                price=price,
                now=now,
                today_str=today_str,
                reason=CloseReason.REVERSE_SIGNAL,
            )

    def _try_enter(
        self,
        *,
        signal: str,
        confidence: str,
        prob: float,
        consensus: bool,
        price: float,
        now: datetime,
        today_str: str,
        state: DailyState,
        opt_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """진입 게이트를 모두 통과하면 포지션을 개설한다.

        Phase 3 추가 게이트:
          - 게이트 4: Dealer Gamma 방향 일치 (gamma_gate_enabled=True 시)
        Phase 3 동적 목표/손절:
          - ATM IV 기반으로 target_pt / stop_pt 동적 계산 (iv_dynamic_enabled=True 시)
        """
        cfg = self._cfg

        # ── 게이트 1: 신호 연속성 ──
        if state.consecutive_count < cfg.min_consecutive_signals:
            logger.debug(
                "[GATE] 진입 차단 — 연속성 부족 (필요 %d, 현재 %d %s)",
                cfg.min_consecutive_signals,
                state.consecutive_count,
                signal,
            )
            return

        # ── 게이트 2: 신호 품질 (confidence / prob / consensus) ──
        if not cfg.confidence_ok(confidence):
            logger.debug("[GATE] 진입 차단 — confidence 부족 (%s)", confidence)
            return
        if signal == "BUY" and prob < cfg.min_prob_buy:
            logger.debug("[GATE] 진입 차단 — prob 부족 (BUY %.3f < %.3f)", prob, cfg.min_prob_buy)
            return
        if signal == "SELL" and prob > cfg.max_prob_sell:
            logger.debug("[GATE] 진입 차단 — prob 부족 (SELL %.3f > %.3f)", prob, cfg.max_prob_sell)
            return
        if cfg.require_consensus and not consensus:
            logger.debug("[GATE] 진입 차단 — consensus 미달")
            return

        # ── 게이트 3: 시간대 슬롯 / 일일 횟수 ──
        slot = get_trade_slot(
            now,
            slot_a_end=cfg.slot_a_end,
            slot_b_end=cfg.slot_b_end,
            force_close=cfg.force_close_time,
            market_open=cfg.market_open_time,
        )
        if slot is None:
            logger.debug("[GATE] 진입 차단 — 진입 금지 시간대")
            return

        # ── 슬롯별 성과 기반 할당 (활성화 시) ──
        if cfg.slot_performance_enabled and state.slot_performance:
            # 사용 가능한 슬롯 목록 (현재 시간대에 해당하고 아직 사용하지 않은 슬롯)
            available_slots = []
            current_time = now.time()
            from datetime import time as _time_cls
            try:
                a_end = _time_cls(*map(int, cfg.slot_a_end.split(":")))
                b_end = _time_cls(*map(int, cfg.slot_b_end.split(":")))
            except Exception:
                a_end = _time_cls(10, 30)
                b_end = _time_cls(13, 0)

            # 현재 시간대에 해당하는 슬롯 확인
            if current_time < a_end:
                slot_candidates = [TradeSlot.A]
            elif current_time < b_end:
                slot_candidates = [TradeSlot.B]
            else:
                slot_candidates = [TradeSlot.C]

            # 사용하지 않은 슬롯 필터링
            for s in slot_candidates:
                if not state.slot_used(s):
                    available_slots.append(s)

            # 성과 기반 정렬 (승률 높은 순, 승률 같으면 총 손익 높은 순)
            if len(available_slots) > 1:
                def slot_score(s: TradeSlot) -> tuple:
                    perf = state.slot_performance.get(s.value, {"total": 0, "wins": 0, "pnl": 0.0})
                    total = perf.get("total", 0)
                    wins = perf.get("wins", 0)
                    pnl = perf.get("pnl", 0.0)
                    win_rate = (wins / total) if total > 0 else 0.5  # 데이터 없으면 중립
                    return (win_rate, pnl)  # 승률, 총 손익

                available_slots.sort(key=slot_score, reverse=True)
                slot = available_slots[0]
                logger.debug("[GATE] 슬롯별 성과 기반 할당: %s 선택", slot.value)
        if state.daily_count >= cfg.max_daily_trades:
            logger.debug("[GATE] 진입 차단 — 일일 한도 초과 (%d회)", state.daily_count)
            return
        if state.slot_used(slot):
            logger.debug("[GATE] 진입 차단 — 슬롯 %s 이미 사용", slot.value)
            return

        # ── 리스크 관리: 연속 손실 제한 ──
        if cfg.max_consecutive_losses > 0 and state.consecutive_losses >= cfg.max_consecutive_losses:
            logger.warning("[GATE] 진입 차단 — 연속 손실 한도 초과 (%d회)", state.consecutive_losses)
            if self._event_bus:
                self._event_bus.publish(RiskLimitEvent(
                    timestamp=datetime.now(),
                    limit_type="CONSECUTIVE_LOSS",
                    current_value=float(state.consecutive_losses),
                    limit_value=float(cfg.max_consecutive_losses),
                    action="BLOCK_ENTRY"
                ))
            return

        # ── 리스크 관리: 일일 최대 손실 제한 ──
        if cfg.max_daily_loss_pt > 0 and state.total_pnl_pt <= -cfg.max_daily_loss_pt:
            logger.warning("[GATE] 진입 차단 — 일일 손실 한도 초과 (%.2fpt / %.2fpt)", 
                          state.total_pnl_pt, cfg.max_daily_loss_pt)
            if self._event_bus:
                self._event_bus.publish(RiskLimitEvent(
                    timestamp=datetime.now(),
                    limit_type="DAILY_LOSS",
                    current_value=abs(state.total_pnl_pt),
                    limit_value=cfg.max_daily_loss_pt,
                    action="BLOCK_ENTRY"
                ))
            return

        # ── 게이트 4 (Phase 3): Dealer Gamma 방향 일치 ──
        opt = opt_context or {}
        net_gamma = float(opt.get("net_gamma_proxy") or 0.0)
        above_vt  = float(opt.get("above_vol_trigger") if opt.get("above_vol_trigger") is not None else 1.0)

        if cfg.gamma_gate_enabled:
            # 딜러 Long Gamma(net_gamma>0) + Vol Trigger 위(above_vt=1.0): 안정 구간
            #   → BUY는 허용, SELL은 딜러가 하락을 억제하므로 차단
            # 딜러 Short Gamma(net_gamma<0) + Vol Trigger 아래(above_vt=0.0): 불안정 구간
            #   → SELL은 허용, BUY는 딜러가 상승을 억제하므로 차단
            # net_gamma=0 이면 데이터 없음 → 게이트 통과 (안전 방향)
            if net_gamma != 0.0:
                dealer_long = net_gamma > 0.0
                if signal == "BUY" and not dealer_long and above_vt < 0.5:
                    logger.debug(
                        "[GATE] 진입 차단 — Gamma 게이트 (BUY, dealer_short_gamma, above_vt=%.1f)",
                        above_vt,
                    )
                    return
                if signal == "SELL" and dealer_long and above_vt >= 0.5:
                    logger.debug(
                        "[GATE] 진입 차단 — Gamma 게이트 (SELL, dealer_long_gamma, above_vt=%.1f)",
                        above_vt,
                    )
                    return

        # ── 현재가 확인 ──
        if price <= 0.0:
            logger.warning("[GATE] 진입 차단 — 현재가 미확인 (%.2f)", price)
            return

        # ── Phase 3: ATM IV 기반 동적 목표/손절 계산 (기준가: 당일 시가) ──
        atm_iv    = float(opt.get("atm_iv") or 0.0)
        atm_delta = float(opt.get("atm_call_delta") or opt.get("atm_delta") or 0.0)
        # daily_open_price가 아직 기록 안 됐으면 현재가를 fallback으로 사용
        _daily_open = float(state.daily_open_price) if state.daily_open_price > 0.0 else price
        target_pt, stop_pt = self._calc_dynamic_targets(atm_iv=atm_iv, daily_open=_daily_open, confidence=confidence)

        side = PositionSide.LONG if signal == "BUY" else PositionSide.SHORT
        
        # ── 포지션 사이징 계산 ──
        # 현재 자본 (일일 손익 반영)
        current_capital = self._cfg.initial_capital  # BUG-5 수정: 설정에서 초기 자본 사용
        if state.total_pnl_pct != 0:  # BUG-5 수정: 포인트 대신 퍼센트 사용
            current_capital *= (1 + state.total_pnl_pct / 100)
        
        # 손절 가격 계산
        stop_loss_price = None
        if side == PositionSide.LONG:
            stop_loss_price = price - stop_pt
        else:
            stop_loss_price = price + stop_pt
        
        # 승률/평균 승패 (과거 데이터에서 추정)
        win_rate = 0.5
        avg_win = target_pt
        avg_loss = stop_pt
        
        # ATR (옵션 데이터에서 추정)
        atr = None
        if atm_iv > 0:
            atr = price * atm_iv
        
        # 포지션 사이즈 계산
        sizing_result = self._sizer.calculate(
            capital=current_capital,
            entry_price=price,
            stop_loss_price=stop_loss_price,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            atr=atr
        )
        
        record = TradeRecord(
            slot=slot,
            side=side,
            entry_price=price,
            entry_time=now,
            entry_signal=signal,
            entry_confidence=confidence,
            entry_prob=prob,
            # Phase 3 컨텍스트
            entry_atm_iv=atm_iv,
            entry_atm_delta=abs(atm_delta),
            entry_net_gamma=net_gamma,
            entry_above_vol_trigger=above_vt,
            entry_target_pt=target_pt,
            entry_stop_pt=stop_pt,
            # 포지션 사이징
            position_size=sizing_result.size,
            capital_used=sizing_result.capital_used,
            risk_amount=sizing_result.risk_amount,
            risk_pct=sizing_result.risk_pct,
            sizing_method=sizing_result.method_used,
        )
        active = ActivePosition(record=record, trailing_stop_price=0.0)

        def _open(s: DailyState) -> None:
            s.active = active
            s.used_slots.append(slot)
        self._state.update(today_str, _open)

        logger.info(
            "[GATE] 진입 %s %.2f 슬롯=%s confidence=%s prob=%.3f "
            "target=%.2fpt stop=%.2fpt iv=%.1f%% (오늘 %d/%d회 연속손실=%d)",
            signal, price, slot.value, confidence, prob,
            target_pt, stop_pt, atm_iv * 100,
            state.daily_count + 1, cfg.max_daily_trades, state.consecutive_losses,
        )
        self._send_entry_message(record, daily_count=state.daily_count + 1)
        
        # 이벤트 발행: 진입
        if self._event_bus:
            self._event_bus.publish(TradeEntryEvent(
                timestamp=now,
                side=side.value,
                price=price,
                size=sizing_result.size,
                confidence=confidence,
                prob=prob,
                slot=slot.value,
                signal=signal
            ))

    # ── 내부 로직: 청산 ───────────────────────────────────────────────────────

    def _check_close_inner(self, *, current_price: float) -> None:
        """목표수익/손절/강제청산 조건을 점검한다."""
        state = self._state.get_state()
        if not state.has_position or state.active is None:
            return

        pos = state.active
        price = float(current_price or 0.0)
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        # ── 강제청산 시각 도달 여부 (force_close_time 이후인지만 판단) ──
        # get_trade_slot() 은 market_open 이전도 None 을 반환하므로
        # 강제청산 전용으로 시각을 직접 비교한다.
        if price > 0.0 and self._is_after_force_close(now):
            logger.info("[GATE] 강제청산 시각 도달 %.2f", price)
            self._execute_close(price=price, now=now, today_str=today_str, reason=CloseReason.FORCE_CLOSE)
            return

        if price <= 0.0:
            return

        # ── 목표수익 / 손절 — Phase 3: entry_target_pt/stop_pt 우선 사용 ──
        # 진입 시 동적으로 계산된 값이 기록돼 있으면 그것을 사용하고,
        # 없으면(0.0) config 기본값으로 fallback한다.
        rec = pos.record

        pnl = (
            price - pos.entry_price if pos.side == PositionSide.LONG
            else pos.entry_price - price
        )

        # ── Trailing Stop-loss ──
        if self._cfg.trailing_stop_enabled:
            activation_pt = self._cfg.trailing_stop_activation_pt
            distance_pt = self._cfg.trailing_stop_distance_pt

            # Trailing Stop 활성화 조건: 이익이 activation_pt 이상
            if pnl >= activation_pt:
                # Trailing Stop 가격 계산
                if pos.side == PositionSide.LONG:
                    new_trailing_stop = price - distance_pt
                    # 기존 trailing stop보다 높으면 업데이트 (이익 보호)
                    if new_trailing_stop > pos.trailing_stop_price:
                        pos.trailing_stop_price = new_trailing_stop
                        logger.debug(
                            "[GATE] Trailing Stop 업데이트 LONG: %.2f (pnl=%.2fpt)",
                            pos.trailing_stop_price, pnl,
                        )
                else:  # SHORT
                    new_trailing_stop = price + distance_pt
                    # 기존 trailing stop보다 낮으면 업데이트 (이익 보호)
                    if new_trailing_stop < pos.trailing_stop_price:  # BUG-4 수정: == 0.0 조건 삭제
                        pos.trailing_stop_price = new_trailing_stop
                        logger.debug(
                            "[GATE] Trailing Stop 업데이트 SHORT: %.2f (pnl=%.2fpt)",
                            pos.trailing_stop_price, pnl,
                        )

            # Trailing Stop 체크 (활성화된 경우)
            if pos.trailing_stop_price > 0.0:
                if pos.side == PositionSide.LONG:
                    if price <= pos.trailing_stop_price:
                        logger.info(
                            "[GATE] Trailing Stop 청산 LONG: %.2f <= %.2f (pnl=%.2fpt)",
                            price, pos.trailing_stop_price, pnl,
                        )
                        self._execute_close(price=price, now=now, today_str=today_str, reason=CloseReason.TRAILING_STOP)
                        return
                else:  # SHORT
                    if price >= pos.trailing_stop_price:
                        logger.info(
                            "[GATE] Trailing Stop 청산 SHORT: %.2f >= %.2f (pnl=%.2fpt)",
                            price, pos.trailing_stop_price, pnl,
                        )
                        self._execute_close(price=price, now=now, today_str=today_str, reason=CloseReason.TRAILING_STOP)
                        return

        # ── 목표수익 / 손절 — Phase 3: entry_target_pt/stop_pt 우선 사용 ──
        # 진입 시 동적으로 계산된 값이 기록돼 있으면 그것을 사용하고,
        # 없으면(0.0) config 기본값으로 fallback한다.
        rec = pos.record
        eff_target = rec.entry_target_pt if rec.entry_target_pt > 0.0 else self._cfg.target_profit_pt
        eff_stop   = rec.entry_stop_pt   if rec.entry_stop_pt   > 0.0 else self._cfg.stop_loss_pt

        if pnl >= eff_target:
            logger.info(
                "[GATE] 목표수익 도달 pnl=%.2fpt target=%.2fpt entry=%.2f close=%.2f",
                pnl, eff_target, pos.entry_price, price,
            )
            self._execute_close(price=price, now=now, today_str=today_str, reason=CloseReason.TARGET_PROFIT)
        elif pnl <= -eff_stop:
            logger.info(
                "[GATE] 손절 발동 pnl=%.2fpt stop=%.2fpt entry=%.2f close=%.2f",
                pnl, eff_stop, pos.entry_price, price,
            )
            self._execute_close(price=price, now=now, today_str=today_str, reason=CloseReason.STOP_LOSS)

    def _is_after_force_close(self, now: datetime) -> bool:
        """현재 시각이 force_close_time 이후인지 반환한다."""
        from datetime import time as _time_cls
        try:
            h, m = self._cfg.force_close_time.split(":")
            fc = _time_cls(int(h), int(m))
            return now.time() >= fc
        except Exception:
            return False

    def _calc_dynamic_targets(
        self,
        *,
        atm_iv: float,
        daily_open: float,
        confidence: str = "MEDIUM",
    ) -> tuple[float, float]:
        """ATM IV 및 신뢰도 기반으로 목표수익(pt)과 손절(pt)을 동적 계산한다.

        기준가로 current_price가 아닌 daily_open(당일 시가)을 사용한다.
        시가를 기준으로 삼으면 장중 가격 변동과 무관하게 일관된 목표값을 유지한다.

        공식:
            # 1단계: ATM IV 기반 기본값 계산
            base_target = clamp(ATM_IV × daily_open × iv_target_mult,
                                iv_target_min, iv_target_max)
            base_stop   = clamp(ATM_IV × daily_open × iv_stop_mult,
                                iv_stop_min,   iv_stop_max)
            
            # 2단계: 신뢰도 기반 배수 적용 (활성화 시)
            if confidence_dynamic_enabled:
                target_mult, stop_mult = get_confidence_multiplier(confidence)
                target = base_target × target_mult
                stop   = base_stop × stop_mult
            else:
                target = base_target
                stop   = base_stop

        ATM_IV 가 0 이거나 iv_dynamic_enabled=False 이거나 daily_open=0 이면
        config 기본값(target_profit_pt / stop_loss_pt)을 그대로 반환한다.

        예시 (ATM_IV=0.20, daily_open=820, confidence=HIGH):
            base_target = clamp(0.20 × 820 × 0.5, 1.5, 5.0) = 5.0pt
            base_stop   = clamp(0.20 × 820 × 0.25, 0.75, 2.5) = 2.5pt
            target = 5.0 × 1.5 = 7.5pt  # HIGH confidence: 공격적 목표
            stop   = 2.5 × 0.8 = 2.0pt  # HIGH confidence: 타이트 손절
        """
        cfg = self._cfg
        if not cfg.iv_dynamic_enabled or atm_iv <= 0.0 or daily_open <= 0.0:
            return cfg.target_profit_pt, cfg.stop_loss_pt

        try:
            # 1단계: ATM IV 기반 기본값 계산
            raw_target = atm_iv * daily_open * cfg.iv_target_mult
            raw_stop   = atm_iv * daily_open * cfg.iv_stop_mult
            base_target = float(max(cfg.iv_target_min, min(cfg.iv_target_max, raw_target)))
            base_stop   = float(max(cfg.iv_stop_min,   min(cfg.iv_stop_max,   raw_stop)))
            
            # 2단계: 신뢰도 기반 배수 적용
            if cfg.confidence_dynamic_enabled:
                target_mult, stop_mult = cfg.get_confidence_multiplier(confidence)
                target = base_target * target_mult
                stop   = base_stop * stop_mult
                logger.debug(
                    "[GATE] 신뢰도 기반 동적 목표/손절 confidence=%s mult=(%.2f,%.2f) "
                    "iv=%.1f%% daily_open=%.2f → target=%.2fpt stop=%.2fpt",
                    confidence, target_mult, stop_mult,
                    atm_iv * 100, daily_open, target, stop,
                )
            else:
                target = base_target
                stop   = base_stop
                logger.debug(
                    "[GATE] 동적 목표/손절 iv=%.1f%% daily_open=%.2f → target=%.2fpt stop=%.2fpt",
                    atm_iv * 100, daily_open, target, stop,
                )
            
            return target, stop
        except Exception:
            return cfg.target_profit_pt, cfg.stop_loss_pt

    def _save_history(self, record: "TradeRecord") -> None:
        """완결된 거래 기록을 일별 JSONL 파일에 저장한다.

        파일 형식: {history_dir}/YYYY-MM-DD.jsonl
        한 줄 = 거래 1건의 JSON (append 모드).
        """
        if not self._cfg.history_save_enabled:
            return
        try:
            import json as _json
            import os as _os
            from pathlib import Path
            
            date_str = (
                record.entry_time.strftime("%Y-%m-%d")
                if record.entry_time else datetime.now().strftime("%Y-%m-%d")
            )
            
            # 1. 기본 history_dir에 저장
            path = _os.path.join(self._cfg.history_dir, f"{date_str}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            logger.debug("[GATE] 이력 저장: %s", path)
            
            # 2. logs/trades 디렉토리에도 저장 (GUI 뷰어용)
            logs_dir = Path("logs/trades")
            logs_dir.mkdir(parents=True, exist_ok=True)
            logs_path = logs_dir / f"trades_{date_str}.jsonl"
            with open(logs_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            logger.debug("[GATE] 이력 저장 (GUI용): %s", logs_path)
        except Exception as e:
            logger.warning("[GATE] 이력 저장 실패: %s", e)

    def _execute_close(
        self,
        *,
        price: float,
        now: datetime,
        today_str: str,
        reason: CloseReason,
    ) -> None:
        """포지션을 청산하고 trade_log 에 기록한다."""
        closed_record: Optional[TradeRecord] = None

        def _close(state: DailyState) -> None:
            nonlocal closed_record
            if state.active is None:
                return
            rec = state.active.record
            rec.close_price = price
            rec.close_time = now
            rec.close_reason = reason
            state.trade_log.append(rec)
            
            # 신뢰도별 통계 업데이트
            conf = rec.entry_confidence.upper()
            if conf not in state.confidence_stats:
                state.confidence_stats[conf] = {"total": 0, "wins": 0}
            state.confidence_stats[conf]["total"] += 1
            if rec.pnl_pt > 0:
                state.confidence_stats[conf]["wins"] += 1
                state.consecutive_losses = 0  # 승리 시 연속 손실 초기화
            else:
                state.consecutive_losses += 1  # 손실 시 연속 손실 증가

            # 슬롯별 성과 업데이트
            slot_key = rec.slot.value
            if slot_key not in state.slot_performance:
                state.slot_performance[slot_key] = {"total": 0, "wins": 0, "pnl": 0.0}
            state.slot_performance[slot_key]["total"] += 1
            if rec.pnl_pt > 0:
                state.slot_performance[slot_key]["wins"] += 1
            state.slot_performance[slot_key]["pnl"] += rec.pnl_pt
            
            # BUG-6 수정: total_pnl_pct 갱신
            if rec.entry_price > 0:
                profit_pct = rec.pnl_pt / rec.entry_price * 100
                state.total_pnl_pct += profit_pct
            
            state.active = None
            closed_record = rec

        self._state.update(today_str, _close)

        if closed_record is not None:
            logger.info(
                "[GATE] 청산 %s pnl=%.2fpt 사유=%s 진입신호=%s confidence=%s",
                closed_record.side.value, closed_record.pnl_pt, reason.value,
                closed_record.entry_signal, closed_record.entry_confidence,
            )
            self._send_close_message(closed_record)
            self._save_history(closed_record)  # Phase 3 — 일별 JSONL 저장
            
            # 이벤트 발행: 청산
            if self._event_bus:
                self._event_bus.publish(TradeExitEvent(
                    timestamp=now,
                    side=closed_record.side.value,
                    entry_price=closed_record.entry_price,
                    exit_price=price,
                    size=closed_record.position_size,
                    pnl=closed_record.pnl_pt,
                    pnl_pct=closed_record.pnl_pt / closed_record.entry_price * 100 if closed_record.entry_price > 0 else 0,
                    reason=reason.value,
                    hold_minutes=closed_record.hold_minutes,
                    slot=closed_record.slot.value
                ))

    # ── 일일 결산 ─────────────────────────────────────────────────────────────

    def _maybe_send_daily_summary(self, now: datetime, today_str: str) -> None:
        """15:05 이후 최초 1회 일일 결산을 전송한다."""
        from datetime import time as _time_cls
        if now.time() < _time_cls(15, 5):
            return
        with self._summary_lock:
            if self._last_summary_date == today_str:
                return
        self._send_daily_summary_inner()

    def _send_daily_summary_inner(self, *, force: bool = False) -> bool:
        today_str = datetime.now().strftime("%Y-%m-%d")
        with self._summary_lock:
            if not force and self._last_summary_date == today_str:
                return False
            self._last_summary_date = today_str

        state = self._state.get_state()
        return self._send_summary_message(state)

    # ── 텔레그램 메시지 포맷 ──────────────────────────────────────────────────

    def _send_entry_message(self, record: TradeRecord, *, daily_count: int) -> bool:
        """진입 알림 메시지 전송."""
        try:
            cfg = self._cfg
            emoji = "🟢" if record.side == PositionSide.LONG else "🔴"
            side_str = "매수" if record.side == PositionSide.LONG else "매도"

            # Phase 3: 동적 목표/손절 우선 사용
            eff_target_pt = record.entry_target_pt if record.entry_target_pt > 0.0 else cfg.target_profit_pt
            eff_stop_pt   = record.entry_stop_pt   if record.entry_stop_pt   > 0.0 else cfg.stop_loss_pt

            target = (
                record.entry_price + eff_target_pt
                if record.side == PositionSide.LONG
                else record.entry_price - eff_target_pt
            )
            stop = (
                record.entry_price - eff_stop_pt
                if record.side == PositionSide.LONG
                else record.entry_price + eff_stop_pt
            )
            time_str = record.entry_time.strftime("%H:%M:%S")

            # IV / Gamma 부가 정보 (데이터 있을 때만 표시)
            iv_line = ""
            if record.entry_atm_iv > 0.0:
                gamma_dir = (
                    "Long Gamma" if record.entry_net_gamma > 0
                    else ("Short Gamma" if record.entry_net_gamma < 0 else "-")
                )
                vt_str = "↑" if record.entry_above_vol_trigger >= 0.5 else "↓"
                iv_line = (
                    f"\nATM IV:   {record.entry_atm_iv:.1%}  "
                    f"Gamma: {gamma_dir}  VT: {vt_str}"
                )

            msg = (
                f"{emoji} <b>진입 알림 ({side_str})</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"진입가:   <code>{record.entry_price:.2f}</code>\n"
                f"시각:     {time_str}  슬롯 {record.slot.value}\n"
                f"신호 근거: {record.entry_signal} {record.entry_confidence}"
                f"  (prob {record.entry_prob:.2f})\n"
                f"목표:     <code>{target:.2f}</code>  (+{eff_target_pt:.2f}pt)  "
                f"손절: <code>{stop:.2f}</code>  (-{eff_stop_pt:.2f}pt)"
                f"{iv_line}\n"
                f"오늘 진입: {daily_count} / {cfg.max_daily_trades}회"
            )
            return bool(self._notifier.send_text(msg, parse_mode="HTML"))
        except Exception:
            logger.exception("[GATE] 진입 알림 전송 실패")
            return False

    def _send_close_message(self, record: TradeRecord) -> bool:
        """청산 알림 메시지 전송."""
        try:
            pnl = record.pnl_pt
            pnl_emoji = "✅" if pnl > 0 else ("❌" if pnl < 0 else "➖")
            pnl_str = f"{pnl:+.2f}pt"
            hold_str = f"{record.hold_minutes:.0f}분"
            reason_str = record.close_reason.value if record.close_reason else "-"
            time_str = record.close_time.strftime("%H:%M:%S") if record.close_time else "-"
            side_str = "매수" if record.side == PositionSide.LONG else "매도"
            msg = (
                f"🚪 <b>청산 알림 ({side_str})</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"청산가:   <code>{record.close_price:.2f}</code>  ({time_str})\n"
                f"진입가:   <code>{record.entry_price:.2f}</code>\n"
                f"{pnl_emoji} 손익:   <b>{pnl_str}</b>\n"
                f"보유 시간: {hold_str}\n"
                f"진입 신호: {record.entry_signal} {record.entry_confidence} (prob {record.entry_prob:.2f})\n"
                f"청산 사유: {reason_str}"
            )
            return bool(self._notifier.send_text(msg, parse_mode="HTML"))
        except Exception:
            logger.exception("[GATE] 청산 알림 전송 실패")
            return False

    def _send_summary_message(self, state: DailyState) -> bool:
        """일일 결산 메시지 전송."""
        try:
            total = state.daily_count
            wins = state.wins
            losses = state.losses
            draws = total - wins - losses
            pnl = state.total_pnl_pt
            pnl_emoji = "📈" if pnl > 0 else ("📉" if pnl < 0 else "➖")
            
            # Confidence별 통계
            conf_stats = state.confidence_stats  # {"HIGH": wins, "MEDIUM": wins, ...}
            conf_lines = []
            if conf_stats:
                for conf in ["HIGH", "MEDIUM", "LOW"]:
                    if conf in conf_stats:
                        c_total = conf_stats[conf].get("total", 0)
                        c_wins = conf_stats[conf].get("wins", 0)
                        c_rate = (c_wins / c_total * 100) if c_total > 0 else 0
                        conf_lines.append(f"{conf}: {c_wins}/{c_total} ({c_rate:.0f}%)")
            
            lines = [
                f"📊 <b>일일 결산</b>  ({state.date})",
                "━━━━━━━━━━━━━━━━━━━",
                f"총 진입: {total}회 / {self._cfg.max_daily_trades}회",
                f"승: {wins}  패: {losses}" + (f"  무: {draws}" if draws else ""),
                f"{pnl_emoji} 손익 합계: <b>{pnl:+.2f}pt</b>",
            ]

            # 리스크 관리 정보
            if self._cfg.max_consecutive_losses > 0 or self._cfg.max_daily_loss_pt > 0:
                risk_lines = []
                if self._cfg.max_consecutive_losses > 0:
                    risk_lines.append(f"연속 손실: {state.consecutive_losses}/{self._cfg.max_consecutive_losses}회")
                if self._cfg.max_daily_loss_pt > 0:
                    loss_ratio = abs(pnl) / self._cfg.max_daily_loss_pt if self._cfg.max_daily_loss_pt > 0 else 0
                    risk_lines.append(f"손실 한도: {abs(pnl):.2f}/{self._cfg.max_daily_loss_pt:.2f}pt ({loss_ratio:.0%})")
                if risk_lines:
                    lines.append("")
                    lines.append("🛡️ 리스크 관리:")
                    lines.extend(risk_lines)
            
            if conf_lines:
                lines.append("")
                lines.append("📈 신뢰도별 승률:")
                lines.extend(conf_lines)
            
            msg = "\n".join(lines)
            return bool(self._notifier.send_text(msg, parse_mode="HTML"))
        except Exception:
            logger.exception("[GATE] 일일 결산 전송 실패")
            return False

    # ── Phase 3: 텔레그램 명령 핸들러 ────────────────────────────────────────

    def handle_telegram_command(self, text: str) -> bool:
        """/trade_status 및 /trade_gate on|off 명령을 처리한다.

        PipelineTelegramBridge 의 메시지 수신 루프에서 호출한다.

        지원 명령:
            ``/trade_status``           — 현재 포지션 및 오늘 결과 조회
            ``/trade_gate on``          — TradeExecutionGate 런타임 활성화
            ``/trade_gate off``         — TradeExecutionGate 런타임 비활성화

        Args:
            text: 텔레그램에서 수신한 명령 문자열 (공백/개행 strip 후 전달).

        Returns:
            True  — 명령을 처리하고 응답을 전송했을 때.
            False — 알 수 없는 명령이거나 처리 실패 시.
        """
        cmd = text.strip().lower()

        if cmd == "/trade_status":
            return self._cmd_trade_status()

        if cmd in ("/trade_gate on", "/trade_gate off"):
            enable = cmd.endswith("on")
            return self._cmd_trade_gate_toggle(enable)

        return False

    def _cmd_trade_status(self) -> bool:
        """``/trade_status`` — 현재 포지션 + 오늘 거래 결과를 전송한다."""
        try:
            state = self._state.get_state()
            lines = [
                "📋 <b>거래 현황</b>",
                "━━━━━━━━━━━━━━━━━━━",
                f"게이트: {'🟢 활성' if self._cfg.enabled else '⛔ 비활성'}",
                "",
            ]

            # ── 현재 포지션 ──
            if state.has_position and state.active is not None:
                pos = state.active
                side_str = "매수(Long)" if pos.side == PositionSide.LONG else "매도(Short)"
                entry_t  = pos.entry_time.strftime("%H:%M:%S")
                lines += [
                    f"📌 <b>보유 포지션: {side_str}</b>",
                    f"  진입가: <code>{pos.entry_price:.2f}</code>  ({entry_t})",
                    f"  슬롯: {pos.record.slot.value}",
                ]
                rec = pos.record
                eff_t = rec.entry_target_pt if rec.entry_target_pt > 0.0 else self._cfg.target_profit_pt
                eff_s = rec.entry_stop_pt   if rec.entry_stop_pt   > 0.0 else self._cfg.stop_loss_pt
                lines += [
                    f"  목표: +{eff_t:.2f}pt  손절: -{eff_s:.2f}pt",
                    f"  연속 반대신호: {pos.consecutive_reverse}/{self._cfg.reverse_close_count}",
                ]
                if rec.entry_atm_iv > 0.0:
                    gamma_dir = (
                        "Long Gamma" if rec.entry_net_gamma > 0
                        else ("Short Gamma" if rec.entry_net_gamma < 0 else "-")
                    )
                    lines.append(f"  ATM IV: {rec.entry_atm_iv:.1%}  Gamma: {gamma_dir}")
            else:
                lines.append("📌 포지션 없음")

            lines.append("")

            # ── 오늘 결과 ──
            total = state.daily_count
            wins  = state.wins
            losses = state.losses
            pnl   = state.total_pnl_pt
            pnl_emoji = "📈" if pnl > 0 else ("📉" if pnl < 0 else "➖")
            win_rate_str = f"{wins/total:.0%}" if total > 0 else "—"

            lines += [
                f"📊 <b>오늘 결과</b>  ({state.date or '—'})",
                f"  진입: {total}/{self._cfg.max_daily_trades}회  "
                f"승률: {win_rate_str} ({wins}승 {losses}패)",
                f"  {pnl_emoji} 손익: <b>{pnl:+.2f}pt</b>",
            ]

            if state.trade_log:
                lines.append("")
                for i, r in enumerate(state.trade_log, 1):
                    s     = "▲" if r.side == PositionSide.LONG else "▼"
                    t     = r.entry_time.strftime("%H:%M")
                    pnl_i = f"{r.pnl_pt:+.2f}pt"
                    rsn   = r.close_reason.value if r.close_reason else "진행중"
                    lines.append(f"  {i}. {s} {t}  {pnl_i}  ({rsn})")

            msg = "\n".join(lines)
            return bool(self._notifier.send_text(msg, parse_mode="HTML"))
        except Exception:
            logger.exception("[GATE] /trade_status 처리 실패")
            return False

    def _cmd_trade_gate_toggle(self, enable: bool) -> bool:
        """``/trade_gate on|off`` — 런타임 활성화/비활성화."""
        try:
            if self._cfg.enabled == enable:
                status = "이미 활성" if enable else "이미 비활성"
                msg = f"ℹ️ TradeExecutionGate {status} 상태입니다."
            else:
                self._cfg.enabled = enable
                if enable:
                    msg = "✅ <b>TradeExecutionGate 활성화</b>\n거래 신호 처리를 시작합니다."
                    logger.info("[GATE] 런타임 활성화 (텔레그램 명령)")
                else:
                    msg = "⛔ <b>TradeExecutionGate 비활성화</b>\n거래 신호 처리를 중단합니다."
                    logger.info("[GATE] 런타임 비활성화 (텔레그램 명령)")
            return bool(self._notifier.send_text(msg, parse_mode="HTML"))
        except Exception:
            logger.exception("[GATE] /trade_gate on|off 처리 실패")
            return False
