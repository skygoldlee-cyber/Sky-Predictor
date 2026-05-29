"""
tests/test_trade_gate.py
========================
TradeExecutionGate / TradeState 단위 테스트.

외부 의존성(텔레그램, eBest API) 없이 순수 로직만 검증한다.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trading.state import (
    CloseReason,
    DailyState,
    PositionSide,
    TradeRecord,
    TradeSlot,
    TradeStateManager,
    get_trade_slot,
)
from trading.gate import TradeExecutionGate, TradeGateConfig


# ── 픽스처 ────────────────────────────────────────────────────────────────────

def _cfg(**kwargs) -> TradeGateConfig:
    defaults = dict(
        enabled=True,
        max_daily_trades=3,
        min_consecutive_signals=2,
        min_confidence="MEDIUM",
        min_prob_buy=0.62,
        max_prob_sell=0.38,
        require_consensus=True,
        target_profit_pt=2.0,
        stop_loss_pt=1.0,
        force_close_time="14:50",
        slot_a_end="10:30",
        slot_b_end="13:00",
        market_open_time="09:05",
        reverse_close_count=2,
        # 리스크 관리 기본값
        max_consecutive_losses=3,
        max_daily_loss_pt=5.0,
        slot_performance_enabled=False,
        # Trailing Stop 기본값
        trailing_stop_enabled=False,
        trailing_stop_activation_pt=1.0,
        trailing_stop_distance_pt=0.5,
        # Phase 3 기본값
        iv_dynamic_enabled=False,
        iv_target_mult=0.5, iv_stop_mult=0.25,
        iv_target_min=1.5, iv_target_max=5.0,
        iv_stop_min=0.75, iv_stop_max=2.5,
        gamma_gate_enabled=False,
        confidence_dynamic_enabled=False,
        confidence_high_target_mult=1.5, confidence_high_stop_mult=0.8,
        confidence_medium_target_mult=1.0, confidence_medium_stop_mult=1.0,
        confidence_low_target_mult=0.7, confidence_low_stop_mult=1.3,
        history_save_enabled=False,
        history_dir="trade_history",
    )
    defaults.update(kwargs)
    return TradeGateConfig(**defaults)


def _notifier() -> MagicMock:
    n = MagicMock()
    n.send_text.return_value = True
    return n


def _gate(**kwargs) -> TradeExecutionGate:
    return TradeExecutionGate(_notifier(), _cfg(**kwargs))


def _result(signal="BUY", confidence="HIGH", prob=0.75, consensus=True, price=382.5) -> dict:
    return {
        "signal": signal,
        "confidence": confidence,
        "prob": prob,
        "consensus": consensus,
        "current_price": price,
    }


# ── get_trade_slot ────────────────────────────────────────────────────────────

class TestGetTradeSlot:
    def _now(self, hh, mm) -> datetime:
        return datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)

    def test_before_market_open_returns_none(self):
        assert get_trade_slot(self._now(9, 4)) is None

    def test_slot_a(self):
        assert get_trade_slot(self._now(9, 5)) == TradeSlot.A
        assert get_trade_slot(self._now(10, 29)) == TradeSlot.A

    def test_slot_b(self):
        assert get_trade_slot(self._now(10, 30)) == TradeSlot.B
        assert get_trade_slot(self._now(12, 59)) == TradeSlot.B

    def test_slot_c(self):
        assert get_trade_slot(self._now(13, 0)) == TradeSlot.C
        assert get_trade_slot(self._now(14, 49)) == TradeSlot.C

    def test_after_force_close_returns_none(self):
        assert get_trade_slot(self._now(14, 50)) is None
        assert get_trade_slot(self._now(15, 30)) is None


# ── TradeGateConfig ────────────────────────────────────────────────────────────

class TestTradeGateConfig:
    def test_defaults_disabled(self):
        cfg = TradeGateConfig()
        assert cfg.enabled is False

    def test_confidence_ok(self):
        cfg = _cfg(min_confidence="MEDIUM")
        assert cfg.confidence_ok("HIGH") is True
        assert cfg.confidence_ok("MEDIUM") is True
        assert cfg.confidence_ok("LOW") is False

    def test_from_dict(self):
        cfg = TradeGateConfig.from_dict({
            "enabled": True,
            "max_daily_trades": 2,
            "target_profit_pt": 3.0,
        })
        assert cfg.enabled is True
        assert cfg.max_daily_trades == 2
        assert cfg.target_profit_pt == 3.0

    def test_from_dict_empty(self):
        cfg = TradeGateConfig.from_dict({})
        assert cfg.enabled is False


# ── TradeStateManager ─────────────────────────────────────────────────────────

class TestTradeStateManager:
    def test_date_reset(self):
        mgr = TradeStateManager()
        mgr.update("2026-03-01", lambda s: setattr(s, "consecutive_count", 5))
        assert mgr.get_state().consecutive_count == 5
        # 날짜 바뀌면 리셋
        mgr.update("2026-03-02", lambda s: None)
        assert mgr.get_state().consecutive_count == 0

    def test_same_date_persists(self):
        mgr = TradeStateManager()
        mgr.update("2026-03-01", lambda s: setattr(s, "consecutive_count", 3))
        mgr.update("2026-03-01", lambda s: None)
        assert mgr.get_state().consecutive_count == 3


# ── TradeRecord ───────────────────────────────────────────────────────────────

class TestTradeRecord:
    def _record(self, side=PositionSide.LONG, entry=382.0) -> TradeRecord:
        return TradeRecord(
            slot=TradeSlot.A,
            side=side,
            entry_price=entry,
            entry_time=datetime.now(),
            entry_signal="BUY" if side == PositionSide.LONG else "SELL",
            entry_confidence="HIGH",
            entry_prob=0.75,
        )

    def test_pnl_long_win(self):
        r = self._record(PositionSide.LONG, 382.0)
        r.close_price = 384.0
        r.close_time = datetime.now()
        r.close_reason = CloseReason.TARGET_PROFIT
        assert abs(r.pnl_pt - 2.0) < 1e-9

    def test_pnl_short_win(self):
        r = self._record(PositionSide.SHORT, 382.0)
        r.close_price = 380.0
        r.close_time = datetime.now()
        r.close_reason = CloseReason.TARGET_PROFIT
        assert abs(r.pnl_pt - 2.0) < 1e-9

    def test_pnl_long_loss(self):
        r = self._record(PositionSide.LONG, 382.0)
        r.close_price = 381.0
        r.close_time = datetime.now()
        r.close_reason = CloseReason.STOP_LOSS
        assert abs(r.pnl_pt - (-1.0)) < 1e-9

    def test_not_closed_pnl_zero(self):
        r = self._record()
        assert r.pnl_pt == 0.0
        assert r.is_closed is False


# ── TradeExecutionGate — 진입 게이트 ─────────────────────────────────────────

class TestEntryGate:

    def _signal_n_times(self, gate, n, signal="BUY", price=382.5):
        """동일 신호를 n번 연속 전달."""
        for _ in range(n):
            gate.on_signal(_result(signal=signal, price=price), current_price=price)

    @patch("trading.gate.datetime")
    def test_entry_after_consecutive(self, mock_dt):
        """연속 신호 2회 후 진입해야 한다."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(min_consecutive_signals=2)
        self._signal_n_times(gate, 2)
        state = gate._state.get_state()
        assert state.has_position is True
        assert state.active.side == PositionSide.LONG

    @patch("trading.gate.datetime")
    def test_no_entry_before_consecutive(self, mock_dt):
        """연속 횟수 미달 시 진입하면 안 된다."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(min_consecutive_signals=3)
        self._signal_n_times(gate, 2)
        assert gate._state.get_state().has_position is False

    @patch("trading.gate.datetime")
    def test_no_entry_low_confidence(self, mock_dt):
        """LOW confidence는 진입 차단."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(min_confidence="MEDIUM", min_consecutive_signals=1)
        gate.on_signal(_result(confidence="LOW"), current_price=382.5)
        assert gate._state.get_state().has_position is False

    @patch("trading.gate.datetime")
    def test_no_entry_prob_too_low(self, mock_dt):
        """prob 기준 미달 시 진입 차단."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(min_prob_buy=0.65, min_consecutive_signals=1)
        gate.on_signal(_result(prob=0.63), current_price=382.5)
        assert gate._state.get_state().has_position is False

    @patch("trading.gate.datetime")
    def test_no_entry_no_consensus(self, mock_dt):
        """consensus 미달 시 진입 차단."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(require_consensus=True, min_consecutive_signals=1)
        gate.on_signal(_result(consensus=False), current_price=382.5)
        assert gate._state.get_state().has_position is False

    @patch("trading.gate.datetime")
    def test_slot_reuse_blocked(self, mock_dt):
        """같은 슬롯에서 두 번째 진입은 차단해야 한다."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(min_consecutive_signals=1)
        # 첫 번째 진입
        gate.on_signal(_result(), current_price=382.5)
        state = gate._state.get_state()
        assert state.has_position

        # 수동으로 청산 처리
        today = mock_dt.now.return_value.strftime("%Y-%m-%d")
        gate._execute_close(price=384.0, now=mock_dt.now.return_value, today_str=today, reason=CloseReason.TARGET_PROFIT)

        # 같은 슬롯 재진입 시도
        gate._update_consecutive("BUY", today)
        gate._try_enter(
            signal="BUY", confidence="HIGH", prob=0.75, consensus=True,
            price=383.0, now=mock_dt.now.return_value, today_str=today,
            state=gate._state.get_state(),
        )
        # 슬롯 A 이미 사용 → 진입 안 됨
        assert gate._state.get_state().has_position is False

    @patch("trading.gate.datetime")
    def test_max_daily_trades_blocked(self, mock_dt):
        """일일 최대 거래 횟수 초과 시 진입 차단."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(max_daily_trades=1, min_consecutive_signals=1)
        today = mock_dt.now.return_value.strftime("%Y-%m-%d")

        # trade_log에 직접 1건 기록
        fake_record = TradeRecord(
            slot=TradeSlot.B, side=PositionSide.LONG,
            entry_price=380.0, entry_time=mock_dt.now.return_value,
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.7,
        )
        fake_record.close_price = 382.0
        fake_record.close_time = mock_dt.now.return_value
        fake_record.close_reason = CloseReason.TARGET_PROFIT
        def _inject(s: DailyState):
            s.trade_log.append(fake_record)
        gate._state.update(today, _inject)

        gate.on_signal(_result(), current_price=382.5)
        assert gate._state.get_state().has_position is False

    @patch("trading.gate.datetime")
    def test_no_entry_before_market_open(self, mock_dt):
        """장 시작 전(09:04) 진입 차단."""
        mock_dt.now.return_value = datetime.now().replace(hour=9, minute=4)
        gate = _gate(min_consecutive_signals=1)
        gate.on_signal(_result(), current_price=382.5)
        assert gate._state.get_state().has_position is False

    @patch("trading.gate.datetime")
    def test_no_entry_after_force_close_time(self, mock_dt):
        """14:50 이후 진입 차단."""
        mock_dt.now.return_value = datetime.now().replace(hour=14, minute=50)
        gate = _gate(min_consecutive_signals=1)
        gate.on_signal(_result(), current_price=382.5)
        assert gate._state.get_state().has_position is False


# ── TradeExecutionGate — 청산 로직 ────────────────────────────────────────────

class TestCloseLogic:

    def _enter(self, gate, price=382.0, hhmm=(10, 0)):
        """슬롯 A에 매수 포지션 직접 개설."""
        from trading.state import ActivePosition
        now = datetime.now().replace(hour=hhmm[0], minute=hhmm[1])
        today = now.strftime("%Y-%m-%d")
        record = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=price, entry_time=now,
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        active = ActivePosition(record=record)
        def _open(s: DailyState):
            s.active = active
            s.used_slots.append(TradeSlot.A)
        gate._state.update(today, _open)
        return today

    @patch("trading.gate.datetime")
    def test_target_profit_close(self, mock_dt):
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate()
        self._enter(gate, price=382.0)
        gate.check_close(current_price=384.1)  # +2.1pt > target 2.0pt
        state = gate._state.get_state()
        assert state.has_position is False
        assert len(state.trade_log) == 1
        assert state.trade_log[0].close_reason == CloseReason.TARGET_PROFIT
        assert state.trade_log[0].pnl_pt > 0

    @patch("trading.gate.datetime")
    def test_stop_loss_close(self, mock_dt):
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate()
        self._enter(gate, price=382.0)
        gate.check_close(current_price=380.9)  # -1.1pt < -stop 1.0pt
        state = gate._state.get_state()
        assert state.has_position is False
        assert state.trade_log[0].close_reason == CloseReason.STOP_LOSS
        assert state.trade_log[0].pnl_pt < 0

    @patch("trading.gate.datetime")
    def test_no_close_within_range(self, mock_dt):
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate()
        self._enter(gate, price=382.0)
        gate.check_close(current_price=383.0)  # +1.0pt — 목표 미달
        assert gate._state.get_state().has_position is True

    @patch("trading.gate.datetime")
    def test_force_close_after_1450(self, mock_dt):
        """14:50 이후 check_close 호출 시 강제청산."""
        mock_dt.now.return_value = datetime.now().replace(hour=14, minute=51)
        gate = _gate()
        self._enter(gate, price=382.0)
        gate.check_close(current_price=382.5)
        state = gate._state.get_state()
        assert state.has_position is False
        assert state.trade_log[0].close_reason == CloseReason.FORCE_CLOSE

    @patch("trading.gate.datetime")
    def test_reverse_signal_close(self, mock_dt):
        """반대 신호 2회 연속 수신 시 청산."""
        mock_dt.now.return_value = datetime.now().replace(hour=11, minute=0)
        gate = _gate(reverse_close_count=2, min_consecutive_signals=1)
        today = mock_dt.now.return_value.strftime("%Y-%m-%d")

        # 진입 (BUY, LONG)
        gate.on_signal(_result(signal="BUY"), current_price=382.0)
        assert gate._state.get_state().has_position is True

        # 반대 신호 1회 (청산 안 됨)
        gate.on_signal(_result(signal="SELL", prob=0.30), current_price=382.5)
        assert gate._state.get_state().has_position is True

        # 반대 신호 2회 → 청산
        gate.on_signal(_result(signal="SELL", prob=0.30), current_price=382.5)
        assert gate._state.get_state().has_position is False
        assert gate._state.get_state().trade_log[0].close_reason == CloseReason.REVERSE_SIGNAL


# ── 일일 결산 ────────────────────────────────────────────────────────────────

class TestDailySummary:

    def test_summary_dict(self):
        gate = _gate()
        today = datetime.now().strftime("%Y-%m-%d")
        r = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=datetime.now(),
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        r.close_price = 384.0
        r.close_time = datetime.now()
        r.close_reason = CloseReason.TARGET_PROFIT
        def _inject(s: DailyState):
            s.trade_log.append(r)
        gate._state.update(today, _inject)

        summary = gate.get_daily_summary_dict()
        assert summary["count"] == 1
        assert summary["wins"] == 1
        assert summary["losses"] == 0
        assert abs(summary["total_pnl_pt"] - 2.0) < 1e-9

    def test_no_double_summary(self):
        gate = _gate()
        n = gate._notifier
        gate.send_daily_summary()
        gate.send_daily_summary()
        # force=False 이므로 두 번째는 전송 안 됨
        assert n.send_text.call_count == 1

    def test_force_resend(self):
        gate = _gate()
        n = gate._notifier
        gate.send_daily_summary()
        gate.send_daily_summary(force=True)
        assert n.send_text.call_count == 2


# ── disabled 상태 ─────────────────────────────────────────────────────────────

class TestDisabledGate:
    def test_disabled_no_op(self):
        gate = TradeExecutionGate(_notifier(), TradeGateConfig(enabled=False))
        gate.on_signal(_result(), current_price=382.5)
        gate.check_close(current_price=384.0)
        assert gate._state.get_state().has_position is False
        assert gate._notifier.send_text.call_count == 0


# ── Phase 3: 동적 목표/손절 ───────────────────────────────────────────────────

class TestDynamicTargets:

    def test_iv_based_target_clamped_to_max(self):
        """IV=20%, daily_open=820 → raw=82pt → max 클램프 → 5.0pt"""
        gate = _gate(
            iv_dynamic_enabled=True,
            iv_target_mult=0.5, iv_stop_mult=0.25,
            iv_target_min=1.5, iv_target_max=5.0,
            iv_stop_min=0.75, iv_stop_max=2.5,
        )
        t, s = gate._calc_dynamic_targets(atm_iv=0.20, daily_open=820.0)
        assert abs(t - 5.0) < 1e-9
        assert abs(s - 2.5) < 1e-9

    def test_iv_based_target_clamped_to_min(self):
        """IV=0.003%, daily_open=820 → raw=1.23pt → min 클램프 → 1.5pt"""
        gate = _gate(iv_dynamic_enabled=True,
                     iv_target_mult=0.5, iv_target_min=1.5, iv_target_max=5.0,
                     iv_stop_mult=0.25, iv_stop_min=0.75, iv_stop_max=2.5)
        t, s = gate._calc_dynamic_targets(atm_iv=0.003, daily_open=820.0)
        assert abs(t - 1.5) < 1e-9
        assert abs(s - 0.75) < 1e-9

    def test_iv_mid_range(self):
        """IV=1.5%, daily_open=820 → target=clamp(6.15,1.5,5.0)=5.0, stop=clamp(3.075,0.75,2.5)=2.5"""
        gate = _gate(iv_dynamic_enabled=True,
                     iv_target_mult=0.5, iv_target_min=1.5, iv_target_max=5.0,
                     iv_stop_mult=0.25, iv_stop_min=0.75, iv_stop_max=2.5)
        t, s = gate._calc_dynamic_targets(atm_iv=0.015, daily_open=820.0)
        assert abs(t - 5.0) < 1e-9
        assert abs(s - 2.5) < 1e-9

    def test_iv_disabled_returns_config_defaults(self):
        """iv_dynamic_enabled=False → config 기본값 그대로"""
        gate = _gate(
            iv_dynamic_enabled=False,
            target_profit_pt=2.0, stop_loss_pt=1.0,
        )
        t, s = gate._calc_dynamic_targets(atm_iv=0.25, daily_open=820.0)
        assert abs(t - 2.0) < 1e-9
        assert abs(s - 1.0) < 1e-9

    def test_iv_zero_returns_config_defaults(self):
        """IV=0 → config 기본값 fallback"""
        gate = _gate(iv_dynamic_enabled=True, target_profit_pt=2.0, stop_loss_pt=1.0)
        t, s = gate._calc_dynamic_targets(atm_iv=0.0, daily_open=820.0)
        assert abs(t - 2.0) < 1e-9
        assert abs(s - 1.0) < 1e-9

    def test_daily_open_zero_returns_config_defaults(self):
        """daily_open=0 (시가 미기록) → config 기본값 fallback"""
        gate = _gate(iv_dynamic_enabled=True, target_profit_pt=2.0, stop_loss_pt=1.0)
        t, s = gate._calc_dynamic_targets(atm_iv=0.20, daily_open=0.0)
        assert abs(t - 2.0) < 1e-9
        assert abs(s - 1.0) < 1e-9

    def test_daily_open_used_not_current_price(self):
        """기준가는 daily_open이며 current_price와 다른 값을 사용해야 한다."""
        gate = _gate(
            iv_dynamic_enabled=True,
            iv_target_mult=0.5, iv_target_min=0.1, iv_target_max=100.0,
            iv_stop_mult=0.25, iv_stop_min=0.1, iv_stop_max=100.0,
        )
        # daily_open=820, current_price=850 → 계산에 820이 사용되어야 함
        t_open, s_open = gate._calc_dynamic_targets(atm_iv=0.01, daily_open=820.0)
        t_curr, s_curr = gate._calc_dynamic_targets(atm_iv=0.01, daily_open=850.0)
        # 820 기준: 0.01×820×0.5=4.1pt, 0.01×820×0.25=2.05pt
        assert t_open == pytest.approx(0.01 * 820.0 * 0.5)
        assert t_curr == pytest.approx(0.01 * 850.0 * 0.5)
        assert t_open != t_curr

    @patch("trading.gate.datetime")
    def test_entry_record_stores_dynamic_targets(self, mock_dt):
        """진입 시 entry_target_pt / entry_stop_pt 가 TradeRecord에 저장되어야 한다.
        기준가로 daily_open_price(시가)가 사용되어야 한다."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            min_consecutive_signals=1,
            iv_dynamic_enabled=True,
            iv_target_mult=0.5, iv_target_min=1.5, iv_target_max=5.0,
            iv_stop_mult=0.25, iv_stop_min=0.75, iv_stop_max=2.5,
        )
        result = _result(signal="BUY")
        result["options"] = {"atm_iv": 0.20, "net_gamma_proxy": 1.0, "above_vol_trigger": 1.0}
        # current_price=820 → 시가로 기록 후 동적 계산에 사용
        gate.on_signal(result, current_price=820.0)

        state = gate._state.get_state()
        assert state.has_position
        # 시가가 기록됐어야 함
        assert state.daily_open_price == pytest.approx(820.0)
        rec = state.active.record
        assert rec.entry_atm_iv == pytest.approx(0.20)
        # IV=20%, daily_open=820 → raw=82pt → clamp → 5.0pt
        assert rec.entry_target_pt == pytest.approx(5.0)
        assert rec.entry_stop_pt   == pytest.approx(2.5)

    @patch("trading.gate.datetime")
    def test_daily_open_recorded_on_first_tick(self, mock_dt):
        """09:05 이후 첫 번째 유효 가격이 시가로 기록되어야 한다."""
        mock_dt.now.return_value = datetime.now().replace(hour=9, minute=5)
        gate = _gate(min_consecutive_signals=1)
        # 첫 틱 820
        gate.on_signal(_result(signal="BUY"), current_price=820.0)
        assert gate._state.get_state().daily_open_price == pytest.approx(820.0)
        # 두 번째 틱 830 → 시가 변경 없음
        gate.on_signal(_result(signal="BUY"), current_price=830.0)
        assert gate._state.get_state().daily_open_price == pytest.approx(820.0)

    @patch("trading.gate.datetime")
    def test_close_uses_dynamic_target_based_on_open(self, mock_dt):
        """청산 시 daily_open 기반 동적 target이 적용되어야 한다."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            min_consecutive_signals=1,
            target_profit_pt=2.0,          # config 기본
            iv_dynamic_enabled=True,
            iv_target_mult=0.5, iv_target_min=1.5, iv_target_max=3.0,
            iv_stop_mult=0.25, iv_stop_min=0.75, iv_stop_max=2.5,
        )
        # IV=0.3%, daily_open=820 → target=clamp(0.003×820×0.5=1.23, 1.5, 3.0)=1.5pt
        result = _result(signal="BUY")
        result["options"] = {"atm_iv": 0.003, "net_gamma_proxy": 0.0, "above_vol_trigger": 1.0}
        gate.on_signal(result, current_price=820.0)

        # config 기본(2.0pt)은 미달이지만 동적 하한(1.5pt) 초과 → 청산 발동
        gate.check_close(current_price=822.0)  # pnl=2.0pt > dynamic_target=1.5pt
        state = gate._state.get_state()
        assert not state.has_position
        assert state.trade_log[0].close_reason.value == "목표수익"


# ── Phase 3: Dealer Gamma 게이트 ─────────────────────────────────────────────

class TestGammaGate:

    @patch("trading.gate.datetime")
    def test_buy_blocked_dealer_short_gamma(self, mock_dt):
        """딜러 Short Gamma + Vol Trigger 아래 → BUY 차단"""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(gamma_gate_enabled=True, min_consecutive_signals=1)
        result = _result(signal="BUY")
        result["options"] = {"net_gamma_proxy": -2.0, "above_vol_trigger": 0.0}
        gate.on_signal(result, current_price=382.5)
        assert gate._state.get_state().has_position is False

    @patch("trading.gate.datetime")
    def test_sell_blocked_dealer_long_gamma(self, mock_dt):
        """딜러 Long Gamma + Vol Trigger 위 → SELL 차단"""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(gamma_gate_enabled=True, min_consecutive_signals=1)
        result = _result(signal="SELL", prob=0.30)
        result["options"] = {"net_gamma_proxy": 3.0, "above_vol_trigger": 1.0}
        gate.on_signal(result, current_price=382.5)
        assert gate._state.get_state().has_position is False

    @patch("trading.gate.datetime")
    def test_buy_allowed_dealer_long_gamma(self, mock_dt):
        """딜러 Long Gamma → BUY 허용"""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(gamma_gate_enabled=True, min_consecutive_signals=1)
        result = _result(signal="BUY")
        result["options"] = {"net_gamma_proxy": 2.0, "above_vol_trigger": 1.0}
        gate.on_signal(result, current_price=382.5)
        assert gate._state.get_state().has_position is True

    @patch("trading.gate.datetime")
    def test_gamma_zero_passes(self, mock_dt):
        """net_gamma=0 (데이터 없음) → 게이트 통과"""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(gamma_gate_enabled=True, min_consecutive_signals=1)
        result = _result(signal="BUY")
        result["options"] = {"net_gamma_proxy": 0.0, "above_vol_trigger": 1.0}
        gate.on_signal(result, current_price=382.5)
        assert gate._state.get_state().has_position is True

    @patch("trading.gate.datetime")
    def test_gamma_gate_disabled(self, mock_dt):
        """gamma_gate_enabled=False → 딜러 Short Gamma여도 BUY 허용"""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(gamma_gate_enabled=False, min_consecutive_signals=1)
        result = _result(signal="BUY")
        result["options"] = {"net_gamma_proxy": -5.0, "above_vol_trigger": 0.0}
        gate.on_signal(result, current_price=382.5)
        assert gate._state.get_state().has_position is True


# ── Phase 3: 이력 저장 ───────────────────────────────────────────────────────

class TestHistorySave:

    def test_save_history_creates_jsonl(self, tmp_path):
        """청산 시 JSONL 파일이 생성되어야 한다."""
        gate = TradeExecutionGate(
            _notifier(),
            _cfg(history_save_enabled=True, history_dir=str(tmp_path)),
        )
        rec = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=datetime.now(),
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        rec.close_price = 384.0
        rec.close_time  = datetime.now()
        rec.close_reason = CloseReason.TARGET_PROFIT
        gate._save_history(rec)

        date_str = rec.entry_time.strftime("%Y-%m-%d")
        jsonl_path = tmp_path / f"{date_str}.jsonl"
        assert jsonl_path.exists()

        import json
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["side"] == "LONG"
        assert abs(data["pnl_pt"] - 2.0) < 1e-9

    def test_save_history_appends(self, tmp_path):
        """복수 청산 시 같은 파일에 append 되어야 한다."""
        gate = TradeExecutionGate(
            _notifier(),
            _cfg(history_save_enabled=True, history_dir=str(tmp_path)),
        )
        for _ in range(3):
            rec = TradeRecord(
                slot=TradeSlot.A, side=PositionSide.LONG,
                entry_price=382.0, entry_time=datetime.now(),
                entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
            )
            rec.close_price = 384.0
            rec.close_time  = datetime.now()
            rec.close_reason = CloseReason.TARGET_PROFIT
            gate._save_history(rec)

        date_str = rec.entry_time.strftime("%Y-%m-%d")
        lines = (tmp_path / f"{date_str}.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_save_history_disabled(self, tmp_path):
        """history_save_enabled=False → 파일 미생성"""
        gate = TradeExecutionGate(
            _notifier(),
            _cfg(history_save_enabled=False, history_dir=str(tmp_path)),
        )
        rec = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=datetime.now(),
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        rec.close_price = 384.0
        rec.close_time  = datetime.now()
        rec.close_reason = CloseReason.TARGET_PROFIT
        gate._save_history(rec)
        assert not any(tmp_path.iterdir())


# ── Phase 3: TradeGateConfig from_dict ───────────────────────────────────────

class TestPhase3Config:

    def test_from_dict_phase3_keys(self):
        cfg = TradeGateConfig.from_dict({
            "enabled": True,
            "iv_dynamic_enabled": False,
            "iv_target_mult": 0.8,
            "iv_target_min": 2.0,
            "iv_target_max": 6.0,
            "iv_stop_mult": 0.4,
            "iv_stop_min": 1.0,
            "iv_stop_max": 3.0,
            "gamma_gate_enabled": True,
            "history_save_enabled": False,
            "history_dir": "/tmp/th",
        })
        assert cfg.iv_dynamic_enabled is False
        assert cfg.iv_target_mult == pytest.approx(0.8)
        assert cfg.iv_target_max == pytest.approx(6.0)
        assert cfg.gamma_gate_enabled is True
        assert cfg.history_save_enabled is False
        assert cfg.history_dir == "/tmp/th"

    def test_phase3_defaults(self):
        cfg = TradeGateConfig.from_dict({})
        assert cfg.iv_dynamic_enabled is True
        assert cfg.gamma_gate_enabled is False
        assert cfg.history_save_enabled is True


# ── Phase 3: trade_id 마이크로초 ID ──────────────────────────────────────────

class TestTradeId:

    def test_trade_id_auto_generated(self):
        """trade_id 가 자동으로 생성되어야 한다."""
        rec = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=datetime.now(),
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        assert rec.trade_id != ""

    def test_trade_id_format(self):
        """trade_id 형식이 YYYYMMDD_HHMMSS_ffffff 이어야 한다."""
        import re
        now = datetime(2026, 3, 22, 10, 5, 12, 834291)
        rec = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=now,
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        assert re.match(r"^\d{8}_\d{6}_\d{6}$", rec.trade_id), f"형식 오류: {rec.trade_id}"
        assert rec.trade_id.startswith("20260322_100512_834291")

    def test_trade_id_unique_per_microsecond(self):
        """entry_time 이 다르면 trade_id 도 달라야 한다."""
        t1 = datetime(2026, 3, 22, 10, 0, 0, 1)
        t2 = datetime(2026, 3, 22, 10, 0, 0, 2)
        r1 = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=t1,
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        r2 = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=t2,
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        assert r1.trade_id != r2.trade_id

    def test_trade_id_explicit_override(self):
        """명시적으로 지정한 trade_id 는 덮어씌워지지 않아야 한다."""
        rec = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=datetime.now(),
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
            trade_id="CUSTOM_ID_001",
        )
        assert rec.trade_id == "CUSTOM_ID_001"

    def test_trade_id_in_to_dict(self):
        """to_dict() 에 trade_id 가 포함되어야 한다."""
        rec = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=datetime.now(),
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        d = rec.to_dict()
        assert "trade_id" in d
        assert d["trade_id"] == rec.trade_id

    def test_trade_id_from_dict_roundtrip(self):
        """to_dict() → from_dict() 후에도 trade_id 가 보존되어야 한다."""
        from trading.state import TradeRecord as TR
        rec = TR(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=datetime(2026, 3, 22, 10, 0, 0, 123456),
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        rec.close_price  = 384.0
        rec.close_time   = datetime(2026, 3, 22, 10, 30, 0)
        rec.close_reason = CloseReason.TARGET_PROFIT
        restored = TR.from_dict(rec.to_dict())
        assert restored.trade_id == rec.trade_id


# ── Phase 3: 텔레그램 명령 핸들러 ────────────────────────────────────────────

class TestTelegramCommands:

    def test_trade_status_no_position(self):
        """/trade_status — 포지션 없을 때 응답을 전송해야 한다."""
        gate = _gate()
        result = gate.handle_telegram_command("/trade_status")
        assert result is True
        gate._notifier.send_text.assert_called_once()
        msg = gate._notifier.send_text.call_args[0][0]
        assert "포지션 없음" in msg

    @patch("trading.gate.datetime")
    def test_trade_status_with_position(self, mock_dt):
        """/trade_status — 포지션 보유 중 진입가·슬롯이 포함되어야 한다."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(min_consecutive_signals=1)
        gate.on_signal(_result(), current_price=382.5)
        assert gate._state.get_state().has_position

        gate._notifier.send_text.reset_mock()
        result = gate.handle_telegram_command("/trade_status")
        assert result is True
        msg = gate._notifier.send_text.call_args[0][0]
        assert "382.5" in msg or "382.50" in msg
        assert "슬롯" in msg

    def test_trade_gate_on(self):
        """/trade_gate on — 비활성 → 활성 전환이 되어야 한다."""
        gate = TradeExecutionGate(_notifier(), TradeGateConfig(enabled=False))
        assert gate.enabled is False
        result = gate.handle_telegram_command("/trade_gate on")
        assert result is True
        assert gate.enabled is True

    def test_trade_gate_off(self):
        """/trade_gate off — 활성 → 비활성 전환이 되어야 한다."""
        gate = _gate()
        assert gate.enabled is True
        result = gate.handle_telegram_command("/trade_gate off")
        assert result is True
        assert gate.enabled is False

    def test_trade_gate_on_already_on(self):
        """/trade_gate on — 이미 활성이면 '이미 활성' 메시지 전송."""
        gate = _gate()
        result = gate.handle_telegram_command("/trade_gate on")
        assert result is True
        msg = gate._notifier.send_text.call_args[0][0]
        assert "이미" in msg

    def test_trade_gate_off_already_off(self):
        """/trade_gate off — 이미 비활성이면 '이미 비활성' 메시지 전송."""
        gate = TradeExecutionGate(_notifier(), TradeGateConfig(enabled=False))
        result = gate.handle_telegram_command("/trade_gate off")
        assert result is True
        msg = gate._notifier.send_text.call_args[0][0]
        assert "이미" in msg

    def test_unknown_command_returns_false(self):
        """알 수 없는 명령은 False 를 반환해야 한다."""
        gate = _gate()
        result = gate.handle_telegram_command("/unknown_cmd")
        assert result is False
        gate._notifier.send_text.assert_not_called()

    def test_trade_gate_off_then_on_signal_no_op(self):
        """/trade_gate off 후 on_signal 은 no-op 이어야 한다."""
        gate = _gate()
        gate.handle_telegram_command("/trade_gate off")
        gate.on_signal(_result(), current_price=382.5)
        assert gate._state.get_state().has_position is False

    def test_trade_status_shows_today_pnl(self):
        """/trade_status — 오늘 손익 합계가 포함되어야 한다."""
        gate = _gate()
        today = datetime.now().strftime("%Y-%m-%d")
        r = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=datetime.now(),
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        r.close_price  = 384.0
        r.close_time   = datetime.now()
        r.close_reason = CloseReason.TARGET_PROFIT
        def _inject(s: DailyState):
            s.trade_log.append(r)
        gate._state.update(today, _inject)

        gate._notifier.send_text.reset_mock()
        gate.handle_telegram_command("/trade_status")
        msg = gate._notifier.send_text.call_args[0][0]
        # 손익 +2.00pt 가 메시지에 포함되어야 함
        assert "+2.00" in msg

    def test_trade_status_win_rate(self):
        """/trade_status — 승률이 메시지에 포함되어야 한다."""
        gate = _gate()
        today = datetime.now().strftime("%Y-%m-%d")
        for entry, close, reason in [
            (380.0, 382.0, CloseReason.TARGET_PROFIT),
            (380.0, 379.0, CloseReason.STOP_LOSS),
        ]:
            r = TradeRecord(
                slot=TradeSlot.A, side=PositionSide.LONG,
                entry_price=entry, entry_time=datetime.now(),
                entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
            )
            r.close_price  = close
            r.close_time   = datetime.now()
            r.close_reason = reason
            def _inject(s: DailyState, rec=r):
                s.trade_log.append(rec)
            gate._state.update(today, _inject)

        gate._notifier.send_text.reset_mock()
        gate.handle_telegram_command("/trade_status")
        msg = gate._notifier.send_text.call_args[0][0]
        # 2건 중 1승 → 50%
        assert "50%" in msg


# ── DailyState.summary_dict win_rate 필드 ─────────────────────────────────

class TestSummaryDictWinRate:

    def test_win_rate_zero_trades(self):
        """거래 없을 때 win_rate = 0.0."""
        state = DailyState(date="2026-03-22")
        d = state.summary_dict()
        assert d["win_rate"] == 0.0

    def test_win_rate_all_wins(self):
        """전부 승리 시 win_rate = 1.0."""
        state = DailyState(date="2026-03-22")
        for i in range(3):
            r = TradeRecord(
                slot=TradeSlot.A, side=PositionSide.LONG,
                entry_price=380.0, entry_time=datetime.now(),
                entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
            )
            r.close_price  = 382.0
            r.close_time   = datetime.now()
            r.close_reason = CloseReason.TARGET_PROFIT
            state.trade_log.append(r)
        d = state.summary_dict()
        assert d["win_rate"] == pytest.approx(1.0)

    def test_win_rate_half(self):
        """1승 1패 → win_rate = 0.5."""
        state = DailyState(date="2026-03-22")
        for entry, close, reason in [
            (380.0, 382.0, CloseReason.TARGET_PROFIT),
            (382.0, 381.0, CloseReason.STOP_LOSS),
        ]:
            r = TradeRecord(
                slot=TradeSlot.A, side=PositionSide.LONG,
                entry_price=entry, entry_time=datetime.now(),
                entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
            )
            r.close_price  = close
            r.close_time   = datetime.now()
            r.close_reason = reason
            state.trade_log.append(r)
        d = state.summary_dict()
        assert d["win_rate"] == pytest.approx(0.5)

    def test_win_rate_excludes_draw(self):
        """무승부(pnl=0)는 win_rate 분모에서 제외 — 1승 1무 → 1.0."""
        state = DailyState(date="2026-03-22")
        for entry, close, reason in [
            (380.0, 382.0, CloseReason.TARGET_PROFIT),   # 승
            (382.0, 382.0, CloseReason.FORCE_CLOSE),     # 무 (pnl=0)
        ]:
            r = TradeRecord(
                slot=TradeSlot.A, side=PositionSide.LONG,
                entry_price=entry, entry_time=datetime.now(),
                entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
            )
            r.close_price  = close
            r.close_time   = datetime.now()
            r.close_reason = reason
            state.trade_log.append(r)
        d = state.summary_dict()
        # wins=1, losses=0 → win_rate = 1/(1+0) = 1.0
        assert d["win_rate"] == pytest.approx(1.0)


# ── 리스크 관리: 최대 연속 손실 제한 ─────────────────────────────────────────

class TestConsecutiveLossLimit:

    @patch("trading.gate.datetime")
    def test_entry_blocked_after_consecutive_losses(self, mock_dt):
        """연속 손실 3회 후 진입 차단."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            max_consecutive_losses=3,
            min_consecutive_signals=1,
        )
        today = mock_dt.now.return_value.strftime("%Y-%m-%d")

        # 3회 연속 손실 기록
        for _ in range(3):
            rec = TradeRecord(
                slot=TradeSlot.A, side=PositionSide.LONG,
                entry_price=382.0, entry_time=mock_dt.now.return_value,
                entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
            )
            rec.close_price = 381.0
            rec.close_time = mock_dt.now.return_value
            rec.close_reason = CloseReason.STOP_LOSS
            def _inject(s: DailyState):
                s.trade_log.append(rec)
                s.consecutive_losses += 1
            gate._state.update(today, _inject)

        # 4번째 진입 시도 → 차단
        gate.on_signal(_result(), current_price=382.5)
        assert gate._state.get_state().has_position is False

    @patch("trading.gate.datetime")
    def test_consecutive_losses_reset_on_win(self, mock_dt):
        """승리 시 연속 손실 초기화."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            max_consecutive_losses=3,
            max_daily_trades=10,  # 충분히 높게 설정
            min_consecutive_signals=1,
        )
        today = mock_dt.now.return_value.strftime("%Y-%m-%d")

        # 2회 손실 (슬롯 A, B 사용)
        for i, slot in enumerate([TradeSlot.A, TradeSlot.B]):
            rec = TradeRecord(
                slot=slot, side=PositionSide.LONG,
                entry_price=382.0, entry_time=mock_dt.now.return_value,
                entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
            )
            rec.close_price = 381.0
            rec.close_time = mock_dt.now.return_value
            rec.close_reason = CloseReason.STOP_LOSS
            def _inject(s: DailyState):
                s.trade_log.append(rec)
                s.consecutive_losses += 1
                s.used_slots.append(slot)
            gate._state.update(today, _inject)

        # 1회 승리 (슬롯 C) → 초기화
        rec_win = TradeRecord(
            slot=TradeSlot.C, side=PositionSide.LONG,
            entry_price=382.0, entry_time=mock_dt.now.return_value,
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        rec_win.close_price = 384.0
        rec_win.close_time = mock_dt.now.return_value
        rec_win.close_reason = CloseReason.TARGET_PROFIT
        def _inject_win(s: DailyState):
            s.trade_log.append(rec_win)
            s.consecutive_losses = 0
            s.used_slots.append(TradeSlot.C)
        gate._state.update(today, _inject_win)

        # 진입 허용 (슬롯 A 재사용 가능하도록 초기화)
        def _reset_slots(s: DailyState):
            s.used_slots = []
        gate._state.update(today, _reset_slots)
        gate.on_signal(_result(), current_price=382.5)
        assert gate._state.get_state().has_position is True

    @patch("trading.gate.datetime")
    def test_consecutive_losses_disabled(self, mock_dt):
        """max_consecutive_losses=0 → 비활성."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            max_consecutive_losses=0,
            max_daily_trades=20,  # 충분히 높게 설정
            max_daily_loss_pt=0.0,  # 비활성
            min_consecutive_signals=1,
        )
        today = mock_dt.now.return_value.strftime("%Y-%m-%d")

        # 10회 손실 기록 (슬롯 순환 사용)
        for i in range(10):
            slot = TradeSlot.A if i % 3 == 0 else (TradeSlot.B if i % 3 == 1 else TradeSlot.C)
            rec = TradeRecord(
                slot=slot, side=PositionSide.LONG,
                entry_price=382.0, entry_time=mock_dt.now.return_value,
                entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
            )
            rec.close_price = 381.0
            rec.close_time = mock_dt.now.return_value
            rec.close_reason = CloseReason.STOP_LOSS
            def _inject(s: DailyState):
                s.trade_log.append(rec)
                s.consecutive_losses += 1
                s.used_slots.append(slot)
            gate._state.update(today, _inject)

        # 진입 허용 (슬롯 초기화 후)
        def _reset_slots(s: DailyState):
            s.used_slots = []
        gate._state.update(today, _reset_slots)
        gate.on_signal(_result(), current_price=382.5)
        assert gate._state.get_state().has_position is True


# ── 리스크 관리: 일일 최대 손실 제한 ─────────────────────────────────────────

class TestDailyLossLimit:

    @patch("trading.gate.datetime")
    def test_entry_blocked_after_daily_loss_limit(self, mock_dt):
        """일일 손실 5pt 초과 시 진입 차단."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            max_daily_loss_pt=5.0,
            min_consecutive_signals=1,
        )
        today = mock_dt.now.return_value.strftime("%Y-%m-%d")

        # 6pt 손실 기록
        rec = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=mock_dt.now.return_value,
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        rec.close_price = 376.0
        rec.close_time = mock_dt.now.return_value
        rec.close_reason = CloseReason.STOP_LOSS
        def _inject(s: DailyState):
            s.trade_log.append(rec)
        gate._state.update(today, _inject)

        # 진입 시도 → 차단
        gate.on_signal(_result(), current_price=382.5)
        assert gate._state.get_state().has_position is False

    @patch("trading.gate.datetime")
    def test_daily_loss_disabled(self, mock_dt):
        """max_daily_loss_pt=0 → 비활성."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            max_daily_loss_pt=0.0,
            min_consecutive_signals=1,
        )
        today = mock_dt.now.return_value.strftime("%Y-%m-%d")

        # 10pt 손실 기록
        rec = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=mock_dt.now.return_value,
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        rec.close_price = 372.0
        rec.close_time = mock_dt.now.return_value
        rec.close_reason = CloseReason.STOP_LOSS
        def _inject(s: DailyState):
            s.trade_log.append(rec)
        gate._state.update(today, _inject)

        # 진입 허용
        gate.on_signal(_result(), current_price=382.5)
        assert gate._state.get_state().has_position is True


# ── 리스크 관리: 슬롯별 성과 기반 할당 ───────────────────────────────────────

class TestSlotPerformance:

    @patch("trading.gate.datetime")
    def test_slot_performance_tracking(self, mock_dt):
        """청산 시 슬롯별 성과 추적."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(min_consecutive_signals=1, max_daily_trades=10)
        today = mock_dt.now.return_value.strftime("%Y-%m-%d")

        # 슬롯 A: 승리 (실제 청산 로직 사용)
        from trading.state import ActivePosition
        rec_a = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=382.0, entry_time=mock_dt.now.return_value,
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        active_a = ActivePosition(record=rec_a, trailing_stop_price=0.0)
        def _open_a(s: DailyState):
            s.active = active_a
            s.used_slots.append(TradeSlot.A)
        gate._state.update(today, _open_a)
        gate._execute_close(price=384.0, now=mock_dt.now.return_value, today_str=today, reason=CloseReason.TARGET_PROFIT)

        # 슬롯 B: 패배
        rec_b = TradeRecord(
            slot=TradeSlot.B, side=PositionSide.LONG,
            entry_price=382.0, entry_time=mock_dt.now.return_value,
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        active_b = ActivePosition(record=rec_b, trailing_stop_price=0.0)
        def _open_b(s: DailyState):
            s.active = active_b
            s.used_slots.append(TradeSlot.B)
        gate._state.update(today, _open_b)
        gate._execute_close(price=381.0, now=mock_dt.now.return_value, today_str=today, reason=CloseReason.STOP_LOSS)

        state = gate._state.get_state()
        assert state.slot_performance["A"]["total"] == 1
        assert state.slot_performance["A"]["wins"] == 1
        assert state.slot_performance["A"]["pnl"] == pytest.approx(2.0)
        assert state.slot_performance["B"]["total"] == 1
        assert state.slot_performance["B"]["wins"] == 0
        assert state.slot_performance["B"]["pnl"] == pytest.approx(-1.0)


# ── Trailing Stop-loss ───────────────────────────────────────────────────────

class TestTrailingStop:

    def _enter(self, gate, price=382.0, hhmm=(10, 0)):
        """슬롯 A에 매수 포지션 직접 개설."""
        from trading.state import ActivePosition
        now = datetime.now().replace(hour=hhmm[0], minute=hhmm[1])
        today = now.strftime("%Y-%m-%d")
        record = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.LONG,
            entry_price=price, entry_time=now,
            entry_signal="BUY", entry_confidence="HIGH", entry_prob=0.75,
        )
        active = ActivePosition(record=record, trailing_stop_price=0.0)
        def _open(s: DailyState):
            s.active = active
            s.used_slots.append(TradeSlot.A)
        gate._state.update(today, _open)
        return today

    @patch("trading.gate.datetime")
    def test_trailing_stop_activation(self, mock_dt):
        """이익 1.0pt 도달 시 Trailing Stop 활성화."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            trailing_stop_enabled=True,
            trailing_stop_activation_pt=1.0,
            trailing_stop_distance_pt=0.5,
            target_profit_pt=100.0,  # 목표수익 청산 방지
            stop_loss_pt=100.0,      # 손절 청산 방지
        )
        self._enter(gate, price=382.0)

        # 가격 383.0 → pnl=1.0pt → trailing_stop=382.5 (활성화)
        gate.check_close(current_price=383.0)
        state = gate._state.get_state()
        assert state.active.trailing_stop_price == pytest.approx(382.5)

    @patch("trading.gate.datetime")
    def test_trailing_stop_update(self, mock_dt):
        """가격 상승 시 trailing_stop_price 상향 이동."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            trailing_stop_enabled=True,
            trailing_stop_activation_pt=1.0,
            trailing_stop_distance_pt=0.5,
            target_profit_pt=100.0,  # 목표수익 청산 방지
            stop_loss_pt=100.0,      # 손절 청산 방지
        )
        self._enter(gate, price=382.0)

        # 가격 383.0 → pnl=1.0pt → trailing_stop=382.5
        gate.check_close(current_price=383.0)
        # 가격 384.0 → pnl=2.0pt → trailing_stop=383.5 (업데이트)
        gate.check_close(current_price=384.0)
        state = gate._state.get_state()
        assert state.active.trailing_stop_price == pytest.approx(383.5)

    @patch("trading.gate.datetime")
    def test_trailing_stop_close(self, mock_dt):
        """가격이 trailing_stop_price 이하 도달 시 청산."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            trailing_stop_enabled=True,
            trailing_stop_activation_pt=1.0,
            trailing_stop_distance_pt=0.5,
            target_profit_pt=100.0,  # 목표수익 청산 방지
            stop_loss_pt=100.0,      # 손절 청산 방지
        )
        self._enter(gate, price=382.0)

        # 가격 383.0 → trailing_stop=382.5
        gate.check_close(current_price=383.0)
        # 가격 382.3 → 382.3 ≤ 382.5 → 청산
        gate.check_close(current_price=382.3)
        state = gate._state.get_state()
        assert state.has_position is False
        assert state.trade_log[0].close_reason == CloseReason.TRAILING_STOP

    @patch("trading.gate.datetime")
    def test_trailing_stop_disabled(self, mock_dt):
        """trailing_stop_enabled=False → 동작 안 함."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            trailing_stop_enabled=False,
            trailing_stop_activation_pt=1.0,
            trailing_stop_distance_pt=0.5,
            target_profit_pt=100.0,  # 목표수익 청산 방지
            stop_loss_pt=100.0,      # 손절 청산 방지
        )
        self._enter(gate, price=382.0)

        gate.check_close(current_price=383.0)
        state = gate._state.get_state()
        assert state.active.trailing_stop_price == 0.0

    @patch("trading.gate.datetime")
    def test_trailing_stop_short(self, mock_dt):
        """SHORT 포지션 Trailing Stop."""
        from trading.state import ActivePosition
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            trailing_stop_enabled=True,
            trailing_stop_activation_pt=1.0,
            trailing_stop_distance_pt=0.5,
            target_profit_pt=100.0,  # 목표수익 청산 방지
            stop_loss_pt=100.0,      # 손절 청산 방지
        )
        # SHORT 진입
        now = mock_dt.now.return_value
        today = now.strftime("%Y-%m-%d")
        record = TradeRecord(
            slot=TradeSlot.A, side=PositionSide.SHORT,
            entry_price=382.0, entry_time=now,
            entry_signal="SELL", entry_confidence="HIGH", entry_prob=0.25,
        )
        active = ActivePosition(record=record, trailing_stop_price=0.0)
        def _open(s: DailyState):
            s.active = active
            s.used_slots.append(TradeSlot.A)
        gate._state.update(today, _open)

        # 가격 381.0 → pnl=1.0pt → trailing_stop=381.5 (활성화)
        gate.check_close(current_price=381.0)
        state = gate._state.get_state()
        assert state.active.trailing_stop_price == pytest.approx(381.5)

        # 가격 381.6 → 381.6 ≥ 381.5 → 청산
        gate.check_close(current_price=381.6)
        assert state.has_position is False
        assert state.trade_log[0].close_reason == CloseReason.TRAILING_STOP


# ── 신뢰도 기반 동적 목표/손절 ───────────────────────────────────────────────

class TestConfidenceDynamicTargets:

    def test_get_confidence_multiplier_high(self):
        """HIGH confidence 배수 반환."""
        cfg = _cfg(
            confidence_dynamic_enabled=True,
            confidence_high_target_mult=1.5,
            confidence_high_stop_mult=0.8,
        )
        target_mult, stop_mult = cfg.get_confidence_multiplier("HIGH")
        assert target_mult == pytest.approx(1.5)
        assert stop_mult == pytest.approx(0.8)

    def test_get_confidence_multiplier_medium(self):
        """MEDIUM confidence 배수 반환."""
        cfg = _cfg(
            confidence_dynamic_enabled=True,
            confidence_medium_target_mult=1.0,
            confidence_medium_stop_mult=1.0,
        )
        target_mult, stop_mult = cfg.get_confidence_multiplier("MEDIUM")
        assert target_mult == pytest.approx(1.0)
        assert stop_mult == pytest.approx(1.0)

    def test_get_confidence_multiplier_low(self):
        """LOW confidence 배수 반환."""
        cfg = _cfg(
            confidence_dynamic_enabled=True,
            confidence_low_target_mult=0.7,
            confidence_low_stop_mult=1.3,
        )
        target_mult, stop_mult = cfg.get_confidence_multiplier("LOW")
        assert target_mult == pytest.approx(0.7)
        assert stop_mult == pytest.approx(1.3)

    @patch("trading.gate.datetime")
    def test_dynamic_targets_with_confidence(self, mock_dt):
        """신뢰도 기반 배수 적용 테스트."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            min_consecutive_signals=1,
            iv_dynamic_enabled=True,
            iv_target_mult=0.5, iv_target_min=1.5, iv_target_max=5.0,
            iv_stop_mult=0.25, iv_stop_min=0.75, iv_stop_max=2.5,
            confidence_dynamic_enabled=True,
            confidence_high_target_mult=1.5,
            confidence_high_stop_mult=0.8,
        )
        # IV=20%, daily_open=820 → base_target=5.0, base_stop=2.5
        # HIGH confidence → target=7.5, stop=2.0
        t, s = gate._calc_dynamic_targets(
            atm_iv=0.20, daily_open=820.0, confidence="HIGH"
        )
        assert t == pytest.approx(7.5)
        assert s == pytest.approx(2.0)

    @patch("trading.gate.datetime")
    def test_confidence_disabled_uses_base(self, mock_dt):
        """confidence_dynamic_enabled=False → 기본값 사용."""
        mock_dt.now.return_value = datetime.now().replace(hour=10, minute=0)
        gate = _gate(
            iv_dynamic_enabled=True,
            iv_target_mult=0.5, iv_target_min=1.5, iv_target_max=5.0,
            iv_stop_mult=0.25, iv_stop_min=0.75, iv_stop_max=2.5,
            confidence_dynamic_enabled=False,
        )
        # IV=20%, daily_open=820 → base_target=5.0, base_stop=2.5
        # confidence 무시 → target=5.0, stop=2.5
        t, s = gate._calc_dynamic_targets(
            atm_iv=0.20, daily_open=820.0, confidence="HIGH"
        )
        assert t == pytest.approx(5.0)
        assert s == pytest.approx(2.5)

