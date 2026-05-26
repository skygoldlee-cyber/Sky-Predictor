"""MarketStructureBreak + OIStructureGate 단위 테스트

검증 항목
---------
1. MarketStructureBreak
   - BOS_UP : swing high 상향 돌파 탐지
   - BOS_DOWN : swing low 하향 돌파 탐지
   - CHoCH_UP : 하락 구조 중 swing high 돌파 → 반전 신호
   - CHoCH_DOWN : 상승 구조 중 swing low 돌파 → 반전 신호
   - 외부 PivotPoint 주입 (ATRAdaptivePivot 연동)
   - 내부 스윙 탐지 (독립 모드)
   - 구조 분석 (UPTREND / DOWNTREND / RANGING)
   - msb_score 범위 [0, 1]
   - get_transformer_features 키 및 값 범위
   - reset() 후 상태 초기화

2. OIStructureGate
   - OI peak 근접 시 점수 부스트
   - OI peak 미근접 시 msb_score 그대로
   - OI levels = None 시 base score 반환
   - get_transformer_features 키 집합

3. ATRAdaptivePivot + MSB 통합
   - 두 지표 동시 사용 예외 없음
   - pivot_points 주입 시 swing 목록 동기화
"""
from __future__ import annotations

import datetime
import math
from typing import List

import pytest

from indicators import (
    ATRAdaptivePivot,
    ATRAdaptivePivotConfig,
    MarketStructureBreak,
    MSBConfig,
    MSBState,
    OIStructureGate,
    OIStructureConfig,
    BOSType,
    StructureType,
    PivotPoint,
    PivotType,
)


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

DT = datetime.datetime(2026, 5, 2, 9, 0)


def _t(minutes: int) -> datetime.datetime:
    return DT + datetime.timedelta(minutes=minutes)


def _msb(bos_buffer_pct: float = 0.0, **kw) -> MarketStructureBreak:
    return MarketStructureBreak(MSBConfig(bos_buffer_pct=bos_buffer_pct, **kw))


def _fake_pivot(idx: int, price: float, ptype: PivotType, t: str = "09:00") -> PivotPoint:
    return PivotPoint(index=idx, price=price, pivot_type=ptype,
                      atr=1.0, bar_time=t)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BOS 탐지
# ─────────────────────────────────────────────────────────────────────────────

class TestBOSDetection:

    def test_bos_up_detected(self) -> None:
        """이전 swing high 를 상향 돌파하면 BOS_UP 이 발생해야 한다."""
        msb = _msb(bos_buffer_pct=0.0, swing_lookback=2, choch_enabled=False)

        # 스윙 주입: SH=365, SL=355, SH=368 → ref_sh=365
        pivots = [
            _fake_pivot(0,  365.0, PivotType.HIGH, "09:00"),
            _fake_pivot(5,  355.0, PivotType.LOW,  "09:05"),
            _fake_pivot(10, 368.0, PivotType.HIGH, "09:10"),
        ]

        signals = []
        for i in range(20):
            # bar 15: 370.0 돌파 (ref_sh=365)
            h = 370.0 if i == 15 else 360.0
            s = msb.update(high=h, low=h - 1.0, close=h - 0.5,
                           bar_time=_t(i), pivot_points=pivots)
            signals.append(s.bos_signal)

        assert BOSType.BOS_UP in signals, \
            f"BOS_UP 미발생: {[s.value for s in signals if s != BOSType.NONE]}"

    def test_bos_down_detected(self) -> None:
        """이전 swing low 를 하향 돌파하면 BOS_DOWN 이 발생해야 한다."""
        msb = _msb(bos_buffer_pct=0.0, swing_lookback=2, choch_enabled=False)

        pivots = [
            _fake_pivot(0,  355.0, PivotType.LOW,  "09:00"),
            _fake_pivot(5,  365.0, PivotType.HIGH, "09:05"),
            _fake_pivot(10, 352.0, PivotType.LOW,  "09:10"),
        ]

        signals = []
        for i in range(20):
            l = 349.0 if i == 15 else 360.0
            s = msb.update(high=l + 2.0, low=l, close=l + 1.0,
                           bar_time=_t(i), pivot_points=pivots)
            signals.append(s.bos_signal)

        assert BOSType.BOS_DOWN in signals, \
            f"BOS_DOWN 미발생: {[s.value for s in signals if s != BOSType.NONE]}"

    def test_no_bos_without_breakout(self) -> None:
        """스윙 레벨을 돌파하지 않으면 BOS 가 발생하지 않아야 한다."""
        msb = _msb(bos_buffer_pct=0.0)

        pivots = [
            _fake_pivot(0, 365.0, PivotType.HIGH),
            _fake_pivot(5, 355.0, PivotType.LOW),
        ]

        for i in range(15):
            # 355 ~ 365 사이 유지
            h = 363.0; l = 357.0
            s = msb.update(high=h, low=l, close=360.0,
                           bar_time=_t(i), pivot_points=pivots)
            assert s.bos_signal == BOSType.NONE, \
                f"bar={i} 돌파 없음에도 BOS 발생: {s.bos_signal}"

    def test_bos_buffer_prevents_premature_signal(self) -> None:
        """bos_buffer_pct 설정 시 정확히 돌파하지 않으면 BOS 미발생."""
        msb = _msb(bos_buffer_pct=0.5)   # 0.5% 버퍼

        pivots = [
            _fake_pivot(0, 365.0, PivotType.HIGH),
            _fake_pivot(5, 355.0, PivotType.LOW),
        ]

        # 365.0 × (1 + 0.005) = 366.825 미만 → BOS 미발생
        for i in range(10):
            h = 365.5  # 버퍼 미달
            s = msb.update(high=h, low=360.0, close=362.0,
                           bar_time=_t(i), pivot_points=pivots)
            assert s.bos_signal == BOSType.NONE, \
                f"버퍼 미달인데 BOS 발생: {s.bos_signal}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHoCH 탐지
# ─────────────────────────────────────────────────────────────────────────────

class TestCHoCHDetection:

    def _setup_downtrend(self) -> MarketStructureBreak:
        """하락 구조를 구성한 MSB 반환."""
        msb = _msb(bos_buffer_pct=0.0, choch_enabled=True,
                   structure_lookback_pivots=4)

        # 하락 구조: LH + LL 패턴
        pivots = [
            _fake_pivot(0,  370.0, PivotType.HIGH, "09:00"),
            _fake_pivot(5,  358.0, PivotType.LOW,  "09:05"),
            _fake_pivot(10, 367.0, PivotType.HIGH, "09:10"),
            _fake_pivot(15, 354.0, PivotType.LOW,  "09:15"),
            _fake_pivot(20, 364.0, PivotType.HIGH, "09:20"),
            _fake_pivot(25, 350.0, PivotType.LOW,  "09:25"),
        ]

        # 초기화 (flat)
        for i in range(26):
            msb.update(high=360.0, low=358.0, close=359.0,
                       bar_time=_t(i), pivot_points=pivots)
        return msb

    def test_choch_up_in_downtrend(self) -> None:
        """하락 구조 중 swing high 돌파 → CHoCH_UP."""
        msb = self._setup_downtrend()

        # 구조 확인 후 실제 ref_sh (swing_highs[-2]) 사용
        ref_sh = msb._swing_highs[-2][1] if len(msb._swing_highs) >= 2 else 364.0
        signals = []
        for i in range(26, 42):
            h = ref_sh + 2.0 if i == 32 else 361.0
            s = msb.update(high=h, low=h - 2.0, close=h - 1.0,
                           bar_time=_t(i))
            signals.append(s.bos_signal)

        assert BOSType.CHOCH_UP in signals, \
            f"CHoCH_UP 미발생 (하락구조 상향돌파): {[s.value for s in signals]}"

    def test_choch_disabled(self) -> None:
        """choch_enabled=False 시 CHoCH 대신 BOS 로 처리."""
        msb = _msb(bos_buffer_pct=0.0, choch_enabled=False,
                   structure_lookback_pivots=4)

        pivots = [
            _fake_pivot(0,  370.0, PivotType.HIGH),
            _fake_pivot(5,  358.0, PivotType.LOW),
            _fake_pivot(10, 367.0, PivotType.HIGH),
            _fake_pivot(15, 354.0, PivotType.LOW),
        ]

        signals = []
        for i in range(20):
            h = 375.0 if i == 18 else 360.0
            s = msb.update(high=h, low=h - 2.0, close=h - 1.0,
                           bar_time=_t(i), pivot_points=pivots)
            signals.append(s.bos_signal)

        # CHoCH 는 없어야 하고, BOS_UP 은 있어야 함
        assert BOSType.CHOCH_UP not in signals
        assert BOSType.BOS_UP in signals


# ─────────────────────────────────────────────────────────────────────────────
# 3. 내부 스윙 탐지 (독립 모드)
# ─────────────────────────────────────────────────────────────────────────────

class TestInternalSwingDetection:

    def test_internal_swing_builds_history(self) -> None:
        """pivot_points 없이 update() 만으로 스윙 목록이 구성되어야 한다."""
        msb = _msb(swing_lookback=2, min_swing_gap_bars=1)

        # 고점 패턴 삽입
        prices = [360, 361, 365, 362, 360,   # 고점 프랙탈: bar2=365
                  359, 355, 358, 360, 361]    # 저점 프랙탈: bar6=355

        for i, p in enumerate(prices):
            msb.update(high=float(p) + 0.5, low=float(p) - 0.5,
                       close=float(p), bar_time=_t(i))

        # 내부 스윙이 1개 이상 탐지되어야 함
        assert len(msb._swing_highs) > 0 or len(msb._swing_lows) > 0, \
            "내부 스윙 미탐지"

    def test_no_external_pivots_no_error(self) -> None:
        """pivot_points=None 으로 60봉 연속 처리해도 예외 없어야 한다."""
        msb = MarketStructureBreak()
        import random; random.seed(7)
        for i in range(60):
            p = 360.0 + random.uniform(-3, 3)
            msb.update(high=p + 0.5, low=p - 0.5, close=p, bar_time=_t(i))
        assert msb.state is not None


# ─────────────────────────────────────────────────────────────────────────────
# 4. 구조 분석
# ─────────────────────────────────────────────────────────────────────────────

class TestStructureAnalysis:

    def test_uptrend_structure(self) -> None:
        """HH + HL 패턴 → UPTREND."""
        msb = _msb(structure_lookback_pivots=6)

        # HH + HL 패턴 피봇
        pivots = [
            _fake_pivot(0,  355.0, PivotType.LOW),
            _fake_pivot(5,  362.0, PivotType.HIGH),
            _fake_pivot(10, 358.0, PivotType.LOW),
            _fake_pivot(15, 367.0, PivotType.HIGH),
            _fake_pivot(20, 363.0, PivotType.LOW),
            _fake_pivot(25, 372.0, PivotType.HIGH),
        ]

        for i in range(30):
            s = msb.update(high=360.0, low=358.0, close=359.0,
                           bar_time=_t(i), pivot_points=pivots)

        assert msb.state.structure == StructureType.UPTREND, \
            f"HH+HL 패턴인데 구조={msb.state.structure}"

    def test_downtrend_structure(self) -> None:
        """LH + LL 패턴 → DOWNTREND."""
        msb = _msb(structure_lookback_pivots=6)

        pivots = [
            _fake_pivot(0,  372.0, PivotType.HIGH),
            _fake_pivot(5,  363.0, PivotType.LOW),
            _fake_pivot(10, 367.0, PivotType.HIGH),
            _fake_pivot(15, 358.0, PivotType.LOW),
            _fake_pivot(20, 362.0, PivotType.HIGH),
            _fake_pivot(25, 355.0, PivotType.LOW),
        ]

        for i in range(30):
            s = msb.update(high=360.0, low=358.0, close=359.0,
                           bar_time=_t(i), pivot_points=pivots)

        assert msb.state.structure == StructureType.DOWNTREND, \
            f"LH+LL 패턴인데 구조={msb.state.structure}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. MSB Score / Features
# ─────────────────────────────────────────────────────────────────────────────

class TestMSBScoreAndFeatures:

    def test_msb_score_in_range(self) -> None:
        """msb_score 는 항상 [0, 1] 이어야 한다."""
        msb = MarketStructureBreak()
        pivots = [
            _fake_pivot(0, 365.0, PivotType.HIGH),
            _fake_pivot(5, 355.0, PivotType.LOW),
        ]
        import random; random.seed(3)
        for i in range(50):
            p = 360.0 + random.uniform(-5, 5)
            s = msb.update(high=p + 0.5, low=p - 0.5, close=p,
                           bar_time=_t(i), pivot_points=pivots)
            assert 0.0 <= s.msb_score <= 1.0, \
                f"bar={i} msb_score={s.msb_score} 범위 초과"

    def test_transformer_features_keys(self) -> None:
        """get_transformer_features 필수 키 존재 + 유한수."""
        required = {
            "msb_bos_signal", "msb_structure", "msb_hh_ratio",
            "msb_ll_ratio", "msb_sh_dist", "msb_sl_dist",
            "msb_score", "msb_choch",
        }
        msb = MarketStructureBreak()
        for i in range(20):
            msb.update(high=360.5, low=359.5, close=360.0, bar_time=_t(i))

        feats = msb.get_transformer_features(360.0)
        missing = required - feats.keys()
        assert not missing, f"누락 키: {missing}"
        for k, v in feats.items():
            assert math.isfinite(v), f"{k}={v} 비유한수"

    def test_reset_clears_state(self) -> None:
        """reset() 후 스윙 목록과 구조가 초기화되어야 한다."""
        msb = MarketStructureBreak()
        for i in range(20):
            msb.update(high=365.0, low=355.0, close=360.0, bar_time=_t(i))

        msb.reset()
        assert msb._swing_highs == []
        assert msb._swing_lows  == []
        assert msb.state.structure == StructureType.UNKNOWN
        assert msb.state.bos_signal == BOSType.NONE


# ─────────────────────────────────────────────────────────────────────────────
# 6. OIStructureGate
# ─────────────────────────────────────────────────────────────────────────────

class TestOIStructureGate:

    def _bos_state(self, bos: BOSType, base_score: float = 0.4) -> MSBState:
        s = MSBState()
        s.bos_signal = bos
        s.msb_score  = base_score
        s.structure  = StructureType.DOWNTREND
        return s

    def test_no_oi_returns_base_score(self) -> None:
        """OI levels = None → msb_score 그대로 반환."""
        gate = OIStructureGate()
        state = self._bos_state(BOSType.CHOCH_UP, 0.5)
        result = gate.score(state, close=360.0, oi_levels=None)
        assert abs(result - 0.5) < 1e-6

    def test_choch_near_oi_boosted(self) -> None:
        """CHoCH + OI peak 근접 → 점수 부스트."""
        cfg   = OIStructureConfig(oi_proximity_pct=0.5, choch_oi_boost=1.5)
        gate  = OIStructureGate(cfg)

        # close=360, call_peak=361 (근접)
        state = self._bos_state(BOSType.CHOCH_UP, 0.5)
        oi    = {"call_oi_peak": 361.0, "put_oi_peak": 350.0}
        result = gate.score(state, close=360.0, oi_levels=oi)

        assert result > 0.5, f"CHoCH+OI 근접인데 부스트 없음: {result}"

    def test_bos_far_from_oi_no_boost(self) -> None:
        """BOS 발생하지만 OI peak 에서 멀면 부스트 없어야 한다."""
        gate  = OIStructureGate(OIStructureConfig(oi_proximity_pct=0.3))

        state = self._bos_state(BOSType.BOS_UP, 0.4)
        # close=360, call_peak=370 (거리 2.7% → 0.3% 초과)
        oi    = {"call_oi_peak": 370.0, "put_oi_peak": 350.0}
        result = gate.score(state, close=360.0, oi_levels=oi)

        assert abs(result - 0.4) < 1e-6, f"OI 거리 먼데 부스트 발생: {result}"

    def test_oi_gate_features_keys(self) -> None:
        """get_transformer_features 키 집합 검증."""
        gate  = OIStructureGate()
        state = self._bos_state(BOSType.BOS_DOWN, 0.3)
        oi    = {"call_oi_peak": 365.0, "put_oi_peak": 355.0}
        feats = gate.get_transformer_features(state, close=360.0, oi_levels=oi)

        required = {
            "oi_msb_score", "oi_near_call", "oi_near_put",
            "oi_call_dist", "oi_put_dist", "oi_bos_boosted",
        }
        missing = required - feats.keys()
        assert not missing, f"누락 키: {missing}"
        for k, v in feats.items():
            assert math.isfinite(v), f"{k}={v} 비유한수"

    def test_score_clamped_to_one(self) -> None:
        """부스트 후 결과가 1.0 을 초과하지 않아야 한다."""
        cfg   = OIStructureConfig(oi_proximity_pct=1.0, choch_oi_boost=3.0)
        gate  = OIStructureGate(cfg)
        state = self._bos_state(BOSType.CHOCH_UP, 0.9)
        oi    = {"call_oi_peak": 360.5, "put_oi_peak": 350.0}
        result = gate.score(state, close=360.0, oi_levels=oi)
        assert result <= 1.0, f"점수 1 초과: {result}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. ATRAdaptivePivot + MSB 통합
# ─────────────────────────────────────────────────────────────────────────────

class TestPivotMSBIntegration:

    def test_pivot_points_injected_to_msb(self) -> None:
        """ATRAdaptivePivot 확정 피봇이 MSB 스윙 목록에 반영되어야 한다."""
        pivot = ATRAdaptivePivot(ATRAdaptivePivotConfig(
            warmup_bars=15, atr_period=5, confirmation_bars=1,
        ))
        msb = MarketStructureBreak()

        for i in range(50):
            p = 360.0 + (i % 10 - 5) * 0.8
            t = _t(i)
            pivot.update(high=p + 0.5, low=p - 0.5, close=p, bar_time=t)
            msb.update(high=p + 0.5, low=p - 0.5, close=p,
                       bar_time=t, pivot_points=pivot.confirmed_pivots)

        # pivot 이 1개 이상 확정되면 MSB 스윙 목록도 채워져야 함
        if pivot.confirmed_pivots:
            assert len(msb._swing_highs) > 0 or len(msb._swing_lows) > 0, \
                "피봇 주입 후 MSB 스윙 목록 비어있음"

    def test_combined_score_in_range(self) -> None:
        """ATR pivot_score + msb_score 가중 합산 결과가 [0, 1] 이어야 한다."""
        pivot = ATRAdaptivePivot(ATRAdaptivePivotConfig(warmup_bars=10))
        msb   = MarketStructureBreak()
        gate  = OIStructureGate()

        import random; random.seed(11)
        for i in range(60):
            p = 360.0 + random.uniform(-4, 4)
            t = _t(i)
            ps = pivot.update(high=p + 0.5, low=p - 0.5, close=p, bar_time=t)
            ms = msb.update(high=p + 0.5, low=p - 0.5, close=p,
                            bar_time=t, pivot_points=pivot.confirmed_pivots)
            oi_score = gate.score(ms, close=p, oi_levels=None)

            total = ps.pivot_score * 0.4 + ms.msb_score * 0.4 + oi_score * 0.2
            assert 0.0 <= total <= 1.0, \
                f"bar={i} total_score={total:.4f} 범위 초과"

    def test_no_exception_full_run(self) -> None:
        """60봉 전체 처리 중 예외 없어야 한다."""
        pivot = ATRAdaptivePivot()
        msb   = MarketStructureBreak()
        oi_gate = OIStructureGate()
        oi_levels = {"call_oi_peak": 365.0, "put_oi_peak": 355.0}

        import random; random.seed(99)
        for i in range(60):
            p = 360.0 + random.uniform(-5, 5)
            t = _t(i)
            ps = pivot.update(high=p + 0.5, low=p - 0.5, close=p, bar_time=t)
            ms = msb.update(high=p + 0.5, low=p - 0.5, close=p,
                            bar_time=t, pivot_points=pivot.confirmed_pivots)
            oi_gate.score(ms, close=p, oi_levels=oi_levels)

        assert True  # 예외 없이 완료
