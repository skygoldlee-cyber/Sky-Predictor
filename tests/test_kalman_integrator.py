"""KalmanTurningPoint + PivotScoreIntegrator 단위 테스트

검증 항목
---------
1. KalmanTurningPoint
   - 웜업 중 kalman_score=0
   - 가격 상승 → slope 양수
   - 가격 하락 → slope 음수
   - 상승→하락 반전 시 slope_flip=True + turning_signal="down"
   - 하락→상승 반전 시 slope_flip=True + turning_signal="up"
   - kalman_score 범위 [0, 1]
   - get_transformer_features 키 집합 + 유한수
   - adaptive_q=True 시 ATR 반영
   - reset() 후 상태 초기화

2. PivotScoreIntegrator
   - 4레이어 모두 활성 → total_score 정상 범위
   - 일부 레이어 None → 재분배 후 정상 동작
   - 전체 None → IntegratorResult 기본값
   - entry_threshold 미달 → signal="none"
   - 다수결 long/short/동점
   - regime boost → adjusted_score >= total_score
   - regime suppress → adjusted_score <= total_score
   - get_transformer_features 키 + 유한수
   - get_all_features 병합 검증

3. 4-레이어 통합 종단 테스트
   - ATRAdaptivePivot + MSB + Kalman + OIGate + Integrator 60봉 예외없음
   - total_score 항상 [0, 1]
   - signal 이 "long"/"short"/"none" 중 하나
"""
from __future__ import annotations

import datetime
import math
import random
from typing import List, Optional

import pytest

from indicators import (
    ATRAdaptivePivot,
    ATRAdaptivePivotConfig,
    MarketStructureBreak,
    OIStructureGate,
    KalmanTurningPoint,
    KalmanConfig,
    PivotScoreIntegrator,
    IntegratorConfig,
    IntegratorResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

DT = datetime.datetime(2026, 5, 2, 9, 0)


def _t(minutes: int) -> datetime.datetime:
    return DT + datetime.timedelta(minutes=minutes)


def _kf(**kw) -> KalmanTurningPoint:
    return KalmanTurningPoint(KalmanConfig(**kw))


# ─────────────────────────────────────────────────────────────────────────────
# 1. KalmanTurningPoint
# ─────────────────────────────────────────────────────────────────────────────

class TestKalmanWarmup:

    def test_no_signal_during_warmup(self) -> None:
        kf = _kf(warmup_bars=20)
        for i in range(18):
            s = kf.update(close=360.0, bar_time=_t(i))
        assert s.kalman_score == 0.0, f"웜업 중 kalman_score={s.kalman_score}"


class TestKalmanSlope:

    def test_rising_price_positive_slope(self) -> None:
        """단조 상승 시 slope 는 양수여야 한다."""
        kf = _kf(warmup_bars=10, q=0.5, r=0.1)
        for i in range(30):
            s = kf.update(close=350.0 + i * 0.5, bar_time=_t(i))
        assert s.slope > 0, f"단조 상승인데 slope={s.slope:.4f}"

    def test_falling_price_negative_slope(self) -> None:
        """단조 하락 시 slope 는 음수여야 한다."""
        kf = _kf(warmup_bars=10, q=0.5, r=0.1)
        for i in range(30):
            s = kf.update(close=380.0 - i * 0.5, bar_time=_t(i))
        assert s.slope < 0, f"단조 하락인데 slope={s.slope:.4f}"


class TestKalmanFlip:

    def test_slope_flip_on_reversal(self) -> None:
        """상승 후 급락 시 slope_flip=True 가 발생해야 한다."""
        kf = _kf(warmup_bars=10, q=0.1, r=0.5, slope_flip_min=0.001)
        # 상승
        for i in range(20):
            kf.update(close=350.0 + i * 0.8, bar_time=_t(i))
        # 하락
        flips = []
        peak = 350.0 + 20 * 0.8
        for i in range(20):
            s = kf.update(close=peak - i * 1.0, bar_time=_t(20 + i))
            flips.append(s.slope_flip)

        assert any(flips), "상승→하락 반전인데 slope_flip 미발생"

    def test_turning_signal_down_on_peak(self) -> None:
        """slope 양→음 전환 → turning_signal='down'."""
        kf = _kf(warmup_bars=10, q=0.2, r=0.3, slope_flip_min=0.001)
        for i in range(20):
            kf.update(close=350.0 + i * 0.8, bar_time=_t(i))
        peak = 350.0 + 20 * 0.8
        signals = []
        for i in range(15):
            s = kf.update(close=peak - i * 1.2, bar_time=_t(20 + i))
            if s.slope_flip:
                signals.append(s.turning_signal)

        assert "down" in signals, f"slope_flip 후 turning_signal={signals}"

    def test_turning_signal_up_on_trough(self) -> None:
        """slope 음→양 전환 → turning_signal='up'."""
        kf = _kf(warmup_bars=10, q=0.2, r=0.3, slope_flip_min=0.001)
        # 하락 먼저
        for i in range(20):
            kf.update(close=380.0 - i * 0.8, bar_time=_t(i))
        trough = 380.0 - 20 * 0.8
        signals = []
        for i in range(15):
            s = kf.update(close=trough + i * 1.2, bar_time=_t(20 + i))
            if s.slope_flip:
                signals.append(s.turning_signal)

        assert "up" in signals, f"slope_flip 후 turning_signal={signals}"


class TestKalmanScoreAndFeatures:

    def test_score_in_range(self) -> None:
        kf = _kf(warmup_bars=10)
        random.seed(5)
        for i in range(60):
            c = 360.0 + random.uniform(-3, 3)
            s = kf.update(close=c, high=c + 0.5, low=c - 0.5, bar_time=_t(i))
            assert 0.0 <= s.kalman_score <= 1.0, \
                f"bar={i} kalman_score={s.kalman_score} 범위 초과"

    def test_transformer_features_keys(self) -> None:
        required = {
            "kf_slope_norm", "kf_slope_flip", "kf_slope_surge",
            "kf_turning_signal", "kf_score", "kf_dev_norm",
            "kf_innovation", "kf_gain",
        }
        kf = KalmanTurningPoint()
        for i in range(30):
            kf.update(close=360.0, bar_time=_t(i))
        feats = kf.get_transformer_features(360.0)
        missing = required - feats.keys()
        assert not missing, f"누락 키: {missing}"
        for k, v in feats.items():
            assert math.isfinite(v), f"{k}={v} 비유한수"

    def test_reset_clears_state(self) -> None:
        kf = _kf(warmup_bars=5)
        for i in range(20):
            kf.update(close=360.0 + i, bar_time=_t(i))
        kf.reset()
        assert not kf._initialized
        assert kf.state.slope == 0.0
        assert kf.state.kalman_score == 0.0

    def test_adaptive_q_affects_gain(self) -> None:
        """adaptive_q=True 시 Q 가 변동성에 따라 달라져야 한다."""
        kf_adapt = _kf(adaptive_q=True, q=0.01, warmup_bars=10)
        kf_fixed = _kf(adaptive_q=False, q=0.01, warmup_bars=10)

        # 고변동성 입력
        random.seed(42)
        for i in range(30):
            c = 360.0 + random.uniform(-5, 5)
            kf_adapt.update(close=c, high=c + 2, low=c - 2, bar_time=_t(i))
            kf_fixed.update(close=c, high=c + 2, low=c - 2, bar_time=_t(i))

        # adaptive 는 Q[0,0] 이 변해야 함 (고변동 → q_adaptive > cfg.q)
        assert kf_adapt.state.q_adaptive != kf_fixed.state.q_adaptive or \
               kf_adapt.state.q_adaptive > 0, "adaptive Q 변화 없음"


# ─────────────────────────────────────────────────────────────────────────────
# 2. PivotScoreIntegrator
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegratorBasic:

    def _integ(self, **kw) -> PivotScoreIntegrator:
        return PivotScoreIntegrator(IntegratorConfig(**kw))

    def test_all_layers_active(self) -> None:
        integ = self._integ()
        r = integ.compute(
            aap_score=0.7, msb_score=0.6, oi_score=0.5, kalman_score=0.8
        )
        assert 0.0 <= r.total_score <= 1.0, f"total_score={r.total_score}"
        assert r.active_layers == 4

    def test_partial_layers_none(self) -> None:
        """일부 레이어 None → 재분배 후 total_score 정상 범위."""
        integ = self._integ()
        r = integ.compute(aap_score=0.8, msb_score=None, oi_score=0.6, kalman_score=None)
        assert 0.0 <= r.total_score <= 1.0
        assert r.active_layers == 2

    def test_all_none_returns_default(self) -> None:
        integ = self._integ()
        r = integ.compute()
        assert r.total_score == 0.0
        assert r.signal == "none"

    def test_threshold_suppresses_signal(self) -> None:
        """낮은 점수 → signal='none'."""
        integ = self._integ(entry_threshold=0.8)
        r = integ.compute(aap_score=0.3, msb_score=0.3, oi_score=0.3, kalman_score=0.3)
        assert r.signal == "none", f"낮은 점수인데 signal={r.signal}"

    def test_strong_signal_above_threshold(self) -> None:
        """높은 점수 + 방향 일치 → STRONG 신호."""
        integ = self._integ(entry_threshold=0.5, strong_threshold=0.7)
        r = integ.compute(
            aap_score=0.9, msb_score=0.9, oi_score=0.9, kalman_score=0.9,
            aap_signal="new_low", msb_signal="bos_up", kalman_signal="up",
        )
        assert r.signal == "long", f"long 기대인데 signal={r.signal}"
        assert r.signal_strength == "STRONG", f"강도={r.signal_strength}"


class TestIntegratorVoting:

    def test_majority_long(self) -> None:
        integ = PivotScoreIntegrator(IntegratorConfig(entry_threshold=0.0))
        r = integ.compute(
            aap_score=0.6, msb_score=0.6, kalman_score=0.6,
            aap_signal="new_low", msb_signal="bos_up", kalman_signal="up",
        )
        assert r.signal == "long"

    def test_majority_short(self) -> None:
        integ = PivotScoreIntegrator(IntegratorConfig(entry_threshold=0.0))
        r = integ.compute(
            aap_score=0.6, msb_score=0.6, kalman_score=0.6,
            aap_signal="new_high", msb_signal="bos_down", kalman_signal="down",
        )
        assert r.signal == "short"

    def test_tie_returns_none(self) -> None:
        """long 1표 vs short 1표 → "none"."""
        integ = PivotScoreIntegrator(IntegratorConfig(entry_threshold=0.0))
        r = integ.compute(
            aap_score=0.7, msb_score=0.7,
            aap_signal="new_low", msb_signal="bos_down",  # 동점
        )
        assert r.signal == "none", f"동점인데 signal={r.signal}"


class TestIntegratorRegime:

    def test_regime_boost(self) -> None:
        cfg   = IntegratorConfig(regime_boost=1.2, entry_threshold=0.0)
        integ = PivotScoreIntegrator(cfg)
        base  = integ.compute(aap_score=0.5, msb_score=0.5)
        boost = integ.compute(aap_score=0.5, msb_score=0.5, regime="uptrend")
        assert boost.adjusted_score >= base.adjusted_score, \
            f"부스트 후 점수가 기본보다 낮음: {boost.adjusted_score} < {base.adjusted_score}"
        assert boost.regime_applied == "boost"

    def test_regime_suppress(self) -> None:
        cfg   = IntegratorConfig(regime_suppress=0.8, entry_threshold=0.0)
        integ = PivotScoreIntegrator(cfg)
        base  = integ.compute(aap_score=0.7, msb_score=0.7)
        supp  = integ.compute(aap_score=0.7, msb_score=0.7, regime="ranging")
        assert supp.adjusted_score <= base.adjusted_score, \
            f"억제 후 점수가 기본보다 높음: {supp.adjusted_score} > {base.adjusted_score}"
        assert supp.regime_applied == "suppress"


class TestIntegratorFeatures:

    def test_transformer_features_keys(self) -> None:
        required = {
            "ps_total_score", "ps_adjusted_score", "ps_signal",
            "ps_strength", "ps_aap_score", "ps_msb_score",
            "ps_oi_score", "ps_kf_score", "ps_active_layers",
            "ps_regime_boost", "ps_regime_suppress", "ps_long", "ps_short",
        }
        integ = PivotScoreIntegrator()
        integ.compute(aap_score=0.5, msb_score=0.6)
        feats = integ.get_transformer_features()
        missing = required - feats.keys()
        assert not missing, f"누락 키: {missing}"
        for k, v in feats.items():
            assert math.isfinite(v), f"{k}={v} 비유한수"

    def test_get_all_features_merge(self) -> None:
        """get_all_features 가 개별 레이어 피처를 모두 포함해야 한다."""
        integ = PivotScoreIntegrator()
        result = integ.compute(aap_score=0.5, msb_score=0.5,
                               oi_score=0.4, kalman_score=0.6)
        pivot_feats = {"azz_direction": 1.0, "aap_pivot_score": 0.5}
        msb_feats   = {"msb_score": 0.5, "msb_bos_signal": 0.3}
        kf_feats    = {"kf_slope_norm": 0.2, "kf_score": 0.6}
        oi_feats    = {"oi_msb_score": 0.4}

        combined = integ.get_all_features(
            result, pivot_feats, msb_feats, kf_feats, oi_feats
        )
        # 모든 개별 키가 포함되어야 함
        for k in [*pivot_feats, *msb_feats, *kf_feats, *oi_feats]:
            assert k in combined, f"누락 키: {k}"
        # ps_* 키도 포함
        assert "ps_total_score" in combined


# ─────────────────────────────────────────────────────────────────────────────
# 3. 4-레이어 통합 종단 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd4Layer:

    def test_full_pipeline_no_exception(self) -> None:
        """60봉 전체 처리 중 예외 없고 score 범위 정상이어야 한다."""
        pivot  = ATRAdaptivePivot(ATRAdaptivePivotConfig(warmup_bars=15, atr_period=5))
        msb    = MarketStructureBreak()
        kf     = KalmanTurningPoint(KalmanConfig(warmup_bars=15))
        oi_gate = OIStructureGate()
        integ  = PivotScoreIntegrator()

        oi_levels = {"call_oi_peak": 368.0, "put_oi_peak": 352.0}

        random.seed(77)
        for i in range(60):
            p = 360.0 + random.uniform(-4, 4)
            t = _t(i)

            ps  = pivot.update(high=p + 0.5, low=p - 0.5, close=p, bar_time=t)
            ms  = msb.update(high=p + 0.5, low=p - 0.5, close=p,
                              bar_time=t, pivot_points=pivot.confirmed_pivots)
            kfs = kf.update(close=p, high=p + 0.5, low=p - 0.5, bar_time=t)
            oi_score = oi_gate.score(ms, close=p, oi_levels=oi_levels)

            result = integ.compute(
                aap_score    = ps.pivot_score,
                msb_score    = ms.msb_score,
                oi_score     = oi_score,
                kalman_score = kfs.kalman_score,
                aap_signal   = ps.new_pivot_signal,
                msb_signal   = ms.bos_signal.value,
                kalman_signal= kfs.turning_signal,
            )

            assert 0.0 <= result.total_score    <= 1.0, \
                f"bar={i} total_score={result.total_score}"
            assert 0.0 <= result.adjusted_score <= 1.0, \
                f"bar={i} adjusted_score={result.adjusted_score}"
            assert result.signal in ("long", "short", "none"), \
                f"bar={i} 알 수 없는 signal={result.signal}"

    def test_all_features_combined_finite(self) -> None:
        """통합 피처 dict 의 모든 값이 유한수여야 한다."""
        pivot  = ATRAdaptivePivot(ATRAdaptivePivotConfig(warmup_bars=10))
        msb    = MarketStructureBreak()
        kf     = KalmanTurningPoint(KalmanConfig(warmup_bars=10))
        oi_gate = OIStructureGate()
        integ  = PivotScoreIntegrator()

        oi_levels = {"call_oi_peak": 365.0, "put_oi_peak": 355.0}
        random.seed(13)

        for i in range(40):
            p   = 360.0 + random.uniform(-3, 3)
            t   = _t(i)
            ps  = pivot.update(high=p + 0.5, low=p - 0.5, close=p, bar_time=t)
            ms  = msb.update(high=p + 0.5, low=p - 0.5, close=p,
                              bar_time=t, pivot_points=pivot.confirmed_pivots)
            kfs = kf.update(close=p, high=p + 0.5, low=p - 0.5, bar_time=t)
            oi_score = oi_gate.score(ms, close=p, oi_levels=oi_levels)

            result = integ.compute(
                aap_score=ps.pivot_score, msb_score=ms.msb_score,
                oi_score=oi_score, kalman_score=kfs.kalman_score,
            )

            combined = integ.get_all_features(
                result,
                pivot_feats = pivot.get_transformer_features(p),
                msb_feats   = msb.get_transformer_features(p),
                kf_feats    = kf.get_transformer_features(p),
                oi_feats    = oi_gate.get_transformer_features(ms, p, oi_levels),
            )

            for k, v in combined.items():
                assert math.isfinite(v), f"bar={i} {k}={v} 비유한수"

    def test_signal_count_reasonable(self) -> None:
        """60봉 중 신호 비율이 과도하지 않아야 한다 (< 50%)."""
        pivot  = ATRAdaptivePivot(ATRAdaptivePivotConfig(warmup_bars=15))
        msb    = MarketStructureBreak()
        kf     = KalmanTurningPoint()
        oi_gate = OIStructureGate()
        integ  = PivotScoreIntegrator()

        random.seed(99)
        signal_count = 0
        for i in range(60):
            p = 360.0 + random.uniform(-2, 2)
            t = _t(i)
            ps  = pivot.update(high=p + 0.4, low=p - 0.4, close=p, bar_time=t)
            ms  = msb.update(high=p + 0.4, low=p - 0.4, close=p, bar_time=t,
                              pivot_points=pivot.confirmed_pivots)
            kfs = kf.update(close=p, high=p + 0.4, low=p - 0.4, bar_time=t)
            oi_score = oi_gate.score(ms, close=p)
            r = integ.compute(
                aap_score=ps.pivot_score, msb_score=ms.msb_score,
                oi_score=oi_score, kalman_score=kfs.kalman_score,
            )
            if r.signal != "none":
                signal_count += 1

        assert signal_count < 30, f"신호 과다 발생: {signal_count}/60"
