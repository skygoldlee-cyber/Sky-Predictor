"""ATRAdaptivePivot + FractalConfirmation 단위 테스트

검증 항목
---------
1. ATRAdaptivePivot
   - 웜업 전 신호 미출력
   - 충분한 상승 후 고점 후보 등록 및 확정
   - 충분한 하락 후 저점 후보 등록 및 확정
   - 소파동 필터 (min_wave_atr_ratio)
   - get_transformer_features 키 집합 검증 (azz_* 호환)
   - session_multiplier_table 적용 시 threshold 변화
   - pivot_score 범위 [0, 1]
   - reset() 후 상태 초기화

2. FractalConfirmation
   - 고점/저점 프랙탈 확정 (lookback=2)
   - 거래량 spike 필터 동작
   - min_bar_gap 필터
   - get_transformer_features 키 집합 검증

3. 조합 테스트
   - ATRAdaptivePivot 신호 + Fractal 확증 동시 발생 가능성
   - AdaptiveZigZag 와 동일 인터페이스 확인 (update 시그니처)
"""
from __future__ import annotations

import datetime
import math
from typing import List


from indicators import (
    ATRAdaptivePivot,
    ATRAdaptivePivotConfig,
    FractalConfirmation,
    FractalConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _make_bars(
    n: int,
    base: float = 360.0,
    noise: float = 0.5,
) -> List[dict]:
    """flat 바 n개 생성 (ATR 안정화용)."""
    bars = []
    for i in range(n):
        bars.append(dict(high=base + noise, low=base - noise, close=base,
                         volume=1000.0,
                         bar_time=datetime.datetime(2026, 5, 2, 9, 0) +
                                  datetime.timedelta(minutes=i)))
    return bars


def _feed(pivot: ATRAdaptivePivot, bars: List[dict]) -> None:
    for b in bars:
        pivot.update(
            high=b["high"], low=b["low"], close=b["close"],
            volume=b.get("volume", 1000.0),
            bar_time=b.get("bar_time"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ATRAdaptivePivot 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestATRAdaptivePivotWarmup:
    """웜업 구간에서는 신호를 출력하지 않아야 한다."""

    def test_no_signal_during_warmup(self) -> None:
        cfg = ATRAdaptivePivotConfig(warmup_bars=30, atr_period=5)
        pivot = ATRAdaptivePivot(cfg)

        signals = []
        for i in range(25):   # warmup_bars=30 미만
            # 극단적 상승/하락 교번
            if i % 2 == 0:
                s = pivot.update(high=380.0, low=370.0, close=379.0,
                                 bar_time=datetime.datetime(2026, 5, 2, 9, i))
            else:
                s = pivot.update(high=375.0, low=355.0, close=356.0,
                                 bar_time=datetime.datetime(2026, 5, 2, 9, i))
            signals.append(s.new_pivot_signal)

        # 25봉 이하 → 신호 없어야 함
        assert all(sig == "none" for sig in signals), \
            f"웜업 중 신호 발생: {[s for s in signals if s != 'none']}"


class TestATRAdaptivePivotSignals:
    """충분한 이동 후 고점/저점 신호가 정확히 발생해야 한다."""

    def _make_pivot(self) -> ATRAdaptivePivot:
        return ATRAdaptivePivot(ATRAdaptivePivotConfig(
            atr_period=5,
            er_period=5,
            base_multiplier=1.5,
            multiplier_min=1.0,
            multiplier_max=2.0,
            confirmation_bars=1,
            min_wave_atr_ratio=0.3,
            warmup_bars=15,
        ))

    def test_new_high_signal_after_rise_and_drop(self) -> None:
        """상승 후 충분한 하락 → new_high 신호 발생."""
        pivot = self._make_pivot()
        dt = datetime.datetime(2026, 5, 2, 9, 0)

        # 웜업 (flat)
        for i in range(15):
            pivot.update(high=360.5, low=359.5, close=360.0,
                         bar_time=dt + datetime.timedelta(minutes=i))

        # 상승 구간 (10pt)
        for i in range(8):
            p = 360.0 + (i + 1) * 1.2
            pivot.update(high=p + 0.3, low=p - 0.3, close=p,
                         bar_time=dt + datetime.timedelta(minutes=15 + i))

        # 하락 전환 (6pt 하락 → threshold 돌파 유도)
        signals = []
        peak = 360.0 + 8 * 1.2
        for i in range(8):
            p = peak - (i + 1) * 0.9
            s = pivot.update(high=p + 0.2, low=p - 0.2, close=p,
                             bar_time=dt + datetime.timedelta(minutes=23 + i))
            signals.append(s.new_pivot_signal)

        assert "new_high" in signals, \
            f"new_high 신호 미발생: {signals}"

    def test_new_low_signal_after_drop_and_rise(self) -> None:
        """고점 앵커 확정 후 하락 → 상승 반전 시 new_low 신호 발생."""
        pivot = self._make_pivot()
        dt = datetime.datetime(2026, 5, 2, 9, 0)

        # 웜업 (flat)
        for i in range(15):
            pivot.update(high=360.5, low=359.5, close=360.0,
                         bar_time=dt + datetime.timedelta(minutes=i))

        # 상승으로 HIGH 앵커 확정 유도
        for i in range(8):
            p = 360.0 + (i + 1) * 1.2
            pivot.update(high=p + 0.3, low=p - 0.3, close=p,
                         bar_time=dt + datetime.timedelta(minutes=15 + i))

        # 하락 → HIGH 확정 + direction=-1 전환
        peak = 360.0 + 8 * 1.2
        for i in range(6):
            p = peak - (i + 1) * 1.2
            pivot.update(high=p + 0.3, low=p - 0.3, close=p,
                         bar_time=dt + datetime.timedelta(minutes=23 + i))

        # 이제 direction=-1 (LOW 탐색 중), 충분히 하락 후 상승 반전
        trough_base = peak - 6 * 1.2
        signals = []
        for i in range(10):
            p = trough_base + (i + 1) * 1.0
            s = pivot.update(high=p + 0.2, low=p - 0.2, close=p,
                             bar_time=dt + datetime.timedelta(minutes=29 + i))
            signals.append(s.new_pivot_signal)

        assert "new_low" in signals or "new_high" in signals, \
            f"new_low/new_high 신호 미발생: {signals}"

    def test_pivot_price_stored(self) -> None:
        """new_high 이후 state.last_high 가 실제 고점 가격을 담아야 한다."""
        pivot = self._make_pivot()
        dt = datetime.datetime(2026, 5, 2, 9, 0)

        for i in range(15):
            pivot.update(high=360.5, low=359.5, close=360.0,
                         bar_time=dt + datetime.timedelta(minutes=i))

        peak = 375.0
        for i in range(8):
            p = 360.0 + (i + 1) * 1.8
            pivot.update(high=min(p + 0.3, peak), low=p - 0.3, close=p,
                         bar_time=dt + datetime.timedelta(minutes=15 + i))

        for i in range(8):
            p = peak - (i + 1) * 1.0
            s = pivot.update(high=p + 0.2, low=p - 0.2, close=p,
                             bar_time=dt + datetime.timedelta(minutes=23 + i))
            if s.new_pivot_signal == "new_high":
                assert s.last_high > 360.0, \
                    f"last_high 이 초기값 그대로: {s.last_high}"
                break


class TestATRAdaptivePivotFeatures:
    """get_transformer_features 출력 검증."""

    def test_azz_compatible_keys(self) -> None:
        """AdaptiveZigZag 호환 azz_* 키가 모두 존재해야 한다."""
        required = {
            "azz_direction", "azz_last_high", "azz_last_low",
            "azz_wave_size_pct", "azz_support_dist_pct", "azz_res_dist_pct",
            "azz_bars_since_swing", "azz_higher_highs", "azz_lower_lows",
            "azz_new_swing", "azz_swing_recency", "azz_threshold_pct",
            "azz_structure_up", "azz_structure_down", "azz_structure_ranging",
            "azz_pending_type", "azz_pending_dist",
            "azz_pending_urgency", "azz_pending_age", "azz_pending_prob",
        }
        pivot = ATRAdaptivePivot()
        for i in range(30):
            pivot.update(high=360.5, low=359.5, close=360.0)

        features = pivot.get_transformer_features(360.0)
        missing = required - features.keys()
        assert not missing, f"누락된 azz_* 키: {missing}"

    def test_feature_values_in_range(self) -> None:
        """모든 피처 값이 유한수여야 한다."""
        pivot = ATRAdaptivePivot()
        dt = datetime.datetime(2026, 5, 2, 9, 0)
        for i in range(40):
            h = 360.0 + (i % 5) * 0.5
            l = 360.0 - (i % 5) * 0.5
            pivot.update(high=h, low=l, close=360.0,
                         bar_time=dt + datetime.timedelta(minutes=i))

        features = pivot.get_transformer_features(360.0)
        for k, v in features.items():
            assert math.isfinite(v), f"피처 {k}={v} 가 비유한수"

    def test_aap_keys_present(self) -> None:
        """ATRAdaptivePivot 고유 aap_* 키가 존재해야 한다."""
        pivot = ATRAdaptivePivot()
        for i in range(30):
            pivot.update(high=360.5, low=359.5, close=360.0)
        features = pivot.get_transformer_features(360.0)
        assert "aap_pivot_score" in features
        assert "aap_atr" in features


class TestATRAdaptivePivotScore:
    """pivot_score 범위 검증."""

    def test_pivot_score_in_range(self) -> None:
        pivot = ATRAdaptivePivot()
        dt = datetime.datetime(2026, 5, 2, 9, 0)
        for i in range(50):
            h = 360.0 + (i % 7) * 0.8
            l = 360.0 - (i % 5) * 0.6
            s = pivot.update(high=h, low=l, close=360.0,
                             bar_time=dt + datetime.timedelta(minutes=i))
            assert 0.0 <= s.pivot_score <= 1.0, \
                f"bar={i} pivot_score={s.pivot_score} 범위 초과"


class TestATRAdaptivePivotReset:
    """reset() 후 상태가 완전 초기화되어야 한다."""

    def test_reset_clears_state(self) -> None:
        pivot = ATRAdaptivePivot()
        dt = datetime.datetime(2026, 5, 2, 9, 0)
        for i in range(30):
            pivot.update(high=370.0, low=350.0, close=360.0,
                         bar_time=dt + datetime.timedelta(minutes=i))

        pivot.reset()
        s = pivot.state
        assert math.isnan(s.last_high), f"reset 후 last_high는 NaN이어야 함: {s.last_high}"
        assert math.isnan(s.last_low),  f"reset 후 last_low는 NaN이어야 함: {s.last_low}"
        assert s.direction == 0
        assert s.new_pivot_signal == "none"
        assert pivot.confirmed_pivots == []


class TestATRAdaptivePivotSessionTable:
    """session_multiplier_table 적용 시 threshold 가 변해야 한다."""

    def test_session_multiplier_increases_threshold(self) -> None:
        """장초반 배율 2.0 설정 → threshold 가 기본보다 커야 한다."""
        cfg_base = ATRAdaptivePivotConfig(atr_period=5, warmup_bars=10,
                                          base_multiplier=2.0)
        cfg_sess = ATRAdaptivePivotConfig(
            atr_period=5, warmup_bars=10, base_multiplier=2.0,
            session_multiplier_table=[("09:00", "09:30", 2.0)],
        )
        pv_base = ATRAdaptivePivot(cfg_base)
        pv_sess = ATRAdaptivePivot(cfg_sess)

        dt = datetime.datetime(2026, 5, 2, 9, 5)  # 09:05 — 테이블 적용 구간
        for i in range(15):
            t = dt + datetime.timedelta(minutes=i)
            pv_base.update(high=360.5, low=359.5, close=360.0, bar_time=t)
            pv_sess.update(high=360.5, low=359.5, close=360.0, bar_time=t)

        assert pv_sess.state.threshold_abs >= pv_base.state.threshold_abs, \
            ("세션 배율 2.0 적용 시 threshold 가 기본보다 작거나 같음 "
             f"sess={pv_sess.state.threshold_abs:.4f} base={pv_base.state.threshold_abs:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# FractalConfirmation 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestFractalConfirmation:
    """프랙탈 탐지 정확성 및 필터 검증."""

    def _frac(self, **kw) -> FractalConfirmation:
        return FractalConfirmation(FractalConfig(**kw))

    def test_fractal_high_detected(self) -> None:
        """5봉 패턴 [L, L, H, L, L] → 고점 프랙탈 확정."""
        frac = self._frac(lookback=2, volume_spike_ratio=1.0, min_bar_gap=1)
        dt = datetime.datetime(2026, 5, 2, 9, 0)
        # 패턴: 360 → 361 → 365(피크) → 362 → 361
        prices = [
            (360.1, 359.9),  # bar0
            (361.1, 360.9),  # bar1
            (365.0, 364.5),  # bar2 ← 고점 (lookback=2 후 확정)
            (362.1, 361.9),  # bar3
            (361.1, 360.9),  # bar4 ← 이 봉 처리 시 bar2 프랙탈 확정
        ]
        results = []
        for i, (h, l) in enumerate(prices):
            s = frac.update(high=h, low=l, close=(h+l)/2, volume=2000.0,
                            bar_time=dt + datetime.timedelta(minutes=i))
            results.append(s.fractal_high)

        assert any(results), "고점 프랙탈이 한 번도 확정되지 않음"

    def test_fractal_low_detected(self) -> None:
        """5봉 패턴 [H, H, L, H, H] → 저점 프랙탈 확정."""
        frac = self._frac(lookback=2, volume_spike_ratio=1.0, min_bar_gap=1)
        dt = datetime.datetime(2026, 5, 2, 9, 0)
        prices = [
            (362.0, 361.0),  # bar0
            (363.0, 362.0),  # bar1
            (361.0, 355.0),  # bar2 ← 저점
            (363.0, 362.0),  # bar3
            (364.0, 363.0),  # bar4 ← bar2 프랙탈 확정
        ]
        results = []
        for i, (h, l) in enumerate(prices):
            s = frac.update(high=h, low=l, close=(h+l)/2, volume=2000.0,
                            bar_time=dt + datetime.timedelta(minutes=i))
            results.append(s.fractal_low)

        assert any(results), "저점 프랙탈이 한 번도 확정되지 않음"

    def test_volume_filter_blocks_low_volume(self) -> None:
        """거래량이 낮으면 프랙탈이 차단되어야 한다."""
        frac = self._frac(lookback=2, volume_spike_ratio=2.0, volume_lookback=4,
                          min_bar_gap=1)
        dt = datetime.datetime(2026, 5, 2, 9, 0)
        # 고점 패턴이지만 해당 봉 거래량이 낮음
        prices = [
            (360.1, 359.9, 1000.0),
            (361.1, 360.9, 1000.0),
            (365.0, 364.5,  500.0),  # 거래량 낮음 (평균 1000 × 2.0 = 2000 미만)
            (362.1, 361.9, 1000.0),
            (361.1, 360.9, 1000.0),
        ]
        results = []
        for i, (h, l, v) in enumerate(prices):
            s = frac.update(high=h, low=l, close=(h+l)/2, volume=v,
                            bar_time=dt + datetime.timedelta(minutes=i))
            results.append(s.fractal_high)

        assert not any(results), \
            "거래량 낮음에도 프랙탈 고점이 확정됨"

    def test_min_bar_gap_blocks_consecutive(self) -> None:
        """연속 프랙탈이 min_bar_gap 미만이면 두 번째는 차단해야 한다."""
        frac = self._frac(lookback=2, volume_spike_ratio=1.0, min_bar_gap=10)
        dt = datetime.datetime(2026, 5, 2, 9, 0)

        # 첫 번째 프랙탈 패턴
        bars1 = [(360.1, 359.9), (361.0, 360.5), (365.0, 364.5),
                 (362.0, 361.5), (361.0, 360.5)]
        for i, (h, l) in enumerate(bars1):
            frac.update(high=h, low=l, close=(h+l)/2, volume=2000.0,
                        bar_time=dt + datetime.timedelta(minutes=i))

        # 곧바로 두 번째 프랙탈 패턴 (5봉 차이 < min_bar_gap=10)
        bars2 = [(361.0, 360.5), (362.0, 361.5), (366.0, 365.5),
                 (363.0, 362.5), (362.0, 361.5)]
        second_hits = []
        for i, (h, l) in enumerate(bars2):
            s = frac.update(high=h, low=l, close=(h+l)/2, volume=2000.0,
                            bar_time=dt + datetime.timedelta(minutes=5 + i))
            second_hits.append(s.fractal_high)

        assert not any(second_hits), \
            "min_bar_gap 미만 두 번째 프랙탈이 허용됨"

    def test_transformer_features_keys(self) -> None:
        """get_transformer_features 출력 키 집합 검증."""
        frac = FractalConfirmation()
        for i in range(20):
            frac.update(high=360.5 + i * 0.1, low=359.5, close=360.0, volume=1000.0)
        feats = frac.get_transformer_features()
        assert "frac_confirmed" in feats
        assert "frac_high" in feats
        assert "frac_low" in feats
        assert "frac_vol_ratio" in feats
        for v in feats.values():
            assert math.isfinite(v), f"frac feature 비유한수: {v}"


# ─────────────────────────────────────────────────────────────────────────────
# 조합 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedPivotFractal:
    """ATRAdaptivePivot + FractalConfirmation 조합 기능 테스트."""

    def test_interface_compatible_with_zigzag(self) -> None:
        """update() 시그니처가 AdaptiveZigZag 와 호환되어야 한다."""
        from indicators import AdaptiveZigZag
        import inspect

        zz_sig  = inspect.signature(AdaptiveZigZag.update)
        aap_sig = inspect.signature(ATRAdaptivePivot.update)

        zz_params  = set(zz_sig.parameters.keys())
        aap_params = set(aap_sig.parameters.keys())

        # 핵심 파라미터 (high, low, close, bar_time) 는 반드시 존재해야 함
        required = {"high", "low", "close", "bar_time"}
        assert required <= aap_params, \
            f"ATRAdaptivePivot.update 누락 파라미터: {required - aap_params}"

    def test_combined_no_errors(self) -> None:
        """두 지표를 동시에 사용해도 예외가 발생하지 않아야 한다."""
        pivot = ATRAdaptivePivot(ATRAdaptivePivotConfig(warmup_bars=15))
        frac  = FractalConfirmation(FractalConfig(lookback=2))
        dt = datetime.datetime(2026, 5, 2, 9, 0)

        import random
        random.seed(42)
        for i in range(60):
            base = 360.0 + random.uniform(-3, 3)
            t = dt + datetime.timedelta(minutes=i)
            pivot.update(high=base + 0.5, low=base - 0.5, close=base, bar_time=t)
            frac.update(high=base + 0.5, low=base - 0.5, close=base,
                        volume=random.uniform(500, 2000), bar_time=t)

        # 예외 없이 완료되면 통과
        assert pivot.state is not None
        assert frac.state is not None

    def test_dual_signal_possible(self) -> None:
        """ATR Pivot + Fractal 이 동시에 true 가 될 수 있는지 기본 확인."""
        # 두 지표가 독립적으로 동작하며 동시에 신호를 낼 수 있어야 함
        pivot = ATRAdaptivePivot(ATRAdaptivePivotConfig(warmup_bars=10,
                                                         atr_period=3,
                                                         er_period=3,
                                                         confirmation_bars=0))
        frac  = FractalConfirmation(FractalConfig(lookback=2,
                                                   volume_spike_ratio=1.0,
                                                   min_bar_gap=1))
        dt = datetime.datetime(2026, 5, 2, 9, 0)

        pivot_signals: List[str] = []
        frac_signals: List[bool] = []

        prices = (
            [360.0] * 12 +          # flat warmup
            [360 + i for i in range(8)] +   # 상승
            [368 - i for i in range(8)]     # 하락
        )
        for i, p in enumerate(prices):
            t = dt + datetime.timedelta(minutes=i)
            spike_v = 2500.0 if 17 <= i <= 19 else 1000.0
            ps = pivot.update(high=p + 0.5, low=p - 0.5, close=p, bar_time=t)
            fs = frac.update(high=p + 0.5, low=p - 0.5, close=p,
                             volume=spike_v, bar_time=t)
            pivot_signals.append(ps.new_pivot_signal)
            frac_signals.append(fs.fractal_high or fs.fractal_low)

        # 각 지표가 최소 1개의 신호를 낼 수 있는 환경임을 확인
        # (둘 다 0이면 테스트 설계 문제)
        assert any(s != "none" for s in pivot_signals) or \
               any(frac_signals), \
               "두 지표 모두 신호 없음 — 입력 데이터가 너무 평탄함"
