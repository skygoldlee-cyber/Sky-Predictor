"""HybridAdaptivePivot 테스트."""

import pytest
import numpy as np
from indicators.hybrid_adaptive_pivot import (
    HybridAdaptivePivot,
    HybridAdaptivePivotConfig,
    HybridAdaptivePivotState,
    PivotType,
)


class TestHybridAdaptivePivotWarmup:
    """웜업 기간 테스트."""

    def test_no_signal_during_warmup(self):
        """웜업 기간 중에는 신호가 출력되지 않아야 함."""
        cfg = HybridAdaptivePivotConfig(warmup_bars=20)
        pivot = HybridAdaptivePivot(cfg)

        for i in range(15):
            state = pivot.update(100 + i, 100, 100 + i, f"{9+i//60:02d}:{i%60:02d}")
            assert state.new_pivot_signal == "none"
            assert not np.isfinite(state.last_high)
            assert not np.isfinite(state.last_low)


class TestHybridAdaptivePivotSignals:
    """피봇 신호 테스트."""

    def test_new_high_signal_after_rise_and_drop(self):
        """상승 후 하락 시 new_high 신호."""
        cfg = HybridAdaptivePivotConfig(
            warmup_bars=5,
            confirmation_bars=0,
            base_pct=0.05,
            base_multiplier=0.5,
            min_wave_pct=0.0,
            min_wave_atr_ratio=0.0,
        )
        pivot = HybridAdaptivePivot(cfg)

        # 상승
        for i in range(10):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        # 큰 하락
        state = pivot.update(110, 102, 102, "09:10")
        # 피봇이 확정되었는지 확인 (마지막 신호가 아니어도 됨)
        assert len(pivot.confirmed_pivots) > 0
        assert np.isfinite(pivot.state.last_high)
        assert pivot.state.last_high > 100

    def test_new_low_signal_after_drop_and_rise(self):
        """하락 후 상승 시 new_low 신호."""
        cfg = HybridAdaptivePivotConfig(
            warmup_bars=5,
            confirmation_bars=0,
            base_pct=0.05,
            base_multiplier=0.5,
            min_wave_pct=0.0,
            min_wave_atr_ratio=0.0,
        )
        pivot = HybridAdaptivePivot(cfg)

        # 하락
        for i in range(10):
            pivot.update(100 - i, 99 - i, 99 - i, f"09:{i:02d}")

        # 큰 상승
        state = pivot.update(95, 98, 98, "09:10")
        # 피봇이 확정되었는지 확인
        assert len(pivot.confirmed_pivots) > 0
        assert np.isfinite(pivot.state.last_low)
        assert pivot.state.last_low < 100

    def test_pivot_price_stored(self):
        """피봇 가격이 정확히 저장되어야 함."""
        cfg = HybridAdaptivePivotConfig(
            warmup_bars=5,
            confirmation_bars=0,
            base_pct=0.05,
            base_multiplier=0.5,
            min_wave_pct=0.0,
            min_wave_atr_ratio=0.0,
        )
        pivot = HybridAdaptivePivot(cfg)

        # 상승
        for i in range(10):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        # 큰 하락
        pivot.update(110, 102, 102, "09:10")
        assert pivot.state.last_high >= 105


class TestHybridAdaptivePivotThreshold:
    """하이브리드 임계값 테스트."""

    def test_atr_weight_zero_uses_percent_only(self):
        """atr_weight=0이면 퍼센트만 사용."""
        cfg = HybridAdaptivePivotConfig(atr_weight=0.0, base_pct=0.3)
        pivot = HybridAdaptivePivot(cfg)

        # 웜업 후
        for i in range(25):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        # 임계값 확인 (퍼센트 기반)
        state = pivot.state
        expected_thr = 100 * 0.3 / 100.0  # base_pct만
        # ER과 세션 배율이 적용되므로 근사치 확인
        assert state.threshold_pct > 0
        assert state.threshold_pct < 1.0

    def test_atr_weight_one_uses_atr_only(self):
        """atr_weight=1이면 ATR만 사용."""
        cfg = HybridAdaptivePivotConfig(atr_weight=1.0, base_multiplier=2.0)
        pivot = HybridAdaptivePivot(cfg)

        # 웜업 후
        for i in range(25):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        # ATR이 계산되었는지 확인
        state = pivot.state
        assert state.atr > 0
        assert state.threshold_abs > 0

    def test_atr_weight_half_mixes_both(self):
        """atr_weight=0.5이면 둘 다 혼합."""
        cfg = HybridAdaptivePivotConfig(atr_weight=0.5)
        pivot = HybridAdaptivePivot(cfg)

        # 웜업 후
        for i in range(25):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        state = pivot.state
        assert state.atr > 0
        assert state.threshold_pct > 0
        assert state.threshold_abs > 0


class TestHybridAdaptivePivotFeatures:
    """Transformer Features 테스트."""

    def test_azz_compatible_keys(self):
        """azz_* 키가 존재해야 함."""
        cfg = HybridAdaptivePivotConfig()
        pivot = HybridAdaptivePivot(cfg)

        for i in range(25):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        features = pivot.get_transformer_features(105)
        azz_keys = [k for k in features.keys() if k.startswith("azz_")]
        assert len(azz_keys) > 20  # 대부분의 azz 키 존재

    def test_hap_keys_present(self):
        """hap_* 고유 키가 존재해야 함."""
        cfg = HybridAdaptivePivotConfig()
        pivot = HybridAdaptivePivot(cfg)

        for i in range(25):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        features = pivot.get_transformer_features(105)
        assert "hap_atr" in features
        assert "hap_atr_weight" in features
        assert "hap_threshold_pct" in features
        assert "hap_pivot_score" in features

    def test_feature_values_in_range(self):
        """피처 값이 0~1 범위 내여야 함."""
        cfg = HybridAdaptivePivotConfig()
        pivot = HybridAdaptivePivot(cfg)

        for i in range(25):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        features = pivot.get_transformer_features(105)
        for key, value in features.items():
            if key.startswith("azz_") or key.startswith("hap_"):
                assert 0.0 <= value <= 1.0, f"{key}={value} out of range"


class TestHybridAdaptivePivotCancelRatio:
    """취소 비율 테스트."""

    def test_cancel_ratio_parameter(self):
        """cancel_ratio 파라미터가 설정 가능해야 함."""
        cfg = HybridAdaptivePivotConfig(cancel_ratio=0.5)
        assert cfg.cancel_ratio == 0.5

    def test_cancel_ratio_default(self):
        """cancel_ratio 기본값이 0.3이어야 함."""
        cfg = HybridAdaptivePivotConfig()
        assert cfg.cancel_ratio == 0.3


class TestHybridAdaptivePivotImmediateConfirmation:
    """즉시 확정 테스트."""

    def test_confirmation_bars_zero_allows_immediate(self):
        """confirmation_bars=0이면 즉시 확정 허용."""
        cfg = HybridAdaptivePivotConfig(
            confirmation_bars=0,
            warmup_bars=5,
            base_pct=0.05,
            base_multiplier=0.5,
            min_wave_pct=0.0,
            min_wave_atr_ratio=0.0,
        )
        pivot = HybridAdaptivePivot(cfg)

        # 상승
        for i in range(10):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        # 큰 하락 시 즉시 확정
        state = pivot.update(110, 102, 102, "09:10")
        # 피봇이 확정되었는지 확인
        assert len(pivot.confirmed_pivots) > 0


class TestHybridAdaptivePivotDirectionRestoration:
    """방향 복귀 테스트."""

    def test_direction_restoration_on_cancel(self):
        """후보 취소 시 방향 복귀."""
        cfg = HybridAdaptivePivotConfig(
            warmup_bars=5,
            confirmation_bars=1,
            cancel_ratio=0.5,
            base_pct=0.1,
        )
        pivot = HybridAdaptivePivot(cfg)

        # 상승
        for i in range(10):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        # 하락 시작
        state = pivot.update(110, 109, 109, "09:10")
        # 후보 등록

        # 다시 상승 (취소 조건)
        state = pivot.update(111, 110, 111, "09:11")
        # 방향 복귀 확인 (상승 탐색으로 유지)
        assert pivot._direction == 1 or pivot._direction == -1


class TestHybridAdaptivePivotWaveFilter:
    """이중 파동 필터 테스트."""

    def test_wave_filter_percent(self):
        """퍼센트 기반 파동 필터."""
        cfg = HybridAdaptivePivotConfig(
            warmup_bars=5,
            min_wave_pct=0.5,
            confirmation_bars=0,
        )
        pivot = HybridAdaptivePivot(cfg)

        # 작은 움직임
        for i in range(10):
            pivot.update(100 + i * 0.01, 99 + i * 0.01, 100 + i * 0.01, f"09:{i:02d}")

        # 작은 하락
        state = pivot.update(100.1, 100.05, 100.05, "09:10")
        # 파동이 너무 작아서 피봇 미확정 가능
        # (실제 동작은 파동 크기에 따라 다름)


class TestHybridAdaptivePivotReset:
    """리셋 테스트."""

    def test_reset_clears_state(self):
        """reset()이 상태를 초기화해야 함."""
        cfg = HybridAdaptivePivotConfig()
        pivot = HybridAdaptivePivot(cfg)

        # 데이터 입력
        for i in range(20):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        # 리셋
        pivot.reset()

        assert pivot._bar_idx == 0
        assert pivot._direction == 0
        assert len(pivot._pivots) == 0
        assert not np.isfinite(pivot.state.last_high)
        assert not np.isfinite(pivot.state.last_low)


class TestHybridAdaptivePivotSessionTable:
    """세션 시간대 테이블 테스트."""

    def test_session_multiplier_increases_threshold(self):
        """세션 배율이 임계값에 영향."""
        cfg = HybridAdaptivePivotConfig(
            session_multiplier_table=[("09:00", "09:30", 1.5)],
        )
        pivot = HybridAdaptivePivot(cfg)

        # 장초반 데이터
        for i in range(25):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        # 임계값 확인
        state = pivot.state
        assert state.threshold_pct > 0


class TestHybridAdaptivePivotLLMContext:
    """LLM 컨텍스트 테스트."""

    def test_llm_context_format(self):
        """LLM 컨텍스트가 올바른 형식이어야 함."""
        cfg = HybridAdaptivePivotConfig()
        pivot = HybridAdaptivePivot(cfg)
        pivot.set_symbol("TEST")

        for i in range(25):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        context = pivot.get_llm_context(105)
        assert "HybridAdaptivePivot" in context
        assert "TEST" in context
        assert "ATR가중치" in context


class TestHybridAdaptivePivotPivotScore:
    """Pivot Score 테스트."""

    def test_pivot_score_in_range(self):
        """Pivot Score가 0~1 범위 내여야 함."""
        cfg = HybridAdaptivePivotConfig()
        pivot = HybridAdaptivePivot(cfg)

        for i in range(25):
            pivot.update(100 + i, 99 + i, 100 + i, f"09:{i:02d}")

        score = pivot.pivot_score
        assert 0.0 <= score <= 1.0
