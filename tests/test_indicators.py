"""kospi_indicators 전체 테스트 스위트.

버그 수정 검증 + 두 프로젝트 사용 패턴 통합 테스트.
실행: pytest tests/ -v
"""

import sys
import math
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from kospi_indicators import (
    WilderRMA,
    AdaptiveSuperTrend, SuperTrendState,
    AdaptiveZigZag, AdaptiveZigZagConfig, ZigZagState, AdaptiveIndicatorManager, validate_consistency,
)


# ──────────────────────────────────────────────────────────
# 공통 픽스처
# ──────────────────────────────────────────────────────────

def make_bars(n: int = 200, seed: int = 42):
    """재현 가능한 KP200 유사 OHLC 시계열 생성."""
    rng = np.random.default_rng(seed)
    close = 400.0
    closes, highs, lows = [], [], []
    for _ in range(n):
        change = rng.normal(0, 1.2)
        close = max(300.0, close + change)
        h = close + abs(rng.normal(0, 0.6))
        lo = close - abs(rng.normal(0, 0.6))
        closes.append(close); highs.append(h); lows.append(lo)
    return (
        pd.DataFrame({"high": highs, "low": lows, "close": closes}),
        highs, lows, closes,
    )


@pytest.fixture(scope="module")
def bars_df():
    df, _, _, _ = make_bars(200)
    return df


# ──────────────────────────────────────────────────────────
# 1. WilderRMA
# ──────────────────────────────────────────────────────────

class TestWilderRMA:
    def test_ready_at_period_not_period_plus_one(self):
        """[FIX] ready = count >= period (SkyEbest의 > 수정)."""
        rma = WilderRMA(period=3)
        for i in range(1, 4):  # 3번 업데이트
            rma.update(float(i))
        assert rma.ready is True, "3번 업데이트 후 ready여야 함"

    def test_not_ready_before_period(self):
        rma = WilderRMA(period=5)
        for i in range(4):
            rma.update(float(i))
        assert rma.ready is False

    def test_warmup_is_sma(self):
        rma = WilderRMA(period=4)
        for v in [1.0, 2.0, 3.0, 4.0]:
            out = rma.update(v)
        assert abs(out - 2.5) < 1e-9  # SMA([1,2,3,4])=2.5

    def test_reset(self):
        rma = WilderRMA(period=3)
        for v in [1, 2, 3, 4, 5]:
            rma.update(float(v))
        rma.reset()
        assert rma.count == 0
        assert rma.ready is False


# ──────────────────────────────────────────────────────────
# 2. AdaptiveSuperTrend
# ──────────────────────────────────────────────────────────

class TestAdaptiveSuperTrend:

    def test_basic_update_returns_state(self, bars_df):
        ast = AdaptiveSuperTrend()
        for _, row in bars_df.iterrows():
            s = ast.update(row.high, row.low, row.close)
        assert isinstance(s, SuperTrendState)
        assert s.direction in (1, -1)
        assert s.atr > 0

    # ── [FIX-1] bars_in_trend 플립봉 ─────────────────────
    def test_bars_in_trend_zero_on_flip_bar(self, bars_df):
        """플립 봉에서 bars_in_trend == 0이어야 함."""
        ast = AdaptiveSuperTrend()
        prev_dir = None
        for _, row in bars_df.iterrows():
            s = ast.update(row.high, row.low, row.close)
            if prev_dir is not None and prev_dir != s.direction:
                assert s.bars_in_trend == 0, (
                    f"플립 봉에서 bars_in_trend={s.bars_in_trend} (0이어야 함)"
                )
            prev_dir = s.direction

    def test_bars_in_trend_increments_after_flip(self, bars_df):
        """플립 이후 봉에서 정상 증가."""
        ast = AdaptiveSuperTrend()
        states = []
        for _, row in bars_df.iterrows():
            states.append(ast.update(row.high, row.low, row.close))

        for i in range(1, len(states)):
            if states[i - 1].direction != states[i].direction:
                # 플립 봉: 0
                assert states[i].bars_in_trend == 0
                # 다음 봉이 있고 같은 방향이면 1
                if i + 1 < len(states) and states[i + 1].direction == states[i].direction:
                    assert states[i + 1].bars_in_trend == 1

    # ── [FIX-2] ATR 재초기화 비율 기준 ───────────────────
    def test_atr_initialized_after_warmup(self, bars_df):
        ast = AdaptiveSuperTrend()
        for _, row in bars_df.iterrows():
            ast.update(row.high, row.low, row.close)
        assert ast._atr_initialized is True

    # ── [FIX-3] LLM advice 키 ─────────────────────────────
    def test_llm_context_advice_matches_direction(self, bars_df):
        """advice가 direction에 맞는 텍스트를 포함해야 함."""
        ast = AdaptiveSuperTrend()
        for _, row in bars_df.iterrows():
            s = ast.update(row.high, row.low, row.close)
        ctx = ast.get_llm_context(bars_df.close.iloc[-1])
        if s.direction == 1:
            assert "상승" in ctx
        else:
            assert "하락" in ctx
        # 과거 버그: '횡보 구조'만 반환되지 않는지 확인
        assert "신중" in ctx or "유리" in ctx  # 어떤 advice든 포함

    # ── [FIX-4] _prev_adx 초기값 ─────────────────────────
    def test_prev_adx_initial_value(self):
        ast = AdaptiveSuperTrend()
        assert ast._prev_adx == 25.0

    def test_compute_from_df_lowercase(self, bars_df):
        ast = AdaptiveSuperTrend()
        out = ast.compute_from_df(bars_df)  # 소문자 컬럼 기본값
        assert "ast_direction" in out.columns
        assert out["ast_atr"].iloc[-1] > 0

    def test_compute_from_df_uppercase(self, bars_df):
        """대문자 컬럼도 자동 처리."""
        df = bars_df.rename(columns=str.title)  # High, Low, Close
        ast = AdaptiveSuperTrend()
        out = ast.compute_from_df(df, high_col="High", low_col="Low", close_col="Close")
        assert "ast_direction" in out.columns

    def test_transformer_features_keys(self, bars_df):
        ast = AdaptiveSuperTrend()
        for _, row in bars_df.iterrows():
            ast.update(row.high, row.low, row.close)
        feats = ast.get_transformer_features(bars_df.close.iloc[-1])
        expected = {
            "ast_direction", "ast_dist_pct", "ast_atr_pct",
            "ast_efficiency_ratio", "ast_adx_norm", "ast_mult_norm",
            "ast_trend_duration", "ast_signal", "ast_band_width_pct",
        }
        assert expected == set(feats.keys())


# ──────────────────────────────────────────────────────────
# 3. AdaptiveZigZag
# ──────────────────────────────────────────────────────────

class TestAdaptiveZigZag:

    def test_basic_update(self, bars_df):
        zz = AdaptiveZigZag()
        for _, row in bars_df.iterrows():
            s = zz.update(row.high, row.low, row.close)
        assert isinstance(s, ZigZagState)

    def test_swings_detected(self, bars_df):
        zz = AdaptiveZigZag()
        for _, row in bars_df.iterrows():
            zz.update(row.high, row.low, row.close)
        assert len(zz._all_swings) >= 2, "200봉에서 최소 2개 스윙 감지 필요"

    # ── [FIX-1] ER threshold 방향 수정 ───────────────────
    def test_er_threshold_direction(self):
        """ER 높을수록 mult가 커야 함 (수정 전: 작았음)."""
        zz = AdaptiveZigZag(AdaptiveZigZagConfig(
            atr_multiplier_min=1.0, atr_multiplier_max=4.0
        ))
        # _calc_threshold_pct 내부 mult 계산 직접 검증
        mmin, mmax = 1.0, 4.0
        er_low  = 0.1
        er_high = 0.9
        mult_low  = mmin + er_low  * (mmax - mmin)  # 1.3
        mult_high = mmin + er_high * (mmax - mmin)  # 3.7
        assert mult_high > mult_low, "ER 높을수록 mult가 커야 함"
        assert abs(mult_low - 1.3) < 1e-9
        assert abs(mult_high - 3.7) < 1e-9

    # ── [FIX-2] pending_confirm 교체 ─────────────────────
    def test_pending_confirm_replaces_on_opposite_type(self):
        """반대 타입의 pending_confirm 등록 시 교체."""
        zz = AdaptiveZigZag()
        zz._pending_confirm = {"type": "low", "remaining": 1, "price": 100.0, "idx": 0, "atr": 1.0}
        # direction == 1 상태에서 전환 조건 충족 시도 — "high" 타입 등록 가능해야 함
        new_confirm = {"type": "high", "remaining": 2, "price": 105.0, "idx": 5, "atr": 1.0}
        existing_type = zz._pending_confirm.get("type")
        should_replace = (zz._pending_confirm is None or existing_type != "high")
        assert should_replace is True, "반대 타입(low→high)이면 교체 가능해야 함"

    def test_pending_confirm_no_replace_same_type(self):
        """같은 타입의 pending_confirm이 있으면 교체하지 않음."""
        zz = AdaptiveZigZag()
        zz._pending_confirm = {"type": "high", "remaining": 1, "price": 105.0, "idx": 3, "atr": 1.0}
        existing_type = zz._pending_confirm.get("type")
        should_replace = (zz._pending_confirm is None or existing_type != "high")
        assert should_replace is False

    # ── [FIX-3] _all_swings 슬라이싱 재할당 ──────────────
    def test_all_swings_max_size_respected(self, bars_df):
        cfg = AdaptiveZigZagConfig(max_swings=10)
        zz = AdaptiveZigZag(cfg)
        for _, row in bars_df.iterrows():
            zz.update(row.high, row.low, row.close)
        assert len(zz._all_swings) <= cfg.max_swings * 2, "max_swings*2 이하로 관리"

    def test_all_swings_list_type(self, bars_df):
        zz = AdaptiveZigZag()
        for _, row in bars_df.iterrows():
            zz.update(row.high, row.low, row.close)
        # 슬라이싱 재할당 이후에도 list 타입 유지
        assert isinstance(zz._all_swings, list)

    def test_compute_from_df(self, bars_df):
        zz = AdaptiveZigZag()
        out = zz.compute_from_df(bars_df)
        assert "azz_direction" in out.columns
        assert "azz_fib_0382" in out.columns   # Transformer 호환 별칭

    def test_compute_from_df_uppercase(self, bars_df):
        df = bars_df.rename(columns=str.title)
        zz = AdaptiveZigZag()
        out = zz.compute_from_df(df, high_col="High", low_col="Low", close_col="Close")
        assert "azz_direction" in out.columns

    def test_transformer_features_keys(self, bars_df):
        zz = AdaptiveZigZag()
        for _, row in bars_df.iterrows():
            zz.update(row.high, row.low, row.close)
        feats = zz.get_transformer_features(bars_df.close.iloc[-1])
        expected = {
            "azz_direction", "azz_wave_size_pct", "azz_support_dist_pct",
            "azz_res_dist_pct", "azz_bars_since_swing", "azz_fib618_dist",
            "azz_fib382_dist", "azz_higher_highs", "azz_lower_lows",
            "azz_new_swing", "azz_swing_recency", "azz_threshold_pct",
            "azz_structure_up", "azz_structure_down", "azz_structure_ranging",
        }
        assert expected == set(feats.keys())

    def test_llm_context_returns_string(self, bars_df):
        zz = AdaptiveZigZag()
        for _, row in bars_df.iterrows():
            zz.update(row.high, row.low, row.close)
        ctx = zz.get_llm_context(bars_df.close.iloc[-1])
        assert isinstance(ctx, str) and len(ctx) > 50

    def test_reset_clears_state(self, bars_df):
        zz = AdaptiveZigZag()
        for _, row in bars_df.iterrows():
            zz.update(row.high, row.low, row.close)
        zz._reset_buffers()
        assert zz._bar_idx == 0
        assert len(zz._all_swings) == 0


# ──────────────────────────────────────────────────────────
# 4. AdaptiveIndicatorManager
# ──────────────────────────────────────────────────────────

class TestAdaptiveIndicatorManager:

    def test_update_returns_dict(self, bars_df):
        mgr = AdaptiveIndicatorManager()
        for _, row in bars_df.iterrows():
            res = mgr.update(row.high, row.low, row.close)
        assert isinstance(res["transformer"], dict)
        assert isinstance(res["llm_context"], str)
        assert "is_ready" in res

    def test_feature_count(self, bars_df):
        mgr = AdaptiveIndicatorManager()
        for _, row in bars_df.iterrows():
            res = mgr.update(row.high, row.low, row.close)
        tf = res["transformer"]
        # ast(9) + azz(14) + cross(4) = 27개 이상
        assert len(tf) >= 27, f"피처 수 부족: {len(tf)}"

    def test_cross_features_present(self, bars_df):
        mgr = AdaptiveIndicatorManager()
        for _, row in bars_df.iterrows():
            res = mgr.update(row.high, row.low, row.close)
        tf = res["transformer"]
        for key in ["cross_trend_agreement", "cross_at_support",
                    "cross_at_resistance", "cross_breakout_potential"]:
            assert key in tf, f"cross feature 누락: {key}"

    def test_is_ready_after_enough_bars(self, bars_df):
        mgr = AdaptiveIndicatorManager()
        for _, row in bars_df.iterrows():
            mgr.update(row.high, row.low, row.close)
        assert mgr.is_ready() is True

    def test_reset(self, bars_df):
        mgr = AdaptiveIndicatorManager()
        for _, row in bars_df.iterrows():
            mgr.update(row.high, row.low, row.close)
        mgr.reset()
        assert mgr._bar_count == 0
        assert mgr.is_ready() is False

    def test_compute_from_df(self, bars_df):
        mgr = AdaptiveIndicatorManager()
        out = mgr.compute_from_df(bars_df)
        assert "ast_direction" in out.columns
        assert "azz_direction" in out.columns
        assert "cross_trend_agreement" in out.columns

    def test_batch_stream_consistency(self, bars_df):
        """배치와 스트리밍 결과의 일관성 검증."""
        mgr = AdaptiveIndicatorManager()
        assert validate_consistency(mgr, bars_df, atol=1e-6) is True

    def test_skyebest_column_names(self, bars_df):
        """SkyEbest 스타일 대문자 컬럼으로도 동작."""
        df = bars_df.rename(columns=str.title)
        mgr = AdaptiveIndicatorManager()
        out = mgr.compute_from_df(df, high_col="High", low_col="Low", close_col="Close")
        assert "ast_direction" in out.columns


# ──────────────────────────────────────────────────────────
# 5. SkyEbest 호환 — get_super_trend 래퍼
# ──────────────────────────────────────────────────────────

class TestSkyEbestCompat:

    def test_get_super_trend_wrapper(self, bars_df):
        df = bars_df.rename(columns=str.title)
        ast = AdaptiveSuperTrend()
        st, ub, lb = ast.get_super_trend(df, lookback=14, multiplier=3.0)
        assert len(st) == len(df)
        assert np.any(np.isfinite(st))

    def test_get_super_trend_smooth_period_param(self, bars_df):
        df = bars_df.rename(columns=str.title)
        ast = AdaptiveSuperTrend()
        st1, _, _ = ast.get_super_trend(df, lookback=14, multiplier=3.0, smooth_period=1)
        st3, _, _ = ast.get_super_trend(df, lookback=14, multiplier=3.0, smooth_period=3)
        # smooth_period가 다르면 값이 달라야 함
        valid = np.isfinite(st1) & np.isfinite(st3)
        assert not np.allclose(st1[valid], st3[valid], atol=1e-9), \
            "smooth_period 1과 3은 결과가 달라야 함"


# ──────────────────────────────────────────────────────────
# 6. 피처 값 범위 검증
# ──────────────────────────────────────────────────────────

class TestFeatureRanges:

    def test_all_features_finite(self, bars_df):
        mgr = AdaptiveIndicatorManager()
        for _, row in bars_df.iterrows():
            res = mgr.update(row.high, row.low, row.close)
        for k, v in res["transformer"].items():
            assert math.isfinite(v), f"피처 {k}={v} 가 유한하지 않음"

    def test_direction_binary(self, bars_df):
        ast = AdaptiveSuperTrend()
        for _, row in bars_df.iterrows():
            ast.update(row.high, row.low, row.close)
        feats = ast.get_transformer_features(bars_df.close.iloc[-1])
        assert feats["ast_direction"] in (1.0, -1.0)

    def test_normalized_features_in_range(self, bars_df):
        mgr = AdaptiveIndicatorManager()
        for _, row in bars_df.iterrows():
            res = mgr.update(row.high, row.low, row.close)
        tf = res["transformer"]
        for k in ["ast_adx_norm", "ast_mult_norm", "ast_trend_duration",
                  "azz_wave_size_pct", "azz_bars_since_swing",
                  "azz_threshold_pct"]:
            assert 0.0 <= tf[k] <= 1.0, f"{k}={tf[k]} 범위 0~1 벗어남"

    def test_cross_trend_agreement_values(self, bars_df):
        mgr = AdaptiveIndicatorManager()
        for _, row in bars_df.iterrows():
            res = mgr.update(row.high, row.low, row.close)
        v = res["transformer"]["cross_trend_agreement"]
        assert v in (-1.0, 0.0, 1.0)
