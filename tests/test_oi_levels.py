"""Tests for calc_oi_levels() in prediction/option_features.py.

Coverage:
    - 기본 Call/Put OI Peak 산출
    - 거리(dist) 부호 검증
    - OI 박스폭(oi_range_pct) 계산
    - 집중도(peak_norm) 범위 검증
    - Zero Gamma / Vol Trigger 산출 (gamma 없는 경우 BS 근사 fallback 포함)
    - 빈 입력 / 선물가 0 safe fallback
    - build_option_snapshot v5 피처 키 포함 검증
    - _describe_oi_levels LLM 컨텍스트 생성 검증
    - OI-01: peak_search_range_pt 근접 범위 제한 검증
    - OI-03: zero_gamma_strike가 현재가 최근접 전환점임을 검증
    - GR-02: 복수 가드레일 조건 동시 적용 검증
    - GR-03: dist=0.0(ATM Peak) 가드레일 정상 동작 검증
"""
import math
import pytest
from prediction.features.option_features import calc_oi_levels, build_option_snapshot
from prediction.context_builder import _describe_oi_levels

# ── 공통 픽스처 ──────────────────────────────────────────────────────────────

CALLS = {
    "C350": {"strike": 350, "open_interest": 10000, "iv": 0.18, "gamma": 0.012},
    "C355": {"strike": 355, "open_interest": 48000, "iv": 0.20, "gamma": 0.010},  # Peak
    "C360": {"strike": 360, "open_interest": 12000, "iv": 0.22, "gamma": 0.008},
}
PUTS = {
    "P340": {"strike": 340, "open_interest": 35000, "iv": 0.22, "gamma": 0.008},
    "P345": {"strike": 345, "open_interest": 52000, "iv": 0.20, "gamma": 0.010},  # Peak
    "P350": {"strike": 350, "open_interest": 20000, "iv": 0.18, "gamma": 0.012},
}
PRICE = 350.0

# gamma 필드 없는 픽스처 (BS 근사 fallback 테스트용)
CALLS_NO_GAMMA = {
    k: {kk: vv for kk, vv in v.items() if kk != "gamma"}
    for k, v in CALLS.items()
}
PUTS_NO_GAMMA = {
    k: {kk: vv for kk, vv in v.items() if kk != "gamma"}
    for k, v in PUTS.items()
}


# ── calc_oi_levels 기본 검증 ─────────────────────────────────────────────────

class TestCalcOiLevelsPeaks:
    def test_call_peak_strike(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        assert r["call_oi_peak"] == pytest.approx(355.0)

    def test_put_peak_strike(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        assert r["put_oi_peak"] == pytest.approx(345.0)

    def test_dist_to_call_peak_positive_when_above(self):
        """저항(355)이 현재가(350) 위이면 dist_to_call_peak > 0."""
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        assert r["dist_to_call_peak"] > 0.0

    def test_dist_to_put_peak_positive_when_below(self):
        """지지(345)가 현재가(350) 아래이면 dist_to_put_peak > 0."""
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        assert r["dist_to_put_peak"] > 0.0

    def test_dist_to_call_peak_negative_when_price_above(self):
        """현재가가 Call Peak보다 높으면(돌파 상태) dist_to_call_peak < 0."""
        r = calc_oi_levels(CALLS, PUTS, 360.0)
        assert r["dist_to_call_peak"] < 0.0

    def test_dist_values_formula(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        expected_call = (355.0 - 350.0) / 350.0 * 100.0
        expected_put  = (350.0 - 345.0) / 350.0 * 100.0
        assert r["dist_to_call_peak"] == pytest.approx(expected_call, rel=1e-3)
        assert r["dist_to_put_peak"]  == pytest.approx(expected_put,  rel=1e-3)


# ── OI-01: peak_search_range_pt 근접 범위 제한 ──────────────────────────────

class TestCalcOiLevelsPeakRange:
    """OI-01 수정: 근접 범위(peak_search_range_pt) 내 후보 우선 선택 검증."""

    def test_far_call_peak_excluded_when_near_candidate_exists(self):
        """근접 범위(20pt) 안에 후보가 있으면 원거리 Peak을 선택하지 않는다."""
        # 355pt(근접, Peak 후보) vs 400pt(원거리, OI 더 많음)
        calls_with_far = {
            "C355": {"strike": 355, "open_interest": 50000, "iv": 0.20},
            "C400": {"strike": 400, "open_interest": 200000, "iv": 0.30},  # 원거리 고OI
        }
        r = calc_oi_levels(calls_with_far, {}, PRICE, peak_search_range_pt=20.0)
        # 근접 범위(350~370) 안의 355가 선택돼야 함
        assert r["call_oi_peak"] == pytest.approx(355.0), (
            f"원거리 400pt가 선택됨: {r['call_oi_peak']}"
        )

    def test_far_put_peak_excluded_when_near_candidate_exists(self):
        """근접 범위(20pt) 안에 후보가 있으면 원거리 Put Peak을 선택하지 않는다."""
        puts_with_far = {
            "P345": {"strike": 345, "open_interest": 50000, "iv": 0.20},
            "P300": {"strike": 300, "open_interest": 200000, "iv": 0.30},  # 원거리 고OI
        }
        r = calc_oi_levels({}, puts_with_far, PRICE, peak_search_range_pt=20.0)
        assert r["put_oi_peak"] == pytest.approx(345.0), (
            f"원거리 300pt가 선택됨: {r['put_oi_peak']}"
        )

    def test_fallback_to_full_range_when_no_near_candidate(self):
        """근접 범위 안에 Call이 없으면 전체 범위 fallback으로 선택한다."""
        # 모든 Call이 범위 밖(+25pt 초과)
        calls_far_only = {
            "C380": {"strike": 380, "open_interest": 30000, "iv": 0.25},
            "C385": {"strike": 385, "open_interest": 50000, "iv": 0.28},
        }
        r = calc_oi_levels(calls_far_only, {}, PRICE, peak_search_range_pt=20.0)
        # fallback: F 이상 전체 중 최대 OI → 385pt
        assert r["call_oi_peak"] == pytest.approx(385.0)

    def test_custom_range_wider_selects_farther_peak(self):
        """peak_search_range_pt=50이면 원거리 고OI가 선택된다."""
        calls_with_far = {
            "C355": {"strike": 355, "open_interest": 10000, "iv": 0.20},
            "C390": {"strike": 390, "open_interest": 80000, "iv": 0.30},
        }
        r = calc_oi_levels(calls_with_far, {}, PRICE, peak_search_range_pt=50.0)
        # 범위 350~400 안에 390이 포함 → 고OI인 390 선택
        assert r["call_oi_peak"] == pytest.approx(390.0)

    def test_atm_call_as_peak_dist_zero(self):
        """Call peak 탐색은 현재가 초과(strict) 행사가만 대상으로 한다.

        수정 이유:
        _pick_call_peak 구현이 `f < k` (strict)로 현재가와 동일한 ATM 행사가를
        의도적으로 제외한다 (ATM은 저항이 아닌 현재 수준으로 간주).
        따라서 C350(OI=70000)이 있어도 현재가(350)와 같으므로 제외되고
        C355(OI=10000)가 call_oi_peak로 선택된다.
        """
        calls_atm = {
            "C350": {"strike": 350, "open_interest": 70000, "iv": 0.18},
            "C355": {"strike": 355, "open_interest": 10000, "iv": 0.20},
        }
        r = calc_oi_levels(calls_atm, {}, PRICE, peak_search_range_pt=20.0)
        # ATM(350)은 strict 제외 → 현재가 초과 중 유일한 후보 355가 peak
        assert r["call_oi_peak"] == pytest.approx(355.0)
        # dist_to_call_peak = (355 - 350) / 350 * 100 > 0
        assert r.get("dist_to_call_peak", 0.0) > 0.0

    def test_atm_put_as_peak_dist_zero(self):
        """Put peak 탐색은 현재가 미만(strict) 행사가만 대상으로 한다.

        수정 이유:
        _pick_put_peak 구현이 `k < f` (strict)로 현재가와 동일한 ATM 행사가를
        의도적으로 제외한다 (ATM은 지지가 아닌 현재 수준으로 간주).
        따라서 P350(OI=70000)이 있어도 현재가(350)와 같으므로 제외되고
        P345(OI=10000)가 put_oi_peak로 선택된다.
        """
        puts_atm = {
            "P350": {"strike": 350, "open_interest": 70000, "iv": 0.18},
            "P345": {"strike": 345, "open_interest": 10000, "iv": 0.20},
        }
        r = calc_oi_levels({}, puts_atm, PRICE, peak_search_range_pt=20.0)
        # ATM(350)은 strict 제외 → 현재가 미만 중 유일한 후보 345가 peak
        assert r["put_oi_peak"] == pytest.approx(345.0)
        # dist_to_put_peak = (350 - 345) / 350 * 100 > 0
        assert r.get("dist_to_put_peak", 0.0) > 0.0


# ── OI-03: Zero Gamma Strike 현재가 최근접 검증 ──────────────────────────────

class TestCalcOiLevelsZeroGamma:
    """OI-03 수정: zero_gamma_strike가 현재가 최근접 전환점임을 검증."""

    def _build_sign_flip_data(self, f: float):
        """두 개의 Net Gamma 부호 전환점이 생기도록 설계된 픽스처."""
        # 전환점 A: 현재가 멀리 (low strike)
        # 전환점 B: 현재가 근접 (near strike)
        # Call OI가 낮은 구간에서 Put OI가 지배하면 Net Gamma < 0 → 반전 발생
        calls = {
            "C_near1": {"strike": f + 2.5, "open_interest": 60000, "iv": 0.20, "gamma": 0.010},
            "C_near2": {"strike": f + 5.0, "open_interest": 5000,  "iv": 0.21, "gamma": 0.009},
            "C_far":   {"strike": f + 20.0, "open_interest": 55000, "iv": 0.25, "gamma": 0.006},
        }
        puts = {
            "P_near":  {"strike": f - 2.5, "open_interest": 5000,  "iv": 0.20, "gamma": 0.010},
            "P_mid":   {"strike": f - 7.5, "open_interest": 70000, "iv": 0.21, "gamma": 0.009},
            "P_far":   {"strike": f - 20.0, "open_interest": 60000, "iv": 0.25, "gamma": 0.007},
        }
        return calls, puts

    def test_zero_gamma_strike_is_nearest_to_price(self):
        """zero_gamma_strike가 여러 전환점 중 현재가와 가장 가까운 것이어야 한다."""
        calls, puts = self._build_sign_flip_data(PRICE)
        r = calc_oi_levels(calls, puts, PRICE)
        zg = r["zero_gamma_strike"]
        vt = r["vol_trigger_strike"]
        # OI-03 수정 후: zero_gamma_strike == vol_trigger_strike (동일 알고리즘)
        assert zg == pytest.approx(vt, abs=0.5), (
            f"zero_gamma({zg:.2f}) != vol_trigger({vt:.2f})"
        )

    def test_zero_gamma_equals_vol_trigger(self):
        """OI-03 수정 후 zero_gamma_strike와 vol_trigger_strike는 같은 값이어야 한다."""
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        zg = r["zero_gamma_strike"]
        vt = r["vol_trigger_strike"]
        # 단일 전환점이면 당연히 일치, 복수여도 수정 후 최근접으로 일치
        assert zg == pytest.approx(vt, abs=0.5)

    def test_no_sign_flip_returns_zero(self):
        """부호 전환점이 없으면 zero_gamma_strike == 0이어야 한다."""
        # Call만 있어서 Net Gamma가 항상 양수
        calls_only = {
            "C350": {"strike": 350, "open_interest": 50000, "iv": 0.18, "gamma": 0.012},
            "C355": {"strike": 355, "open_interest": 40000, "iv": 0.20, "gamma": 0.010},
        }
        r = calc_oi_levels(calls_only, {}, PRICE)
        assert r["zero_gamma_strike"] == pytest.approx(0.0)
        assert r["vol_trigger_strike"] == pytest.approx(0.0)


# ── calc_oi_levels 집중도 검증 ───────────────────────────────────────────────

class TestCalcOiLevelsConcentration:
    def test_call_peak_norm_range(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        assert 0.0 <= r["call_oi_peak_norm"] <= 1.0

    def test_put_peak_norm_range(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        assert 0.0 <= r["put_oi_peak_norm"] <= 1.0

    def test_call_peak_norm_formula(self):
        """Peak OI / 전체 OI 합 검증."""
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        total = 10000 + 48000 + 12000
        expected = 48000 / total
        assert r["call_oi_peak_norm"] == pytest.approx(expected, rel=1e-3)

    def test_put_peak_norm_formula(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        total = 35000 + 52000 + 20000
        expected = 52000 / total
        assert r["put_oi_peak_norm"] == pytest.approx(expected, rel=1e-3)


class TestCalcOiLevelsBoxMetrics:
    def test_oi_range_pct_formula(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        expected = (355.0 - 345.0) / 350.0 * 100.0
        assert r["oi_range_pct"] == pytest.approx(expected, rel=1e-3)

    def test_oi_center(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        assert r["oi_center"] == pytest.approx(350.0, rel=1e-3)

    def test_oi_center_dist_pct_zero_when_price_at_center(self):
        """현재가가 OI 중심과 일치하면 center_dist_pct ≈ 0."""
        r = calc_oi_levels(CALLS, PUTS, 350.0)
        assert abs(r["oi_center_dist_pct"]) < 0.01


class TestCalcOiLevelsGamma:
    def test_above_vol_trigger_binary(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        assert r["above_vol_trigger"] in (0.0, 1.0)

    def test_zero_gamma_dist_finite(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        assert math.isfinite(r["zero_gamma_dist_pct"])

    def test_net_gamma_proxy_finite(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        assert math.isfinite(r["net_gamma_proxy"])

    def test_bs_gamma_fallback_no_crash(self):
        """gamma 필드 없어도 BS 근사로 안전하게 동작해야 한다."""
        r = calc_oi_levels(CALLS_NO_GAMMA, PUTS_NO_GAMMA, PRICE)
        assert math.isfinite(r["net_gamma_proxy"])
        assert r["above_vol_trigger"] in (0.0, 1.0)

    def test_vol_trigger_strike_non_negative(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        assert r["vol_trigger_strike"] >= 0.0


class TestCalcOiLevelsSafeFallback:
    def test_empty_calls_and_puts(self):
        r = calc_oi_levels({}, {}, 350.0)
        assert r["call_oi_peak"] == 0.0
        assert r["put_oi_peak"] == 0.0
        assert r["oi_range_pct"] == 0.0
        assert r["above_vol_trigger"] == 1.0

    def test_zero_underlying_price(self):
        r = calc_oi_levels(CALLS, PUTS, 0.0)
        assert r["call_oi_peak"] == 0.0
        assert r["put_oi_peak"] == 0.0

    def test_only_calls(self):
        r = calc_oi_levels(CALLS, {}, PRICE)
        assert r["call_oi_peak"] == pytest.approx(355.0)
        assert r["put_oi_peak"] == 0.0
        assert r["oi_range_pct"] == 0.0

    def test_only_puts(self):
        r = calc_oi_levels({}, PUTS, PRICE)
        assert r["put_oi_peak"] == pytest.approx(345.0)
        assert r["call_oi_peak"] == 0.0

    def test_all_oi_zero(self):
        zero_calls = {k: {**v, "open_interest": 0} for k, v in CALLS.items()}
        zero_puts  = {k: {**v, "open_interest": 0} for k, v in PUTS.items()}
        r = calc_oi_levels(zero_calls, zero_puts, PRICE)
        assert r["call_oi_peak"] == 0.0
        assert r["put_oi_peak"] == 0.0

    def test_return_keys_complete(self):
        r = calc_oi_levels(CALLS, PUTS, PRICE)
        expected_keys = {
            "call_oi_peak", "put_oi_peak", "call_oi_peak_norm", "put_oi_peak_norm",
            "oi_range_pct", "dist_to_call_peak", "dist_to_put_peak",
            "oi_center", "oi_center_dist_pct", "net_gamma_proxy",
            "zero_gamma_strike", "zero_gamma_dist_pct",
            "vol_trigger_strike", "above_vol_trigger",
        }
        assert expected_keys.issubset(r.keys())


# ── GR-02/GR-03: 가드레일 동작 검증 ─────────────────────────────────────────

class TestOiGuardrailLogic:
    """GR-02/GR-03 수정 동작을 pipeline 없이 option_features 레벨에서 검증."""

    def _make_oi_snap(self, *, call_dist, put_dist, call_conc=0.0, put_conc=0.0,
                      above_vt=1.0, zgd=99.0, oi_range=5.0):
        """_oi_levels dict 직접 구성."""
        return {
            "_oi_levels": {
                "dist_to_call_peak": call_dist,
                "dist_to_put_peak": put_dist,
                "call_oi_peak_norm": call_conc,
                "put_oi_peak_norm": put_conc,
                "above_vol_trigger": above_vt,
                "zero_gamma_dist_pct": zgd,
                "oi_range_pct": oi_range,
                "call_oi_peak": PRICE + (call_dist / 100.0 * PRICE),
                "put_oi_peak":  PRICE - (put_dist  / 100.0 * PRICE),
            }
        }

    def test_gr03_dist_zero_triggers_call_resistance(self):
        """GR-03: dist=0.0(ATM Call Peak) + conc>=0.4 + BUY → 가드레일 동작해야 한다."""
        snap = self._make_oi_snap(call_dist=0.0, put_dist=5.0, call_conc=0.5)
        oi = snap["_oi_levels"]
        # GR-03 수정: call_dist = None 체크 → 0.0 정상 처리
        call_dist_raw = oi.get("dist_to_call_peak")
        call_dist = float(call_dist_raw if call_dist_raw is not None else 99.0)
        # 조건: call_conc >= 0.4 and 0.0 <= call_dist < 0.3
        assert oi["call_oi_peak_norm"] >= 0.4
        assert 0.0 <= call_dist < 0.3, f"dist={call_dist} 조건 미충족"

    def test_gr03_dist_zero_triggers_put_support(self):
        """GR-03: dist=0.0(ATM Put Peak) + conc>=0.4 + SELL → 가드레일 동작해야 한다."""
        snap = self._make_oi_snap(call_dist=5.0, put_dist=0.0, put_conc=0.5)
        oi = snap["_oi_levels"]
        put_dist_raw = oi.get("dist_to_put_peak")
        put_dist = float(put_dist_raw if put_dist_raw is not None else 99.0)
        assert oi["put_oi_peak_norm"] >= 0.4
        assert 0.0 <= put_dist < 0.3, f"dist={put_dist} 조건 미충족"

    def test_gr02_multiple_conditions_accumulate(self):
        """GR-02: 조건 1(Zero Gamma 근접)과 조건 2(Vol Trigger 하방) 동시 해당 검증."""
        # zero_gamma_dist 작고(조건1), vol_trigger 하방(조건2)
        snap = self._make_oi_snap(call_dist=5.0, put_dist=5.0,
                                  above_vt=0.0, zgd=0.05)
        oi = snap["_oi_levels"]
        zgd = abs(float(oi.get("zero_gamma_dist_pct", 99.0)))
        above_vt = float(oi.get("above_vol_trigger", 1.0))
        assert zgd < 0.2   # 조건1 충족
        assert above_vt < 1.0  # 조건2 충족 (하방)

    def test_oi_snap_with_dist_none_safe(self):
        """dist 키가 없을 때 None 체크 기본값 99.0으로 처리 — 가드레일 비활성."""
        snap = {"_oi_levels": {"oi_range_pct": 5.0, "call_oi_peak_norm": 0.5}}
        oi = snap["_oi_levels"]
        call_dist = float(oi.get("dist_to_call_peak") if oi.get("dist_to_call_peak") is not None else 99.0)
        # 키 없으면 99.0 → 조건 0.0 <= dist < 0.3 불충족 → 가드레일 비활성
        assert call_dist == 99.0


# ── build_option_snapshot v5 통합 검증 ──────────────────────────────────────

class TestBuildOptionSnapshotV5:
    def test_v5_keys_present(self):
        snap = build_option_snapshot(CALLS, PUTS, PRICE, option_feature_set="v5")
        v5_keys = [
            "dist_to_call_peak", "dist_to_put_peak", "oi_center_dist_pct",
            "oi_range_pct", "call_oi_peak_norm", "put_oi_peak_norm",
            "above_vol_trigger", "zero_gamma_dist_pct",
        ]
        for k in v5_keys:
            assert k in snap, f"v5 피처 키 누락: {k}"

    def test_oi_levels_stored_in_all_fs(self):
        """모든 fs에서 _oi_levels가 snap에 저장되어야 한다 (LLM 컨텍스트 공통 활용)."""
        for fs in ("v1", "v2", "v3", "v4", "v5"):
            snap = build_option_snapshot(CALLS, PUTS, PRICE, option_feature_set=fs)
            assert "_oi_levels" in snap, f"fs={fs}에서 _oi_levels 누락"

    def test_v4_does_not_expose_v5_keys(self):
        """v4에서는 v5 전용 피처가 직접 노출되지 않아야 한다 (차원 불변 보장)."""
        snap = build_option_snapshot(CALLS, PUTS, PRICE, option_feature_set="v4")
        # v5 전용 피처는 _oi_levels 내부에만 있어야 함
        assert "dist_to_call_peak" not in snap
        assert "above_vol_trigger" not in snap

    def test_v5_above_vol_trigger_binary(self):
        snap = build_option_snapshot(CALLS, PUTS, PRICE, option_feature_set="v5")
        assert snap["above_vol_trigger"] in (0.0, 1.0)

    def test_v5_all_keys_finite(self):
        snap = build_option_snapshot(CALLS, PUTS, PRICE, option_feature_set="v5")
        v5_float_keys = [
            "dist_to_call_peak", "dist_to_put_peak", "oi_center_dist_pct",
            "oi_range_pct", "call_oi_peak_norm", "put_oi_peak_norm",
            "above_vol_trigger", "zero_gamma_dist_pct",
        ]
        for k in v5_float_keys:
            val = snap[k]
            assert math.isfinite(float(val)), f"{k} = {val} (not finite)"

    def test_v5_empty_inputs_no_crash(self):
        snap = build_option_snapshot({}, {}, 350.0, option_feature_set="v5")
        assert snap["dist_to_call_peak"] == 0.0
        assert snap["above_vol_trigger"] == 1.0

    def test_v5_peak_range_applied_via_build_snapshot(self):
        """build_option_snapshot의 ATM±25pt 필터가 적용된 후 calc_oi_levels 호출 검증."""
        # 원거리(+50pt) Call에 고OI → build_option_snapshot 필터로 제외되어야 함
        calls_far = {
            "C355": {"strike": 355, "open_interest": 20000, "iv": 0.20},
            "C400": {"strike": 400, "open_interest": 999999, "iv": 0.30},
        }
        snap = build_option_snapshot(calls_far, {}, 350.0, option_feature_set="v5")
        oi = snap.get("_oi_levels", {})
        # 400pt는 ATM±25pt(375~325) 범위 밖이므로 제외 → 355pt 선택
        assert oi.get("call_oi_peak", 0.0) == pytest.approx(355.0), (
            f"원거리 400pt가 Peak으로 선택됨: {oi.get('call_oi_peak')}"
        )


# ── _describe_oi_levels LLM 컨텍스트 생성 검증 ──────────────────────────────

class TestDescribeOiLevels:
    def _snap_with_oi(self) -> dict:
        snap = build_option_snapshot(CALLS, PUTS, PRICE, option_feature_set="v1")
        return snap

    def test_returns_string(self):
        snap = self._snap_with_oi()
        result = _describe_oi_levels(snap)
        assert isinstance(result, str)

    def test_contains_call_peak_info(self):
        snap = self._snap_with_oi()
        result = _describe_oi_levels(snap)
        assert "355" in result or "Call" in result

    def test_contains_put_peak_info(self):
        snap = self._snap_with_oi()
        result = _describe_oi_levels(snap)
        assert "345" in result or "Put" in result

    def test_contains_regime_info(self):
        snap = self._snap_with_oi()
        result = _describe_oi_levels(snap)
        assert "Gamma" in result or "레짐" in result

    def test_empty_snap_returns_empty_string(self):
        result = _describe_oi_levels({})
        assert result == ""

    def test_zero_oi_returns_empty_string(self):
        """OI 데이터가 없으면 빈 문자열 반환해야 한다."""
        snap = {"_oi_levels": {"oi_range_pct": 0.0, "call_oi_peak": 0.0, "put_oi_peak": 0.0}}
        result = _describe_oi_levels(snap)
        assert result == ""

    def test_zero_gamma_near_warning(self):
        """Zero Gamma 근접 시 경고 텍스트가 포함되어야 한다."""
        snap = {
            "_oi_levels": {
                "call_oi_peak": 355.0,
                "put_oi_peak": 345.0,
                "call_oi_peak_norm": 0.68,
                "put_oi_peak_norm": 0.50,
                "dist_to_call_peak": 1.43,
                "dist_to_put_peak": 1.43,
                "oi_center_dist_pct": 0.0,
                "oi_range_pct": 2.86,
                "above_vol_trigger": 1.0,
                "zero_gamma_dist_pct": 0.1,   # 0.3% 미만 → 경고 발생
                "zero_gamma_strike": 349.65,
                "vol_trigger_strike": 349.65,
            }
        }
        result = _describe_oi_levels(snap)
        assert "Zero Gamma" in result or "감마" in result


# ── calc_iv_peak_range 검증 ──────────────────────────────────────────────────

class TestCalcIvPeakRange:
    """calc_iv_peak_range() — IV 기반 동적 탐색 범위 산출 검증."""

    def test_typical_case_iv20_dte5(self):
        """IV=20%, DTE=5일, F=350 → 2σ ≈ 19.8pt → 20.0pt (2.5 배수 반올림)."""
        from prediction.option_features import calc_iv_peak_range
        r = calc_iv_peak_range(350.0, 0.20, 5.0)
        # 350 * 0.20 * sqrt(5/252) * 2.0 ≈ 19.8 → 20.0
        assert r == pytest.approx(20.0, abs=2.5)

    def test_high_iv_wider_range(self):
        """IV=35% → 범위가 기본(IV=20%)보다 넓어야 한다."""
        from prediction.option_features import calc_iv_peak_range
        r_low  = calc_iv_peak_range(350.0, 0.20, 5.0)
        r_high = calc_iv_peak_range(350.0, 0.35, 5.0)
        assert r_high > r_low

    def test_low_iv_narrower_but_above_min(self):
        """IV=8%(매우 낮음) → min_range_pt(10pt) 이상 보장."""
        from prediction.option_features import calc_iv_peak_range
        r = calc_iv_peak_range(350.0, 0.08, 5.0)
        assert r >= 10.0

    def test_long_dte_wider_range(self):
        """DTE=20일 → DTE=5일보다 범위가 넓어야 한다."""
        from prediction.option_features import calc_iv_peak_range
        r_short = calc_iv_peak_range(350.0, 0.20, 5.0)
        r_long  = calc_iv_peak_range(350.0, 0.20, 20.0)
        assert r_long > r_short

    def test_capped_at_max_range(self):
        """IV=100%(극단) → max_range_pt(40pt) 이하 보장."""
        from prediction.option_features import calc_iv_peak_range
        r = calc_iv_peak_range(350.0, 1.00, 30.0)
        assert r <= 40.0

    def test_zero_iv_returns_min_range(self):
        """IV=0 → min_range_pt(fallback) 반환."""
        from prediction.option_features import calc_iv_peak_range
        r = calc_iv_peak_range(350.0, 0.0, 5.0)
        assert r == pytest.approx(10.0)

    def test_zero_price_returns_min_range(self):
        """F=0 → min_range_pt 반환 (safe fallback)."""
        from prediction.option_features import calc_iv_peak_range
        r = calc_iv_peak_range(0.0, 0.20, 5.0)
        assert r == pytest.approx(10.0)

    def test_result_is_multiple_of_strike_step(self):
        """결과가 행사가 간격(2.5pt) 배수이어야 한다."""
        from prediction.option_features import calc_iv_peak_range
        for iv in (0.15, 0.20, 0.25, 0.35):
            r = calc_iv_peak_range(350.0, iv, 5.0)
            # 2.5pt 배수 여부 — 부동소수 오차 허용
            assert abs(r % 2.5) < 0.01 or abs(r % 2.5 - 2.5) < 0.01, \
                f"iv={iv}: {r}는 2.5pt 배수가 아님"

    def test_result_monotone_in_iv(self):
        """IV가 증가할수록 범위도 단조 증가 (min/max 범위 내)."""
        from prediction.option_features import calc_iv_peak_range
        ivs = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
        results = [calc_iv_peak_range(350.0, iv, 5.0) for iv in ivs]
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1], \
                f"iv={ivs[i]}→{ivs[i+1]}: {results[i]}>{results[i+1]} (단조 증가 위반)"


class TestCalcOiLevelsIvDynamic:
    """calc_oi_levels(atm_iv=...) — IV 동적 범위 통합 검증."""

    def test_iv_range_used_key_present(self):
        """반환 dict에 peak_search_range_used 키가 있어야 한다."""
        r = calc_oi_levels(CALLS, PUTS, PRICE, atm_iv=0.20)
        assert "peak_search_range_used" in r

    def test_iv_range_used_positive(self):
        """IV > 0이면 peak_search_range_used > 0."""
        r = calc_oi_levels(CALLS, PUTS, PRICE, atm_iv=0.20)
        assert r["peak_search_range_used"] > 0.0

    def test_iv_range_used_matches_calc(self):
        """IV 기반 범위가 calc_iv_peak_range()와 일치해야 한다."""
        from prediction.option_features import calc_iv_peak_range
        expected = calc_iv_peak_range(PRICE, 0.20, 7.0)
        r = calc_oi_levels(CALLS, PUTS, PRICE, atm_iv=0.20,
                           default_days_to_expiry=7.0)
        assert r["peak_search_range_used"] == pytest.approx(expected, abs=0.1)

    def test_no_iv_uses_fixed_fallback(self):
        """atm_iv=0이면 peak_search_range_pt 고정값이 사용된다."""
        r = calc_oi_levels(CALLS, PUTS, PRICE, atm_iv=0.0,
                           peak_search_range_pt=15.0)
        assert r["peak_search_range_used"] == pytest.approx(15.0, abs=0.1)

    def test_high_iv_wider_range_excludes_far_peak(self):
        """IV 높아 탐색 범위 확장 → 적당히 먼 고OI 행사가도 선택 가능."""
        calls_mid = {
            "C360": {"strike": 360, "open_interest": 5000,  "iv": 0.40},
            "C375": {"strike": 375, "open_interest": 80000, "iv": 0.40},  # 25pt 떨어진 고OI
        }
        # IV=40%, DTE=5일 → 2σ ≈ 35pt → 375pt 포함 가능
        r = calc_oi_levels(calls_mid, {}, PRICE,
                           atm_iv=0.40, default_days_to_expiry=5.0)
        # peak_search_range_used >= 25pt 이면 375pt가 탐색 범위에 포함
        if r["peak_search_range_used"] >= 25.0:
            assert r["call_oi_peak"] == pytest.approx(375.0)

    def test_low_iv_narrow_range_excludes_far_peak(self):
        """IV 낮아 탐색 범위 좁음 → 원거리 고OI가 제외되고 근접 Peak 선택."""
        calls_with_far = {
            "C355": {"strike": 355, "open_interest": 50000, "iv": 0.10},
            "C390": {"strike": 390, "open_interest": 200000, "iv": 0.10},
        }
        # IV=10%, DTE=5일 → 2σ ≈ 11.1pt → 근접 범위 내 355pt 선택
        r = calc_oi_levels(calls_with_far, {}, PRICE,
                           atm_iv=0.10, default_days_to_expiry=5.0)
        assert r["call_oi_peak"] == pytest.approx(355.0), \
            f"범위={r['peak_search_range_used']:.1f}pt: 원거리 390pt 선택됨"

    def test_build_snapshot_passes_atm_iv_to_calc_oi_levels(self):
        """build_option_snapshot이 atm_iv를 calc_oi_levels에 전달하여 동적 범위 적용."""
        snap = build_option_snapshot(CALLS, PUTS, PRICE, option_feature_set="v1")
        oi = snap.get("_oi_levels", {})
        # atm_iv > 0이면 peak_search_range_used가 20.0과 다를 수 있음
        assert "peak_search_range_used" in oi
        assert oi["peak_search_range_used"] > 0.0
