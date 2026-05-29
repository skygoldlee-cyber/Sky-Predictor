"""OI(미결제약정) 기반 지지저항 및 진폭 계산.

calc_expected_amplitude  : 당일 기대 진폭(pts) 계산
calc_oi_levels           : Call/Put OI Peak, Zero Gamma, Vol Trigger 레벨
calc_otm_premium_change  : OTM 프리미엄 변화율 (방향성 흐름 감지)
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import math
import numpy as np

from ..option_core import _find_atm_strike, _bs_gamma_proxy, calc_iv_peak_range

def calc_expected_amplitude(
    underlying_price: float,
    atm_iv: float,
    days_to_expiry: float,
    *,
    session_high: float = 0.0,
    session_low: float = 0.0,
    session_open: float = 0.0,
    sigma_multiplier: float = 1.0,
    min_amplitude_pt: float = 3.0,
    max_amplitude_pt: float = 50.0,
    oi_levels: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """당일 선물 진폭을 IV 기반 + OI 앵커 혼합으로 예측하고 장중 실현 진폭과 비교한다.

    공식:
        IV 기반:  iv_amp  = F * ATM_IV * sqrt(1/252) * sigma_multiplier * 2
                            (DTE는 장중 당일 예측이므로 항상 1일 고정)
        OI 기반:  oi_amp  = call_oi_peak - put_oi_peak
                            (OI 집중도 기반 가중치 적용, 박스 과대 시 신뢰도 감소)
        혼합:     expected = oi_amp * oi_weight + iv_amp * (1 - oi_weight)

    OI 가중치 산출:
        oi_weight = min(1.0, (call_oi_peak_norm + put_oi_peak_norm) / 2)
        OI 박스폭이 현재가 대비 5% 초과 시 oi_weight *= 0.3 (비현실적 박스 페널티)
        OI 없으면 oi_weight = 0 → IV 단독 사용

    Args:
        underlying_price:  현재 선물가(F).
        atm_iv:            ATM 내재변동성(소수. 0.20 = 20%). 0이면 IV항 0 처리.
        days_to_expiry:    잔존기간(일). 장중 당일 진폭 예측이므로 내부에서 1.0 고정.
                           파라미터는 하위 호환성을 위해 유지하나 실제로는 사용하지 않음.
        session_high:      장중 누적 고가. 0이면 실현 진폭 계산 생략.
        session_low:       장중 누적 저가. 0이면 실현 진폭 계산 생략.
        session_open:      장 시작 시가. 0이면 open_dist_pct 계산 생략.
        sigma_multiplier:  IV 항 배율. 기본 1.0σ (단방향). 양방향은 내부 ×2.
        min_amplitude_pt:  반환 최솟값(pt). 기본 3.0pt.
        max_amplitude_pt:  반환 최댓값(pt). 기본 50.0pt.
        oi_levels:         calc_oi_levels() 반환 dict. None이면 OI 항 비활성.

    Returns:
        dict with keys:
            expected_amplitude_pt   : 예상 진폭(pt, H-L 기준). 0 = 계산 불가.
            realized_hl_range_pt    : 장중 실현 진폭(pt, H-L). 0 = 데이터 없음.
            amplitude_exhaustion    : 실현/예상 비율 [0, ∞). 1.0 초과 = 예상 범위 소진.
            remaining_amplitude_pt  : 남은 예상 진폭(pt). max(0, expected - realized).
            open_dist_pct           : 현재가 vs 시가 거리(%). 양수=시가 위, 음수=시가 아래.
            session_open            : 사용된 시가값(pt). 0 = 없음.
            oi_box_pt               : OI 박스폭(pt). call_peak - put_peak. 0 = OI 없음.
            oi_vs_amplitude         : OI 박스폭 / 예상 진폭 비율. 1.0 = 동일.
            call_dist_pt            : 현재가 → Call OI Peak 거리(pt). 양수 = 위.
            put_dist_pt             : 현재가 → Put OI Peak 거리(pt). 양수 = 아래.
            _oi_weight              : OI 항에 적용된 실제 가중치 [0, 1]. 진단용.
            _iv_amplitude_pt        : IV 단독 예상 진폭(pt). 혼합 전 원본값. 진단용.
            _oi_amplitude_pt        : OI 박스폭 기반 진폭(pt). 혼합 전 원본값. 진단용.

    Notes:
        - DTE 고정: 장중 당일 예측이 목적이므로 DTE=1이 항상 올바른 기준이다.
          DTE를 만기까지 잔존일(수십일)로 사용하면 sqrt()에 의해 진폭이 크게 과대평가되어
          max_amplitude_pt clamp(50pt)에 붙는 버그가 발생한다.
        - OI 혼합: OI 지지·저항이 유의미할 때(집중도 높음) OI 박스폭을 진폭 앵커로 활용.
          장의 등락에 따라 OI 레벨이 달라지면 예상 진폭도 동적으로 변한다.
        - amplitude_exhaustion > 0.8: 추가 움직임 여력 제한적.
    """
    import math as _math

    empty: Dict[str, float] = {
        "expected_amplitude_pt":    0.0,
        "realized_hl_range_pt":     0.0,
        "amplitude_exhaustion":     0.0,
        "remaining_amplitude_pt":   0.0,
        # [FIX-AMP-4] 방향별 잔여 진폭 (시가 중심 대칭 가정)
        "upside_remaining_pt":      0.0,   # 위쪽 잔여: predicted_high - max(current, session_high)
        "downside_remaining_pt":    0.0,   # 아래쪽 잔여: min(current, session_low) - predicted_low
        "open_dist_pct":            0.0,
        "session_open":             0.0,
        "oi_box_pt":                0.0,
        "oi_vs_amplitude":          0.0,
        "call_dist_pt":             0.0,
        "put_dist_pt":              0.0,
        "_oi_weight":               0.0,
        "_iv_amplitude_pt":         0.0,
        "_oi_amplitude_pt":       0.0,
    }

    try:
        F   = float(underlying_price or 0.0)
        iv  = float(atm_iv or 0.0)
        s_h = float(session_high or 0.0)
        s_l = float(session_low or 0.0)
        s_o = float(session_open or 0.0)

        if F <= 0.0:
            return empty

        # ── 1. IV 기반 진폭 (DTE=1 고정) ────────────────────────────────────
        # 장중 당일 진폭 예측이 목적이므로 DTE는 항상 1일을 사용한다.
        # DTE를 만기까지 잔존일(수십일)로 쓰면 sqrt() 과대평가 → clamp 50pt 고착 버그.
        iv_amp = 0.0
        if iv > 0.0:
            try:
                sigma_move = F * iv * _math.sqrt(1.0 / 252.0) * float(sigma_multiplier)
                iv_raw     = sigma_move * 2.0   # 양방향(H-L) 기대 범위
                iv_amp     = max(float(min_amplitude_pt),
                                 min(float(max_amplitude_pt), iv_raw))
                if not _math.isfinite(iv_amp):
                    iv_amp = 0.0
            except Exception:
                iv_amp = 0.0

        # ── 2. OI 앵커 기반 진폭 ────────────────────────────────────────────
        # call_oi_peak(저항) - put_oi_peak(지지) = OI 박스폭을 진폭 앵커로 활용.
        # OI 집중도(peak_norm) 평균으로 가중치를 결정한다.
        oi_amp    = 0.0
        oi_weight = 0.0
        oi_box_pt = 0.0

        try:
            _oi = oi_levels if isinstance(oi_levels, dict) else {}
            _call_peak      = float(_oi.get("call_oi_peak")      or 0.0)
            _put_peak       = float(_oi.get("put_oi_peak")       or 0.0)
            _call_norm      = float(_oi.get("call_oi_peak_norm") or 0.0)
            _put_norm       = float(_oi.get("put_oi_peak_norm")  or 0.0)
            _oi_range_pct   = float(_oi.get("oi_range_pct") or 0.0)

            if _call_peak > 0.0 and _put_peak > 0.0 and _call_peak > _put_peak:
                oi_box_pt = float(_call_peak - _put_peak)

                # OI 가중치: 집중도 평균 (0~1)
                _raw_weight = (_call_norm + _put_norm) / 2.0
                _raw_weight = min(1.0, max(0.0, _raw_weight))

                # OI 박스가 현재가 대비 과도하게 넓으면(%) IV 쪽 비중 확대
                if _oi_range_pct > 7.0:
                    _raw_weight *= 0.4
                elif _oi_range_pct > 4.0:
                    _raw_weight *= 0.7

                # OI 박스폭이 현재가 대비 5% 초과 시 신뢰도 페널티
                # (원거리 행사가 OI가 잡힌 경우 과대 진폭 방지)
                if F > 0.0 and oi_box_pt > F * 0.05:
                    _raw_weight *= 0.3

                if _math.isfinite(_raw_weight) and _raw_weight > 0.0:
                    oi_amp    = oi_box_pt
                    oi_weight = _raw_weight
        except Exception:
            oi_amp    = 0.0
            oi_weight = 0.0

        # ── 3. IV + OI 혼합 ──────────────────────────────────────────────────
        # OI가 유효(weight > 0)하면 가중 혼합, 없으면 IV 단독 사용.
        expected_pt = 0.0
        if oi_amp > 0.0 and oi_weight > 0.0 and iv_amp > 0.0:
            blended = oi_amp * oi_weight + iv_amp * (1.0 - oi_weight)
            expected_pt = max(float(min_amplitude_pt),
                              min(float(max_amplitude_pt), blended))
        elif iv_amp > 0.0:
            expected_pt = iv_amp
        # iv도 oi도 없으면 expected_pt = 0.0 유지

        if not _math.isfinite(expected_pt):
            expected_pt = 0.0

        # ── 4. 장중 실현 진폭 ───────────────────────────────────────────────
        realized_pt = 0.0
        if s_h > 0.0 and s_l > 0.0 and s_h >= s_l:
            try:
                realized_pt = float(s_h - s_l)
            except Exception:
                realized_pt = 0.0

        # ── 5. 소진 비율 / 잔여 진폭 ────────────────────────────────────────
        exhaustion = 0.0
        remaining  = 0.0
        if expected_pt > 0.0:
            try:
                exhaustion = float(realized_pt / expected_pt)
                if not _math.isfinite(exhaustion):
                    exhaustion = 0.0
                exhaustion = max(0.0, exhaustion)
            except Exception:
                exhaustion = 0.0
            remaining = float(max(0.0, expected_pt - realized_pt))

        # ── 5b. 방향별 잔여 진폭 ────────────────────────────────────────────
        # [FIX-AMP-4] remaining_amplitude_pt(H-L 기준 총 잔여)는 방향 정보가 없다.
        # 시가를 중심으로 상하 대칭 분포를 가정하여 위/아래 방향별 잔여를 계산한다.
        #
        # 공식 (s_o > 0 인 경우):
        #   predicted_high = s_o + expected_pt / 2
        #   predicted_low  = s_o - expected_pt / 2
        #   upside_remaining   = max(0, predicted_high - max(F, s_h))
        #     = 예측 고가까지 아직 닿지 않은 위쪽 여력
        #   downside_remaining = max(0, min(F, s_l) - predicted_low)
        #     = 예측 저가까지 아직 닿지 않은 아래쪽 여력
        #
        # s_o == 0인 경우 현재가(F) 기준으로 symmetric fallback.
        upside_remaining   = 0.0
        downside_remaining = 0.0
        try:
            if expected_pt > 0.0 and F > 0.0:
                _half = expected_pt / 2.0
                if s_o > 0.0:
                    _pred_hi = s_o + _half
                    _pred_lo = s_o - _half
                else:
                    # 시가 없음 → 현재가 기준
                    _pred_hi = F + _half
                    _pred_lo = F - _half

                # 이미 도달한 최고가/최저가를 반영 (s_h, s_l이 없으면 F로 대체)
                _cur_hi = s_h if s_h > 0.0 else F
                _cur_lo = s_l if s_l > 0.0 else F

                upside_remaining   = float(max(0.0, _pred_hi - max(F, _cur_hi)))
                downside_remaining = float(max(0.0, min(F, _cur_lo) - _pred_lo))

                if not _math.isfinite(upside_remaining):
                    upside_remaining = 0.0
                if not _math.isfinite(downside_remaining):
                    downside_remaining = 0.0
        except Exception:
            upside_remaining   = 0.0
            downside_remaining = 0.0

        # ── 6. 현재가 vs 시가 거리 ──────────────────────────────────────────
        open_dist_pct = 0.0
        if s_o > 0.0 and F > 0.0:
            try:
                open_dist_pct = float((F - s_o) / s_o * 100.0)
                if not _math.isfinite(open_dist_pct):
                    open_dist_pct = 0.0
            except Exception:
                open_dist_pct = 0.0

        # ── 7. OI dist 필드 (현재가 → peak 거리(pt)) ───────────────────────
        call_dist_pt_v = 0.0
        put_dist_pt_v  = 0.0
        oi_vs_amp      = 0.0
        try:
            _oi = oi_levels if isinstance(oi_levels, dict) else {}
            _call_peak     = float(_oi.get("call_oi_peak")    or 0.0)
            _put_peak      = float(_oi.get("put_oi_peak")     or 0.0)
            _call_dist_pct = float(_oi.get("dist_to_call_peak") or 0.0)
            _put_dist_pct  = float(_oi.get("dist_to_put_peak")  or 0.0)

            if _call_peak > 0.0 and _call_dist_pct > 0.0:
                _f_cur_c = _call_peak / (1.0 + _call_dist_pct / 100.0)
                call_dist_pt_v = float(_call_peak - _f_cur_c)
            if _put_peak > 0.0 and _put_dist_pct > 0.0:
                _f_cur_p = _put_peak / (1.0 - _put_dist_pct / 100.0)
                put_dist_pt_v = float(_f_cur_p - _put_peak)
            if expected_pt > 0.0 and oi_box_pt > 0.0 and _math.isfinite(oi_box_pt):
                oi_vs_amp = round(oi_box_pt / expected_pt, 4)
        except Exception:
            pass

        return {
            "expected_amplitude_pt":    round(expected_pt, 2),
            "realized_hl_range_pt":     round(realized_pt, 2),
            "amplitude_exhaustion":     round(exhaustion, 4),
            "remaining_amplitude_pt":   round(remaining, 2),
            # [FIX-AMP-4] 방향별 잔여 진폭
            "upside_remaining_pt":      round(upside_remaining, 2),
            "downside_remaining_pt":    round(downside_remaining, 2),
            "open_dist_pct":            round(open_dist_pct, 4),
            "session_open":             round(s_o, 2),
            "oi_box_pt":                round(oi_box_pt, 2),
            "oi_vs_amplitude":          round(oi_vs_amp, 4),
            "call_dist_pt":             round(call_dist_pt_v, 2),
            "put_dist_pt":              round(put_dist_pt_v, 2),
            "_oi_weight":               round(oi_weight, 4),
            "_iv_amplitude_pt":         round(iv_amp, 2),
            "_oi_amplitude_pt":         round(oi_amp, 2),
        }

    except Exception:
        return empty


def calc_oi_levels(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
    *,
    default_days_to_expiry: float = 7.0,
    contract_multiplier: float = 100.0,
    peak_search_range_pt: float = 20.0,
    atm_iv: float = 0.0,
    realized_hl_range_pt: float = 0.0,
    session_high: float = 0.0,
    session_low: float = 0.0,
) -> Dict[str, float]:
    """Strike별 OI 분포에서 지지·저항 레벨 및 딜러 감마 레벨을 산출한다.

    Args:
        peak_search_range_pt: OI-01 수정 — Call/Put Peak을 탐색할 ATM 기준 최대 거리(pt).
            atm_iv > 0이면 무시되고 IV 기반 동적 범위가 우선 사용된다.
            atm_iv == 0(IV 없음)일 때만 이 고정값 fallback이 사용된다. 기본 20pt.
        atm_iv: ATM 내재변동성(소수. 예: 0.20 = 20%).
            0보다 크면 calc_iv_peak_range()로 IV·DTE 기반 동적 탐색 범위를 산출한다.
            build_option_snapshot()이 calc_iv_skew() 결과(atm_call_iv)를 전달한다.
        realized_hl_range_pt: 당일 장중 실현 진폭(H-L, pt). (진단/호환용)
        session_high: 당일 장중 누적 고가(선물).
            >0이면 Call Peak 1차 탐색 상단을 session_high + 10pt 로 설정한다.
        session_low: 당일 장중 누적 저가(선물).
            >0이면 Put Peak 1차 탐색 하단을 session_low - 10pt 로 설정한다.

    Returns:
        call_oi_peak        : Call OI 최대 행사가 (상단 저항 대표).
        put_oi_peak         : Put OI 최대 행사가 (하단 지지 대표).
        call_oi_peak_norm   : Call OI Peak 집중도 [0, 1]. 클수록 저항 강도가 높음.
        put_oi_peak_norm    : Put OI Peak 집중도 [0, 1]. 클수록 지지 강도가 높음.
        oi_range_pct        : (call_peak - put_peak) / underlying * 100. OI 박스 폭(%).
        dist_to_call_peak   : (call_peak - F) / F * 100. 양수 = 저항이 현재가 위.
        dist_to_put_peak    : (F - put_peak) / F * 100. 양수 = 지지가 현재가 아래.
        oi_center           : (call_peak + put_peak) / 2. OI 박스 중심가격.
        oi_center_dist_pct  : (F - oi_center) / oi_center * 100. 중심 대비 거리(%).
        net_gamma_proxy     : Σ(call_oi*γ) - Σ(put_oi*γ). 양수 = 딜러 Long Gamma.
        zero_gamma_strike   : 현재가 최근접 Net Gamma 부호 전환 행사가. 없으면 0.0.
        zero_gamma_dist_pct : (F - zero_gamma_strike) / F * 100. 0 근처 = 반전 임박.
        vol_trigger_strike  : 현재가 근방 Dealer Gamma Long→Short 전환 레벨. 없으면 0.0.
        above_vol_trigger   : 1.0 = 선물이 vol_trigger 위(안정권), 0.0 = 아래(불안정권).
        peak_search_range_used : 실제 적용된 탐색 범위(pt). IV 기반이면 IV 연동값.

    Notes:
        - gamma 필드가 없으면 Black-Scholes 공식으로 근사 (calc_gex()와 동일 방식).
        - OI 데이터가 모두 0이면 all-zeros 반환 (장 시작 직후 safe fallback).
        - Best-effort: 모든 예외를 포착하여 zeros 반환.
        - OI-01 수정: Call/Put Peak을 ATM 기준 peak_search_range_pt 이내로 제한.
          원거리 행사가(예: 40pt 이상 이격)가 지지/저항으로 선정되는 것을 방지.
        - IV 동적 범위: atm_iv > 0이면 calc_iv_peak_range()로 탐색 반경 자동 산출.
          IV=20%, DTE=5일, F=350 기준 2σ ≈ 20pt. 변동성 급등 시 자동 확장.
        - OI-03 수정: zero_gamma_strike도 vol_trigger_strike와 동일하게
          zg_candidates 중 현재가 최근접 전환점을 사용. 두 값이 동일 레벨을 가리킨다.
    """
    empty: Dict[str, float] = {
        "call_oi_peak": 0.0,
        "call_oi_peak2": 0.0,           # 2nd Call OI Peak 행사가
        "put_oi_peak": 0.0,
        "put_oi_peak2": 0.0,            # 2nd Put OI Peak 행사가
        "call_oi_peak_global": 0.0,     # 탐색범위 무관 Call OI 절대 최대 행사가
        "put_oi_peak_global": 0.0,      # 탐색범위 무관 Put OI 절대 최대 행사가
        "call_oi_peak_norm": 0.0,
        "call_oi_peak2_norm": 0.0,      # 2nd Call Peak 집중도
        "put_oi_peak_norm": 0.0,
        "put_oi_peak2_norm": 0.0,       # 2nd Put Peak 집중도
        "call_oi_peak_global_norm": 0.0,  # 절대 최대 Call OI 집중도
        "put_oi_peak_global_norm": 0.0,   # 절대 최대 Put OI 집중도
        "oi_range_pct": 0.0,
        "dist_to_call_peak": 0.0,
        "dist_to_put_peak": 0.0,
        "oi_center": 0.0,
        "oi_center_dist_pct": 0.0,
        "net_gamma_proxy": 0.0,
        "zero_gamma_strike": 0.0,
        "zero_gamma_dist_pct": 0.0,
        "vol_trigger_strike": 0.0,
        "above_vol_trigger": 1.0,
        "peak_search_range_used": float(peak_search_range_pt),
    }

    F = float(underlying_price or 0.0)
    if F <= 0.0:
        return empty

    # ── 기본 탐색 반경 산출(IV 기반, 세션 고저 미존재 시 fallback) ───────────────
    # session_high/session_low가 있으면 1차 탐색은
    #   - Call: (F, session_high+10]
    #   - Put : [session_low-10, F)
    # 를 사용한다. 이 값들이 없을 때만 _peak_range(F±)를 사용한다.
    _iv = float(atm_iv or 0.0)
    if _iv > 0.0:
        _peak_range_base = calc_iv_peak_range(
            underlying_price=F,
            atm_iv=_iv,
            days_to_expiry=float(default_days_to_expiry or 7.0),
        )
    else:
        _peak_range_base = max(float(peak_search_range_pt or 20.0), 2.5)
    _peak_range = float(_peak_range_base)
    try:
        _sh = float(session_high or 0.0)
    except Exception:
        _sh = 0.0
    try:
        _sl = float(session_low or 0.0)
    except Exception:
        _sl = 0.0
    _call_upper_bound = (_sh + 10.0) if _sh > 0.0 else (F + _peak_range)
    _put_lower_bound = (_sl - 10.0) if _sl > 0.0 else (F - _peak_range)

    # ── Call/Put OI per Strike ──────────────────────────────────────────────
    call_oi_by_k: Dict[float, float] = {}
    call_iv_by_k: Dict[float, float] = {}
    call_gamma_by_k: Dict[float, float] = {}

    for v in (calls or {}).values():
        try:
            k = float(v.get("strike") or 0.0)
            oi = float(v.get("open_interest") or 0.0)
            if k > 0.0 and oi > 0.0:
                call_oi_by_k[k] = call_oi_by_k.get(k, 0.0) + oi
                iv = float(v.get("iv") or v.get("impv") or 0.0)
                if iv > 0.0:
                    call_iv_by_k[k] = iv
                g = float(v.get("gamma") or 0.0)
                if g > 0.0:
                    call_gamma_by_k[k] = g
        except Exception:
            continue

    put_oi_by_k: Dict[float, float] = {}
    put_iv_by_k: Dict[float, float] = {}
    put_gamma_by_k: Dict[float, float] = {}

    for v in (puts or {}).values():
        try:
            k = float(v.get("strike") or 0.0)
            oi = float(v.get("open_interest") or 0.0)
            if k > 0.0 and oi > 0.0:
                put_oi_by_k[k] = put_oi_by_k.get(k, 0.0) + oi
                iv = float(v.get("iv") or v.get("impv") or 0.0)
                if iv > 0.0:
                    put_iv_by_k[k] = iv
                g = float(v.get("gamma") or 0.0)
                if g > 0.0:
                    put_gamma_by_k[k] = g
        except Exception:
            continue

    if not call_oi_by_k and not put_oi_by_k:
        return empty

    # ── Call/Put OI Peak (지지·저항 대표 행사가) ────────────────────────────
    # OI-01 수정: 3단계 탐색 — 근접 범위 우선, 없으면 전체 fallback
    # 1단계: F 이상 + ATM ±_peak_range 이내 (저항 기본 탐색 구간)
    # 2단계: F 이상 전체 (near_range에 후보 없는 경우)
    # 3단계: 전체 행사가 (F 이상에도 OI 없는 경우 — 심층 OTM 콜만 존재)
    def _pick_top2(oi_by_k: Dict[float, float], candidates: Dict[float, float]) -> tuple:
        """후보 딕셔너리에서 OI 상위 2개 행사가를 반환한다.
        Returns: (peak1, peak2) — peak2는 없으면 0.0.
        """
        if not candidates:
            return 0.0, 0.0
        sorted_k = sorted(candidates, key=candidates.__getitem__, reverse=True)
        peak1 = float(sorted_k[0])
        peak2 = float(sorted_k[1]) if len(sorted_k) >= 2 else 0.0
        return peak1, peak2

    def _pick_call_peak(oi_by_k: Dict[float, float], f: float, rng: float, upper_bound: float) -> tuple:
        """상단 OI 상위 2개 행사가를 반환한다. Returns: (peak1, peak2).

        Call peak은 현재가를 초과(strict)하는 행사가에서만 탐색한다.
        현재가와 동일한 행사가는 저항이 아닌 현재 수준이므로 제외한다.
        """
        # 1단계: f 초과 + 1차 상단 경계(session_high+10 또는 F+rng)
        ub1 = float(upper_bound if upper_bound > f else (f + rng))
        near1 = {k: v for k, v in oi_by_k.items() if f < k <= ub1}
        if near1:
            return _pick_top2(oi_by_k, near1)

        # 2단계: 1차보다 확장된 상단 경계(항상 ub2 > ub1 보장)
        _expand = max(10.0, float(rng))
        ub2 = float(ub1 + _expand)
        near2 = {k: v for k, v in oi_by_k.items() if f < k <= ub2}
        if near2:
            return _pick_top2(oi_by_k, near2)

        # 3단계: f 초과 전체 fallback
        above_all = {k: v for k, v in oi_by_k.items() if k > f}
        if above_all:
            return _pick_top2(oi_by_k, above_all)

        # 최종 fallback: 전체(현재가 이하만 존재하는 특수 케이스)
        return _pick_top2(oi_by_k, oi_by_k)

    def _pick_put_peak(oi_by_k: Dict[float, float], f: float, rng: float, lower_bound: float) -> tuple:
        """하단 OI 상위 2개 행사가를 반환한다. Returns: (peak1, peak2).

        Put peak은 현재가 미만(strict)인 행사가에서만 탐색한다.
        현재가와 동일한 행사가는 지지가 아닌 현재 수준이므로 제외한다.
        """
        # 1단계: f 미만 + 1차 하단 경계(session_low-10 또는 F-rng)
        lb1 = float(lower_bound if lower_bound < f else (f - rng))
        near1 = {k: v for k, v in oi_by_k.items() if lb1 <= k < f}
        if near1:
            return _pick_top2(oi_by_k, near1)

        # 2단계: 1차보다 확장된 하단 경계(항상 lb2 < lb1 보장)
        _expand = max(10.0, float(rng))
        lb2 = float(lb1 - _expand)
        near2 = {k: v for k, v in oi_by_k.items() if lb2 <= k < f}
        if near2:
            return _pick_top2(oi_by_k, near2)

        # 3단계: f 미만 전체 fallback
        below_all = {k: v for k, v in oi_by_k.items() if k < f}
        if below_all:
            return _pick_top2(oi_by_k, below_all)

        # 최종 fallback: 전체(현재가 이상만 존재하는 특수 케이스)
        return _pick_top2(oi_by_k, oi_by_k)

    call_oi_peak = 0.0;  call_oi_peak2 = 0.0
    call_oi_peak_norm = 0.0;  call_oi_peak2_norm = 0.0
    call_oi_peak_global = 0.0; call_oi_peak_global_norm = 0.0
    if call_oi_by_k:
        try:
            call_oi_peak, call_oi_peak2 = _pick_call_peak(
                call_oi_by_k,
                F,
                _peak_range,
                _call_upper_bound,
            )
            total_call_oi = sum(call_oi_by_k.values())
            if total_call_oi > 0.0:
                call_oi_peak_norm = float(
                    call_oi_by_k.get(call_oi_peak, 0.0) / total_call_oi
                )
                # 탐색범위와 무관한 절대 최대 OI 행사가
                call_oi_peak_global = float(max(call_oi_by_k, key=call_oi_by_k.get))
                call_oi_peak_global_norm = float(
                    call_oi_by_k.get(call_oi_peak_global, 0.0) / total_call_oi
                )
                if call_oi_peak2 > 0.0:
                    call_oi_peak2_norm = float(
                        call_oi_by_k.get(call_oi_peak2, 0.0) / total_call_oi
                    )
        except Exception:
            call_oi_peak = 0.0;  call_oi_peak2 = 0.0
            call_oi_peak_norm = 0.0;  call_oi_peak2_norm = 0.0
            call_oi_peak_global = 0.0; call_oi_peak_global_norm = 0.0

    put_oi_peak = 0.0;  put_oi_peak2 = 0.0
    put_oi_peak_norm = 0.0;  put_oi_peak2_norm = 0.0
    put_oi_peak_global = 0.0; put_oi_peak_global_norm = 0.0
    if put_oi_by_k:
        try:
            put_oi_peak, put_oi_peak2 = _pick_put_peak(
                put_oi_by_k,
                F,
                _peak_range,
                _put_lower_bound,
            )
            total_put_oi = sum(put_oi_by_k.values())
            if total_put_oi > 0.0:
                put_oi_peak_norm = float(
                    put_oi_by_k.get(put_oi_peak, 0.0) / total_put_oi
                )
                # 탐색범위와 무관한 절대 최대 OI 행사가
                put_oi_peak_global = float(max(put_oi_by_k, key=put_oi_by_k.get))
                put_oi_peak_global_norm = float(
                    put_oi_by_k.get(put_oi_peak_global, 0.0) / total_put_oi
                )
                if put_oi_peak2 > 0.0:
                    put_oi_peak2_norm = float(
                        put_oi_by_k.get(put_oi_peak2, 0.0) / total_put_oi
                    )
        except Exception:
            put_oi_peak = 0.0;  put_oi_peak2 = 0.0
            put_oi_peak_norm = 0.0;  put_oi_peak2_norm = 0.0
            put_oi_peak_global = 0.0; put_oi_peak_global_norm = 0.0

    # ── 거리 및 박스 폭 ──────────────────────────────────────────────────────
    dist_to_call_peak = float((call_oi_peak - F) / F * 100.0) if call_oi_peak > 0.0 else 0.0
    dist_to_put_peak  = float((F - put_oi_peak)  / F * 100.0) if put_oi_peak  > 0.0 else 0.0
    oi_range_pct = (
        float((call_oi_peak - put_oi_peak) / F * 100.0)
        if call_oi_peak > 0.0 and put_oi_peak > 0.0
        else 0.0
    )
    oi_center = (
        float((call_oi_peak + put_oi_peak) / 2.0)
        if call_oi_peak > 0.0 and put_oi_peak > 0.0
        else F
    )
    oi_center_dist_pct = float((F - oi_center) / oi_center * 100.0) if oi_center > 0.0 else 0.0

    # ── BS Gamma 헬퍼 — 모듈 레벨 _bs_gamma_proxy 사용 (중복 제거) ───────────
    T = max(float(default_days_to_expiry or 7.0), 0.5) / 365.0

    def _get_gamma(k: float, gamma_by_k: Dict[float, float], iv_by_k: Dict[float, float]) -> float:
        g = gamma_by_k.get(k, 0.0)
        if g > 0.0:
            return float(g)
        iv = iv_by_k.get(k, 0.0)
        return _bs_gamma_proxy(F, k, iv, T)

    # ── Net Gamma Proxy (Strike별 가중 합산) ────────────────────────────────
    all_strikes = sorted(set(call_oi_by_k) | set(put_oi_by_k))
    net_gamma_proxy = 0.0
    net_gamma_by_k: Dict[float, float] = {}
    try:
        for k in all_strikes:
            c_oi = float(call_oi_by_k.get(k, 0.0))
            p_oi = float(put_oi_by_k.get(k, 0.0))
            c_g  = _get_gamma(k, call_gamma_by_k, call_iv_by_k)
            p_g  = _get_gamma(k, put_gamma_by_k, put_iv_by_k)
            ng = c_oi * c_g * float(contract_multiplier) * F * F \
               - p_oi * p_g * float(contract_multiplier) * F * F
            net_gamma_by_k[k] = float(ng)
        net_gamma_proxy = float(sum(net_gamma_by_k.values()))
        if not math.isfinite(net_gamma_proxy):
            net_gamma_proxy = 0.0
    except Exception:
        net_gamma_proxy = 0.0

    # ── Zero Gamma / Vol Trigger (모든 부호 전환점 수집 후 현재가 최근접) ──────
    # OI-03 수정: zero_gamma_strike와 vol_trigger_strike를 동일한 알고리즘으로
    # 산출한다. 두 값은 각각 "전체 전환점 중 현재가 최근접" 을 가리키며,
    # 이전의 zero_gamma(첫 번째 전환점) vs vol_trigger(최근접 전환점) 불일치를 해소한다.
    # 두 이름을 구분 유지하는 이유: 텔레그램/LLM 컨텍스트 출력 레이블이 분리되어 있음.
    zg_candidates: list = []
    try:
        sorted_ks = sorted(net_gamma_by_k.keys())
        for i in range(len(sorted_ks) - 1):
            k1, k2 = sorted_ks[i], sorted_ks[i + 1]
            ng1, ng2 = net_gamma_by_k[k1], net_gamma_by_k[k2]
            if ng1 * ng2 < 0.0:
                denom = ng2 - ng1
                if abs(denom) > 1e-12:
                    zg_k = float(k1 + (k2 - k1) * (-ng1) / denom)
                    zg_candidates.append(zg_k)
    except Exception:
        zg_candidates = []

    zero_gamma_strike = 0.0
    zero_gamma_dist_pct = 0.0
    vol_trigger_strike = 0.0
    above_vol_trigger = 1.0

    if zg_candidates:
        try:
            # zero_gamma_strike: 현재가 최근접 전환점 (OI-03 수정: 첫 번째 → 최근접)
            zero_gamma_strike = float(min(zg_candidates, key=lambda k: abs(k - F)))
            if F > 0.0:
                zero_gamma_dist_pct = float((F - zero_gamma_strike) / F * 100.0)
                if not math.isfinite(zero_gamma_dist_pct):
                    zero_gamma_dist_pct = 0.0
        except Exception:
            zero_gamma_strike = 0.0
            zero_gamma_dist_pct = 0.0

        try:
            # vol_trigger_strike: zero_gamma_strike와 동일 알고리즘 (최근접 전환점)
            vol_trigger_strike = zero_gamma_strike
            above_vol_trigger = 1.0 if F >= vol_trigger_strike else 0.0
        except Exception:
            vol_trigger_strike = 0.0
            above_vol_trigger = 1.0
    else:
        # 전환점 없으면 net_gamma_proxy 부호로 레짐 판단
        above_vol_trigger = 1.0 if net_gamma_proxy >= 0.0 else 0.0

    return {
        "call_oi_peak":       round(float(call_oi_peak), 2),
        "call_oi_peak2":      round(float(call_oi_peak2), 2),
        "put_oi_peak":        round(float(put_oi_peak), 2),
        "put_oi_peak2":       round(float(put_oi_peak2), 2),
        "call_oi_peak_global": round(float(call_oi_peak_global), 2),
        "put_oi_peak_global":  round(float(put_oi_peak_global), 2),
        "call_oi_peak_norm":  round(float(call_oi_peak_norm), 4),
        "call_oi_peak2_norm": round(float(call_oi_peak2_norm), 4),
        "put_oi_peak_norm":   round(float(put_oi_peak_norm), 4),
        "put_oi_peak2_norm":  round(float(put_oi_peak2_norm), 4),
        "call_oi_peak_global_norm": round(float(call_oi_peak_global_norm), 4),
        "put_oi_peak_global_norm":  round(float(put_oi_peak_global_norm), 4),
        "oi_range_pct":       round(float(oi_range_pct), 4),
        "dist_to_call_peak":  round(float(dist_to_call_peak), 4),
        "dist_to_put_peak":   round(float(dist_to_put_peak), 4),
        "oi_center":          round(float(oi_center), 2),
        "oi_center_dist_pct": round(float(oi_center_dist_pct), 4),
        "net_gamma_proxy":    round(float(net_gamma_proxy), 2),
        "zero_gamma_strike":  round(float(zero_gamma_strike), 2),
        "zero_gamma_dist_pct": round(float(zero_gamma_dist_pct), 4),
        "vol_trigger_strike": round(float(vol_trigger_strike), 2),
        "above_vol_trigger":  float(above_vol_trigger),
        "peak_search_range_used": round(float(_peak_range), 2),
    }



def calc_otm_premium_change(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
    *,
    otm_open_min: float = 0.30,
) -> Dict[str, Any]:
    """OTM 옵션의 시가 대비 현재가 증감률 단순 평균을 계산한다.

    각 OTM 종목의 프리미엄 변화율 = (현재가 - 시가) / 시가.
    시가(open_price)는 tick_processor.set_option_open_map()으로 주입된 값을 사용한다.

    필터 기준:
        - 콜: strike > atm_strike
        - 풋: strike < atm_strike
        - 시가(open_price) >= otm_open_min
        - 현재가(price) >= otm_open_min  [OTM-2]
          open_price/price 양쪽 하한이 보장되므로 추가 clipping 불필요.
          극값은 실제 옵션 플로우 신호로 그대로 반영한다.

    변경 이력:
        [OTM-WGT] 시가(open_price) 가중 평균 → 단순 평균.
            가중 평균은 근접 OTM을 과대 반영해 방향성 신호를 왜곡.
        [OTM-2] price >= otm_open_min 필터: 사실상 무가치 종목 제외.

    Args:
        calls:            콜옵션 딕셔너리 {symbol: {..., "price", "open_price", "strike"}}.
        puts:             풋옵션 딕셔너리 (동일 구조).
        underlying_price: 현재 선물가(ATM 탐색 기준).
        otm_open_min:     시가·현재가 최소 기준. 기본 0.30pt.

    Returns:
        call_otm_prem_chg  : 콜 OTM 단순 평균 변화율. 종목 없으면 None.
        put_otm_prem_chg   : 풋 OTM 단순 평균 변화율. 종목 없으면 None.
        call_otm_count     : 계산에 사용된 콜 OTM 종목 수.
        put_otm_count      : 계산에 사용된 풋 OTM 종목 수.
        call_otm_symbols   : 계산에 포함된 콜 OTM 심볼 목록 (디버그용).
        put_otm_symbols    : 계산에 포함된 풋 OTM 심볼 목록 (디버그용).
        call_otm_avg_open  : 콜 OTM 평균 시가(pt). 진단용.
        put_otm_avg_open   : 풋 OTM 평균 시가(pt). 진단용.
        call_strike_range  : 콜 OTM 포함 행사가 [min, max]. 디버그용.
        put_strike_range   : 풋 OTM 포함 행사가 [min, max]. 디버그용.
    """
    empty: Dict[str, Any] = {
        "call_otm_prem_chg": None,
        "put_otm_prem_chg":  None,
        "call_otm_count":    0,
        "put_otm_count":     0,
        "call_otm_symbols":  [],
        "put_otm_symbols":   [],
        "call_otm_avg_open": 0.0,
        "put_otm_avg_open":  0.0,
        "call_strike_range": [],
        "put_strike_range":  [],
    }

    F = float(underlying_price or 0.0)
    if F <= 0.0:
        return empty

    all_strikes: list = []
    try:
        for v in list((calls or {}).values()) + list((puts or {}).values()):
            k = float(v.get("strike") or 0.0)
            if k > 0.0:
                all_strikes.append(k)
    except Exception:
        pass
    if not all_strikes:
        return empty

    atm = _find_atm_strike(sorted(set(all_strikes)), F)
    if atm is None:
        return empty

    _otm_min = float(otm_open_min or 0.0)

    # ── 콜 OTM ───────────────────────────────────────────────────────────
    call_changes: list = []
    call_opens:   list = []
    call_strikes: list = []
    call_syms:    list = []
    try:
        for sym, v in (calls or {}).items():
            try:
                strike = float(v.get("strike") or 0.0)
                if strike <= atm:
                    continue
                open_p = float(v.get("open_price") or 0.0)
                if open_p < _otm_min:
                    continue
                price = float(v.get("price") or 0.0)
                if price < _otm_min:
                    continue
                chg = (price - open_p) / open_p
                if math.isfinite(chg):
                    call_changes.append(chg)
                    call_opens.append(open_p)
                    call_strikes.append(strike)
                    call_syms.append(str(sym))
            except Exception:
                continue
    except Exception:
        pass

    # ── 풋 OTM ───────────────────────────────────────────────────────────
    put_changes: list = []
    put_opens:   list = []
    put_strikes: list = []
    put_syms:    list = []
    try:
        for sym, v in (puts or {}).items():
            try:
                strike = float(v.get("strike") or 0.0)
                if strike >= atm:
                    continue
                open_p = float(v.get("open_price") or 0.0)
                if open_p < _otm_min:
                    continue
                price = float(v.get("price") or 0.0)
                if price < _otm_min:
                    continue
                chg = (price - open_p) / open_p
                if math.isfinite(chg):
                    put_changes.append(chg)
                    put_opens.append(open_p)
                    put_strikes.append(strike)
                    put_syms.append(str(sym))
            except Exception:
                continue
    except Exception:
        pass

    # ── 단순 평균 ──────────────────────────────────────────────────────────
    call_avg = round(float(np.mean(call_changes)), 6) if call_changes else None
    put_avg  = round(float(np.mean(put_changes)),  6) if put_changes  else None

    call_avg_open = round(float(np.mean(call_opens)), 2) if call_opens else 0.0
    put_avg_open  = round(float(np.mean(put_opens)),  2) if put_opens  else 0.0
    call_str_rng  = sorted(set(call_strikes)) if call_strikes else []
    put_str_rng   = sorted(set(put_strikes))  if put_strikes  else []

    return {
        "call_otm_prem_chg": call_avg,
        "put_otm_prem_chg":  put_avg,
        "call_otm_count":    int(len(call_changes)),
        "put_otm_count":     int(len(put_changes)),
        "call_otm_symbols":  call_syms,
        "put_otm_symbols":   put_syms,
        "call_otm_avg_open": call_avg_open,
        "put_otm_avg_open":  put_avg_open,
        "call_strike_range": [call_str_rng[0], call_str_rng[-1]] if len(call_str_rng) >= 2 else call_str_rng,
        "put_strike_range":  [put_str_rng[0],  put_str_rng[-1]]  if len(put_str_rng)  >= 2 else put_str_rng,
    }

