"""Call-Put 패리티 이탈 및 프리미엄 블리드 계산.

만기 근접 시 옵션 가격의 이론가 이탈(parity divergence)과
프리미엄 시간가치 수축(premium bleed)을 측정한다.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import math
import numpy as np

from ..option_core import _find_atm_strike, _get_atm_option_price

def calc_parity_divergence(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
    *,
    days_to_expiry: float = 7.0,
    risk_free_rate: float = 0.035,
    prev_underlying_price: Optional[float] = None,
    prev_atm_call_price: Optional[float] = None,
    prev_atm_put_price: Optional[float] = None,
) -> Dict[str, float]:
    """만기주 ATM 콜-풋 패리티 이탈 지표를 계산한다.

    콜-풋 패리티(C - P = F - K·e^{-rT})에서 이탈하는 정도와 방향을 수치화한다.
    만기가 가까울수록(days_to_expiry → 0) 이 관계가 무너지는 경우를 탐지한다.

    Returns:
        parity_spread        : C - P - (F - K·e^{-rT}). 이론상 0에 가까워야 함.
        parity_spread_pct    : parity_spread / F * 100. 비율 정규화 (%).
        call_delta_proxy     : C / (C + P). ATM에서는 이론상 0.5.
        straddle_price       : C + P. 내재 변동성의 직접 지표.
        straddle_vs_fut_move : straddle_price / max(|F - K|, 0.1). 선물 이동 대비 스트래들 배율.
        call_vs_fut_ret_diff : 콜 수익률 - (0.5 * 선물 수익률). 직전 틱 대비 추종 이탈.
        dte_weight_norm      : 만기 근접도 [0, 1]. 1 = 만기 당일, 0.1 ≈ 만기 10일 전.
        parity_divergence_score : 종합 이탈 스코어 [-1, 1]. DTE 가중 적용.

    Notes:
        - tick_processor.process_option_tick()이 저장하는 'price' 필드를 사용한다.
        - prev_* 인자가 None이면 call_vs_fut_ret_diff는 0.0으로 반환된다.
        - Best-effort: 모든 계산 실패 시 zeros를 반환한다.
    """
    empty: Dict[str, float] = {
        "parity_spread": 0.0,
        "parity_spread_pct": 0.0,
        "call_delta_proxy": 0.5,
        "straddle_price": 0.0,
        "straddle_vs_fut_move": 0.0,
        "call_vs_fut_ret_diff": 0.0,
        "dte_weight_norm": 0.0,
        "parity_divergence_score": 0.0,
    }

    F = float(underlying_price or 0.0)
    if F <= 0.0:
        return empty

    # --- ATM 행사가 탐색 (calc_iv_skew와 동일 로직) ---
    call_by_k: Dict[float, Dict[str, Any]] = {}
    for v in (calls or {}).values():
        try:
            k = float(v.get("strike") or 0.0)
            if k > 0.0:
                call_by_k[k] = v
        except Exception:
            continue

    put_by_k: Dict[float, Dict[str, Any]] = {}
    for v in (puts or {}).values():
        try:
            k = float(v.get("strike") or 0.0)
            if k > 0.0:
                put_by_k[k] = v
        except Exception:
            continue

    all_strikes = sorted(set(call_by_k) | set(put_by_k))
    if not all_strikes:
        return empty

    atm_strike_found = _find_atm_strike(all_strikes, F)
    if atm_strike_found is None:
        return empty
    atm_strike = atm_strike_found

    # --- ATM 옵션 체결가 추출 (process_option_tick의 'price' 필드 사용) ---
    atm_call_price = _get_atm_option_price(calls, atm_strike)
    atm_put_price = _get_atm_option_price(puts, atm_strike)

    if atm_call_price <= 0.0 or atm_put_price <= 0.0:
        return empty

    # --- 패리티 스프레드 ---
    try:
        T = max(float(days_to_expiry or 0.0), 0.0) / 365.0
        r = float(risk_free_rate or 0.035)
        theoretical_diff = F - atm_strike * math.exp(-r * T)
        actual_diff = atm_call_price - atm_put_price
        parity_spread = float(actual_diff - theoretical_diff)
        parity_spread_pct = float(parity_spread / F * 100.0) if F > 0.0 else 0.0
    except Exception:
        parity_spread = 0.0
        parity_spread_pct = 0.0

    # --- 콜 델타 프록시 ---
    straddle_price = float(atm_call_price + atm_put_price)
    call_delta_proxy = float(atm_call_price / straddle_price) if straddle_price > 0.0 else 0.5

    # --- 스트래들 vs 선물 이동 배율 ---
    try:
        fut_move = max(abs(F - atm_strike), 0.1)
        straddle_vs_fut_move = float(straddle_price / fut_move)
    except Exception:
        straddle_vs_fut_move = 0.0

    # --- 콜 수익률 vs 선물 수익률 차이 (직전 틱 필요) ---
    call_vs_fut_ret_diff = 0.0
    try:
        prev_F = float(prev_underlying_price or 0.0)
        prev_C = float(prev_atm_call_price or 0.0)
        if prev_F > 0.0 and prev_C > 0.0 and abs(prev_F - F) > 1e-9:
            fut_ret = (F - prev_F) / prev_F
            call_ret = (atm_call_price - prev_C) / prev_C
            # ATM 콜의 이론 델타 ≈ 0.5 이므로 call_ret ≈ 0.5 * fut_ret 이어야 함
            expected_call_ret = 0.5 * fut_ret
            call_vs_fut_ret_diff = float(call_ret - expected_call_ret)
            if not np.isfinite(call_vs_fut_ret_diff):
                call_vs_fut_ret_diff = 0.0
    except Exception:
        call_vs_fut_ret_diff = 0.0

    # --- DTE 가중치 ---
    # 설계: min(1 / (max(dte, 0.1) * 10), 1.0)
    #   dte=0일 → 1.000, dte=1일 → 0.100, dte=3일 → 0.033, dte=7일 → 0.014
    # 만기 당일에만 신호가 의미 있고, 7일 이상이면 사실상 0에 수렴.
    # 가드레일 임계값 dte_w >= 0.33 = 만기 3일 이내, dte_w >= 1.0 = 만기 당일.
    try:
        dte = max(float(days_to_expiry or 0.0), 0.0)
        dte_weight_norm = float(min(1.0 / (max(dte, 0.1) * 10.0), 1.0))
        if not np.isfinite(dte_weight_norm):
            dte_weight_norm = 0.0
    except Exception:
        dte_weight_norm = 0.0

    # --- 종합 이탈 스코어 [-1, 1] ---
    # 설계 계수: parity_spread_pct * 0.4 + (delta-0.5)*2 * 0.3 + ret_diff*10 * 0.3
    # 각 항을 [-1,1]로 클리핑 후 DTE 가중치로 신호 증폭 (최대 2배).
    try:
        raw_score = (
            float(np.clip(parity_spread_pct * 0.4, -1.0, 1.0)) * 0.4
            + float(np.clip((call_delta_proxy - 0.5) * 2.0, -1.0, 1.0)) * 0.3
            + float(np.clip(call_vs_fut_ret_diff * 10.0, -1.0, 1.0)) * 0.3
        )
        parity_divergence_score = float(np.clip(raw_score * (1.0 + dte_weight_norm), -1.0, 1.0))
        if not np.isfinite(parity_divergence_score):
            parity_divergence_score = 0.0
    except Exception:
        parity_divergence_score = 0.0

    return {
        "parity_spread": round(float(parity_spread), 4),
        "parity_spread_pct": round(float(parity_spread_pct), 4),
        "call_delta_proxy": round(float(call_delta_proxy), 4),
        "straddle_price": round(float(straddle_price), 4),
        "straddle_vs_fut_move": round(float(straddle_vs_fut_move), 4),
        "call_vs_fut_ret_diff": round(float(call_vs_fut_ret_diff), 4),
        "dte_weight_norm": round(float(dte_weight_norm), 4),
        "parity_divergence_score": round(float(parity_divergence_score), 4),
    }


def calc_premium_bleed(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
    *,
    days_to_expiry: float = 7.0,
    prev_underlying_price: Optional[float] = None,
    prev_atm_call_price: Optional[float] = None,
    prev_atm_put_price: Optional[float] = None,
) -> Dict[str, float]:
    """만기주 선물 상승 중 옵션 프리미엄 수축(Premium Bleed) 지표를 계산한다.

    콜-풋 패리티 이탈과 달리, 선물이 방향성을 보이는 동안 스트래들 전체 가치가
    수축하는 현상을 탐지한다. 원인: Theta 급가속, IV Crush, MM 롤오버.

    Returns:
        straddle_decay_vs_fut  : straddle 수익률 - |선물 수익률| * 0.5.
                                 음수일수록 선물 상승 대비 프리미엄이 비정상 수축.
        iv_crush_proxy         : ATM IV 방향 근사 변화율.
                                 = (sigma_now - sigma_prev) / sigma_prev.
                                 sigma = straddle / (F * sqrt(T) * sqrt(2/pi)).
        fut_ret                : 직전 틱 대비 선물 수익률. 방향 확인용.
        straddle_now           : 현재 ATM 스트래들 가격 (C+P).
        straddle_prev          : 직전 틱 ATM 스트래들 가격. 0이면 prev 없음.
        premium_bleed_score    : 종합 수축 스코어 [-1, 1].
                                 -1 = 강한 프리미엄 수축 (선물 방향성 대비).
                                  0 = 중립 또는 prev 없음.
                                 +1 = 프리미엄 팽창 (IV 급등 등).

    Notes:
        - prev_* 인자가 None이면 스코어는 0.0으로 반환된다.
        - dte_weight_norm은 calc_parity_divergence()의 값을 재활용하므로 별도 반환하지 않는다.
        - 선물 수익률이 0.03% 미만인 경우 노이즈로 간주하고 스코어 0.0 반환.
    """
    empty: Dict[str, float] = {
        "straddle_decay_vs_fut": 0.0,
        "iv_crush_proxy": 0.0,
        "fut_ret": 0.0,
        "straddle_now": 0.0,
        "straddle_prev": 0.0,
        "premium_bleed_score": 0.0,
    }

    F = float(underlying_price or 0.0)
    if F <= 0.0:
        return empty

    # --- ATM 행사가 탐색 (calc_parity_divergence와 동일 로직) ---
    call_by_k: Dict[float, Dict[str, Any]] = {}
    for v in (calls or {}).values():
        try:
            k = float(v.get("strike") or 0.0)
            if k > 0.0:
                call_by_k[k] = v
        except Exception:
            continue

    put_by_k: Dict[float, Dict[str, Any]] = {}
    for v in (puts or {}).values():
        try:
            k = float(v.get("strike") or 0.0)
            if k > 0.0:
                put_by_k[k] = v
        except Exception:
            continue

    all_strikes = sorted(set(call_by_k) | set(put_by_k))
    if not all_strikes:
        return empty

    atm_strike_found = _find_atm_strike(all_strikes, F)
    if atm_strike_found is None:
        return empty
    atm_strike = atm_strike_found

    atm_call_price = _get_atm_option_price(calls, atm_strike)
    atm_put_price  = _get_atm_option_price(puts, atm_strike)

    if atm_call_price <= 0.0 or atm_put_price <= 0.0:
        return empty

    straddle_now = float(atm_call_price + atm_put_price)

    # --- prev 없으면 부분 반환 ---
    prev_F = float(prev_underlying_price or 0.0)
    prev_C = float(prev_atm_call_price or 0.0)
    prev_P = float(prev_atm_put_price or 0.0)

    empty["straddle_now"] = round(straddle_now, 4)

    if prev_F <= 0.0 or prev_C <= 0.0 or prev_P <= 0.0:
        return empty

    straddle_prev = float(prev_C + prev_P)
    empty["straddle_prev"] = round(straddle_prev, 4)

    # --- 선물 수익률 ---
    try:
        fut_ret = (F - prev_F) / prev_F
        if not np.isfinite(fut_ret):
            fut_ret = 0.0
    except Exception:
        fut_ret = 0.0

    # 선물 움직임이 너무 작으면 노이즈 → 스코어 0
    # 기준: 0.03% 미만 (KP200 선물 1틱 ≈ 0.05pt, 350pt 기준 약 0.014%)
    if abs(fut_ret) < 0.0003:
        return {
            "straddle_decay_vs_fut": 0.0,
            "iv_crush_proxy": 0.0,
            "fut_ret": round(float(fut_ret), 6),
            "straddle_now": round(straddle_now, 4),
            "straddle_prev": round(straddle_prev, 4),
            "premium_bleed_score": 0.0,
        }

    # --- 스트래들 수축률 vs 선물 수익률 ---
    # 이론: 선물 1% 상승 시 ATM 스트래들은 거의 변화 없어야 함.
    # 실제 수축이면 → IV Crush 또는 Theta 급가속 신호.
    try:
        if straddle_prev > 0.0:
            straddle_ret = (straddle_now - straddle_prev) / straddle_prev
        else:
            straddle_ret = 0.0
        # 이론 기대치: |fut_ret| * 0.5 (ATM 델타 ≈ 0.5 → 콜 상승 + 풋 하락 상쇄)
        straddle_decay_vs_fut = float(straddle_ret - abs(fut_ret) * 0.5)
        if not np.isfinite(straddle_decay_vs_fut):
            straddle_decay_vs_fut = 0.0
    except Exception:
        straddle_decay_vs_fut = 0.0

    # --- IV Crush 근사 ---
    # ATM 이론 스트래들 ≈ F * σ * sqrt(T) * sqrt(2/π)
    # σ_proxy = straddle / (F * sqrt(T) * sqrt(2/π))
    iv_crush_proxy = 0.0
    try:
        T = max(float(days_to_expiry or 0.0), 0.0) / 365.0
        if T <= 0.0:
            T = 1.0 / 365.0  # 만기 당일: 1일로 처리
        scale = F * math.sqrt(T) * math.sqrt(2.0 / math.pi)
        if scale > 0.0:
            sigma_now  = straddle_now  / scale
            sigma_prev = straddle_prev / scale
            if sigma_prev > 0.0:
                iv_crush_proxy = float((sigma_now - sigma_prev) / sigma_prev)
                if not np.isfinite(iv_crush_proxy):
                    iv_crush_proxy = 0.0
    except Exception:
        iv_crush_proxy = 0.0

    # --- DTE 가중치 (calc_parity_divergence와 동일 공식) ---
    try:
        dte = max(float(days_to_expiry or 0.0), 0.0)
        dte_weight_norm = float(min(1.0 / (max(dte, 0.1) * 10.0), 1.0))
        if not np.isfinite(dte_weight_norm):
            dte_weight_norm = 0.0
    except Exception:
        dte_weight_norm = 0.0

    # --- 종합 프리미엄 수축 스코어 [-1, 1] ---
    # decay_component: 음수일수록 수축 강함. 계수 20배 후 클리핑.
    # iv_component:    음수일수록 IV 감소. 계수 5배 후 클리핑.
    # dte_weight 증폭: 만기 당일 최대 2배.
    try:
        decay_component = float(np.clip(straddle_decay_vs_fut * 20.0, -1.0, 1.0))
        iv_component    = float(np.clip(iv_crush_proxy * 5.0,         -1.0, 1.0))
        raw_score = decay_component * 0.6 + iv_component * 0.4
        premium_bleed_score = float(np.clip(raw_score * (1.0 + dte_weight_norm), -1.0, 1.0))
        if not np.isfinite(premium_bleed_score):
            premium_bleed_score = 0.0
    except Exception:
        premium_bleed_score = 0.0

    return {
        "straddle_decay_vs_fut": round(float(straddle_decay_vs_fut), 6),
        "iv_crush_proxy":        round(float(iv_crush_proxy), 6),
        "fut_ret":               round(float(fut_ret), 6),
        "straddle_now":          round(float(straddle_now), 4),
        "straddle_prev":         round(float(straddle_prev), 4),
        "premium_bleed_score":   round(float(premium_bleed_score), 4),
    }


