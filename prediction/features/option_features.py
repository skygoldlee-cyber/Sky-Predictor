"""Option-derived feature calculators — facade module.

이 파일은 하위 호환성을 위한 re-export facade다.
실제 구현은 다음 하위 모듈에 분리되어 있다:

    prediction/option_core.py        : ATM 탐색, BS Gamma proxy (공통 헬퍼)
    prediction/parity_features.py    : calc_parity_divergence, calc_premium_bleed
    prediction/oi_features.py        : calc_expected_amplitude, calc_oi_levels,
                                       calc_otm_premium_change
    prediction/similarity_features.py: FuturesCallSimilarity

외부 코드는 이 파일에서 그대로 import 가능:
    from ..option_features import build_option_snapshot, calc_oi_levels
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

# ── 공통 헬퍼 (re-export) ────────────────────────────────────────────────────
from ..option_core import (
    _find_atm_strike,
    _bs_gamma_proxy,
)

# ── 패리티/블리드 (re-export) ─────────────────────────────────────────────────
from .parity_features import calc_parity_divergence, calc_premium_bleed

# ── OI/진폭/OTM (re-export) ───────────────────────────────────────────────────
from .oi_features import (
    calc_oi_levels,
    calc_otm_premium_change,
)

# ── 선물-콜 유사도 (re-export) ────────────────────────────────────────────────

# ── 이 파일에 직접 구현된 함수들 ─────────────────────────────────────────────
# calc_iv_peak_range, calc_pcr, calc_iv_skew, calc_gex,
# _strike_to_symbol_map, calc_atm_microstructure, calc_max_pain,
# _get_atm_option_price, build_option_snapshot

def calc_iv_peak_range(
    underlying_price: float,
    atm_iv: float,
    days_to_expiry: float,
    *,
    sigma_multiplier: float = 2.0,
    min_range_pt: float = 10.0,
    max_range_pt: float = 40.0,
    strike_step: float = 2.5,
) -> float:
    """ATM IV와 잔존기간(DTE)으로 OI Peak 탐색 범위(pt)를 동적 산출한다.

    공식:
        expected_move_pt = F * ATM_IV * sqrt(DTE / 252) * sigma_multiplier

    근거:
        - 로그정규 분포 가정 하에 DTE일 동안의 1σ 예상 이동폭.
        - KP200 선물(F≈350): IV=20%, DTE=5일 → 1σ ≈ 9.9pt, 2σ ≈ 19.8pt.
        - sigma_multiplier=2.0: 일중에 2σ 이상 이동은 드물므로 그 범위 이내 Peak 탐색.

    Args:
        underlying_price:  현재 선물가(F).
        atm_iv:            ATM 옵션의 내재변동성(소수. 예: 0.20 = 20%).
                           0이면 min_range_pt(고정 fallback) 반환.
        days_to_expiry:    잔존기간(일). 장중 OI Peak 탐색에는 당일 기준(DTE=1~7) 사용.
        sigma_multiplier:  탐색 범위를 몇 σ로 설정할지. 기본 2.0σ.
        min_range_pt:      반환 최솟값(pt). IV가 극히 낮아도 최소 범위 보장. 기본 10pt.
        max_range_pt:      반환 최댓값(pt). 폭등 국면에서 범위 무한 확장 방지. 기본 40pt.
        strike_step:       행사가 간격(pt). 결과를 이 값의 배수로 반올림. 기본 2.5pt.

    Returns:
        peak_search_range_pt(float): OI Peak 탐색 반경(pt). min~max 범위 내 보장.

    Examples:
        # IV=20%, DTE=5일, F=350 → 2σ ≈ 19.8pt → 20.0pt 반환 (2.5 배수 반올림)
        calc_iv_peak_range(350.0, 0.20, 5.0)  # → 20.0

        # IV=35%(급등 장), DTE=3일, F=350 → 2σ ≈ 21.2pt → 22.5pt 반환
        calc_iv_peak_range(350.0, 0.35, 3.0)  # → 22.5

        # IV=10%(저변동), DTE=7일, F=350 → 2σ ≈ 13.1pt → 12.5pt 반환
        calc_iv_peak_range(350.0, 0.10, 7.0)  # → 12.5
    """
    try:
        F   = float(underlying_price or 0.0)
        iv  = float(atm_iv or 0.0)
        dte = float(days_to_expiry or 0.0)

        if F <= 0.0 or iv <= 0.0 or dte <= 0.0:
            return float(min_range_pt)

        # 1σ 예상 이동폭(pt): F * σ * sqrt(DTE/252)
        import math as _math
        sigma_move_pt = F * iv * _math.sqrt(dte / 252.0)

        # sigma_multiplier σ 기준 탐색 반경
        raw_range = sigma_move_pt * float(sigma_multiplier)

        # 행사가 간격(2.5pt) 배수로 반올림 (nearest even)
        step = max(float(strike_step), 1.0)
        rounded = round(raw_range / step) * step

        # min/max 클리핑
        clamped = max(float(min_range_pt), min(float(max_range_pt), rounded))

        if not _math.isfinite(clamped):
            return float(min_range_pt)

        return float(clamped)

    except Exception:
        return float(min_range_pt)


def _strikes_for_pcr_atm_window(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
    n_each_side: int = 5,
) -> Optional[set]:
    """콜·풋에 등장하는 행사가를 합쳐 정렬한 뒤 ATM 기준 위·아래 각 n개만 포함하는 집합.

    Returns:
        포함할 행사가 집합. 실패 시 None → 호출부에서 전체 합산(fallback).
    """
    try:
        F = float(underlying_price or 0.0)
        if F <= 0.0:
            return None
        n = max(0, int(n_each_side))
        strikes_set: set[float] = set()
        for v in list((calls or {}).values()) + list((puts or {}).values()):
            try:
                k = float(v.get("strike") or 0.0)
                if k > 0.0:
                    strikes_set.add(k)
            except Exception:
                continue
        if not strikes_set:
            return None
        strikes = sorted(strikes_set)
        atm = _find_atm_strike(strikes, F)
        if atm is None:
            return None
        idx = min(range(len(strikes)), key=lambda j: abs(strikes[j] - atm))
        lo = max(0, idx - n)
        hi = min(len(strikes) - 1, idx + n)
        return set(strikes[lo : hi + 1])
    except Exception:
        return None


def _pcr_row_in_strikes(v: Dict[str, Any], strike_filter: Optional[set]) -> bool:
    if strike_filter is None:
        return True
    try:
        k = float(v.get("strike") or 0.0)
        if k <= 0.0:
            return False
        return k in strike_filter
    except Exception:
        return False


def calc_pcr(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: Optional[float] = None,
    *,
    atm_strikes_each_side: int = 5,
) -> Dict[str, float]:
    """Compute put/call ratios (volume and open-interest).

    기본: 기초자산가로 ATM을 잡고, **그 행사가를 기준 위·아래 각 N개**(총 최대 2N+1개
    행사가)에 대해서만 콜·풋 각각 거래량·OI를 합산합니다. 풋 행사가가 쪽수상 더 많아도
    전체 행사가를 한꺼번에 넣어 비율이 흔들리는 문제를 줄입니다.

    ``underlying_price``가 없거나 0이면, 또는 ATM 윈도를 만들 수 없으면 기존처럼
    전체 심볼 합산으로 폴백합니다.
    """
    strike_filter: Optional[set] = None
    try:
        if underlying_price is not None and float(underlying_price) > 0.0:
            strike_filter = _strikes_for_pcr_atm_window(
                calls or {},
                puts or {},
                float(underlying_price),
                int(atm_strikes_each_side),
            )
    except Exception:
        strike_filter = None

    call_vol = sum(
        float(v.get("volume") or 0.0)
        for v in (calls or {}).values()
        if _pcr_row_in_strikes(v, strike_filter)
    )
    put_vol = sum(
        float(v.get("volume") or 0.0)
        for v in (puts or {}).values()
        if _pcr_row_in_strikes(v, strike_filter)
    )
    call_oi = sum(
        float(v.get("open_interest") or 0.0)
        for v in (calls or {}).values()
        if _pcr_row_in_strikes(v, strike_filter)
    )
    put_oi = sum(
        float(v.get("open_interest") or 0.0)
        for v in (puts or {}).values()
        if _pcr_row_in_strikes(v, strike_filter)
    )

    return {
        "pcr_volume": round(put_vol / call_vol if call_vol > 0.0 else 1.0, 4),
        "pcr_oi": round(put_oi / call_oi if call_oi > 0.0 else 1.0, 4),
        "call_vol": int(call_vol),
        "put_vol": int(put_vol),
        "call_oi": int(call_oi),
        "put_oi": int(put_oi),
    }


def calc_iv_skew(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
) -> Dict[str, float]:
    """Compute ATM IV skew (ATM put IV / ATM call IV)."""

    empty = {"iv_skew": 1.0, "atm_strike": 0.0, "atm_call_iv": 0.0, "atm_put_iv": 0.0}
    if not calls or not puts or float(underlying_price or 0.0) <= 0.0:
        return empty

    call_iv: Dict[float, float] = {}
    for v in calls.values():
        try:
            k = float(v.get("strike") or 0.0)
            iv = float(v.get("iv") or 0.0)
            if k > 0.0 and iv > 0.0:
                call_iv[k] = iv
        except Exception:
            continue

    put_iv: Dict[float, float] = {}
    for v in puts.values():
        try:
            k = float(v.get("strike") or 0.0)
            iv = float(v.get("iv") or 0.0)
            if k > 0.0 and iv > 0.0:
                put_iv[k] = iv
        except Exception:
            continue

    all_strikes = sorted(set(call_iv) | set(put_iv))
    if not all_strikes:
        return empty

    upx = float(underlying_price)
    atm = _find_atm_strike(all_strikes, upx)
    if atm is None:
        return empty

    atm_c = float(call_iv.get(atm) or 0.0)
    atm_p = float(put_iv.get(atm) or 0.0)

    if atm_c <= 0.0 or atm_p <= 0.0:
        return empty
    skew = atm_p / atm_c
    return {
        "iv_skew": round(float(skew), 4),
        "atm_strike": float(atm),
        "atm_call_iv": round(atm_c, 4),
        "atm_put_iv": round(atm_p, 4),
    }


def calc_gex(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
    *,
    default_days_to_expiry: float = 7.0,
    contract_multiplier: float = 100.0,
) -> Dict[str, float]:
    empty = {"gex": 0.0, "gex_calls": 0.0, "gex_puts": 0.0, "gex_gamma_count": 0.0}
    S = float(underlying_price or 0.0)
    if S <= 0.0:
        return empty

    try:
        T = float(default_days_to_expiry) / 365.0
    except Exception:
        T = 7.0 / 365.0
    if T <= 0.0:
        T = 7.0 / 365.0

    def _gamma_from_opt(v: Dict[str, Any]) -> float:
        try:
            g = float(v.get("gamma") or 0.0)
            if g > 0.0:
                return float(g)
        except Exception as _e:
            logger.debug("[_gamma_from_opt] 오류 무시: %s", _e)
        try:
            K = float(v.get("strike") or 0.0)
            iv = float(v.get("iv") or v.get("impv") or v.get("imp_vol") or 0.0)
            return _bs_gamma_proxy(S, K, iv, T)
        except Exception:
            return 0.0

    def _oi(v: Dict[str, Any]) -> float:
        try:
            return float(v.get("open_interest") or 0.0)
        except Exception:
            return 0.0

    gex_calls = 0.0
    gex_puts = 0.0
    g_cnt = 0.0

    for v in (calls or {}).values():
        g = _gamma_from_opt(v)
        oi = _oi(v)
        if g > 0.0 and oi > 0.0:
            g_cnt += 1.0
            gex_calls += float(g) * float(oi) * float(contract_multiplier) * float(S) * float(S)

    for v in (puts or {}).values():
        g = _gamma_from_opt(v)
        oi = _oi(v)
        if g > 0.0 and oi > 0.0:
            g_cnt += 1.0
            gex_puts += float(g) * float(oi) * float(contract_multiplier) * float(S) * float(S)

    gex = float(gex_calls) - float(gex_puts)
    return {
        "gex": float(gex),
        "gex_calls": float(gex_calls),
        "gex_puts": float(gex_puts),
        "gex_gamma_count": float(g_cnt),
    }


def _strike_to_symbol_map(opts: Dict[str, Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    for v in (opts or {}).values():
        try:
            k = float(v.get("strike") or 0.0)
            if k > 0.0:
                out[k] = v
        except Exception:
            continue
    return out


def calc_atm_microstructure(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
) -> Dict[str, float]:
    """Compute OH0-derived microstructure features around ATM.

    Notes:
    - Uses bid/ask and depth/qty fields when present (typically from OH0).
    - Best-effort: returns zeros when insufficient information is available.
    """

    empty = {
        "atm_spread_pct": 0.0,
        "atm_orderbook_imb": 0.0,
        "atm_liquidity_log": 0.0,
    }
    if float(underlying_price or 0.0) <= 0.0:
        return empty

    call_by_k = _strike_to_symbol_map(calls or {})
    put_by_k = _strike_to_symbol_map(puts or {})
    all_strikes = sorted(set(call_by_k) | set(put_by_k))
    if not all_strikes:
        return empty

    upx = float(underlying_price)
    atm = float(min(all_strikes, key=lambda s: abs(float(s) - upx)))

    # Prefer having both call+put at the same strike; fallback to whichever exists.
    c = call_by_k.get(atm) or {}
    p = put_by_k.get(atm) or {}
    targets = [v for v in (c, p) if isinstance(v, dict) and v]
    if not targets:
        return empty

    # Spread: average of available (ask-bid)/mid across call/put
    spreads = []
    for v in targets:
        try:
            bid = float(v.get("bid") or 0.0)
            ask = float(v.get("ask") or 0.0)
            if bid > 0.0 and ask > 0.0 and ask >= bid:
                mid = (ask + bid) / 2.0
                if mid > 0.0:
                    spreads.append((ask - bid) / mid * 100.0)
        except Exception:
            continue
    spread_pct = float(np.mean(spreads)) if spreads else 0.0

    # Imbalance: average L1 qty imbalance across call/put
    imbs = []
    for v in targets:
        try:
            bidq = 0.0
            askq = 0.0
            bqd = v.get("bid_qty_depth")
            aqd = v.get("ask_qty_depth")
            if isinstance(bqd, list) and bqd:
                bidq = float(bqd[0] or 0.0)
            if isinstance(aqd, list) and aqd:
                askq = float(aqd[0] or 0.0)
            denom = bidq + askq
            if denom > 0.0:
                imbs.append((bidq - askq) / denom)
        except Exception:
            continue
    imb = float(np.mean(imbs)) if imbs else 0.0

    # Liquidity: sum of depth qty for call+put, log-scaled
    liq = 0.0
    for v in targets:
        try:
            bqd = v.get("bid_qty_depth")
            aqd = v.get("ask_qty_depth")
            if isinstance(bqd, list):
                liq += float(np.nansum([float(x or 0.0) for x in bqd[:5]]))
            if isinstance(aqd, list):
                liq += float(np.nansum([float(x or 0.0) for x in aqd[:5]]))
        except Exception:
            continue
    liq_log = float(np.log1p(max(0.0, liq)))

    return {
        "atm_spread_pct": round(float(spread_pct), 4),
        "atm_orderbook_imb": round(float(imb), 4),
        "atm_liquidity_log": round(float(liq_log), 4),
    }


def calc_max_pain(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
) -> Dict[str, Any]:
    """Compute max pain strike and distance (%)."""

    empty = {"max_pain_price": 0.0, "max_pain_dist_pct": 0.0}

    call_oi: Dict[float, float] = {}
    for v in (calls or {}).values():
        try:
            k = float(v.get("strike") or 0.0)
            oi = float(v.get("open_interest") or 0.0)
            if k > 0.0:
                call_oi[k] = call_oi.get(k, 0.0) + oi
        except Exception:
            continue

    put_oi: Dict[float, float] = {}
    for v in (puts or {}).values():
        try:
            k = float(v.get("strike") or 0.0)
            oi = float(v.get("open_interest") or 0.0)
            if k > 0.0:
                put_oi[k] = put_oi.get(k, 0.0) + oi
        except Exception:
            continue

    all_strikes = sorted(set(call_oi) | set(put_oi))
    if len(all_strikes) < 2:
        return empty

    max_pain_s = float(all_strikes[0])
    try:
        strikes_arr = np.array(all_strikes, dtype=np.float64)

        call_k = np.array(list(call_oi.keys()), dtype=np.float64)
        call_v = np.array([float(call_oi[k]) for k in call_oi.keys()], dtype=np.float64)
        put_k = np.array(list(put_oi.keys()), dtype=np.float64)
        put_v = np.array([float(put_oi[k]) for k in put_oi.keys()], dtype=np.float64)

        pain_call = np.zeros((len(strikes_arr),), dtype=np.float64)
        if call_k.size:
            pain_call = np.sum(np.maximum(0.0, strikes_arr[:, None] - call_k[None, :]) * call_v[None, :], axis=1)

        pain_put = np.zeros((len(strikes_arr),), dtype=np.float64)
        if put_k.size:
            pain_put = np.sum(np.maximum(0.0, put_k[None, :] - strikes_arr[:, None]) * put_v[None, :], axis=1)

        pain_total = pain_call + pain_put
        max_pain_s = float(strikes_arr[int(np.argmin(pain_total))])
    except Exception:
        best_pain = float("inf")
        for s in all_strikes:
            pain = 0.0
            try:
                pain += sum(float(oi) * (float(s) - float(k)) for k, oi in call_oi.items() if float(k) < float(s))
                pain += sum(float(oi) * (float(k) - float(s)) for k, oi in put_oi.items() if float(k) > float(s))
            except Exception:
                continue

            if pain < best_pain:
                best_pain = float(pain)
                max_pain_s = float(s)

    upx = float(underlying_price or 0.0)
    dist_pct = round((upx - max_pain_s) / upx * 100.0, 4) if upx > 0.0 else 0.0

    return {
        "max_pain_price": round(float(max_pain_s), 2),
        "max_pain_dist_pct": float(dist_pct),
    }


def _get_atm_option_price(
    opts: Dict[str, Dict[str, Any]],
    atm_strike: float,
) -> float:
    """행사가가 atm_strike인 옵션의 체결가를 반환한다.

    tick_processor가 저장하는 'price' 필드를 1차로 사용하고,
    없으면 bid/ask 중간값으로 fallback한다.
    """
    for v in (opts or {}).values():
        try:
            if abs(float(v.get("strike") or 0.0) - float(atm_strike)) < 0.01:
                px = float(v.get("price") or 0.0)
                if px > 0.0:
                    return px
                # fallback: bid/ask 중간값
                bid = float(v.get("bid") or 0.0)
                ask = float(v.get("ask") or 0.0)
                if bid > 0.0 and ask > 0.0:
                    return (bid + ask) / 2.0
                if bid > 0.0:
                    return bid
                if ask > 0.0:
                    return ask
        except Exception:
            continue
    return 0.0



def build_option_snapshot(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
    *,
    tick_processor: Optional[Any] = None,
    option_feature_set: str = "v1",
    prev_underlying_price: Optional[float] = None,
    prev_atm_call_price: Optional[float] = None,
    prev_atm_put_price: Optional[float] = None,
    otm_open_min: float = 0.30,
    pcr_atm_strikes_each_side: int = 5,
    prev_oi_levels: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build a consolidated option feature snapshot.

    Args:
        prev_underlying_price: 직전 틱 선물가. v3/v4에서 call_vs_fut_ret_diff 계산에 사용.
        prev_atm_call_price:   직전 틱 ATM 콜 체결가. v3/v4에서만 사용.
        prev_atm_put_price:    직전 틱 ATM 풋 체결가. v3/v4에서만 사용.

    Feature sets:
        v1 : 기본 7개 (PCR, IV skew, max pain, microstructure)
        v2 : v1 + option micro-movement 9개
        v3 : v2 + parity divergence 7개
        v4 : v3 + premium bleed 6개
        v5 : v4 + OI 기반 지지저항 레벨 8개  ← 신규

    pcr_atm_strikes_each_side:
        ``calc_pcr``의 ATM±N 행사가 윈도(기본 5). 0이면 ATM 1줄만 합산.
    """

    snap: Dict[str, Any] = {
        "call_count": int(len(calls or {})),
        "put_count": int(len(puts or {})),
    }

    try:
        snap.update(
            calc_pcr(
                calls or {},
                puts or {},
                float(underlying_price or 0.0) or None,
                atm_strikes_each_side=max(0, min(50, int(pcr_atm_strikes_each_side))),
            )
        )
    except Exception:
        snap.update({"pcr_volume": 1.0, "pcr_oi": 1.0})

    try:
        snap.update(calc_iv_skew(calls or {}, puts or {}, float(underlying_price or 0.0)))
    except Exception:
        snap.update({"iv_skew": 1.0, "atm_call_iv": 0.0, "atm_put_iv": 0.0})

    # Backward-compat alias for OPT_KEYS.
    try:
        snap["atm_iv"] = float(snap.get("atm_call_iv") or 0.0)
    except Exception:
        snap["atm_iv"] = 0.0

    try:
        snap.update(calc_max_pain(calls or {}, puts or {}, float(underlying_price or 0.0)))
    except Exception:
        snap.update({"max_pain_price": 0.0, "max_pain_dist_pct": 0.0})

    try:
        snap.update(calc_atm_microstructure(calls or {}, puts or {}, float(underlying_price or 0.0)))
    except Exception:
        snap.update({"atm_spread_pct": 0.0, "atm_orderbook_imb": 0.0, "atm_liquidity_log": 0.0})

    try:
        snap.update(calc_gex(calls or {}, puts or {}, float(underlying_price or 0.0)))
    except Exception:
        snap.update({"gex": 0.0, "gex_calls": 0.0, "gex_puts": 0.0, "gex_gamma_count": 0.0})

    # Optional v2/v3/v4/v5: option micro-movement features from option_minute_ohlcv.
    try:
        fs = str(option_feature_set or "v1").strip().lower()
    except Exception:
        fs = "v1"
    if fs in ("v2", "v3", "v4", "v5"):
        try:
            from .option_flow_features import calc_option_minute_micro_features

            snap.update(
                calc_option_minute_micro_features(
                    tick_processor=tick_processor,
                    calls=calls or {},
                    puts=puts or {},
                    underlying_price=float(underlying_price or 0.0),
                    lookback_minutes=5,
                )
            )
        except Exception:
            snap.update(
                {
                    "optm_call_ret": 0.0,
                    "optm_put_ret": 0.0,
                    "optm_straddle_ret": 0.0,
                    "optm_call_range_pct": 0.0,
                    "optm_put_range_pct": 0.0,
                    "optm_straddle_range_pct": 0.0,
                    "optm_call_vol": 0.0,
                    "optm_put_vol": 0.0,
                    "optm_straddle_vol": 0.0,
                }
            )

    # Optional v3/v4/v5: call-put parity divergence features.
    # prev_* 파라미터는 pipeline._build_option_snapshot_safe()에서 직접 전달된다.
    # (이전의 snap.pop(_prev_*) 방식은 이중 실행을 유발하여 제거.)
    if fs in ("v3", "v4", "v5"):
        try:
            from core.utils import get_expiry_week_info
            expiry_info = get_expiry_week_info()
            dte = float(expiry_info.get("days_to_expiry") or 7.0)
        except Exception:
            dte = 7.0
        # Medium-07: DTE 역산 오차 방지 — snap에 days_to_expiry를 직접 노출.
        # pipeline._calc_amplitude_snapshot이 dte_weight_norm → DTE 역산 없이
        # 이 값을 직접 사용하면 클리핑 오차(max(1.0, min(30.0, ...)))가 제거된다.
        snap["days_to_expiry"] = float(dte)

        try:
            snap.update(
                calc_parity_divergence(
                    calls or {},
                    puts or {},
                    float(underlying_price or 0.0),
                    days_to_expiry=dte,
                    prev_underlying_price=prev_underlying_price,
                    prev_atm_call_price=prev_atm_call_price,
                    prev_atm_put_price=prev_atm_put_price,
                )
            )
        except Exception:
            snap.update(
                {
                    "parity_spread": 0.0,
                    "parity_spread_pct": 0.0,
                    "call_delta_proxy": 0.5,
                    "straddle_price": 0.0,
                    "straddle_vs_fut_move": 0.0,
                    "call_vs_fut_ret_diff": 0.0,
                    "dte_weight_norm": 0.0,
                    "parity_divergence_score": 0.0,
                }
            )

    # Optional v4/v5: premium bleed features (선물 상승 중 옵션 프리미엄 수축 탐지).
    # v3의 parity divergence 이후에 계산하며 dte를 재활용한다.
    # _dte_days: dte_weight_norm 역산 또는 get_expiry_week_info() 공통값.
    if fs in ("v4", "v5"):
        try:
            _dte_w = float(snap.get("dte_weight_norm") or 0.0)
            if _dte_w > 0.0:
                _dte_days = 1.0 / (_dte_w * 10.0)
            else:
                try:
                    from core.utils import get_expiry_week_info
                    _dte_days = float(get_expiry_week_info().get("days_to_expiry") or 7.0)
                except Exception:
                    _dte_days = 7.0
        except Exception:
            _dte_days = 7.0

        try:
            snap.update(
                calc_premium_bleed(
                    calls or {},
                    puts or {},
                    float(underlying_price or 0.0),
                    days_to_expiry=_dte_days,
                    prev_underlying_price=prev_underlying_price,
                    prev_atm_call_price=prev_atm_call_price,
                    prev_atm_put_price=prev_atm_put_price,
                )
            )
        except Exception:
            snap.update(
                {
                    "straddle_decay_vs_fut": 0.0,
                    "iv_crush_proxy": 0.0,
                    "fut_ret": 0.0,
                    "straddle_now": 0.0,
                    "straddle_prev": 0.0,
                    "premium_bleed_score": 0.0,
                }
            )

    # Optional v5: OI 기반 지지저항 레벨 피처.
    # Dealer Gamma Hedge Flow 구조(Zero Gamma, Vol Trigger) 포함.
    # 모델 차원 변경을 수반하므로 v5 전용으로 분리. v5 가중치 파일 필요.
    # [즉시 활용 옵션] LLM 컨텍스트용으로는 모든 fs에서 '_oi_levels' 키로 저장.
    # _oi_dte: v4/v5에서 이미 계산된 _dte_days 재사용, 그 외 별도 계산.
    try:
        _oi_dte = float(_dte_days)  # v4/v5: 위에서 이미 계산됨
    except NameError:
        try:
            _oi_dte_w = float(snap.get("dte_weight_norm") or 0.0)
            if _oi_dte_w > 0.0:
                _oi_dte = float(min(1.0 / (_oi_dte_w * 10.0), 365.0))
            else:
                try:
                    from core.utils import get_expiry_week_info
                    _oi_dte = float(get_expiry_week_info().get("days_to_expiry") or 7.0)
                except Exception:
                    _oi_dte = 7.0
        except Exception:
            _oi_dte = 7.0

    # ATM IV: calc_iv_skew()가 이미 snap에 저장한 값을 재사용한다.
    # IV 기반 동적 탐색 범위 산출 — IV 없으면 고정 fallback(20pt).
    _atm_iv_for_oi = float(snap.get("atm_call_iv") or snap.get("atm_iv") or 0.0)
    _iv_range_pt = calc_iv_peak_range(
        underlying_price=float(underlying_price or 0.0),
        atm_iv=_atm_iv_for_oi,
        days_to_expiry=_oi_dte,
    ) if _atm_iv_for_oi > 0.0 else 20.0

    try:
        _oi_calls = calls or {}
        _oi_puts = puts or {}
        try:
            _F = float(underlying_price or 0.0)
        except Exception:
            _F = 0.0
        if _F > 0.0:
            try:
                _gap = 2.5
                _atm = float(int(_F / _gap + 0.5) * _gap)
                # 선행 필터: IV 기반 범위 + 행사가 1칸 여유 확보.
                # calc_oi_levels 내부 _peak_range보다 조금 넓게 잡아
                # fallback 2단계(F 이상 전체)가 작동할 수 있도록 한다.
                _filter_half = float(_iv_range_pt) + float(_gap)
                _lo = float(_atm - _filter_half)
                _hi = float(_atm + _filter_half)
                _oi_calls = {
                    k: v
                    for k, v in (calls or {}).items()
                    if isinstance(v, dict) and _lo <= float(v.get("strike") or 0.0) <= _hi
                }
                _oi_puts = {
                    k: v
                    for k, v in (puts or {}).items()
                    if isinstance(v, dict) and _lo <= float(v.get("strike") or 0.0) <= _hi
                }
            except Exception:
                _oi_calls = calls or {}
                _oi_puts = puts or {}
        _realized_hl = 0.0
        _session_high = 0.0
        _session_low = 0.0
        try:
            if tick_processor is not None and hasattr(tick_processor, "get_daily_session_ohlc"):
                _daily = tick_processor.get_daily_session_ohlc() or {}
                _sh = float(_daily.get("session_high") or 0.0)
                _sl = float(_daily.get("session_low") or 0.0)
                _session_high = float(_sh)
                _session_low = float(_sl)
                if _sh > 0.0 and _sl > 0.0 and _sh >= _sl:
                    _realized_hl = float(_sh - _sl)
        except Exception:
            _realized_hl = 0.0
            _session_high = 0.0
            _session_low = 0.0

        _oi_result = calc_oi_levels(
            _oi_calls,
            _oi_puts,
            float(underlying_price or 0.0),
            default_days_to_expiry=_oi_dte,
            atm_iv=_atm_iv_for_oi,
            realized_hl_range_pt=float(_realized_hl),
            session_high=float(_session_high),
            session_low=float(_session_low),
        )
        # 모든 fs: LLM 컨텍스트 참조용으로 _oi_levels 저장 (모델 피처 차원 불변)
        snap["_oi_levels"] = _oi_result
        # ── OI 변화율(velocity) 계산 — 이전 스냅샷 대비 ──────────────────────
        # prev_oi_levels가 전달되면 call/put peak의 절대·정규화 변화량을 산출한다.
        # OI 급감: 포지션 청산 → 기존 지지/저항 약화 신호
        # OI 급증: 신규 포지션 → 지지/저항 강화 신호
        try:
            _prev = prev_oi_levels if isinstance(prev_oi_levels, dict) else {}
            _oi_delta: Dict[str, float] = {
                "call_peak_delta":  float(_oi_result.get("call_oi_peak", 0.0))
                                  - float(_prev.get("call_oi_peak", 0.0)),
                "put_peak_delta":   float(_oi_result.get("put_oi_peak", 0.0))
                                  - float(_prev.get("put_oi_peak", 0.0)),
                "call_norm_delta":  float(_oi_result.get("call_oi_peak_norm", 0.0))
                                  - float(_prev.get("call_oi_peak_norm", 0.0)),
                "put_norm_delta":   float(_oi_result.get("put_oi_peak_norm", 0.0))
                                  - float(_prev.get("put_oi_peak_norm", 0.0)),
                "range_delta":      float(_oi_result.get("oi_range_pct", 0.0))
                                  - float(_prev.get("oi_range_pct", 0.0)),
            }
            snap["_oi_delta"] = _oi_delta
        except Exception:
            snap["_oi_delta"] = {}
        import logging as _logging
        _logger_oif = _logging.getLogger(__name__)
        _call_pk = float(_oi_result.get("call_oi_peak") or 0.0)
        _put_pk  = float(_oi_result.get("put_oi_peak")  or 0.0)
        _rng_used = float(_oi_result.get("peak_search_range_used") or 0.0)
        _logger_oif.debug(
            "[build_option_snapshot] _oi_levels 세팅 완료 "
            "call_peak=%.2f put_peak=%.2f oi_range_pct=%.4f "
            "call_norm=%.3f put_norm=%.3f peak_search_range=%.1fpt fs=%s",
            _call_pk,
            _put_pk,
            float(_oi_result.get("oi_range_pct") or 0.0),
            float(_oi_result.get("call_oi_peak_norm") or 0.0),
            float(_oi_result.get("put_oi_peak_norm")  or 0.0),
            _rng_used,
            fs,
        )
        if _call_pk <= 0.0 and _put_pk <= 0.0:
            _logger_oif.debug(
                "[build_option_snapshot] _oi_levels call/put peak 모두 0 — "
                "OI 데이터 없음 또는 peak_search_range(%.1fpt) 내 후보 없음 fs=%s",
                _rng_used, fs,
            )
        # v5 전용: OPT_KEYS_V5에 포함되는 8개 피처를 snap에 직접 노출
        if fs == "v5":
            snap.update({
                "dist_to_call_peak":  _oi_result["dist_to_call_peak"],
                "dist_to_put_peak":   _oi_result["dist_to_put_peak"],
                "oi_center_dist_pct": _oi_result["oi_center_dist_pct"],
                "oi_range_pct":       _oi_result["oi_range_pct"],
                "call_oi_peak_norm":  _oi_result["call_oi_peak_norm"],
                "put_oi_peak_norm":   _oi_result["put_oi_peak_norm"],
                "above_vol_trigger":  _oi_result["above_vol_trigger"],
                "zero_gamma_dist_pct": _oi_result["zero_gamma_dist_pct"],
            })
    except Exception as _oi_exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[build_option_snapshot] calc_oi_levels 실패 — _oi_levels={} fallback: %s", _oi_exc
        )
        snap["_oi_levels"] = {}
        if fs == "v5":
            snap.update({
                "dist_to_call_peak":  0.0,
                "dist_to_put_peak":   0.0,
                "oi_center_dist_pct": 0.0,
                "oi_range_pct":       0.0,
                "call_oi_peak_norm":  0.0,
                "put_oi_peak_norm":   0.0,
                "above_vol_trigger":  1.0,
                "zero_gamma_dist_pct": 0.0,
            })

    # OTM 프리미엄 변화율 — 모든 feature set에서 공통 계산.
    # tick_processor.set_option_open_map()으로 주입된 open_price 기반.
    # open_price가 없는 종목은 자동 제외되므로 장 초기에는 count=0으로 반환됨.
    try:
        snap["otm_premium"] = calc_otm_premium_change(
            calls or {},
            puts or {},
            float(underlying_price or 0.0),
            otm_open_min=float(otm_open_min),
        )
    except Exception:
        snap["otm_premium"] = {
            "call_otm_prem_chg": None,
            "put_otm_prem_chg":  None,
            "call_otm_count":    0,
            "put_otm_count":     0,
        }

    # ── 옵션 가격 레벨 탐색 (고가/저가 기준) ─────────────────────────────────
    # 탐색 레벨: 1.20, 2.50, 3.50, 4.85, 5.50 (기본값; config 오버라이드 가능)
    # 결과는 '_price_level_scan' 키에 저장 — 모델 피처 차원에 영향 없음.
    # context_builder와 telegram_formatters에서 LLM 컨텍스트·알림 생성에 활용.
    try:
        from .option_price_level_scan import scan_option_price_levels
        snap["_price_level_scan"] = scan_option_price_levels(
            calls or {},
            puts or {},
            underlying_price=float(underlying_price or 0.0),
        )
    except Exception as _pls_exc:
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "[build_option_snapshot] price_level_scan 실패(무시): %s", _pls_exc
        )
        snap["_price_level_scan"] = {
            "call_hits": [], "put_hits": [],
            "hit_levels_call": [], "hit_levels_put": [],  # set() → [] (JSON 직렬화 호환)
            "has_hit": False, "summary": "",
        }

    return snap


# ═══════════════════════════════════════════════════════════════════════════════
# FuturesCallSimilarity — 선물 vs ATM 콜 그래프 유사도 측정 (만기 이탈 탐지)
# ═══════════════════════════════════════════════════════════════════════════════

