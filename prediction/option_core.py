"""option_features 공통 헬퍼 함수.

ATM 탐색, Black-Scholes Gamma proxy 등 여러 calculator가 공유하는 순수 함수.
이 파일은 다른 option_* 모듈들이 import하는 공통 기반이다.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import numpy as np

def _find_atm_strike(
    strikes: list,
    underlying_price: float,
    tol: float = 2.5,
) -> Optional[float]:
    """행사가 목록에서 ATM 행사가를 찾는다.

    KP200 옵션 행사가 간격(2.5pt) 기준으로 nearest-even 반올림하여 ATM anchor를
    구한 뒤, tol 범위 내에서 가장 가까운 실제 행사가를 반환한다.

    Args:
        strikes:          정렬된 float 행사가 리스트.
        underlying_price: 현재 기초자산(선물) 가격.
        tol:              ATM anchor 기준 탐색 허용 오차(pt). 기본 2.5.

    Returns:
        ATM 행사가(float). 후보가 없으면 리스트 중 가장 가까운 값 반환.
        strikes가 빈 리스트면 None.
    """
    if not strikes:
        return None
    F = float(underlying_price or 0.0)
    if F <= 0.0:
        return None
    try:
        anchor = round(F * 2.0) / 2.0
    except Exception:
        anchor = F
    candidates = [float(s) for s in strikes if abs(float(s) - anchor) <= float(tol)]
    if not candidates:
        candidates = [float(s) for s in strikes]
    return float(min(candidates, key=lambda s: abs(s - anchor)))


def _bs_gamma_proxy(
    S: float,
    K: float,
    iv: float,
    T: float,
) -> float:
    """Black-Scholes Gamma 근사값을 반환한다.

    calc_gex()와 calc_oi_levels()가 동일한 공식을 내부 중첩 함수로 각각 구현하던
    것을 모듈 레벨 단일 함수로 통합한다.

    Args:
        S:  현재 기초자산 가격.
        K:  행사가.
        iv: 내재변동성 (소수. 예: 0.20 = 20%).
        T:  잔존기간(년). 예: 7/365.

    Returns:
        Gamma 값(float). 계산 불가 시 0.0.
    """
    try:
        if S <= 0.0 or K <= 0.0 or iv <= 0.0 or T <= 0.0:
            return 0.0
        d1 = (math.log(S / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
        phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        return float(phi / (S * iv * math.sqrt(T)))
    except Exception:
        return 0.0

def _get_atm_option_price(
    opts: Dict[str, Any],
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


