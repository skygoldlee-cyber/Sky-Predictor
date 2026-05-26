"""Option symbol utilities used by eBest live mode."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from core.strike_utils import extract_strike_pt

logger = logging.getLogger(__name__)


CALL_SYMBOL_PREFIX = "B016"
PUT_SYMBOL_PREFIX = "C016"
STRIKE_GAP = 2.5


def _extract_strike_quiet(symbol: str) -> Optional[float]:
    """심볼 코드에서 행사가를 파싱한다(조용히 실패).

    eBest 옵션 심볼(B016xxx / C016xxx)의 마지막 3자리 행사가 코드를
    strike_utils.extract_strike_pt()에 위임하여 변환한다.

    Args:
        symbol: 옵션 심볼 문자열 (예: "B016250385", "C016250A01").

    Returns:
        행사가(float). 파싱 실패/범위 밖이면 None.
    """
    s = str(symbol or "").strip()
    if len(s) < 8:
        return None
    if not (s.startswith(CALL_SYMBOL_PREFIX) or s.startswith(PUT_SYMBOL_PREFIX)):
        logger.debug("[STRIKE] unknown symbol prefix: %s", s[:4])
        return None

    return extract_strike_pt(s)


def _round_to_strike_gap(price: float, gap: float = STRIKE_GAP) -> float:
    """현재가를 행사가 그리드(기본 2.5)로 반올림한다."""
    p, g = float(price), float(gap)
    if p <= 0 or g <= 0:
        return 0.0
    return float(int(p / g + 0.5) * g)


def _filter_option_symbols_by_atm(
    predictor: Any,
    *,
    calls: List[str],
    puts: List[str],
    itm_count: int,
    otm_count_call: int,
    otm_count_put: int,
    underlying_price: Optional[float] = None,
) -> Tuple[List[str], List[str], Optional[float]]:
    """ATM 기준으로 ITM/OTM 옵션 심볼을 선택한다.

    Args:
        predictor: `tick_processor.get_current_price()`를 제공하는 객체.
        calls/puts: 월물에 해당하는 콜/풋 옵션 심볼 리스트.
        itm_count: ATM 기준 ITM 쪽으로 선택할 개수.
        otm_count_call: OTM 콜 선택 개수.
        otm_count_put: OTM 풋 선택 개수.
        underlying_price: 기초자산 현재가(주어지지 않으면 predictor에서 조회).

    Returns:
        (sel_calls, sel_puts, atm_strike)
        - sel_calls/sel_puts: 구독 대상 심볼 리스트
        - atm_strike: 선택된 ATM 행사가
    """
    try:
        current_price = float(
            underlying_price
            if underlying_price is not None
            else (predictor.tick_processor.get_current_price() or 0.0)
        )
    except Exception:
        current_price = 0.0

    if float(current_price or 0.0) <= 0.0:
        try:
            pc = getattr(predictor, "kp200_prev_close", None)
            if pc is not None and float(pc) > 0.0:
                current_price = float(pc)
        except Exception:
            pass

    call_items = [(s, k) for s in calls if (k := _extract_strike_quiet(s)) is not None]
    put_items = [(s, k) for s in puts if (k := _extract_strike_quiet(s)) is not None]

    strikes = sorted({k for _, k in call_items} | {k for _, k in put_items})
    if not strikes:
        return [], [], None

    target_atm = _round_to_strike_gap(current_price, STRIKE_GAP) if current_price > 0 else None
    atm_strike = (
        min(strikes, key=lambda x: abs(x - target_atm))
        if target_atm is not None
        else strikes[len(strikes) // 2]
    )

    call_items_sorted = sorted(call_items, key=lambda x: x[1])
    put_items_sorted = sorted(put_items, key=lambda x: x[1])

    call_itm = [(s, k) for s, k in call_items_sorted if k < atm_strike]
    call_atm = [(s, k) for s, k in call_items_sorted if k == atm_strike]
    call_otm = [(s, k) for s, k in call_items_sorted if k > atm_strike]

    put_otm = [(s, k) for s, k in put_items_sorted if k < atm_strike]
    put_atm = [(s, k) for s, k in put_items_sorted if k == atm_strike]
    put_itm = [(s, k) for s, k in put_items_sorted if k > atm_strike]

    sel_calls = [s for s, _ in call_itm[-itm_count:] + call_atm[:1] + call_otm[:otm_count_call]]
    sel_puts = [s for s, _ in put_otm[-otm_count_put:] + put_atm[:1] + put_itm[:itm_count]]

    return sel_calls, sel_puts, float(atm_strike)


def filter_option_symbols_dynamic_otm_by_open(
    predictor: Any,
    *,
    calls: List[str],
    puts: List[str],
    itm_count: int,
    underlying_price: Optional[float] = None,
    call_open_map: Optional[dict] = None,
    put_open_map: Optional[dict] = None,
    otm_open_min: float = 0.30,
    max_otm_calls: int = 0,
    max_otm_puts: int = 0,
) -> Tuple[List[str], List[str], Optional[float]]:
    try:
        current_price = float(
            underlying_price
            if underlying_price is not None
            else (predictor.tick_processor.get_current_price() or 0.0)
        )
    except Exception:
        current_price = 0.0

    call_items = [(s, k) for s in calls if (k := _extract_strike_quiet(s)) is not None]
    put_items = [(s, k) for s in puts if (k := _extract_strike_quiet(s)) is not None]

    strikes = sorted({k for _, k in call_items} | {k for _, k in put_items})
    if not strikes:
        return [], [], None

    target_atm = _round_to_strike_gap(current_price, STRIKE_GAP) if current_price > 0 else None
    atm_strike = (
        min(strikes, key=lambda x: abs(x - target_atm))
        if target_atm is not None
        else strikes[len(strikes) // 2]
    )

    call_items_sorted = sorted(call_items, key=lambda x: x[1])
    put_items_sorted = sorted(put_items, key=lambda x: x[1])

    call_itm = [(s, k) for s, k in call_items_sorted if k < atm_strike]
    call_atm = [(s, k) for s, k in call_items_sorted if k == atm_strike]
    call_otm = [(s, k) for s, k in call_items_sorted if k > atm_strike]

    put_otm = [(s, k) for s, k in put_items_sorted if k < atm_strike]
    put_atm = [(s, k) for s, k in put_items_sorted if k == atm_strike]
    put_itm = [(s, k) for s, k in put_items_sorted if k > atm_strike]

    call_open_map = call_open_map if isinstance(call_open_map, dict) else {}
    put_open_map = put_open_map if isinstance(put_open_map, dict) else {}

    sel_calls_itm_atm = [s for s, _ in (call_itm[-itm_count:] + call_atm[:1])]
    sel_puts_otm_atm_itm = [s for s, _ in (put_atm[:1] + put_itm[:itm_count])]

    sel_calls_otm = [
        s
        for s, _ in call_otm
        if float(call_open_map.get(s) or 0.0) >= float(otm_open_min)
    ]
    sel_puts_otm = [
        s
        for s, _ in put_otm
        if float(put_open_map.get(s) or 0.0) >= float(otm_open_min)
    ]

    try:
        mcc = int(max_otm_calls or 0)
    except Exception:
        mcc = 0
    try:
        mpp = int(max_otm_puts or 0)
    except Exception:
        mpp = 0

    if mcc > 0:
        sel_calls_otm = sel_calls_otm[:mcc]
    if mpp > 0:
        sel_puts_otm = sel_puts_otm[-mpp:]

    sel_calls = sel_calls_itm_atm + sel_calls_otm
    sel_puts = sel_puts_otm + sel_puts_otm_atm_itm
    return sel_calls, sel_puts, float(atm_strike)


def select_oi_window_symbols(
    calls: List[str],
    puts: List[str],
    underlying_price: float,
    *,
    itm_count: int = 10,
    otm_count: int = 10,
    strike_gap: float = STRIKE_GAP,
) -> Tuple[List[str], List[str], Optional[float]]:
    """OI 지지저항 분석을 위해 ATM 기준 내가(ITM) N개 + 외가(OTM) N개 심볼을 선택한다.

    OI 로직이 제대로 작동하려면 ATM 양쪽 충분한 행사가의 open_interest 데이터가
    실시간으로 수신돼야 한다. 이 함수는 콜/풋 각각 독립적으로 선택하여
    ATM을 중심으로 고르게 분포된 구독 목록을 반환한다.

    선택 기준 (콜 기준):
        - ITM 콜 : ATM 아래 행사가 중 가까운 순으로 itm_count개
        - ATM 콜 : 정확히 1개 (ATM 행사가)
        - OTM 콜 : ATM 위 행사가 중 가까운 순으로 otm_count개

    풋은 ITM/OTM 방향이 콜과 반대이나, 결과적으로 ATM 양쪽 동일 행사가 범위를 커버한다.

    Args:
        calls:            전체 콜 옵션 심볼 리스트 (월물 필터 완료 상태).
        puts:             전체 풋 옵션 심볼 리스트 (월물 필터 완료 상태).
        underlying_price: 현재 기초자산(선물) 가격.
        itm_count:        ATM 기준 내가 방향 선택 개수 (기본 10).
        otm_count:        ATM 기준 외가 방향 선택 개수 (기본 10).
        strike_gap:       행사가 간격 (기본 2.5pt).

    Returns:
        (sel_calls, sel_puts, atm_strike)
        - sel_calls : 최대 itm_count + 1 + otm_count 개의 콜 심볼 리스트
        - sel_puts  : 최대 itm_count + 1 + otm_count 개의 풋 심볼 리스트
        - atm_strike: 선택된 ATM 행사가 (float). 선택 불가 시 None.

    Notes:
        - 심볼 목록에 해당 행사가가 없으면 해당 슬롯은 건너뜁니다.
        - itm_count=10, otm_count=10 설정 시 콜/풋 각 최대 21개(ATM 포함),
          총 최대 42개 심볼을 반환합니다.
        - OI 계산은 OC0(틱) 수신 데이터의 open_interest 필드를 사용하므로
          반드시 OC0로 구독된 심볼이어야 합니다.
    """
    upx = float(underlying_price or 0.0)
    if upx <= 0.0:
        return [], [], None

    g = max(float(strike_gap or STRIKE_GAP), 0.5)
    atm_target = float(int(upx / g + 0.5) * g)

    call_items = [(s, k) for s in (calls or []) if (k := _extract_strike_quiet(s)) is not None]
    put_items  = [(s, k) for s in (puts  or []) if (k := _extract_strike_quiet(s)) is not None]

    all_strikes = sorted({k for _, k in call_items} | {k for _, k in put_items})
    if not all_strikes:
        return [], [], None

    atm_strike = float(min(all_strikes, key=lambda x: abs(x - atm_target)))

    # 행사가 → 심볼 매핑 (중복 행사가는 마지막 심볼 사용)
    call_by_k: Dict[float, str] = {}
    for s, k in call_items:
        call_by_k[k] = s
    put_by_k: Dict[float, str] = {}
    for s, k in put_items:
        put_by_k[k] = s

    sorted_strikes = sorted(all_strikes)
    try:
        atm_idx = sorted_strikes.index(atm_strike)
    except ValueError:
        atm_idx = len(sorted_strikes) // 2

    # ATM 기준 내가/외가 행사가 슬라이스
    # lower_strikes: ATM 아래 (가까운 순, 즉 역순 정렬 후 슬라이스)
    # upper_strikes: ATM 위 (가까운 순)
    lower_strikes = sorted_strikes[max(0, atm_idx - itm_count): atm_idx]   # ATM 아래 itm_count개
    upper_strikes = sorted_strikes[atm_idx + 1: atm_idx + 1 + otm_count]   # ATM 위 otm_count개

    # ── 콜: ITM = lower(행사가 낮음), ATM = atm, OTM = upper(행사가 높음)
    sel_calls: List[str] = []
    for k in lower_strikes:                 # 콜 ITM (낮은 → 높은 순)
        if k in call_by_k:
            sel_calls.append(call_by_k[k])
    if atm_strike in call_by_k:            # 콜 ATM
        sel_calls.append(call_by_k[atm_strike])
    for k in upper_strikes:                # 콜 OTM (낮은 → 높은 순)
        if k in call_by_k:
            sel_calls.append(call_by_k[k])

    # ── 풋: OTM = lower(행사가 낮음), ATM = atm, ITM = upper(행사가 높음)
    # OI 분포 파악 목적이므로 콜과 동일한 행사가 범위를 커버하면 충분함
    sel_puts: List[str] = []
    for k in lower_strikes:                # 풋 OTM (낮은 → 높은 순)
        if k in put_by_k:
            sel_puts.append(put_by_k[k])
    if atm_strike in put_by_k:            # 풋 ATM
        sel_puts.append(put_by_k[atm_strike])
    for k in upper_strikes:               # 풋 ITM (낮은 → 높은 순)
        if k in put_by_k:
            sel_puts.append(put_by_k[k])

    return sel_calls, sel_puts, float(atm_strike)
