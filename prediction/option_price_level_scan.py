"""옵션 콜·풋 고가/저가 레벨 터치 탐색 모듈.

특정 가격 레벨(1.20, 2.50, 3.50, 4.85, 5.50)에 당일 고가 또는 저가가
정확히 일치했는지 스캔하여 결과를 반환한다.

판정 기준:
    - 허용 오차 = 0.0pt (완전 일치).
    - 부동소수점 비교 오차 방지: round(value, 2) 후 == level 비교.
    - 예: high=2.5000001 → round → 2.50 → 레벨 2.50 매칭.
    - 예: high=2.51 → round → 2.51 → 불일치 → 제외.

데이터 소스:
    1. OC0 실시간 틱 → tick_processor.call_options/put_options["high"/"low"/"timestamp"]
    2. t2301 주기 갱신 → update_oi_from_t2301() 경유 동일 딕셔너리

hit 항목 구조::
    {
        "symbol":   str,    # 옵션 심볼 코드
        "opt_type": str,    # "call" | "put"
        "strike":   float,  # 행사가(pt)
        "field":    str,    # "high" | "low" | "price"
        "value":    float,  # round(실제값, 2)
        "level":    float,  # 매칭된 레벨
        "time_str": str,    # 발생시각 HHMM (예: "0911"), 없으면 ""
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── 기본 탐색 레벨 ────────────────────────────────────────────────────────────
DEFAULT_SCAN_LEVELS: List[float] = [1.20, 2.50, 3.50, 4.85, 5.50]

# 허용 오차 = 0.0 → round(value, 2) == level 완전 일치만 통과
DEFAULT_TOLERANCE: float = 0.0


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_time_str(raw_ts: Any) -> str:
    """OC0 chetime(HHMMSS 6자리)에서 HHMM 4자리를 추출한다.

    Args:
        raw_ts: tick["timestamp"] 값. 예: "091130", "09:11:30".

    Returns:
        "HHMM" 4자리 문자열. 파싱 불가이면 "" 반환.
    """
    try:
        digits = "".join(c for c in str(raw_ts or "") if c.isdigit())
        if len(digits) >= 4:
            return digits[:4]
    except Exception:
        pass
    return ""


def _match_level(value: float, levels: List[float]) -> Optional[float]:
    """round(value, 2) 가 levels 중 하나와 정확히 일치하면 해당 레벨을 반환.

    부동소수점 == 비교는 round 후 수행하므로 누적 오차를 흡수한다.
    """
    if value <= 0.0:
        return None
    v = round(value, 2)
    for lv in levels:
        if v == lv:
            return lv
    return None


# ── 공개 API ──────────────────────────────────────────────────────────────────

def scan_option_price_levels(
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    *,
    levels: Optional[List[float]] = None,
    check_high: bool = True,
    check_low: bool = True,
    check_price: bool = False,
    underlying_price: float = 0.0,
    atm_range_pt: float = 0.0,
) -> Dict[str, Any]:
    """콜·풋 옵션 고가/저가가 지정 레벨과 정확히 일치하는지 스캔한다.

    판정: round(high or low, 2) == level (완전 일치, 허용 오차 없음).

    Args:
        calls:            tick_processor.call_options (symbol → data dict).
        puts:             tick_processor.put_options  (symbol → data dict).
        levels:           탐색 레벨 리스트. None → DEFAULT_SCAN_LEVELS.
        check_high:       당일 고가 탐색. 기본 True.
        check_low:        당일 저가 탐색. 기본 True.
        check_price:      현재가 탐색. 기본 False.
        underlying_price: 선물 현재가(pt). atm_range_pt > 0 이면 ATM 근방 필터 활성.
        atm_range_pt:     ATM 근방 필터 반경(pt). 0 → 전 행사가 스캔.

    Returns:
        {
            "call_hits":       List[hit],
            "put_hits":        List[hit],
            "hit_levels_call": Set[float],
            "hit_levels_put":  Set[float],
            "has_hit":         bool,
            "summary":         str,
            "levels_used":     List[float],
        }
    """
    _levels: List[float] = [
        round(float(lv), 2)
        for lv in (levels or DEFAULT_SCAN_LEVELS)
        if float(lv) > 0.0
    ]
    if not _levels:
        return _empty_result(_levels)

    _upx = _safe_float(underlying_price)
    _rng = _safe_float(atm_range_pt)
    use_atm_filter = (_upx > 0.0 and _rng > 0.0)

    call_hits: List[Dict[str, Any]] = []
    put_hits:  List[Dict[str, Any]] = []
    hit_levels_call: Set[float] = set()
    hit_levels_put:  Set[float] = set()

    fields_to_check: List[str] = []
    if check_high:
        fields_to_check.append("high")
    if check_low:
        fields_to_check.append("low")
    if check_price:
        fields_to_check.append("price")
    if not fields_to_check:
        return _empty_result(_levels)

    def _scan_opts(
        opts: Dict[str, Dict[str, Any]],
        opt_type: str,
        hit_list: List[Dict[str, Any]],
        hit_set: Set[float],
    ) -> None:
        for sym, data in (opts or {}).items():
            if not isinstance(data, dict):
                continue
            strike = _safe_float(data.get("strike"))
            if use_atm_filter and abs(strike - _upx) > _rng:
                continue

            # OC0 틱의 chetime이 "timestamp" 키(HHMMSS 문자열)로 저장됨
            time_str = _parse_time_str(data.get("timestamp") or "")

            for fld in fields_to_check:
                val_raw = _safe_float(data.get(fld))
                matched_lv = _match_level(val_raw, _levels)
                if matched_lv is None:
                    continue
                hit_list.append({
                    "symbol":   str(sym),
                    "opt_type": opt_type,
                    "strike":   strike,
                    "field":    fld,
                    "value":    round(val_raw, 2),
                    "level":    matched_lv,
                    "time_str": time_str,
                })
                hit_set.add(matched_lv)

    try:
        _scan_opts(calls or {}, "call", call_hits, hit_levels_call)
    except Exception as exc:
        logger.warning("[PriceLevelScan] 콜 스캔 실패: %s", exc)

    try:
        _scan_opts(puts or {}, "put", put_hits, hit_levels_put)
    except Exception as exc:
        logger.warning("[PriceLevelScan] 풋 스캔 실패: %s", exc)

    call_hits.sort(key=lambda x: (x["level"], x["strike"]))
    put_hits.sort(key=lambda x: (x["level"], x["strike"]))

    has_hit = bool(call_hits or put_hits)
    summary = _build_summary(call_hits, put_hits)

    if has_hit:
        logger.debug(
            "[PriceLevelScan] 터치 감지: call=%d put=%d | %s",
            len(call_hits), len(put_hits), summary,
        )

    return {
        "call_hits":       call_hits,
        "put_hits":        put_hits,
        "hit_levels_call": sorted(hit_levels_call),  # set → list (JSON 직렬화 호환)
        "hit_levels_put":  sorted(hit_levels_put),   # set → list (JSON 직렬화 호환)
        "has_hit":         has_hit,
        "summary":         summary,
        "levels_used":     _levels,
    }


def _empty_result(levels: List[float]) -> Dict[str, Any]:
    return {
        "call_hits":       [],
        "put_hits":        [],
        "hit_levels_call": [],  # set() → [] (JSON 직렬화 호환)
        "hit_levels_put":  [],  # set() → [] (JSON 직렬화 호환)
        "has_hit":         False,
        "summary":         "",
        "levels_used":     levels,
    }


def _build_summary(
    call_hits: List[Dict[str, Any]],
    put_hits:  List[Dict[str, Any]],
) -> str:
    """hit 목록에서 로그용 한 줄 요약을 생성한다.

    예: "콜 385.0 고가=2.50@0911 | 풋 382.5 저가=1.20@1023"
    """
    _FIELD_KR = {"high": "고가", "low": "저가", "price": "현재가"}
    parts: List[str] = []

    for hits, label in ((call_hits, "콜"), (put_hits, "풋")):
        if not hits:
            continue
        by_level: Dict[float, List[Dict]] = {}
        for h in hits:
            by_level.setdefault(h["level"], []).append(h)
        for lv in sorted(by_level):
            grp = by_level[lv]
            fld_kr = _FIELD_KR.get(grp[0]["field"], grp[0]["field"])
            items = []
            for h in grp[:4]:
                t = f"@{h['time_str']}" if h.get("time_str") else ""
                items.append(f"{h['strike']:.1f} {fld_kr}={lv:.2f}{t}")
            if len(grp) > 4:
                items.append(f"외{len(grp)-4}")
            parts.append(f"{label} " + " / ".join(items))

    return " | ".join(parts) if parts else ""


# ── config 연동 헬퍼 ──────────────────────────────────────────────────────────

def scan_levels_from_config(config: Any) -> List[float]:
    """AppConfig에서 레벨 리스트를 읽는다.

    config.json 예시::
        {
          "prediction": {
            "price_level_scan": {
              "levels": [1.20, 2.50, 3.50, 4.85, 5.50]
            }
          }
        }

    Returns:
        레벨 리스트 (없으면 DEFAULT_SCAN_LEVELS).
    """
    try:
        pred = getattr(config, "prediction", None)
        scan_cfg = getattr(pred, "price_level_scan", None)
        raw_levels = (
            scan_cfg.get("levels") if isinstance(scan_cfg, dict)
            else getattr(scan_cfg, "levels", None)
        )
        if isinstance(raw_levels, (list, tuple)) and raw_levels:
            return [round(float(lv), 2) for lv in raw_levels if float(lv) > 0.0]
    except Exception:
        pass
    return list(DEFAULT_SCAN_LEVELS)
