from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _nearest_atm_symbols(
    *,
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (call_symbol, put_symbol) nearest to ATM strike, best-effort."""

    upx = float(underlying_price or 0.0)
    if upx <= 0.0:
        return None, None

    call_by_k: Dict[float, str] = {}
    for sym, v in (calls or {}).items():
        try:
            k = float((v or {}).get("strike") or 0.0)
            if k > 0.0:
                call_by_k[k] = str(sym)
        except Exception:
            continue

    put_by_k: Dict[float, str] = {}
    for sym, v in (puts or {}).items():
        try:
            k = float((v or {}).get("strike") or 0.0)
            if k > 0.0:
                put_by_k[k] = str(sym)
        except Exception:
            continue

    strikes = sorted(set(call_by_k.keys()) | set(put_by_k.keys()))
    if not strikes:
        return None, None

    atm = float(min(strikes, key=lambda s: abs(float(s) - upx)))
    return call_by_k.get(atm), put_by_k.get(atm)


def _last_minute_bar_features(df) -> Dict[str, float]:
    """Return return/range/volume from the last row of an option minute DF."""

    if df is None or getattr(df, "empty", True):
        return {"ret": 0.0, "range_pct": 0.0, "vol": 0.0}

    try:
        row = df.iloc[-1]
    except Exception:
        return {"ret": 0.0, "range_pct": 0.0, "vol": 0.0}

    # [IMP-8-4] pd.Series에는 .get()이 있으므로 직접 사용.
    # hasattr 체크로 dict/Series 모두 안전하게 처리한다.
    def _row_get(key: str) -> float:
        try:
            if hasattr(row, "get"):
                return _safe_float(row.get(key, 0.0))
            return _safe_float(row[key])
        except Exception:
            return 0.0

    o = _row_get("open")
    h = _row_get("high")
    l = _row_get("low")
    c = _row_get("close")
    v = _row_get("volume")

    ret = (c - o) / o if o > 0.0 else 0.0
    range_pct = (h - l) / c if c > 0.0 else 0.0

    if not np.isfinite(ret):
        ret = 0.0
    if not np.isfinite(range_pct):
        range_pct = 0.0
    if not np.isfinite(v):
        v = 0.0

    return {"ret": float(ret), "range_pct": float(range_pct), "vol": float(max(0.0, v))}


def calc_option_minute_micro_features(
    *,
    tick_processor: Any,
    calls: Dict[str, Dict[str, Any]],
    puts: Dict[str, Dict[str, Any]],
    underlying_price: float,
    lookback_minutes: int = 5,
) -> Dict[str, float]:
    """Compute extra option micro-movement features from option_minute_ohlcv.

    Requires tick_processor.get_option_minute_df(symbol, minutes).
    Best-effort: returns zeros if unavailable.
    """

    out: Dict[str, float] = {
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

    getter = getattr(tick_processor, "get_option_minute_df", None)
    if not callable(getter):
        return out

    call_sym, put_sym = _nearest_atm_symbols(calls=calls, puts=puts, underlying_price=float(underlying_price or 0.0))
    if not call_sym and not put_sym:
        return out

    c_df = None
    p_df = None
    try:
        if call_sym:
            c_df = getter(call_sym, minutes=int(lookback_minutes))
    except Exception:
        c_df = None
    try:
        if put_sym:
            p_df = getter(put_sym, minutes=int(lookback_minutes))
    except Exception:
        p_df = None

    cf = _last_minute_bar_features(c_df)
    pf = _last_minute_bar_features(p_df)

    out["optm_call_ret"] = float(cf["ret"])
    out["optm_put_ret"] = float(pf["ret"])
    out["optm_call_range_pct"] = float(cf["range_pct"])
    out["optm_put_range_pct"] = float(pf["range_pct"])
    out["optm_call_vol"] = float(cf["vol"])
    out["optm_put_vol"] = float(pf["vol"])

    # Straddle: combine best-effort.
    out["optm_straddle_ret"] = float(cf["ret"] + pf["ret"])
    out["optm_straddle_range_pct"] = float(cf["range_pct"] + pf["range_pct"])
    out["optm_straddle_vol"] = float(cf["vol"] + pf["vol"])

    return out
