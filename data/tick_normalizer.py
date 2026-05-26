from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import TRCode  # NW-QUA-02: TR 코드 매직 문자열 → enum
from core.utils import safe_float, safe_int


def normalize_realtime_tick(*, trcode: str, symbol: str, tick: Any) -> Dict[str, Any]:
    """Normalize eBest realtime payloads into a stable schema.

    This function is intentionally conservative:
    - It never mutates the original `tick`.
    - It returns a new dict with a small set of common keys + TR-specific keys.

    Normalized common keys:
    - trcode, symbol
    - chetime (as provided; no datetime parsing here)

    FC0/OC0 normalized keys:
    - price, open, high, low (float)
    - cvolume (int, trade volume)
    - volume (int, cumulative volume)
    - bid1, ask1 (float)
    - openyak (int)
    - k200jisu (float, when present)

    FH0/OH0 normalized keys:
    - hotime (str)
    - offerhos, bidhos (List[float], len=5)
    - offerrems, bidrems (List[float], len=5)
    - offercnts, bidcnts (List[float], len=5)
    - totofferrem, totbidrem, totoffercnt, totbidcnt (float)
    """

    t = tick if isinstance(tick, dict) else {}
    tc = str(trcode or "").strip().upper()

    out: Dict[str, Any] = {
        "trcode": tc,
        "symbol": str(symbol or "").strip(),
    }

    # Most realtime payloads carry chetime (HHMMSS) or hotime (HHMMSS)
    chetime = t.get("chetime")
    if chetime is not None and str(chetime).strip() != "":
        out["chetime"] = str(chetime)

    if tc in (TRCode.FUTURES.value, TRCode.OPTIONS.value):
        cvol = safe_int(t.get("cvolume"))
        cumvol = safe_int(t.get("volume"))
        if cvol <= 0 and cumvol > 0:
            cvol = int(cumvol)

        out.update(
            {
                "price": safe_float(t.get("price")),
                "open": safe_float(t.get("open")),
                "high": safe_float(t.get("high")),
                "low": safe_float(t.get("low")),
                "cvolume": int(cvol),
                "volume": int(cumvol),
                "value": safe_float(t.get("value")),
                "bid1": safe_float(t.get("bidho1")) or safe_float(t.get("bidho")),
                "ask1": safe_float(t.get("offerho1")) or safe_float(t.get("offerho")),
                "openyak": safe_int(t.get("openyak")),
                "k200jisu": safe_float(t.get("k200jisu")) or safe_float(t.get("kospijisu")),
                "theoryprice": safe_float(t.get("theoryprice")),
                # BASIS 필드 (FC0 OutBlock — eBest 직접 제공)
                "sbasis": safe_float(t.get("sbasis")),    # 시장BASIS (선물 - KP200현물)
                "ibasis": safe_float(t.get("ibasis")),    # 이론BASIS
                "kasis":  safe_float(t.get("kasis")),     # 괴리율
            }
        )

        # OC0-only fields
        if tc == TRCode.OPTIONS.value:
            out.update(
                {
                    "optcode": str(t.get("optcode") or out.get("symbol") or ""),
                    "impv": safe_float(t.get("impv")),
                    "timevalue": safe_float(t.get("timevalue")),
                    "eqva": safe_float(t.get("eqva")),
                }
            )

        # FC0-only fields
        if tc == TRCode.FUTURES.value:
            out["futcode"] = str(t.get("futcode") or out.get("symbol") or "")

        return out

    if tc in (TRCode.FUTURES_BOOK.value, TRCode.OPTIONS_QUOTE.value):
        hotime = t.get("hotime") or t.get("chetime")
        if hotime is not None and str(hotime).strip() != "":
            out["hotime"] = str(hotime)

        def _depth_list(prefix: str, n: int = 5) -> List[float]:
            """_depth_list.

Args:
    prefix:
    n:
"""
            return [safe_float(t.get(f"{prefix}{i}")) for i in range(1, n + 1)]

        out.update(
            {
                "offerhos": _depth_list("offerho"),
                "bidhos": _depth_list("bidho"),
                "offerrems": _depth_list("offerrem"),
                "bidrems": _depth_list("bidrem"),
                "offercnts": _depth_list("offercnt"),
                "bidcnts": _depth_list("bidcnt"),
                "totofferrem": safe_float(t.get("totofferrem")),
                "totbidrem": safe_float(t.get("totbidrem")),
                "totoffercnt": safe_float(t.get("totoffercnt")),
                "totbidcnt": safe_float(t.get("totbidcnt")),
                "danhochk": str(t.get("danhochk") or ""),
                "alloc_gubun": str(t.get("alloc_gubun") or ""),
            }
        )

        if tc == TRCode.FUTURES_BOOK.value:
            out["futcode"] = str(t.get("futcode") or out.get("symbol") or "")
        else:
            out["optcode"] = str(t.get("optcode") or out.get("symbol") or "")

        return out

    # Fallback: pass through minimal
    if "hotime" in t and t.get("hotime") is not None:
        out["hotime"] = str(t.get("hotime"))
    return out
