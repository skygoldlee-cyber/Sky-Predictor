"""eBest API helpers (auth + REST requests).

This module is split out from `ebest_live.py` to keep the live loop orchestration thin.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.utils import safe_float

logger = logging.getLogger(__name__)


def _mask_sensitive(msg: str) -> str:
    """예외 메시지에서 API 키 등 민감 정보를 마스킹한다.

    NW-SEC-01: 일부 HTTP 라이브러리는 인증 실패 예외에 요청 URL(키 포함)을 담는다.
    appkey/secret 관련 문자열을 '***'로 치환해 로그에 키가 노출되지 않도록 한다.
    """
    if not msg:
        return msg
    # appkey=값, appsecretkey=값, Authorization: Bearer 값 패턴
    msg = re.sub(r'(appkey|appsecret(?:key)?)[=:]\s*\S+', r'\1=***', msg, flags=re.IGNORECASE)
    msg = re.sub(r'(Bearer\s+)\S+', r'\1***', msg, flags=re.IGNORECASE)
    return msg


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(base, dict):
        base = {}
    if not isinstance(override, dict):
        return dict(base)
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out.get(k) or {}, v)
        else:
            out[k] = v
    return out


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load JSON config as a dict (best-effort).

    This is used only for a small number of runtime toggles (e.g. test-time injection).
    """
    try:
        config_path_s = str(config_path or "config.json")
        with open(config_path_s, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg = cfg if isinstance(cfg, dict) else {}

        secrets_path = os.environ.get("APP_SECRETS_CONFIG")
        if not secrets_path:
            try:
                import os as _os

                secrets_path = _os.path.join(_os.path.dirname(config_path_s) or ".", "config.secrets.json")
            except Exception:
                secrets_path = "config.secrets.json"

        try:
            with open(str(secrets_path), "r", encoding="utf-8") as f:
                secrets_cfg = json.load(f)
            if isinstance(secrets_cfg, dict) and secrets_cfg:
                cfg = _deep_merge_dict(cfg, secrets_cfg)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("[CONFIG] secrets load failed: %s", e)

        return cfg
    except Exception as e:
        logger.warning("[CONFIG] load failed: %s", e)
        return {}


def _get_option_subscribe_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract option subscription settings from config.

    Args:
        cfg: Parsed config dict.

    Returns:
        The `options_subscription` sub-dict when present; otherwise an empty dict.

    Notes:
        This helper is used by the live runtime to determine how many option symbols
        to subscribe to around ATM.
    """
    v = cfg.get("options_subscription")
    return v if isinstance(v, dict) else {}


def _get_ebest_keys(*, config_path: str = "config.json") -> Tuple[str, str]:
    """Resolve eBest appkey/appsecret from env vars or config.json.

    Resolution order:
    1) Environment variables: `EBEST_APPKEY/EBEST_APPSECRET` (or legacy aliases)
    2) `config.json` under `ebest.appkey/ebest.appsecretkey`

    Args:
        config_path: Path to config JSON.

    Returns:
        (appkey, appsecretkey) as strings. Missing values are returned as empty strings.
    """
    appkey = os.environ.get("EBEST_APPKEY") or os.environ.get("EBEST_APP_KEY")
    appsecret = os.environ.get("EBEST_APPSECRET") or os.environ.get("EBEST_APP_SECRET")

    if not appkey or not appsecret:
        try:
            cfg = _load_config(str(config_path or "config.json"))
            eb = (cfg or {}).get("ebest") or {}
            appkey = appkey or str(eb.get("appkey") or "")
            appsecret = appsecret or str(eb.get("appsecretkey") or "")
        except Exception as e:
            logger.warning("[KEYS] config read failed: %s", e)

    return (str(appkey or ""), str(appsecret or ""))


async def _ebest_login(api: Any, *, appkey: str, appsecretkey: str) -> bool:
    """Login wrapper with error handling.

    Args:
        api: eBest wrapper client.
        appkey: eBest app key.
        appsecretkey: eBest secret key.

    Returns:
        True on success, False on failure.
    """
    try:
        ok = await api.login(appkey, appsecretkey)
        return bool(ok)
    except Exception as e:
        logger.warning("[LOGIN] failed: %s", _mask_sensitive(str(e)))
        return False


async def _ebest_fetch_kp200_symbol(api: Any) -> Optional[str]:
    """Fetch the KP200 futures symbol via `t8432` (best-effort).

    Args:
        api: eBest wrapper client.

    Returns:
        Futures symbol code (e.g. "101V3000") when available; otherwise None.
    """
    try:
        res = await api.request("t8432", {"t8432InBlock": {"gubun": "0"}})
        items = (getattr(res, "body", None) or {}).get("t8432OutBlock") or []
        shcode = str(items[0].get("shcode") or "").strip() if items else ""
        return shcode or None
    except Exception as e:
        logger.warning("[KP200_SYMBOL] fetch failed: %s", e)
        return None


async def _ebest_fetch_kp200_symbol_and_prev_close(api: Any) -> Tuple[Optional[str], Optional[float]]:
    """Fetch the KP200 futures symbol and previous close via `t8432` (best-effort).

    Returns:
        (symbol, prev_close) where either may be None on failure.
    """
    try:
        res = await api.request("t8432", {"t8432InBlock": {"gubun": "0"}})
        items = (getattr(res, "body", None) or {}).get("t8432OutBlock") or []
        if not isinstance(items, list) or not items:
            return None, None
        first = items[0] if isinstance(items[0], dict) else {}
        shcode = str(first.get("shcode") or "").strip()
        prev_close = safe_float(first.get("jnilclose"))
        return (shcode or None, float(prev_close) if prev_close > 0.0 else None)
    except Exception as e:
        logger.warning("[KP200_SYMBOL] fetch symbol/prev_close failed: %s", e)
        return None, None


async def _ebest_fetch_t2301_open_map(
    api: Any,
    *,
    yyyymm: str,
    gubun: str = "G",
) -> Optional[Dict[str, Dict[str, float]]]:
    try:
        req = {"t2301InBlock": {"yyyymm": str(yyyymm), "gubun": str(gubun)}}
        res = await api.request("t2301", req)
        body = getattr(res, "body", None) or {}

        calls = body.get("t2301OutBlock1") or []
        puts = body.get("t2301OutBlock2") or []
        if not isinstance(calls, list):
            calls = []
        if not isinstance(puts, list):
            puts = []

        call_open: Dict[str, float] = {}
        put_open: Dict[str, float] = {}

        for it in calls:
            if not isinstance(it, dict):
                continue
            code = str(it.get("optcode") or "").strip()
            if not code:
                continue
            op = safe_float(it.get("open"))
            if (op or 0.0) <= 0.0:
                op = safe_float(it.get("price"))
            if (op or 0.0) > 0.0:
                call_open[code] = float(op)

        for it in puts:
            if not isinstance(it, dict):
                continue
            code = str(it.get("optcode") or "").strip()
            if not code:
                continue
            op = safe_float(it.get("open"))
            if (op or 0.0) <= 0.0:
                op = safe_float(it.get("price"))
            if (op or 0.0) > 0.0:
                put_open[code] = float(op)

        return {"calls": call_open, "puts": put_open}
    except Exception as e:
        logger.warning("[T2301] open map fetch failed yyyymm=%s gubun=%s: %s", yyyymm, gubun, e)
        return None


async def _ebest_fetch_kp200_price_t8415(api: Any, *, symbol: str, yyyymmdd: Optional[str] = None) -> Optional[float]:
    """Fetch the latest close price via `t8415` (best-effort).

    Args:
        api: eBest wrapper client.
        symbol: Futures symbol code.
        yyyymmdd: Optional date override (YYYYMMDD). Defaults to today.

    Returns:
        Latest close price as float when available; otherwise None.
    """
    try:
        date = str(yyyymmdd) if yyyymmdd else datetime.now().strftime("%Y%m%d")
        req = {
            "t8415InBlock": {
                "shcode": str(symbol),
                "ncnt": 1,
                "qrycnt": 1,
                "nday": "0",
                "sdate": "",
                "stime": "",
                "edate": date,
                "etime": "",
                "cts_date": "",
                "cts_time": "",
                "comp_yn": "N",
            }
        }
        res = await api.request("t8415", req)
        body = getattr(res, "body", None) or {}
        items = body.get("t8415OutBlock1") or []
        last = items[-1] if isinstance(items, list) and items else None
        if not isinstance(last, dict):
            return None
        close = safe_float(last.get("close"), 0.0)
        return float(close) if close > 0.0 else None
    except Exception as e:
        logger.warning("[KP200_PRICE] fetch failed: %s", e)
        return None


async def _ebest_fetch_kp200_ohlcv_t8415(api: Any, *, symbol: str, yyyymmdd: Optional[str] = None, ncnt: int = 1) -> Optional[List[Dict[str, Any]]]:
    """Fetch OHLCV data via `t8415` for a specific date.

    Args:
        api: eBest wrapper client.
        symbol: Futures symbol code.
        yyyymmdd: Target date (YYYYMMDD). Defaults to today.
        ncnt: Unit in minutes (0=30sec, 1=1min, 5=5min, etc.)

    Returns:
        List of OHLCV bars when available; otherwise None.
        Each bar contains: time, open, high, low, close, volume
    """
    try:
        date = str(yyyymmdd) if yyyymmdd else datetime.now().strftime("%Y%m%d")
        req = {
            "t8415InBlock": {
                "shcode": str(symbol),
                "ncnt": ncnt,      # 분봉 단위 (숫자)
                "qrycnt": 1,       # 최대 조회 수
                "nday": "",        # 일수 (빈 문자열)
                "sdate": date,     # 시작 날짜
                "stime": "",       # 시작 시간 (빈 문자열)
                "edate": date,     # 종료 날짜
                "etime": "",       # 종료 시간 (빈 문자열)
                "cts_date": "",    # 연속 조회 날짜 (빈 문자열)
                "cts_time": "",    # 연속 조회 시간 (빈 문자열)
                "comp_yn": "N",    # 압축 여부 (미압축)
            }
        }
        res = await api.request("t8415", req)
        body = getattr(res, "body", None) or {}
        items = body.get("t8415OutBlock1") or []

        # 디버깅: API 응답 로깅
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("[t8415] API response: body keys=%s, items count=%d, items=%s",
                     list(body.keys()) if isinstance(body, dict) else "not dict",
                     len(items) if isinstance(items, list) else "not list",
                     str(items)[:500] if items else "empty")

        if not isinstance(items, list):
            return None

        # 데이터 변환
        bars = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                bar = {
                    "date": str(item.get("date") or ""),
                    "time": str(item.get("time") or ""),
                    "open": safe_float(item.get("open"), 0.0),
                    "high": safe_float(item.get("high"), 0.0),
                    "low": safe_float(item.get("low"), 0.0),
                    "close": safe_float(item.get("close"), 0.0),
                    "volume": safe_float(item.get("jdiff_vol"), 0.0),
                }
                bars.append(bar)
            except Exception:
                continue
        return bars if bars else None
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("[t8418] request failed: %s", e)
        return None


async def _ebest_fetch_kospi_ohlcv_t8418(api: Any, *, symbol: str = "001", yyyymmdd: Optional[str] = None, ncnt: int = 1) -> Optional[List[Dict[str, Any]]]:
    """Fetch KOSPI index OHLCV data via `t8418` for a specific date.

    Args:
        api: eBest wrapper client.
        symbol: KOSPI index symbol code (default: "001").
        yyyymmdd: Target date (YYYYMMDD). Defaults to today.
        ncnt: Unit in minutes (0=30sec, 1=1min, 5=5min, etc.)

    Returns:
        List of OHLCV bars when available; otherwise None.
        Each bar contains: time, open, high, low, close, volume
    """
    try:
        date = str(yyyymmdd) if yyyymmdd else datetime.now().strftime("%Y%m%d")
        req = {
            "t8418InBlock": {
                "shcode": str(symbol),
                "ncnt": ncnt,       # 분봉 단위 (숫자)
                "qrycnt": 1,        # 최대 조회 수
                "nday": "",         # 일수 (빈 문자열)
                "sdate": date,      # 시작 날짜
                "stime": "",        # 시작 시간 (빈 문자열)
                "edate": date,      # 종료 날짜
                "etime": "",        # 종료 시간 (빈 문자열)
                "cts_date": "",     # 연속 조회 날짜 (빈 문자열)
                "cts_time": "",     # 연속 조회 시간 (빈 문자열)
                "comp_yn": "N",     # 압축 여부 (미압축)
            }
        }
        res = await api.request("t8418", req)
        body = getattr(res, "body", None) or {}
        items = body.get("t8418OutBlock1") or []

        # 디버깅: API 응답 로깅
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("[t8418] API response: body keys=%s, items count=%d, items=%s",
                     list(body.keys()) if isinstance(body, dict) else "not dict",
                     len(items) if isinstance(items, list) else "not list",
                     str(items)[:500] if items else "empty")

        if not isinstance(items, list):
            return None
        
        # 데이터 변환
        bars = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                bar = {
                    "date": str(item.get("date") or ""),
                    "time": str(item.get("time") or ""),
                    "open": safe_float(item.get("open"), 0.0),
                    "high": safe_float(item.get("high"), 0.0),
                    "low": safe_float(item.get("low"), 0.0),
                    "close": safe_float(item.get("close"), 0.0),
                    "volume": safe_float(item.get("jdiff_vol"), 0.0),
                }
                if bar["close"] > 0:
                    bars.append(bar)
            except Exception:
                continue
        
        return bars if bars else None
        
    except Exception as e:
        logger.warning("[KP200_OHLCV] fetch failed: %s", e)
        return None


async def _ebest_fetch_t2101_snapshot(api: Any, *, focode: str) -> Optional[Dict[str, Any]]:
    """Fetch a best-effort snapshot via `t2101`.

    Ref: t2101_SCHEMA.md

    Args:
        api: eBest wrapper client.
        focode: Futures code.

    Returns:
        A small stable dict snapshot for logging/LLM context, or None.

    Note on `open`:
        t2101 응답의 `open` 필드는 당일 장 시작(08:45) 시가를 의미한다.
        그러나 장 초기 또는 특정 환경에서 0 또는 이론가(theoryprice)가 반환될 수 있다.
        호출자(pipeline.set_market_snapshots)에서 유효성 검증을 추가로 수행하므로
        여기서는 원시 값을 그대로 반환하고 `_open_raw` 키로 원본을 보존한다.
    """
    try:
        req = {"t2101InBlock": {"focode": str(focode)}}
        res = await api.request("t2101", req)
        body = getattr(res, "body", None) or {}
        out = body.get("t2101OutBlock") or {}
        if not isinstance(out, dict):
            return None

        _open_raw  = safe_float(out.get("open"))
        _price_raw = safe_float(out.get("price"))
        _theory    = safe_float(out.get("theoryprice"))

        # open이 0이고 price가 유효하면 price로 보완 (장 초기 open 미확정 방어)
        # open과 theoryprice가 동일하면 이론가가 혼입된 것일 수 있으므로 0으로 마킹
        _open_eff = _open_raw
        if _open_eff == 0.0 and _price_raw > 0.0:
            _open_eff = 0.0  # price != open이므로 보정하지 않음 — 호출자가 fallback 처리
        if _theory > 0.0 and abs(_open_raw - _theory) < 0.01 and _price_raw > 0.0:
            # theoryprice와 open이 동일 → open이 이론가로 오염됐을 가능성
            logger.warning(
                "[T2101] open(%.2f) == theoryprice(%.2f) — open 신뢰도 낮음, 0으로 마킹",
                _open_raw, _theory,
            )
            _open_eff = 0.0

        snap = {
            "focode":      str(out.get("focode") or focode),
            "hname":       str(out.get("hname") or ""),
            "price":       _price_raw,
            "open":        _open_eff,
            "_open_raw":   _open_raw,   # 원본 보존 (디버깅용)
            "high":        safe_float(out.get("high")),
            "low":         safe_float(out.get("low")),
            "volume":      safe_float(out.get("volume")),
            "value":       safe_float(out.get("value")),
            "mgjv":        safe_float(out.get("mgjv")),
            "basis":       safe_float(out.get("basis")),
            "theoryprice": _theory,
            "kospijisu":   safe_float(out.get("kospijisu")),
            "impv":        safe_float(out.get("impv")),
            "delt":        safe_float(out.get("delt")),
            "gama":        safe_float(out.get("gama")),
            "ceta":        safe_float(out.get("ceta")),
            "vega":        safe_float(out.get("vega")),
            "rhox":        safe_float(out.get("rhox")),
        }
        return snap
    except Exception as e:
        logger.warning("[T2101] fetch failed focode=%s: %s", focode, e)
        return None


async def _ebest_fetch_ij_snapshot(api: Any, *, tr_key: str = "101") -> Optional[Dict[str, Any]]:
    """Fetch a best-effort index snapshot via `IJ`.

    Args:
        api: eBest wrapper client.
        tr_key: Index code string (e.g. "001" KOSPI, "301" KOSDAQ, "101" KP200).

    Returns:
        A small stable dict snapshot for logging/LLM context, or None.
    """
    try:
        # Wrapper/environment differences observed:
        # - Some wrappers require only `IJInBlock: {tr_key}`
        # - Some ignore `tr_cd`
        # - Out block key can vary (IJOutBlock, ijOutBlock, OutBlock, etc.)
        req_candidates = [
            {"IJInBlock": {"tr_cd": "IJ", "tr_key": str(tr_key)}},
            {"IJInBlock": {"tr_key": str(tr_key)}},
            {"InBlock": {"tr_key": str(tr_key)}},
            {"ijInBlock": {"tr_key": str(tr_key)}},
        ]

        res = None
        body: Dict[str, Any] = {}
        last_exc: Optional[Exception] = None
        for req in req_candidates:
            try:
                res = await api.request("IJ", req)
                body = getattr(res, "body", None) or {}
                if isinstance(body, dict) and body:
                    break
            except Exception as e:
                last_exc = e
                continue

        if not isinstance(body, dict) or not body:
            if last_exc is not None:
                raise last_exc
            return None

        out = (
            body.get("IJOutBlock")
            or body.get("ijOutBlock")
            or body.get("OutBlock")
            or body.get("outblock")
            or body.get("output")
            or {}
        )
        if not isinstance(out, dict):
            return None

        # Some wrappers nest under `body['rsp']` or similar.
        if not out and isinstance(body.get("rsp"), dict):
            rsp = body.get("rsp") or {}
            out = rsp.get("IJOutBlock") or rsp.get("ijOutBlock") or rsp.get("OutBlock") or {}
            if not isinstance(out, dict):
                out = {}

        def _pick_float(d: Dict[str, Any], keys: list[str]) -> float:
            for k in keys:
                try:
                    v = safe_float(d.get(k))
                    if v is None:
                        continue
                    # allow 0.0 but prefer non-zero candidates first
                    return float(v)
                except Exception:
                    continue
            return 0.0

        # jisu key may vary.
        jisu = _pick_float(out, ["jisu", "index", "close", "price", "curjisu", "nowjisu", "jisu1"])
        # If still zero, try scanning for any numeric-looking field containing 'jisu'.
        if float(jisu or 0.0) == 0.0:
            try:
                for k, v in out.items():
                    if "jisu" in str(k).lower():
                        vv = safe_float(v)
                        if vv and float(vv) != 0.0:
                            jisu = float(vv)
                            break
            except Exception:
                pass

        snap = {
            "tr_key": str(tr_key),
            "time": str(out.get("time") or ""),
            "jisu": float(jisu),
            "sign": str(out.get("sign") or ""),
            "change": safe_float(out.get("change")),
            "drate": safe_float(out.get("drate")),
            "cvolume": safe_float(out.get("cvolume")),
            "volume": safe_float(out.get("volume")),
            "value": safe_float(out.get("value")),
            "upjo": safe_float(out.get("upjo")),
            "highjo": safe_float(out.get("highjo")),
            "unchgjo": safe_float(out.get("unchgjo")),
            "lowjo": safe_float(out.get("lowjo")),
            "downjo": safe_float(out.get("downjo")),
            "upjrate": safe_float(out.get("upjrate")),
            "openjisu": safe_float(out.get("openjisu")),
            "opentime": str(out.get("opentime") or ""),
            "highjisu": safe_float(out.get("highjisu")),
            "hightime": str(out.get("hightime") or ""),
            "lowjisu": safe_float(out.get("lowjisu")),
            "lowtime": str(out.get("lowtime") or ""),
            "frgsvolume": safe_float(out.get("frgsvolume")),
            "orgsvolume": safe_float(out.get("orgsvolume")),
            "frgsvalue": safe_float(out.get("frgsvalue")),
            "orgsvalue": safe_float(out.get("orgsvalue")),
            "upcode": str(out.get("upcode") or ""),
        }
        return snap
    except Exception as e:
        logger.warning("[IJ] fetch failed tr_key=%s: %s", tr_key, e)
        return None


async def _ebest_fetch_t2301_snapshot(
    api: Any,
    *,
    yyyymm: str,
    gubun: str = "G",
    sample_n: int = 5,
) -> Optional[Dict[str, Any]]:
    """Fetch a best-effort option IV/chain snapshot via `t2301`.

    Args:
        api: eBest wrapper client.
        yyyymm: Option month (YYYYMM).
        gubun: Option type (e.g. "G" regular / "W" weekly; depends on API).
        sample_n: Number of chain rows to keep as a small sample (for diagnostics).

    Returns:
        Snapshot dict or None on failure.
        전 행사가 OI(mgjv), Greeks, IV를 포함한 oi_calls / oi_puts 리스트를 포함한다.
        OI 지지저항 분석은 이 필드를 사용하므로 OC0 실시간 구독 범위와 무관하게 동작한다.
    """
    try:
        req = {"t2301InBlock": {"yyyymm": str(yyyymm), "gubun": str(gubun)}}
        res = await api.request("t2301", req)
        body = getattr(res, "body", None) or {}

        out0 = body.get("t2301OutBlock") or {}
        calls = body.get("t2301OutBlock1") or []
        puts = body.get("t2301OutBlock2") or []

        if not isinstance(out0, dict):
            out0 = {}
        if not isinstance(calls, list):
            calls = []
        if not isinstance(puts, list):
            puts = []

        snap: Dict[str, Any] = {
            "yyyymm": str(yyyymm),
            "gubun": str(gubun),
            "histimpv": safe_float(out0.get("histimpv")),
            "jandatecnt": safe_float(out0.get("jandatecnt")),
            "cimpv": safe_float(out0.get("cimpv")),
            "pimpv": safe_float(out0.get("pimpv")),
            "gmprice": safe_float(out0.get("gmprice")),
            "gmchange": safe_float(out0.get("gmchange")),
            "gmdiff": safe_float(out0.get("gmdiff")),
            "gmvolume": safe_float(out0.get("gmvolume")),
            "gmshcode": str(out0.get("gmshcode") or ""),
            "call_count": int(len(calls)),
            "put_count": int(len(puts)),
        }

        def _parse_chain_row(it: Dict[str, Any], opt_type: str) -> Dict[str, Any]:
            """t2301 OutBlock1(콜)/OutBlock2(풋) 행 → 공통 포맷으로 변환."""
            return {
                "strike": safe_float(it.get("actprice")),
                "optcode": str(it.get("optcode") or ""),
                "option_type": opt_type,
                "price": safe_float(it.get("price")),
                "open_interest": safe_float(it.get("mgjv")),   # ← 미결제약정(OI)
                "oi_change": safe_float(it.get("mgjvupdn")),   # 미결제증감
                "volume": safe_float(it.get("volume")),
                "iv": safe_float(it.get("iv")) / 100.0 if it.get("iv") else 0.0,
                "delta": safe_float(it.get("delt")),
                "gamma": safe_float(it.get("gama")),
                "vega": safe_float(it.get("vega")),
                "theta": safe_float(it.get("ceta")),
                "rho": safe_float(it.get("rhox")),
                "theory_price": safe_float(it.get("theoryprice")),
                "bid": safe_float(it.get("bidho1")),
                "ask": safe_float(it.get("offerho1")),
                "bid_qty": safe_float(it.get("bidrem1")),
                "ask_qty": safe_float(it.get("offerrem1")),
                "atmgubun": str(it.get("atmgubun") or ""),
                # ── 가격 레벨 탐색용 당일 OHLC ──────────────────────────────
                # t2301 OutBlock1/2 스키마: open/high/low 필드 (6.2 format)
                "open": safe_float(it.get("open")),
                "high": safe_float(it.get("high")),
                "low":  safe_float(it.get("low")),
            }

        # 전 행사가 OI 포함 리스트 — OI 지지저항 분석 핵심 소스
        oi_calls: List[Dict[str, Any]] = []
        for it in calls:
            if isinstance(it, dict):
                row = _parse_chain_row(it, "call")
                if row["strike"] > 0.0:
                    oi_calls.append(row)

        oi_puts: List[Dict[str, Any]] = []
        for it in puts:
            if isinstance(it, dict):
                row = _parse_chain_row(it, "put")
                if row["strike"] > 0.0:
                    oi_puts.append(row)

        snap["oi_calls"] = oi_calls    # 전 행사가 콜 OI 리스트
        snap["oi_puts"] = oi_puts      # 전 행사가 풋 OI 리스트
        snap["oi_fetched_at"] = float(__import__("time").time())  # 갱신 시각(epoch)

        def _sample_chain(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out_items: List[Dict[str, Any]] = []
            for it in items[: max(0, int(sample_n))]:
                if not isinstance(it, dict):
                    continue
                out_items.append(
                    {
                        "actprice": safe_float(it.get("actprice")),
                        "optcode": str(it.get("optcode") or ""),
                        "price": safe_float(it.get("price")),
                        "iv": safe_float(it.get("iv")),
                        "mgjv": safe_float(it.get("mgjv")),
                        "offerho1": safe_float(it.get("offerho1")),
                        "bidho1": safe_float(it.get("bidho1")),
                        "offerrem1": safe_float(it.get("offerrem1")),
                        "bidrem1": safe_float(it.get("bidrem1")),
                        "volume": safe_float(it.get("volume")),
                        "cvolume": safe_float(it.get("cvolume")),
                        "atmgubun": str(it.get("atmgubun") or ""),
                    }
                )
            return out_items

        snap["calls_sample"] = _sample_chain(calls)
        snap["puts_sample"] = _sample_chain(puts)
        return snap
    except Exception as e:
        logger.warning("[T2301] fetch failed yyyymm=%s gubun=%s: %s", yyyymm, gubun, e)
        return None


async def _ebest_fetch_option_symbols(api: Any, *, option_month_info: str) -> Tuple[List[str], List[str]]:
    """Fetch option symbols (calls, puts) matching a given option month label.

    Args:
        api: eBest wrapper client.
        option_month_info: Month label embedded in hname (implementation-specific).

    Returns:
        (calls, puts) lists. Empty lists on failure.
    """
    try:
        res = await api.request("t8433", {"t8433InBlock": {"dummy": ""}})
        items = (getattr(res, "body", None) or {}).get("t8433OutBlock") or []

        filtered = sorted(
            [it for it in items if str(option_month_info) in str(it.get("hname") or "")],
            key=lambda x: str(x.get("shcode") or ""),
        )
        shcodes = [s for s in (str(it.get("shcode") or "").strip() for it in filtered) if s]

        call_prefix = next((s[:4] for s in shcodes if s.startswith("B")), None)
        put_prefix = next((s[:4] for s in shcodes if s.startswith("C")), None)

        if not call_prefix or not put_prefix:
            logger.warning("[OPTIONS] could not determine call/put prefix from symbols")
            return [], []

        calls = [s for s in shcodes if s.startswith(call_prefix)]
        puts = [s for s in shcodes if s.startswith(put_prefix)]
        return calls, puts
    except Exception as e:
        logger.warning("[OPTIONS] symbol fetch failed: %s", e)
        return [], []


async def _ebest_fetch_front_month_and_all_option_symbols(
    api: Any,
    *,
    option_month_info: str,
) -> Dict[str, Any]:
    """Fetch front-month futures symbol and all option symbols for a given month label.

    This is a convenience wrapper intended for "after login" initialization.

    Returns:
        {
          "kp200_symbol": Optional[str],
          "kp200_prev_close": Optional[float],
          "calls": List[str],
          "puts": List[str],
        }
    """
    kp200_symbol, kp200_prev_close = await _ebest_fetch_kp200_symbol_and_prev_close(api)
    calls, puts = await _ebest_fetch_option_symbols(api, option_month_info=str(option_month_info or "").strip())
    return {
        "kp200_symbol": kp200_symbol,
        "kp200_prev_close": kp200_prev_close,
        "calls": calls,
        "puts": puts,
    }


async def _ebest_register_realtime(api: Any, *, trcode: str, symbol: str) -> bool:
    """Register realtime subscription for a given trcode/symbol (best-effort).

    Args:
        api: eBest wrapper client.
        trcode: TR code to subscribe (e.g. FC0/FH0/OC0/OH0).
        symbol: Target symbol.

    Returns:
        True when request is accepted; False when an exception is raised.
    """
    try:
        ok = await api.add_realtime(trcode, symbol)
        return bool(ok) if ok is not None else True
    except Exception as e:
        logger.warning("[REALTIME] register failed trcode=%s symbol=%s: %s", trcode, symbol, e)
        return False
