"""Realtime/message callback helpers for eBest live mode."""

from __future__ import annotations

from collections import deque
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from zoneinfo import ZoneInfo

from data.tick_normalizer import normalize_realtime_tick
from core.utils import make_json_safe, parse_chetime, write_jsonl_line
from config import TRCode  # QUA-07: TR 코드 매직 문자열 → enum

logger = logging.getLogger(__name__)


REALTIME_TRCODES = (TRCode.FUTURES.value, TRCode.OPTIONS.value, TRCode.OPTIONS_QUOTE.value, TRCode.FUTURES_BOOK.value, "JIF", "IJ_")


def _oc0_option_is_call(symbol: str) -> Optional[bool]:
    """OC0 옵션 종목코드 선두: B=콜, C=풋(대소문자 무시). 그 외·빈 문자열은 None."""
    s = str(symbol or "").strip()
    if not s:
        return None
    c = s[0].upper()
    if c == "B":
        return True
    if c == "C":
        return False
    return None


_KST = ZoneInfo("Asia/Seoul")


def _kst_chetime_to_utc_ms(chetime: Any) -> int:
    try:
        che = str(chetime or "").strip()
        if che:
            dt_kst = parse_chetime(che, reference=datetime.now(tz=_KST))
        else:
            dt_kst = datetime.now(tz=_KST).replace(microsecond=0)
    except Exception:
        dt_kst = datetime.now(tz=_KST).replace(microsecond=0)

    try:
        return int(dt_kst.astimezone(timezone.utc).timestamp() * 1000)
    except Exception:
        return int(datetime.now(tz=timezone.utc).replace(microsecond=0).timestamp() * 1000)


def _tick_to_compact_numeric(tick: Any) -> Any:
    if not isinstance(tick, dict):
        return make_json_safe(tick)

    out: dict[str, Any] = {}
    for k, v in tick.items():
        ks = str(k)
        if isinstance(v, str):
            s = v.strip()

            # price-like fields (2 decimals) -> int scaled by 100
            if ks.startswith("offerho") or ks.startswith("bidho"):
                try:
                    out[ks] = int(round(float(s) * 100))
                    continue
                except Exception:
                    pass

            # count-like fields -> int
            if ks.startswith("tot") or ks.endswith("cnt") or ks.endswith("rem"):
                try:
                    out[ks] = int(s) if s else 0
                    continue
                except Exception:
                    pass

            # misc numeric integers stored as strings
            if s.isdigit():
                try:
                    out[ks] = int(s)
                    continue
                except Exception:
                    pass

        out[ks] = make_json_safe(v)

    return out

_GUI_TICK_LOCK = threading.Lock()
_GUI_TICK_STATS: Dict[str, Any] = {
    "counts": {},
    "last_ts": None,
    "last_trcode": "",
    "last_symbol": "",
    "last_chetime": "",
    "last_fc0_ts": None,
    "last_fc0_symbol": None,
    "last_fc0_price": None,
    "last_oc0_ts": None,
    "last_oc0_symbol": None,
    "last_oc0_price": None,
    "call_now": None,
    "put_now": None,
    "last_oh0_call_ts": None,
    "last_oh0_call_symbol": None,
    "last_oh0_put_ts": None,
    "last_oh0_put_symbol": None,
    "oh0_call_count": 0,
    "oh0_put_count": 0,
    "eval_dir_hits": 0,
    "eval_dir_total": 0,
    "eval_dir_rate": 0.0,
    "eval_hold_count": 0,
    "opt_sr_h": None,
    "opt_sr_l": None,
    # 종목별 당일 OC0 체결 고저(센트). 전역 최대/최소가 아님 — 의미가 알림은 이 기준.
    "opt_day_high_by_symbol": {},
    "opt_day_low_by_symbol": {},
    "opt_day_high_cents": None,
    "opt_day_high_symbol": None,
    "opt_day_low_cents": None,
    "opt_day_low_symbol": None,
    "last_jif_log_ms": None,
    "last_jif_jstatus": None,
    "spot_index": None,
    "spot_time": None,
    "fut_prices": deque(maxlen=2000),
}


_DEFAULT_MEANINGFUL_OPT_LEVEL_CENTS = {
    120,  # 1.20
    250,  # 2.50
    350,  # 3.50
    485,  # 4.85
    550,  # 5.50
}

# Meaningful option price levels ("의미가") configured as exact cent values.
# Can be updated at runtime from config.json via `update_meaningful_option_levels`.
_MEANINGFUL_OPT_LEVEL_CENTS = set(_DEFAULT_MEANINGFUL_OPT_LEVEL_CENTS)

# Telegram notifier for meaningful option alerts (set by runtime)
_TELEGRAM_NOTIFIER: Optional[Any] = None

# 의미가 유지 상태 알림(신규 발생 외) 스팸 방지용
_MEANINGFUL_OPT_KEEP_LAST_EPOCH: Dict[str, float] = {}
_MEANINGFUL_OPT_KEEP_COOLDOWN_SEC: float = 180.0

# 의미가 처음 발생 시간 저장 (유지 알림 시 원래 발생 시간 표시용)
_MEANINGFUL_OPT_FIRST_TIMESTAMP: Dict[str, str] = {}


def update_meaningful_option_levels(levels: Any) -> None:
    """Update meaningful option levels from config.

    Accepts an iterable of price levels (floats/strings). Stored as integer cents.
    """
    global _MEANINGFUL_OPT_LEVEL_CENTS
    try:
        if not isinstance(levels, (list, tuple, set)):
            return
        new_set = set()
        for x in levels:
            try:
                new_set.add(int(round(float(x) * 100.0)))
            except Exception:
                continue
        if new_set:
            old_set = _MEANINGFUL_OPT_LEVEL_CENTS
            _MEANINGFUL_OPT_LEVEL_CENTS = set(new_set)
            # 로그 추가: 설정 변경 사항 기록
            levels_str = ", ".join(sorted([f"{x/100.0:.2f}" for x in new_set]))
            logger.info(f"[MEANINGFUL_OPT] 레벨 업데이트: {levels_str} (총 {len(new_set)}개)")
            if old_set != new_set:
                added = new_set - old_set
                removed = old_set - new_set
                if added:
                    added_str = ", ".join(sorted([f"{x/100.0:.2f}" for x in added]))
                    logger.info(f"[MEANINGFUL_OPT] 추가: {added_str}")
                if removed:
                    removed_str = ", ".join(sorted([f"{x/100.0:.2f}" for x in removed]))
                    logger.info(f"[MEANINGFUL_OPT] 제거: {removed_str}")
    except Exception as e:
        logger.warning(f"[MEANINGFUL_OPT] 레벨 업데이트 실패: {e}")
        return


def set_meaningful_option_telegram_notifier(notifier: Any) -> None:
    """Set telegram notifier for meaningful option alerts.
    
    Args:
        notifier: TelegramNotifier instance or None to disable.
    """
    global _TELEGRAM_NOTIFIER
    _TELEGRAM_NOTIFIER = notifier
    if notifier is not None:
        logger.info("[MEANINGFUL_OPT] 텔레그램 알림 활성화")
    else:
        logger.info("[MEANINGFUL_OPT] 텔레그램 알림 비활성화")


def _send_meaningful_option_telegram(
    signal_type: str,
    message: str,
    timestamp: str,
    symbol: str = "",
) -> None:
    """Send meaningful option alert to Telegram.

    Args:
        signal_type: "SRH" / "SRL" (도달) 또는 "SRH_REL" / "SRL_REL" (의미가에서 이탈)
        message: Formatted message (e.g., "B0164787 H2.50")
        timestamp: ISO timestamp string
        symbol: OC0 종목코드(B=콜, C=풋). 종류 표시는 이 코드 선두만 사용(SRH/SRL로 추정하지 않음).
    """
    global _TELEGRAM_NOTIFIER
    if _TELEGRAM_NOTIFIER is None:
        return

    try:
        # 이모지 및 포맷
        if signal_type == "SRH":
            emoji = "🔴"
        elif signal_type == "SRL":
            emoji = "🔵"
        elif signal_type in ("SRH_REL", "SRL_REL"):
            emoji = "🟡"
        else:
            emoji = "⚪"
        # timestamp 길이에 의존한 슬라이싱은 "2026-04-" 같은 잘림을 유발할 수 있어
        # ISO 파싱 우선 + 안전 폴백으로 표시 문자열을 만든다.
        raw_ts = str(timestamp or "").strip()
        time_str = raw_ts
        try:
            if raw_ts:
                iso_s = raw_ts[:-1] + "+00:00" if raw_ts.endswith("Z") else raw_ts
                dt = datetime.fromisoformat(iso_s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_KST)
                time_str = dt.astimezone(_KST).strftime("%H:%M:%S")
        except Exception:
            if "T" in raw_ts and len(raw_ts) >= 19:
                time_str = raw_ts[:19].replace("T", " ")
            elif len(raw_ts) >= 19:
                time_str = raw_ts[:19]
            else:
                time_str = raw_ts

        sym_s = str(symbol or "").strip()
        cp = _oc0_option_is_call(sym_s)
        if cp is True:
            kind = "콜"
        elif cp is False:
            kind = "풋"
        else:
            kind = "미확인"

        label = {
            "SRH": "SRH",
            "SRL": "SRL",
            "SRH_REL": "SRH(의미가 해제)",
            "SRL_REL": "SRL(의미가 해제)",
            "SRH_KEEP": "SRH(의미가 유지)",
            "SRL_KEEP": "SRL(의미가 유지)",
        }.get(signal_type, signal_type)
        title = (
            "옵션 의미가 해제"
            if signal_type in ("SRH_REL", "SRL_REL")
            else ("옵션 의미가 유지" if signal_type in ("SRH_KEEP", "SRL_KEEP") else "옵션 의미가 신호")
        )

        # 메시지 구성 — 종목·종류는 동일 심볼에서만 판별(문구/실제 불일치 방지)
        telegram_msg = f"{emoji} <b>{title}</b>\n\n"
        if sym_s:
            telegram_msg += f"<b>종목:</b> <code>{sym_s}</code>\n"
        telegram_msg += f"<b>{label}:</b> {message}\n"
        telegram_msg += f"<b>시간:</b> {time_str}\n"
        telegram_msg += f"<b>종류:</b> {kind} 옵션"
        
        # 텔레그램 전송
        success = _TELEGRAM_NOTIFIER.send_text(
            telegram_msg,
            parse_mode="HTML",
            debug_context={"kind": "meaningful_option", "signal_type": signal_type}
        )
        
        if success:
            logger.info(f"[MEANINGFUL_OPT] 텔레그램 전송 성공: {signal_type} {message}")
        else:
            logger.warning(f"[MEANINGFUL_OPT] 텔레그램 전송 실패: {signal_type} {message}")
            
    except Exception as e:
        logger.error(f"[MEANINGFUL_OPT] 텔레그램 전송 오류: {e}")


def _price_to_cents(v: Any) -> Any:
    try:
        if v is None:
            return None
        return int(round(float(v) * 100.0))
    except Exception:
        return None


def _send_meaningful_option_keepalive(
    *,
    symbol: str,
    side: str,
    level_cents: int,
    timestamp: str,
) -> None:
    """신규 신호 외 의미가 유지 상태를 주기적으로 송출한다."""
    global _MEANINGFUL_OPT_KEEP_LAST_EPOCH, _MEANINGFUL_OPT_FIRST_TIMESTAMP
    try:
        sym = str(symbol or "").strip()
        side_u = str(side or "").strip().upper()
        lv = int(level_cents or 0)
        if (not sym) or side_u not in ("H", "L") or lv <= 0:
            return

        now_epoch = float(datetime.now(tz=timezone.utc).timestamp())
        key = f"{sym}:{side_u}:{lv}"
        last_epoch = float(_MEANINGFUL_OPT_KEEP_LAST_EPOCH.get(key, 0.0) or 0.0)
        if (now_epoch - last_epoch) < float(_MEANINGFUL_OPT_KEEP_COOLDOWN_SEC):
            return
        _MEANINGFUL_OPT_KEEP_LAST_EPOCH[key] = now_epoch

        # 의미가 처음 발생 시간 사용 (없으면 현재 시간 사용)
        first_timestamp = _MEANINGFUL_OPT_FIRST_TIMESTAMP.get(key, timestamp)

        if side_u == "H":
            sig = "SRH_KEEP"
            msg = f"{sym} H{lv / 100.0:.2f} (고가 의미가 유지)"
        else:
            sig = "SRL_KEEP"
            msg = f"{sym} L{lv / 100.0:.2f} (저가 의미가 유지)"

        _send_meaningful_option_telegram(sig, msg, first_timestamp, sym)
    except Exception as e:
        logger.debug("[MEANINGFUL_OPT] 유지 알림 처리 실패: %s", e)


def update_gui_eval_dir_stats(*, hits: int, total: int, rate: float, hold: int = 0) -> None:
    """Update GUI-visible evaluation stats (best-effort, thread-safe)."""
    try:
        with _GUI_TICK_LOCK:
            _GUI_TICK_STATS["eval_dir_hits"] = int(hits or 0)
            _GUI_TICK_STATS["eval_dir_total"] = int(total or 0)
            _GUI_TICK_STATS["eval_dir_rate"] = float(rate or 0.0)
            _GUI_TICK_STATS["eval_hold_count"] = int(hold or 0)
    except Exception:
        return


def update_gui_spot_index(*, spot_index: Any, spot_time: Any = None) -> None:
    """Update GUI-visible spot index snapshot (best-effort, thread-safe)."""
    try:
        with _GUI_TICK_LOCK:
            try:
                _GUI_TICK_STATS["spot_index"] = float(spot_index) if spot_index is not None else None
            except Exception:
                _GUI_TICK_STATS["spot_index"] = None
            _GUI_TICK_STATS["spot_time"] = spot_time
    except Exception:
        return


def get_gui_tick_stats() -> Dict[str, Any]:
    try:
        with _GUI_TICK_LOCK:
            fut_prices = _GUI_TICK_STATS.get("fut_prices")
            if not isinstance(fut_prices, deque):
                fut_prices = deque(maxlen=2000)
                _GUI_TICK_STATS["fut_prices"] = fut_prices

            fut_now: float = 0.0
            fut_5m: float = 0.0
            try:
                if len(fut_prices) > 0:
                    fut_now = float(fut_prices[-1][1])
                    now_ts_ms = int(fut_prices[-1][0])
                    target_ts_ms = now_ts_ms - (5 * 60 * 1000)

                    # Find last price at or before target_ts_ms.
                    for ts_ms, px in reversed(fut_prices):
                        if int(ts_ms) <= target_ts_ms:
                            fut_5m = float(px)
                            break
            except Exception:
                fut_now = 0.0
                fut_5m = 0.0

            def _as_int(v: Any) -> int:
                try:
                    if v is None:
                        return 0
                    if isinstance(v, bool):
                        return int(v)
                    if isinstance(v, (int, float)):
                        return int(v)
                    if isinstance(v, str) and v.strip() != "":
                        return int(float(v))
                except Exception:
                    return 0
                return 0

            def _as_float(v: Any) -> float:
                try:
                    if v is None:
                        return 0.0
                    if isinstance(v, bool):
                        return float(int(v))
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str) and v.strip() != "":
                        return float(v)
                except Exception:
                    return 0.0
                return 0.0

            counts_obj = _GUI_TICK_STATS.get("counts")
            counts: Dict[str, Any] = {}
            if isinstance(counts_obj, dict):
                counts = dict(counts_obj)
            return {
                "counts": counts,
                "last_ts": _GUI_TICK_STATS.get("last_ts"),
                "last_trcode": _GUI_TICK_STATS.get("last_trcode"),
                "last_symbol": _GUI_TICK_STATS.get("last_symbol"),
                "last_chetime": _GUI_TICK_STATS.get("last_chetime"),
                "last_fc0_ts": _GUI_TICK_STATS.get("last_fc0_ts"),
                "last_fc0_symbol": _GUI_TICK_STATS.get("last_fc0_symbol"),
                "last_fc0_price": _GUI_TICK_STATS.get("last_fc0_price"),
                "last_oc0_ts": _GUI_TICK_STATS.get("last_oc0_ts"),
                "last_oc0_symbol": _GUI_TICK_STATS.get("last_oc0_symbol"),
                "last_oc0_price": _GUI_TICK_STATS.get("last_oc0_price"),
                "call_now": _as_float(_GUI_TICK_STATS.get("call_now")),
                "put_now": _as_float(_GUI_TICK_STATS.get("put_now")),
                "oc0_call_count": _as_int(_GUI_TICK_STATS.get("oc0_call_count")),
                "oc0_put_count": _as_int(_GUI_TICK_STATS.get("oc0_put_count")),
                "last_oh0_call_ts": _GUI_TICK_STATS.get("last_oh0_call_ts"),
                "last_oh0_call_symbol": _GUI_TICK_STATS.get("last_oh0_call_symbol"),
                "last_oh0_put_ts": _GUI_TICK_STATS.get("last_oh0_put_ts"),
                "last_oh0_put_symbol": _GUI_TICK_STATS.get("last_oh0_put_symbol"),
                "oh0_call_count": _as_int(_GUI_TICK_STATS.get("oh0_call_count")),
                "oh0_put_count": _as_int(_GUI_TICK_STATS.get("oh0_put_count")),
                "eval_dir_hits": _as_int(_GUI_TICK_STATS.get("eval_dir_hits")),
                "eval_dir_total": _as_int(_GUI_TICK_STATS.get("eval_dir_total")),
                "eval_dir_rate": _as_float(_GUI_TICK_STATS.get("eval_dir_rate")),
                "eval_hold_count": _as_int(_GUI_TICK_STATS.get("eval_hold_count")),
                "opt_sr_h": _GUI_TICK_STATS.get("opt_sr_h"),
                "opt_sr_l": _GUI_TICK_STATS.get("opt_sr_l"),
                "opt_sr_h_ts": _GUI_TICK_STATS.get("opt_sr_h_ts"),
                "opt_sr_l_ts": _GUI_TICK_STATS.get("opt_sr_l_ts"),
                "spot_index": _GUI_TICK_STATS.get("spot_index"),
                "spot_time": _GUI_TICK_STATS.get("spot_time"),
                "fut_now": fut_now,
                "fut_5m": fut_5m,
            }
    except Exception:
        return {"counts": {}}


def _is_ack_message(msg: Any) -> bool:
    """Return True when `msg` looks like a realtime subscription ACK.

    This is best-effort and supports multiple message shapes:
    - plain strings
    - dict-like objects with `errcode` / `rsp_cd` / `success` fields
    - objects exposing `.body` or `.__dict__`

    Strategy:
    1) Prefer structured codes/booleans when present.
    2) If the message is JSON, attempt to parse and re-evaluate.
    3) Fallback to simple Korean keyword matching.
    """
    try:
        if msg is None:
            return False

        if isinstance(msg, dict):
            d = msg
        else:
            d = None
            try:
                body = getattr(msg, "body", None)
                if isinstance(body, dict):
                    d = body
            except Exception:
                d = None

            if d is None:
                try:
                    d = getattr(msg, "__dict__", None)
                    if not isinstance(d, dict):
                        d = None
                except Exception:
                    d = None

        if isinstance(d, dict):
            for k in ("errcode", "err_code", "error_code", "rsp_cd", "code"):
                if k in d:
                    v = d.get(k)
                    if str(v).strip() in ("0", "00", "00000", "OK", "ok"):
                        return True

            for k in ("success", "ok", "is_ok"):
                if k in d and bool(d.get(k)):
                    return True

            for k in ("message", "msg", "text", "description"):
                v = d.get(k)
                if isinstance(v, str) and v:
                    s = v
                    if ("정상" in s and "처리" in s) or ("등록" in s and any(x in s for x in ("완료", "되", "성공"))):
                        return True

        s = str(msg)
        if s:
            try:
                j = json.loads(s)
                if isinstance(j, dict) and _is_ack_message(j):
                    return True
            except Exception:
                pass

            if ("정상" in s and "처리" in s) or ("등록" in s and any(x in s for x in ("완료", "되", "성공"))):
                return True
    except Exception:
        return False

    return False


def _make_realtime_callback(predictor: Any, state: Any, ticks_fh: Any) -> Any:
    """Create eBest realtime callback.

    The callback:
    - parses (trcode, symbol, tick) from wrapper arguments
    - writes ticks to JSONL when `ticks_fh` is provided
    - attaches `tick_norm` when possible
    - forwards payload into `predictor.add_realtime_tick()`

    Args:
        predictor: PredictionPipeline-like object.
        state: LiveState-like object (mutated for counters).
        ticks_fh: JSONL file handle or None.
    """

    def _on_realtime(*args):
        """Handle a realtime tick event from the eBest wrapper.

        Expected arg patterns (best-effort):
        - (sender, trcode, symbol, tick)
        - (trcode, symbol, tick)
        """
        trcode = symbol = tick = None

        def _normalize_trcode(v: Any) -> str:
            try:
                s = str(v or "").strip().upper()
            except Exception:
                return ""
            if not s:
                return ""
            if s in REALTIME_TRCODES:
                return s
            # Some wrappers include suffixes/prefixes; keep the first 3 chars when it matches.
            try:
                s3 = s[:3]
                if s3 in REALTIME_TRCODES:
                    return s3
            except Exception:
                pass
            return s

        try:
            # Common shapes:
            # - (sender, trcode, symbol, tick)
            # - (trcode, symbol, tick)
            if len(args) >= 4:
                trcode, symbol, tick = args[1], args[2], args[3]
            elif len(args) == 3:
                trcode, symbol, tick = args[0], args[1], args[2]
            else:
                # Fallback: search for a known trcode and a dict-like tick in args.
                cand_tr = None
                cand_tick = None
                cand_sym = None
                for a in args:
                    if cand_tick is None and isinstance(a, dict):
                        cand_tick = a
                        continue
                    s = _normalize_trcode(a)
                    if cand_tr is None and s in REALTIME_TRCODES:
                        cand_tr = s
                        continue
                # Try to use any string arg as symbol.
                for a in args:
                    try:
                        if isinstance(a, str) and a and _normalize_trcode(a) not in REALTIME_TRCODES:
                            cand_sym = a
                            break
                    except Exception:
                        continue
                trcode = cand_tr
                symbol = cand_sym
                tick = cand_tick
        except Exception as e:
            logger.warning("[RT] arg parse error: %s", e)
            return

        trcode_s = _normalize_trcode(trcode)
        symbol_s = str(symbol or "").strip()

        if trcode_s not in REALTIME_TRCODES:
            return

        # Some wrappers deliver tick as a JSON string; normalize to dict.
        if tick is not None and not isinstance(tick, dict):
            try:
                if isinstance(tick, str) and tick.strip():
                    j = json.loads(tick)
                    if isinstance(j, dict):
                        tick = j
            except Exception:
                pass

        try:
            lock = getattr(state, "_lock", None)
            if lock is not None:
                with lock:
                    state.tick_counts[trcode_s] = state.tick_counts.get(trcode_s, 0) + 1
            else:
                state.tick_counts[trcode_s] = state.tick_counts.get(trcode_s, 0) + 1
        except Exception:
            pass

        # If we are receiving any realtime ticks, consider the market gate open.
        # Starting the runtime after 08:45 may not deliver the JIF "open" transition.
        try:
            if trcode_s != "JIF":
                opened = bool(state.market_opened)
                already_logged = bool(state._market_opened_by_tick_logged)
                if not opened:
                    state.market_opened = True
                    if not already_logged:
                        state._market_opened_by_tick_logged = True
                        logger.info(
                            "[GATE_BY_TICK] market_opened=True by realtime tick trcode=%s symbol=%s",
                            trcode_s,
                            symbol_s,
                        )
        except Exception:
            pass

        # IJ_ 콜백: 항상 처리 (KOSPI 데이터용)
        if trcode_s == "IJ_":
            try:
                ij_snap = None
                try:
                    ij_snap = {
                        "tr_key": str(symbol_s or ""),
                        "time": str((tick or {}).get("time") or ""),
                        "jisu": float((tick or {}).get("jisu") or 0.0),
                        "sign": str((tick or {}).get("sign") or ""),
                        "change": float((tick or {}).get("change") or 0.0),
                        "drate": float((tick or {}).get("drate") or 0.0),
                    }
                except Exception:
                    ij_snap = None

                try:
                    update_gui_spot_index(spot_index=(tick or {}).get("jisu"), spot_time=(tick or {}).get("time"))
                except Exception:
                    pass

                try:
                    setter = getattr(predictor, "set_market_snapshots", None)
                    if callable(setter) and isinstance(ij_snap, dict):
                        setter(ij_=ij_snap)
                except Exception:
                    pass

                try:
                    tp = getattr(predictor, "tick_processor", None)
                    fn_ij = getattr(tp, "process_spot_index_tick", None) if tp is not None else None
                    if callable(fn_ij):
                        fn_ij({"trcode": "IJ_", "symbol": symbol_s, "tick": tick})
                except Exception:
                    pass
            except Exception:
                pass

        try:
            ts_iso = None
            try:
                che = (tick or {}).get("hotime") or (tick or {}).get("chetime")
                ts_iso = parse_chetime(che).isoformat()
            except Exception:
                ts_iso = datetime.now().isoformat()
            with _GUI_TICK_LOCK:
                c = _GUI_TICK_STATS.get("counts")
                if not isinstance(c, dict):
                    c = {TRCode.FUTURES.value: 0, TRCode.OPTIONS.value: 0, TRCode.OPTIONS_QUOTE.value: 0, TRCode.FUTURES_BOOK.value: 0, "JIF": 0, "IJ_": 0}
                    _GUI_TICK_STATS["counts"] = c
                c[trcode_s] = int(c.get(trcode_s, 0) or 0) + 1
                _GUI_TICK_STATS["last_ts"] = ts_iso
                _GUI_TICK_STATS["last_trcode"] = trcode_s
                _GUI_TICK_STATS["last_symbol"] = symbol_s
                _GUI_TICK_STATS["last_chetime"] = (tick or {}).get("hotime") or (tick or {}).get("chetime")

                # IJ_ GUI stats: 항상 처리 (KOSPI 데이터용)
                if trcode_s == "IJ_":
                    try:
                        _GUI_TICK_STATS["spot_index"] = float((tick or {}).get("jisu") or 0.0) or None
                    except Exception:
                        _GUI_TICK_STATS["spot_index"] = None
                    try:
                        _GUI_TICK_STATS["spot_time"] = (tick or {}).get("time")
                    except Exception:
                        _GUI_TICK_STATS["spot_time"] = None

                if trcode_s == TRCode.FUTURES.value:
                    _GUI_TICK_STATS["last_fc0_ts"] = ts_iso
                    _GUI_TICK_STATS["last_fc0_symbol"] = symbol_s
                    try:
                        _GUI_TICK_STATS["last_fc0_price"] = float((tick or {}).get("price"))
                    except Exception:
                        _GUI_TICK_STATS["last_fc0_price"] = None

                if trcode_s == TRCode.OPTIONS.value:
                    _GUI_TICK_STATS["last_oc0_ts"] = ts_iso
                    _GUI_TICK_STATS["last_oc0_symbol"] = symbol_s
                    try:
                        oc0_px = float((tick or {}).get("price"))
                        _GUI_TICK_STATS["last_oc0_price"] = float(oc0_px)
                    except Exception:
                        _GUI_TICK_STATS["last_oc0_price"] = None

                    try:
                        is_call = _oc0_option_is_call(str(symbol_s)) is True
                    except Exception:
                        is_call = False

                    if is_call:
                        try:
                            _GUI_TICK_STATS["oc0_call_count"] = int(_GUI_TICK_STATS.get("oc0_call_count") or 0) + 1
                        except Exception:
                            _GUI_TICK_STATS["oc0_call_count"] = 1
                    else:
                        try:
                            _GUI_TICK_STATS["oc0_put_count"] = int(_GUI_TICK_STATS.get("oc0_put_count") or 0) + 1
                        except Exception:
                            _GUI_TICK_STATS["oc0_put_count"] = 1

                    try:
                        if _GUI_TICK_STATS.get("last_oc0_price") is not None:
                            if is_call:
                                _GUI_TICK_STATS["call_now"] = _GUI_TICK_STATS.get("last_oc0_price")
                            else:
                                _GUI_TICK_STATS["put_now"] = _GUI_TICK_STATS.get("last_oc0_price")
                    except Exception:
                        pass

                    # 종목별 당일 OC0 체결 고저(센트). 의미가 알림은 해당 종목의 당일 고/저 갱신 시에만.
                    px_cents = _price_to_cents((tick or {}).get("price"))
                    if px_cents is not None:
                        sym_key = str(symbol_s)
                        byh = _GUI_TICK_STATS.get("opt_day_high_by_symbol")
                        byl = _GUI_TICK_STATS.get("opt_day_low_by_symbol")
                        if not isinstance(byh, dict):
                            byh = {}
                            _GUI_TICK_STATS["opt_day_high_by_symbol"] = byh
                        if not isinstance(byl, dict):
                            byl = {}
                            _GUI_TICK_STATS["opt_day_low_by_symbol"] = byl

                        prev_hi = byh.get(sym_key)
                        prev_lo = byl.get(sym_key)

                        is_new_hi = prev_hi is None or int(px_cents) > int(prev_hi)
                        is_new_lo = prev_lo is None or int(px_cents) < int(prev_lo)
                        # 해당 종목 첫 OC0 틱이면 동일 가격이 당일 고저를 동시에 갱신 → SRL 이중 알림 방지
                        first_tick_sym = prev_hi is None and prev_lo is None

                        if is_new_hi:
                            hi_c = int(px_cents)
                            byh[sym_key] = hi_c
                            _GUI_TICK_STATS["opt_day_high_cents"] = hi_c
                            _GUI_TICK_STATS["opt_day_high_symbol"] = symbol_s
                            if hi_c in _MEANINGFUL_OPT_LEVEL_CENTS:
                                sr_h_msg = f"{symbol_s} H{hi_c / 100.0:.2f}"
                                _GUI_TICK_STATS["opt_sr_h"] = sr_h_msg
                                _GUI_TICK_STATS["opt_sr_h_ts"] = ts_iso
                                logger.info(
                                    f"[MEANINGFUL_OPT] SRH 감지: {sr_h_msg} (종목 당일 신고가)"
                                )
                                # 의미가 처음 발생 시간 저장
                                _MEANINGFUL_OPT_FIRST_TIMESTAMP[f"{sym_key}:H:{hi_c}"] = ts_iso
                                _send_meaningful_option_telegram("SRH", sr_h_msg, ts_iso, sym_key)
                            elif (
                                prev_hi is not None
                                and int(prev_hi) in _MEANINGFUL_OPT_LEVEL_CENTS
                                and hi_c not in _MEANINGFUL_OPT_LEVEL_CENTS
                            ):
                                prev_s = int(prev_hi) / 100.0
                                sr_h_rel = (
                                    f"{symbol_s} H{hi_c / 100.0:.2f} "
                                    f"(고가 의미가 해제, 이전 {prev_s:.2f})"
                                )
                                _GUI_TICK_STATS["opt_sr_h"] = sr_h_rel
                                logger.info(
                                    f"[MEANINGFUL_OPT] SRH(해제): {sr_h_rel} (신고가가 의미가 밖으로)"
                                )
                                # 의미가 해제 시 저장된 시간 삭제
                                first_ts_key = f"{sym_key}:H:{int(prev_hi)}"
                                _MEANINGFUL_OPT_FIRST_TIMESTAMP.pop(first_ts_key, None)
                                _send_meaningful_option_telegram(
                                    "SRH_REL", sr_h_rel, ts_iso, sym_key
                                )
                            else:
                                _GUI_TICK_STATS["opt_sr_h"] = None
                                _GUI_TICK_STATS["opt_sr_h_ts"] = None
                        elif prev_hi is not None and int(prev_hi) in _MEANINGFUL_OPT_LEVEL_CENTS:
                            _send_meaningful_option_keepalive(
                                symbol=sym_key,
                                side="H",
                                level_cents=int(prev_hi),
                                timestamp=ts_iso,
                            )

                        if is_new_lo:
                            lo_c = int(px_cents)
                            byl[sym_key] = lo_c
                            _GUI_TICK_STATS["opt_day_low_cents"] = lo_c
                            _GUI_TICK_STATS["opt_day_low_symbol"] = symbol_s
                            if lo_c in _MEANINGFUL_OPT_LEVEL_CENTS:
                                if first_tick_sym and is_new_hi:
                                    pass
                                else:
                                    sr_l_msg = f"{symbol_s} L{lo_c / 100.0:.2f}"
                                    _GUI_TICK_STATS["opt_sr_l"] = sr_l_msg
                                    _GUI_TICK_STATS["opt_sr_l_ts"] = ts_iso
                                    logger.info(
                                        f"[MEANINGFUL_OPT] SRL 감지: {sr_l_msg} (종목 당일 신저가)"
                                    )
                                    # 의미가 처음 발생 시간 저장
                                    _MEANINGFUL_OPT_FIRST_TIMESTAMP[f"{sym_key}:L:{lo_c}"] = ts_iso
                                    _send_meaningful_option_telegram("SRL", sr_l_msg, ts_iso, sym_key)
                            elif (
                                prev_lo is not None
                                and int(prev_lo) in _MEANINGFUL_OPT_LEVEL_CENTS
                                and lo_c not in _MEANINGFUL_OPT_LEVEL_CENTS
                            ):
                                prev_s = int(prev_lo) / 100.0
                                sr_l_rel = (
                                    f"{symbol_s} L{lo_c / 100.0:.2f} "
                                    f"(저가 의미가 해제, 이전 {prev_s:.2f})"
                                )
                                _GUI_TICK_STATS["opt_sr_l"] = sr_l_rel
                                logger.info(
                                    f"[MEANINGFUL_OPT] SRL(해제): {sr_l_rel} (신저가가 의미가 밖으로)"
                                )
                                # 의미가 해제 시 저장된 시간 삭제
                                first_ts_key = f"{sym_key}:L:{int(prev_lo)}"
                                _MEANINGFUL_OPT_FIRST_TIMESTAMP.pop(first_ts_key, None)
                                _send_meaningful_option_telegram(
                                    "SRL_REL", sr_l_rel, ts_iso, sym_key
                                )
                            else:
                                _GUI_TICK_STATS["opt_sr_l"] = None
                                _GUI_TICK_STATS["opt_sr_l_ts"] = None
                        elif prev_lo is not None and int(prev_lo) in _MEANINGFUL_OPT_LEVEL_CENTS:
                            _send_meaningful_option_keepalive(
                                symbol=sym_key,
                                side="L",
                                level_cents=int(prev_lo),
                                timestamp=ts_iso,
                            )

                if trcode_s == TRCode.OPTIONS_QUOTE.value:
                    try:
                        is_call = _oc0_option_is_call(str(symbol_s)) is True
                    except Exception:
                        is_call = False
                    if is_call:
                        _GUI_TICK_STATS["last_oh0_call_ts"] = ts_iso
                        _GUI_TICK_STATS["last_oh0_call_symbol"] = symbol_s
                        try:
                            _GUI_TICK_STATS["oh0_call_count"] = int(_GUI_TICK_STATS.get("oh0_call_count") or 0) + 1
                        except Exception:
                            _GUI_TICK_STATS["oh0_call_count"] = 1
                    else:
                        _GUI_TICK_STATS["last_oh0_put_ts"] = ts_iso
                        _GUI_TICK_STATS["last_oh0_put_symbol"] = symbol_s
                        try:
                            _GUI_TICK_STATS["oh0_put_count"] = int(_GUI_TICK_STATS.get("oh0_put_count") or 0) + 1
                        except Exception:
                            _GUI_TICK_STATS["oh0_put_count"] = 1

                if trcode_s == TRCode.FUTURES.value:
                    try:
                        fut_prices = _GUI_TICK_STATS.get("fut_prices")
                        if not isinstance(fut_prices, deque):
                            fut_prices = deque(maxlen=2000)
                            _GUI_TICK_STATS["fut_prices"] = fut_prices

                        che = (tick or {}).get("hotime") or (tick or {}).get("chetime")
                        ts_ms = _kst_chetime_to_utc_ms(che)
                        px = float((tick or {}).get("price"))
                        fut_prices.append((int(ts_ms), float(px)))

                        # Prune old samples beyond ~15 minutes from the newest ts.
                        cutoff = int(ts_ms) - (15 * 60 * 1000)
                        while len(fut_prices) > 0 and int(fut_prices[0][0]) < cutoff:
                            fut_prices.popleft()
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            if trcode_s not in state.first_tick_printed:
                state.first_tick_printed.add(trcode_s)
                if trcode_s in (TRCode.FUTURES_BOOK.value, TRCode.OPTIONS_QUOTE.value):
                    ho = (tick or {}).get("hotime")
                    logger.info("[DBG][RT] first %s tick symbol=%s hotime=%s", trcode_s, symbol_s, ho)
                else:
                    che = (tick or {}).get("chetime")
                    logger.info("[DBG][RT] first %s tick symbol=%s chetime=%s", trcode_s, symbol_s, che)
        except Exception:
            pass

        fh = ticks_fh
        if fh is not None:
            try:
                if bool(getattr(fh, "closed", False)):
                    fh = None
            except Exception:
                pass
        if fh is not None:
            che = (tick or {}).get("hotime") or (tick or {}).get("chetime")
            ts_ms = _kst_chetime_to_utc_ms(che)
            try:
                write_jsonl_line(
                    fh,
                    {
                        "ts_ms": ts_ms,
                        "trcode": trcode_s,
                        "symbol": symbol_s,
                        "tick": _tick_to_compact_numeric(tick),
                    },
                )
            except Exception as e:
                logger.warning("[RT] tick write failed: %s", e)

        if trcode_s == "JIF":
            try:
                try:
                    jstatus = (tick or {}).get("jstatus")
                    jstatus_s = str(jstatus or "").strip()
                except Exception:
                    jstatus_s = ""

                try:
                    jangubun_s = str((tick or {}).get("jangubun") or "").strip()
                except Exception:
                    jangubun_s = ""

                # Always log on jstatus changes (including the first observed).
                try:
                    with _GUI_TICK_LOCK:
                        prev_js = _GUI_TICK_STATS.get("last_jif_jstatus")
                        changed = (prev_js is None) or (str(prev_js) != str(jstatus_s))
                        if changed:
                            _GUI_TICK_STATS["last_jif_jstatus"] = str(jstatus_s)
                except Exception:
                    changed = False

                if changed:
                    try:
                        che = (tick or {}).get("hotime") or (tick or {}).get("chetime")
                    except Exception:
                        che = None
                    logger.info(
                        "[JIF_STATUS] symbol=%s jangubun=%s jstatus=%s chetime=%s tick=%s",
                        symbol_s,
                        str((tick or {}).get("jangubun") or ""),
                        str(jstatus_s),
                        che,
                        _tick_to_compact_numeric(tick),
                    )

                if jangubun_s == "5" and jstatus_s == "41":
                    try:
                        already = bool(state.stop_requested)
                    except Exception:
                        already = False
                    if not already:
                        try:
                            state.stop_requested = True
                            state.stop_reason = "JIF jangubun=5 jstatus=41"
                        except Exception:
                            pass
                        try:
                            tp = getattr(predictor, "tick_processor", None)
                            if tp is not None:
                                tp.set_market_closed(True)
                        except Exception:
                            pass
                        try:
                            che = (tick or {}).get("hotime") or (tick or {}).get("chetime")
                        except Exception:
                            che = None
                        logger.info(
                            "[JIF_CLOSE] jangubun=5 jstatus=41 chetime=%s tick=%s",
                            che,
                            _tick_to_compact_numeric(tick),
                        )

                        # 장마감 후 당일 매매 백테스트 실행 (비동기)
                        try:
                            def run_backtest_async():
                                try:
                                    import sys
                                    from pathlib import Path

                                    # 프로젝트 루트 경로 추가
                                    project_root = Path(__file__).parent.parent
                                    if str(project_root) not in sys.path:
                                        sys.path.insert(0, str(project_root))

                                    from scripts.run_daily_backtest import run_daily_backtest_with_ohlcv

                                    logger.info("[JIF_CLOSE] 당일 매매 백테스트 시작 (비동기)")
                                    success = run_daily_backtest_with_ohlcv()
                                    if success:
                                        logger.info("[JIF_CLOSE] 당일 매매 백테스트 완료")
                                    else:
                                        logger.warning("[JIF_CLOSE] 당일 매매 백테스트 실패")
                                except Exception as e:
                                    logger.exception("[JIF_CLOSE] 백테스트 실행 예외: %s", e)

                            # 백그라운드 스레드에서 실행
                            backtest_thread = threading.Thread(target=run_backtest_async, daemon=True)
                            backtest_thread.start()
                            logger.info("[JIF_CLOSE] 백테스트 스레드 시작")
                        except Exception as e:
                            logger.warning("[JIF_CLOSE] 백테스트 스레드 시작 실패: %s", e)

                if jangubun_s == "5" and jstatus_s == "21":
                    try:
                        opened = bool(state.market_opened)
                    except Exception:
                        opened = False
                    if not opened:
                        try:
                            state.market_opened = True
                        except Exception:
                            pass
                        try:
                            che = (tick or {}).get("hotime") or (tick or {}).get("chetime")
                        except Exception:
                            che = None
                        logger.info(
                            "[JIF_OPEN] jangubun=5 jstatus=21 chetime=%s tick=%s",
                            che,
                            _tick_to_compact_numeric(tick),
                        )

                # JIF can be chatty; keep a lightweight rate limit but still record arrivals.
                now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
                with _GUI_TICK_LOCK:
                    last_ms = _GUI_TICK_STATS.get("last_jif_log_ms")
                    do_log = last_ms is None or (int(now_ms) - int(last_ms)) >= 1000
                    if do_log:
                        _GUI_TICK_STATS["last_jif_log_ms"] = int(now_ms)
                if do_log:
                    che = (tick or {}).get("hotime") or (tick or {}).get("chetime")
                    logger.info(
                        "[JIF] symbol=%s jangubun=%s jstatus=%s chetime=%s tick=%s",
                        symbol_s,
                        str((tick or {}).get("jangubun") or ""),
                        str(jstatus_s),
                        che,
                        _tick_to_compact_numeric(tick),
                    )
            except Exception:
                pass
            return

        try:
            tick_norm = None
            try:
                tick_norm = normalize_realtime_tick(trcode=trcode_s, symbol=symbol_s, tick=tick)
            except Exception:
                tick_norm = None
            payload = {"trcode": trcode_s, "symbol": symbol_s, "tick": tick}
            if isinstance(tick_norm, dict) and tick_norm:
                payload["tick_norm"] = tick_norm
            predictor.add_realtime_tick(payload)
        except Exception as e:
            logger.warning("[RT] predictor.add_realtime_tick failed: %s", e)

    return _on_realtime


def _make_message_callback(state: Any) -> Any:
    """Create message callback used to track subscription ACK progress.

    Side effects:
    - prints raw messages
    - increments `state.realtime_response_count` on ACK-like messages
    """

    def _on_message(*args):
        """Handle message events emitted by the eBest wrapper (ACK/progress)."""
        try:
            msg = args[1] if len(args) > 1 else args[0]
            s = str(msg)
        except Exception:
            return

        if s:
            logger.info("%s", s)

        is_ack = _is_ack_message(msg)
        if is_ack:
            try:
                state.realtime_response_count = min(
                    state.realtime_response_count + 1,
                    state.expected_realtime_responses,
                )
                if state.expected_realtime_responses > 0:
                    progress = state.realtime_response_count / state.expected_realtime_responses * 100
                    logger.info(
                        "[ACK] %d/%d (%.1f%%)",
                        state.realtime_response_count,
                        state.expected_realtime_responses,
                        progress,
                    )
            except Exception:
                return

    return _on_message
