"""
eBest API 실시간 연동

이 모듈은 원본 financial_derivatives_predict.py의 run_ebest_live_mode 로직을
모듈로 분리한 버전입니다.

주의:
- 프로젝트에 ebest 래퍼(예: ebest.OpenApi)가 설치/존재해야 실제 실행됩니다.
- 본 함수는 ebest 모듈 import 실패 시 에러 dict를 반환합니다.
"""

import asyncio
import html
import json
import logging
import os
import time
import gzip
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from zoneinfo import ZoneInfo

from .api import (
    _ebest_fetch_kp200_price_t8415,
    _ebest_fetch_kp200_symbol,
    _ebest_fetch_option_symbols,
    _ebest_fetch_front_month_and_all_option_symbols,
    _ebest_fetch_t2101_snapshot,
    _ebest_fetch_ij_snapshot,
    _ebest_fetch_t2301_open_map,
    _ebest_fetch_t2301_snapshot,
    _ebest_login,
    _ebest_register_realtime,
    _get_ebest_keys,
    _load_config,
)
from .callbacks import (
    _make_message_callback,
    _make_realtime_callback,
    update_gui_eval_dir_stats,
    update_gui_spot_index,
    update_meaningful_option_levels,
    set_meaningful_option_telegram_notifier,
)
from .options import (
    _filter_option_symbols_by_atm,
    filter_option_symbols_dynamic_otm_by_open,
    select_oi_window_symbols,
)
from core.utils import (
    get_option_month_yyyymm,
    get_pipeline_adaptive_indicator_symbol,
    normalize_adaptive_indicator_symbol,
)
from config import TRCode  # QUA-07: 매직 문자열 TRCode.FUTURES.value/TRCode.OPTIONS.value 등 → TRCode enum

logger = logging.getLogger(__name__)

_KST = ZoneInfo("Asia/Seoul")


def _last_complete_minute_ts_from_df(df: Any, now_dt: datetime) -> Optional[datetime]:
    """진행 중인 분봉을 제외한 마지막 완결 분 시각 (adaptive_mixin._pick_last_complete_ts 와 동일 규칙)."""
    try:
        if df is None or len(df) < 1:
            return None
        frame = df
        if not isinstance(frame.index, pd.DatetimeIndex):
            # tick_processor 분봉은 timestamp 컬럼을 가질 수 있으므로 인덱스로 승격
            try:
                if "timestamp" in frame.columns:
                    frame = frame.copy()
                    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
                    frame = frame.set_index("timestamp")
            except Exception as e:
                logger.debug("[eBest] timestamp conversion failed: %s", e)
                return None
        if not isinstance(frame.index, pd.DatetimeIndex):
            return None
        if len(frame.index) < 2:
            return frame.index[-1].to_pydatetime().replace(second=0, microsecond=0)
        now_min = now_dt.replace(second=0, microsecond=0)
        last_ts = frame.index[-1].to_pydatetime().replace(second=0, microsecond=0)
        if last_ts >= now_min:
            return frame.index[-2].to_pydatetime().replace(second=0, microsecond=0)
        return last_ts
    except Exception as e:
        logger.debug("[eBest] _get_reference_bar_time failed: %s", e)
        return None


def _ebest_adaptive_minute_df(predictor: Any, warmup_bars: Optional[int] = None) -> Any:
    """dual_mode가 항상 true이므로 KOSPI 현물(IJ_) 분봉 DataFrame을 반환한다."""
    tp = getattr(predictor, "tick_processor", None)
    if tp is None:
        return None
    wb = warmup_bars
    if wb is None:
        try:
            wb = int(getattr(tp, "default_futures_minutes", 120) or 120)
        except Exception as e:
            logger.debug("[eBest] warmup_bars parse failed, using default 120: %s", e)
            wb = 120
    try:
        wb = max(1, int(wb))
    except Exception as e:
        logger.debug("[eBest] warmup_bars validation failed, using default 120: %s", e)
        wb = 120
    return tp.get_kospi_minute_df(int(wb))


def _ebest_ij_tr_key(predictor: Any) -> str:
    """IJ_ 지수 키를 반환한다.

    dual_mode는 항상 true이므로 항상 KOSPI 지수("001")을 반환한다.
    """
    return "001"


def install_ebest_zz_confirm_telegram_hook(predictor: Any, bridge: Any) -> None:
    """[ZZ_CONFIRM_TRIGGER]와 동일 시점에 ZigZag 확정 알림을 텔레그램으로 보내는 훅을 predictor에 설치한다.

    dual_mode가 항상 true이므로 KOSPI 지수와 KP200 선물 모두 전송된다.
    가격·파동은 호출부에서 넘기는 `zigzag_state`(없으면 transformer features)로 표시해 지수/선물 분봉과 일치시킨다.
    bridge가 None이면 훅을 제거한다. (구버전 `_ebest_telegram_zz_confirm_hook`도 함께 제거)
    """
    if predictor is None:
        return
    try:
        setattr(predictor, "_ebest_telegram_zz_confirm_hook", None)
    except Exception:
        pass
    if bridge is None:
        try:
            setattr(predictor, "_ebest_telegram_zz_confirm_message_hook", None)
        except Exception:
            pass
        return

    pred_ref = predictor

    def _msg_hook(azz: int, bar_ts: Any, features: Any, zigzag_state: Any = None) -> None:
        try:
            n = getattr(bridge, "_notifier", None)
            if n is None or not bool(getattr(n, "is_configured", False)):
                return
            ai = int(azz)
            if ai > 0:
                lbl = "고점 확정 ▼"
            elif ai < 0:
                lbl = "저점 확정 ▲"
            else:
                return

            _raw_sym = str(get_pipeline_adaptive_indicator_symbol(pred_ref) or "").strip()
            _norm_sym = normalize_adaptive_indicator_symbol(_raw_sym)

            if _norm_sym == "KOSPI지수":
                _title = "📐 <b>KOSPI 지수 지그재그 피봇 확정</b>"
            elif _norm_sym == "KP200선물":
                _title = "📐 <b>KP200 선물 지그재그 피봇 확정</b>"
            else:
                _title = "📐 <b>적응형 지그재그 피봇 확정</b>"

            # 스윙 고/저/파동: ZigZagState 우선(지수/선물 분봉과 동일 출처), 없을 때만 transformer dict
            hi_v = lo_v = wave_v = None
            pivot_time = None
            confirm_time = None
            lag_bars = None
            if zigzag_state is not None:
                try:
                    v = float(getattr(zigzag_state, "last_swing_high", 0.0) or 0.0)
                    if v > 0:
                        hi_v = f"{v:.2f}"
                except Exception:
                    pass
                try:
                    v = float(getattr(zigzag_state, "last_swing_low", 0.0) or 0.0)
                    if v > 0:
                        lo_v = f"{v:.2f}"
                except Exception:
                    pass
                try:
                    wp = float(getattr(zigzag_state, "wave_size_pct", 0.0) or 0.0)
                    if wp > 0:
                        wave_v = f"{wp:.2f}%"
                except Exception:
                    pass
                try:
                    if ai > 0:
                        pivot_time = getattr(zigzag_state, "last_swing_high_time", None)
                        confirm_time = getattr(zigzag_state, "last_swing_high_confirm_time", None)
                        lag_bars = int(getattr(zigzag_state, "last_swing_high_lag_bars", 0) or 0)
                    else:
                        pivot_time = getattr(zigzag_state, "last_swing_low_time", None)
                        confirm_time = getattr(zigzag_state, "last_swing_low_confirm_time", None)
                        lag_bars = int(getattr(zigzag_state, "last_swing_low_lag_bars", 0) or 0)
                except Exception:
                    pass
            if isinstance(features, dict):
                if hi_v is None:
                    try:
                        v = float(features.get("azz_last_high") or 0.0)
                        if v > 0:
                            hi_v = f"{v:.2f}"
                    except Exception:
                        pass
                if lo_v is None:
                    try:
                        v = float(features.get("azz_last_low") or 0.0)
                        if v > 0:
                            lo_v = f"{v:.2f}"
                    except Exception:
                        pass
                if wave_v is None:
                    try:
                        # azz_wave_size_pct 는 정규화값(÷10) → 역산
                        raw = float(features.get("azz_wave_size_pct") or 0.0)
                        pct = raw * 10.0
                        if pct > 0:
                            wave_v = f"{pct:.2f}%"
                    except Exception:
                        pass

            # 확정된 피봇 가격 (고점 확정 → last_high, 저점 확정 → last_low)
            pivot_price = hi_v if ai > 0 else lo_v

            lines = [
                _title,
            ]
            if _raw_sym:
                lines.append(f"적응형 심볼: <code>{html.escape(_raw_sym)}</code>")
            lines.extend(
                [
                f"유형: <b>{html.escape(lbl)}</b>",
                ]
            )
            if pivot_time:
                lines.append(f"피봇봉: <code>{html.escape(pivot_time)}</code>")
            if pivot_price:
                lines.append(f"피봇가: <code>{html.escape(pivot_price)}</code>")
            if confirm_time:
                lines.append(f"확정봉: <code>{html.escape(confirm_time)}</code>")
            if lag_bars is not None and lag_bars > 0:
                lines.append(f"지연: <code>{lag_bars}봉</code>")
            if hi_v and lo_v:
                lines.append(
                    f"최근 스윙  고: <code>{html.escape(hi_v)}</code>"
                    f"  저: <code>{html.escape(lo_v)}</code>"
                )
            if wave_v:
                lines.append(f"파동 크기: <code>{html.escape(wave_v)}</code>")

            msg = "\n".join(lines)
            try:
                _log(
                    "[ZZ_TG_SEND] sym=%s azz=%d pivot_time=%s pivot_price=%s confirm_time=%s lag_bars=%s",
                    str(_raw_sym or ""),
                    int(ai),
                    str(pivot_time or ""),
                    str(pivot_price or ""),
                    str(confirm_time or ""),
                    str(lag_bars or ""),
                    level="debug",
                )
            except Exception:
                pass
            n.send_text(
                msg,
                parse_mode="HTML",
                debug_context={"kind": "zz_confirm_trigger"},
            )
        except Exception as e:
            _log("[ZZ_TG_SEND_FAIL] %s", str(e), level="warning")

    try:
        setattr(predictor, "_ebest_telegram_zz_confirm_message_hook", _msg_hook)
    except Exception:
        pass


def install_ebest_zz_candidate_telegram_hook(predictor: Any, bridge: Any) -> None:
    """피봇 후보 등록/갱신/취소 시 텔레그램으로 알림을 보내는 훅을 predictor에 설치한다.
    
    bridge가 None이면 훅을 제거한다.
    """
    if predictor is None:
        return
    
    # ZigZag 인스턴스 가져오기
    try:
        zz = getattr(predictor, "_adaptive_zigzag", None)
        if zz is None:
            return
    except Exception:
        return
    
    # 기존 훅 제거
    try:
        setattr(zz, "_telegram_event_callback", None)
    except Exception:
        pass
    
    if bridge is None:
        return
    
    pred_ref = predictor
    zz_ref = zz
    
    def _candidate_hook(action: str, close: float, payload: Dict[str, Any]) -> None:
        try:
            n = getattr(bridge, "_notifier", None)
            if n is None or not bool(getattr(n, "is_configured", False)):
                return
            
            _raw_sym = str(get_pipeline_adaptive_indicator_symbol(pred_ref) or "").strip()
            _norm_sym = normalize_adaptive_indicator_symbol(_raw_sym)
            
            # KOSPI 지수인 경우에만 전송 (피봇 확정과 동일한 정책)
            if _norm_sym != "KOSPI지수":
                return
            
            _title = "📊 <b>KOSPI 지수 피봇 후보</b>"
            
            # 액션별 메시지 구성
            if action == "후보등록":
                ctype = str(payload.get("candidate") or "").upper()[:1] or "?"
                swing_time = str(payload.get("swing_time") or "?")
                swing_px = float(payload.get("swing_price") or 0.0)
                rem = int(payload.get("remaining") or 0)
                thr = float(payload.get("thr_abs") or 0.0)
                prob = float(payload.get("prob", 0.0) or 0.0)
                
                lbl = "고점 후보 ▼" if ctype == "H" else "저점 후보 ▲"
                
                lines = [
                    _title,
                    f"유형: <b>{html.escape(lbl)}</b>",
                ]
                if _raw_sym:
                    lines.append(f"적응형 심볼: <code>{html.escape(_raw_sym)}</code>")
                if swing_time:
                    lines.append(f"후보봉: <code>{html.escape(swing_time)}</code>")
                if swing_px:
                    lines.append(f"후보가: <code>{swing_px:.2f}</code>")
                if rem:
                    lines.append(f"대기: <code>{rem}봉</code>")
                if thr:
                    lines.append(f"역전임계: <code>{thr:.2f}pt</code>")
                if prob:
                    lines.append(f"확률: <code>{prob:.2f}</code>")
                
            elif action == "후보갱신":
                ctype = str(payload.get("candidate") or "").upper()[:1] or "?"
                swing_time = str(payload.get("swing_time") or "?")
                swing_px = float(payload.get("swing_price") or 0.0)
                rem = int(payload.get("remaining") or 0)
                reason = str(payload.get("reason") or "")
                prob = float(payload.get("prob", 0.0) or 0.0)
                
                lbl = "고점 후보" if ctype == "H" else "저점 후보"
                
                lines = [
                    _title,
                    f"유형: <b>{html.escape(lbl)} 갱신</b>",
                ]
                if _raw_sym:
                    lines.append(f"적응형 심볼: <code>{html.escape(_raw_sym)}</code>")
                if swing_time:
                    lines.append(f"후보봉: <code>{html.escape(swing_time)}</code>")
                if swing_px:
                    lines.append(f"후보가: <code>{swing_px:.2f}</code>")
                if rem:
                    lines.append(f"잔여: <code>{rem}봉</code>")
                if reason:
                    lines.append(f"사유: <code>{html.escape(reason)}</code>")
                if prob:
                    lines.append(f"확률: <code>{prob:.2f}</code>")
                
            elif action == "취소":
                ctype = str(payload.get("prev_type") or "").upper()[:1] or "?"
                swing_time = str(payload.get("prev_time") or "?")
                swing_px = float(payload.get("prev_price") or 0.0)
                reason = str(payload.get("reason") or "")
                
                lbl = "고점 후보" if ctype == "H" else "저점 후보"
                
                lines = [
                    _title,
                    f"유형: <b>{html.escape(lbl)} 취소</b>",
                ]
                if _raw_sym:
                    lines.append(f"적응형 심볼: <code>{html.escape(_raw_sym)}</code>")
                if swing_time:
                    lines.append(f"후보봉: <code>{html.escape(swing_time)}</code>")
                if swing_px:
                    lines.append(f"후보가: <code>{swing_px:.2f}</code>")
                if reason:
                    lines.append(f"사유: <code>{html.escape(reason)}</code>")
            else:
                return
            
            if close:
                lines.append(f"현재가: <code>{close:.2f}</code>")
            
            msg = "\n".join(lines)
            try:
                n.send_text(
                    msg,
                    parse_mode="HTML",
                    debug_context={"kind": "zz_candidate_trigger"},
                )
            except Exception as e:
                _log("[ZZ_CANDIDATE_TG_SEND_FAIL] %s", str(e), level="warning")
        except Exception as e:
            _log("[ZZ_CANDIDATE_TG_HOOK_FAIL] %s", str(e), level="warning")
    
    try:
        setattr(zz_ref, "_telegram_event_callback", _candidate_hook)
    except Exception:
        pass
    
    # ── [ATR-MONITOR] ATR 급격 변동 텔레그램 콜백 설정 ─────────────────
    def _atr_spike_hook(message: str) -> None:
        try:
            n = getattr(bridge, "_notifier", None)
            if n is None or not bool(getattr(n, "is_configured", False)):
                return
            
            _raw_sym = str(get_pipeline_adaptive_indicator_symbol(pred_ref) or "").strip()
            _norm_sym = normalize_adaptive_indicator_symbol(_raw_sym)
            
            # KOSPI 지수와 KP200 선물 모두 전송
            if _norm_sym not in ("KOSPI지수", "KP200선물"):
                return
            
            try:
                n.send_text(
                    message,
                    parse_mode="HTML",
                    debug_context={"kind": "atr_spike"},
                )
            except Exception as e:
                _log("[ATR_SPIKE_TG_SEND_FAIL] %s", str(e), level="warning")
        except Exception as e:
            _log("[ATR_SPIKE_TG_HOOK_FAIL] %s", str(e), level="warning")
    
    try:
        zz_ref.set_atr_telegram_callback(_atr_spike_hook)
    except Exception as e:
        _log("[ATR_SPIKE_HOOK_SETUP_FAIL] %s", str(e), level="warning")


def _log(msg: str, *args, level: str = "info") -> None:
    """print 와 logger 를 동시에 처리하는 단일 유틸리티."""
    getattr(logger, level)(msg, *args)


def _fmt_atm_strike(v: Any) -> str:
    try:
        if v is None:
            return ""
        f = float(v)
        return str(int(f))
    except Exception:
        try:
            return str(v)
        except Exception:
            return ""


# ──────────────────────────────────────────────
# 공유 상태 dataclass
# ──────────────────────────────────────────────
@dataclass
class LiveState:
    """Mutable runtime state for the live mode loop.

    ARC-05: 이전에 setattr(state, ...) 로 동적으로 추가되던 속성들을
    모두 명시적 필드로 선언하고, 관련 메서드를 추가했다.
    IDE 타입 추론이 가능하며 오타가 런타임이 아닌 정적 분석 단계에서 발견된다.
    """
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    tick_counts: Dict[str, int] = field(
        default_factory=lambda: {TRCode.FUTURES.value: 0, TRCode.OPTIONS.value: 0, TRCode.OPTIONS_QUOTE.value: 0, TRCode.FUTURES_BOOK.value: 0, "JIF": 0, "IJ_": 0}
    )
    expected_realtime_responses: int = 0
    realtime_response_count: int = 0
    first_tick_printed: Set[str] = field(default_factory=set)

    # Market open gate (set by JIF callback).
    market_opened: bool = False

    # 장 종료 플래그 (CON-04: setattr 대신 명시적 필드 사용)
    stop_requested: bool = False
    stop_reason: str = ""

    kp200_prev_close: Optional[float] = None

    prediction_count: int = 0
    pending_evals: List[Dict[str, Any]] = field(default_factory=list)
    eval_count: int = 0
    eval_abs_err_sum: float = 0.0
    eval_dir_count: int = 0
    eval_dir_hit_count: int = 0
    eval_hold_count: int = 0

    option_calls_all: List[str] = field(default_factory=list)
    option_puts_all: List[str] = field(default_factory=list)
    subscribed_oh0: Set[str] = field(default_factory=set)
    subscribed_oc0: Set[str] = field(default_factory=set)
    open_oc0_subscribed: bool = False

    last_ij_refresh_epoch: Optional[float] = None
    last_t2101_refresh_epoch: Optional[float] = None

    last_result: Dict[str, Any] = field(default_factory=dict)

    # ARC-05: setattr으로 동적 추가되던 속성들을 명시적 필드로 선언
    _market_opened_by_tick_logged: bool = field(default=False, repr=False)

    def request_stop(self, reason: str) -> None:
        """stop_requested + stop_reason을 원자적으로 설정한다."""
        self.stop_requested = True
        self.stop_reason = reason

    def mark_open_oc0_subscribed(self) -> None:
        """open_oc0_subscribed를 True로 설정하고 로깅 중복을 방지한다."""
        self.open_oc0_subscribed = True


async def _initialize_api(
    api: Any,
    predictor: Any,
    state: LiveState,
    *,
    config_path: str,
    include_options: bool,
    option_month_info: Optional[str],
    opt_itm: int,
    opt_wait_sec: int,
    out_ticks: Optional[str],
    save_ticks_enabled: bool,
) -> tuple[Optional[Any], Optional[str]]:
    """Initialize eBest API session and register realtime subscriptions."""
    cfg_root: Dict[str, Any] = {}
    try:
        cfg_root = _load_config(config_path=str(config_path or "config.json")) or {}
        if not isinstance(cfg_root, dict):
            cfg_root = {}
    except Exception as e:
        logger.warning("[eBest] config load failed, using empty config: %s", e)
        cfg_root = {}

    opt_cfg = cfg_root.get("options_subscription") if isinstance(cfg_root.get("options_subscription"), dict) else {}
    ebest_cfg = cfg_root.get("ebest") if isinstance(cfg_root.get("ebest"), dict) else {}

    # Normalize/clamp option subscription knobs (defensive against config/CLI type issues)
    try:
        # config.json의 itm 값 우선 사용, CLI opt_itm은 fallback
        config_itm = int(opt_cfg.get("itm", 10))
        opt_itm_eff = int(opt_itm) if opt_itm is not None else config_itm
    except Exception as e:
        logger.debug("[eBest] opt_itm parse failed, using default 10: %s", e)
        opt_itm_eff = 10
    if opt_itm_eff < 0:
        opt_itm_eff = 0
    if opt_itm_eff > 20:
        opt_itm_eff = 20

    try:
        opt_wait_sec_eff = int(opt_wait_sec)
    except Exception as e:
        logger.debug("[eBest] opt_wait_sec parse failed, using default 2: %s", e)
        opt_wait_sec_eff = 2
    if opt_wait_sec_eff < 0:
        opt_wait_sec_eff = 0
    if opt_wait_sec_eff > 60:
        opt_wait_sec_eff = 60

    try:
        otm_open_min = float(opt_cfg.get("otm_open_min", 0.30) or 0.30)
    except Exception as e:
        logger.debug("[eBest] otm_open_min parse failed, using default 0.30: %s", e)
        otm_open_min = 0.30
    try:
        max_otm_calls = int(opt_cfg.get("max_otm_calls", 0) or 0)
    except Exception as e:
        logger.debug("[eBest] max_otm_calls parse failed, using default 0: %s", e)
        max_otm_calls = 0
    try:
        max_otm_puts = int(opt_cfg.get("max_otm_puts", 0) or 0)
    except Exception as e:
        logger.debug("[eBest] max_otm_puts parse failed, using default 0: %s", e)
        max_otm_puts = 0

    try:
        preopen_oh0_window = int(opt_cfg.get("preopen_oh0_window", 2) or 2)
    except Exception as e:
        logger.debug("[eBest] preopen_oh0_window parse failed, using default 2: %s", e)
        preopen_oh0_window = 2
    if preopen_oh0_window < 0:
        preopen_oh0_window = 0
    if preopen_oh0_window > 20:
        preopen_oh0_window = 20

    # OI 지지저항 분석용 구독 창 설정
    # oi_itm_count / oi_otm_count: ATM 기준 내가/외가 각 방향 개수.
    # 기본 10 → 콜/풋 각 최대 21개(ATM 포함), 총 최대 42개 심볼.
    # oi_rebalance_interval_sec: 장중 OC0 재구독 갱신 주기(초). 기본 60.
    try:
        oi_itm_count = int(opt_cfg.get("oi_itm_count", 10) or 10)
    except Exception as e:
        logger.debug("[eBest] oi_itm_count parse failed, using default 10: %s", e)
        oi_itm_count = 10
    if oi_itm_count < 1:
        oi_itm_count = 1
    if oi_itm_count > 30:
        oi_itm_count = 30

    try:
        oi_otm_count = int(opt_cfg.get("oi_otm_count", 10) or 10)
    except Exception as e:
        logger.debug("[eBest] oi_otm_count parse failed, using default 10: %s", e)
        oi_otm_count = 10
    if oi_otm_count < 1:
        oi_otm_count = 1
    if oi_otm_count > 30:
        oi_otm_count = 30

    try:
        oi_rebalance_interval_sec = float(opt_cfg.get("oi_rebalance_interval_sec", 60.0) or 60.0)
    except Exception as e:
        logger.debug("[eBest] oi_rebalance_interval_sec parse failed, using default 60.0: %s", e)
        oi_rebalance_interval_sec = 60.0
    if oi_rebalance_interval_sec < 10.0:
        oi_rebalance_interval_sec = 10.0
    if oi_rebalance_interval_sec > 600.0:
        oi_rebalance_interval_sec = 600.0

    try:
        _log(
            "[OPTIONS_CFG] opt_itm=%s->%d wait_sec=%s->%d otm_open_min=%.3f max_otm_calls=%d max_otm_puts=%d preopen_oh0_window=%d oi_itm=%d oi_otm=%d oi_rebalance_sec=%.0f",
            str(opt_itm),
            int(opt_itm_eff),
            str(opt_wait_sec),
            int(opt_wait_sec_eff),
            float(otm_open_min),
            int(max_otm_calls),
            int(max_otm_puts),
            int(preopen_oh0_window),
            int(oi_itm_count),
            int(oi_otm_count),
            float(oi_rebalance_interval_sec),
        )
    except Exception:
        pass

    # 1) 로그인
    appkey, appsecretkey = _get_ebest_keys(config_path=str(config_path or "config.json"))
    if not appkey or not appsecretkey:
        raise ValueError(
            "missing EBEST_APPKEY/EBEST_APPSECRET (or config.json ebest.appkey/ebest.appsecretkey)"
        )

    _log("⏳ Ebest login...")
    ok = await _ebest_login(api, appkey=appkey, appsecretkey=appsecretkey)
    if not ok:
        msg = getattr(api, "last_message", None)
        raise ConnectionError(f"eBest login failed: {msg}")

    is_sim = getattr(api, "is_simulation", None)
    server = "SIMULATION" if is_sim else ("LIVE" if is_sim is not None else "unknown")
    _log("✅ Ebest login success (server=%s)", server)

    kp200_symbol = None
    kp200_prev_close = None
    try:
        init_pack = await _ebest_fetch_front_month_and_all_option_symbols(
            api,
            option_month_info=str(option_month_info or "").strip() or get_option_month_yyyymm(datetime.now())[2:],
        )
        if isinstance(init_pack, dict):
            kp200_symbol = init_pack.get("kp200_symbol")
            kp200_prev_close = init_pack.get("kp200_prev_close")
    except Exception as e:
        logger.warning("[eBest] front month symbol fetch failed: %s", e)
        init_pack = None

    if not kp200_symbol:
        kp200_symbol = await _ebest_fetch_kp200_symbol(api)
    if not kp200_symbol:
        raise RuntimeError("failed to fetch kp200 symbol (t8432)")

    if kp200_prev_close is not None:
        try:
            _log("[eBest] KP200 prev_close=%.2f symbol=%s", float(kp200_prev_close), kp200_symbol)
        except Exception as e:
            logger.debug("[eBest] prev_close log failed: %s", e)
        try:
            state.kp200_prev_close = float(kp200_prev_close)
        except Exception as e:
            logger.debug("[eBest] prev_close state update failed: %s", e)
            state.kp200_prev_close = None
        try:
            setattr(predictor, "kp200_prev_close", state.kp200_prev_close)
        except Exception as e:
            logger.debug("[eBest] prev_close predictor setattr failed: %s", e)

    async def _post_open_init() -> None:
        """Defer snapshot queries & option subscriptions until market open JIF is seen."""
        start_wait = time.monotonic()
        try:
            while not bool(state.market_opened):
                # Fallback: JIF may not arrive depending on wrapper/env.
                # If we are inside market hours and have been waiting long enough,
                # open the gate so snapshots/context are still populated.
                try:
                    waited = float(time.monotonic() - float(start_wait))
                except Exception:
                    waited = 0.0

                if waited >= 30.0:
                    try:
                        now_kst = datetime.now(tz=_KST)
                        if int(now_kst.weekday()) < 5:
                            open_dt = now_kst.replace(hour=9, minute=0, second=0, microsecond=0)
                            close_dt = now_kst.replace(hour=15, minute=45, second=0, microsecond=0)
                            if open_dt <= now_kst <= close_dt:
                                state.market_opened = True
                                _log(
                                    "[GATE_FALLBACK] JIF open not received; opening gate by time policy now=%s waited=%.1fs",
                                    now_kst.strftime("%Y-%m-%d %H:%M:%S"),
                                    float(waited),
                                )
                                break
                    except Exception:
                        pass

                await asyncio.sleep(0.5)
        except Exception:
            return

        _log("[GATE] market open detected; starting t8415/t2101/t2301 snapshots")
        try:
            _log(
                "[OPEN_FLOW] include_options=%s option_month_info=%s opt_itm=%s otm_open_min=%.3f max_otm_calls=%d max_otm_puts=%d",
                str(bool(include_options)),
                str(option_month_info or ""),
                str(opt_itm),
                float(otm_open_min),
                int(max_otm_calls),
                int(max_otm_puts),
            )
        except Exception:
            pass

        kp200_price = None
        try:
            kp200_price = await _ebest_fetch_kp200_price_t8415(api, symbol=kp200_symbol)
            if kp200_price is not None:
                _log("[eBest] KP200 t8415 price=%.2f symbol=%s", kp200_price, kp200_symbol)
            else:
                _log("[eBest] KP200 t8415 price unavailable symbol=%s", kp200_symbol)
        except Exception:
            _log("[eBest] KP200 t8415 price unavailable symbol=%s", kp200_symbol)

        kp200_snap = None
        try:
            kp200_snap = await _ebest_fetch_t2101_snapshot(api, focode=kp200_symbol)
            if isinstance(kp200_snap, dict) and kp200_snap:
                try:
                    _log(
                        "[eBest] KP200 t2101 price=%.2f open=%.2f high=%.2f low=%.2f mgjv=%.0f impv=%.2f focode=%s",
                        float(kp200_snap.get("price") or 0.0),
                        float(kp200_snap.get("open") or 0.0),
                        float(kp200_snap.get("high") or 0.0),
                        float(kp200_snap.get("low") or 0.0),
                        float(kp200_snap.get("mgjv") or 0.0),
                        float(kp200_snap.get("impv") or 0.0),
                        str(kp200_snap.get("focode") or kp200_symbol),
                    )
                except Exception:
                    _log("[eBest] KP200 t2101 snapshot received (focode=%s)", kp200_symbol)
        except Exception:
            kp200_snap = None

        # Attach to pipeline/predictor if supported (LLM context enrichment)
        try:
            setter = getattr(predictor, "set_market_snapshots", None)
            if callable(setter):
                setter(t2101=kp200_snap)
        except Exception:
            pass

        # IJ 스냅샷: 항상 요청 (KOSPI 데이터용)
        _ij_key = _ebest_ij_tr_key(predictor)
        try:
            ij_snap = await _ebest_fetch_ij_snapshot(api, tr_key=_ij_key)
            if isinstance(ij_snap, dict) and ij_snap:
                try:
                    _log(
                        "[eBest] IJ(%s) spot jisu=%.2f time=%s",
                        str(_ij_key),
                        float(ij_snap.get("jisu") or 0.0),
                        str(ij_snap.get("time") or ""),
                    )
                except Exception:
                    _log("[eBest] IJ(%s) snapshot received", str(_ij_key))

                try:
                    setter = getattr(predictor, "set_market_snapshots", None)
                    if callable(setter):
                        setter(ij_=ij_snap)
                except Exception:
                    pass
                try:
                    update_gui_spot_index(spot_index=ij_snap.get("jisu"), spot_time=ij_snap.get("time"))
                except Exception:
                    pass
                try:
                    state.last_ij_refresh_epoch = float(time.time())
                except Exception:
                    state.last_ij_refresh_epoch = None
        except Exception:
            pass

        if not bool(include_options):
            try:
                _log("[OPEN_FLOW] skip post-open OC0 subscribe: include_options=False")
            except Exception:
                pass
            return

        # After open: fetch t2301 snapshot/open map and subscribe missing OC0 (best-effort).
        try:
            ym_raw = str(option_month_info).strip() if option_month_info else ""
            ym6 = ym_raw
            if len(ym6) == 4 and ym6.isdigit():
                ym6 = "20" + ym6
            if len(ym6) != 6:
                ym6 = str(get_option_month_yyyymm(datetime.now()))
            gubun = "W" if str(option_month_info).strip().upper().startswith("W") else "G"

            t2301 = await _ebest_fetch_t2301_snapshot(api, yyyymm=str(ym6), gubun=gubun)
            try:
                open_map = await _ebest_fetch_t2301_open_map(api, yyyymm=str(ym6), gubun=gubun)
            except Exception:
                open_map = None

            try:
                if not isinstance(open_map, dict) or not open_map:
                    _log("[OPEN_FLOW] t2301 open_map unavailable (yyyymm=%s gubun=%s); OTM filtering will be disabled", str(ym6), str(gubun))
            except Exception:
                pass

            if isinstance(t2301, dict) and t2301:
                _log(
                    "[eBest] t2301 yyyymm=%s gubun=%s cimpv=%.3f pimpv=%.3f histimpv=%.3f jandate=%.0f calls=%d puts=%d",
                    str(t2301.get("yyyymm") or ym6),
                    str(t2301.get("gubun") or gubun),
                    float(t2301.get("cimpv") or 0.0),
                    float(t2301.get("pimpv") or 0.0),
                    float(t2301.get("histimpv") or 0.0),
                    float(t2301.get("jandatecnt") or 0.0),
                    int(t2301.get("call_count") or 0),
                    int(t2301.get("put_count") or 0),
                )

            try:
                setter = getattr(predictor, "set_market_snapshots", None)
                if callable(setter):
                    setter(t2301=t2301)
            except Exception:
                pass

            try:
                if not bool(state.open_oc0_subscribed):
                    underlying_open = None
                    try:
                        if isinstance(kp200_snap, dict):
                            underlying_open = float(kp200_snap.get("open") or 0.0)
                    except Exception:
                        underlying_open = None
                    try:
                        if not underlying_open:
                            underlying_open = float(predictor.tick_processor.get_current_price() or 0.0)
                    except Exception:
                        underlying_open = 0.0

                    call_open_map = None
                    put_open_map = None
                    try:
                        if isinstance(open_map, dict):
                            # _ebest_fetch_t2301_open_map 반환 키: "calls"/"puts"
                            call_open_map = open_map.get("calls") or open_map.get("call_open_map")
                            put_open_map  = open_map.get("puts")  or open_map.get("put_open_map")
                    except Exception:
                        call_open_map = None
                        put_open_map = None

                    # tick_processor에 시가 맵 주입 — OTM 프리미엄 변화율 계산 기반 데이터
                    try:
                        if isinstance(call_open_map, dict) or isinstance(put_open_map, dict):
                            predictor.tick_processor.set_option_open_map(
                                call_open_map=call_open_map or {},
                                put_open_map=put_open_map or {},
                            )
                    except Exception as _e:
                        _log("[OPEN_FLOW] set_option_open_map 실패: %s", _e)

                    try:
                        c_n = len(call_open_map) if isinstance(call_open_map, dict) else 0
                        p_n = len(put_open_map) if isinstance(put_open_map, dict) else 0
                        _log("[OPEN_FLOW] open_map sizes: call=%d put=%d", int(c_n), int(p_n))
                    except Exception:
                        pass

                    new_calls, new_puts, new_atm = filter_option_symbols_dynamic_otm_by_open(
                        predictor,
                        calls=list(state.option_calls_all or []),
                        puts=list(state.option_puts_all or []),
                        itm_count=int(opt_itm_eff),
                        underlying_price=float(underlying_open or 0.0),
                        call_open_map=call_open_map,
                        put_open_map=put_open_map,
                        otm_open_min=float(otm_open_min),
                        max_otm_calls=int(max_otm_calls),
                        max_otm_puts=int(max_otm_puts),
                    )

                    # OI 지지/저항 분석을 위해 ATM 기준 oi_itm_count/oi_otm_count 범위의
                    # 심볼을 추가 구독한다. ATM이 동적으로 변하므로 장 개시 시점의
                    # 실제 기초자산 가격을 기준으로 select_oi_window_symbols로 재계산한다.
                    try:
                        oi_open_calls, oi_open_puts, _ = select_oi_window_symbols(
                            calls=list(state.option_calls_all or []),
                            puts=list(state.option_puts_all or []),
                            underlying_price=float(underlying_open or 0.0),
                            itm_count=int(oi_itm_count),
                            otm_count=int(oi_otm_count),
                        )
                        oi_open_syms = set(
                            str(x) for x in (list(oi_open_calls or []) + list(oi_open_puts or [])) if x
                        )
                    except Exception as _oi_e:
                        oi_open_syms = set()
                        _log("[OPEN_FLOW] select_oi_window_symbols 실패(post-open): %s", str(_oi_e))
                    # OTM 프리미엄 구독 목록과 OI 창 목록을 합산
                    _otm_set = set(str(x) for x in (list(new_calls or []) + list(new_puts or [])) if x)
                    desired = _otm_set | oi_open_syms

                    try:
                        atm2 = f" ATM={_fmt_atm_strike(new_atm)}" if new_atm is not None else ""
                        _log(
                            "[eBest] subscribe OC0 (post-open) otm_calls=%d otm_puts=%d oi_window=%d total=%d open=%.2f%s (otm_open_min=%.3f otm_caps call=%d put=%d)",
                            int(len(new_calls or [])),
                            int(len(new_puts or [])),
                            int(len(oi_open_syms)),
                            int(len(desired)),
                            float(underlying_open or 0.0),
                            atm2,
                            float(otm_open_min),
                            int(max_otm_calls),
                            int(max_otm_puts),
                        )
                    except Exception:
                        pass

                    try:
                        already = set([str(x) for x in (state.subscribed_oc0 or set()) if x])
                    except Exception:
                        already = set()
                    missing = sorted(list(desired - already))

                    try:
                        _log(
                            "[eBest] post-open subscription breakdown: OC0 calls=%d puts=%d desired=%d already=%d missing=%d",
                            int(len(new_calls or [])),
                            int(len(new_puts or [])),
                            int(len(desired)),
                            int(len(already)),
                            int(len(missing)),
                        )
                    except Exception:
                        pass

                    atm_str = f" ATM={_fmt_atm_strike(new_atm)}" if new_atm is not None else ""
                    _log(
                        "[OPEN][OC0] desired=%d missing=%d open=%.2f%s",
                        int(len(desired)),
                        int(len(missing)),
                        float(underlying_open or 0.0),
                        atm_str,
                    )
                    added = 0
                    for sym in missing:
                        try:
                            await _ebest_register_realtime(api, trcode=TRCode.OPTIONS.value, symbol=str(sym))
                            added += 1
                            try:
                                state.subscribed_oc0.add(str(sym))
                            except Exception:
                                pass
                        except Exception:
                            pass
                        await asyncio.sleep(0.05)
                    _log("[OPEN][OC0] added_missing=%d", int(added))
                    try:
                        state.open_oc0_subscribed = True
                    except Exception:
                        pass
                else:
                    try:
                        _log("[OPEN_FLOW] skip post-open OC0 subscribe: already executed (open_oc0_subscribed=True)")
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    ticks_fh = None
    if save_ticks_enabled:
        path = str(out_ticks) if out_ticks else f"ticks_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        try:
            if str(path).lower().endswith(".gz"):
                ticks_fh = gzip.open(path, "at", encoding="utf-8")
            else:
                ticks_fh = open(path, "a", encoding="utf-8")
            logger.info("[TICKS] saving to: %s", path)
        except Exception as e:
            logger.warning("[TICKS] could not open tick file: %s", e)
            ticks_fh = None

    # 콜백 등록
    on_realtime = _make_realtime_callback(predictor, state, ticks_fh)
    on_message  = _make_message_callback(state)

    for sig_name, cb in (("on_realtime", on_realtime), ("on_message", on_message)):
        sig = getattr(api, sig_name, None)
        if sig is not None and hasattr(sig, "connect"):
            try:
                sig.connect(cb)
            except Exception as e:
                logger.warning("[SIGNAL] connect %s failed: %s", sig_name, e)

    # 선물 체결(FC0) + 호가/현재가(FH0; FO0 대체) + 장운영(JIF) + 현물지수(IJ_)
    # IJ_ '001'(KOSPI 지수)는 항상 구독 (KOSPI 데이터용)
    state.expected_realtime_responses = 4
    _ij_key = _ebest_ij_tr_key(predictor)
    _log(
        "실시간 시세를 요청합니다. (선물 2개 + 장운영 1개 + 지수 1개)\n  └─ FC0 %s\n  └─ FH0 %s\n  └─ JIF (key='0')\n  └─ IJ_ (key='%s')",
        kp200_symbol,
        kp200_symbol,
        str(_ij_key),
    )
    await _ebest_register_realtime(api, trcode=TRCode.FUTURES.value, symbol=kp200_symbol)
    await _ebest_register_realtime(api, trcode=TRCode.FUTURES_BOOK.value, symbol=kp200_symbol)

    try:
        ok = await _ebest_register_realtime(api, trcode="JIF", symbol="0")
        if not ok:
            _log("[eBest] JIF realtime subscribe failed (ignored)")
    except Exception:
        _log("[eBest] JIF realtime subscribe failed (ignored)")

    try:
        ok = await _ebest_register_realtime(api, trcode="IJ_", symbol=str(_ij_key))
        if not ok:
            _log("[eBest] IJ_ realtime subscribe failed (ignored)")
    except Exception:
        _log("[eBest] IJ_ realtime subscribe failed (ignored)")

    # t8415/t8418 분봉 데이터 요청 (include_options와 상관없이 항상 실행)
    try:
        from .api import _ebest_fetch_kp200_ohlcv_t8415, _ebest_fetch_kospi_ohlcv_t8418
        import pandas as pd
        # config에서 target_day 읽기 (prediction 섹션)
        target_date = cfg_root.get("prediction", {}).get("target_day", "")
        # 빈 문자열이면 None로 처리
        if target_date == "" or target_date is None:
            target_date = None
        today_date = datetime.now(tz=_KST).strftime("%Y%m%d")
        now_kst = datetime.now(tz=_KST)

        # target_day가 설정되어 있으면 그 날짜 사용, 없으면 오늘 날짜 사용
        if target_date:
            # target_day가 설정되어 있으면 그 날짜 사용
            pass
        else:
            # target_day가 없으면 오늘 날짜 사용
            target_date = today_date

        # KP200 분봉 요청 (t8415)
        _log("[t8415] KP200 분봉 데이터 요청 시작 (symbol=%s, date=%s, current_time=%s)",
             kp200_symbol, target_date, now_kst.strftime("%H:%M:%S"))
        kp200_ohlcv = await _ebest_fetch_kp200_ohlcv_t8415(api, symbol=kp200_symbol, yyyymmdd=target_date, ncnt=1)
        is_today = (target_date == today_date)
        if kp200_ohlcv:
            _log("[t8415] KP200 분봉 데이터 수신 완료 (bars=%d)", len(kp200_ohlcv))
            if kp200_ohlcv:
                latest = kp200_ohlcv[-1]
                _log("[t8415] KP200 최신 분봉: time=%s open=%.2f high=%.2f low=%.2f close=%.2f volume=%.0f",
                     latest.get("time", ""), latest.get("open", 0), latest.get("high", 0),
                     latest.get("low", 0), latest.get("close", 0), latest.get("volume", 0))

            # DataFrame으로 변환하여 predictor에 전달
            try:
                df = pd.DataFrame(kp200_ohlcv)
                # timestamp를 datetime으로 변환 (rename 전에 수행)
                df["timestamp"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S")
                df = df.drop(columns=["date", "time"])
                df = df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                })
                df = df.set_index("timestamp").sort_index()

                # predictor에 데이터 설정 (tick_processor 초기화)
                try:
                    if predictor and hasattr(predictor, "tick_processor"):
                        # 틱 데이터 초기화 (t8415 데이터로 대체)
                        predictor.tick_processor.futures_ticks = []
                        predictor.tick_processor._futures_minute_df = df
                        _log("[t8415] KP200 분봉 데이터를 tick_processor에 설정 완료 (bars=%d)", len(df))
                except Exception as e:
                    _log("[t8415] tick_processor 설정 실패: %s", str(e))
            except Exception as e:
                _log("[t8415] DataFrame 변환 실패: %s", str(e))
        else:
            _log("[t8415] KP200 분봉 데이터 없음 (장 마감 후이거나 데이터 없음)")
    except Exception as e:
        _log("[t8415] KP200 분봉 요청 실패: %s", str(e))

    # KOSPI 분봉 요청 (t8418 - KOSPI 지수는 t8418 사용)
    try:
        kospi_symbol = "001"  # KOSPI 지수 코드
        _log("[t8418] KOSPI 분봉 데이터 요청 시작 (symbol=%s, date=%s, current_time=%s)",
             kospi_symbol, target_date, now_kst.strftime("%H:%M:%S"))
        kospi_ohlcv = await _ebest_fetch_kospi_ohlcv_t8418(api, symbol=kospi_symbol, yyyymmdd=target_date, ncnt=1)
        if kospi_ohlcv:
            _log("[t8418] KOSPI 분봉 데이터 수신 완료 (bars=%d)", len(kospi_ohlcv))
            latest = kospi_ohlcv[-1]
            _log("[t8418] KOSPI 최신 분봉: time=%s open=%.2f high=%.2f low=%.2f close=%.2f volume=%.0f",
                 latest.get("time", ""), latest.get("open", 0), latest.get("high", 0),
                 latest.get("low", 0), latest.get("close", 0), latest.get("volume", 0))

            # DataFrame으로 변환하여 predictor에 전달
            try:
                df = pd.DataFrame(kospi_ohlcv)
                # timestamp를 datetime으로 변환 (rename 전에 수행)
                df["timestamp"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S")
                df = df.drop(columns=["date", "time"])
                df = df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                })
                df = df.set_index("timestamp").sort_index()

                # predictor에 데이터 설정 (kospi 초기화)
                try:
                    if predictor and hasattr(predictor, "tick_processor"):
                        predictor.tick_processor._kospi_minute_df = df
                        _log("[t8418] KOSPI 분봉 데이터를 tick_processor에 설정 완료 (bars=%d)", len(df))
                except Exception as e:
                    _log("[t8418] tick_processor 설정 실패: %s", str(e))
            except Exception as e:
                _log("[t8418] DataFrame 변환 실패: %s", str(e))
        else:
            _log("[t8418] KOSPI 분봉 데이터 없음 (장 마감 후이거나 데이터 없음)")
    except Exception as e:
        _log("[t8418] KOSPI 분봉 요청 실패: %s", str(e))

    # 옵션 구독
    if include_options and not option_month_info:
        option_month_info = get_option_month_yyyymm(datetime.now())[2:]

    if include_options:
        await asyncio.sleep(max(0, opt_wait_sec_eff))
        calls = []
        puts = []
        try:
            if isinstance(init_pack, dict):
                calls = init_pack.get("calls") or []
                puts = init_pack.get("puts") or []
        except Exception:
            calls = []
            puts = []

        if not calls or not puts:
            calls, puts = await _ebest_fetch_option_symbols(api, option_month_info=option_month_info)

        try:
            state.option_calls_all = list(calls or [])
            state.option_puts_all = list(puts or [])
        except Exception:
            state.option_calls_all = []
            state.option_puts_all = []

        preopen_underlying = None
        try:
            pc = state.kp200_prev_close
            if pc is None:
                pc = getattr(predictor, "kp200_prev_close", None)
            if pc is not None and float(pc) > 0.0:
                preopen_underlying = float(pc)
        except Exception:
            preopen_underlying = None

        # Pre-open: 설정대로 ITM + OTM 옵션 선택
        # ITM 10개 (콜/풋 각각) + OTM 콜 20개 + OTM 풋 30개 = 총 60개
        sel_calls, sel_puts, atm = _filter_option_symbols_by_atm(
            predictor,
            calls=list(calls or []),
            puts=list(puts or []),
            itm_count=int(opt_itm_eff),
            otm_count_call=int(max_otm_calls),
            otm_count_put=int(max_otm_puts),
            underlying_price=preopen_underlying,
        )
        
        # fallback: 설정대로 선택 실패 시 OI 기준 선택
        if not sel_calls and not sel_puts:
            sel_calls, sel_puts, atm = select_oi_window_symbols(
                calls=list(calls or []),
                puts=list(puts or []),
                underlying_price=float(preopen_underlying or 0.0) if preopen_underlying else 0.0,
                itm_count=int(oi_itm_count),
                otm_count=int(oi_otm_count),
            )
        atm_str = f" ATM={_fmt_atm_strike(atm)}" if atm is not None else ""
        prev_close_str = ""
        try:
            pc = state.kp200_prev_close
            if pc is None:
                pc = getattr(predictor, "kp200_prev_close", None)
            if pc is not None:
                prev_close_str = f" prev_close={float(pc):.2f}"
        except Exception:
            prev_close_str = ""
        _log(
            "[eBest] subscribe OC0 (pre-open) calls=%d puts=%d%s%s (itm=%d otm_caps call=%d put=%d)",
            len(sel_calls),
            len(sel_puts),
            atm_str,
            prev_close_str,
            int(opt_itm_eff),
            int(max_otm_calls),
            int(max_otm_puts),
        )

        # OH0는 호가 데이터(bid/ask/IV) 제공. OI 분석과 동일한 행사가 범위를 커버한다.
        # preopen_oh0_window와 oi_itm_count/oi_otm_count 중 큰 값으로 선택.
        _oh0_window = max(int(preopen_oh0_window), int(oi_itm_count), int(oi_otm_count))
        oh0_calls, oh0_puts, oh0_atm = _filter_option_symbols_by_atm(
            predictor,
            calls=list(calls or []),
            puts=list(puts or []),
            itm_count=int(_oh0_window),
            otm_count_call=int(_oh0_window),
            otm_count_put=int(_oh0_window),
            underlying_price=preopen_underlying,
        )
        oh0_syms = list(dict.fromkeys(list(oh0_calls or []) + list(oh0_puts or [])))
        oh0_atm_str = f" ATM={_fmt_atm_strike(oh0_atm)}" if oh0_atm is not None else ""
        _log(
            "[eBest] subscribe OH0 (pre-open, ATM±%d) symbols=%d%s%s",
            int(preopen_oh0_window),
            len(oh0_syms),
            oh0_atm_str,
            prev_close_str,
        )

        try:
            _log(
                "[eBest] pre-open subscription breakdown: OC0 calls=%d puts=%d OH0 symbols=%d total=%d",
                int(len(sel_calls)),
                int(len(sel_puts)),
                int(len(oh0_syms)),
                int(3 + int(len(sel_calls) + len(sel_puts)) + int(len(oh0_syms))),
            )
        except Exception:
            pass

        state.expected_realtime_responses = 3 + int(len(sel_calls) + len(sel_puts)) + int(len(oh0_syms))
        _log("실시간 시세를 요청합니다. (총 %d개)", state.expected_realtime_responses)

        for sym in sel_calls + sel_puts:
            _log("  └─ OC0 %s", sym)
            await _ebest_register_realtime(api, trcode=TRCode.OPTIONS.value, symbol=sym)
            try:
                state.subscribed_oc0.add(str(sym))
            except Exception:
                pass
            await asyncio.sleep(0.05)

        for sym in oh0_syms:
            try:
                state.subscribed_oh0.add(str(sym))
            except Exception:
                pass
            _log("  └─ OH0 %s", sym)
            await _ebest_register_realtime(api, trcode=TRCode.OPTIONS_QUOTE.value, symbol=sym)
            await asyncio.sleep(0.05)

    # Gate snapshot queries + option subscriptions until JIF market open.
    try:
        asyncio.create_task(_post_open_init())
    except Exception:
        pass

    return ticks_fh, kp200_symbol


def _try_evaluate_pending(
    state: LiveState,
    predictor: Any,
    df: Any,
) -> None:
    """Evaluate pending predictions once target time has passed."""
    try:
        state._lock.acquire()
    except Exception:
        pass
    done_idx = []
    for i, ev in enumerate(state.pending_evals):
        target_ts = ev.get("target_timestamp")
        try:
            target_dt = pd.to_datetime(target_ts) if target_ts else None
        except Exception:
            target_dt = None

        if target_dt is None:
            continue

        try:
            last_ts  = pd.to_datetime(df.iloc[-1].get("timestamp"))
            can_eval = last_ts >= target_dt
        except Exception:
            can_eval = False

        if not can_eval:
            continue

        done_idx.append(i)

        try:
            _ts  = pd.to_datetime(df["timestamp"], errors="coerce")
            tmp  = df.copy()
            tmp["_ts"] = _ts
            tmp  = tmp.dropna(subset=["_ts"]).sort_values("_ts")
            row  = tmp[tmp["_ts"] >= target_dt].head(1)
            if row is None or len(row) == 0:
                row = tmp.tail(1)
            v = row.iloc[0].get("Close")
            if v is None:
                v = row.iloc[0].get("close")
            actual_price = float(v)
        except Exception:
            actual_price = None

        predicted_price = None
        base_price = None
        signal = None
        transformer_prob = None
        tft_prob = None
        try:
            if ev.get("predicted_price") is not None:
                predicted_price = float(ev.get("predicted_price"))
            base_price = float(ev.get("base_price"))
            signal = ev.get("signal")
            if ev.get("transformer_prob") is not None:
                transformer_prob = float(ev.get("transformer_prob"))
            if ev.get("tft_prob") is not None:
                tft_prob = float(ev.get("tft_prob"))
        except Exception:
            predicted_price = None
            base_price = None
            signal = None
            transformer_prob = None
            tft_prob = None

        if actual_price is None or base_price is None:
            continue

        act_d = actual_price - base_price

        # Adaptive ensemble weight update (best-effort) when probs exist.
        try:
            if transformer_prob is not None and tft_prob is not None and act_d != 0:
                act_up = bool(act_d > 0)
                t_up = bool(float(transformer_prob) >= 0.5)
                f_up = bool(float(tft_prob) >= 0.5)
                t_ok = (t_up == act_up)
                f_ok = (f_up == act_up)
                npred = getattr(predictor, "numeric_predictor", None)
                fn = getattr(npred, "update_adaptive_weights", None)
                if callable(fn):
                    fn(transformer_correct=bool(t_ok), tft_correct=bool(f_ok))
        except Exception:
            pass

        # price-based evaluation (legacy)
        if predicted_price is not None:
            abs_err = abs(actual_price - predicted_price)
            pred_d = predicted_price - base_price
            dir_hit = (pred_d > 0) == (act_d > 0) if pred_d != 0 and act_d != 0 else None

            state.eval_count += 1
            state.eval_abs_err_sum += abs_err
            try:
                s = str(signal or "").strip().upper()
            except Exception:
                s = ""

            if s == "HOLD":
                try:
                    state.eval_hold_count += 1
                except Exception:
                    pass

            if s in ("BUY", "SELL") and dir_hit is not None:
                state.eval_dir_count += 1
                if dir_hit:
                    state.eval_dir_hit_count += 1
            continue

        # direction-only evaluation (pipeline)
        try:
            s = str(signal or "").strip().upper()
        except Exception:
            s = ""

        dir_hit2 = None
        if s == "BUY":
            dir_hit2 = bool(act_d > 0)
        elif s == "SELL":
            dir_hit2 = bool(act_d < 0)
        elif s == "HOLD":
            try:
                state.eval_hold_count += 1
            except Exception:
                pass

        if dir_hit2 is not None:
            state.eval_dir_count += 1
            if dir_hit2:
                state.eval_dir_hit_count += 1

    # 완료된 항목을 역순으로 제거 (인덱스 안전)
    state.pending_evals = [ev for i, ev in enumerate(state.pending_evals) if i not in done_idx]
    try:
        state._lock.release()
    except Exception:
        pass


def _append_eval_metrics(result: Dict[str, Any], state: LiveState) -> None:
    """Attach running evaluation metrics to the result dict (best-effort)."""
    try:
        state._lock.acquire()
    except Exception:
        pass
    if state.eval_count > 0:
        result["eval_mae"] = state.eval_abs_err_sum / state.eval_count

    if state.eval_dir_count > 0:
        result["eval_dir_hits"] = state.eval_dir_hit_count
        result["eval_dir_total"] = state.eval_dir_count
        result["eval_dir_rate"] = state.eval_dir_hit_count / state.eval_dir_count * 100.0

    try:
        result["eval_hold_count"] = int(state.eval_hold_count or 0)
    except Exception:
        pass

    try:
        hits = int(state.eval_dir_hit_count or 0)
        total = int(state.eval_dir_count or 0)
        rate = float(hits / total * 100.0) if total > 0 else 0.0
        hold = int(state.eval_hold_count or 0)
        update_gui_eval_dir_stats(hits=hits, total=total, rate=rate, hold=hold)
    except Exception:
        pass
    try:
        state._lock.release()
    except Exception:
        pass


def _log_model_outputs(result: Dict[str, Any]) -> None:
    """Pretty-print optional model_outputs blocks for debugging."""
    model_outputs = result.get("model_outputs", {}) if isinstance(result, dict) else {}

    round_time = None
    try:
        if isinstance(result, dict):
            round_time = result.get("prediction_time") or result.get("round_time")
    except Exception:
        round_time = None

    def _normalize_payload(payload: Any) -> Any:
        """Normalize/round selected numeric fields for stable logging."""
        if not isinstance(payload, dict):
            return payload

        p = dict(payload)
        if round_time is not None and "round_time" not in p:
            p["round_time"] = round_time

        def _round_field(key: str, ndigits: int) -> None:
            """Round a numeric-like field in-place when present."""
            try:
                v = p.get(key)
                if isinstance(v, bool) or v is None:
                    return
                if isinstance(v, (int, float)):
                    p[key] = round(float(v), int(ndigits))
                    return
                if isinstance(v, str) and v.strip() != "":
                    p[key] = round(float(v), int(ndigits))
            except Exception:
                return

        for k in (
            "current_price",
            "predicted_price",
            "base_price",
            "price_change",
            "pred_range_low",
            "pred_range_high",
            "expiry_final_prediction_price",
        ):
            _round_field(k, 2)
        _round_field("price_change_pct", 6)
        _round_field("confidence", 1)
        _round_field("eval_mae", 4)
        _round_field("eval_dir_rate", 2)

        return p

    def _emit(tag: str, payload: Any) -> None:
        """Emit a pretty-printed block for a model tag (PIPELINE/HEURISTIC/GPT/...)."""
        if payload is None:
            return
        if not isinstance(payload, dict):
            payload = {"value": payload}

        # Keep logs concise: omit large raw LLM output blobs from provider blocks.
        try:
            if tag in ("GPT", "GEMINI") and "raw" in payload:
                payload = dict(payload)
                payload.pop("raw", None)
        except Exception:
            pass

        payload = _normalize_payload(payload)
        sep = "=" * 70

        def _format_payload_lines(p: Dict[str, Any]) -> List[str]:
            try:
                import textwrap
            except Exception:
                textwrap = None  # type: ignore

            out_lines: List[str] = ["{"]
            width_default = 120
            width_long_text = 80
            for k, v in p.items():
                key = str(k)
                if isinstance(v, str):
                    s = v
                    if "\n" in s:
                        try:
                            s = " ".join([x for x in s.splitlines() if x is not None])
                        except Exception:
                            s = v

                    w = width_long_text if key.lower() in ("rationale", "caution") else width_default

                    if textwrap is not None and len(s) > w:
                        wrapped = textwrap.wrap(s, width=w)
                        if not wrapped:
                            wrapped = [""]
                        out_lines.append(f"  \"{key}\": {json.dumps(wrapped[0], ensure_ascii=False)}")
                        for w in wrapped[1:]:
                            out_lines.append(f"               {json.dumps(w, ensure_ascii=False)}")
                        out_lines[-1] = out_lines[-1] + ","
                    else:
                        out_lines.append(f"  \"{key}\": {json.dumps(s, ensure_ascii=False)},")
                    continue

                try:
                    out_lines.append(f"  \"{key}\": {json.dumps(v, ensure_ascii=False)},")
                except TypeError:
                    out_lines.append(f"  \"{key}\": {json.dumps(str(v), ensure_ascii=False)},")

            if len(out_lines) > 1:
                out_lines[-1] = out_lines[-1].rstrip(",")
            out_lines.append("}")
            return out_lines

        lines = [sep, f"[{tag}]", *_format_payload_lines(payload), sep]
        for line in lines:
            logger.info(line)
        logger.info("")

    try:
        if isinstance(result, dict):
            pipeline_payload = {
                "round_time": round_time,
                "current_price": result.get("current_price"),
                "ob_records_len": result.get("ob_records_len"),
                "fo0_age_sec": result.get("fo0_age_sec"),
                "prob": result.get("prob"),
                "signal": result.get("signal"),
                "confidence": result.get("confidence"),
                "llm_action": result.get("llm_action"),
                "llm_provider": result.get("llm_provider"),
                "llm_timed_out": result.get("llm_timed_out"),
                "risk_level": result.get("risk_level"),
                "rationale": result.get("rationale"),
                "caution": result.get("caution"),
                "llm_raw": result.get("llm_raw"),
                "consensus": result.get("consensus"),
                "guardrail_applied": result.get("guardrail_applied"),
                "guardrail_reason": result.get("guardrail_reason") or None,
                "eval_mae": result.get("eval_mae"),
                "eval_dir_hits": result.get("eval_dir_hits"),
                "eval_dir_total": result.get("eval_dir_total"),
                "eval_dir_rate": result.get("eval_dir_rate"),
            }
            try:
                if pipeline_payload.get("fo0_age_sec") is not None:
                    pipeline_payload["fo0_age_sec"] = round(float(pipeline_payload["fo0_age_sec"]), 2)
            except Exception:
                pass
            if any(v is not None for v in pipeline_payload.values()):
                _emit("PIPELINE", pipeline_payload)
    except Exception:
        pass

    heuristic_payload = model_outputs.get("heuristic")
    try:
        if not isinstance(heuristic_payload, dict):
            heuristic_payload = {
                "provider": "adaptive_indicator",
                "action": "HOLD",
                "is_ready": False,
                "reason": "missing_in_model_outputs (adaptive disabled or not ready)",
            }
    except Exception:
        heuristic_payload = model_outputs.get("heuristic")

    _emit("HEURISTIC", heuristic_payload)

    # In single-LLM mode (dual_llm=False), PredictionPipeline may not populate
    # model_outputs["gpt"|"gemini"]. In that case, synthesize a lightweight payload
    # from the flattened llm_* fields so runtime logs still show LLM decisions.
    gpt_payload = model_outputs.get("gpt")
    gemini_payload = model_outputs.get("gemini")
    try:
        if not isinstance(gpt_payload, dict) and not isinstance(gemini_payload, dict) and isinstance(result, dict):
            prov = str(result.get("llm_provider") or "").strip().lower()
            action = result.get("llm_action")
            if prov in ("gpt", "gemini") and action is not None:
                synthesized = {
                    "provider": prov,
                    "action": action,
                    "risk_level": result.get("risk_level"),
                    "rationale": result.get("rationale"),
                    "caution": result.get("caution"),
                    "timed_out": bool(result.get("llm_timed_out")) if result.get("llm_timed_out") is not None else None,
                }
                if prov == "gpt":
                    gpt_payload = synthesized
                else:
                    gemini_payload = synthesized
    except Exception:
        pass

    _emit("GPT", gpt_payload)
    _emit("GEMINI", gemini_payload)

    def _get_direction(payload: Any) -> Optional[str]:
        """Extract a normalized direction from a model payload.

        Note: LLM providers in this project typically return `action` (BUY/SELL/HOLD)
        rather than an explicit `direction` key.
        """
        if not isinstance(payload, dict):
            return None
        try:
            d = payload.get("direction")
            if d is None:
                d = payload.get("action")
            if d is None:
                return None

            s = str(d).strip().lower()
            if not s:
                return None
            if s in ("buy", "bull", "bullish", "long", "up", "uptrend"):
                return "buy"
            if s in ("sell", "bear", "bearish", "short", "down", "downtrend"):
                return "sell"
            if s in ("hold", "neutral", "flat"):
                return "hold"
            return s
        except Exception:
            return None

    h_dir = _get_direction(heuristic_payload)
    g_dir = _get_direction(gpt_payload)
    m_dir = _get_direction(gemini_payload)

    counts: Dict[str, int] = {}
    for d in (h_dir, g_dir, m_dir):
        if d is None:
            continue
        counts[d] = counts.get(d, 0) + 1

    consensus = None
    if counts:
        best_dir = None
        best_cnt = 0
        for k, v in counts.items():
            if v > best_cnt:
                best_dir = k
                best_cnt = v
            elif v == best_cnt:
                best_dir = None
        consensus = best_dir

    def _norm_dir(v: Optional[str]) -> str:
        try:
            s = str(v or "-").strip().lower()
        except Exception:
            s = "-"
        if s in ("buy", "sell", "hold"):
            return s.upper()
        if not s or s == "none":
            return "-"
        return s.upper()

    h_s = _norm_dir(h_dir)
    g_s = _norm_dir(g_dir)
    m_s = _norm_dir(m_dir)
    c_s = _norm_dir(consensus)
    try:
        votes = int(best_cnt) if consensus is not None else 0
    except Exception:
        votes = 0

    # Keep a single INFO line for easy grep; make it visually scannable.
    # Example:
    # [DIR_SUMMARY] H=SELL | GPT=SELL | GEM=SELL => CONS=SELL (votes=3/3)
    summary = (
        f"[DIR_SUMMARY] HEURISTIC={h_s:<4} | GPT={g_s:<4} | GEM={m_s:<4} "
        f"=> CONS={c_s:<4} (votes={votes}/3)"
    )
    try:
        logger.info("=" * 70)
    except Exception:
        pass
    logger.info(summary)
    try:
        logger.info("=" * 70)
    except Exception:
        pass


async def _run_prediction_loop(
    api: Any,
    predictor: Any,
    state: LiveState,
    *,
    kp200_symbol: str,
    duration_sec: int,
    option_month_info: Optional[str] = None,
    oi_itm_count: int = 10,
    oi_otm_count: int = 10,
    oc0_rebalance_sec: float = 60.0,
    t2301_refresh_sec: float = 60.0,
    test_now_injection_enabled: bool = False,
    test_now_fixed_dt: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Main live loop: periodically call predictor and track evaluation."""
    prediction_minutes    = int(getattr(predictor, "prediction_minutes",         5)  or 5)
    min_bars_required     = int(getattr(predictor, "min_minute_bars_required",   30) or 30)
    prediction_interval   = prediction_minutes * 60

    last_heuristic_action: Optional[str] = None
    # heuristic flip이 연속 발생 시 LLM 과호출 방지 — 최소 간격(초)
    _min_iv = getattr(predictor, "_heuristic_flip_min_interval_sec", None)
    try:
        if _min_iv is not None:
            _HEUR_FLIP_MIN_INTERVAL = max(30.0, float(_min_iv))
        else:
            _HEUR_FLIP_MIN_INTERVAL = max(60.0, float(prediction_minutes) * 30.0)
    except Exception:
        _HEUR_FLIP_MIN_INTERVAL = max(60.0, float(prediction_minutes) * 30.0)
    _heur_flip_include_hold = bool(getattr(predictor, "_heuristic_flip_include_hold_transition", False))
    last_heuristic_flip_ts: float = 0.0
    # ZigZag 스윙 확정(confirmation_bars 종료) — 휴리스틱 flip과 별도로 off-boundary 예측 트리거
    last_zz_confirm_log_key: Optional[Tuple[Any, ...]] = None  # [ZZ_CONFIRM_TRIGGER] 중복 방지(동일 확정 1회)
    last_zz_confirm_trigger_key: Optional[Tuple[Any, ...]] = None
    last_zz_confirm_flip_ts: float = 0.0

    def _handle_zz_confirm(
        *,
        af: Any,
        zz_state: Any,
        minute_df: Any,
        now_dt_ref: datetime,
        now_ts_ref: float,
    ) -> bool:
        """ZigZag 확정 로그/텔레그램/오프바운더리 트리거를 공통 처리."""
        nonlocal last_zz_confirm_log_key, last_zz_confirm_trigger_key, last_zz_confirm_flip_ts
        try:
            _azz = 0
            if isinstance(af, dict):
                _azz = int(float(af.get("azz_new_swing") or 0.0))
        except Exception:
            _azz = 0
        if _azz == 0:
            return False

        _lct = _last_complete_minute_ts_from_df(minute_df, now_dt_ref)
        _zkey = (_lct, int(_azz))

        if _zkey != last_zz_confirm_log_key:
            last_zz_confirm_log_key = _zkey
            try:
                _zz_dir = int(_azz)  # +1=고점, -1=저점
                _dir_kor = "고점(▼)" if _zz_dir > 0 else "저점(▲)"
                _pivot_px = 0.0
                _hi_px = 0.0
                _lo_px = 0.0
                _wave_pct = 0.0
                _struct = "unknown"
                try:
                    if zz_state is not None:
                        _hi_px = float(getattr(zz_state, "last_swing_high", 0.0) or 0.0)
                        _lo_px = float(getattr(zz_state, "last_swing_low", 0.0) or 0.0)
                        _wave_pct = float(getattr(zz_state, "wave_size_pct", 0.0) or 0.0)
                        _struct = str(getattr(zz_state, "structure", "unknown") or "unknown")
                        _pivot_px = _hi_px if _zz_dir > 0 else _lo_px
                except Exception:
                    pass
                _sym_label = str(get_pipeline_adaptive_indicator_symbol(predictor) or "")
                _log(
                    "[ZZ_CONFIRM] %s %s 피봇=%.2f  고=%.2f 저=%.2f  파동=%.2f%%  구조=%s  bar_ts=%s",
                    _sym_label,
                    _dir_kor,
                    _pivot_px,
                    _hi_px,
                    _lo_px,
                    _wave_pct,
                    _struct,
                    str(_lct),
                )
            except Exception:
                pass
            try:
                fn_msg = getattr(predictor, "_ebest_telegram_zz_confirm_message_hook", None)
                if callable(fn_msg):
                    fn_msg(
                        int(_azz),
                        _lct,
                        af if isinstance(af, dict) else None,
                        zz_state,
                    )
            except Exception:
                pass

        if _zkey != last_zz_confirm_trigger_key:
            now_ts_zz = float(now_ts_ref)
            if (now_ts_zz - float(last_zz_confirm_flip_ts)) >= float(_HEUR_FLIP_MIN_INTERVAL):
                last_zz_confirm_trigger_key = _zkey
                last_zz_confirm_flip_ts = now_ts_zz
                try:
                    fn = getattr(predictor, "_metrics_inc", None)
                    if callable(fn):
                        fn("zz_confirm_triggered")
                except Exception:
                    pass
                return True
            try:
                fn = getattr(predictor, "_metrics_inc", None)
                if callable(fn):
                    fn("zz_confirm_skipped_interval")
            except Exception:
                pass
            try:
                _log(
                    "[ZZ_CONFIRM_SKIP] too soon (%.0fs < min %.0fs)",
                    now_ts_zz - float(last_zz_confirm_flip_ts),
                    float(_HEUR_FLIP_MIN_INTERVAL),
                    level="debug",
                )
            except Exception:
                pass
        return False

    if int(duration_sec) <= 0:
        end_time = float("inf")
    else:
        end_time = time.monotonic() + max(0, int(duration_sec))
    next_prediction_ts    = 0.0
    next_heartbeat_ts     = 0.0
    next_oh0_refresh_ts   = 0.0
    next_oc0_rebalance_ts = 0.0   # OI 창 OC0 재구독 갱신 타이머
    next_t2301_refresh_ts = 0.0   # t2301 전 행사가 OI 갱신 타이머
    next_fc0_recover_ts   = 0.0   # FC0 stale 시 재구독 타이머
    fc0_recover_fail_streak = 0   # FC0/FH0 재구독 연속 실패 횟수
    fc0_recover_warn_after = 3    # n회 이상 연속 실패 시 WARNING 승격
    # 월물 코드 캐시 (t2301 갱신에 사용)
    try:
        _ym_raw = str(option_month_info or "").strip()
        if len(_ym_raw) == 4 and _ym_raw.isdigit():
            _t2301_yyyymm: Optional[str] = "20" + _ym_raw
        elif len(_ym_raw) == 6:
            _t2301_yyyymm = _ym_raw
        else:
            _t2301_yyyymm = str(get_option_month_yyyymm(datetime.now()))
        _t2301_gubun: str = "W" if _ym_raw.upper().startswith("W") else "G"
    except Exception:
        _t2301_yyyymm = None
        _t2301_gubun = "G"
    try:
        _oi_itm_count_eff = int(oi_itm_count)
    except Exception:
        _oi_itm_count_eff = 10
    if _oi_itm_count_eff < 1:
        _oi_itm_count_eff = 1
    if _oi_itm_count_eff > 30:
        _oi_itm_count_eff = 30

    try:
        _oi_otm_count_eff = int(oi_otm_count)
    except Exception:
        _oi_otm_count_eff = 10
    if _oi_otm_count_eff < 1:
        _oi_otm_count_eff = 1
    if _oi_otm_count_eff > 30:
        _oi_otm_count_eff = 30

    _oc0_rebalance_sec_eff = max(10.0, min(600.0, float(oc0_rebalance_sec or 60.0)))
    _t2301_refresh_sec_eff = max(30.0, float(t2301_refresh_sec or 60.0))
    wait_start            = time.monotonic()
    next_wait_log_sec     = 0.0
    ready_logged          = False

    while time.monotonic() < end_time:
        now_ts = time.monotonic()

        off_boundary_trigger = False

        try:
            if bool(state.stop_requested):
                reason = state.stop_reason
                _log("[STOP] requested: %s", str(reason or ""))
                break
        except Exception:
            pass

        # ── FC0 stale 자동 복구(재구독) ─────────────────────────
        # FC0 체결 틱이 stale 임계값을 넘기면 FC0/FH0를 재구독해 피드 복구를 시도한다.
        try:
            if now_ts >= next_fc0_recover_ts:
                tp = getattr(predictor, "tick_processor", None)
                market_closed = bool(getattr(tp, "market_closed", False)) if tp is not None else False
                if not market_closed:
                    stale_thr = float(getattr(predictor, "_fc0_stale_threshold_sec", 10.0) or 10.0)
                    recover_cd = float(getattr(predictor, "_fc0_stale_cooldown_sec", 60.0) or 60.0)
                    recover_cd = max(30.0, recover_cd)
                    last_fc0 = getattr(predictor, "_last_fc0_seen_epoch", None)
                    if last_fc0 is not None:
                        age_sec = float(time.time()) - float(last_fc0)
                        if age_sec > stale_thr:
                            _log(
                                "[FC0_RECOVER] stale detected age_sec=%.1f > %.1f — re-subscribe FC0/FH0",
                                float(age_sec),
                                float(stale_thr),
                            )
                            _recover_errors = []
                            try:
                                await _ebest_register_realtime(
                                    api,
                                    trcode=TRCode.FUTURES.value,
                                    symbol=kp200_symbol,
                                )
                            except Exception as _e_fc0:
                                _recover_errors.append(f"FC0:{str(_e_fc0)}")
                            try:
                                await _ebest_register_realtime(
                                    api,
                                    trcode=TRCode.FUTURES_BOOK.value,
                                    symbol=kp200_symbol,
                                )
                            except Exception as _e_fh0:
                                _recover_errors.append(f"FH0:{str(_e_fh0)}")

                            if _recover_errors:
                                fc0_recover_fail_streak = int(fc0_recover_fail_streak) + 1
                                _lv = "warning" if int(fc0_recover_fail_streak) >= int(fc0_recover_warn_after) else "info"
                                _log(
                                    "[FC0_RECOVER] 재구독 실패 streak=%d/%d errs=%s",
                                    int(fc0_recover_fail_streak),
                                    int(fc0_recover_warn_after),
                                    " | ".join(_recover_errors),
                                    level=_lv,
                                )
                            else:
                                if int(fc0_recover_fail_streak) > 0:
                                    _log(
                                        "[FC0_RECOVER] 재구독 복구 성공 (fail_streak_reset from %d)",
                                        int(fc0_recover_fail_streak),
                                    )
                                fc0_recover_fail_streak = 0
                            next_fc0_recover_ts = now_ts + recover_cd
        except Exception:
            pass

        # ── OH0 (ATM±2) 동적 구독 갱신 ─────────────────────────
        if now_ts >= next_oh0_refresh_ts and bool(state.option_calls_all) and bool(state.option_puts_all):
            try:
                upx = float(predictor.tick_processor.get_current_price() or 0.0)
            except Exception:
                upx = 0.0

            if upx > 0.0:
                try:
                    new_calls, new_puts, new_atm = _filter_option_symbols_by_atm(
                        predictor,
                        calls=list(state.option_calls_all or []),
                        puts=list(state.option_puts_all or []),
                        itm_count=2,
                        otm_count_call=2,
                        otm_count_put=2,
                        underlying_price=upx,
                    )
                    desired = set([str(x) for x in (new_calls or []) + (new_puts or []) if x])
                except Exception:
                    desired = set()
                    new_atm = None

                try:
                    already = set([str(x) for x in (state.subscribed_oh0 or set()) if x])
                except Exception:
                    already = set()

                new_syms = sorted(list(desired - already))
                if new_syms:
                    atm_str = f" ATM={_fmt_atm_strike(new_atm)}" if new_atm is not None else ""
                    _log("[OH0_REFRESH] add=%d total=%d upx=%.2f%s", int(len(new_syms)), int(len(already) + len(new_syms)), float(upx), atm_str)
                    for sym in new_syms:
                        _log("[OH0_REFRESH] subscribe OH0 %s", str(sym))
                        try:
                            state.subscribed_oh0.add(str(sym))
                        except Exception:
                            pass
                        try:
                            await _ebest_register_realtime(api, trcode=TRCode.OPTIONS_QUOTE.value, symbol=str(sym))
                        except Exception:
                            pass
                        await asyncio.sleep(0.05)

            next_oh0_refresh_ts = now_ts + 60.0

        # ── OC0 (OI 창) 장중 재구독 갱신 ────────────────────────
        # ATM이 이동하면 기존 구독 목록이 새 ATM 기준 ±N 범위를 벗어난다.
        # oi_rebalance_interval_sec(기본 60초) 주기로 ATM 기준 내가/외가 목록을 재계산하여
        # 누락된 심볼만 추가 구독한다. 이미 구독 중인 심볼은 재구독하지 않는다.
        # (eBest API는 구독 해제 없이 추가만 가능하므로 누적 방식으로 관리)
        if (
            now_ts >= next_oc0_rebalance_ts
            and bool(state.option_calls_all)
            and bool(state.option_puts_all)
            and bool(state.open_oc0_subscribed)   # 장 개시 후 최초 구독 완료 이후에만 실행
        ):
            try:
                upx_reb = float(predictor.tick_processor.get_current_price() or 0.0)
            except Exception:
                upx_reb = 0.0

            if upx_reb > 0.0:
                try:
                    reb_calls, reb_puts, reb_atm = select_oi_window_symbols(
                        calls=list(state.option_calls_all or []),
                        puts=list(state.option_puts_all or []),
                        underlying_price=float(upx_reb),
                        itm_count=int(_oi_itm_count_eff),
                        otm_count=int(_oi_otm_count_eff),
                    )
                    desired_reb = set(
                        str(x) for x in (list(reb_calls or []) + list(reb_puts or [])) if x
                    )
                except Exception as _e:
                    desired_reb = set()
                    reb_atm = None
                    _log("[OC0_REBAL] select_oi_window_symbols 실패: %s", str(_e))

                try:
                    already_oc0 = set(str(x) for x in (state.subscribed_oc0 or set()) if x)
                except Exception:
                    already_oc0 = set()

                missing_reb = sorted(desired_reb - already_oc0)
                if missing_reb:
                    atm_reb_str = f" ATM={reb_atm:.2f}" if reb_atm is not None else ""
                    _log(
                        "[OC0_REBAL] ATM 이동 감지 — 추가구독 %d개 upx=%.2f%s",
                        int(len(missing_reb)),
                        float(upx_reb),
                        atm_reb_str,
                    )
                    added_reb = 0
                    for sym in missing_reb:
                        try:
                            await _ebest_register_realtime(
                                api, trcode=TRCode.OPTIONS.value, symbol=str(sym)
                            )
                            added_reb += 1
                            try:
                                state.subscribed_oc0.add(str(sym))
                            except Exception:
                                pass
                        except Exception as _se:
                            _log("[OC0_REBAL] 구독 실패 %s: %s", str(sym), str(_se))
                        await asyncio.sleep(0.05)
                    _log("[OC0_REBAL] 추가완료 %d개 / 전체 OC0 구독 %d개",
                         int(added_reb), int(len(already_oc0) + added_reb))

                    # OH0도 동일 범위로 갱신 (호가/IV 데이터 확보)
                    try:
                        already_oh0 = set(str(x) for x in (state.subscribed_oh0 or set()) if x)
                        missing_oh0 = sorted(desired_reb - already_oh0)
                        for sym in missing_oh0:
                            try:
                                await _ebest_register_realtime(
                                    api, trcode=TRCode.OPTIONS_QUOTE.value, symbol=str(sym)
                                )
                                state.subscribed_oh0.add(str(sym))
                            except Exception:
                                pass
                            await asyncio.sleep(0.05)
                        if missing_oh0:
                            _log("[OC0_REBAL] OH0 추가 %d개", int(len(missing_oh0)))
                    except Exception:
                        pass

            next_oc0_rebalance_ts = now_ts + float(_oc0_rebalance_sec_eff)

        # ── t2301 전 행사가 OI 주기 갱신 ──────────────────────────
        # OC0 실시간 구독은 ATM ±N개 범위만 커버하므로, t2301 REST 조회로
        # 전 행사가 OI를 주기적으로 갱신하여 ATM 이동과 무관하게 OI 분포를 보장한다.
        # - 갱신 주기: t2301_refresh_sec (기본 60초)
        # - 장 개시(open_oc0_subscribed) 이후에만 실행
        # - 월물 코드(_t2301_yyyymm)가 없으면 건너뜀
        if (
            now_ts >= next_t2301_refresh_ts
            and bool(state.open_oc0_subscribed)
            and _t2301_yyyymm is not None
        ):
            try:
                _t2301_snap = await _ebest_fetch_t2301_snapshot(
                    api,
                    yyyymm=str(_t2301_yyyymm),
                    gubun=str(_t2301_gubun),
                )
                if isinstance(_t2301_snap, dict) and (
                    _t2301_snap.get("oi_calls") or _t2301_snap.get("oi_puts")
                ):
                    try:
                        _n_updated = predictor.tick_processor.update_oi_from_t2301(_t2301_snap)
                    except Exception as _ue:
                        _n_updated = 0
                        _log("[T2301_REFRESH] update_oi_from_t2301 실패: %s", str(_ue))

                    # pipeline의 _t2301_snapshot도 최신 값으로 교체
                    try:
                        setter = getattr(predictor, "set_market_snapshots", None)
                        if callable(setter):
                            setter(t2301=_t2301_snap)
                    except Exception:
                        pass

                    _log(
                        "[T2301_REFRESH] OI 갱신 완료: 콜%d개 풋%d개 → tick_processor %d건 갱신",
                        int(len(_t2301_snap.get("oi_calls") or [])),
                        int(len(_t2301_snap.get("oi_puts") or [])),
                        int(_n_updated),
                    )
                else:
                    _log("[T2301_REFRESH] 응답 없음 또는 OI 데이터 없음 (yyyymm=%s gubun=%s)",
                         str(_t2301_yyyymm), str(_t2301_gubun))
            except Exception as _te:
                _log("[T2301_REFRESH] t2301 조회 실패: %s", str(_te))

            next_t2301_refresh_ts = now_ts + float(_t2301_refresh_sec_eff)

        try:
            df        = _ebest_adaptive_minute_df(predictor, None)
            bar_count = len(df) if df is not None else 0
        except Exception:
            df        = None
            bar_count = 0

        # ── 평가 ──────────────────────────────
        if state.pending_evals and df is not None and bar_count > 0:
            _try_evaluate_pending(state, predictor, df)

        # ── 데이터 수집 대기 ───────────────────
        if bar_count < min_bars_required:
            waited = now_ts - wait_start
            if waited >= next_wait_log_sec:
                ft = ct = pt = 0
                try:
                    ft = len(getattr(predictor.tick_processor, "futures_ticks", []) or [])
                    ct = getattr(predictor.tick_processor, "call_option_ticks", 0) or 0
                    pt = getattr(predictor.tick_processor, "put_option_ticks",  0) or 0
                except Exception:
                    pass
                # If no realtime ticks are arriving at all, suppress the minute-bar WAIT log.
                # In this case, other warnings (e.g., FC0_STALE) are more actionable.
                if int(ft) <= 0 and int(ct) <= 0 and int(pt) <= 0:
                    next_wait_log_sec = waited + 60.0
                    await asyncio.sleep(1)
                    continue
                remaining = max(0, int(min_bars_required) - int(bar_count))
                eta_sec = float(remaining) * 60.0
                try:
                    w = float(waited)
                    b = float(bar_count)
                    if w > 5.0 and b > 0.0:
                        rate = b / w
                        if rate > 0:
                            eta_sec = float(remaining) / rate
                except Exception:
                    eta_sec = float(remaining) * 60.0
                try:
                    mo = bool(state.market_opened)
                except Exception:
                    mo = False
                _log(
                    "[WAIT] collecting minute bars: %d/%d remaining=%d waited=%.0fs eta=%.1fmin market_opened=%s futures_ticks=%d call_ticks=%d put_ticks=%d",
                    int(bar_count),
                    int(min_bars_required),
                    int(remaining),
                    float(waited),
                    float(eta_sec) / 60.0,
                    str(mo),
                    int(ft),
                    int(ct),
                    int(pt),
                )
                next_wait_log_sec = waited + 60.0
            await asyncio.sleep(1)
            continue

        # ── 하트비트 ──────────────────────────
        if now_ts >= next_heartbeat_ts:
            ft = ct = pt = 0
            try:
                ft = len(getattr(predictor.tick_processor, "futures_ticks", []) or [])
                ct = getattr(predictor.tick_processor, "call_option_ticks", 0) or 0
                pt = getattr(predictor.tick_processor, "put_option_ticks",  0) or 0
            except Exception:
                pass
            sec_to_next = max(0.0, next_prediction_ts - now_ts)
            try:
                mo = bool(state.market_opened)
            except Exception:
                mo = False
            _log(
                "[HB] bars=%d/%d(required=%d) market_opened=%s futures_ticks=%d call_ticks=%d put_ticks=%d next_predict_in=%.1fs",
                bar_count, bar_count, min_bars_required, str(mo), ft, ct, pt, sec_to_next,
                level="debug",
            )
            next_heartbeat_ts = now_ts + 60.0

        if not ready_logged:
            ready_logged = True
            _log("[READY] minute bars reached: %d/%d", bar_count, min_bars_required)

        # ── Heuristic flip 트리거 (BUY<->SELL) ─────────────────
        # prediction_minutes 경계가 아니어도 Heuristic이 BUY↔SELL로 바뀌면
        # 즉시 LLM 포함 예측(get_prediction) 실행을 트리거합니다.
        try:
            if now_ts < next_prediction_ts:
                now_dt2 = test_now_fixed_dt or datetime.fromtimestamp(float(now_ts))
                try:
                    warmup_bars = int(getattr(predictor, "_adaptive_warmup_bars", 45) or 45)
                except Exception:
                    warmup_bars = 45
                df_h = _ebest_adaptive_minute_df(predictor, int(warmup_bars))
                if df_h is not None and len(df_h) >= int(min_bars_required):
                    try:
                        _af, _ac, _st, _zz, mo = predictor._compute_adaptive_bundle(df=df_h, now_dt=now_dt2)
                    except Exception:
                        _af = None
                        mo = {}
                    heur = None
                    try:
                        heur = (mo or {}).get("heuristic") if isinstance(mo, dict) else None
                    except Exception:
                        heur = None
                    cur_a = ""
                    try:
                        if isinstance(heur, dict):
                            cur_a = str(heur.get("action") or "").strip().upper()
                    except Exception:
                        cur_a = ""

                    prev_a = str(last_heuristic_action or "").strip().upper()
                    if cur_a in ("BUY", "SELL", "HOLD"):
                        _flip = False
                        if prev_a in ("BUY", "SELL") and cur_a in ("BUY", "SELL") and cur_a != prev_a:
                            _flip = True
                        elif _heur_flip_include_hold:
                            if prev_a == "HOLD" and cur_a in ("BUY", "SELL"):
                                _flip = True
                            elif cur_a == "HOLD" and prev_a in ("BUY", "SELL"):
                                _flip = True
                        if _flip:
                            try:
                                _log(
                                    "[HEUR_FLIP_TRIGGER] %s -> %s (off-boundary)",
                                    str(prev_a),
                                    str(cur_a),
                                )
                            except Exception:
                                pass
                            try:
                                fn = getattr(predictor, "_metrics_inc", None)
                                if callable(fn):
                                    fn("heur_flip_triggered")
                            except Exception:
                                pass
                            # 최소 간격 체크: 너무 잦은 off-boundary 호출 방지
                            now_ts_flip = float(now_ts)
                            if (now_ts_flip - float(last_heuristic_flip_ts)) >= float(_HEUR_FLIP_MIN_INTERVAL):
                                next_prediction_ts = now_ts_flip
                                off_boundary_trigger = True
                                last_heuristic_flip_ts = now_ts_flip
                            else:
                                try:
                                    fn = getattr(predictor, "_metrics_inc", None)
                                    if callable(fn):
                                        fn("heur_flip_skipped_interval")
                                except Exception:
                                    pass
                                _log(
                                    "[HEUR_FLIP_SKIP] too soon (%.0fs < min %.0fs)",
                                    now_ts_flip - float(last_heuristic_flip_ts),
                                    float(_HEUR_FLIP_MIN_INTERVAL),
                                    level="debug",
                                )
                        if cur_a in ("BUY", "SELL"):
                            last_heuristic_action = cur_a
                        elif _heur_flip_include_hold and cur_a == "HOLD":
                            last_heuristic_action = cur_a

                    # ZigZag 스윙 확정 — 로그/텔레그램/트리거 공통 처리
                    if _handle_zz_confirm(
                        af=_af,
                        zz_state=_zz,
                        minute_df=df_h,
                        now_dt_ref=now_dt2,
                        now_ts_ref=float(now_ts),
                    ):
                        next_prediction_ts = float(now_ts)
                        off_boundary_trigger = True
        except Exception:
            pass

        # ── 예측 ──────────────────────────────
        if now_ts >= next_prediction_ts:
            try:
                # Refresh IJ spot snapshot before prediction (best-effort; 60s cadence)
                try:
                    last_ij = float(state.last_ij_refresh_epoch) if state.last_ij_refresh_epoch is not None else None
                except Exception:
                    last_ij = None

                if last_ij is None or (float(now_ts) - float(last_ij)) >= 60.0:
                    try:
                        _ij_key = _ebest_ij_tr_key(predictor)
                        ij_snap = await _ebest_fetch_ij_snapshot(api, tr_key=str(_ij_key))
                        if isinstance(ij_snap, dict) and ij_snap:
                            try:
                                setter = getattr(predictor, "set_market_snapshots", None)
                                if callable(setter):
                                    setter(ij_=ij_snap)
                            except Exception:
                                pass
                            try:
                                update_gui_spot_index(spot_index=ij_snap.get("jisu"), spot_time=ij_snap.get("time"))
                            except Exception:
                                pass
                            try:
                                with state._lock:
                                    state.last_ij_refresh_epoch = float(now_ts)
                            except Exception:
                                try:
                                    with state._lock:
                                        state.last_ij_refresh_epoch = None
                                except Exception:
                                    pass
                            try:
                                _log(
                                    "[IJ_REFRESH] tr_key=%s jisu=%.2f time=%s",
                                    str(_ij_key),
                                    float(ij_snap.get("jisu") or 0.0),
                                    str(ij_snap.get("time") or ""),
                                )
                            except Exception:
                                _log("[IJ_REFRESH] tr_key=%s", str(_ij_key))
                    except Exception:
                        pass

                # t2101(선물 당일 시가) 주기적 갱신 — 60초 cadence, IJ와 동일 주기.
                # 장 초기(08:45~09:00)에 t2101 open이 아직 확정되지 않을 수 있으므로
                # 주기적으로 재조회해 session_open을 최신 당일 시가로 유지한다.
                try:
                    last_t2101 = float(state.last_t2101_refresh_epoch) if state.last_t2101_refresh_epoch is not None else None
                except Exception:
                    last_t2101 = None

                if last_t2101 is None or (float(now_ts) - float(last_t2101)) >= 60.0:
                    try:
                        _t2101_refresh = await _ebest_fetch_t2101_snapshot(api, focode=kp200_symbol)
                        if isinstance(_t2101_refresh, dict) and float(_t2101_refresh.get("open") or 0.0) > 0.0:
                            try:
                                setter = getattr(predictor, "set_market_snapshots", None)
                                if callable(setter):
                                    setter(t2101=_t2101_refresh)
                            except Exception:
                                pass
                            try:
                                with state._lock:
                                    state.last_t2101_refresh_epoch = float(now_ts)
                            except Exception:
                                pass
                            try:
                                _log(
                                    "[T2101_REFRESH] open=%.2f high=%.2f low=%.2f price=%.2f focode=%s",
                                    float(_t2101_refresh.get("open") or 0.0),
                                    float(_t2101_refresh.get("high") or 0.0),
                                    float(_t2101_refresh.get("low") or 0.0),
                                    float(_t2101_refresh.get("price") or 0.0),
                                    str(_t2101_refresh.get("focode") or kp200_symbol),
                                )
                            except Exception:
                                _log("[T2101_REFRESH] focode=%s", kp200_symbol)
                    except Exception:
                        pass

                # 경계 시점(now_ts >= next_prediction_ts)에도 ZigZag 확정을 먼저 점검해
                # 텔레그램/로그 누락을 방지한다.
                try:
                    _now_dt_pred = test_now_fixed_dt or datetime.fromtimestamp(float(now_ts))
                    if df is not None and len(df) >= int(min_bars_required):
                        try:
                            _af_b, _ac_b, _st_b, _zz_b, _mo_b = predictor._compute_adaptive_bundle(
                                df=df, now_dt=_now_dt_pred
                            )
                        except Exception:
                            _af_b = None
                            _zz_b = None
                        _ = _handle_zz_confirm(
                            af=_af_b,
                            zz_state=_zz_b,
                            minute_df=df,
                            now_dt_ref=_now_dt_pred,
                            now_ts_ref=float(now_ts),
                        )
                except Exception:
                    pass

                if bool(test_now_injection_enabled):
                    dt = test_now_fixed_dt or datetime.fromtimestamp(float(now_ts))
                    res = predictor.get_prediction(auto_mode=True, _now=dt, off_boundary=bool(off_boundary_trigger))
                else:
                    res = predictor.get_prediction(auto_mode=True, off_boundary=bool(off_boundary_trigger))
            except Exception as e:
                res = {"error": f"prediction_failed: {e}"}

            try:
                with state._lock:
                    state.prediction_count += 1
                    pred_seq = int(state.prediction_count)
            except Exception:
                pred_seq = int(state.prediction_count or 0) + 1
                try:
                    state.prediction_count = int(pred_seq)
                except Exception:
                    pass

            result = dict(res) if isinstance(res, dict) else {"result": res}
            result["prediction_seq"] = int(pred_seq)
            result["minute_bars"]    = bar_count

            try:
                mo = result.get("model_outputs")
                heur = (mo or {}).get("heuristic") if isinstance(mo, dict) else None
                if isinstance(heur, dict):
                    a = str(heur.get("action") or "").strip().upper()
                    if a:
                        last_heuristic_action = a
            except Exception:
                pass

            # pending_eval 등록
            if (
                "error" not in result
                and "current_price" in result
                and df is not None
                and len(df) > 0
                and ("predicted_price" in result or "signal" in result)
            ):
                try:
                    base_ts = pd.to_datetime(df.iloc[-1].get("timestamp"))
                    target_dt = base_ts + timedelta(minutes=prediction_minutes)

                    payload: Dict[str, Any] = {
                        "prediction_seq": int(pred_seq),
                        "base_price": float(result["current_price"]),
                        "base_timestamp": base_ts.isoformat(),
                        "target_timestamp": target_dt.isoformat(),
                    }

                    try:
                        if result.get("transformer_prob") is not None:
                            payload["transformer_prob"] = float(result.get("transformer_prob"))
                        if result.get("tft_prob") is not None:
                            payload["tft_prob"] = float(result.get("tft_prob"))
                    except Exception:
                        pass

                    if "predicted_price" in result and result.get("predicted_price") is not None:
                        payload["predicted_price"] = float(result["predicted_price"])
                    if "signal" in result and result.get("signal") is not None:
                        payload["signal"] = str(result.get("signal"))

                    try:
                        with state._lock:
                            state.pending_evals.append(payload)
                    except Exception:
                        state.pending_evals.append(payload)
                except Exception as e:
                    logger.warning("[EVAL] failed to register pending eval: %s", e)

            try:
                with state._lock:
                    result["ebest_tick_counts"] = dict(state.tick_counts)
                    kp200_prev_close = state.kp200_prev_close
            except Exception:
                result["ebest_tick_counts"] = dict(state.tick_counts or {})
                kp200_prev_close = state.kp200_prev_close
            result["ebest_kp200_symbol"] = kp200_symbol
            result["ebest_kp200_prev_close"] = kp200_prev_close

            _append_eval_metrics(result, state)

            _log("[PREDICT] seq=%d bars=%d", int(pred_seq), bar_count)
            _log_model_outputs(result)
            _log("[SCHEDULE] next_predict_in=%.1fs", float(prediction_interval))

            try:
                with state._lock:
                    state.last_result = result
            except Exception:
                state.last_result = result
            next_prediction_ts   = now_ts + prediction_interval

        await asyncio.sleep(1)

    try:
        with state._lock:
            return state.last_result
    except Exception:
        return state.last_result


# ──────────────────────────────────────────────
# 공개 진입점
# ──────────────────────────────────────────────
async def run_ebest_live_mode(
    predictor: Any,
    duration_sec: int,
    include_options: bool,
    option_month_info: Optional[str],
    config_path: str,
    opt_itm: int,
    opt_wait_sec: int,
    out_ticks: Optional[str],
    save_ticks_enabled: bool,
) -> Dict[str, Any]:
    """eBest 실시간 모드 실행."""
    try:
        import ebest  # type: ignore
    except Exception as e:
        return {"error": f"ImportError: {e}"}

    api      = ebest.OpenApi()
    state    = LiveState()
    ticks_fh = None

    prediction_minutes = int(getattr(predictor, "prediction_minutes",       5)  or 5)
    min_bars_required  = int(getattr(predictor, "min_minute_bars_required", 30) or 30)
    _log(
        "[EBEST_LIVE] file=%s prediction_minutes=%d min_minute_bars_required=%d",
        __file__, prediction_minutes, min_bars_required,
    )

    try:
        cfg       = _load_config(str(config_path or "config.json"))
        test_now_injection_enabled = bool(cfg.get("test_now_injection_enabled") or False)
        test_now_fixed_iso = str(cfg.get("test_now_fixed_iso") or "").strip()
        test_now_fixed_dt: Optional[datetime] = None
        if test_now_injection_enabled and test_now_fixed_iso:
            try:
                test_now_fixed_dt = datetime.fromisoformat(test_now_fixed_iso)
            except Exception as e:
                logger.warning("[CONFIG] test_now_fixed_iso parse failed: %s", e)

        try:
            update_meaningful_option_levels(cfg.get("meaningful_option_levels"))
        except Exception:
            pass

        # 텔레그램 notifier 설정 (의미가 옵션 알림) — secrets 파일 경로 전달 (dict 아님)
        try:
            from telegram.notifier import create_notifier_from_config

            _cfg_p = os.path.abspath(str(config_path or "config.json"))
            _secrets = os.environ.get("APP_SECRETS_CONFIG") or os.path.join(
                os.path.dirname(_cfg_p) or ".", "config.secrets.json"
            )
            notifier = create_notifier_from_config(str(_secrets))
            set_meaningful_option_telegram_notifier(notifier)
        except Exception as e:
            logger.warning(f"[CONFIG] 텔레그램 notifier 설정 실패: {e}")
            set_meaningful_option_telegram_notifier(None)

        try:
            opt_cfg = cfg.get("options_subscription") if isinstance(cfg.get("options_subscription"), dict) else {}
            oi_itm_count = int(opt_cfg.get("oi_itm_count", 10) or 10)
            oi_otm_count = int(opt_cfg.get("oi_otm_count", 10) or 10)
            oi_rebalance_interval_sec = float(opt_cfg.get("oi_rebalance_interval_sec", 60.0) or 60.0)
        except Exception:
            oi_itm_count = 10
            oi_otm_count = 10
            oi_rebalance_interval_sec = 60.0
        if oi_itm_count < 1:
            oi_itm_count = 1
        if oi_itm_count > 30:
            oi_itm_count = 30
        if oi_otm_count < 1:
            oi_otm_count = 1
        if oi_otm_count > 30:
            oi_otm_count = 30
        if oi_rebalance_interval_sec < 10.0:
            oi_rebalance_interval_sec = 10.0
        if oi_rebalance_interval_sec > 600.0:
            oi_rebalance_interval_sec = 600.0
    except Exception as e:
        logger.warning("[CONFIG] option subscribe settings parse failed: %s", e)
        test_now_injection_enabled = False
        test_now_fixed_dt = None
        oi_itm_count = 10
        oi_otm_count = 10
        oi_rebalance_interval_sec = 60.0

    try:
        ticks_fh, kp200_symbol = await _initialize_api(
            api, predictor, state,
            config_path=str(config_path or "config.json"),
            include_options=include_options,
            option_month_info=option_month_info,
            opt_itm=opt_itm,
            opt_wait_sec=opt_wait_sec,
            out_ticks=out_ticks,
            save_ticks_enabled=bool(save_ticks_enabled),
        )
    except (ValueError, ConnectionError, RuntimeError) as e:
        return {"error": str(e)}

    try:
        oi_reb_eff = float(oi_rebalance_interval_sec)
    except Exception:
        oi_reb_eff = 60.0
    if oi_reb_eff < 10.0:
        oi_reb_eff = 10.0
    if oi_reb_eff > 600.0:
        oi_reb_eff = 600.0

    try:
        return await _run_prediction_loop(
            api, predictor, state,
            kp200_symbol=kp200_symbol,
            duration_sec=duration_sec,
            option_month_info=option_month_info,
            oi_itm_count=int(oi_itm_count),
            oi_otm_count=int(oi_otm_count),
            oc0_rebalance_sec=float(oi_reb_eff),
            t2301_refresh_sec=float(oi_reb_eff),
        )
    except Exception as e:
        return {"error": str(e)}
    finally:
        if ticks_fh is not None:
            try:
                ticks_fh.close()
            except Exception:
                pass
        # NW-ARC-04: PredictionPipeline 리소스 해제 (ThreadPoolExecutor, LLM HTTP 세션)
        try:
            if predictor is not None and hasattr(predictor, "close"):
                predictor.close()
                _log("[LIVE] PredictionPipeline closed")
        except Exception as e:
            _log("[LIVE] predictor.close() 실패 (무시): %s", str(e), level="warning")
