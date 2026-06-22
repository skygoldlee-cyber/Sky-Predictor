"""텔레그램 예측 결과 송수신 모듈 (SkyEbest / Transformer Pipeline 연동)

사용법:
    from .notifier import create_notifier_from_config, PipelineTelegramBridge

    notifier = create_notifier_from_config()   # secrets 자동 로드
    bridge = PipelineTelegramBridge(pipeline, notifier)
    bridge.start()           # 주기적 예측 루프
    bridge.start_polling()   # 명령 수신

설정:
    - config.secrets.json: { "telegram": { "bot_token": "...", "chat_id": "..." } }
    - 환경변수: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    - secrets 경로 오버라이드: APP_SECRETS_CONFIG 환경변수

의존성:
    표준 라이브러리(urllib)만 사용 — 추가 설치 불필요
"""

from __future__ import annotations

import json
import logging
import os
import re
import html
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import urllib.parse
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)
try:
    # Inherit effective level from root so INFO logs (e.g., [TG][SEND]) are not suppressed.
    logger.setLevel(logging.NOTSET)
except Exception:
    pass

try:
    _TG_DEBUG = str(os.environ.get("TELEGRAM_DEBUG") or "").strip().lower() not in ("", "0", "false", "no")
except Exception:
    _TG_DEBUG = False

# trade_gate 는 순환 임포트 방지를 위해 런타임에 지연 로드
try:
    from trading.gate import TradeExecutionGate, TradeGateConfig as _TradeGateConfig
    _TRADE_GATE_AVAILABLE = True
except ImportError:
    TradeExecutionGate = None  # type: ignore[assignment,misc]
    _TradeGateConfig = None    # type: ignore[assignment]
    _TRADE_GATE_AVAILABLE = False

# ── 포매터는 telegram_formatters.py에서 분리 관리 ────────────────────────────
# SRP 원칙: 순수 함수(포매터)와 I/O 클래스(Notifier/Bridge)를 분리한다.
# 하위 호환을 위해 이 모듈에서 re-export한다.
from .formatters import (  # noqa: E402
    _esc_mdv2,
    format_prediction_message,
    format_premium_bleed_alert,
    format_futures_call_divergence_alert,
    format_price_level_touch_alert,
    format_error_message,
)

__all__ = [
    "format_prediction_message",
    "format_premium_bleed_alert",
    "format_futures_call_divergence_alert",
    "format_price_level_touch_alert",
    "format_error_message",
    "TelegramNotifier",
    "PipelineTelegramBridge",
    "create_notifier_from_config",
]


class TelegramNotifier:
    """텔레그램 봇을 통해 예측 결과를 전송하고, 명령을 수신하는 클래스.

    Args:
        bot_token: 텔레그램 봇 토큰. None이면 환경변수 TELEGRAM_BOT_TOKEN 사용.
        chat_id: 전송 대상 채팅 ID. None이면 환경변수 TELEGRAM_CHAT_ID 사용.
        min_interval_sec: [Deprecated] 더 이상 사용되지 않음. 신호 변경 기반 필터로 대체.
        only_actionable: True면 HOLD 신호는 전송 생략. 기본 False.
        timeout: HTTP 타임아웃 (초). 기본 10초.
    """

    BASE_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        min_interval_sec: float = 60.0,   # Deprecated: 더 이상 사용되지 않음
        only_actionable: bool = False,
        timeout: float = 30.0,
        proxy_url: Optional[str] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = str(chat_id or os.environ.get("TELEGRAM_CHAT_ID", ""))
        self._only_actionable = bool(only_actionable)
        self._timeout = float(timeout)
        self._now_fn = now_fn if now_fn is not None else datetime.now

        # 프록시 설정: proxy_url 지정 시 해당 프록시를 통해 요청
        # 예) "http://127.0.0.1:7890"  또는  "socks5://127.0.0.1:1080"
        # 미지정 시 환경변수 HTTPS_PROXY / HTTP_PROXY 자동 적용 (urllib 기본 동작)
        _proxy_url = proxy_url or os.environ.get("TELEGRAM_PROXY_URL", "")
        if _proxy_url:
            _proxies = {"http": _proxy_url, "https": _proxy_url}
            self._opener: Optional[urllib.request.OpenerDirector] = (
                urllib.request.build_opener(urllib.request.ProxyHandler(_proxies))
            )
        else:
            self._opener = None

        self._last_signal: str = ""
        # _last_signal은 _predict_loop 스레드와 /reset 명령 처리 스레드에서 동시에
        # 읽고 쓰므로 Lock으로 직렬화한다. (3-1 수정)
        self._signal_lock = threading.Lock()

        # 프리미엄 블리드 알림 중복 방지 쿨다운 (기본 5분)
        self._last_bleed_alert_epoch: float = 0.0
        self._bleed_alert_cooldown_sec: float = 300.0

        # OI 구조 알림 중복 방지 쿨다운 (기본 10분)
        self._last_oi_alert_epoch: float = 0.0
        self._oi_alert_cooldown_sec: float = 600.0

        # 선물-콜 추적 이탈(CDS) 알림 쿨다운 (기본 5분)
        self._last_divergence_alert_epoch: float = 0.0
        self._divergence_alert_cooldown_sec: float = 300.0

        # 옵션 가격 레벨 터치 알림 쿨다운 (기본 3분)
        # 동일 레벨이 반복 터치될 때 과다 알림을 방지한다.
        self._last_price_level_alert_epoch: float = 0.0
        self._price_level_alert_cooldown_sec: float = 180.0

        self._send_count_lock = threading.Lock()
        self._send_count_total: int = 0

        self._polling_thread: Optional[threading.Thread] = None
        self._polling_stop = threading.Event()
        self._last_update_id: int = 0

        if not self._token:
            logger.warning("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")
        if not self._chat_id:
            logger.warning("TELEGRAM_CHAT_ID가 설정되지 않았습니다.")

    @property
    def is_configured(self) -> bool:
        return bool(self._token) and bool(self._chat_id)

    def get_send_count_total(self) -> int:
        try:
            with self._send_count_lock:
                return int(self._send_count_total)
        except Exception:
            try:
                return int(self._send_count_total)
            except Exception:
                return 0

    # ──────────────────────────────────────────
    # 공개 메서드
    # ──────────────────────────────────────────

    def send_premium_bleed_alert(
        self,
        opt_snap: Dict[str, Any],
        current_price: float,
        *,
        dte_days: Optional[float] = None,
        min_score: float = 0.3,
        cooldown_sec: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """선물 상승 중 옵션 프리미엄 수축 감지 시 텔레그램으로 독립 알림 전송.

        예측 주기와 무관하게 opt_snap 변화를 감지할 때마다 호출한다.
        쿨다운 내 중복 전송을 방지하며, 신호 강도가 min_score 미만이면 억제한다.

        Args:
            opt_snap:      build_option_snapshot() 반환 dict (v4 기준).
            current_price: 현재 선물 가격.
            dte_days:      만기 잔존일. None이면 dte_weight_norm 역산.
            min_score:     전송 최소 |premium_bleed_score| (기본 0.3).
            cooldown_sec:  알림 간 최소 간격(초). None이면 인스턴스 기본값(300초) 사용.
            force:         True면 쿨다운/점수 필터 무시.

        Returns:
            전송 성공 여부.
        """
        try:
            score = float(opt_snap.get("premium_bleed_score") or 0.0)
            dte_w = float(opt_snap.get("dte_weight_norm") or 0.0)
        except Exception:
            return False

        # 만기 7일 이상(dte_w < 0.1)이거나 prev 없으면 전송 안 함
        straddle_p = float(opt_snap.get("straddle_prev") or 0.0)
        if not force:
            if dte_w < 0.1:
                logger.debug("[TG][BLEED] dte_w=%.3f < 0.1 — 전송 생략", dte_w)
                return False
            if straddle_p <= 0.0:
                logger.debug("[TG][BLEED] straddle_prev=0 (prev 없음) — 전송 생략")
                return False
            if abs(score) < float(min_score):
                logger.debug("[TG][BLEED] score=%.3f < min_score=%.3f — 전송 생략", score, min_score)
                return False

            # 쿨다운 체크
            _cooldown = float(cooldown_sec if cooldown_sec is not None else self._bleed_alert_cooldown_sec)
            now_epoch = float(time.time())
            elapsed   = now_epoch - float(self._last_bleed_alert_epoch)
            if elapsed < _cooldown:
                logger.debug(
                    "[TG][BLEED] 쿨다운 중 (elapsed=%.0f초 < cooldown=%.0f초) — 전송 생략",
                    elapsed, _cooldown,
                )
                return False

        text = format_premium_bleed_alert(
            opt_snap,
            current_price,
            dte_days=dte_days,
        )
        ok = self._send_message(
            text,
            parse_mode="MarkdownV2",
            debug_context={
                "kind": "premium_bleed",
                "score": score,
                "dte_w": dte_w,
            },
        )
        if ok:
            try:
                self._last_bleed_alert_epoch = float(time.time())
            except Exception:
                pass
            logger.info(
                "[TG][BLEED] 프리미엄 블리드 알림 전송 (score=%.2f, dte_w=%.3f)",
                score, dte_w,
            )
        return ok

    def send_oi_structure_alert(
        self,
        opt_snap: Dict[str, Any],
        current_price: float,
        *,
        min_call_conc: float = 0.3,
        min_put_conc: float = 0.3,
        cooldown_sec: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """OI 구조 변화 알림 — 만기 주간 또는 OI 집중도 급변 시 독립 전송.

        Call/Put OI Peak, Vol Trigger, Zero Gamma Level을 요약하여 전송한다.
        집중도(call_oi_peak_norm / put_oi_peak_norm) 임계값 미만이면 전송하지 않는다.

        Args:
            opt_snap:       build_option_snapshot() 반환 dict.
                            _oi_levels 키 또는 v5 직접 노출 피처를 사용한다.
            current_price:  현재 선물 가격.
            min_call_conc:  전송 최소 Call OI 집중도 (기본 0.3 = 30%).
            min_put_conc:   전송 최소 Put OI 집중도 (기본 0.3 = 30%).
            cooldown_sec:   알림 간 최소 간격(초). None이면 인스턴스 기본값(600초).
            force:          True면 쿨다운/집중도 필터 무시.

        Returns:
            전송 성공 여부.
        """
        # _oi_levels 우선, 없으면 opt_snap 직접 참조
        oi = opt_snap.get("_oi_levels") if isinstance(opt_snap, dict) else None
        if not isinstance(oi, dict) or not oi:
            oi = opt_snap if isinstance(opt_snap, dict) else {}

        try:
            call_peak  = float(oi.get("call_oi_peak") or 0.0)
            put_peak   = float(oi.get("put_oi_peak") or 0.0)
            call_peak_global = float(oi.get("call_oi_peak_global") or 0.0)
            put_peak_global = float(oi.get("put_oi_peak_global") or 0.0)
            call_conc  = float(oi.get("call_oi_peak_norm") or 0.0)
            put_conc   = float(oi.get("put_oi_peak_norm") or 0.0)
            call_conc_global = float(oi.get("call_oi_peak_global_norm") or 0.0)
            put_conc_global = float(oi.get("put_oi_peak_global_norm") or 0.0)
            vt_strike  = float(oi.get("vol_trigger_strike") or 0.0)
            zg_strike  = float(oi.get("zero_gamma_strike") or 0.0)
            above_vt   = float(oi.get("above_vol_trigger") if oi.get("above_vol_trigger") is not None else 1.0)
            range_pct  = float(oi.get("oi_range_pct") or 0.0)
            zgd        = float(oi.get("zero_gamma_dist_pct") or 0.0)
            iv_range   = float(oi.get("peak_search_range_used") or 0.0)
        except Exception:
            return False

        # OI 데이터 없으면 전송 안 함
        if call_peak <= 0.0 and put_peak <= 0.0:
            return False

        if not force:
            # 집중도 임계값 미만이면 생략
            if call_conc < float(min_call_conc) and put_conc < float(min_put_conc):
                logger.debug(
                    "[TG][OI] 집중도 미달 (call=%.2f, put=%.2f) — 전송 생략",
                    call_conc, put_conc,
                )
                return False

            # 쿨다운 체크
            _cooldown = float(cooldown_sec if cooldown_sec is not None else self._oi_alert_cooldown_sec)
            elapsed = float(time.time()) - float(self._last_oi_alert_epoch)
            if elapsed < _cooldown:
                logger.debug(
                    "[TG][OI] 쿨다운 중 (elapsed=%.0f초 < cooldown=%.0f초) — 전송 생략",
                    elapsed, _cooldown,
                )
                return False

        def _esc(s: str) -> str:
            """MarkdownV2 이스케이프 (간이 버전)."""
            for c in r"\_*[]()~`>#+-=|{}.!":
                s = s.replace(c, f"\\{c}")
            return s

        regime_str = (
            "📗 Long Gamma \\(안정권\\)" if above_vt >= 1.0
            else "📕 Short Gamma ⚠️ \\(추세 가속 가능\\)"
        )

        F_str = _esc(f"{current_price:.2f}")
        lines = [
            "📊 *OI 구조 업데이트*",
            f"현재가: `{F_str}`",
            "━━━━━━━━━━━━",
        ]

        if call_peak > 0.0:
            lines.append(
                f"🔴 저항\\(Call OI Peak\\): `{_esc(f'{call_peak:.2f}')}` "
                f"집중도 {_esc(f'{call_conc:.1%}')}"
            )
        if put_peak > 0.0:
            lines.append(
                f"🟢 지지\\(Put OI Peak\\):  `{_esc(f'{put_peak:.2f}')}` "
                f"집중도 {_esc(f'{put_conc:.1%}')}"
            )
        if call_peak_global > 0.0:
            lines.append(
                f"🔺 전체최대 Call OI: `{_esc(f'{call_peak_global:.2f}')}` "
                f"집중도 {_esc(f'{call_conc_global:.1%}')}"
            )
        if put_peak_global > 0.0:
            lines.append(
                f"🔻 전체최대 Put OI:  `{_esc(f'{put_peak_global:.2f}')}` "
                f"집중도 {_esc(f'{put_conc_global:.1%}')}"
            )
        if range_pct > 0.0:
            iv_rng_str = f"  \\(탐색반경 ±{_esc(f'{iv_range:.1f}')}pt\\)" if iv_range > 0.0 else ""
            lines.append(f"📏 OI 박스폭: {_esc(f'{range_pct:.2f}')}%{iv_rng_str}")

        if vt_strike > 0.0:
            lines.append(f"⚡ Vol Trigger: `{_esc(f'{vt_strike:.2f}')}`")
        if zg_strike > 0.0:
            zg_warn = " ⚠️" if abs(zgd) < 0.3 else ""
            lines.append(
                f"🔀 Zero Gamma: `{_esc(f'{zg_strike:.2f}')}`{_esc(zg_warn)} "
                f"\\(거리 {_esc(f'{zgd:+.2f}')}%\\)"
            )

        lines.append(f"레짐: {regime_str}")

        text = "\n".join(lines)
        ok = self._send_message(
            text,
            parse_mode="MarkdownV2",
            debug_context={
                "kind": "oi_structure",
                "call_peak": call_peak,
                "put_peak": put_peak,
                "above_vt": above_vt,
            },
        )
        if ok:
            try:
                self._last_oi_alert_epoch = float(time.time())
            except Exception:
                pass
            logger.info(
                "[TG][OI] OI 구조 알림 전송 (call_peak=%.2f, put_peak=%.2f, above_vt=%.0f)",
                call_peak, put_peak, above_vt,
            )
        return ok

    def send_futures_call_divergence_alert(
        self,
        cds_result: Dict[str, Any],
        current_price: float,
        atm_strike: float,
        *,
        dte_days: Optional[float] = None,
        min_cds: float = 0.3,
        cooldown_sec: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """선물-ATM 콜 추적 이탈(CDS) 알림을 텔레그램으로 전송한다.

        FuturesCallSimilarity.composite_divergence_score() 결과를 받아
        CDS가 임계값 이상일 때, 쿨다운 조건을 만족하면 알림을 전송한다.

        Args:
            cds_result:    composite_divergence_score() 반환 dict.
            current_price: 현재 선물가.
            atm_strike:    ATM 행사가.
            dte_days:      만기 잔존일 (표시용). None이면 생략.
            min_cds:       전송 최소 CDS 값 (기본 0.3).
            cooldown_sec:  알림 간 최소 간격(초). None이면 인스턴스 기본값(300초) 사용.
            force:         True면 쿨다운·점수 필터 무시하고 강제 전송.

        Returns:
            True면 전송 성공, False면 필터/오류로 생략.
        """
        if not self.is_configured:
            return False

        try:
            cds = float(cds_result.get("cds") or 0.0)
        except Exception:
            cds = 0.0

        if not force:
            if cds < float(min_cds):
                logger.debug(
                    "[TG][DIV] CDS 미달 (cds=%.3f < min=%.3f) — 전송 생략",
                    cds, min_cds,
                )
                return False

            now_epoch = float(time.time())
            _cooldown = float(
                cooldown_sec if cooldown_sec is not None
                else self._divergence_alert_cooldown_sec
            )
            elapsed = now_epoch - float(self._last_divergence_alert_epoch)
            if elapsed < _cooldown:
                logger.debug(
                    "[TG][DIV] 쿨다운 중 (elapsed=%.0f초 < cooldown=%.0f초) — 전송 생략",
                    elapsed, _cooldown,
                )
                return False

        try:
            text = format_futures_call_divergence_alert(
                cds_result,
                current_price,
                atm_strike,
                dte_days=dte_days,
            )
        except Exception as e:
            logger.warning("[TG][DIV] 메시지 포맷 실패: %s", e)
            return False

        ok = self._send_message(text, parse_mode="MarkdownV2")
        if ok:
            self._last_divergence_alert_epoch = float(time.time())
            logger.info(
                "[TG][DIV] 선물-콜 이탈 알림 전송 (cds=%.3f, corr=%.3f, r2=%.3f)",
                cds,
                float(cds_result.get("corr") or 0.0),
                float(cds_result.get("r2") or 0.0),
            )
        return ok

    def send_price_level_touch_alert(
        self,
        opt_snap: Dict[str, Any],
        current_price: float,
        *,
        cooldown_sec: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """옵션 고가/저가가 주요 가격 레벨에 터치 시 텔레그램 알림 전송.

        build_option_snapshot()의 '_price_level_scan' 키를 소비한다.
        터치 항목이 없거나 쿨다운 중이면 전송하지 않는다.

        Args:
            opt_snap:      build_option_snapshot() 반환 dict.
            current_price: 현재 선물 가격(pt).
            cooldown_sec:  알림 간 최소 간격(초). None이면 기본값(180초) 사용.
            force:         True면 쿨다운 필터 무시하고 강제 전송.

        Returns:
            True면 전송 성공, False면 필터/오류로 생략.
        """
        if not self.is_configured:
            return False

        scan = opt_snap.get("_price_level_scan")
        if not isinstance(scan, dict) or not scan.get("has_hit"):
            return False

        if not force:
            _cooldown = float(
                cooldown_sec if cooldown_sec is not None
                else self._price_level_alert_cooldown_sec
            )
            elapsed = float(time.time()) - float(self._last_price_level_alert_epoch)
            if elapsed < _cooldown:
                logger.debug(
                    "[TG][LEVEL] 쿨다운 중 (elapsed=%.0f초 < cooldown=%.0f초) — 전송 생략",
                    elapsed, _cooldown,
                )
                return False

        try:
            text = format_price_level_touch_alert(opt_snap, current_price)
        except Exception as e:
            logger.warning("[TG][LEVEL] 메시지 포맷 실패: %s", e)
            return False

        if not text:
            return False

        ok = self._send_message(text, parse_mode="MarkdownV2")
        if ok:
            self._last_price_level_alert_epoch = float(time.time())
            call_hits = scan.get("call_hits") or []
            put_hits  = scan.get("put_hits")  or []
            summary   = str(scan.get("summary") or "")
            logger.info(
                "[TG][LEVEL] 옵션 레벨 터치 알림 전송 "
                "(call=%d put=%d) %s",
                len(call_hits), len(put_hits), summary,
            )
        return ok

    def send_prediction(
        self,
        result: Dict[str, Any],
        *,
        force: bool = False,
        include_dir_summary: bool = True,
        symbol: str = "",
    ) -> bool:
        """예측 결과 dict를 텔레그램으로 전송.

        Args:
            result: PredictionPipeline.get_prediction() 반환값.
            force: True면 중복/간격 필터링 무시.
            include_dir_summary: True면 메시지 하단에 [DIR_SUMMARY] 모델별 action 포함.
                                  prediction_minutes 경계 틱에서만 True로 호출하는 것을 권장.

        Returns:
            전송 성공 여부.
        """
        if "error" in result:
            try:
                logger.info(
                    "[TG][SUPPRESS] prediction error not sent: %s",
                    str(result.get("error") or "unknown"),
                )
            except Exception:
                pass
            return False

        signal = str(result.get("signal", "HOLD")).upper()
        llm_action = str(result.get("llm_action", "") or "").upper()
        llm_actionable = llm_action in ("BUY", "SELL")

        # LLM이 disabled 상태이면 KP200 선물 예측 전송 건너뜀
        llm_disabled = bool(result.get("llm_disabled", False))
        if llm_disabled and not force:
            logger.info("[TG][SUPPRESS] LLM disabled — KP200 선물 예측 전송 건너뜀")
            return False

        if not force:
            # 운영 요구: LLM이 BUY/SELL을 내면 중복 신호 억제를 우회해 전송한다.
            if llm_actionable:
                force = True
                try:
                    logger.info(
                        "[TG][BYPASS] llm_action=%s actionable → duplicate suppression bypass",
                        llm_action,
                    )
                except Exception:
                    pass

            # 신호 전환 여부 먼저 확인 — 전환 시에는 only_actionable 차단을 바이패스
            with self._signal_lock:
                _prev = str(self._last_signal or "").strip().upper()
            _changed = bool(_prev) and (signal != _prev)

            if self._only_actionable and signal == "HOLD" and not _changed:
                logger.debug("HOLD 신호 (동일 반복) — 전송 생략")
                return False
            if not self._should_send(signal):
                logger.debug("중복 신호 또는 간격 미달 — 전송 생략 (signal=%s)", signal)
                return False

        with self._signal_lock:
            prev_signal_for_fmt = self._last_signal
        text = format_prediction_message(
            result,
            include_dir_summary=include_dir_summary,
            prev_signal=prev_signal_for_fmt,
            symbol=symbol,
        )
        try:
            next_cnt = int(self.get_send_count_total() or 0) + 1
            text = str(text or "") + "\n\n" + f"_TG sent: `{_esc_mdv2(str(next_cnt))}`_"
        except Exception:
            pass
        ok = self._send_message(
            text,
            parse_mode="MarkdownV2",
            debug_context={
                "kind": "prediction",
                "signal": signal,
                "include_dir_summary": include_dir_summary,
                "keys": list(result.keys()),
            },
        )
        with self._signal_lock:
            if ok:
                self._last_signal = signal
            else:
                # 전송 실패 시에도 _last_signal을 갱신하여
                # 다음 틱에서 동일 신호가 _should_send 규칙1(최초)을 재발동하는 것을 방지
                if not self._last_signal:
                    self._last_signal = signal
        return ok

    def send_option_flow_status(
        self,
        result: Dict[str, Any],
        *,
        interp: Optional[Dict[str, float]] = None,
    ) -> bool:
        """옵션 PCR/틱유입 상태를 예측 본문과 별개로 항상 전송한다."""
        try:
            if not isinstance(result, dict) or ("error" in result):
                return False
            options = result.get("options")
            if not isinstance(options, dict) or not options:
                return False
            tf = options.get("_tick_flow")
            if not isinstance(tf, dict):
                return False

            pcr_v = float(options.get("pcr_volume") or 0.0)
            pcr_oi = float(options.get("pcr_oi") or 0.0)
            t1 = float(tf.get("ticks_1m") or 0.0)
            a20 = float(tf.get("ticks_avg20m") or 0.0)
            sr = float(tf.get("surge_ratio") or 0.0)
            imb = float(tf.get("cp_imbalance") or 0.0)
            pt = float(tf.get("per_tick_move_pt") or 0.0)
            bias = "중립"
            if imb > 0.02:
                bias = "콜 우세"
            elif imb < -0.02:
                bias = "풋 우세"

            # 간단 해석 라벨 생성 (임계값은 interp dict로 오버라이드 가능)
            cfg = interp if isinstance(interp, dict) else {}
            sr_warn = float(cfg.get("sr_warn", 1.5) or 1.5)
            sr_hot = float(cfg.get("sr_hot", 2.0) or 2.0)
            pt_low = float(cfg.get("pt_low", 0.008) or 0.008)
            pt_high = float(cfg.get("pt_high", 0.03) or 0.03)
            pcr_v_low = float(cfg.get("pcr_v_low", 0.90) or 0.90)
            pcr_v_high = float(cfg.get("pcr_v_high", 1.10) or 1.10)
            pcr_oi_low = float(cfg.get("pcr_oi_low", 0.95) or 0.95)
            pcr_oi_high = float(cfg.get("pcr_oi_high", 1.05) or 1.05)

            pcr_view = "중립"
            if pcr_v >= pcr_v_high and pcr_oi >= pcr_oi_high:
                pcr_view = "풋 우위(하방/헤지 성향)"
            elif pcr_v <= pcr_v_low and pcr_oi <= pcr_oi_low:
                pcr_view = "콜 우위(상방 성향)"
            elif (pcr_v - 1.0) * (pcr_oi - 1.0) < 0:
                pcr_view = "체결/포지션 엇갈림(혼조)"

            # 요청사항: PCR(V), PCR(OI) 조합 기반 3분류 자동 라벨
            # - 둘 다 콜 우위 임계 하회: 상방확정형
            # - 둘 다 풋 우위 임계 상회: 하방확정형
            # - 그 외: 혼조형
            regime_label = "혼조형"
            if pcr_v <= pcr_v_low and pcr_oi <= pcr_oi_low:
                regime_label = "상방확정형"
            elif pcr_v >= pcr_v_high and pcr_oi >= pcr_oi_high:
                regime_label = "하방확정형"

            flow_view = "평시 유입"
            if sr >= sr_hot:
                flow_view = "유입 급증(변동성 확대 경계)"
            elif sr >= sr_warn:
                flow_view = "유입 증가"
            elif sr <= 0.7:
                flow_view = "유입 둔화"

            impact_view = "틱당 충격 보통"
            if pt >= pt_high:
                impact_view = "틱당 충격 큼(얇은 호가/급변 가능)"
            elif pt <= pt_low:
                impact_view = "틱당 충격 낮음(흡수 가능)"

            lines = [
                "📡 <b>옵션 마이크로 플로우</b>",
                f"PCR(V)/PCR(OI): <code>{pcr_v:.2f}</code> / <code>{pcr_oi:.2f}</code>",
                (
                    f"옵션 틱 유입: <code>{sr:.2f}x</code> "
                    f"(1m=<code>{int(round(t1))}</code>, avg20m=<code>{int(round(a20))}</code>) 🔥"
                ),
                f"옵션 틱 편향: <code>{imb:+.2f}</code> ({bias}), 틱당 변동: <code>{pt:.3f}pt</code>",
                (
                    "해석: "
                    f"<b>{pcr_view}</b> · <b>{flow_view}</b> · <b>{impact_view}</b>"
                ),
                f"라벨: <b>{regime_label}</b>",
            ]
            
            # OI 지지/저항 변경 정보 추가
            oi_level_change = options.get("_oi_level_change")
            oi_level_change_fired = bool(options.get("_oi_level_change_fired", False))
            
            if oi_level_change_fired and isinstance(oi_level_change, dict):
                change_type = oi_level_change.get("type", "")
                old_level = oi_level_change.get("old_level", 0.0)
                new_level = oi_level_change.get("new_level", 0.0)
                direction = oi_level_change.get("direction", "")
                
                lines.append("")
                lines.append("⚠️ <b>OI 지지/저항 변경 감지</b>")
                lines.append(f"유형: <code>{change_type}</code>")
                lines.append(f"방향: <code>{direction}</code>")
                lines.append(f"변경: <code>{old_level:.2f}</code> → <code>{new_level:.2f}</code>")
            
            return bool(
                self.send_text(
                    "\n".join(lines),
                    parse_mode="HTML",
                    debug_context={"kind": "option_flow_status"},
                )
            )
        except Exception as exc:
            try:
                logger.debug("[TG][OPTION_FLOW] status send skipped: %s", exc)
            except Exception:
                pass
            return False

    def send_realtime_ingest_status(self, result: Dict[str, Any]) -> bool:
        """실시간 수신 누적 상태 텔레그램 송출 비활성화."""
        return False

    def send_error(self, result: Dict[str, Any]) -> bool:
        """에러 결과 dict를 텔레그램으로 전송."""
        text = format_error_message(result)
        try:
            next_cnt = int(self.get_send_count_total() or 0) + 1
            text = str(text or "") + "\n\n" + f"_TG sent: `{_esc_mdv2(str(next_cnt))}`_"
        except Exception:
            pass
        return self._send_message(
            text,
            parse_mode="MarkdownV2",
            debug_context={
                "kind": "error",
                "keys": list(result.keys()),
            },
        )

    def send_text(
        self,
        text: str,
        parse_mode: str = "HTML",
        *,
        debug_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """임의 텍스트를 텔레그램으로 전송."""
        try:
            next_cnt = int(self.get_send_count_total() or 0) + 1
        except Exception:
            next_cnt = None

        try:
            if next_cnt is not None:
                if str(parse_mode or "").upper() == "HTML":
                    text = str(text or "") + f"\n\n<i>TG sent: <code>{int(next_cnt)}</code></i>"
                elif str(parse_mode or "") == "":
                    text = str(text or "") + f"\n\nTG sent: {int(next_cnt)}"
        except Exception:
            pass
        ctx: Dict[str, Any] = {
            "kind": "text",
            "parse_mode": parse_mode,
            "len": len(str(text or "")),
        }
        try:
            if isinstance(debug_context, dict) and debug_context:
                ctx.update(dict(debug_context))
        except Exception:
            pass
        return self._send_message(text, parse_mode=parse_mode, debug_context=ctx)

    def send_json_result(self, result: Dict[str, Any]) -> bool:
        """예측 결과를 JSON 코드블록으로 전송 (디버그용)."""
        snippet = json.dumps(result, ensure_ascii=False, indent=2)
        # 텔레그램 메시지 최대 4096자 제한
        if len(snippet) > 3800:
            snippet = snippet[:3800] + "\n... (생략)"
        text = f"<pre>{snippet}</pre>"
        return self._send_message(text, parse_mode="HTML")

    def send_option_sentiment_alert(
        self,
        signal: Dict[str, Any],
        current_price: float,
        *,
        cooldown_sec: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """옵션 센티먼트 이벤트 알림을 텔레그램으로 전송.

        방향성 전환 또는 신뢰도 급증 시 알림을 전송한다.

        Args:
            signal: OptionSentimentAnalyzer.analyze() 반환 SentimentSignal의 dict 표현.
            current_price: 현재 선물 가격.
            cooldown_sec: 알림 간 최소 간격(초). None이면 기본값(300초) 사용.
            force: True면 쿨다운 필터 무시하고 강제 전송.

        Returns:
            전송 성공 여부.
        """
        if not self.is_configured:
            return False

        try:
            event_type = signal.get("event_type", "none")
            direction = signal.get("direction", "neutral")
            confidence = float(signal.get("confidence") or 0.0)
            event_timestamp = signal.get("event_timestamp")  # 의미 발생 시각
        except Exception:
            return False

        # 이벤트가 없으면 전송 안 함
        if event_type == "none" and not force:
            return False

        # 쿨다운 체크 (이벤트 타입별 별도 쿨다운)
        if not force:
            _cooldown = float(cooldown_sec if cooldown_sec is not None else 300.0)

            # 이벤트 타입별 마지막 알림 시간 가져오기
            last_alerts = getattr(self, "_last_sentiment_alert_epochs", {})
            if not isinstance(last_alerts, dict):
                last_alerts = {}
                self._last_sentiment_alert_epochs = last_alerts

            last_epoch = float(last_alerts.get(event_type, 0.0))
            elapsed = float(time.time()) - last_epoch

            if elapsed < _cooldown:
                logger.debug(
                    "[TG][SENTIMENT] 쿨다운 중 (event_type=%s, elapsed=%.0f초 < cooldown=%.0f초) — 전송 생략",
                    event_type, elapsed, _cooldown,
                )
                return False

        # 메시지 포맷
        direction_emoji = {
            "bullish": "📈",
            "bearish": "📉",
            "neutral": "➡️",
        }.get(direction, "➡️")

        direction_kor = {
            "bullish": "상승",
            "bearish": "하락",
            "neutral": "중립",
        }.get(direction, "중립")

        event_emoji = "⚠️" if event_type != "none" else ""

        try:
            skew = float(signal.get("skew") or 0.0) * 100
            volume_pcr = float(signal.get("volume_pcr") or 0.0)
            oi_pcr = float(signal.get("oi_pcr") or 0.0)
            prev_direction = signal.get("prev_direction")
            prev_confidence = float(signal.get("prev_confidence") or 0.0)
        except Exception:
            skew = volume_pcr = oi_pcr = 0.0
            prev_direction = None
            prev_confidence = 0.0

        # [TIME-FIX] 의미 발생 시각 사용 (없으면 현재 시간)
        if event_timestamp:
            if isinstance(event_timestamp, (int, float)):
                # Unix timestamp
                event_dt = datetime.fromtimestamp(event_timestamp, datetime.timezone.utc)
                event_dt = event_dt.astimezone(datetime.timezone(datetime.timedelta(hours=9)))  # KST
                time_str = event_dt.strftime("%H:%M:%S")
            elif isinstance(event_timestamp, datetime):
                event_dt = event_timestamp
                time_str = event_dt.strftime("%H:%M:%S")
            else:
                time_str = str(event_timestamp)
        else:
            time_str = self._now_fn().strftime("%H:%M:%S")

        lines = [
            f"{direction_emoji} <b>옵션 센티먼트 이벤트</b> {event_emoji}",
            f"발생 시각: <code>{time_str}</code>",
            f"종합 방향: <b>{direction_kor}</b> (신뢰도: <code>{confidence*100:.1f}%</code>)",
            f"현재가: <code>{current_price:.2f}</code>",
            "━━━━━━━━━━━━",
            f"Skew: <code>{skew:+.2f}%</code>",
            f"Volume PCR: <code>{volume_pcr:.2f}</code>",
            f"OI PCR: <code>{oi_pcr:.2f}</code>",
        ]

        if event_type == "direction_change" and prev_direction:
            prev_dir_kor = {
                "bullish": "상승",
                "bearish": "하락",
                "neutral": "중립",
            }.get(prev_direction, "알 수 없음")
            lines.append(f"<b>이벤트: 방향성 전환</b> ({prev_dir_kor} → {direction_kor})")
        elif event_type == "confidence_spike":
            lines.append(f"<b>이벤트: 신뢰도 급증</b> ({prev_confidence*100:.1f}% → {confidence*100:.1f}%)")
        elif "direction_change" in event_type and "confidence_spike" in event_type:
            # 복합 이벤트: 방향성 전환 + 신뢰도 급증
            if prev_direction:
                prev_dir_kor = {
                    "bullish": "상승",
                    "bearish": "하락",
                    "neutral": "중립",
                }.get(prev_direction, "알 수 없음")
                lines.append(f"<b>이벤트: 방향성 전환 + 신뢰도 급증</b> ({prev_dir_kor} → {direction_kor}, {prev_confidence*100:.1f}% → {confidence*100:.1f}%)")
            else:
                lines.append(f"<b>이벤트: 방향성 전환 + 신뢰도 급증</b> ({prev_confidence*100:.1f}% → {confidence*100:.1f}%)")
        elif "direction_change" in event_type:
            # 복합 이벤트의 일부로 방향성 전환만 포함
            if prev_direction:
                prev_dir_kor = {
                    "bullish": "상승",
                    "bearish": "하락",
                    "neutral": "중립",
                }.get(prev_direction, "알 수 없음")
                lines.append(f"<b>이벤트: 방향성 전환</b> ({prev_dir_kor} → {direction_kor})")
        elif "confidence_spike" in event_type:
            # 복합 이벤트의 일부로 신뢰도 급증만 포함
            lines.append(f"<b>이벤트: 신뢰도 급증</b> ({prev_confidence*100:.1f}% → {confidence*100:.1f}%)")

        text = "\n".join(lines)
        ok = self.send_text(text, parse_mode="HTML")

        if ok:
            try:
                # 이벤트 타입별 마지막 알림 시간 업데이트
                last_alerts = getattr(self, "_last_sentiment_alert_epochs", {})
                if not isinstance(last_alerts, dict):
                    last_alerts = {}
                    self._last_sentiment_alert_epochs = last_alerts
                last_alerts[event_type] = float(time.time())
                self._last_sentiment_alert_epochs = last_alerts
            except Exception:
                pass
            logger.info(
                "[TG][SENTIMENT] 옵션 센티먼트 알림 전송 (event=%s, direction=%s, confidence=%.2f)",
                event_type, direction, confidence,
            )
        else:
            logger.warning(
                "[TG][SENTIMENT] 옵션 센티먼트 알림 전송 실패 (event=%s, direction=%s, confidence=%.2f)",
                event_type, direction, confidence,
            )
        return ok

    # ──────────────────────────────────────────
    # 명령 수신 폴링
    # ──────────────────────────────────────────

    def start_polling(
        self,
        on_command: Callable[[str, int], None],
        poll_interval: float = 2.0,
    ) -> None:
        """백그라운드에서 텔레그램 업데이트를 폴링하며 명령을 수신합니다.

        Args:
            on_command: 명령 수신 시 호출되는 콜백.
                        (command: str, chat_id: int) → None
                        예: on_command("/predict", 123456789)
            poll_interval: 폴링 주기 (초). 기본 2초.

        지원 명령:
            /predict   — 즉시 예측 요청
            /status    — 현재 상태 조회
            /pause     — 예측 알림 일시 정지
            /resume    — 예측 알림 재개
            /help      — 도움말
        """
        if self._polling_thread and self._polling_thread.is_alive():
            logger.warning("폴링 스레드가 이미 실행 중입니다.")
            return

        self._polling_stop.clear()

        def _loop() -> None:
            logger.info("텔레그램 폴링 시작 (간격: %.1f초)", poll_interval)
            while not self._polling_stop.is_set():
                try:
                    updates = self._get_updates()
                    for update in updates:
                        uid = int(update.get("update_id", 0))
                        if uid <= self._last_update_id:
                            continue
                        self._last_update_id = uid
                        self._dispatch_update(update, on_command)
                except Exception as exc:
                    logger.warning("폴링 오류: %s", exc)
                self._polling_stop.wait(timeout=poll_interval)
            logger.info("텔레그램 폴링 종료")

        self._polling_thread = threading.Thread(target=_loop, daemon=True, name="TelegramPoller")
        self._polling_thread.start()

    def stop_polling(self) -> None:
        """폴링 스레드를 정지합니다."""
        self._polling_stop.set()

    # ──────────────────────────────────────────
    # 내부 유틸리티
    # ──────────────────────────────────────────

    def _increment_send_count(self) -> None:
        """전송 카운터를 thread-safe하게 1 증가."""
        try:
            with self._send_count_lock:
                self._send_count_total += 1
        except Exception:
            try:
                self._send_count_total += 1
            except Exception:
                pass

    def _should_send(self, signal: str) -> bool:
        """신호 변경 필터.

        규칙:
          1. 이전 신호가 없으면 (최초) → 전송
          2. 신호가 변경된 경우 (BUY/SELL/HOLD 간 전환) → 전송
          3. 동일 신호 반복 (BUY→BUY, SELL→SELL, HOLD→HOLD) → 억제

        주의: only_actionable에 의한 HOLD 필터링은 호출 전에 처리한다.
              신호 전환(예: BUY→HOLD)이면 only_actionable 차단을 바이패스하므로
              이 메서드까지 도달한 경우는 항상 규칙 2에 의해 전송된다.
        """
        try:
            with self._signal_lock:
                prev = str(self._last_signal or "").strip().upper()
        except Exception:
            prev = ""
        cur = str(signal or "").strip().upper()

        # 규칙 1: 최초 전송 (이전 신호 없음)
        if not prev:
            return True

        # 규칙 2: 신호 변경 시 전송 (BUY↔SELL↔HOLD 전환)
        if cur != prev:
            return True

        # 규칙 3: 동일 신호 반복 → 억제 (HOLD→HOLD, BUY→BUY, SELL→SELL 모두 차단)
        logger.debug(
            "[TG][SUPPRESS] 동일 신호 반복 억제 (prev=%s cur=%s)",
            prev, cur,
        )
        return False

    def _api_url(self, method: str) -> str:
        return self.BASE_URL.format(token=self._token, method=method)

    def _http_post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """urllib로 JSON POST 요청. proxy_url 설정 시 프록시를 경유한다."""
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            _open = self._opener.open if self._opener else urllib.request.urlopen
            with _open(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                raw = ""
            try:
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
            return {"ok": False, "error_code": getattr(e, "code", None), "description": str(e)}
        except Exception as e:
            return {"ok": False, "description": str(e)}

    def _http_get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """urllib로 GET 요청. proxy_url 설정 시 프록시를 경유한다."""
        full_url = url + "?" + urllib.parse.urlencode(params)
        try:
            _open = self._opener.open if self._opener else urllib.request.urlopen
            with _open(full_url, timeout=self._timeout + 2) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                raw = ""
            try:
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
            return {"ok": False, "error_code": getattr(e, "code", None), "description": str(e)}
        except Exception as e:
            return {"ok": False, "description": str(e)}

    def _extract_telegram_byte_offset(self, description: str) -> Optional[int]:
        m = re.search(r"byte offset (\d+)", description)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _message_snippet_at_byte_offset(self, text: str, byte_offset: int, context_bytes: int = 160) -> str:
        try:
            b = text.encode("utf-8", errors="replace")
            start = max(0, byte_offset - context_bytes)
            end = min(len(b), byte_offset + context_bytes)
            snippet = b[start:end].decode("utf-8", errors="replace")
            return snippet
        except Exception:
            return ""

    def _to_plain_text(self, text: str, *, parse_mode: str = "") -> str:
        """parse_mode 기반 포맷 제거 후 읽기 쉬운 plain text로 변환."""
        plain = str(text or "")
        try:
            if str(parse_mode or "").upper() == "HTML":
                # HTML parse_mode 실패 폴백 시 태그가 그대로 노출되지 않도록 제거
                plain = re.sub(r"<br\s*/?>", "\n", plain, flags=re.IGNORECASE)
                plain = re.sub(r"</p\s*>", "\n", plain, flags=re.IGNORECASE)
                plain = re.sub(r"<[^>]+>", "", plain)
                plain = html.unescape(plain)
            else:
                # MarkdownV2/일반 텍스트 폴백: 최소한의 장식 문자 제거
                plain = plain.replace("\\", "")
                plain = plain.replace("*", "").replace("`", "").replace("_", "")
        except Exception:
            pass
        return plain

    def _send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        debug_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Telegram sendMessage API 호출."""
        if not self._token or not self._chat_id:
            logger.error("봇 토큰 또는 채팅 ID가 설정되지 않았습니다.")
            return False
        try:
            data = self._http_post(
                self._api_url("sendMessage"),
                {
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
            )
            if not data.get("ok"):
                if parse_mode != "":
                    desc = str(data.get("description") or "")
                    offset = self._extract_telegram_byte_offset(desc)
                    snippet = (
                        self._message_snippet_at_byte_offset(text, offset)
                        if offset is not None
                        else ""
                    )
                    text_head = text[:500]
                    logger.warning(
                        "메시지 전송 실패 — 일반 텍스트 재시도: %s | offset=%s | snippet=%r | head=%r | len=%d | ctx=%s",
                        desc,
                        offset,
                        snippet,
                        text_head,
                        len(text),
                        debug_context,
                    )
                    return self._send_message_plain(text, parse_mode=parse_mode, debug_context=debug_context)
                logger.error("텔레그램 전송 실패: %s", data)
                return False
            try:
                head = str(text or "")
                if len(head) > 200:
                    head = head[:200] + "..."
                try:
                    kind = (debug_context or {}).get("kind") if isinstance(debug_context, dict) else None
                except Exception:
                    kind = None
                if _TG_DEBUG or kind == "startup":
                    logger.info(
                        "[TG][SEND] ok parse_mode=%s len=%d head=%r ctx=%s",
                        parse_mode,
                        len(text or ""),
                        head,
                        debug_context,
                    )
            except Exception:
                pass

            try:
                self._increment_send_count()
            except Exception:
                pass
            return True
        except Exception as exc:
            logger.error("텔레그램 전송 예외: %s", exc)
            return False

    def _send_message_plain(
        self,
        text: str,
        parse_mode: str = "",
        debug_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """마크다운 없이 일반 텍스트로 전송 (폴백)."""
        plain = self._to_plain_text(text, parse_mode=parse_mode)
        try:
            data = self._http_post(
                self._api_url("sendMessage"),
                {"chat_id": self._chat_id, "text": plain, "disable_web_page_preview": True},
            )
            if not data.get("ok"):
                logger.error("일반 텍스트 전송도 실패: %s | ctx=%s", data, debug_context)
            ok = bool(data.get("ok"))
            if ok:
                self._increment_send_count()
            return ok
        except Exception as exc:
            logger.error("일반 텍스트 전송도 실패: %s", exc)
            return False

    def _get_updates(self) -> list:
        """getUpdates API 호출."""
        params: Dict[str, Any] = {"timeout": 1}
        if self._last_update_id > 0:
            params["offset"] = self._last_update_id + 1
        data = self._http_get(self._api_url("getUpdates"), params)
        if data.get("ok"):
            return data.get("result", [])
        return []

    def _dispatch_update(
        self,
        update: Dict[str, Any],
        on_command: Callable[[str, int], None],
    ) -> None:
        """수신된 업데이트에서 명령어를 추출해 콜백 호출."""
        msg = update.get("message") or {}
        text = str(msg.get("text") or "").strip()
        chat_id = int((msg.get("chat") or {}).get("id") or 0)
        from_id = int((msg.get("from") or {}).get("id") or 0)

        if not text or not chat_id:
            return

        # DS-02: 허가되지 않은 chat_id 로부터의 명령 차단
        if self._chat_id and str(chat_id) != str(self._chat_id):
            logger.warning(
                "[TG][SECURITY] 허가되지 않은 chat_id=%d 명령 무시: %r (허가된 chat_id=%s)",
                chat_id, text, self._chat_id,
            )
            return

        try:
            logger.info("[TG][RECV] chat=%d from=%d text=%r", chat_id, from_id, text)
        except Exception:
            pass

        # 봇 멘션 제거 (예: /predict@MyBot → /predict)
        # NOTE: '/@' 같은 커스텀 숏컷은 '@'를 포함하므로, 단순 split("@")는 사용하면 안 된다.
        try:
            m = re.match(r"^/([A-Za-z0-9_]+)@[A-Za-z0-9_]+(\s|$)", text)
        except Exception:
            m = None
        if m:
            try:
                text = "/" + str(m.group(1)) + text[m.end(0) - 1 :]
            except Exception:
                try:
                    text = "/" + str(m.group(1))
                except Exception:
                    pass

        if text.startswith("/"):
            logger.info("명령 수신: %s (from=%d, chat=%d)", text, from_id, chat_id)
            try:
                on_command(text, chat_id)
            except Exception as exc:
                logger.error("명령 처리 오류 (%s): %s", text, exc)



# ──────────────────────────────────────────────
# PredictionPipeline 연동 헬퍼
# [FIX] 레거시 클래스 제거: telegram.bridge.PipelineTelegramBridge로 단일화.
# main.py에서 `from telegram.notifier import PipelineTelegramBridge` 유지를 위해
# re-export alias를 제공한다. 실제 구현은 telegram/bridge.py 참조.
# ──────────────────────────────────────────────
try:
    from telegram.bridge import PipelineTelegramBridge  # re-export
except ImportError:
    try:
        from .bridge import PipelineTelegramBridge  # type: ignore[assignment]
    except ImportError:
        PipelineTelegramBridge = None  # type: ignore[assignment,misc]


def load_telegram_config(
    secrets_path: Optional[str] = None,
    config_path: str = "config.secrets.json",
) -> Dict[str, Any]:
    """config.secrets.json 또는 환경변수에서 텔레그램 설정을 로드합니다.

    secrets 파일 경로 결정 순서:
      1. secrets_path 인자
      2. 환경변수 APP_SECRETS_CONFIG
      3. config_path 인자 (기본: config.secrets.json)

    우선순위: 환경변수 > secrets 파일

    Returns:
        {"bot_token": str, "chat_id": str}
    """
    import pathlib

    # secrets 파일 경로 결정
    resolved = (
        secrets_path
        or os.environ.get("APP_SECRETS_CONFIG")
        or config_path
    )

    cfg: Dict[str, Any] = {}

    # secrets 파일에서 봇 토큰 / 채팅 ID 읽기
    try:
        path = pathlib.Path(resolved)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            telegram = data.get("telegram") or {}
            if telegram.get("bot_token"):
                cfg["bot_token"] = str(telegram["bot_token"])
            if telegram.get("chat_id"):
                cfg["chat_id"] = str(telegram["chat_id"])
        else:
            logger.debug("secrets 파일 없음: %s", path)
    except Exception as exc:
        logger.warning("secrets 파일 읽기 실패: %s", exc)

    # 환경변수가 파일보다 우선
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        cfg["bot_token"] = os.environ["TELEGRAM_BOT_TOKEN"]
    if os.environ.get("TELEGRAM_CHAT_ID"):
        cfg["chat_id"] = os.environ["TELEGRAM_CHAT_ID"]

    return cfg


def create_notifier_from_config(
    secrets_path: Optional[str] = None,
    **kwargs: Any,
) -> TelegramNotifier:
    """config.secrets.json / 환경변수 / APP_SECRETS_CONFIG 로 TelegramNotifier를 생성합니다.

    secrets 경로는 APP_SECRETS_CONFIG 환경변수로 오버라이드 가능합니다.
    """
    cfg = load_telegram_config(secrets_path)
    return TelegramNotifier(
        bot_token=cfg.get("bot_token"),
        chat_id=cfg.get("chat_id"),
        **kwargs,
    )


def create_bridge_from_config(
    pipeline: Any,
    secrets_path: Optional[str] = None,
    **bridge_kwargs: Any,
) -> Optional["PipelineTelegramBridge"]:
    """config.secrets.json / 환경변수 / APP_SECRETS_CONFIG 로 PipelineTelegramBridge를 생성합니다.

    텔레그램 활성화 여부(예: config.json의 telegram.enabled, GUI 체크박스)는
    엔트리포인트(main.py)에서 결정하는 것을 권장합니다.

    Args:
        pipeline: PredictionPipeline 인스턴스.
        secrets_path: secrets 파일 경로 오버라이드 (없으면 APP_SECRETS_CONFIG → config.secrets.json).
        **bridge_kwargs: PipelineTelegramBridge 생성자 추가 인자
                         (predict_interval_sec, only_consensus, only_actionable 등).

    Returns:
        PipelineTelegramBridge 인스턴스.

    사용 예시 (main.py):
        bridge = create_bridge_from_config(pipeline)
        bridge.start()
        bridge.start_polling()
    """
    cfg = load_telegram_config(secrets_path)
    notifier = TelegramNotifier(
        bot_token=cfg.get("bot_token"),
        chat_id=cfg.get("chat_id"),
    )
    return PipelineTelegramBridge(pipeline, notifier, **bridge_kwargs)


# ──────────────────────────────────────────────
# 단독 테스트용 더미 데이터
# ──────────────────────────────────────────────

# 시나리오별 더미 결과 생성
_SCENARIOS: Dict[str, Any] = {}   # 아래 _build_scenarios()로 채워짐

def _build_scenarios(now_fn: Optional[Callable[[], datetime]] = None) -> Dict[str, Dict[str, Any]]:
    """테스트 시나리오별 더미 예측 결과 딕셔너리를 반환합니다.

    Args:
        now_fn: 시간 함수 (테스트/백테스트용 주입 가능)
    """
    _now = now_fn if now_fn is not None else datetime.now
    now   = _now().isoformat()
    base  = dict(
        prediction_time=now, target_time=now,
        prediction_minutes=5,
        current_price=355.25, spot_index=354.80, basis=0.45,
        ob_records_len=120, fo0_age_sec=0.3,
        options={"pcr_volume": 0.82, "pcr_oi": 0.91},
    )
    return {
        # ── 정상 예측 ──────────────────────────────────────────────
        "buy": {**base,
            "prob": 0.71, "signal": "BUY", "confidence": "HIGH",
            "transformer_prob": 0.73, "tft_prob": 0.69,
            "ensemble_method": "weighted_avg", "model_agreement": True,
            "regime": "STRONG_UP",
            "llm_action": "BUY", "llm_provider": "claude", "llm_timed_out": False,
            "risk_level": "MEDIUM", "consensus": True,
            "rationale": "강한 상승 추세 지속. 옵션 PCR 0.82로 매수 우위.",
            "caution": "오후 장 변동성 확대 가능성.",
            "model_outputs": {
                "heuristic":   {"action": "BUY",  "provider": "heuristic"},
                "transformer": {"action": "BUY",  "provider": "transformer"},
                "tft":         {"action": "BUY",  "provider": "tft"},
                "claude":      {"action": "BUY",  "provider": "claude"},
            },
        },
        "sell": {**base,
            "current_price": 352.10, "spot_index": 353.60, "basis": -1.50,
            "options": {"pcr_volume": 1.42, "pcr_oi": 1.18},
            "prob": 0.27, "signal": "SELL", "confidence": "HIGH",
            "transformer_prob": 0.25, "tft_prob": 0.29,
            "ensemble_method": "weighted_avg", "model_agreement": True,
            "regime": "STRONG_DOWN",
            "llm_action": "SELL", "llm_provider": "gpt", "llm_timed_out": False,
            "risk_level": "HIGH", "consensus": True,
            "rationale": "베이시스 역전 -1.50. 풋 우위 강세, 하락 추세 지속.",
            "caution": "급반등 가능성 — 손절선 필수.",
            "model_outputs": {
                "heuristic":   {"action": "SELL", "provider": "heuristic"},
                "transformer": {"action": "SELL", "provider": "transformer"},
                "tft":         {"action": "SELL", "provider": "tft"},
                "gpt":         {"action": "SELL", "provider": "gpt"},
            },
        },
        "hold": {**base,
            "prob": 0.52, "signal": "HOLD", "confidence": "LOW",
            "transformer_prob": 0.55, "tft_prob": 0.49,
            "ensemble_method": "weighted_avg", "model_agreement": False,
            "regime": "RANGE",
            "llm_action": "HOLD", "llm_provider": "claude", "llm_timed_out": False,
            "risk_level": "LOW", "consensus": True,
            "rationale": "박스권 횡보. 방향성 불명확.",
            "caution": "",
            "model_outputs": {
                "heuristic":   {"action": "HOLD", "provider": "heuristic"},
                "transformer": {"action": "BUY",  "provider": "transformer"},
                "tft":         {"action": "SELL", "provider": "tft"},
                "claude":      {"action": "HOLD", "provider": "claude"},
            },
        },
        # ── 불일치 / 특수 케이스 ───────────────────────────────────
        "disagreement": {**base,
            "prob": 0.63, "signal": "BUY", "confidence": "MEDIUM",
            "transformer_prob": 0.63, "tft_prob": 0.38,
            "ensemble_method": "disagreement_hold", "model_agreement": False,
            "regime": "WEAK_UP",
            "llm_action": "HOLD", "llm_provider": "dual_disagreement_hold",
            "llm_timed_out": False,
            "risk_level": "MEDIUM", "consensus": False,  # ← 컨센서스 불일치
            "rationale": "Transformer BUY vs TFT SELL 불일치. 관망 권장.",
            "caution": "방향 확정 후 진입 권장.",
            "model_outputs": {
                "transformer": {"action": "BUY",  "provider": "transformer"},
                "tft":         {"action": "SELL", "provider": "tft"},
                "gpt":         {"action": "BUY",  "provider": "gpt"},
                "gemini":      {"action": "SELL", "provider": "gemini"},
            },
        },
        "llm_timeout": {**base,
            "prob": 0.65, "signal": "BUY", "confidence": "MEDIUM",
            "transformer_prob": 0.65, "tft_prob": None,
            "ensemble_method": "transformer_only", "model_agreement": None,
            "regime": "WEAK_UP",
            "llm_action": "BUY", "llm_provider": "timeout", "llm_timed_out": True,
            "risk_level": "LOW", "consensus": True,
            "rationale": "LLM 타임아웃 — Transformer 단독 결과 사용.",
            "caution": "",
            "model_outputs": {
                "transformer": {"action": "BUY", "provider": "transformer"},
            },
        },
        # ── 패리티 가드레일 케이스 ──────────────────────────────
        "parity_hold": {**base,
            "prob": 0.68, "signal": "HOLD", "confidence": "LOW",
            "transformer_prob": 0.68, "tft_prob": 0.65,
            "ensemble_method": "weighted_avg", "model_agreement": True,
            "regime": "STRONG_UP",
            "llm_action": "BUY", "llm_provider": "claude", "llm_timed_out": False,
            "risk_level": "LOW", "consensus": False,
            "rationale": "패리티 이탈로 신호 강제 HOLD. 만기 당일 ATM 콜 과매도 감지.",
            "caution": "만기 후 정상화 확인 필요.",
            "guardrail": {
                "applied": True,
                "original_signal": "BUY",
                "original_confidence": "HIGH",
                "reason": "parity_divergence_critical(score=0.83,dte_w=1.00)",
            },
            "options": {"pcr_volume": 0.91, "pcr_oi": 1.05,
                        "parity_divergence_score": -0.83, "dte_weight_norm": 1.0,
                        "parity_spread_pct": -0.41},
            "model_outputs": {
                "transformer": {"action": "BUY",  "provider": "transformer"},
                "tft":         {"action": "BUY",  "provider": "tft"},
                "claude":      {"action": "BUY",  "provider": "claude"},
            },
        },
        "parity_demote": {**base,
            "prob": 0.61, "signal": "BUY", "confidence": "LOW",
            "transformer_prob": 0.61, "tft_prob": 0.58,
            "ensemble_method": "weighted_avg", "model_agreement": True,
            "regime": "WEAK_UP",
            "llm_action": "BUY", "llm_provider": "gpt", "llm_timed_out": False,
            "risk_level": "MEDIUM", "consensus": True,
            "rationale": "패리티 이탈로 신뢰도 강등. 만기 2일 전 콜 수익률 추종 이탈 감지.",
            "caution": "만기 전 패리티 정상화 여부 모니터링 필요.",
            "guardrail": {
                "applied": True,
                "original_signal": "BUY",
                "original_confidence": "MEDIUM",
                "reason": "parity_divergence(score=0.61,dte_w=0.050)",
            },
            "options": {"pcr_volume": 0.87, "pcr_oi": 0.93,
                        "parity_divergence_score": 0.61, "dte_weight_norm": 0.05},
            "model_outputs": {
                "transformer": {"action": "BUY", "provider": "transformer"},
                "tft":         {"action": "BUY", "provider": "tft"},
                "gpt":         {"action": "BUY", "provider": "gpt"},
            },
        },
        # ── 에러 케이스 ───────────────────────────────────────────
        "error_data": {
            "error": "insufficient_minutes",
            "message": "분봉 데이터 부족 (현재: 8개, 필요: 20개)",
        },
        "error_ob": {
            "error": "ob_buffer_empty",
            "message": "호가창 버퍼가 비어 있습니다. FO0 데이터 수신 대기 중.",
        },
    }


def _print_formatted(scenario_name: str, result: Dict[str, Any], include_dir_summary: bool) -> None:
    """포매팅 결과를 콘솔에 출력합니다."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  시나리오: {scenario_name.upper()}")
    print(sep)
    if "error" in result:
        msg = format_error_message(result)
    else:
        msg = format_prediction_message(result, include_dir_summary=include_dir_summary)
    print(msg)
    print(sep)
    print(f"  메시지 길이: {len(msg)}자  (텔레그램 한도: 4096자)")
    print(sep)


# ──────────────────────────────────────────────
# 단독 실행 진입점
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    scenarios = _build_scenarios()
    scenario_names = list(scenarios.keys())

    parser = argparse.ArgumentParser(
        prog="telegram_notifier.py",
        description="TelegramNotifier 단독 테스트 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
        시나리오 목록:
        {chr(10)+"  ".join(f"{k:<18} — {('에러 메시지' if 'error' in v else v.get('signal','?')+' / '+v.get('regime',''))}" for k, v in scenarios.items())}

        사용 예시:
        # 포매팅만 확인 (전송 없음)
        python telegram_notifier.py
        python telegram_notifier.py --scenario sell
        python telegram_notifier.py --all

        # 실제 텔레그램 전송
        python telegram_notifier.py --send --token <BOT_TOKEN> --chat <CHAT_ID>
        python telegram_notifier.py --send --scenario error_data

        # 환경변수로 토큰 설정 후 전송
        set TELEGRAM_BOT_TOKEN=...  &&  set TELEGRAM_CHAT_ID=...
        python telegram_notifier.py --send

        # 폴링 수신 테스트 (봇에 명령 전송하며 응답 확인)
        python telegram_notifier.py --poll --token <BOT_TOKEN> --chat <CHAT_ID>
        """,
            )

    # ── 시나리오 선택 ──────────────────────────────────────────
    parser.add_argument(
        "--scenario", "-s",
        choices=scenario_names,
        default="buy",
        metavar="SCENARIO",
        help=f"테스트 시나리오 선택 (기본: buy). 선택지: {', '.join(scenario_names)}",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="모든 시나리오를 순서대로 포매팅하여 출력합니다.",
    )

    # ── 전송 옵션 ──────────────────────────────────────────────
    parser.add_argument(
        "--send",
        action="store_true",
        help="포매팅 결과를 실제 텔레그램으로 전송합니다 (토큰 필요).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="--send 시 MarkdownV2 대신 JSON 코드블록으로 전송합니다.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="[DIR_SUMMARY] 블록을 메시지에서 제외합니다.",
    )

    # ── 인증 정보 ──────────────────────────────────────────────
    parser.add_argument(
        "--token",
        default=None,
        metavar="BOT_TOKEN",
        help="봇 토큰 직접 지정 (환경변수 TELEGRAM_BOT_TOKEN보다 우선).",
    )
    parser.add_argument(
        "--chat",
        default=None,
        metavar="CHAT_ID",
        help="채팅 ID 직접 지정 (환경변수 TELEGRAM_CHAT_ID보다 우선).",
    )
    parser.add_argument(
        "--secrets",
        default=None,
        metavar="PATH",
        help="config.secrets.json 경로 직접 지정.",
    )

    # ── 폴링 테스트 ────────────────────────────────────────────
    parser.add_argument(
        "--poll",
        action="store_true",
        help="폴링 수신 테스트 — 봇에게 명령을 전송하면 콘솔에 출력합니다.",
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=60.0,
        metavar="SEC",
        help="폴링 테스트 지속 시간(초, 기본: 60).",
    )

    args = parser.parse_args()
    include_dir_summary = not args.no_summary

    # CLI 인자 토큰이 환경변수보다 우선
    if args.token:
        os.environ["TELEGRAM_BOT_TOKEN"] = args.token
    if args.chat:
        os.environ["TELEGRAM_CHAT_ID"] = args.chat

    # ── 1. 모든 시나리오 포매팅 출력 ──────────────────────────
    if args.all:
        for name, result in scenarios.items():
            _print_formatted(name, result, include_dir_summary)
        sys.exit(0)

    # ── 2. 선택 시나리오 포매팅 출력 ──────────────────────────
    result = scenarios[args.scenario]
    _print_formatted(args.scenario, result, include_dir_summary)

    # ── 3. 노티파이어 초기화 ──────────────────────────────────
    notifier = create_notifier_from_config(secrets_path=args.secrets)

    # ── 4. 폴링 수신 테스트 ───────────────────────────────────
    if args.poll:
        if not notifier.is_configured:
            print("\n❌ 폴링 실패: 봇 토큰 / 채팅 ID 미설정")
            print("   --token <BOT_TOKEN> --chat <CHAT_ID> 를 추가하거나 환경변수를 설정하세요.")
            sys.exit(1)

        def _on_command(command: str, chat_id: int) -> None:
            print(f"\n[폴링] 수신: {command!r}  (chat_id={chat_id})")
            notifier.send_text(
                f"✅ <b>수신 확인</b>: <code>{command}</code>\n"
                f"(단독 테스트 모드)",
                parse_mode="HTML",
            )

        print(f"\n폴링 시작 — {args.poll_sec:.0f}초간 대기합니다.")
        print("봇에게 명령을 보내세요: /predict  /status  /pause  /resume  /json  /help")
        notifier.start_polling(on_command=_on_command, poll_interval=2.0)
        try:
            time.sleep(args.poll_sec)
        except KeyboardInterrupt:
            print("\n(Ctrl+C) 중단")
        finally:
            notifier.stop_polling()
        sys.exit(0)

    # ── 5. 실제 전송 ──────────────────────────────────────────
    if args.send:
        if not notifier.is_configured:
            print("\n❌ 전송 실패: 봇 토큰 / 채팅 ID 미설정")
            print("   다음 중 하나로 설정하세요:")
            print("     --token <BOT_TOKEN> --chat <CHAT_ID>")
            print("     환경변수: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
            print('     config.secrets.json: {"telegram": {"bot_token":"...", "chat_id":"..."}}')
            sys.exit(1)

        print(f"\n전송 중... (시나리오: {args.scenario})")
        if args.json:
            ok = notifier.send_json_result(result)
        elif "error" in result:
            ok = notifier.send_error(result)
        else:
            ok = notifier.send_prediction(result, force=True, include_dir_summary=include_dir_summary)

        print("전송 결과:", "✅ 성공" if ok else "❌ 실패")
        sys.exit(0 if ok else 1)

    # ── 6. 토큰 미설정 시 안내 ────────────────────────────────
    if not notifier.is_configured:
        print("\n💡 실제 전송하려면:")
        print("   python telegram_notifier.py --send --token <BOT_TOKEN> --chat <CHAT_ID>")
