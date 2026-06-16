"""PipelineTelegramBridge 분리 모듈.

이 파일은 telegram_notifier.py에서 분리된 Mixin 클래스입니다.
직접 인스턴스화하지 마세요.
"""
from __future__ import annotations
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


from .commands import CommandsMixin
from .monitors import MonitorsMixin

# ── Trade Gate Import ─────────────────────────────────────────────────────
try:
    from trading.gate import TradeExecutionGate, TradeGateConfig
    _TRADE_GATE_AVAILABLE = True
except Exception:
    TradeExecutionGate = None  # type: ignore
    TradeGateConfig = None  # type: ignore
    _TRADE_GATE_AVAILABLE = False

try:
    from trading.pivot_gate import PivotExecutionGate, PivotGateConfig
    _PIVOT_GATE_AVAILABLE = True
except Exception:
    PivotExecutionGate = None  # type: ignore
    PivotGateConfig = None  # type: ignore
    _PIVOT_GATE_AVAILABLE = False


class PipelineTelegramBridge(CommandsMixin, MonitorsMixin):
    """PipelineTelegramBridge 코어.

    Public API (시그니처 불변):
        start()         — 브리지 시작 (모니터링 루프 스레드 기동)
        stop()          — 브리지 종료
        start_polling() — 텔레그램 커맨드 폴링 시작
        predict_now()   — 즉시 예측 실행
    """

    def __init__(
        self,
        pipeline: Any,
        notifier: TelegramNotifier,
        predict_interval_sec: float = 60.0,
        only_consensus: bool = False,
        only_actionable: bool = False,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        """
        Args:
            pipeline: PredictionPipeline 인스턴스.
            notifier: TelegramNotifier 인스턴스.
            predict_interval_sec: 예측 주기 (초). 기본 60초.
            only_consensus: True면 Transformer와 LLM이 합의한 경우에만 전송.
            only_actionable: True면 HOLD 신호는 전송 생략.
            now_fn: 시간 함수 (테스트/백테스트용 주입 가능)
        """
        self._pipeline = pipeline
        self._notifier = notifier
        self._interval = float(predict_interval_sec)
        self._only_consensus = bool(only_consensus)
        self._only_actionable = bool(only_actionable)
        self._now_fn = now_fn if now_fn is not None else datetime.now

        self._run_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # CON-03: _user_paused를 threading.Event로 교체.
        # 복합 조건 평가(not paused and not market_paused)가 비원자적이던 문제 해결.
        # set() = 일시정지, clear() = 재개.
        self._user_pause_event = threading.Event()   # set = 일시정지
        self._market_paused: bool = False   # market_closed 감지 시 자동 제어
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_boundary_minute: int = -1   # DIR_SUMMARY 경계 추적

        self._first_prediction_sent: bool = False
        self._first_prediction_attempt_count: int = 0  # 최초 전송 실패 시 무한 재시도 방지
        self._first_prediction_max_attempts: int = 3   # 최대 재시도 횟수

        self._error_window_sec = 600.0
        self._error_threshold = 3
        self._error_events: deque[float] = deque(maxlen=100)
        self._last_error_alert_epoch: float = 0.0
        self._error_alert_cooldown_sec: float = 600.0

        # 프리미엄 블리드 모니터 설정
        self._bleed_monitor_interval_sec: float = 5.0   # opt_snap 폴링 주기 (초)
        self._bleed_min_score: float = 0.3              # 전송 최소 |score|
        self._bleed_monitor_thread: Optional[threading.Thread] = None

        # OI 구조 모니터 설정 (v5 전용)
        self._oi_monitor_interval_sec: float = 30.0     # OI 폴링 주기 (초, 독립 알림용)
        self._oi_min_call_conc: float = 0.3             # 전송 최소 Call OI 집중도
        self._oi_min_put_conc: float = 0.3              # 전송 최소 Put OI 집중도
        self._oi_monitor_thread: Optional[threading.Thread] = None

        # HEARTBEAT 모니터 설정
        self._heartbeat_interval_sec: float = 3600.0   # HEARTBEAT 주기 (초, 기본 1시간)
        self._heartbeat_monitor_thread: Optional[threading.Thread] = None

        # 선물-콜 추적 이탈(CDS) 모니터 설정 (v4/v5)
        self._divergence_monitor_interval_sec: float = 0.0   # CDS 폴링 주기 (초) - 비활성화
        self._divergence_min_cds: float = 0.3                # 전송 최소 CDS
        self._divergence_window: int = 20                    # FuturesCallSimilarity window
        self._divergence_monitor_thread: Optional[threading.Thread] = None
        # 가격 히스토리 큐 (롤링 window 유지)
        self._fut_price_history: "deque[float]" = deque(maxlen=self._divergence_window)
        self._call_price_history: "deque[float]" = deque(maxlen=self._divergence_window)
        self._divergence_atm_strike: float = 0.0
        self._divergence_delta: float = 0.5
        self._divergence_lock = threading.Lock()

        # ── Trade Gate (장중 진입/청산 판단) ──
        # enabled=False(기본) 시 완전 비활성 — 기존 예측 흐름에 영향 없음.
        # 활성화하려면 TradeGateConfig(enabled=True, ...) 를 주입하거나
        # set_trade_gate_config() 를 호출한다.
        self._trade_gate: Optional[Any] = None
        self._trade_gate_strategy: str = "signal"  # 기본값: signal 기반 전략
        self._trade_monitor_thread: Optional[threading.Thread] = None
        # [FIX-PIVOT-SUMMARY] pipeline 종료 전 피봇 데이터 미리 캡처
        # stop() 호출 시점에는 pipeline이 이미 닫혀 있을 수 있으므로
        # pipeline 종료 직전(_capture_pivot_data_before_close)에 필요한 데이터를 저장한다.
        self._cached_pivot_data: Optional[Dict[str, Any]] = None
        # 기본적으로 TradeExecutionGate 초기화 (strategy="signal")
        if _TRADE_GATE_AVAILABLE and TradeExecutionGate is not None:
            try:
                self._trade_gate = TradeExecutionGate(self._notifier)
            except Exception:
                logger.warning("[BRIDGE] TradeExecutionGate 초기화 실패 — 비활성 상태로 계속")
                self._trade_gate = None

    def set_trade_gate_config(self, config: Any) -> None:
        """TradeGateConfig 또는 PivotGateConfig 를 주입하고 게이트를 재초기화한다.

        config.json 로드 후 또는 런타임 설정 변경 시 호출한다.
        config.strategy 값에 따라 TradeExecutionGate 또는 PivotExecutionGate를 선택한다.

        Args:
            config: TradeGateConfig 또는 PivotGateConfig 인스턴스.
                    strategy="signal" 이면 TradeGateConfig.from_dict() 로 생성.
                    strategy="pivot" 이면 PivotGateConfig.from_dict() 로 생성.
        """
        # strategy 설정 추출
        strategy = getattr(config, "strategy", "signal").lower()
        self._trade_gate_strategy = strategy
        
        enabled = getattr(config, "enabled", False)
        
        if strategy == "pivot":
            # 피봇 기반 전략
            if not _PIVOT_GATE_AVAILABLE or PivotExecutionGate is None:
                logger.warning("[BRIDGE] pivot_gate 모듈 없음 — set_trade_gate_config 무시")
                return
            try:
                self._trade_gate = PivotExecutionGate(self._notifier, config)
                logger.info(
                    "[BRIDGE] PivotExecutionGate 설정 완료 enabled=%s strategy=pivot",
                    enabled,
                )
            except Exception:
                logger.exception("[BRIDGE] PivotExecutionGate 재초기화 실패")
        else:
            # 신호 기반 전략 (기본)
            if not _TRADE_GATE_AVAILABLE or TradeExecutionGate is None:
                logger.warning("[BRIDGE] trade_gate 모듈 없음 — set_trade_gate_config 무시")
                return
            try:
                self._trade_gate = TradeExecutionGate(self._notifier, config)
                logger.info(
                    "[BRIDGE] TradeExecutionGate 설정 완료 enabled=%s strategy=signal",
                    enabled,
                )
            except Exception:
                logger.exception("[BRIDGE] TradeExecutionGate 재초기화 실패")

    def start(self) -> None:
        """예측 루프를 백그라운드 스레드로 시작합니다."""
        if self._run_thread and self._run_thread.is_alive():
            logger.warning("예측 루프가 이미 실행 중입니다.")
            return
        # SDK 미설치 알림을 위해 pipeline에 notifier 주입 (없으면 skip)
        try:
            if getattr(self._pipeline, "_notifier", None) is None:
                self._pipeline._notifier = self._notifier
            judge = getattr(self._pipeline, "judge", None)
            if judge is not None and getattr(judge, "_notifier", None) is None:
                judge._notifier = self._notifier
        except Exception:
            pass
        self._stop_event.clear()
        self._run_thread = threading.Thread(
            target=self._predict_loop, daemon=True, name="PipelineBridge"
        )
        self._run_thread.start()

        # v4 전용: 프리미엄 블리드 모니터 스레드 시작
        try:
            fs = str(getattr(self._pipeline, "_option_feature_set", "") or "")
            if fs == "v4":
                self._bleed_monitor_thread = threading.Thread(
                    target=self._bleed_monitor_loop, daemon=True, name="BleedMonitor"
                )
                self._bleed_monitor_thread.start()
                logger.info("[TG][BLEED] 프리미엄 블리드 모니터 시작 (간격: %.0f초)", self._bleed_monitor_interval_sec)
        except Exception as e:
            logger.warning("[TG][BLEED] 블리드 모니터 시작 실패: %s", e)

        # v5 전용: OI 구조 독립 알림 모니터 스레드 시작
        try:
            fs = str(getattr(self._pipeline, "_option_feature_set", "") or "")
            if fs == "v5":
                self._bleed_monitor_thread = threading.Thread(
                    target=self._bleed_monitor_loop, daemon=True, name="BleedMonitor"
                )
                self._bleed_monitor_thread.start()
                logger.info("[TG][BLEED] 프리미엄 블리드 모니터 시작 (v5, 간격: %.0f초)", self._bleed_monitor_interval_sec)

                self._oi_monitor_thread = threading.Thread(
                    target=self._oi_monitor_loop, daemon=True, name="OIMonitor"
                )
                self._oi_monitor_thread.start()
                logger.info("[TG][OI] OI 구조 모니터 시작 (간격: %.0f초)", self._oi_monitor_interval_sec)
        except Exception as e:
            logger.warning("[TG][OI] OI 모니터 시작 실패: %s", e)

        # HEARTBEAT 모니터 스레드 시작
        try:
            self._heartbeat_monitor_thread = threading.Thread(
                target=self._heartbeat_monitor_loop, daemon=True, name="HeartbeatMonitor"
            )
            self._heartbeat_monitor_thread.start()
            logger.info("[TG][HEARTBEAT] 하트비트 모니터 시작 (간격: %.0f초)", self._heartbeat_interval_sec)
        except Exception as e:
            logger.warning("[TG][HEARTBEAT] 하트비트 모니터 시작 실패: %s", e)

        # v4/v5: 선물-콜 추적 이탈(CDS) 모니터 스레드 시작
        try:
            fs = str(getattr(self._pipeline, "_option_feature_set", "") or "")
            if fs in ("v4", "v5") and self._divergence_monitor_interval_sec > 0:
                self._divergence_monitor_thread = threading.Thread(
                    target=self._divergence_monitor_loop, daemon=True, name="DivergenceMonitor"
                )
                self._divergence_monitor_thread.start()
                logger.info(
                    "[TG][DIV] 선물-콜 이탈 모니터 시작 (간격: %.0f초, min_cds=%.2f)",
                    self._divergence_monitor_interval_sec,
                    self._divergence_min_cds,
                )
        except Exception as e:
            logger.warning("[TG][DIV] 이탈 모니터 시작 실패: %s", e)

        # 진입/청산 감시 모니터 스레드 (TradeExecutionGate 활성 시에만)
        try:
            if self._trade_gate is not None and getattr(self._trade_gate, "enabled", False):
                self._trade_monitor_thread = threading.Thread(
                    target=self._trade_monitor_loop, daemon=True, name="TradeMonitor"
                )
                self._trade_monitor_thread.start()
                logger.info("[TG][TRADE] 진입/청산 감시 모니터 시작 (30초 주기)")
        except Exception as e:
            logger.warning("[TG][TRADE] 트레이드 모니터 시작 실패: %s", e)

        logger.info("PipelineTelegramBridge 시작 (간격: %.0f초)", self._interval)
        try:
            tp = getattr(self._pipeline, "tick_processor", None)
            if tp is not None and bool(tp.market_closed):
                return
        except Exception:
            pass

        try:
            ok = bool(self._notifier.send_text("🚀 <b>SkyEbest 예측 시스템 시작</b>", debug_context={"kind": "startup"}))
        except Exception:
            ok = False

        if not ok:
            try:
                logger.warning(
                    "[TG][STARTUP] start message not sent (configured=%s)",
                    bool(getattr(self._notifier, "is_configured", False)),
                )
            except Exception:
                pass

    def stop(self) -> None:
        """예측 루프를 정지합니다."""
        try:
            if bool(self._stop_event.is_set()):
                return
        except Exception:
            pass

        # [FIX-PIVOT-SUMMARY-1] pipeline 종료 전 피봇 데이터 캡처
        # stop() 진입 시점에 pipeline이 아직 살아 있으므로 여기서 추출한다.
        self._capture_pivot_data_before_close()

        # PivotExecutionGate 종료 정리 (요약 로그 저장 + 텔레그램 전송)
        try:
            if self._trade_gate is not None and self._trade_gate_strategy == "pivot":
                if hasattr(self._trade_gate, "shutdown"):
                    telegram_text = self._trade_gate.shutdown()
                    logger.info("[BRIDGE] PivotExecutionGate 종료 정리 완료")
                    
                    # 텔레그램으로 요약 전송
                    if telegram_text:
                        try:
                            self._notifier.send_text(telegram_text, parse_mode="Markdown")
                            logger.info("[BRIDGE] 피봇 매매 요약 텔레그램 전송 완료")
                        except Exception as e:
                            logger.warning("[BRIDGE] 피봇 매매 요약 텔레그램 전송 실패: %s", e)
        except Exception as e:
            logger.warning("[BRIDGE] PivotExecutionGate 종료 정리 실패: %s", e)

        try:
            self._stop_event.set()
        except Exception:
            pass

        # ── 장마감 피봇 요약 송출 ─────────────────────────────────
        try:
            pivot_summary, backtest_data = self._generate_daily_pivot_summary()
            if pivot_summary:
                # [FIX-PIVOT-SUMMARY-2] 네트워크 실패 시 최대 3회 재시도
                _sent = False
                for _attempt in range(3):
                    try:
                        self._notifier.send_text(pivot_summary, parse_mode="Markdown")
                        logger.info("[PIVOT_SUMMARY] 당일 피봇 요약 전송 완료 (attempt=%d)", _attempt + 1)
                        _sent = True
                        break
                    except Exception as _e:
                        logger.warning(
                            "[PIVOT_SUMMARY] 전송 실패 attempt=%d: %s", _attempt + 1, _e
                        )
                        time.sleep(2)
                if not _sent:
                    logger.error("[PIVOT_SUMMARY] 최종 전송 실패 — 피봇 요약 미전송")
                
                # 백테스트 결과 저장 (strategy가 pivot일 때만)
                if backtest_data is not None and self._trade_gate_strategy == "pivot":
                    self._save_pivot_backtest_results(backtest_data)
            else:
                logger.info("[PIVOT_SUMMARY] 피봇 요약 없음 (생성 실패 또는 피봇 없음)")
        except Exception as _ex:
            logger.warning("[PIVOT_SUMMARY] 요약 생성/전송 중 예외: %s", _ex)

        try:
            self._notifier.send_text("🛑 <b>SkyEbest 예측 시스템 종료</b>")
        except Exception:
            pass

        try:
            self._notifier.stop_polling()
        except Exception:
            pass

        # 이탈 모니터 스레드 정리
        try:
            if self._divergence_monitor_thread and self._divergence_monitor_thread.is_alive():
                self._divergence_monitor_thread.join(timeout=3.0)
        except Exception:
            pass

        # 하트비트 모니터 스레드 정리
        try:
            if self._heartbeat_monitor_thread and self._heartbeat_monitor_thread.is_alive():
                self._heartbeat_monitor_thread.join(timeout=3.0)
        except Exception:
            pass

    def _capture_pivot_data_before_close(self) -> None:
        """pipeline 종료 전에 피봇 관련 데이터를 미리 캡처한다.

        stop() 첫 줄에서 호출. pipeline이 닫힌 후 stop()의 나머지 로직이 실행되므로
        반드시 _stop_event.set() 이전에 실행해야 한다.
        
        듀얼 모드에서는 KOSPI 피봇을 매매 기준으로 사용하고, KP200 선물을 매매 종목으로 사용한다.
        """
        try:
            # 듀얼 모드: KOSPI 피봇을 매매 기준으로 사용
            azz = None
            try:
                mgr = getattr(self._pipeline, "_adaptive_mgr", None)
                # 듀얼 모드면 kospi_zigzag 사용, 아니면 기본 zigzag 사용
                if hasattr(mgr, 'kospi_zigzag') and mgr.kospi_zigzag is not None:
                    azz = mgr.kospi_zigzag
                else:
                    azz = getattr(mgr, "zigzag", None)
            except Exception:
                pass

            # 경로 2: tick_processor → heuristic → zigzag_state 경로
            zz_state = None
            all_swings = None
            bar_hhmm_fn = None

            if azz is not None:
                # AdaptiveZigZag 인스턴스에서 직접 추출
                all_swings = list(getattr(azz, "_all_swings", None) or [])
                # anchor pivot 제외 (초기화용, 실시간 신호 아님) - index==0인 swing는 anchor
                all_swings = [s for s in all_swings if s.index != 0]
                bar_hhmm_fn = getattr(azz, "_bar_hhmm", None)
                zz_state = getattr(azz, "_state", None)
            else:
                # fallback: tick_processor → heuristic 경로
                tp = getattr(self._pipeline, "tick_processor", None)
                heuristic = getattr(tp, "heuristic", None)
                zs = getattr(heuristic, "zigzag_state", None)
                all_swings_raw = getattr(zs, "_all_swings", None)
                if all_swings_raw is not None:
                    all_swings = list(all_swings_raw)
                    # anchor pivot 제외 (초기화용, 실시간 신호 아님) - index==0인 swing는 anchor
                    all_swings = [s for s in all_swings if s.index != 0]
                bar_hhmm_fn = getattr(zs, "_bar_hhmm", None)
                zz_state = zs

            if not all_swings:
                logger.info("[PIVOT_SUMMARY] 캡처 실패: _all_swings 없음")
                return

            # 선물 가격 (비율 계산용) - KP200 선물 가격
            futures_price = 0.0
            try:
                tp2 = getattr(self._pipeline, "tick_processor", None)
                t2101 = getattr(tp2, "_t2101_snapshot", {}) or {}
                futures_price = float(t2101.get("price", 0.0) or 0.0)
            except Exception:
                pass

            self._cached_pivot_data = {
                "all_swings": all_swings,
                "bar_hhmm_fn": bar_hhmm_fn,
                "zz_state": zz_state,
                "futures_price": futures_price,
            }
            logger.info(
                "[PIVOT_SUMMARY] 피봇 데이터 캡처 완료: swings=%d futures_price=%.2f",
                len(all_swings), futures_price,
            )
        except Exception as e:
            logger.warning("[PIVOT_SUMMARY] 캡처 중 예외: %s", e)

    def _generate_daily_pivot_summary(self) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """당일 확정 피봇 요약 메시지 생성.

        [FIX-PIVOT-SUMMARY-3] _capture_pivot_data_before_close()에서 미리 저장한
        캐시(_cached_pivot_data)를 우선 사용한다.
        캐시가 없으면 live pipeline 경로로 fallback 시도한다.
        """
        try:
            # ── 데이터 소스 결정 ──────────────────────────────────
            cache = self._cached_pivot_data
            if cache is not None:
                all_swings   = cache.get("all_swings") or []
                bar_hhmm_fn  = cache.get("bar_hhmm_fn")       # AdaptiveZigZag._bar_hhmm
                zz_state     = cache.get("zz_state")
                futures_price = float(cache.get("futures_price", 0.0) or 0.0)
            else:
                # fallback: pipeline이 아직 살아 있을 때
                logger.warning("[PIVOT_SUMMARY] 캐시 없음 — live pipeline 경로 시도")
                mgr = getattr(self._pipeline, "_adaptive_mgr", None)
                azz = getattr(mgr, "zigzag", None)
                if azz is not None:
                    all_swings  = list(getattr(azz, "_all_swings", None) or [])
                    # anchor pivot 제외 (초기화용, 실시간 신호 아님) - index==0인 swing는 anchor
                    all_swings = [s for s in all_swings if s.index != 0]
                    bar_hhmm_fn = getattr(azz, "_bar_hhmm", None)
                    zz_state    = getattr(azz, "_state", None)
                else:
                    tp = getattr(self._pipeline, "tick_processor", None)
                    heuristic = getattr(tp, "heuristic", None)
                    zs = getattr(heuristic, "zigzag_state", None)
                    if zs is None:
                        return None
                    all_swings  = list(getattr(zs, "_all_swings", None) or [])
                    # anchor pivot 제외 (초기화용, 실시간 신호 아님) - index==0인 swing는 anchor
                    all_swings = [s for s in all_swings if s.index != 0]
                    bar_hhmm_fn = getattr(zs, "_bar_hhmm", None)
                    zz_state    = zs
                futures_price = 0.0
                try:
                    tp2 = getattr(self._pipeline, "tick_processor", None)
                    t2101 = getattr(tp2, "_t2101_snapshot", {}) or {}
                    futures_price = float(t2101.get("price", 0.0) or 0.0)
                except Exception:
                    pass

            if not all_swings:
                return None

            # [FIX-PIVOT-SUMMARY-4] bar_hhmm_fn 정규화
            # AdaptiveZigZag._bar_hhmm(idx) 형태가 올바른 경로.
            # ZigZagState에는 이 메서드가 없어 "?"만 반환됐던 버그 수정.
            def _get_time(idx: int) -> str:
                try:
                    if callable(bar_hhmm_fn):
                        t = bar_hhmm_fn(idx)
                        if t:
                            return str(t)
                except Exception:
                    pass
                return "?"

            # ── confirmed 스윙만 필터 (09:00 anchor 제외) ──────────
            confirmed_swings = [
                sw for sw in all_swings
                if bool(getattr(sw, "confirmed", False))
                and _get_time(int(getattr(sw, "index", -1))) != "09:00"
            ]

            # ── 현재가(마지막 확정 피봇 가격) ──────────────────────
            current_index_price = 0.0
            for sw in reversed(confirmed_swings):
                p = float(getattr(sw, "price", 0.0) or 0.0)
                if p > 0:
                    current_index_price = p
                    break

            # 선물/지수 비율
            futures_index_ratio = 0.0
            if futures_price > 0 and current_index_price > 0:
                futures_index_ratio = futures_price / current_index_price

            # ── 메시지 조립 ───────────────────────────────────────
            lines = []
            lines.append("📊 *당일 피봇 요약*")
            lines.append("")
            lines.append(f"총 확정 피봇: `{len(confirmed_swings)}`개")
            if current_index_price > 0:
                lines.append(f"현재 KOSPI 지수: `{current_index_price:.2f}`")
            if futures_price > 0:
                lines.append(f"현재 KP200 선물: `{futures_price:.2f}`pt")
            lines.append("")

            # 취소된 피봇 후보
            cancelled_candidates = getattr(zz_state, "cancelled_candidates", None)
            if isinstance(cancelled_candidates, list) and cancelled_candidates:
                lines.append("🔄 *취소된 피봇 후보*")
                for cc in cancelled_candidates:
                    c_type        = str(cc.get("type", "")).upper()
                    c_time        = str(cc.get("time", "")).strip()
                    c_price       = float(cc.get("price", 0.0) or 0.0)
                    c_cancel_time = str(cc.get("cancel_time", "")).strip()
                    c_reason      = str(cc.get("reason", "")).strip()
                    lines.append(
                        f"- {c_type} `{c_time}` `{c_price:.2f}` → 취소 `{c_cancel_time}` (사유: {c_reason})"
                    )
                lines.append("")

            if not confirmed_swings:
                lines.append("📌 당일 확정 피봇 없음")
                lines.append("")
                lines.append("💰 *백테스팅 시뮬레이션*")
                lines.append("- 거래 없음")
                backtest_data = {
                    "date": self._now_fn().strftime("%Y-%m-%d"),
                    "total_pivots": 0,
                    "confirmed_pivots": 0,
                    "total_trades": 0,
                    "win_trades": 0,
                    "loss_trades": 0,
                    "stop_loss_trades": 0,
                    "win_rate": 0.0,
                    "total_profit": 0.0,
                    "avg_profit": 0.0,
                    "trades": []
                }
                return "\n".join(lines), backtest_data

            # ── 피봇 확정 지연 통계 ───────────────────────────────
            lag_times = []
            for sw in confirmed_swings:
                confirmed_at_idx = int(getattr(sw, "confirmed_at_idx", -1) or -1)
                pivot_idx        = int(getattr(sw, "index", -1) or -1)
                if confirmed_at_idx >= 0 and pivot_idx >= 0:
                    lag_times.append(confirmed_at_idx - pivot_idx)

            if lag_times:
                lines.append("⏱️ *피봇 확정 지연 시간*")
                lines.append(f"- 최소: `{min(lag_times)}`분")
                lines.append(f"- 최대: `{max(lag_times)}`분")
                lines.append(f"- 평균: `{sum(lag_times)/len(lag_times):.1f}`분")

            # ── 피봇 목록 ─────────────────────────────────────────
            for i, sw in enumerate(confirmed_swings, start=1):
                st_raw      = str(getattr(sw, "swing_type", ""))
                swing_type  = "고점" if "HIGH" in st_raw.upper() else "저점"
                index_price = float(getattr(sw, "price", 0.0) or 0.0)
                fut_price   = index_price * futures_index_ratio if futures_index_ratio > 0 else index_price
                sw_time     = _get_time(int(getattr(sw, "index", -1)))
                emoji       = "🔺" if swing_type == "고점" else "🔻"
                lines.append(
                    f"{i}. {emoji} {sw_time} {swing_type} 선물:`{fut_price:.2f}` 지수:`{index_price:.2f}`"
                )

            lines.append("")
            lines.append("📈 *파동 통계*")
            high_count = sum(1 for sw in confirmed_swings if "HIGH" in str(getattr(sw, "swing_type", "")).upper())
            low_count  = len(confirmed_swings) - high_count
            lines.append(f"- 고점: `{high_count}`개")
            lines.append(f"- 저점: `{low_count}`개")

            # ── 백테스팅 시뮬레이션 ───────────────────────────────
            lines.append("")
            lines.append("💰 *백테스팅 시뮬레이션*")
            lines.append("(저점 매수 → 고점 청산, 고점 매도 → 저점 청산)")
            lines.append("(진입/청산: 피봇 확정 봉 종가, 손절 기준: 이전 피봇 확정 봉 종가)")
            lines.append("(참고: 현실적으로 다음 피봇 확정 시점에 청산 가능)")
            lines.append("(참고: 피봇 확정은 confirmation_bars만큼 지연이 최소이며, 보통은 그 이상 지연됨)")
            lines.append("(매매 기준: KOSPI 피봇, 수익: KOSPI 지수 포인트)")
            lines.append("*참고: 백테스트는 과거 데이터 기반으로 실제 선물 가격 아님*")
            lines.append("")

            total_trades = win_trades = loss_trades = stop_loss_trades = 0
            total_profit = 0.0
            trade_details = []  # 개별 거래 내역

            for i in range(len(confirmed_swings) - 1):
                cur  = confirmed_swings[i]
                nxt  = confirmed_swings[i + 1]
                ct   = str(getattr(cur, "swing_type", "")).upper()
                nt   = str(getattr(nxt, "swing_type", "")).upper()

                # 진입가: 피봇 확정 봉 종가 우선, 없으면 피봇가 (KOSPI 지수 포인트)
                # 참고: confirmation_bars만큼 지연이 최소이며, 보통은 그 이상 지연되어야 피봇 확정 가능
                c_close = float(getattr(cur, "confirmed_close", 0.0) or 0.0)
                n_close = float(getattr(nxt, "confirmed_close", 0.0) or 0.0)
                c_entry = (c_close if c_close > 0 else float(getattr(cur, "price", 0.0) or 0.0))
                n_exit  = (n_close if n_close > 0 else float(getattr(nxt, "price", 0.0) or 0.0))

                c_time = _get_time(int(getattr(cur, "confirmed_at_idx", -1) or -1))
                n_time = _get_time(int(getattr(nxt, "confirmed_at_idx", -1) or -1))
                
                # 이전 피봇 확정 봉 종가를 손절 기준으로 설정 (KOSPI 지수 포인트)
                stop_loss_price = None
                if i > 0:
                    prev = confirmed_swings[i - 1]
                    prev_close = float(getattr(prev, "confirmed_close", 0.0) or 0.0)
                    prev_price = (prev_close if prev_close > 0 else float(getattr(prev, "price", 0.0) or 0.0))
                    
                    if "LOW" in ct:
                        # 저점 매수: 이전 피봇 확정 봉 종가를 손절 기준으로 설정 (가격이 더 떨어지면 손절)
                        stop_loss_price = prev_price
                    elif "HIGH" in ct:
                        # 고점 매도: 이전 피봇 확정 봉 종가를 손절 기준으로 설정 (가격이 더 오르면 손절)
                        stop_loss_price = prev_price

                if "LOW" in ct and "HIGH" in nt:
                    # 저점 매수 → 고점 청산
                    profit = n_exit - c_entry
                    total_trades += 1; total_profit += profit
                    
                    # 손절 체크
                    if stop_loss_price is not None and c_entry < stop_loss_price:
                        # 손절 발생 (진입가가 손절 기준보다 낮음)
                        loss = stop_loss_price - c_entry
                        total_profit -= loss
                        stop_loss_trades += 1
                        lines.append(f"⚠️ {c_time} 매수 `{c_entry:.2f}` → 손절 `{stop_loss_price:.2f}` | 손실: `{loss:.2f}`pt")
                        trade_details.append({
                            "type": "BUY",
                            "entry_time": c_time,
                            "entry_price": c_entry,
                            "exit_time": c_time,
                            "exit_price": stop_loss_price,
                            "profit": -loss,
                            "result": "STOP_LOSS",
                            "exit_reason": "STOP_LOSS"
                        })
                    else:
                        if profit > 0:
                            win_trades += 1
                            lines.append(f"✅ {c_time} 매수 `{c_entry:.2f}` → {n_time} 청산 `{n_exit:.2f}` | 수익: `+{profit:.2f}`pt")
                            trade_details.append({
                                "type": "BUY",
                                "entry_time": c_time,
                                "entry_price": c_entry,
                                "exit_time": n_time,
                                "exit_price": n_exit,
                                "profit": profit,
                                "result": "WIN",
                                "exit_reason": "TARGET"
                            })
                        else:
                            loss_trades += 1
                            lines.append(f"❌ {c_time} 매수 `{c_entry:.2f}` → {n_time} 청산 `{n_exit:.2f}` | 손실: `{profit:.2f}`pt")
                            trade_details.append({
                                "type": "BUY",
                                "entry_time": c_time,
                                "entry_price": c_entry,
                                "exit_time": n_time,
                                "exit_price": n_exit,
                                "profit": profit,
                                "result": "LOSS",
                                "exit_reason": "TARGET"
                            })

                elif "HIGH" in ct and "LOW" in nt:
                    # 고점 매도 → 저점 청산
                    profit = c_entry - n_exit
                    total_trades += 1; total_profit += profit
                    
                    # 손절 체크
                    if stop_loss_price is not None and c_entry > stop_loss_price:
                        # 손절 발생 (진입가가 손절 기준보다 높음)
                        loss = c_entry - stop_loss_price
                        total_profit -= loss
                        stop_loss_trades += 1
                        lines.append(f"⚠️ {c_time} 매도 `{c_entry:.2f}` → 손절 `{stop_loss_price:.2f}` | 손실: `{loss:.2f}`pt")
                        trade_details.append({
                            "type": "SELL",
                            "entry_time": c_time,
                            "entry_price": c_entry,
                            "exit_time": c_time,
                            "exit_price": stop_loss_price,
                            "profit": -loss,
                            "result": "STOP_LOSS",
                            "exit_reason": "STOP_LOSS"
                        })
                    else:
                        if profit > 0:
                            win_trades += 1
                            lines.append(f"✅ {c_time} 매도 `{c_entry:.2f}` → {n_time} 청산 `{n_exit:.2f}` | 수익: `+{profit:.2f}`pt")
                            trade_details.append({
                                "type": "SELL",
                                "entry_time": c_time,
                                "entry_price": c_entry,
                                "exit_time": n_time,
                                "exit_price": n_exit,
                                "profit": profit,
                                "result": "WIN",
                                "exit_reason": "TARGET"
                            })
                        else:
                            loss_trades += 1
                            lines.append(f"❌ {c_time} 매도 `{c_entry:.2f}` → {n_time} 청산 `{n_exit:.2f}` | 손실: `{profit:.2f}`pt")
                            trade_details.append({
                                "type": "SELL",
                                "entry_time": c_time,
                                "entry_price": c_entry,
                                "exit_time": n_time,
                                "exit_price": n_exit,
                                "profit": profit,
                                "result": "LOSS",
                                "exit_reason": "TARGET"
                            })

            if total_trades > 0:
                lines.append("")
                lines.append("📊 *시뮬레이션 결과*")
                lines.append(f"- 총 거래: `{total_trades}`건")
                lines.append(f"- 승리: `{win_trades}`건 | 패배: `{loss_trades}`건 | 손절: `{stop_loss_trades}`건")
                lines.append(f"- 승률: `{win_trades/total_trades*100:.1f}%`")
                lines.append(f"- 총 수익: `{total_profit:+.2f}`pt (KOSPI 지수)")
                lines.append(f"- 평균 수익: `{total_profit/total_trades:+.2f}`pt (KOSPI 지수)")
            else:
                lines.append("- 거래 없음")

            # 백테스트 데이터 구성
            backtest_data = {
                "date": self._now_fn().strftime("%Y-%m-%d"),
                "total_pivots": len(all_swings),
                "confirmed_pivots": len(confirmed_swings),
                "high_count": high_count if total_trades > 0 else 0,
                "low_count": low_count if total_trades > 0 else 0,
                "current_index_price": current_index_price,
                "futures_price": futures_price,
                "futures_index_ratio": futures_index_ratio,
                "total_trades": total_trades,
                "win_trades": win_trades,
                "loss_trades": loss_trades,
                "stop_loss_trades": stop_loss_trades,
                "win_rate": win_trades/total_trades*100 if total_trades > 0 else 0.0,
                "total_profit": total_profit,
                "avg_profit": total_profit/total_trades if total_trades > 0 else 0.0,
                "trades": trade_details
            }

            return "\n".join(lines), backtest_data

        except Exception as e:
            logger.warning("[PIVOT_SUMMARY] 요약 생성 실패: %s", e)
            return None, None

    def _save_pivot_backtest_results(self, pivot_summary_data: Dict[str, Any]) -> None:
        """피봇 기반 백테스트 결과를 JSON 파일로 저장한다.
        
        Args:
            pivot_summary_data: 백테스트 결과 데이터 딕셔너리
        """
        try:
            import json
            import os
            from datetime import datetime
            import math
            
            # 추가 지표 계산
            trades = pivot_summary_data.get("trades", [])
            if trades:
                # 최대 드로우다운 (MDD) 계산
                equity_curve = []
                cumulative_profit = 0.0
                max_equity = 0.0
                max_drawdown = 0.0
                
                for trade in trades:
                    cumulative_profit += trade.get("profit", 0.0)
                    equity_curve.append(cumulative_profit)
                    max_equity = max(max_equity, cumulative_profit)
                    drawdown = (cumulative_profit - max_equity) / max_equity if max_equity > 0 else 0.0
                    max_drawdown = min(max_drawdown, drawdown)
                
                pivot_summary_data["max_drawdown"] = max_drawdown * 100 if max_equity > 0 else 0.0
                
                # 샤프 비율 계산 (단순화: 연환산 없이 일일 기준)
                profits = [t.get("profit", 0.0) for t in trades]
                if len(profits) > 1:
                    avg_profit = sum(profits) / len(profits)
                    std_profit = math.sqrt(sum((p - avg_profit) ** 2 for p in profits) / len(profits))
                    sharpe_ratio = avg_profit / std_profit if std_profit > 0 else 0.0
                    pivot_summary_data["sharpe_ratio"] = sharpe_ratio
                else:
                    pivot_summary_data["sharpe_ratio"] = 0.0
                
                # 평균 보유 시간 계산
                hold_times = []
                for trade in trades:
                    entry_time = trade.get("entry_time", "")
                    exit_time = trade.get("exit_time", "")
                    if entry_time and exit_time:
                        try:
                            # HH:MM 형식 가정
                            h1, m1 = map(int, entry_time.split(":"))
                            h2, m2 = map(int, exit_time.split(":"))
                            hold_minutes = (h2 * 60 + m2) - (h1 * 60 + m1)
                            hold_times.append(hold_minutes)
                        except Exception:
                            pass
                avg_hold_minutes = sum(hold_times) / len(hold_times) if hold_times else 0.0
                pivot_summary_data["avg_hold_minutes"] = avg_hold_minutes
                
                # 손익비 (Profit Factor) 계산
                total_profit = sum(t.get("profit", 0.0) for t in trades if t.get("profit", 0.0) > 0)
                total_loss = abs(sum(t.get("profit", 0.0) for t in trades if t.get("profit", 0.0) < 0))
                profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
                pivot_summary_data["profit_factor"] = profit_factor
                
                # 최대 연속 손실 횟수
                max_consecutive_losses = 0
                current_consecutive_losses = 0
                for trade in trades:
                    if trade.get("result") == "LOSS" or trade.get("result") == "STOP_LOSS":
                        current_consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
                    else:
                        current_consecutive_losses = 0
                pivot_summary_data["max_consecutive_losses"] = max_consecutive_losses
            else:
                # 거래 없는 경우
                pivot_summary_data["max_drawdown"] = 0.0
                pivot_summary_data["sharpe_ratio"] = 0.0
                pivot_summary_data["avg_hold_minutes"] = 0.0
                pivot_summary_data["profit_factor"] = 0.0
                pivot_summary_data["max_consecutive_losses"] = 0
            
            # 저장 경로 결정
            history_dir = "trade_history"
            try:
                # config에서 trade_gate.history_dir 읽기 시도
                if hasattr(self._pipeline, "_config"):
                    tg_cfg = self._pipeline._config.get("trade_gate", {})
                    history_dir = tg_cfg.get("history_dir", "trade_history")
            except Exception:
                pass
            
            # 디렉토리 생성
            os.makedirs(history_dir, exist_ok=True)
            
            # 파일명: pivot_backtest_YYYYMMDD.json
            today = self._now_fn().strftime("%Y%m%d")
            filename = f"pivot_backtest_{today}.json"
            filepath = os.path.join(history_dir, filename)
            
            # JSON 저장
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(pivot_summary_data, f, ensure_ascii=False, indent=2)
            
            logger.info("[PIVOT_BACKTEST] 백테스트 결과 저장 완료: %s", filepath)
        except Exception as e:
            logger.warning("[PIVOT_BACKTEST] 백테스트 결과 저장 실패: %s", e)

    def start_polling(self) -> None:
        """텔레그램 명령 수신 폴링도 시작합니다."""
        self._notifier.start_polling(on_command=self._handle_command)

    def predict_now(
        self,
        force: bool = False,
        include_dir_summary: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """즉시 예측을 실행하고 결과를 텔레그램으로 전송합니다."""
        try:
            tp = getattr(self._pipeline, "tick_processor", None)
            if tp is not None and bool(tp.market_closed):
                return None
        except Exception:
            pass
        try:
            result = self._pipeline.get_prediction()
            self._last_result = result
            if "error" in result:
                return result
            include_summary = include_dir_summary if include_dir_summary is not None else self._is_boundary_tick(result)
            # symbol 정보 추출
            symbol = ""
            try:
                ad = getattr(self._pipeline, "_adaptive_indicator", {})
                symbol = str(ad.get("symbol", "") or "")
            except Exception:
                pass
            self._notifier.send_prediction(result, force=force, include_dir_summary=bool(include_summary), symbol=symbol)
            return result
        except Exception as exc:
            logger.error("즉시 예측 실패: %s", exc)
            try:
                tp = getattr(self._pipeline, "tick_processor", None)
                if tp is not None and bool(tp.market_closed):
                    return None
            except Exception:
                pass
            self._notifier.send_text(f"❌ 예측 실패: {exc}")
            return None


# ── module-level helpers ─────────────────────────────────────────────────────
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
            if telegram.get("proxy_url"):
                cfg["proxy_url"] = str(telegram["proxy_url"])
            if telegram.get("http_timeout") is not None:
                cfg["http_timeout"] = float(telegram["http_timeout"])
        else:
            logger.debug("secrets 파일 없음: %s", path)
    except Exception as exc:
        logger.warning("secrets 파일 읽기 실패: %s", exc)

    # 환경변수가 파일보다 우선
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        cfg["bot_token"] = os.environ["TELEGRAM_BOT_TOKEN"]
    if os.environ.get("TELEGRAM_CHAT_ID"):
        cfg["chat_id"] = os.environ["TELEGRAM_CHAT_ID"]
    if os.environ.get("TELEGRAM_PROXY_URL"):
        cfg["proxy_url"] = os.environ["TELEGRAM_PROXY_URL"]

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
        proxy_url=cfg.get("proxy_url") or None,
        timeout=float(cfg.get("http_timeout", 30.0) or 30.0),
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
        proxy_url=cfg.get("proxy_url") or None,
        timeout=float(cfg.get("http_timeout", 30.0) or 30.0),
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
