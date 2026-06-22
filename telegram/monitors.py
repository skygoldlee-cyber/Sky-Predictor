"""PipelineTelegramBridge 분리 모듈.

이 파일은 telegram_notifier.py에서 분리된 Mixin 클래스입니다.
직접 인스턴스화하지 마세요.
"""
from __future__ import annotations
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MonitorsMixin:
    """모니터링/예측 루프 Mixin.

    루프 목록:
        _bleed_monitor_loop      — 프리미엄 블리드 감시
        _divergence_monitor_loop — 선물-콜 다이버전스 감시
        _trade_monitor_loop      — 트레이드 게이트 상태 감시
        _oi_monitor_loop         — OI 지지저항 감시
        _predict_loop            — 정기 예측 루프
    """

    def _bleed_monitor_loop(self) -> None:
        """v4 전용: 프리미엄 블리드 신호를 주기적으로 폴링하여 텔레그램 독립 알림 전송.

        _predict_loop와 독립적으로 동작하며 _bleed_monitor_interval_sec 주기로 실행된다.
        opt_snap은 pipeline._build_option_snapshot_safe()를 통해 직접 조회한다.
        """
        logger.info("[TG][BLEED] 모니터 루프 시작")
        while not self._stop_event.is_set():
            try:
                # 장 종료 중이거나 일시정지이면 건너뜀
                try:
                    tp = getattr(self._pipeline, "tick_processor", None)
                    if tp is not None and bool(tp.market_closed):
                        self._stop_event.wait(timeout=self._bleed_monitor_interval_sec)
                        continue
                except Exception:
                    pass

                if self._user_pause_event.is_set() or self._market_paused:
                    self._stop_event.wait(timeout=self._bleed_monitor_interval_sec)
                    continue

                # 현재 선물가 조회
                current_price = 0.0
                try:
                    tp = getattr(self._pipeline, "tick_processor", None)
                    if tp is not None:
                        px = getattr(tp, "latest_future_price", None)
                        if callable(px):
                            current_price = float(px() or 0.0)
                        elif px is not None:
                            current_price = float(px or 0.0)
                except Exception:
                    pass

                if current_price <= 0.0:
                    self._stop_event.wait(timeout=self._bleed_monitor_interval_sec)
                    continue

                # opt_snap 조회 (update_prev=False: 블리드 모니터는 상태 갱신 안 함)
                opt_snap: Dict[str, Any] = {}
                try:
                    build_fn = getattr(self._pipeline, "_build_option_snapshot_safe", None)
                    if callable(build_fn):
                        opt_snap = dict(build_fn(current_price=float(current_price), update_prev=False) or {})
                except Exception as e:
                    logger.debug("[TG][BLEED] opt_snap 조회 실패: %s", e)
                    self._stop_event.wait(timeout=self._bleed_monitor_interval_sec)
                    continue

                if not opt_snap:
                    self._stop_event.wait(timeout=self._bleed_monitor_interval_sec)
                    continue

                # 만기 잔존일: 직접값(days_to_expiry) 우선, 없을 때만 역산
                dte_days: Optional[float] = None
                try:
                    _direct = opt_snap.get("days_to_expiry")
                    if _direct is not None:
                        dte_days = float(_direct)
                    else:
                        dte_w = float(opt_snap.get("dte_weight_norm") or 0.0)
                        if dte_w > 0.0:
                            dte_days = 1.0 / (dte_w * 10.0)
                except Exception:
                    pass

                # 알림 전송 (쿨다운 / 점수 필터는 send_premium_bleed_alert 내부에서 처리)
                try:
                    self._notifier.send_premium_bleed_alert(
                        opt_snap,
                        current_price,
                        dte_days=dte_days,
                        min_score=float(self._bleed_min_score),
                    )
                except Exception as e:
                    logger.debug("[TG][BLEED] 알림 전송 실패: %s", e)

                # 옵션 가격 레벨 터치 알림
                # opt_snap에 '_price_level_scan' 키가 있으면 터치 여부를 검사하고
                # 감지 시 독립 알림을 전송한다. 쿨다운 필터는 메서드 내부에서 처리.
                try:
                    self._notifier.send_price_level_touch_alert(
                        opt_snap,
                        current_price,
                    )
                except Exception as e:
                    logger.debug("[TG][LEVEL] 레벨 터치 알림 전송 실패: %s", e)

            except Exception as exc:
                logger.warning("[TG][BLEED] 모니터 루프 오류: %s", exc)

            self._stop_event.wait(timeout=self._bleed_monitor_interval_sec)

        logger.info("[TG][BLEED] 모니터 루프 종료")

    def _divergence_monitor_loop(self) -> None:
        """v4/v5 전용: 선물-ATM 콜 추적 이탈(CDS)을 주기적으로 측정하고 텔레그램 알림 전송.

        동작 방식:
        1. _divergence_monitor_interval_sec 주기로 현재 선물가와 ATM 콜 가격을 수집한다.
        2. 가격을 롤링 큐(_fut_price_history, _call_price_history)에 누적한다.
        3. window(기본 20) 이상 샘플이 모이면 FuturesCallSimilarity로 CDS를 계산한다.
        4. CDS >= _divergence_min_cds 이고 쿨다운이 지났으면 텔레그램 알림을 전송한다.

        가격 수집 전략:
        - 선물가: tick_processor.latest_future_price
        - ATM 콜가: build_option_snapshot_safe 대신 opt_snap 캐시에서 직접 추출.
          _prev_atm_call_price 를 사용하면 snapshot 재계산 없이 현재 ATM 콜가를 얻는다.
          없으면 _build_option_snapshot_safe(update_prev=False) fallback.
        """
        from prediction.option_features import FuturesCallSimilarity

        sim = FuturesCallSimilarity(window=self._divergence_window)
        logger.info("[TG][DIV] 이탈 모니터 루프 시작 (window=%d)", self._divergence_window)

        while not self._stop_event.is_set():
            try:
                # 장 종료 / 일시정지 건너뜀
                try:
                    tp = getattr(self._pipeline, "tick_processor", None)
                    if tp is not None and bool(tp.market_closed):
                        self._stop_event.wait(timeout=self._divergence_monitor_interval_sec)
                        continue
                except Exception:
                    pass

                if self._user_pause_event.is_set() or self._market_paused:
                    self._stop_event.wait(timeout=self._divergence_monitor_interval_sec)
                    continue

                # ── 선물가 수집 ────────────────────────────────────────────────
                current_price = 0.0
                try:
                    tp = getattr(self._pipeline, "tick_processor", None)
                    if tp is not None:
                        px = getattr(tp, "latest_future_price", None)
                        current_price = float(px() if callable(px) else (px or 0.0))
                except Exception:
                    pass

                if current_price <= 0.0:
                    self._stop_event.wait(timeout=self._divergence_monitor_interval_sec)
                    continue

                # ── ATM 콜가 수집 ──────────────────────────────────────────────
                # 1안: pipeline._prev_atm_call_price 직접 참조 (snapshot 재계산 없음)
                atm_call_price = 0.0
                atm_strike = 0.0
                delta = 0.5
                try:
                    atm_call_price = float(
                        getattr(self._pipeline, "_prev_atm_call_price", None) or 0.0
                    )
                except Exception:
                    pass

                # 2안: opt_snap 재계산 fallback
                if atm_call_price <= 0.0:
                    try:
                        build_fn = getattr(self._pipeline, "_build_option_snapshot_safe", None)
                        if callable(build_fn):
                            snap = dict(build_fn(current_price=current_price, update_prev=False) or {})
                            # straddle_now = C+P, call_delta_proxy = C/(C+P)
                            straddle = float(snap.get("straddle_now") or snap.get("straddle_price") or 0.0)
                            cdp = float(snap.get("call_delta_proxy") or 0.5)
                            if straddle > 0.0:
                                atm_call_price = straddle * cdp
                            atm_strike_raw = float(snap.get("atm_strike") or 0.0)
                            if atm_strike_raw > 0.0:
                                atm_strike = atm_strike_raw
                    except Exception:
                        pass

                # ATM 행사가 보완 (pipeline 캐시에서 직접)
                if atm_strike <= 0.0:
                    try:
                        atm_strike = float(
                            getattr(self._pipeline, "_atm_strike", None) or 0.0
                        )
                    except Exception:
                        pass
                if atm_strike <= 0.0:
                    atm_strike = current_price  # fallback: 현재가 근사

                if atm_call_price <= 0.0:
                    self._stop_event.wait(timeout=self._divergence_monitor_interval_sec)
                    continue

                # ── 가격 히스토리 누적 ──────────────────────────────────────────
                with self._divergence_lock:
                    self._fut_price_history.append(float(current_price))
                    self._call_price_history.append(float(atm_call_price))
                    self._divergence_atm_strike = float(atm_strike)
                    fut_arr  = list(self._fut_price_history)
                    call_arr = list(self._call_price_history)

                n_samp = min(len(fut_arr), len(call_arr))
                if n_samp < max(self._divergence_window // 2, 5):
                    # 샘플 부족: 경고 없이 대기
                    self._stop_event.wait(timeout=self._divergence_monitor_interval_sec)
                    continue

                # ── CDS 계산 ────────────────────────────────────────────────────
                try:
                    import numpy as _np
                    cds_result = sim.composite_divergence_score(
                        _np.array(fut_arr, dtype=float),
                        _np.array(call_arr, dtype=float),
                        delta=float(delta),
                    )
                except Exception as e:
                    logger.debug("[TG][DIV] CDS 계산 실패: %s", e)
                    self._stop_event.wait(timeout=self._divergence_monitor_interval_sec)
                    continue

                # ── DTE 역산 ────────────────────────────────────────────────────
                dte_days: Optional[float] = None
                try:
                    from core.utils import get_expiry_week_info
                    dte_days = float(get_expiry_week_info().get("days_to_expiry") or 0.0) or None
                except Exception:
                    pass

                # ── 알림 전송 (쿨다운·점수 필터는 send 내부 처리) ────────────────
                try:
                    self._notifier.send_futures_call_divergence_alert(
                        cds_result,
                        float(current_price),
                        float(atm_strike),
                        dte_days=dte_days,
                        min_cds=float(self._divergence_min_cds),
                    )
                except Exception as e:
                    logger.debug("[TG][DIV] 알림 전송 실패: %s", e)

            except Exception as exc:
                logger.warning("[TG][DIV] 모니터 루프 오류: %s", exc)

            self._stop_event.wait(timeout=self._divergence_monitor_interval_sec)

        logger.info("[TG][DIV] 이탈 모니터 루프 종료")

    def _trade_monitor_loop(self) -> None:
        """TradeExecutionGate 또는 PivotExecutionGate 청산 조건을 30초 주기로 점검한다.

        _predict_loop 와 독립적으로 동작한다.
        strategy="signal" 이면 check_close, strategy="pivot" 이면 check_price_breakout 및 피봇 확정 이벤트를 처리한다.
        """
        logger.info("[TG][TRADE] 트레이드 모니터 루프 시작")
        # 피봇 확정 추적용 마지막 확인 인덱스
        last_pivot_check_idx = -1
        
        while not self._stop_event.is_set():
            try:
                if self._trade_gate is not None:
                    try:
                        tp = getattr(self._pipeline, "tick_processor", None)
                        if tp is not None and bool(getattr(tp, "market_closed", False)):
                            # 장 종료 — 강제청산 조건은 각 게이트 내부에서 처리
                            pass
                        price = float(tp.get_current_price() if tp is not None else 0.0)
                        
                        # strategy에 따라 다른 메서드 호출
                        strategy = getattr(self, "_trade_gate_strategy", "signal")
                        
                        if strategy == "pivot":
                            # 피봇 기반 전략: 가격 돌파 체크
                            if hasattr(self._trade_gate, "check_price_breakout"):
                                self._trade_gate.check_price_breakout(current_price=price)
                            
                            # 피봇 확정 이벤트 감지
                            pivot_info = self._get_latest_pivot_confirmation()
                            if pivot_info is not None:
                                pivot_idx = pivot_info.get("confirmed_at_idx", -1)
                                if pivot_idx > last_pivot_check_idx:
                                    last_pivot_check_idx = pivot_idx
                                    pivot_type = pivot_info.get("pivot_type", "")
                                    pivot_price = float(pivot_info.get("pivot_price", 0.0) or 0.0)
                                    if pivot_price > 0 and hasattr(self._trade_gate, "on_pivot_confirmed"):
                                        self._trade_gate.on_pivot_confirmed(
                                            pivot_type=pivot_type,
                                            pivot_price=pivot_price,
                                            current_price=price,
                                        )
                                        logger.info(
                                            "[TG][TRADE] 피봇 확정 이벤트 전달: type=%s price=%.2f",
                                            pivot_type, pivot_price,
                                        )
                        else:
                            # 신호 기반 전략: 기존 check_close 호출
                            self._trade_gate.check_close(current_price=price)
                    except Exception:
                        logger.debug("[TG][TRADE] 게이트 체크 오류", exc_info=True)
            except Exception:
                pass
            # 가격 체크 주기: 5초 (급격한 가격 변동 대응)
            self._stop_event.wait(timeout=5.0)
        logger.info("[TG][TRADE] 트레이드 모니터 루프 종료")
    
    def _get_latest_pivot_confirmation(self) -> Optional[Dict[str, Any]]:
        """최근 피봇 확정 정보를 추출한다.
        
        Returns:
            {
                "pivot_type": "HIGH" or "LOW",
                "pivot_price": float,
                "confirmed_at_idx": int,
            } or None
        """
        try:
            mgr = getattr(self._pipeline, "_adaptive_mgr", None)
            if mgr is None:
                return None
            
            # 듀얼 모드: KOSPI 피봇을 매매 기준으로 사용
            azz = None
            if hasattr(mgr, 'kospi_zigzag') and mgr.kospi_zigzag is not None:
                azz = mgr.kospi_zigzag
            else:
                azz = getattr(mgr, "zigzag", None)
            
            if azz is None:
                return None
            
            all_swings = list(getattr(azz, "_all_swings", None) or [])
            if not all_swings:
                return None
            
            # anchor pivot 제외 (index==0인 swing는 anchor)
            confirmed_swings = [s for s in all_swings if s.index != 0 and getattr(s, "confirmed", False)]
            
            if not confirmed_swings:
                return None
            
            # 가장 최근 확정 피봇
            latest = confirmed_swings[-1]
            swing_type = str(getattr(latest, "swing_type", "")).upper()
            pivot_price = float(getattr(latest, "price", 0.0) or 0.0)
            confirmed_at_idx = int(getattr(latest, "confirmed_at_idx", -1) or -1)
            
            if pivot_price <= 0 or confirmed_at_idx < 0:
                return None
            
            return {
                "pivot_type": swing_type,
                "pivot_price": pivot_price,
                "confirmed_at_idx": confirmed_at_idx,
            }
        except Exception:
            logger.debug("[TG][TRADE] 피봇 확정 정보 추출 실패", exc_info=True)
            return None

    def _oi_monitor_loop(self) -> None:
        """v5 전용: OI 구조 변화를 주기적으로 폴링하여 텔레그램 독립 알림 전송.

        _predict_loop와 독립적으로 동작하며 _oi_monitor_interval_sec 주기로 실행된다.
        OI 집중도(call_oi_peak_norm / put_oi_peak_norm)가 임계값 이상이고
        쿨다운을 벗어났을 때만 send_oi_structure_alert()를 호출한다.

        전송 조건 (send_oi_structure_alert 내부에서 최종 판단):
            - call_oi_peak_norm >= _oi_min_call_conc  또는
              put_oi_peak_norm  >= _oi_min_put_conc
            - 쿨다운 600초 (TelegramNotifier._oi_alert_cooldown_sec)
        """
        logger.info("[TG][OI] OI 모니터 루프 시작")
        while not self._stop_event.is_set():
            try:
                # 장 종료 또는 일시정지 시 건너뜀
                try:
                    tp = getattr(self._pipeline, "tick_processor", None)
                    if tp is not None and bool(tp.market_closed):
                        self._stop_event.wait(timeout=self._oi_monitor_interval_sec)
                        continue
                except Exception:
                    pass

                if self._user_pause_event.is_set() or self._market_paused:
                    self._stop_event.wait(timeout=self._oi_monitor_interval_sec)
                    continue

                # 현재 선물가 조회
                current_price = 0.0
                try:
                    tp = getattr(self._pipeline, "tick_processor", None)
                    if tp is not None:
                        px = getattr(tp, "latest_future_price", None)
                        if callable(px):
                            current_price = float(px() or 0.0)
                        elif px is not None:
                            current_price = float(px or 0.0)
                except Exception:
                    pass

                if current_price <= 0.0:
                    self._stop_event.wait(timeout=self._oi_monitor_interval_sec)
                    continue

                # opt_snap 조회 (update_prev=False: 모니터는 상태 갱신 안 함)
                opt_snap: Dict[str, Any] = {}
                try:
                    build_fn = getattr(self._pipeline, "_build_option_snapshot_safe", None)
                    if callable(build_fn):
                        opt_snap = dict(
                            build_fn(current_price=float(current_price), update_prev=False) or {}
                        )
                except Exception as e:
                    logger.debug("[TG][OI] opt_snap 조회 실패: %s", e)
                    self._stop_event.wait(timeout=self._oi_monitor_interval_sec)
                    continue

                if not opt_snap:
                    self._stop_event.wait(timeout=self._oi_monitor_interval_sec)
                    continue

                # OI 구조 알림 전송 (쿨다운/집중도 필터는 내부에서 처리)
                try:
                    self._notifier.send_oi_structure_alert(
                        opt_snap,
                        current_price,
                        min_call_conc=float(self._oi_min_call_conc),
                        min_put_conc=float(self._oi_min_put_conc),
                    )
                except Exception as e:
                    logger.debug("[TG][OI] OI 알림 전송 실패: %s", e)

            except Exception as exc:
                logger.warning("[TG][OI] OI 모니터 루프 오류: %s", exc)

            self._stop_event.wait(timeout=self._oi_monitor_interval_sec)

        logger.info("[TG][OI] OI 모니터 루프 종료")

    def _heartbeat_monitor_loop(self) -> None:
        """1시간마다 HEARTBEAT 메시지를 텔레그램으로 전송.
        
        시스템 상태를 확인하고 정상 작동 중임을 알리는 하트비트 메시지를 전송한다.
        """
        logger.info("[TG][HEARTBEAT] 하트비트 모니터 루프 시작 (간격: %.0f초)", self._heartbeat_interval_sec)
        
        while not self._stop_event.is_set():
            try:
                # 하트비트 메시지 전송
                try:
                    from datetime import datetime as dt
                    
                    # 시스템 상태 수집
                    status_info = []
                    
                    # 현재 시간
                    status_info.append(f"⏰ 시간: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # 포지션 상태
                    try:
                        if hasattr(self, '_pipeline') and self._pipeline:
                            pos_tracker = getattr(self._pipeline, 'position_tracker', None)
                            if pos_tracker:
                                position = pos_tracker.get_position()
                                if position:
                                    status_info.append(f"📊 포지션: {position.get('side', 'N/A')} @ {position.get('entry_price', 'N/A')}")
                                else:
                                    status_info.append("📊 포지션: 없음")
                    except Exception:
                        status_info.append("📊 포지션: 확인 불가")
                    
                    # 리스크 상태
                    try:
                        if hasattr(self, '_pipeline') and self._pipeline:
                            tp = getattr(self._pipeline, 'tick_processor', None)
                            if tp:
                                current_price = 0.0
                                try:
                                    px = getattr(tp, 'latest_future_price', None)
                                    if callable(px):
                                        current_price = float(px() or 0.0)
                                    elif px is not None:
                                        current_price = float(px or 0.0)
                                except Exception:
                                    pass
                                
                                if current_price > 0:
                                    status_info.append(f"💰 현재가: {current_price:.2f}")
                    except Exception:
                        status_info.append("💰 현재가: 확인 불가")
                    
                    # 시스템 상태
                    status_info.append("✅ 시스템: 정상 작동 중")
                    
                    # 메시지 조합
                    heartbeat_msg = "💓 <b>HEARTBEAT</b>\n\n" + "\n".join(status_info)
                    
                    # 텔레그램 전송
                    self.send_message(heartbeat_msg, parse_mode="HTML")
                    logger.info("[TG][HEARTBEAT] 하트비트 메시지 전송 완료")
                    
                except Exception as e:
                    logger.error("[TG][HEARTBEAT] 하트비트 메시지 전송 실패: %s", e)
                
            except Exception as exc:
                logger.warning("[TG][HEARTBEAT] 하트비트 루프 오류: %s", exc)
            
            # 대기
            self._stop_event.wait(timeout=self._heartbeat_interval_sec)
        
        logger.info("[TG][HEARTBEAT] 하트비트 모니터 루프 종료")

    def _predict_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._user_pause_event.is_set() and not self._market_paused:
                try:
                    try:
                        tp = getattr(self._pipeline, "tick_processor", None)
                        if tp is not None and bool(tp.market_closed):
                            try:
                                self._error_events.clear()
                            except Exception:
                                pass
                            if not self._market_paused:
                                self._market_paused = True
                                logger.info("[TG] 장 종료 감지 — 예측 루프 일시 정지")
                            self._stop_event.wait(timeout=self._interval)
                            continue
                        else:
                            # BUG-03: 장 재개 시 _market_paused 자동 복원
                            if self._market_paused:
                                self._market_paused = False
                                logger.info("[TG] 장 재개 감지 — 예측 루프 재시작")
                    except Exception:
                        pass

                    result = self._pipeline.get_prediction()
                    self._last_result = result

                    if "error" in result:
                        err_code = ""
                        try:
                            err_code = str(result.get("error") or "").strip().lower()
                        except Exception:
                            err_code = ""

                        actionable_for_alert = True
                        suppress_log = False
                        try:
                            if err_code in ("insufficient_minutes", "insufficient_data", "no_price"):
                                actionable_for_alert = False
                        except Exception:
                            pass

                        try:
                            tp = getattr(self._pipeline, "tick_processor", None)
                            if tp is not None:
                                ft = len(getattr(tp, "futures_ticks", []) or [])
                                ct = int(getattr(tp, "call_option_ticks", 0) or 0)
                                pt = int(getattr(tp, "put_option_ticks", 0) or 0)
                                if int(ft) <= 0 and int(ct) <= 0 and int(pt) <= 0:
                                    actionable_for_alert = False
                                    suppress_log = True
                        except Exception:
                            pass

                        # 예측 오류는 텔레그램으로 전송하지 않음 (내부 로그만)
                        try:
                            if (not suppress_log) or bool(_TG_DEBUG):
                                logger.info(
                                    "[TG][SUPPRESS] loop prediction error not sent: %s",
                                    str(result.get("error") or "unknown"),
                                )
                        except Exception:
                            pass

                        try:
                            if actionable_for_alert:
                                try:
                                    tp = getattr(self._pipeline, "tick_processor", None)
                                    if tp is not None and bool(tp.market_closed):
                                        actionable_for_alert = False
                                except Exception:
                                    pass
                                now_epoch = float(time.time())
                                self._error_events.append(now_epoch)
                                while self._error_events and (now_epoch - float(self._error_events[0])) > float(self._error_window_sec):
                                    self._error_events.popleft()
                                should_alert = (
                                    len(self._error_events) >= int(self._error_threshold)
                                    and (now_epoch - float(self._last_error_alert_epoch)) >= float(self._error_alert_cooldown_sec)
                                )
                                if should_alert:
                                    self._last_error_alert_epoch = float(now_epoch)
                                    self._notifier.send_text(
                                        f"⚠️ <b>예측 오류 급증</b>\n최근 {int(self._error_window_sec)}초 내 오류 {len(self._error_events)}회\n"
                                        f"마지막 오류: {str(result.get('error') or 'unknown')}\n"
                                        f"메시지: {str(result.get('message') or '')}",
                                        parse_mode="HTML",
                                    )
                        except Exception:
                            pass
                    else:
                        try:
                            self._error_events.clear()
                        except Exception:
                            pass
                        signal = str(result.get("signal", "HOLD")).upper()
                        consensus = bool(result.get("consensus", False))

                        # 최초 1회 정상 예측 전송 (신호 변경 필터와 동일 규칙 적용)
                        if not bool(self._first_prediction_sent):
                            include_summary = self._is_boundary_tick(result)
                            ok = False
                            try:
                                # force=False: _should_send 규칙1(최초, _last_signal 없음)에 의해
                                # 첫 신호는 BUY/SELL/HOLD 관계없이 1회 전송됨.
                                # 이후 동일 신호 반복은 _should_send 규칙3에 의해 차단.
                                # symbol 정보 추출
                                symbol = ""
                                try:
                                    ad = getattr(self._pipeline, "_adaptive_indicator", {})
                                    symbol = str(ad.get("symbol", "") or "")
                                except Exception:
                                    pass
                                ok = bool(
                                    self._notifier.send_prediction(
                                        result,
                                        force=False,
                                        include_dir_summary=bool(include_summary),
                                        symbol=symbol,
                                    )
                                )
                            except Exception:
                                ok = False

                            self._first_prediction_attempt_count += 1
                            if ok:
                                self._first_prediction_sent = True
                            elif self._first_prediction_attempt_count >= self._first_prediction_max_attempts:
                                # 최대 재시도 횟수 초과 — 이후 루프는 정상 필터링 모드로 전환
                                self._first_prediction_sent = True
                                try:
                                    logger.warning(
                                        "[TG][FIRST] 최초 전송 %d회 실패 — 정상 필터링 모드로 전환",
                                        self._first_prediction_attempt_count,
                                    )
                                except Exception:
                                    pass
                            try:
                                logger.info(
                                    "[TG][SEND] first prediction force-sent=%s attempt=%d/%d (signal=%s consensus=%s boundary=%s)",
                                    ok,
                                    self._first_prediction_attempt_count,
                                    self._first_prediction_max_attempts,
                                    signal,
                                    consensus,
                                    bool(include_summary),
                                )
                            except Exception:
                                pass
                            self._stop_event.wait(timeout=self._interval)
                            continue

                        # 신호 전환 여부 먼저 확인 (BUY/SELL/HOLD 간 변경)
                        # 전환 시에는 only_actionable / only_consensus 필터를 바이패스하고 반드시 전송
                        with self._notifier._signal_lock:
                            _prev_signal = str(self._notifier._last_signal or "").strip().upper()
                        _signal_changed = bool(_prev_signal) and (signal != _prev_signal)

                        should_send = True
                        if not _signal_changed:
                            # 동일 신호 반복 구간에서만 Bridge 필터 적용
                            if self._only_actionable:
                                should_send = signal != "HOLD"
                            if should_send and self._only_consensus:
                                # rule_based_only 모드(weights 미로드)에서는 LLM이 항상 HOLD를
                                # 내놓아 consensus가 사실상 성립되지 않으므로 필터를 비활성화한다.
                                _ensemble_method = str(result.get("ensemble_method") or "")
                                _is_rule_based = _ensemble_method in ("rule_based_only", "")
                                if _is_rule_based:
                                    logger.debug(
                                        "[TG][CONSENSUS_BYPASS] rule_based_only 모드 — only_consensus 필터 비활성화"
                                    )
                                else:
                                    should_send = consensus
                        if should_send:
                            try:
                                tp = getattr(self._pipeline, "tick_processor", None)
                                if tp is not None and bool(tp.market_closed):
                                    should_send = False
                            except Exception:
                                pass

                        if not should_send:
                            try:
                                logger.info(
                                    "[TG][SKIP] filtered by bridge flags (signal=%s consensus=%s actionable=%s only_consensus=%s signal_changed=%s)",
                                    signal,
                                    consensus,
                                    (not self._only_actionable) or (signal != "HOLD"),
                                    self._only_consensus,
                                    _signal_changed,
                                )
                            except Exception:
                                pass
                        else:
                            include_summary = self._is_boundary_tick(result)
                            ok = False
                            try:
                                # symbol 정보 추출
                                symbol = ""
                                try:
                                    ad = getattr(self._pipeline, "_adaptive_indicator", {})
                                    symbol = str(ad.get("symbol", "") or "")
                                except Exception:
                                    pass
                                ok = bool(self._notifier.send_prediction(result, force=False, include_dir_summary=include_summary, symbol=symbol))
                            except Exception:
                                ok = False

                            if ok:
                                try:
                                    logger.info(
                                        "[TG][SEND] prediction sent (signal=%s consensus=%s boundary=%s)",
                                        signal,
                                        consensus,
                                        bool(include_summary),
                                    )
                                except Exception:
                                    pass

                                # TradeExecutionGate — 매 틱 신호 전달 (enabled=False 시 no-op)
                                try:
                                    if self._trade_gate is not None:
                                        tp = getattr(self._pipeline, "tick_processor", None)
                                        price = float(
                                            tp.get_current_price() if tp is not None else 0.0
                                        )
                                        self._trade_gate.on_signal(result, current_price=price)
                                except Exception:
                                    pass
                            else:
                                # Usually skipped due to duplicate signal suppression in TelegramNotifier._should_send.
                                try:
                                    logger.info(
                                        "[TG][SKIP] duplicate/unchanged signal (signal=%s consensus=%s)",
                                        signal,
                                        consensus,
                                    )
                                except Exception:
                                    pass

                except Exception as exc:
                    logger.error("예측 루프 오류: %s", exc)
                    self._notifier.send_text(f"⚠️ 예측 루프 오류: {exc}")

            self._stop_event.wait(timeout=self._interval)

    def _is_boundary_tick(self, result: Dict[str, Any]) -> bool:
        """prediction_minutes 주기 경계(분)에서 첫 번째 틱이면 True.

        예) prediction_minutes=5 이면 00, 05, 10, 15 … 분에 처음 들어오는 결과가 경계.
        """
        try:
            pred_min = int(result.get("prediction_minutes") or 0)
            if pred_min <= 0:
                return False
            dt = self._parse_prediction_time(result)
            if dt is None:
                return False
            boundary = (dt.minute // pred_min) * pred_min
            key = dt.timetuple().tm_yday * 10000 + dt.hour * 100 + boundary
            if key != self._last_boundary_minute:
                self._last_boundary_minute = key
                return True
        except Exception:
            pass
        return False

    def _parse_prediction_time(self, result: Dict[str, Any]) -> Optional[datetime]:
        try:
            pred_time_str = str(result.get("prediction_time") or "")
            if not pred_time_str:
                return None
            # fromisoformat()은 'Z'를 직접 파싱하지 못하므로 보정
            if pred_time_str.endswith("Z"):
                pred_time_str = pred_time_str[:-1] + "+00:00"
            return datetime.fromisoformat(pred_time_str)
        except Exception:
            return None

