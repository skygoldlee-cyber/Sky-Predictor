"""Mixin extracted from prediction/pipeline.py.

이 파일은 PredictionPipeline의 일부를 Mixin으로 분리한 것입니다.
직접 인스턴스화하지 마십시오. PredictionPipeline을 통해 사용하세요.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

from ..features.option_features import build_option_snapshot


class OptionMixin:
    """Mixin: OptionMixin methods extracted from PredictionPipeline."""

    def _build_option_tick_flow_snapshot(self, *, current_price: Optional[float] = None) -> Dict[str, float]:
        """옵션 틱 유입 강도 스냅샷(1분 카운트/20분 평균/배수)을 계산한다."""
        try:
            now_min = datetime.now().replace(second=0, microsecond=0)
            call_ticks = int(getattr(self.tick_processor, "call_option_ticks", 0) or 0)
            put_ticks = int(getattr(self.tick_processor, "put_option_ticks", 0) or 0)
            total_ticks = int(call_ticks + put_ticks)
            last_min = getattr(self, "_opt_tick_flow_last_minute", None)
            last_price = float(getattr(self, "_opt_tick_flow_last_price", 0.0) or 0.0)
            if last_min is None:
                self._opt_tick_flow_last_minute = now_min
                self._opt_tick_flow_last_total_ticks = int(total_ticks)
                self._opt_tick_flow_last_call_ticks = int(call_ticks)
                self._opt_tick_flow_last_put_ticks = int(put_ticks)
                if current_price is not None:
                    self._opt_tick_flow_last_price = float(current_price)
            elif now_min > last_min:
                delta = max(0, int(total_ticks) - int(getattr(self, "_opt_tick_flow_last_total_ticks", 0) or 0))
                self._opt_tick_flow_window.append(float(delta))
                self._opt_tick_flow_last_minute = now_min
                self._opt_tick_flow_last_total_ticks = int(total_ticks)
                self._opt_tick_flow_last_call_ticks = int(call_ticks)
                self._opt_tick_flow_last_put_ticks = int(put_ticks)
                if current_price is not None:
                    self._opt_tick_flow_last_price = float(current_price)

            one_min = max(0.0, float(total_ticks) - float(getattr(self, "_opt_tick_flow_last_total_ticks", 0) or 0))
            window = list(getattr(self, "_opt_tick_flow_window", []) or [])
            avg20 = float(np.mean(window)) if window else 0.0
            surge = (float(one_min) / float(avg20)) if avg20 > 1e-9 else 0.0

            # 콜/풋 편향: 최근 1분 구간에서의 비율(가능할 때만)
            call_1m = max(0, int(call_ticks) - int(getattr(self, "_opt_tick_flow_last_call_ticks", 0) or 0))
            put_1m = max(0, int(put_ticks) - int(getattr(self, "_opt_tick_flow_last_put_ticks", 0) or 0))
            total_cp = call_1m + put_1m
            cp_imb = 0.0
            if total_cp > 0:
                cp_imb = float(call_1m - put_1m) / float(total_cp)
            per_tick_move_pt = 0.0
            try:
                if current_price is not None and one_min > 0.0 and last_price > 0.0:
                    per_tick_move_pt = abs(float(current_price) - float(last_price)) / float(one_min)
            except Exception:
                per_tick_move_pt = 0.0

            return {
                "ticks_1m": float(one_min),
                "ticks_avg20m": float(avg20),
                "surge_ratio": float(surge),
                "cp_imbalance": float(cp_imb),
                "per_tick_move_pt": float(per_tick_move_pt),
            }
        except Exception:
            return {
                "ticks_1m": 0.0,
                "ticks_avg20m": 0.0,
                "surge_ratio": 0.0,
                "cp_imbalance": 0.0,
                "per_tick_move_pt": 0.0,
            }

    def _init_option_sentiment_analyzer(self, config_path: Optional[str] = None) -> None:
        """옵션 센티먼트 분석기를 초기화한다."""
        try:
            from indicators.option_sentiment import OptionSentimentAnalyzer, load_config_from_dict
            import json

            cfg_path = str(config_path or getattr(self, "_config_path", "config.json"))
            config_dict = {}
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f) or {}
            except Exception:
                config_dict = {}

            sentiment_config = load_config_from_dict(config_dict)

            # 콜백 함수: 이벤트 발생 시 텔레그램 전송
            def on_sentiment_event(signal):
                try:
                    notifier = getattr(self, "_notifier", None)
                    if notifier is None:
                        return

                    signal_dict = {
                        "direction": signal.direction.value,
                        "confidence": signal.confidence,
                        "skew": signal.skew,
                        "volume_pcr": signal.volume_pcr,
                        "oi_pcr": signal.oi_pcr,
                        "event_type": signal.event_type,
                        "prev_direction": signal.prev_direction.value if signal.prev_direction else None,
                        "prev_confidence": signal.prev_confidence,
                    }

                    current_price = float(getattr(self, "_last_underlying_price", 0.0) or 0.0)
                    if current_price <= 0.0:
                        logger.warning(
                            "[OptionSentiment] 기초자산 가격 미수신으로 알림 생략 (event=%s, direction=%s)",
                            signal.event_type, signal.direction.value
                        )
                        return

                    notifier.send_option_sentiment_alert(signal_dict, current_price)
                except Exception as e:
                    logger.error("[OptionSentiment] 콜백에서 텔레그램 전송 실패: %s", e)

            self._option_sentiment_analyzer = OptionSentimentAnalyzer(
                sentiment_config, event_callback=on_sentiment_event
            )
            logger.info("[OptionMixin] 옵션 센티먼트 분석기 초기화 완료")
        except Exception as e:
            logger.warning("[OptionMixin] 옵션 센티먼트 분석기 초기화 실패: %s", e)
            self._option_sentiment_analyzer = None

    def _analyze_option_sentiment(
        self, opt_snap: Dict[str, Any], current_price: float
    ) -> Optional[Dict[str, Any]]:
        """옵션 스냅샷에서 센티먼트를 분석한다.

        Args:
            opt_snap: build_option_snapshot() 반환 dict.
            current_price: 현재 선물 가격.

        Returns:
            센티먼트 신호 dict. 분석 실패 시 None.
        """
        try:
            analyzer = getattr(self, "_option_sentiment_analyzer", None)
            if analyzer is None:
                return None

            # 옵션 스냅샷에서 필요한 데이터 추출
            # None 체크와 0 체크를 분리하여 유효한 0값이 손실되지 않도록 함
            iv_skew_raw = opt_snap.get("iv_skew")
            iv_skew = float(iv_skew_raw) if iv_skew_raw is not None else 1.0
            # iv_skew = put_iv / call_iv (비율)
            # 우리가 필요한 것은 call_iv - put_iv (차이)
            # 따라서 skew = 1 - iv_skew로 근사 계산
            #   예: put_iv=20%, call_iv=18% → iv_skew=1.11 → skew=-0.11 (약세)
            #   예: put_iv=18%, call_iv=20% → iv_skew=0.90 → skew=+0.10 (강세)
            # 주의: 1 - ratio는 퍼센트가 아니라 비율 차이이므로 임계값(0.05)은 비율 기준임
            skew = 1.0 - iv_skew

            pcr_volume_raw = opt_snap.get("pcr_volume")
            pcr_volume = float(pcr_volume_raw) if pcr_volume_raw is not None else 1.0

            pcr_oi_raw = opt_snap.get("pcr_oi")
            pcr_oi = float(pcr_oi_raw) if pcr_oi_raw is not None else 1.0

            # 센티먼트 분석
            signal = analyzer.analyze(skew=skew, volume_pcr=pcr_volume, oi_pcr=pcr_oi)

            return {
                "direction": signal.direction.value,
                "confidence": signal.confidence,
                "skew": signal.skew,
                "volume_pcr": signal.volume_pcr,
                "oi_pcr": signal.oi_pcr,
                "skew_signal": signal.skew_signal,
                "volume_pcr_signal": signal.volume_pcr_signal,
                "oi_pcr_signal": signal.oi_pcr_signal,
                "event_type": signal.event_type,
                "prev_direction": signal.prev_direction.value if signal.prev_direction else None,
                "prev_confidence": signal.prev_confidence,
            }
        except Exception as e:
            logger.error("[OptionMixin] 옵션 센티먼트 분석 실패: %s", e)
            return None

    def _notify_oi_level_change(
        self,
        new_oi: Dict[str, float],
        prev_oi: Dict[str, float],
        current_price: float,
        opt_snapshot: Optional[Dict[str, Any]] = None,
        tick_flow_snapshot: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Call/Put OI Peak 레벨 변경 시 텔레그램 알람을 전송한다.

        감지 조건:
            - call_oi_peak 또는 put_oi_peak 가 min_change_pt(0.5pt) 이상 변동
            - 이전 알람으로부터 _OI_ALERT_COOLDOWN_SEC(기본 5분) 경과
            - notifier가 설정돼 있을 것

        메시지 포맷:
            📊 OI 지지/저항 변경
            ───────────────────
            저항(Call Peak): 390.00 → 392.50  ▲+2.50pt
            지지(Put  Peak): 382.50  (유지)
            현재가: 388.75
            OI 박스: 10.00pt
        """
        try:
            notifier = getattr(self, "_notifier", None)
            if notifier is None:
                return False

            min_change_pt = 0.5   # 최소 변화 임계값 (pt)

            new_call  = float(new_oi.get("call_oi_peak")  or 0.0)
            new_put   = float(new_oi.get("put_oi_peak")   or 0.0)
            old_call  = float(prev_oi.get("call_oi_peak") or 0.0)
            old_put   = float(prev_oi.get("put_oi_peak")  or 0.0)
            # 2nd Peak
            new_call2 = float(new_oi.get("call_oi_peak2") or 0.0)
            new_put2  = float(new_oi.get("put_oi_peak2")  or 0.0)
            new_call2_norm = float(new_oi.get("call_oi_peak2_norm") or 0.0)
            new_put2_norm  = float(new_oi.get("put_oi_peak2_norm")  or 0.0)
            new_call_norm  = float(new_oi.get("call_oi_peak_norm")  or 0.0)
            new_put_norm   = float(new_oi.get("put_oi_peak_norm")   or 0.0)
            new_call_global = float(new_oi.get("call_oi_peak_global") or 0.0)
            new_put_global  = float(new_oi.get("put_oi_peak_global") or 0.0)
            new_call_global_norm = float(new_oi.get("call_oi_peak_global_norm") or 0.0)
            new_put_global_norm  = float(new_oi.get("put_oi_peak_global_norm") or 0.0)

            # 이전값이 없으면 첫 수신 — 알람 없이 기준값만 설정
            if old_call <= 0.0 and old_put <= 0.0:
                return False

            call_changed = abs(new_call - old_call) >= min_change_pt and new_call > 0.0
            put_changed  = abs(new_put  - old_put)  >= min_change_pt and new_put  > 0.0

            if not call_changed and not put_changed:
                return False

            # 쿨다운 체크
            now_t = float(time.time())
            if now_t - float(self._oi_alert_last_epoch) < float(self._OI_ALERT_COOLDOWN_SEC):
                return False
            self._oi_alert_last_epoch = now_t

            # 메시지 조립
            F = float(current_price or 0.0)

            def _level_line(label: str, old_v: float, new_v: float, changed: bool,
                            norm: float = 0.0) -> str:
                norm_str = f"  집중도 {norm:.0%}" if norm > 0.0 else ""
                if not changed or new_v <= 0.0:
                    return f"{label}: <code>{new_v:.2f}</code>  (유지){norm_str}"
                diff = new_v - old_v
                arrow = "▲" if diff > 0 else "▼"
                sign  = "+" if diff > 0 else ""
                return (
                    f"{label}: <code>{old_v:.2f}</code> → <code>{new_v:.2f}</code>  "
                    f"{arrow}<b>{sign}{diff:.2f}pt</b>{norm_str}"
                )

            call_line = _level_line("저항(Call 1st)", old_call, new_call, call_changed, new_call_norm)
            put_line  = _level_line("지지(Put  1st)", old_put,  new_put,  put_changed,  new_put_norm)

            oi_box = new_call - new_put if new_call > 0.0 and new_put > 0.0 else 0.0

            # 현재가 vs 저항/지지 거리
            dist_lines = []
            if F > 0.0:
                if new_call > 0.0:
                    d_call = new_call - F
                    dist_lines.append(f"  현재가 → 저항(1st): <code>{d_call:+.2f}pt</code>")
                if new_put > 0.0:
                    d_put = F - new_put
                    dist_lines.append(f"  현재가 → 지지(1st): <code>{d_put:+.2f}pt</code>")

            lines = [
                "📊 <b>OI 지지/저항 변경</b>",
                "━━━━━━━━━━━━━━━━━━━",
                call_line,
                put_line,
            ]
            # 2nd Peak 정보 (유효한 경우만)
            if new_call2 > 0.0:
                norm2_str = f"  집중도 {new_call2_norm:.0%}" if new_call2_norm > 0.0 else ""
                d2 = new_call2 - F if F > 0.0 else 0.0
                dist2_str = f"  (현재가+{d2:+.2f}pt)" if F > 0.0 else ""
                lines.append(f"  저항 2nd: <code>{new_call2:.2f}</code>{norm2_str}{dist2_str}")
            if new_put2 > 0.0:
                norm2_str = f"  집중도 {new_put2_norm:.0%}" if new_put2_norm > 0.0 else ""
                d2 = F - new_put2 if F > 0.0 else 0.0
                dist2_str = f"  (현재가{d2:+.2f}pt)" if F > 0.0 else ""
                lines.append(f"  지지  2nd: <code>{new_put2:.2f}</code>{norm2_str}{dist2_str}")
            if new_call_global > 0.0:
                gnorm = f"  집중도 {new_call_global_norm:.0%}" if new_call_global_norm > 0.0 else ""
                lines.append(f"  전체최대 Call OI: <code>{new_call_global:.2f}</code>{gnorm}")
            if new_put_global > 0.0:
                gnorm = f"  집중도 {new_put_global_norm:.0%}" if new_put_global_norm > 0.0 else ""
                lines.append(f"  전체최대 Put OI:  <code>{new_put_global:.2f}</code>{gnorm}")
            if dist_lines:
                lines += dist_lines
            if F > 0.0:
                lines.append(f"현재가: <code>{F:.2f}</code>")
            if oi_box > 0.0:
                lines.append(f"OI 박스: <code>{oi_box:.2f}pt</code>")
            msg = "\n".join(lines)
            try:
                notifier.send_text(
                    msg,
                    parse_mode="HTML",
                    debug_context={"kind": "oi_level_change"},
                )
                logger.info("[OI_ALERT] 텔레그램 전송 성공")
                # 요구사항: OI 지지/저항 변경 신호와 옵션 마이크로 플로우를 함께 송출.
                try:
                    if isinstance(opt_snapshot, dict):
                        tf = (
                            tick_flow_snapshot
                            if isinstance(tick_flow_snapshot, dict)
                            else self._build_option_tick_flow_snapshot(current_price=float(F))
                        )
                        if isinstance(tf, dict) and tf:
                            opt_snapshot["_tick_flow"] = dict(tf)
                        logger.info("[OI_ALERT] 동반 옵션 마이크로 플로우 전송")
                        notifier.send_option_flow_status(
                            {"options": dict(opt_snapshot)},
                        )
                    else:
                        logger.info("[OI_ALERT] opt_snapshot이 None이어서 마이크로 플로우 전송 생략")
                except Exception as _flow_e:
                    logger.info("[OI_ALERT] 동반 마이크로 플로우 전송 실패: %s", _flow_e)
                return True
            except Exception as _e:
                logger.info("[OI_ALERT] 텔레그램 전송 실패: %s", _e)
                return False

        except Exception as _e:
            logger.info("[OI_ALERT] 알람 처리 중 예외: %s", _e)
            return False

    def _build_option_snapshot_safe(
        self,
        *,
        current_price: float,
        update_prev: bool = True,
    ) -> Dict[str, Any]:
        """옵션 스냅샷을 안전하게 빌드한다.

        Args:
            current_price: 현재 선물 가격.
            update_prev: True(기본값)이면 계산 후 _prev_* 상태를 갱신한다.
                OB 버퍼 경로(1Hz)에서는 True로 호출하여 매 초 prev를 갱신한다.
                get_prediction() 경로에서는 False로 호출한다.
                OB 버퍼가 이미 prev를 갱신했으므로 이중 갱신을 방지하기 위함이다.
        """
        try:
            # v3/v4: _prev_* 상태를 build_option_snapshot 파라미터로 직접 전달한다.
            # (이전의 snap.pop + extra 재계산 이중실행 방식을 제거.)
            _needs_prev = self._option_feature_set in ("v3", "v4", "v5")
            prev_u = self._prev_underlying_price if _needs_prev else None
            prev_c = self._prev_atm_call_price   if _needs_prev else None
            prev_p = self._prev_atm_put_price    if _needs_prev else None

            snap = build_option_snapshot(
                self.tick_processor.call_options,
                self.tick_processor.put_options,
                current_price,
                tick_processor=self.tick_processor,
                option_feature_set=str(self._option_feature_set),
                prev_underlying_price=prev_u,
                prev_atm_call_price=prev_c,
                prev_atm_put_price=prev_p,
                otm_open_min=float(self._otm_open_min),
                pcr_atm_strikes_each_side=int(getattr(self, "_pcr_atm_strikes_each_side", 5)),
                prev_oi_levels=dict(self._prev_oi_levels) if self._prev_oi_levels else None,
            )
            # OI velocity용 스냅샷 갱신
            try:
                _new_oi = snap.get("_oi_levels")
                snap["_oi_level_change_fired"] = False
                if isinstance(_new_oi, dict) and _new_oi:
                    # OI 레벨 변경 감지 → 텔레그램 알람
                    try:
                        _fired = self._notify_oi_level_change(
                            new_oi=_new_oi,
                            prev_oi=self._prev_oi_levels,
                            current_price=float(current_price),
                            opt_snapshot=snap,
                            tick_flow_snapshot=self._build_option_tick_flow_snapshot(),
                        )
                        snap["_oi_level_change_fired"] = bool(_fired)
                    except Exception:
                        pass
                    self._prev_oi_levels = dict(_new_oi)
            except Exception:
                pass

            # 옵션 센티먼트 분석
            try:
                sentiment_result = self._analyze_option_sentiment(snap, float(current_price))
                if sentiment_result:
                    snap["_sentiment"] = sentiment_result
            except Exception:
                pass
            # DTE 직접 접근용 — _calc_amplitude_snapshot에서 days_to_expiry 우선 사용
            try:
                self._last_opt_snap = snap
            except Exception:
                pass

            # 다음 틱을 위해 현재 ATM 가격 캐싱.
            # snap에는 체결가 키가 없으므로 tick_processor에서 직접 읽는다.
            # v1/v2에서도 underlying_price를 갱신해두면 v3 전환 시 첫 틱부터 diff 계산 가능.
            if update_prev:
                try:
                    self._prev_underlying_price = float(current_price)
                    if _needs_prev:
                        from ..features.option_features import _get_atm_option_price
                        # ATM 행사가는 calc_iv_skew가 snap에 저장한 atm_strike 재사용.
                        # 없으면 underlying_price로 직접 탐색한다.
                        atm_k = float(snap.get("atm_strike") or 0.0)
                        if atm_k <= 0.0:
                            try:
                                strikes = [
                                    float(v.get("strike") or 0.0)
                                    for v in self.tick_processor.call_options.values()
                                    if float(v.get("strike") or 0.0) > 0.0
                                ]
                                if strikes:
                                    atm_k = float(min(strikes, key=lambda s: abs(s - float(current_price))))
                            except Exception:
                                atm_k = 0.0
                        if atm_k > 0.0:
                            cp = _get_atm_option_price(self.tick_processor.call_options, atm_k)
                            pp = _get_atm_option_price(self.tick_processor.put_options, atm_k)
                            self._prev_atm_call_price = float(cp) if cp > 0.0 else self._prev_atm_call_price
                            self._prev_atm_put_price = float(pp) if pp > 0.0 else self._prev_atm_put_price
                except Exception:
                    pass

            return snap
        except Exception:
            return {
                "pcr_volume": 1.0,
                "pcr_oi": 1.0,
                "iv_skew": 1.0,
                "max_pain_dist_pct": 0.0,
                "atm_iv": 0.0,
                "atm_spread_pct": 0.0,
                "atm_orderbook_imb": 0.0,
                "atm_liquidity_log": 0.0,
            }

    def _build_llm_snapshot(
        self,
        *,
        current_price: float,
        spot_index: Optional[float],
        basis: Optional[float],
        prob: float,
        signal: str,
        confidence: str,
        raw_signal: str,
        raw_confidence: str,
        guardrail_applied: bool,
        guardrail_reason: str,
        transformer_prob: float,
        tft_prob: Any,
        ensemble_method: str,
        model_agreement: Any,
        transformer_weight: Optional[float],
        last_ob_snapshot: Dict[str, Any],
        opt_snap: Dict[str, Any],
        adaptive_features: Optional[Dict[str, float]],
        heuristic_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        def _prob_to_signal(p: float) -> str:
            if float(p) >= float(self._buy_threshold):
                return "BUY"
            if float(p) <= float(self._sell_threshold):
                return "SELL"
            return "HOLD"

        snapshot: Dict[str, Any] = {
            "prediction_minutes": int(self.prediction_minutes),
            "transformer": {
                "prob": float(transformer_prob),
                "signal": str(_prob_to_signal(transformer_prob)),
            },
            "ensemble": {
                "prob": float(prob),
                "signal": str(signal),
                "confidence": str(confidence),
                "method": str(ensemble_method),
                "agreement": model_agreement if model_agreement is None else bool(model_agreement),
            },
            "market": {
                "current_price": float(current_price),
                "spot_index": float(spot_index) if spot_index is not None else None,
                "basis": float(basis) if basis is not None else None,
            },
        }

        if tft_prob is not None:
            try:
                snapshot["tft"] = {
                    "prob": float(tft_prob),
                    "signal": str(_prob_to_signal(float(tft_prob))),
                }
            except Exception:
                pass

        if transformer_weight is not None:
            try:
                snapshot["ensemble"]["transformer_weight"] = float(transformer_weight)
            except Exception:
                pass

        bg: Dict[str, Any] = {}
        if isinstance(self._t2101_snapshot, dict) and self._t2101_snapshot:
            bg["t2101"] = dict(self._t2101_snapshot)
        if isinstance(self._t2301_snapshot, dict) and self._t2301_snapshot:
            bg["t2301"] = dict(self._t2301_snapshot)
        if isinstance(self._ij_realtime_snapshot, dict) and self._ij_realtime_snapshot:
            bg["ij_"] = dict(self._ij_realtime_snapshot)
        if bg:
            snapshot["market_background"] = bg

        if last_ob_snapshot:
            snapshot["orderbook"] = dict(last_ob_snapshot)

        if opt_snap:
            snapshot["options"] = dict(opt_snap)

        if adaptive_features:
            snapshot["adaptive"] = dict(adaptive_features)

        # 휴리스틱(지그재그 피봇 기반) 판정 결과를 LLM 입력에 명시한다.
        # 운영 단순화 규칙(L->BUY, H->SELL) 적용 상태를 LLM이 직접 참조할 수 있게 한다.
        if isinstance(heuristic_info, dict) and heuristic_info:
            try:
                _ha = str(heuristic_info.get("action") or "").upper().strip()
                _hr = str(heuristic_info.get("reason") or "").strip()
                _ready = bool(heuristic_info.get("is_ready", False))
                snapshot["heuristic"] = {
                    "action": _ha if _ha in ("BUY", "SELL", "HOLD") else "HOLD",
                    "is_ready": bool(_ready),
                    "reason": _hr,
                }
            except Exception:
                pass

        try:
            if bool(guardrail_applied) or (str(raw_signal) != str(signal)) or (str(raw_confidence) != str(confidence)):
                snapshot["guardrail"] = {
                    "applied": bool(guardrail_applied),
                    "original_signal": str(raw_signal),
                    "original_confidence": str(raw_confidence),
                    "reason": str(guardrail_reason or ""),
                }
        except Exception:
            pass

        return snapshot

