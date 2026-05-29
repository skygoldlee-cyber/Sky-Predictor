"""Mixin extracted from prediction/pipeline.py.

이 파일은 PredictionPipeline의 일부를 Mixin으로 분리한 것입니다.
직접 인스턴스화하지 마십시오. PredictionPipeline을 통해 사용하세요.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class AmplitudeMixin:
    """Mixin: AmplitudeMixin methods extracted from PredictionPipeline."""

    def _update_sigma_multiplier(self, amplitude: Dict[str, Any]) -> None:
        """실현 진폭 피드백으로 sigma_multiplier를 자동 조정한다.

        amplitude_exhaustion > 1.0 (예상 초과) 이 연속 2회 이상이면
        sigma_multiplier를 최대 1.3까지 10% 상향한다.
        exhaustion < 0.6 이 연속 2회 이면 1.0으로 복원한다.
        """
        try:
            exhaust = float(amplitude.get("amplitude_exhaustion") or 0.0)
            if exhaust > 1.0:
                self._exhaust_exceed_count = min(self._exhaust_exceed_count + 1, 5)
                if self._exhaust_exceed_count >= 2:
                    self._sigma_multiplier = min(1.3, self._sigma_multiplier * 1.10)
            elif exhaust < 0.6 and exhaust > 0.0:
                self._exhaust_exceed_count = max(self._exhaust_exceed_count - 1, 0)
                if self._exhaust_exceed_count == 0:
                    self._sigma_multiplier = max(1.0, self._sigma_multiplier * 0.95)
        except Exception as _e:
            logger.debug("[_update_sigma_multiplier] 오류 무시: %s", _e)

    def _update_realized_amplitude_ema(self, amplitude: Dict[str, Any]) -> None:
        """방안C: 당일 실현 진폭으로 EMA를 갱신한다.

        장 마감 무렵 realized_hl_range_pt가 충분히 쌓였을 때(>= 3.0pt) 하루 1회 갱신.
        EMA는 다음 예측 사이클에서 IV 기반 진폭과 혼합하는 데 사용된다.

        갱신 조건:
            - realized_hl_range_pt >= 3.0pt (의미 있는 장중 진폭)
            - 오늘 날짜에 아직 갱신하지 않은 경우 (하루 1회)
            - [FIX-AMP-3] 현재 시각 >= 15:00 (장 마감 근접 시점).
              오전 장 초반에 갱신하면 일중 진폭이 미완성 상태로 EMA가 과소평가된다.
              KP200 선물 정규장 마감은 15:45이므로 15:00 이후면 대부분의 진폭이 확정됨.
        """
        try:
            realized = float(amplitude.get("realized_hl_range_pt") or 0.0)
            if realized < 3.0:
                return

            now_dt = None
            try:
                now_dt = datetime.now()
            except Exception:
                return

            # [FIX-AMP-3] 15:00 이전에는 갱신하지 않음 (일중 진폭 미완성 방지)
            if now_dt.hour < 15:
                return

            today = ""
            try:
                today = now_dt.strftime("%Y%m%d")
            except Exception:
                return

            if today == str(self._realized_amplitude_ema_updated_date or ""):
                return  # 오늘 이미 갱신함

            alpha = float(self._realized_amplitude_ema_alpha or 0.2)
            if float(self._realized_amplitude_ema) <= 0.0:
                # 첫 수신 — 초기화
                self._realized_amplitude_ema = realized
            else:
                self._realized_amplitude_ema = (
                    alpha * realized + (1.0 - alpha) * float(self._realized_amplitude_ema)
                )
            self._realized_amplitude_ema_updated_date = today
            logger.info(
                "[AMP_EMA] realized=%.2fpt → ema=%.2fpt (alpha=%.2f) at %s",
                realized, float(self._realized_amplitude_ema), alpha,
                now_dt.strftime("%H:%M"),
            )
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)

    def _calc_amplitude_snapshot(
        self,
        *,
        current_price: float,
        atm_iv: float = 0.0,
        dte_weight_norm: float = 0.0,
        oi_levels: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """당일 선물 진폭 스냅샷을 산출한다.

        분봉 DataFrame에서 장중 누적 고가/저가/시가를 추출하고
        option_features.calc_expected_amplitude()를 호출하여
        IV 기반 예상 진폭과 실현 진폭을 반환한다.

        Args:
            current_price:    현재 선물가.
            atm_iv:           ATM 내재변동성(소수). 0이면 내부에서 calc_iv_skew 재계산.
                              get_prediction() 흐름에서는 opt_snap["atm_iv"]를 전달하여
                              calc_iv_skew 중복 계산을 방지한다.
            dte_weight_norm:  만기 근접도 [0,1]. opt_snap["dte_weight_norm"]을 전달하면
                              DTE를 역산하여 expected_amplitude_pt 정확도를 높인다.
                              0이면 get_expiry_week_info() fallback 사용.

        Returns:
            dict (모든 키 항상 존재):
                expected_amplitude_pt   : IV 기반 예상 진폭(pt). 0 = 계산 불가.
                realized_hl_range_pt    : 장중 실현 진폭(pt). 0 = 데이터 없음.
                amplitude_exhaustion    : 소진 비율. 1.0 초과 = 예상 범위 초과.
                remaining_amplitude_pt  : 남은 예상 진폭(pt).
                open_dist_pct           : 현재가 vs 시가 거리(%).
                session_open            : 사용된 시가값(pt).
                oi_box_pt               : OI 박스폭(pt). 0 = OI 없음.
                oi_vs_amplitude         : OI 박스폭 / 예상진폭 비율.
                call_dist_pt            : 현재가 → Call OI Peak 거리(pt).
                put_dist_pt             : 현재가 → Put OI Peak 거리(pt).
                _amplitude_source       : 진폭 데이터 소스 ("iv+session" | "session_only" | "none").
        """
        from prediction.option_features import calc_expected_amplitude

        _empty: Dict[str, Any] = {
            "expected_amplitude_pt":  0.0,
            "realized_hl_range_pt":   0.0,
            "amplitude_exhaustion":   0.0,
            "remaining_amplitude_pt":   0.0,
            "upside_remaining_pt":      0.0,
            "downside_remaining_pt":    0.0,
            "open_dist_pct":          0.0,
            "session_open":           0.0,
            "oi_box_pt":              0.0,
            "oi_vs_amplitude":        0.0,
            "call_dist_pt":           0.0,
            "put_dist_pt":            0.0,
            "_amplitude_source":      "none",
            # 방안B 혼합 진단 필드 (oi_features.calc_expected_amplitude 반환값과 동기화)
            "_oi_weight":             0.0,
            "_iv_amplitude_pt":       0.0,
            "_oi_amplitude_pt":       0.0,
            # 방안C EMA 보정 진단 필드
            "_realized_amplitude_ema": 0.0,
            "_ema_blended":            False,
        }

        try:
            # ── 1. 당일 세션 OHLC 추출 ────────────────────────────────────
            # [FIX-AMP-1] 우선순위:
            #   1) tick_processor.get_daily_session_ohlc() — FC0 틱 누적값.
            #      TICK_DATA_RETENTION_HOURS(2h) 제한과 무관하게 장 내내 유효.
            #   2) t2101 스냅샷 — REST 1~2회 수신, open 기준값으로 활용.
            #   3) 분봉 DataFrame — 2시간 범위만 유효하므로 고가/저가 fallback 전용.
            session_high = 0.0
            session_low  = 0.0
            session_open = 0.0

            # 1) tick_processor 당일 누적 OHLC (주 소스)
            try:
                _daily = self.tick_processor.get_daily_session_ohlc()
                session_high = float(_daily.get("session_high") or 0.0)
                session_low  = float(_daily.get("session_low")  or 0.0)
                session_open = float(_daily.get("session_open") or 0.0)
            except Exception as _e:
                logger.debug("오류 무시: %s", _e)

            # 2) t2101 open: REST 스냅샷이 더 정확한 경우 open을 덮어씀
            #    (FC0 틱 open 필드가 장 초기에 0이거나 전일 값인 경우 방어)
            try:
                _t2101_open = float((self._t2101_snapshot or {}).get("open") or 0.0)
                _cp = float(current_price or 0.0)
                if _t2101_open > 0.0 and _cp > 0.0:
                    _open_ratio = abs(_t2101_open - _cp) / _cp
                    if _open_ratio <= 0.30:  # ±30% 이내만 신뢰
                        if session_open == 0.0 or _t2101_open < session_open:
                            session_open = _t2101_open
            except Exception as _e:
                logger.debug("오류 무시: %s", _e)

            # 3) 분봉 DataFrame fallback: high/low가 아직 0인 경우 보완
            #    (장 직후 첫 틱 수신 전 등 극초단 구간)
            if session_high <= 0.0 or session_low <= 0.0:
                try:
                    df_all = self.tick_processor.get_futures_minute_df(411)
                    if df_all is not None and not df_all.empty:
                        if session_high <= 0.0:
                            for col_h in ("High", "high"):
                                if col_h in df_all.columns:
                                    _h = float(df_all[col_h].max())
                                    if _h > 0.0:
                                        session_high = _h
                                    break
                        if session_low <= 0.0:
                            for col_l in ("Low", "low"):
                                if col_l in df_all.columns:
                                    _l = float(df_all[col_l].min())
                                    if _l > 0.0:
                                        session_low = _l
                                    break
                        # [FIX-AMP-2] open fallback: 08:45 이후 첫 행만 사용
                        if session_open <= 0.0:
                            try:
                                from datetime import time as _time
                                _idx = df_all.index
                                if hasattr(_idx, "time"):
                                    _mask = _idx.time >= _time(8, 45)
                                    _open_rows = df_all.loc[_mask]
                                else:
                                    _open_rows = df_all
                                if not _open_rows.empty:
                                    for col_o in ("Open", "open"):
                                        if col_o in _open_rows.columns:
                                            _o = float(_open_rows[col_o].iloc[0])
                                            if _o > 0.0:
                                                session_open = _o
                                            break
                            except Exception as _e:
                                logger.debug("오류 무시: %s", _e)
                except Exception as _e:
                    logger.debug("오류 무시: %s", _e)

            # 모든 소스에서 고가/저가 추출 실패 시 현재가로 fallback (실현 진폭 0pt)
            if session_high <= 0.0:
                session_high = float(current_price)
            if session_low <= 0.0:
                session_low = float(current_price)

            # ── 2. atm_iv 결정 ──────────────────────────────────────────
            # 호출자가 opt_snap["atm_iv"]를 전달했으면 그대로 사용(중복 계산 없음).
            # 전달되지 않은 경우(atm_iv=0.0)에만 calc_iv_skew를 재계산한다.
            _atm_iv = float(atm_iv or 0.0)
            source  = "session_only"
            if _atm_iv <= 0.0:
                try:
                    _tp = getattr(self, "tick_processor", None)
                    if _tp is not None:
                        _calls = getattr(_tp, "call_options", {}) or {}
                        _puts  = getattr(_tp, "put_options",  {}) or {}
                        if _calls or _puts:
                            from prediction.option_features import calc_iv_skew
                            _skew = calc_iv_skew(_calls, _puts, float(current_price))
                            _iv = float(_skew.get("atm_call_iv") or _skew.get("atm_iv") or 0.0)
                            if _iv > 0.0:
                                _atm_iv = _iv
                except Exception:
                    _atm_iv = 0.0

            if _atm_iv > 0.0:
                source = "iv+session"

            # ── 3. DTE 결정 ─────────────────────────────────────────────
            # 우선순위 (Medium-07 개선):
            #   1) opt_snap["days_to_expiry"] — build_option_snapshot이 직접 노출 (가장 정확)
            #   2) dte_weight_norm 역산: dte = 1 / (dte_w * 10)  (클리핑 오차 있음)
            #   3) get_expiry_week_info() 직접 조회
            #   4) 기본값 1.0 (당일 기준)
            _dte = 1.0
            try:
                # 1) days_to_expiry 직접 노출값 우선
                _direct_dte = float(dte_weight_norm or 0.0)  # 파라미터명 유지 (하위호환)
                # opt_snap은 호출자가 dte_weight_norm 파라미터로 전달하는 대신
                # days_to_expiry가 snap에 있으면 그쪽을 우선한다.
                # _calc_amplitude_snapshot 호출 시 opt_snap 자체를 참조하는 경로 확보
                _snap_dte = float(
                    (getattr(self, "_last_opt_snap", None) or {}).get("days_to_expiry") or 0.0
                )
                if _snap_dte > 0.0:
                    _dte = max(1.0, min(30.0, _snap_dte))
                elif _direct_dte > 0.0:
                    # 2) dte_weight_norm 역산
                    _dte = float(1.0 / (_direct_dte * 10.0))
                    _dte = max(1.0, min(30.0, _dte))
                else:
                    # 3) get_expiry_week_info fallback
                    from core.utils import get_expiry_week_info
                    _dte = float(get_expiry_week_info().get("days_to_expiry") or 1.0)
                    _dte = max(1.0, float(_dte))
            except Exception:
                _dte = 1.0

            # ── 4. 진폭 계산 ─────────────────────────────────────────────
            amp = calc_expected_amplitude(
                underlying_price=float(current_price),
                atm_iv=float(_atm_iv),
                days_to_expiry=float(_dte),
                session_high=float(session_high),
                session_low=float(session_low),
                session_open=float(session_open),
                oi_levels=oi_levels,
                sigma_multiplier=float(getattr(self, "_sigma_multiplier", 1.0)),
            )
            amp["_amplitude_source"] = str(source)
            amp["_sigma_multiplier"] = float(getattr(self, "_sigma_multiplier", 1.0))
            amp["_dte_weight_norm_ref"] = float(dte_weight_norm or 0.0)
            # 방안C: 실현 진폭 EMA 갱신 (하루 1회, 장중 진폭이 쌓인 시점)
            try:
                self._update_realized_amplitude_ema(amp)
            except Exception as _e:
                logger.debug("오류 무시: %s", _e)
            # EMA 보정값을 amp에 기록 (context_builder 참조용)
            try:
                _ema = float(self._realized_amplitude_ema or 0.0)
                amp["_realized_amplitude_ema"] = round(_ema, 2)
                # EMA가 유효하면 expected_amplitude_pt를 IV+OI 혼합값과 EMA로 재보정
                # 가중치: IV+OI 60% + EMA 40% (EMA 미초기화 시 IV+OI 단독 유지)
                if _ema > 0.0:
                    _cur_exp = float(amp.get("expected_amplitude_pt") or 0.0)
                    if _cur_exp > 0.0:
                        _blended = 0.6 * _cur_exp + 0.4 * _ema
                        _blended = max(3.0, min(50.0, _blended))
                        amp["expected_amplitude_pt"] = round(_blended, 2)
                        amp["_ema_blended"] = True
                    else:
                        amp["_ema_blended"] = False
                else:
                    amp["_realized_amplitude_ema"] = 0.0
                    amp["_ema_blended"] = False
            except Exception as _e:
                logger.debug("오류 무시: %s", _e)
            # 피드백 루프: 실현 진폭 결과로 다음 예측 배율 조정
            try:
                self._update_sigma_multiplier(amp)
            except Exception as _e:
                logger.debug("오류 무시: %s", _e)
            return amp

        except Exception:
            return _empty

