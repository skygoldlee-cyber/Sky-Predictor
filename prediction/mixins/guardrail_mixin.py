"""Mixin extracted from prediction/pipeline.py.

이 파일은 PredictionPipeline의 일부를 Mixin으로 분리한 것입니다.
직접 인스턴스화하지 마십시오. PredictionPipeline을 통해 사용하세요.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class GuardrailMixin:
    """Mixin: GuardrailMixin methods extracted from PredictionPipeline."""

    def _apply_option_guardrail(self, *, signal: str, confidence: str, opt_snap: Dict[str, Any]) -> tuple[str, str, Optional[str]]:
        try:
            atm_spread_pct = float((opt_snap or {}).get("atm_spread_pct") or 0.0)
            atm_liq_log = float((opt_snap or {}).get("atm_liquidity_log") or 0.0)
            opt_call_cnt = int((opt_snap or {}).get("call_count") or 0)
            opt_put_cnt = int((opt_snap or {}).get("put_count") or 0)
            s_up = str(signal or "").strip().upper()
            c_up = str(confidence or "").strip().upper()

            has_atm_opt = (opt_call_cnt > 0) or (opt_put_cnt > 0)
            wide = atm_spread_pct >= float(self._guard_atm_spread_pct_thr)
            illiq = atm_liq_log <= float(self._guard_atm_liq_log_thr)
            # 장종료 전후/비정상 호가 구간 대응: 극단 스프레드는 즉시 HOLD 처리
            extreme_spread_thr = max(10.0, float(self._guard_atm_spread_pct_thr) * 6.0)
            extreme_wide = atm_spread_pct >= extreme_spread_thr
            market_closed = bool(getattr(getattr(self, "tick_processor", None), "market_closed", False))

            if has_atm_opt:
                if market_closed and s_up in ("BUY", "SELL"):
                    return "HOLD", "LOW", f"option_guardrail_market_closed(spread={atm_spread_pct:.2f}%,liq={atm_liq_log:.2f})"

                if extreme_wide and s_up in ("BUY", "SELL"):
                    return "HOLD", "LOW", f"option_guardrail_extreme_spread(spread={atm_spread_pct:.2f}%,thr={extreme_spread_thr:.2f}%)"

            if has_atm_opt and (wide or illiq):
                if (s_up in ("BUY", "SELL")) and wide and illiq:
                    return "HOLD", "LOW", f"option_guardrail_hold(spread={atm_spread_pct:.2f}%,liq={atm_liq_log:.2f})"
                if c_up == "HIGH":
                    return str(signal), "MEDIUM", f"option_guardrail_downgrade(spread={atm_spread_pct:.2f}%,liq={atm_liq_log:.2f})"
                if c_up == "MEDIUM":
                    return str(signal), "LOW", f"option_guardrail_downgrade(spread={atm_spread_pct:.2f}%,liq={atm_liq_log:.2f})"
        except Exception:
            pass
        return str(signal), str(confidence), None

    def _apply_basis_guardrail(
        self, *, signal: str, confidence: str, current_price: float
    ) -> tuple[str, str, Optional[float], Optional[float], Optional[str]]:
        spot_index = None
        basis = None
        try:
            # symbol 확인 (KP200 선물 vs KOSPI 지수)
            ad = getattr(self, "_adaptive_indicator", {})
            symbol = str(ad.get("symbol", "") or "").strip()

            if "KP200" in symbol or "선물" in symbol:
                # KP200 선물: FC0 틱의 sbasis 필드를 직접 사용
                # sbasis = KP200선물 - KP200현물 (eBest 서버가 직접 계산해 제공)
                #
                # 1순위: FC0 틱의 sbasis 필드 (가장 정확)
                try:
                    tp = getattr(self, "tick_processor", None)
                    if tp is not None and callable(getattr(tp, "get_latest_sbasis", None)):
                        _sbasis = tp.get_latest_sbasis()
                        if _sbasis is not None:
                            basis = float(_sbasis)
                            # spot_index = 선물가 - basis (역산)
                            spot_index = float(current_price) - basis
                except Exception:
                    pass

                # 2순위: FC0 k200_index로 직접 계산 (sbasis 미수신 시 fallback)
                if basis is None:
                    try:
                        if tp is not None and callable(getattr(tp, "get_latest_k200_index", None)):
                            kp200_spot = float(tp.get_latest_k200_index() or 0.0)
                            if kp200_spot > 0.0:
                                spot_index = kp200_spot
                                basis = float(current_price) - kp200_spot
                    except Exception:
                        pass

                # 3순위: IJ_ key="101" 스냅샷 (FC0 틱 미수신 시 최종 fallback)
                if basis is None:
                    if isinstance(self._ij_realtime_snapshot, dict) and self._ij_realtime_snapshot:
                        kp200_spot = float(self._ij_realtime_snapshot.get("jisu") or 0.0)
                        if kp200_spot > 0.0:
                            spot_index = kp200_spot
                            basis = float(current_price) - kp200_spot
            else:
                # KOSPI 지수: IJ_ key="001" snap의 jisu = KOSPI 현물지수
                if isinstance(self._ij_realtime_snapshot, dict) and self._ij_realtime_snapshot:
                    spot = float(self._ij_realtime_snapshot.get("jisu") or 0.0)
                    if spot > 0.0:
                        spot_index = float(spot)
                        basis = float(current_price) - float(spot)
        except Exception:
            spot_index = None
            basis = None

        try:
            if spot_index is not None and basis is not None:
                basis_abs = abs(float(basis))
                hold_thr = float(self._guard_basis_hold_thr)
                downgrade_thr = float(self._guard_basis_downgrade_thr)

                s_up = str(signal or "").strip().upper()
                c_up = str(confidence or "").strip().upper()

                if basis_abs >= float(hold_thr) and s_up in ("BUY", "SELL"):
                    return "HOLD", "LOW", spot_index, basis, f"basis_guardrail_hold(basis={basis:.3f},thr={hold_thr:.1f})"
                if basis_abs >= float(downgrade_thr):
                    if c_up == "HIGH":
                        return str(signal), "MEDIUM", spot_index, basis, f"basis_guardrail_downgrade(basis={basis:.3f},thr={downgrade_thr:.1f})"
                    if c_up == "MEDIUM":
                        return str(signal), "LOW", spot_index, basis, f"basis_guardrail_downgrade(basis={basis:.3f},thr={downgrade_thr:.1f})"
        except Exception:
            pass

        return str(signal), str(confidence), spot_index, basis, None

    def _apply_parity_guardrail(
        self,
        *,
        signal: str,
        confidence: str,
        opt_snap: Optional[Dict[str, Any]],
    ) -> tuple[str, str, Optional[str]]:
        """만기주 패리티 이탈이 클 경우 신호 신뢰도를 낮추는 가드레일.

        조건:
            - option_feature_set == "v3"/"v4" 일 때만 동작한다.
            - |parity_divergence_score| >= 0.5 이고 dte_weight_norm >= 0.33 (만기 3일 이내):
              MEDIUM → LOW 강등.
            - |parity_divergence_score| >= 0.8 이고 dte_weight_norm >= 0.9 (만기 1일 이내):
              BUY/SELL → HOLD.

        Returns:
            (signal, confidence, reason_or_None)
        """
        if not getattr(self, "_guardrail_parity_enabled", self._option_feature_set in ("v3", "v4", "v5")):
            return str(signal), str(confidence), None
        if not isinstance(opt_snap, dict):
            return str(signal), str(confidence), None

        try:
            score = abs(float(opt_snap.get("parity_divergence_score") or 0.0))
            dte_w = float(opt_snap.get("dte_weight_norm") or 0.0)
        except Exception:
            return str(signal), str(confidence), None

        reason: Optional[str] = None

        try:
            s_up = str(signal or "").strip().upper()
            c_up = str(confidence or "").strip().upper()

            # 만기 당일(dte_weight_norm >= 1.0) + 강한 이탈 → HOLD
            # dte_weight_norm = min(1/(max(dte,0.1)*10), 1.0) → dte=0일 이면 1.0
            if dte_w >= 1.0 and score >= 0.8 and s_up in ("BUY", "SELL"):
                return "HOLD", "LOW", f"parity_divergence_critical(score={score:.2f},dte_w={dte_w:.2f})"

            # 만기 3일 이내(dte_weight_norm >= 0.033) + 중간 이탈 → MEDIUM→LOW 강등
            # (설계 원문: dte_w >= 0.33은 만기 3일 이내 기준 — 공식 변경 후 0.033으로 조정)
            if dte_w >= 0.033 and score >= 0.5:
                if c_up == "MEDIUM":
                    reason = f"parity_divergence(score={score:.2f},dte_w={dte_w:.2f})"
                    return str(signal), "LOW", reason
        except Exception:
            pass

        return str(signal), str(confidence), None

    def _apply_bleed_guardrail(
        self,
        *,
        signal: str,
        confidence: str,
        opt_snap: Optional[Dict[str, Any]],
    ) -> tuple[str, str, Optional[str]]:
        """프리미엄 블리드(Premium Bleed) 감지 시 신호 신뢰도를 낮추는 가드레일.

        option_feature_set == 'v4' 일 때만 동작한다.
        premium_bleed_score가 opt_snap에 포함되어 있어야 한다.
        (build_option_snapshot v4 경로에서 calc_premium_bleed() 결과를 포함시켜야 함.)

        조건:
            - premium_bleed_score <= -0.5 (강한 수축): MEDIUM → LOW 강등.
            - premium_bleed_score <= -0.8 (극단 수축): BUY/SELL → HOLD.
        """
        if not getattr(self, "_guardrail_bleed_enabled", self._option_feature_set in ("v4", "v5")):
            return str(signal), str(confidence), None
        if not isinstance(opt_snap, dict):
            return str(signal), str(confidence), None

        try:
            score = float(opt_snap.get("premium_bleed_score") or 0.0)
        except Exception:
            return str(signal), str(confidence), None

        try:
            s_up = str(signal or "").strip().upper()
            c_up = str(confidence or "").strip().upper()

            if score <= -0.8 and s_up in ("BUY", "SELL"):
                return "HOLD", "LOW", f"premium_bleed_critical(score={score:.2f})"

            if score <= -0.5 and c_up == "MEDIUM":
                return str(signal), "LOW", f"premium_bleed(score={score:.2f})"
        except Exception:
            pass

        return str(signal), str(confidence), None

    def _apply_oi_guardrail(
        self,
        *,
        signal: str,
        confidence: str,
        opt_snap: Optional[Dict[str, Any]],
        amplitude: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str, Optional[str]]:
        """OI 기반 지지저항 이탈·레짐 전환 시 신호를 보정하는 가드레일.

        조건 (option_feature_set == "v5" 일 때만 동작):
            1. Zero Gamma Level 근접(|dist| < 0.2%) + MEDIUM:
               → LOW 강등 (딜러 감마 방향 불확실, 전 신호 신뢰도 저하)
            2. Vol Trigger 하방(Dealer Short Gamma) + BUY + MEDIUM:
               → LOW 강등 (Short Gamma 구간 상승 배팅 위험)
            3. Call OI Peak 직전 0.3% 이내 + 집중도 >= 0.4 + BUY:
               → HOLD / LOW (강한 저항 직전 매수 억제)
            4. Put OI Peak 직전 0.3% 이내 + 집중도 >= 0.4 + SELL:
               → HOLD / LOW (강한 지지 직전 매도 억제)

        GR-02 수정: 조건이 중복 해당될 때 모든 reason을 누적하여 반환한다.
            (이전: if/elif 구조로 첫 번째 조건만 적용)
        GR-03 수정: dist 필드를 `or 99.0` 대신 None 체크로 읽어
            dist=0.0(ATM이 Peak)인 경우를 올바르게 처리한다.

        Returns:
            (signal, confidence, reason_or_None)
        """
        # GR-04 수정: v5 전용 제한 해제 — _oi_levels가 유효하면 모든 fs에서 동작.
        # v1~v4에서도 build_option_snapshot()이 _oi_levels를 저장하므로
        # Call/Put OI 저항 직전 BUY/SELL 억제 조건을 동일하게 적용할 수 있다.
        if not isinstance(opt_snap, dict):
            return str(signal), str(confidence), None

        # _oi_levels 우선, 없으면 opt_snap 직접 참조 (v5에서 직접 노출)
        oi = opt_snap.get("_oi_levels")
        if not isinstance(oi, dict) or not oi:
            # _guardrail_oi_enabled=True + _oi_levels 없음: opt_snap 직접 시도
            if not getattr(self, "_guardrail_oi_enabled", self._option_feature_set == "v5"):
                return str(signal), str(confidence), None
            oi = opt_snap

        try:
            above_vt  = float(oi.get("above_vol_trigger") if oi.get("above_vol_trigger") is not None else 1.0)
            call_conc = float(oi.get("call_oi_peak_norm") or 0.0)
            put_conc  = float(oi.get("put_oi_peak_norm") or 0.0)
            # GR-03 수정: `or 99.0` 제거 → dist=0.0(ATM Peak)도 정상값으로 처리.
            # None(키 없음)인 경우에만 99.0 기본값 사용.
            _call_dist_raw = oi.get("dist_to_call_peak")
            _put_dist_raw  = oi.get("dist_to_put_peak")
            call_dist = float(_call_dist_raw if _call_dist_raw is not None else 99.0)
            put_dist  = float(_put_dist_raw  if _put_dist_raw  is not None else 99.0)
            zgd       = abs(float(oi.get("zero_gamma_dist_pct") if oi.get("zero_gamma_dist_pct") is not None else 99.0))
            oi_range  = float(oi.get("oi_range_pct") or 0.0)
        except Exception:
            return str(signal), str(confidence), None

        # OI 데이터가 없으면 (장 시작 직후) 가드레일 비활성화
        if oi_range <= 0.0 and call_conc <= 0.0 and put_conc <= 0.0:
            try:
                fn = getattr(self, "_metrics_inc", None)
                if callable(fn):
                    fn("guardrail_oi_skipped_no_data")
            except Exception:
                pass
            return str(signal), str(confidence), None

        s_up = str(signal or "").strip().upper()
        c_up = str(confidence or "").strip().upper()

        # GR-02 수정: 복수 조건이 동시에 해당될 수 있으므로 reasons를 누적한다.
        # 신호/신뢰도는 가장 보수적인 보정(HOLD > BUY/SELL, LOW > MEDIUM)으로 수렴.
        reasons: list = []
        try:
            # 조건 1: Zero Gamma Level 근접 → MEDIUM 신호 불확실성 증가
            if zgd < 0.2 and c_up == "MEDIUM":
                c_up = "LOW"
                reasons.append(f"oi_zero_gamma_near(dist={zgd:.3f}%)")

            # 조건 2: Vol Trigger 하방 + BUY → Short Gamma 구간 매수 억제
            if above_vt < 1.0 and s_up == "BUY" and c_up in ("MEDIUM", "LOW"):
                c_up = "LOW"
                reasons.append("oi_vol_trigger_below(dealer_short_gamma)")

            # 조건 3: 강한 Call OI 저항 직전 BUY → HOLD
            # dist=0.0(ATM Peak) 포함: 0.0 <= call_dist < 0.3 범위로 확장.
            if call_conc >= 0.4 and 0.0 <= call_dist < 0.3 and s_up == "BUY":
                s_up = "HOLD"
                c_up = "LOW"
                reasons.append(
                    f"oi_call_resistance(conc={call_conc:.2f},dist={call_dist:.3f}%)"
                )

            # 조건 4: 강한 Put OI 지지 직전 SELL → HOLD
            # dist=0.0(ATM Peak) 포함: 0.0 <= put_dist < 0.3 범위로 확장.
            if put_conc >= 0.4 and 0.0 <= put_dist < 0.3 and s_up == "SELL":
                s_up = "HOLD"
                c_up = "LOW"
                reasons.append(
                    f"oi_put_support(conc={put_conc:.2f},dist={put_dist:.3f}%)"
                )

            # 조건 5 (Medium-10): OI Peak 근접 + 진폭 소진 이중 억제
            # amplitude 데이터가 전달된 경우에만 동작. 조건 3·4와 독립적으로 적용.
            if isinstance(amplitude, dict) and amplitude:
                try:
                    _exhaust = float(amplitude.get("amplitude_exhaustion") or 0.0)
                    _remain  = float(amplitude.get("remaining_amplitude_pt") or 0.0)
                    # 진폭 소진율 85% 이상 + OI Peak 0.5% 이내 → BUY/SELL 모두 억제
                    if _exhaust >= 0.85:
                        if call_dist < 0.5 and s_up == "BUY":
                            s_up = "HOLD"
                            c_up = "LOW"
                            reasons.append(
                                f"oi_amp_double_suppress_call("
                                f"exhaust={_exhaust:.0%},dist={call_dist:.3f}%,"
                                f"remain={_remain:.1f}pt)"
                            )
                        if put_dist < 0.5 and s_up == "SELL":
                            s_up = "HOLD"
                            c_up = "LOW"
                            reasons.append(
                                f"oi_amp_double_suppress_put("
                                f"exhaust={_exhaust:.0%},dist={put_dist:.3f}%,"
                                f"remain={_remain:.1f}pt)"
                            )
                except Exception:
                    pass
        except Exception:
            pass

        if reasons:
            return s_up, c_up, ",".join(reasons)
        return s_up, c_up, None

