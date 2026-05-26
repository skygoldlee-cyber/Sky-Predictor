"""Mixin extracted from prediction/pipeline.py.

이 파일은 PredictionPipeline의 일부를 Mixin으로 분리한 것입니다.
직접 인스턴스화하지 마십시오. PredictionPipeline을 통해 사용하세요.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from config import FUTURE_KNOWN_DIM, HORIZON_SEC
from core.utils import adaptive_uses_kospi_spot_index_minute_bars
from ..predictor import ModelInput
from ..features.features import MS5_KEYS, MS15_KEYS, build_sequence, calc_multiscale_features, calc_multiscale_features_15m
from ..features.time_features import build_time_features
from ..context_builder import build_llm_context, build_llm_prompt


@dataclass(slots=True)
class _NumericResult:
    t_res: Any
    ob_records: List[Dict[str, Any]]
    last_ob_snapshot: Dict[str, Any]
    ob_len: int
    prob: float
    signal: str
    confidence: str
    raw_signal: str
    raw_confidence: str
    guardrail_applied: bool
    guardrail_reason: str
    spot_index: Any
    basis: Any
    transformer_prob: float
    tft_prob: Any
    ensemble_method: str
    model_agreement: Any
    transformer_weight: Any


class PredictionMixin:
    """Mixin: PredictionMixin methods extracted from PredictionPipeline."""

    def _build_and_predict_numeric(
        self,
        *,
        df: "pd.DataFrame",
        now_dt: datetime,
        current_price: float,
        adaptive_features: Optional[Dict[str, float]],
        opt_snap: Dict[str, Any],
        candle_df=None,
    ):

        seq = None
        ob_records_snapshot: List[Dict[str, Any]] = []
        last_ob_snapshot: Dict[str, Any] = {}
        ob_len_snapshot = 0
        try:
            with self._ob_lock:
                ob_records_snapshot = list(self._ob_records)
                last_ob_snapshot = dict(self._last_ob_snapshot) if self._last_ob_snapshot else {}
                ob_len_snapshot = int(len(self._ob_records))
        except Exception:
            ob_records_snapshot = []
            last_ob_snapshot = {}
            ob_len_snapshot = 0

        try:
            # ── 멀티스케일 피처 계산 (5분봉 + 15분봉) ─────────────────────────
            multiscale_snap: Optional[Dict[str, pd.DataFrame]] = None
            if getattr(self, "_multiscale_5m", False) or getattr(self, "_multiscale_enabled", False):
                try:
                    # _tick_processor 의 futures_ohlcv (1분봉 DataFrame) 를 가져옴
                    tp = getattr(self, "_tick_processor", None)
                    raw_1m = None
                    if tp is not None:
                        try:
                            raw_1m = tp.get_futures_ohlcv()
                        except Exception:
                            raw_1m = None
                    if raw_1m is not None and not raw_1m.empty:
                        multiscale_snap = {}
                        # 5분봉 피처 계산
                        ms5_df = calc_multiscale_features(raw_1m)
                        if not ms5_df.empty:
                            multiscale_snap["ms5"] = ms5_df
                        # 15분봉 피처 계산
                        ms15_df = calc_multiscale_features_15m(raw_1m)
                        if not ms15_df.empty:
                            multiscale_snap["ms15"] = ms15_df
                except Exception:
                    multiscale_snap = None

            seq = build_sequence(
                ob_records_snapshot,
                candle_df,
                seq_len=int(self._seq_len),
                opt_features=opt_snap,
                adaptive_features=adaptive_features,
                opt_keys_override=list(self._opt_keys),
                multiscale_features=multiscale_snap,
            )
        except Exception:
            seq = None

        horizon_steps = int(max(1, int(getattr(self, "_tft_horizon", HORIZON_SEC) or HORIZON_SEC)))

        past_known = None
        try:
            pk_arr = np.zeros((int(self._seq_len), FUTURE_KNOWN_DIM), dtype=np.float32)
            tail = ob_records_snapshot[-int(self._seq_len) :]
            start_idx = int(self._seq_len) - int(len(tail))
            for i, rec in enumerate(tail):
                try:
                    ts = float((rec or {}).get("_ts_epoch") or 0.0)
                    dt = datetime.fromtimestamp(ts) if ts > 0 else now_dt
                    pk_arr[start_idx + i] = np.array(build_time_features(dt), dtype=np.float32)
                except Exception:
                    pass
            past_known = pk_arr
        except Exception:
            past_known = None

        future_known = None
        try:
            fk_arr = np.zeros((int(horizon_steps), FUTURE_KNOWN_DIM), dtype=np.float32)
            for step in range(int(horizon_steps)):
                try:
                    fdt = now_dt + timedelta(seconds=int(step))
                    fk_arr[int(step)] = np.array(build_time_features(fdt), dtype=np.float32)
                except Exception:
                    pass
            future_known = fk_arr
        except Exception:
            future_known = None

        try:
            last_ob_snapshot = dict(last_ob_snapshot or {})
            last_ob_snapshot.update(self._compute_flow_features(now_dt=now_dt, current_price=float(current_price)))
        except Exception:
            last_ob_snapshot = last_ob_snapshot

        model_input = ModelInput(
            sequence=seq,
            past_known=past_known,
            future_known=future_known,
            feature_snapshot=last_ob_snapshot,
            meta={
                "prediction_minutes": int(self.prediction_minutes),
                "seq_len": int(self._seq_len),
                "ob_records_len": int(ob_len_snapshot),
            },
            schema_version="v2_ob7_cd5_opt4_tft",
        )
        t_res = self.numeric_predictor.predict(input=model_input)
        return t_res, seq, past_known, future_known, ob_records_snapshot, last_ob_snapshot, ob_len_snapshot

    def _run_numeric_prediction_and_guardrails(
        self,
        *,
        df: "pd.DataFrame",
        now_dt: datetime,
        current_price: float,
        adaptive_features: Optional[Dict[str, float]],
        opt_snap: Dict[str, Any],
        amplitude: Optional[Dict[str, Any]] = None,
    ) -> "_NumericResult":
        t_res, _seq, _past_known, _future_known, ob_records_snapshot, last_ob_snapshot, ob_len_snapshot = (
            self._build_and_predict_numeric(
                df=df,
                now_dt=now_dt,
                current_price=float(current_price),
                adaptive_features=adaptive_features,
                opt_snap=opt_snap,
            )
        )

        prob = float(getattr(t_res, "prob", 0.5) or 0.5)
        signal = str(getattr(t_res, "signal", "HOLD") or "HOLD")
        confidence = str(getattr(t_res, "confidence", "LOW") or "LOW")

        raw_signal = str(signal)
        raw_confidence = str(confidence)
        guardrail_applied = False
        guardrail_reason = ""

        spot_index = None
        basis = None

        signal2, conf2, opt_reason = self._apply_option_guardrail(signal=signal, confidence=confidence, opt_snap=opt_snap)
        try:
            if str(signal2) != str(signal) or str(conf2) != str(confidence):
                guardrail_applied = True
                _opt_tag = str(opt_reason) if opt_reason else "option_guardrail"
                guardrail_reason = (guardrail_reason + "," if guardrail_reason else "") + _opt_tag
                # [P1-FIX-2] 가드레일 발동 메트릭
                self._metrics_inc("guardrail_fires_option")
        except Exception:
            pass
        signal, confidence = str(signal2), str(conf2)
        signal, confidence, spot_index, basis, basis_reason = self._apply_basis_guardrail(
            signal=signal,
            confidence=confidence,
            current_price=float(current_price),
        )
        try:
            if str(signal) != str(raw_signal) or str(confidence) != str(raw_confidence):
                if "basis_guardrail" not in str(guardrail_reason or ""):
                    guardrail_applied = True
                    _basis_tag = str(basis_reason) if basis_reason else "basis_guardrail"
                    guardrail_reason = (guardrail_reason + "," if guardrail_reason else "") + _basis_tag
                    # [P1-FIX-2] 가드레일 발동 메트릭
                    self._metrics_inc("guardrail_fires_basis")
        except Exception:
            pass

        # 패리티 가드레일: _guardrail_parity_enabled 플래그로 제어 (기본값: v3+)
        try:
            signal3, conf3, parity_reason = self._apply_parity_guardrail(
                signal=signal,
                confidence=confidence,
                opt_snap=opt_snap,
            )
            if parity_reason:
                guardrail_applied = True
                guardrail_reason = (guardrail_reason + "," if guardrail_reason else "") + parity_reason
                self._metrics_inc("guardrail_fires_parity")
            signal, confidence = str(signal3), str(conf3)
        except Exception:
            pass

        # 블리드 가드레일: _guardrail_bleed_enabled 플래그로 제어 (기본값: v4+)
        try:
            signal_b, conf_b, bleed_reason = self._apply_bleed_guardrail(
                signal=signal,
                confidence=confidence,
                opt_snap=opt_snap,
            )
            if bleed_reason:
                guardrail_applied = True
                guardrail_reason = (guardrail_reason + "," if guardrail_reason else "") + bleed_reason
                self._metrics_inc("guardrail_fires_bleed")
            signal, confidence = str(signal_b), str(conf_b)
        except Exception:
            pass

        # OI 가드레일: _guardrail_oi_enabled 플래그로 제어 (기본값: v5)
        # [P1-FIX-1] amplitude는 호출자(get_prediction)에서 사전 계산 후 파라미터로 전달.
        try:
            signal4, conf4, oi_reason = self._apply_oi_guardrail(
                signal=signal,
                confidence=confidence,
                opt_snap=opt_snap,
                amplitude=amplitude,
            )
            if oi_reason:
                guardrail_applied = True
                guardrail_reason = (guardrail_reason + "," if guardrail_reason else "") + oi_reason
                self._metrics_inc("guardrail_fires_oi")
            signal, confidence = str(signal4), str(conf4)
        except Exception as _oi_err:
            logger.warning("[OI_GUARDRAIL] 예외 발생 — confidence를 LOW로 강등: %s", _oi_err)
            # 실패 안전(fail-safe): 억제 없이 원신호를 통과시키지 않고 confidence 강등
            confidence = "LOW"
            guardrail_applied = True
            guardrail_reason = (guardrail_reason + "," if guardrail_reason else "") + "oi_guardrail_error"
            self._metrics_inc("guardrail_fires_oi_error")

        transformer_prob = float(getattr(t_res, "transformer_prob", prob) or prob)
        tft_prob = getattr(t_res, "tft_prob", None)
        ensemble_method = str(getattr(t_res, "ensemble_method", "transformer_only") or "transformer_only")
        model_agreement = getattr(t_res, "agreement", None)

        transformer_weight = None
        try:
            fn_w = getattr(self.numeric_predictor, "get_transformer_weight", None)
            if callable(fn_w):
                transformer_weight = float(fn_w())
        except Exception:
            transformer_weight = None

        return _NumericResult(
            t_res=t_res,
            ob_records=ob_records_snapshot,
            last_ob_snapshot=last_ob_snapshot,
            ob_len=int(ob_len_snapshot),
            prob=float(prob),
            signal=str(signal),
            confidence=str(confidence),
            raw_signal=str(raw_signal),
            raw_confidence=str(raw_confidence),
            guardrail_applied=bool(guardrail_applied),
            guardrail_reason=str(guardrail_reason),
            spot_index=spot_index,
            basis=basis,
            transformer_prob=float(transformer_prob),
            tft_prob=tft_prob,
            ensemble_method=str(ensemble_method),
            model_agreement=model_agreement,
            transformer_weight=transformer_weight,
        )

    def _prepare_prediction_inputs(
        self,
        *,
        now_dt: datetime,
        current_price: float,
    ) -> tuple["pd.DataFrame", Optional[Dict[str, float]], str, Any, Any, Dict[str, Any], Optional[str]]:
        try:
            self.tick_processor.update_option_minute_allowed_symbols(underlying_price=float(current_price))
        except Exception:
            pass

        try:
            df = self._get_minute_df_or_error(warmup_bars=int(self._adaptive_warmup_bars or 45))
        except RuntimeError:
            w = int(self._adaptive_warmup_bars or 45)
            if adaptive_uses_kospi_spot_index_minute_bars(self):
                df2 = self.tick_processor.get_kospi_minute_df(w)
            else:
                df2 = self.tick_processor.get_futures_minute_df(w)
            raise RuntimeError(
                f"insufficient_minutes:{0 if df2 is None else len(df2)}:{int(self.min_minute_bars_required)}"
            )

        adaptive_features: Optional[Dict[str, float]] = None
        adaptive_context: str = ""
        adaptive_supertrend_state: Any = None
        adaptive_zigzag_state: Any = None
        model_outputs: Dict[str, Any] = {}
        try:
            adaptive_features, adaptive_context, adaptive_supertrend_state, adaptive_zigzag_state, model_outputs = (
                self._compute_adaptive_bundle(df=df, now_dt=now_dt)
            )
        except Exception as e:
            raise RuntimeError(f"adaptive_failed:{e}")

        regime = None
        try:
            regime = self._compute_regime(
                adaptive_features=adaptive_features,
                adaptive_supertrend_state=adaptive_supertrend_state,
            )
        except Exception:
            regime = None

        return df, adaptive_features, adaptive_context, adaptive_supertrend_state, adaptive_zigzag_state, model_outputs, regime

    def _build_llm_prompt(self, *, snapshot: Dict[str, Any], ob_records: List[Dict[str, Any]], adaptive_context: str, amplitude: Optional[Dict[str, Any]] = None, prob_interval: Optional[tuple] = None) -> tuple[str, str]:
        context = build_llm_context(
            snapshot=snapshot,
            ob_records=ob_records,
            adaptive_context=str(adaptive_context or "") or None,
            amplitude=amplitude,
            prob_interval=prob_interval,
        )
        system, user = build_llm_prompt(context=context, prediction_minutes=int(self.prediction_minutes))
        return system, user

    def _compute_fo0_diagnostics(self) -> Optional[float]:
        fo0_age_sec = None
        try:
            if self._last_fo0_seen_epoch is not None:
                fo0_age_sec = float(time.time()) - float(self._last_fo0_seen_epoch)
        except Exception:
            fo0_age_sec = None

        stale = fo0_age_sec is None or float(fo0_age_sec) > float(self._fo0_stale_sec)
        now_epoch = float(time.time())
        cooldown_sec = 60.0
        warned_recently = (
            self._last_fo0_stale_warn_epoch is not None
            and (now_epoch - float(self._last_fo0_stale_warn_epoch)) < float(cooldown_sec)
        )
        if stale and (not warned_recently):
            # CON-05: len(_ob_records) 조회도 _ob_lock 하에 수행 (다른 스레드의 append와 race 방지)
            with self._ob_lock:
                ob_len_for_log = len(self._ob_records)
            logger.warning(
                "[FO0_STALE] age_sec=%s ob_len=%s (no FO0 in last %ss; check subscription/feed)",
                fo0_age_sec,
                ob_len_for_log,
                int(self._fo0_stale_sec),
            )
            self._last_fo0_stale_warn_epoch = float(now_epoch)
        return fo0_age_sec

    def _build_prediction_output(
        self,
        *,
        now_dt: datetime,
        start: float,
        current_price: float,
        spot_index: Optional[float],
        basis: Optional[float],
        prob: float,
        signal: str,
        confidence: str,
        transformer_prob: float,
        tft_prob: Optional[float],
        ensemble_method: str,
        model_agreement: Optional[bool],
        transformer_weight: Optional[float],
        regime: Optional[str],
        model_outputs: Optional[Dict[str, Any]],
        last_ob_snapshot: Dict[str, Any],
        ob_records_snapshot: List[Dict[str, Any]],
        ob_len_snapshot: int,
        fo0_age_sec: Optional[float],
        llm_action: str,
        llm_provider: str,
        llm_timed_out: bool,
        risk_level: str,
        rationale: str,
        caution: str,
        llm_raw: str,
        opt_snap: Optional[Dict[str, Any]],
        amplitude_snapshot: Optional[Dict[str, Any]] = None,
        guardrail_applied: bool = False,
        guardrail_reason: str = "",
        prob_lower: Optional[float] = None,
        prob_upper: Optional[float] = None,
        adaptive_zigzag_state: Optional[Any] = None,
        llm_disabled: bool = False,
    ) -> Dict[str, Any]:
        """예측 결과 dict를 단일 위치에서 구성한다.

        get_prediction() 내 동일 구조 dict가 2개소에 중복 정의되어 있던 것을 통합.
        한쪽만 수정되어 필드 불일치가 발생하는 문제를 원천 차단한다.
        """
        out: Dict[str, Any] = {
            "prediction_time": now_dt.isoformat(),
            "prediction_minutes": int(self.prediction_minutes),
            "target_time": (now_dt + timedelta(minutes=int(self.prediction_minutes))).isoformat(),
            "current_price": float(current_price),
            "spot_index": float(spot_index) if spot_index is not None else None,
            "basis": float(basis) if basis is not None else None,
            "prob": float(prob),
            "prob_lower": float(prob_lower) if prob_lower is not None else None,
            "prob_upper": float(prob_upper) if prob_upper is not None else None,
            "signal": str(signal),
            "confidence": str(confidence),
            "transformer_prob": float(transformer_prob),
            "tft_prob": float(tft_prob) if tft_prob is not None else None,
            "ensemble_method": str(ensemble_method),
            "model_agreement": model_agreement if model_agreement is None else bool(model_agreement),
            "transformer_weight": float(transformer_weight) if transformer_weight is not None else None,
            "regime": regime,
            "model_outputs": dict(model_outputs) if isinstance(model_outputs, dict) and model_outputs else model_outputs,
            "orderbook": dict(last_ob_snapshot) if last_ob_snapshot else None,
            "ob_records": list(ob_records_snapshot) if isinstance(ob_records_snapshot, list) else [],
            "ob_records_len": int(ob_len_snapshot),
            "fo0_age_sec": float(fo0_age_sec) if fo0_age_sec is not None else None,
            "llm_action": str(llm_action),
            "llm_provider": str(llm_provider),
            "llm_timed_out": bool(llm_timed_out),
            "llm_disabled": bool(llm_disabled),
            "risk_level": str(risk_level),
            "rationale": str(rationale),
            "caution": str(caution),
            "consensus": bool(str(signal).upper() == str(llm_action).upper()),
            "options": dict(opt_snap) if opt_snap else None,
            "amplitude": dict(amplitude_snapshot) if amplitude_snapshot else None,
            "guardrail_reason": str(guardrail_reason or ""),
        }
        try:
            mo = model_outputs if isinstance(model_outputs, dict) else {}
            out["pivot_candidate_probability"] = str(
                mo.get("pivot_candidate_probability") or ""
            )
            out["pivot_candidate_reason"] = str(
                mo.get("pivot_candidate_reason") or ""
            )
        except Exception:
            out["pivot_candidate_probability"] = ""
            out["pivot_candidate_reason"] = ""

        # pivot_confirmed: adaptive_zigzag_state에서 확정 피봇 정보 가져오기
        try:
            pivot_confirmed = "-"
            pivot_candidate_time = "-"
            pivot_confirmed_time = "-"
            pivot_lag = "-"
            # adaptive_zigzag_state가 None이면 _adaptive_last_zigzag_state로 대체
            _zz_state = adaptive_zigzag_state
            if _zz_state is None:
                _zz_state = getattr(self, "_adaptive_last_zigzag_state", None)

            if _zz_state is not None:
                # recent_swings에서 가장 최근 확정 피봇 찾기
                recent_swings = list(getattr(_zz_state, "recent_swings", []) or [])
                confirmed_swings = [s for s in recent_swings if getattr(s, "confirmed", False)]
                confirmed_count = int(getattr(_zz_state, "confirmed_pivot_count", 0) or 0)
                logger.debug("[PredictionMixin] recent_swings: %d, confirmed_swings: %d, confirmed_count: %d",
                           len(recent_swings), len(confirmed_swings), confirmed_count)

                if confirmed_swings:
                    # 가장 최근 확정 피봇 유형 확인
                    last_swing_high_confirm_time = str(getattr(_zz_state, "last_swing_high_confirm_time", "") or "")
                    last_swing_low_confirm_time = str(getattr(_zz_state, "last_swing_low_confirm_time", "") or "")
                    last_swing_high_lag_bars = int(getattr(_zz_state, "last_swing_high_lag_bars", 0) or 0)
                    last_swing_low_lag_bars = int(getattr(_zz_state, "last_swing_low_lag_bars", 0) or 0)
                    logger.debug("[PredictionMixin] high_confirm_time: %s, low_confirm_time: %s, high_lag: %d, low_lag: %d",
                               last_swing_high_confirm_time, last_swing_low_confirm_time,
                               last_swing_high_lag_bars, last_swing_low_lag_bars)

                    # 가장 최근 확정 시각이 있는 피봇 확인
                    if last_swing_high_confirm_time and (not last_swing_low_confirm_time or last_swing_high_confirm_time >= last_swing_low_confirm_time):
                        pivot_type = "HIGH"
                        pivot_candidate_time = str(getattr(_zz_state, "last_swing_high_time", "") or "")
                        pivot_confirmed_time = last_swing_high_confirm_time
                        pivot_lag_bars = last_swing_high_lag_bars
                    elif last_swing_low_confirm_time:
                        pivot_type = "LOW"
                        pivot_candidate_time = str(getattr(_zz_state, "last_swing_low_time", "") or "")
                        pivot_confirmed_time = last_swing_low_confirm_time
                        pivot_lag_bars = last_swing_low_lag_bars
                    else:
                        # recent_swings에서 가장 최근 확정 피봇 확인
                        last_confirmed = confirmed_swings[-1]
                        swing_type = str(getattr(last_confirmed, "swing_type", "") or "").upper()
                        pivot_type = "HIGH" if "HIGH" in swing_type else "LOW"
                        pivot_lag_bars = 0
                        logger.debug("[PredictionMixin] fallback to recent_swings, pivot_type: %s", pivot_type)

                    if pivot_lag_bars > 0:
                        pivot_lag = f"{pivot_lag_bars}봉"

                    pivot_confirmed = f"{pivot_type} (총 {confirmed_count}개)"
                    logger.debug("[PredictionMixin] pivot info: %s, %s, %s, %s",
                               pivot_confirmed, pivot_candidate_time, pivot_confirmed_time, pivot_lag)
                else:
                    pivot_confirmed = "미확정"
            else:
                pivot_confirmed = "ZigZag 미준비"
            out["pivot_confirmed"] = str(pivot_confirmed)
            out["pivot_candidate_time"] = str(pivot_candidate_time)
            out["pivot_confirmed_time"] = str(pivot_confirmed_time)
            out["pivot_lag"] = str(pivot_lag)
        except Exception as e:
            logger.warning("[PredictionMixin] 피봇 정보 가져오기 실패: %s", e, exc_info=True)
            out["pivot_confirmed"] = "-"
            out["pivot_candidate_time"] = "-"
            out["pivot_confirmed_time"] = "-"
            out["pivot_lag"] = "-"
        if llm_raw:
            out["llm_raw"] = llm_raw
        try:
            self._last_result = dict(out)
        except Exception:
            pass
        self._metrics_inc("predictions")
        self._metrics_set("last_latency_ms", float(time.time() - start) * 1000.0)

        # guardrail metrics: HOLD 전환 + confidence 강등 케이스를 모두 추적
        try:
            # 1) signal이 HOLD로 바뀐 경우
            _signal_became_hold = (
                str(out.get("signal", "")).upper() == "HOLD"
                and bool(out.get("guardrail_reason") or "")
            )
            # 2) guardrail 발동으로 confidence가 강등된 경우 (HOLD 아니어도 포함)
            _confidence_downgraded = bool(guardrail_applied) and bool(out.get("guardrail_reason") or "")

            if _signal_became_hold or _confidence_downgraded:
                self._metrics_inc("guardrail_fires_total")
            if _signal_became_hold:
                self._metrics_inc("guardrail_hold_count")

            _total_preds = int(self._metrics_get("predictions") or 1)
            _hold_count  = int(self._metrics_get("guardrail_hold_count") or 0)
            _fire_count  = int(self._metrics_get("guardrail_fires_total") or 0)
            self._metrics_set(
                "guardrail_hold_ratio",
                round(float(_hold_count) / float(max(1, _total_preds)), 4),
            )
            self._metrics_set(
                "guardrail_fire_ratio",
                round(float(_fire_count) / float(max(1, _total_preds)), 4),
            )
        except Exception:
            pass

        return out

    def get_prediction(
        self,
        *,
        off_boundary: bool = False,
        _now: "Optional[datetime]" = None,
        auto_mode: bool = False,
    ) -> Dict[str, Any]:
        """Compute a combined numeric + LLM prediction output.

        Args:
            off_boundary: True이면 정규 예측 주기 사이의 휴리스틱 트리거에 의한 호출임을
                나타낸다. rate-limit 상태에서는 LLM을 건너뛰고 transformer 신호를 그대로
                반환한다.
            _now: 테스트용 시각 오버라이드. None이면 실제 현재 시각을 사용한다.
            auto_mode: 자동 트리거 모드 플래그. 현재는 메타데이터 목적으로 전달된다.

        Returns a dict containing at least:
        - `prob`, `signal`, `confidence`
        - `llm_action`, `llm_provider`, `risk_level`, `rationale`, `caution`
        - `orderbook`, `ob_records_len`, `fo0_age_sec`
        - `consensus`

        On insufficient data, returns an error dict with `error` and `message`.
        """
        start = time.time()
        try:
            now_dt = self._get_now_dt(now_override=_now)
            current_price = self._get_current_price_or_error()
        except Exception as e:
            if str(e) == "no_price":
                self._metrics_inc("prediction_failures")
                return {"error": "insufficient_data", "message": "선물 틱 데이터가 부족합니다"}
            raise

        # [IMP-LLM-01] 캐시 키에서 price_bucket 계산용 현재가 저장
        try:
            self._last_price_for_cache = float(current_price)
        except Exception:
            pass

        self._update_fc0_stale_detection()

        try:
            self._maybe_process_feedback(now_dt=now_dt, current_price=float(current_price))
        except Exception:
            pass

        try:

            try:
                (
                    df,
                    adaptive_features,
                    adaptive_context,
                    adaptive_supertrend_state,
                    adaptive_zigzag_state,
                    model_outputs,
                    regime,
                ) = self._prepare_prediction_inputs(now_dt=now_dt, current_price=float(current_price))
            except RuntimeError as e:
                msg = str(e)
                if msg.startswith("insufficient_minutes:"):
                    try:
                        parts = msg.split(":")
                        cur_n = int(parts[1]) if len(parts) > 1 else 0
                    except Exception:
                        cur_n = 0
                    self._metrics_inc("prediction_failures")
                    return {
                        "error": "insufficient_minutes",
                        "message": f"분봉 데이터 부족 (현재: {cur_n}개, 필요: {int(self.min_minute_bars_required)}개)",
                    }
                if msg.startswith("adaptive_failed:"):
                    self._metrics_inc("prediction_failures")
                    return {"error": "adaptive_failed", "message": msg.split(":", 1)[1] if ":" in msg else msg}
                raise

            opt_snap = self._build_option_snapshot_safe(current_price=float(current_price), update_prev=False)
            try:
                if isinstance(opt_snap, dict):
                    opt_snap["_tick_flow"] = self._build_option_tick_flow_snapshot(current_price=float(current_price))
            except Exception:
                pass

            # [P1-FIX-1] amplitude를 _run_numeric_prediction_and_guardrails 호출 전에
            # 미리 계산해 OI 가드레일에 정상 전달한다. (dir() 버그 수정 연동)
            try:
                _pre_amplitude = self._calc_amplitude_snapshot(
                    current_price=float(current_price),
                    atm_iv=float((opt_snap or {}).get("atm_iv") or 0.0),
                    dte_weight_norm=float((opt_snap or {}).get("dte_weight_norm") or 0.0),
                    oi_levels=(opt_snap or {}).get("_oi_levels") or None,
                )
            except Exception:
                _pre_amplitude = None

            # [IMP-ENS-02] 레짐에 따라 앙상블 가중치 창 크기 및 초기 편향 갱신
            try:
                _ep = self.numeric_predictor
                _set_regime_fn = getattr(_ep, "set_regime", None)
                _set_dte_fn    = getattr(_ep, "set_dte_window", None)
                if callable(_set_regime_fn):
                    _set_regime_fn(regime)
                if callable(_set_dte_fn):
                    _set_dte_fn(float((opt_snap or {}).get("dte_weight_norm") or 0.0))
            except Exception:
                pass

            try:
                _nr = self._run_numeric_prediction_and_guardrails(
                    df=df,
                    now_dt=now_dt,
                    current_price=float(current_price),
                    adaptive_features=adaptive_features,
                    opt_snap=opt_snap,
                    amplitude=_pre_amplitude,
                )
                t_res              = _nr.t_res
                ob_records_snapshot = _nr.ob_records
                last_ob_snapshot   = _nr.last_ob_snapshot
                ob_len_snapshot    = _nr.ob_len
                prob               = _nr.prob
                signal             = _nr.signal
                confidence         = _nr.confidence
                raw_signal         = _nr.raw_signal
                raw_confidence     = _nr.raw_confidence
                guardrail_applied  = _nr.guardrail_applied
                guardrail_reason   = _nr.guardrail_reason
                spot_index         = _nr.spot_index
                basis              = _nr.basis
                transformer_prob   = _nr.transformer_prob
                tft_prob           = _nr.tft_prob
                ensemble_method    = _nr.ensemble_method
                model_agreement    = _nr.model_agreement
                transformer_weight = _nr.transformer_weight
            except Exception as e:
                self._metrics_inc("prediction_failures")
                return {"error": "numeric_failed", "message": str(e)}

            # ── [PIVOT-OVERRIDE] 피봇 확정 시 signal 직접 오버라이드 ──────────────
            # heuristic.action이 zigzag_pivot_low(L)->BUY 또는 zigzag_pivot_high(H)->SELL
            # 일 때 Transformer 수치 기반 signal을 피봇 신호로 교체한다.
            #
            # 설계 원칙:
            #   - Transformer/가드레일 단계 이후에 적용 → 가드레일을 우회하지 않음
            #   - LLM 단계 이전에 signal 교체 → LLM이 교체된 signal 기준으로 판단
            #   - consensus = (signal == llm_action) 계산에도 정상 반영
            #   - reason은 guardrail_reason에 태그로 기록 (기존 이력 보존)
            #   - 피봇이 없으면(HOLD) 아무것도 변경하지 않음
            try:
                _h_info = (model_outputs or {}).get("heuristic") or {}
                _h_action = str(_h_info.get("action") or "").upper().strip()
                _h_reason = str(_h_info.get("reason") or "")
                _h_confidence = str(_h_info.get("confidence") or "LOW")
                _is_pivot_override = (
                    _h_action in ("BUY", "SELL")
                    and (
                        "zigzag_pivot_low(L)->BUY" in _h_reason
                        or "zigzag_pivot_high(H)->SELL" in _h_reason
                    )
                )
                if _is_pivot_override:
                    _prev_signal = signal
                    signal = _h_action
                    # [P7] ML-휴리스틱 앙상블 개선: confidence 통합
                    # 휴리스틱 confidence를 기본으로 사용하되, ML 확률이 높으면 가중치 부여
                    try:
                        transformer_prob = getattr(t_res, "transformer_prob", None)
                        if transformer_prob is not None and _h_confidence == "MEDIUM":
                            # ML 확률이 0.7 이상이면 confidence를 HIGH로 승격
                            if transformer_prob >= 0.7:
                                confidence = "HIGH"
                                logger.info(
                                    "[PIVOT-OVERRIDE-ENSEMBLE] ML 확률 %.2f >= 0.7 → confidence 승격 MEDIUM→HIGH",
                                    transformer_prob
                                )
                        elif _h_confidence == "HIGH":
                            confidence = "HIGH"
                        else:
                            confidence = _h_confidence
                    except Exception:
                        confidence = _h_confidence
                    
                    _tag = f"pivot_override:{_prev_signal}→{signal}(conf={confidence})"
                    guardrail_reason  = (
                        (guardrail_reason + "," if guardrail_reason else "")
                        + _tag
                    )
                    logger.info(
                        "[PIVOT-OVERRIDE] signal %s→%s confidence=%s reason=%s",
                        _prev_signal, signal, confidence, _h_reason,
                    )
            except Exception:
                pass

            try:
                if hasattr(t_res, "transformer_prob") and hasattr(t_res, "tft_prob"):
                    try:
                        ts_epoch = float(now_dt.timestamp())
                    except Exception:
                        ts_epoch = float(time.time())
                    horizon_sec = float(max(1, int(self.prediction_minutes))) * 60.0
                    self._feedback_queue.append(
                        {
                            "ts_epoch": float(ts_epoch),
                            "target_ts_epoch": float(ts_epoch + float(horizon_sec)),
                            "price": float(current_price),
                            "transformer_prob": float(getattr(t_res, "transformer_prob")),
                            "tft_prob": (None if getattr(t_res, "tft_prob", None) is None else float(getattr(t_res, "tft_prob"))),
                            "llm_actions": llm_actions if isinstance(llm_actions, dict) else {},
                        }
                    )
            except Exception:
                pass

            fo0_age_sec = self._compute_fo0_diagnostics()

            snapshot = self._build_llm_snapshot(
                current_price=float(current_price),
                spot_index=spot_index,
                basis=basis,
                prob=float(prob),
                signal=str(signal),
                confidence=str(confidence),
                raw_signal=str(raw_signal),
                raw_confidence=str(raw_confidence),
                guardrail_applied=bool(guardrail_applied),
                guardrail_reason=str(guardrail_reason or ""),
                transformer_prob=float(transformer_prob),
                tft_prob=tft_prob,
                ensemble_method=str(ensemble_method),
                model_agreement=model_agreement,
                transformer_weight=transformer_weight,
                last_ob_snapshot=dict(last_ob_snapshot) if last_ob_snapshot else {},
                opt_snap=dict(opt_snap) if opt_snap else {},
                adaptive_features=adaptive_features,
                heuristic_info=(
                    dict((model_outputs or {}).get("heuristic") or {})
                    if isinstance(model_outputs, dict)
                    else None
                ),
            )

            try:
                _pi = None
                try:
                    if (getattr(t_res, "prob_lower", None) is not None
                            and getattr(t_res, "prob_upper", None) is not None):
                        _pi = (float(t_res.prob_lower), float(t_res.prob_upper))
                except Exception:
                    _pi = None
                system, user = self._build_llm_prompt(
                    snapshot=snapshot,
                    ob_records=ob_records_snapshot,
                    adaptive_context=str(adaptive_context or ""),
                    amplitude=_pre_amplitude,
                    prob_interval=_pi,
                )
            except Exception as e:
                self._metrics_inc("prediction_failures")
                return {"error": "prompt_failed", "message": str(e)}

            # If we have recently hit 429, keep the base schedule intact but skip off-boundary
            # (heuristic flip) LLM calls to avoid request storms.
            # QUA-04: off_boundary는 이제 명시적 파라미터. kwargs 조회 불필요.
            try:
                now_epoch2 = float(now_dt.timestamp())
            except Exception:
                now_epoch2 = float(time.time())
            try:
                in_rl = float(self._llm_rate_limited_until_epoch or 0.0) > 0.0 and float(now_epoch2) < float(
                    self._llm_rate_limited_until_epoch
                )
            except Exception:
                in_rl = False

            if bool(off_boundary) and bool(in_rl):
                llm_action = str(signal)
                llm_provider = "skip_off_boundary_rate_limited"
                llm_timed_out = False
                risk_level = "MEDIUM"
                rationale = "LLM skipped due to recent 429 (off-boundary trigger)"
                caution = ""
                llm_raw = ""
                llm_disabled = not getattr(self, "_use_llm", True)
                return self._build_prediction_output(
                    now_dt=now_dt, start=start,
                    current_price=current_price, spot_index=spot_index, basis=basis,
                    prob=prob, signal=signal, confidence=confidence,
                    transformer_prob=transformer_prob, tft_prob=tft_prob,
                    ensemble_method=ensemble_method, model_agreement=model_agreement,
                    transformer_weight=transformer_weight, regime=regime,
                    model_outputs=model_outputs,
                    last_ob_snapshot=last_ob_snapshot,
                    ob_records_snapshot=ob_records_snapshot, ob_len_snapshot=ob_len_snapshot,
                    fo0_age_sec=fo0_age_sec,
                    llm_action=llm_action, llm_provider=llm_provider, llm_timed_out=llm_timed_out,
                    risk_level=risk_level, rationale=rationale, caution=caution,
                    llm_raw=llm_raw, opt_snap=opt_snap,
                    amplitude_snapshot=_pre_amplitude,
                    guardrail_applied=bool(guardrail_applied),
                    guardrail_reason=str(guardrail_reason or ""),
                    prob_lower=getattr(t_res, "prob_lower", None),
                    adaptive_zigzag_state=adaptive_zigzag_state,
                    prob_upper=getattr(t_res, "prob_upper", None),
                    llm_disabled=llm_disabled,
                )

            try:
                if self._dump_llm_prompt and (not self._llm_prompt_dumped):
                    self._llm_prompt_dumped = True
                    sep = "=" * 70
                    logger.info(sep)
                    logger.info("[LLM_USER_PROMPT_DUMP] (first occurrence only)")
                    try:
                        _u = str(user)
                        _u = _u.replace("\n\n출력은 반드시 JSON 단일 객체만.", "")
                        _u = _u.replace("출력은 반드시 JSON 단일 객체만.", "")
                        _u = _u.replace("출력은 반드시 JSON 단일 객체만", "")
                        logger.info(_u.strip())
                    except Exception:
                        logger.info(user)
                    logger.info(sep)
            except Exception:
                pass

            try:
                llm_action, llm_provider, llm_timed_out, risk_level, rationale, caution, llm_raw, model_outputs = (
                    self._run_llm_judgment(
                        system=system,
                        user=user,
                        t_res=_nr,
                        model_outputs=model_outputs,
                    )
                )
            except Exception as e:
                try:
                    s = str(e or "")
                    if "429" in s or "too many requests" in s.lower():
                        cd = float(LLM_COOLDOWN_SECONDS_ON_429 or 0.0)
                        if cd > 0.0:
                            self._llm_rate_limited_until_epoch = float(time.time()) + cd
                except Exception:
                    pass
                self._metrics_inc("prediction_failures")
                return {"error": "llm_failed", "message": str(e)}

            # ARC-02: _build_prediction_output 헬퍼로 단일화
            llm_disabled = not getattr(self, "_use_llm", True)
            return self._build_prediction_output(
                now_dt=now_dt, start=start,
                current_price=current_price, spot_index=spot_index, basis=basis,
                prob=prob, signal=signal, confidence=confidence,
                transformer_prob=transformer_prob, tft_prob=tft_prob,
                ensemble_method=ensemble_method, model_agreement=model_agreement,
                transformer_weight=transformer_weight, regime=regime,
                model_outputs=model_outputs,
                last_ob_snapshot=last_ob_snapshot,
                ob_records_snapshot=ob_records_snapshot, ob_len_snapshot=ob_len_snapshot,
                fo0_age_sec=fo0_age_sec,
                llm_action=llm_action, llm_provider=llm_provider, llm_timed_out=llm_timed_out,
                risk_level=risk_level, rationale=rationale, caution=caution,
                llm_raw=llm_raw, opt_snap=opt_snap,
                amplitude_snapshot=_pre_amplitude,
                guardrail_applied=bool(guardrail_applied),
                guardrail_reason=str(guardrail_reason or ""),
                prob_lower=getattr(t_res, "prob_lower", None),
                prob_upper=getattr(t_res, "prob_upper", None),
                adaptive_zigzag_state=adaptive_zigzag_state,
                llm_disabled=llm_disabled,
            )
        except Exception as e:
            self._metrics_inc("prediction_failures")
            return {"error": "prediction_failed", "message": str(e)}

