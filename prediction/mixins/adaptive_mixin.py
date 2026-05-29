"""Mixin extracted from prediction/pipeline.py.

이 파일은 PredictionPipeline의 일부를 Mixin으로 분리한 것입니다.
직접 인스턴스화하지 마십시오. PredictionPipeline을 통해 사용하세요.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from core.utils import normalize_ohlcv_columns

# ── Step 1~3 피처 병합 헬퍼 ─────────────────────────────────────────────────
def _merge_step123_features(
    self_obj: Any,
    high:  float,
    low:   float,
    close: float,
    bar_time: Any,
    features: Dict[str, float],
) -> None:
    """ATRAdaptivePivot / PercentAdaptivePivot / MSB / Kalman / Integrator 를 업데이트하고
    결과 피처를 features dict 에 in-place 병합한다.

    self_obj 는 PredictionPipeline 인스턴스이며, 다음 속성을 가져야 한다:
      _aap, _pap, _msb, _kf, _oi_gate, _integrator  (없으면 None)
    _last_opt_snap 이 있으면 OI 레벨을 추출해 OIStructureGate 에 전달한다.
    예외는 모두 흡수하고 기존 features 에 영향을 주지 않는다.
    """
    try:
        aap        = getattr(self_obj, "_aap",        None)
        pap        = getattr(self_obj, "_pap",        None)
        msb        = getattr(self_obj, "_msb",        None)
        kf         = getattr(self_obj, "_kf",         None)
        oi_gate    = getattr(self_obj, "_oi_gate",    None)
        integrator = getattr(self_obj, "_integrator", None)

        if aap is None and pap is None and msb is None and kf is None:
            return  # 모두 비활성 → 스킵

        # 1. ATRAdaptivePivot 업데이트
        aap_state = None
        if aap is not None:
            try:
                aap_state = aap.update(high=high, low=low, close=close, bar_time=bar_time)
                features.update(aap.get_transformer_features(close))
                # 신규 피봇 발생 시 로그
                if hasattr(aap_state, 'new_pivot_signal') and aap_state.new_pivot_signal != "none":
                    logger.info("[PIVOT][AAP] %s @%.2f %s", 
                               aap_state.new_pivot_signal.upper(), 
                               close, 
                               bar_time)
            except Exception as _e:
                logger.debug("[STEP1] aap.update 실패: %s", _e)

        # 1.5. PercentAdaptivePivot 업데이트
        pap_state = None
        if pap is not None:
            try:
                pap_state = pap.update(high=high, low=low, close=close, bar_time=bar_time)
                features.update(pap.get_transformer_features(close))
                # 신규 피봇 발생 시 로그
                if hasattr(pap_state, 'new_pivot_signal') and pap_state.new_pivot_signal != "none":
                    logger.info("[PIVOT][PAP] %s @%.2f %s",
                               pap_state.new_pivot_signal.upper(),
                               close,
                               bar_time)
            except Exception as _e:
                logger.debug("[STEP1.5] pap.update 실패: %s", _e)

        # 2. MarketStructureBreak 업데이트
        msb_state = None
        if msb is not None:
            try:
                # 활성화된 피봇 지표(aap 또는 pap)의 피봇 데이터 사용
                _pivots = list(aap.confirmed_pivots) if aap is not None else (list(pap.confirmed_pivots) if pap is not None else None)
                msb_state = msb.update(
                    high=high, low=low, close=close,
                    bar_time=bar_time, pivot_points=_pivots,
                )
                features.update(msb.get_transformer_features(close))
                # BOS/CHoCH 발생 시 로그
                if hasattr(msb_state, 'bos_signal') and msb_state.bos_signal.value != "none":
                    logger.info("[PIVOT][MSB] %s @%.2f %s", 
                               msb_state.bos_signal.value.upper(), 
                               close, 
                               bar_time)
            except Exception as _e:
                logger.debug("[STEP2] msb.update 실패: %s", _e)

        # 3. KalmanTurningPoint 업데이트
        kf_state = None
        if kf is not None:
            try:
                kf_state = kf.update(close=close, high=high, low=low, bar_time=bar_time)
                features.update(kf.get_transformer_features(close))
                # slope_flip 발생 시 로그
                if hasattr(kf_state, 'turning_signal') and kf_state.turning_signal != "none":
                    logger.info("[PIVOT][KALMAN] %s @%.2f %s", 
                               kf_state.turning_signal.upper(), 
                               close, 
                               bar_time)
            except Exception as _e:
                logger.debug("[STEP3] kf.update 실패: %s", _e)

        # 4. OI 레벨 추출 (_last_opt_snap 에서 가져오기)
        oi_levels = None
        try:
            _snap = getattr(self_obj, "_last_opt_snap", None) or {}
            oi_levels = _snap.get("_oi_levels") or None
        except Exception:
            pass

        # 5. OIStructureGate 점수 + 통합 PivotScore
        if integrator is not None and msb_state is not None:
            try:
                oi_score = 0.0
                if oi_gate is not None:
                    oi_score = float(oi_gate.score(msb_state, close, oi_levels) or 0.0)
                    features.update(oi_gate.get_transformer_features(
                        msb_state, close, oi_levels
                    ))

                # 2-C: regime — MSB 구조 + SuperTrend 방향 교차 판정
                # MSB 스윙 미수렴 시 SuperTrend(ast_direction)를 fallback으로 사용
                _msb_regime = str(msb_state.structure.value) if msb_state else "unknown"
                if _msb_regime == "unknown":
                    # SuperTrend 방향으로 fallback
                    _ast_dir = float(features.get("ast_direction", 0) or 0)
                    if _ast_dir > 0:
                        _regime = "uptrend"
                    elif _ast_dir < 0:
                        _regime = "downtrend"
                    else:
                        _regime = "unknown"
                elif _msb_regime in ("uptrend", "downtrend"):
                    # MSB와 SuperTrend 방향 일치 확인
                    _ast_dir = float(features.get("ast_direction", 0) or 0)
                    _ast_matches = (
                        (_msb_regime == "uptrend"   and _ast_dir > 0) or
                        (_msb_regime == "downtrend" and _ast_dir < 0) or
                        _ast_dir == 0  # SuperTrend 미확정 시 MSB 단독 사용
                    )
                    _regime = _msb_regime if _ast_matches else "ranging"
                else:
                    _regime = _msb_regime  # "ranging"

                result = integrator.compute(
                    aap_score    = float(aap_state.pivot_score)    if aap_state    else None,
                    pap_score    = float(pap_state.pivot_score)    if pap_state    else None,
                    msb_score    = float(msb_state.msb_score)      if msb_state    else None,
                    oi_score     = oi_score                        if oi_gate      else None,
                    kalman_score = float(kf_state.kalman_score)    if kf_state     else None,
                    aap_signal   = str(aap_state.new_pivot_signal) if aap_state    else "none",
                    pap_signal   = str(pap_state.new_pivot_signal) if pap_state    else "none",
                    msb_signal   = msb_state.bos_signal.value      if msb_state    else "none",
                    kalman_signal= str(kf_state.turning_signal)    if kf_state     else "none",
                    regime       = _regime,
                )
                features.update(integrator.get_transformer_features(result))
            except Exception as _e:
                logger.debug("[STEP3] integrator.compute 실패: %s", _e)

    except Exception as _outer:
        logger.debug("[STEP123] 피처 병합 예외 (무시): %s", _outer)



# TradeLogger import (선택적 - 실패해도 동작하도록)
try:
    from prediction.trade_logger import get_trade_logger, get_position_tracker, ExitReason
    TRADE_LOGGING_AVAILABLE = True
except ImportError:
    TRADE_LOGGING_AVAILABLE = False
    logger.warning("[ADAPTIVE_MIXIN] TradeLogger import 실패 - 거래 로깅 비활성화")


def _parse_adaptive_heuristic_features(
    features: Dict[str, Any],
) -> tuple[Optional[int], float, Optional[int]]:
    """ast_direction, ast_signal, azz_new_swing 단일 파싱 (FIX-HEURISTIC v2)."""
    ast_dir: Optional[int] = None
    try:
        d = features.get("ast_direction")
        if d is not None:
            _dv = float(d)
            if _dv > 0:
                ast_dir = 1
            elif _dv < 0:
                ast_dir = -1
    except Exception:
        ast_dir = None
    ast_signal_val = 0.0
    try:
        ast_signal_val = float(features.get("ast_signal") or 0.0)
    except Exception:
        ast_signal_val = 0.0
    azz_swing: Optional[int] = None
    try:
        ns = features.get("azz_new_swing")
        if ns is not None:
            _nsv = float(ns)
            if _nsv > 0:
                azz_swing = 1
            elif _nsv < 0:
                azz_swing = -1
            else:
                azz_swing = 0
    except Exception:
        azz_swing = None
    return ast_dir, ast_signal_val, azz_swing


class AdaptiveMixin:
    """Mixin: AdaptiveMixin methods extracted from PredictionPipeline."""

    def _compute_regime(
        self,
        *,
        adaptive_features: Optional[Dict[str, float]],
        adaptive_supertrend_state: Any,
    ) -> Optional[str]:
        """Best-effort market regime label.

        Returns one of:
        - STRONG_UP / WEAK_UP / RANGE / WEAK_DOWN / STRONG_DOWN
        """
        try:
            direction = None
            strength = None

            try:
                st = adaptive_supertrend_state
                if st is not None:
                    direction = int(getattr(st, "direction", 0) or 0)
                    strength = str(getattr(st, "trend_strength", "") or "").strip().lower() or None
            except Exception:
                direction = direction

            if direction is None:
                try:
                    if isinstance(adaptive_features, dict):
                        direction = int(float(adaptive_features.get("ast_direction") or 0.0))
                except Exception:
                    direction = None

            if strength is None:
                try:
                    if isinstance(adaptive_features, dict):
                        adx_norm = adaptive_features.get("ast_adx_norm")
                        if adx_norm is not None:
                            v = float(adx_norm)
                            if v < (20.0 / 60.0):
                                strength = "weak"
                            elif v < (40.0 / 60.0):
                                strength = "neutral"
                            else:
                                strength = "strong"
                except Exception:
                    strength = None

            if direction is None or int(direction) == 0:
                return None

            if strength == "strong":
                return "STRONG_UP" if int(direction) > 0 else "STRONG_DOWN"
            if strength == "weak":
                return "WEAK_UP" if int(direction) > 0 else "WEAK_DOWN"
            return "RANGE"
        except Exception:
            return None

    def _compute_adaptive_bundle(self, *, df: "pd.DataFrame", now_dt: datetime) -> tuple[
        Optional[Dict[str, float]], str, Any, Any, Dict[str, Any]
    ]:
        adaptive_features: Optional[Dict[str, float]] = None
        adaptive_context: str = ""
        adaptive_supertrend_state: Any = None
        adaptive_zigzag_state: Any = None
        model_outputs: Dict[str, Any] = {}

        # 멀티스케일 데이터 업데이트 (활성화 시)
        if self._adaptive_mgr is not None and (self._multiscale_5m or self._multiscale_enabled):
            try:
                self._update_adaptive_multiscale_data(df)
            except Exception as e:
                logger.debug("[_compute_adaptive_bundle] 멀티스케일 데이터 업데이트 실패: %s", e)

        try:
            if self._adaptive_mgr is not None:
                dfx = df.copy()
                if "timestamp" in dfx.columns:
                    dfx["timestamp"] = dfx["timestamp"].astype("datetime64[ns]")
                    dfx = dfx.set_index("timestamp")

                try:
                    dfx = normalize_ohlcv_columns(dfx)
                except Exception as _e:
                    logger.debug("[_compute_adaptive_bundle] 오류 무시: %s", _e)

                def _sanitize_adaptive_df(frame: "pd.DataFrame") -> "pd.DataFrame":
                    """Adaptive 지표 입력 DF를 일관된 형태로 강제 정규화한다.

                    보장 항목:
                    - timestamp 인덱스(DatetimeIndex)
                    - 시간 오름차순 정렬
                    - 동일 timestamp 중복 제거(마지막 값 유지)
                    - OHLCV 숫자형 변환/비정상값 제거
                    """
                    if frame is None or frame.empty:
                        return frame
                    out = frame.copy()

                    # 1) DatetimeIndex 보장
                    try:
                        if not isinstance(out.index, pd.DatetimeIndex):
                            if "timestamp" in out.columns:
                                out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
                                out = out.dropna(subset=["timestamp"])
                                out = out.set_index("timestamp")
                    except Exception:
                        return out.iloc[0:0]
                    if not isinstance(out.index, pd.DatetimeIndex):
                        return out.iloc[0:0]

                    # 2) 정렬 및 중복 제거(최신 틱 기반 마지막 값 유지)
                    try:
                        out = out[~out.index.duplicated(keep="last")]
                        out = out.sort_index()
                    except Exception as _e:
                        logger.debug("[_sanitize_adaptive_df] 오류 무시: %s", _e)

                    # 3) 필수 컬럼 숫자형 강제 + 유효 범위 필터
                    for col in ("Open", "High", "Low", "Close", "Volume"):
                        if col not in out.columns:
                            out[col] = np.nan
                        try:
                            out[col] = pd.to_numeric(out[col], errors="coerce")
                        except Exception as _e:
                            logger.debug("오류 무시: %s", _e)
                    out = out.dropna(subset=["High", "Low", "Close"], how="any")
                    try:
                        out = out[
                            (out["High"] > 0.0)
                            & (out["Low"] > 0.0)
                            & (out["Close"] > 0.0)
                            & (out["High"] >= out["Low"])
                        ]
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)
                    return out

                def _pick_last_complete_bar(frame: "pd.DataFrame"):
                    try:
                        if frame is None or frame.empty:
                            return None
                        if not isinstance(frame.index, pd.DatetimeIndex):
                            return frame.iloc[-1]
                        if len(frame.index) < 2:
                            return frame.iloc[-1]
                        now_min = now_dt.replace(second=0, microsecond=0)
                        last_ts = frame.index[-1].to_pydatetime()
                        last_ts = last_ts.replace(second=0, microsecond=0)
                        if last_ts >= now_min:
                            return frame.iloc[-2]
                        return frame.iloc[-1]
                    except Exception:
                        try:
                            return frame.iloc[-1]
                        except Exception:
                            return None

                def _pick_last_complete_ts(frame: "pd.DataFrame") -> Optional[datetime]:
                    try:
                        if frame is None or frame.empty:
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
                    except Exception:
                        return None

                def _reset_adaptive_state(reason: str) -> None:
                    def _normalize_rewind_reason_key(msg: str) -> str:
                        """rewind 사유 문자열을 경로 무관한 정규 키로 변환."""
                        try:
                            s = str(msg or "").strip()
                            m = re.search(
                                r"(?:last_complete_ts|last_ts)=([^\s]+)\s+prev=([^\s]+)",
                                s,
                            )
                            if m:
                                return f"minute_df_rewind:{m.group(1)}->{m.group(2)}"
                        except Exception as _e:
                            logger.debug("[_normalize_rewind_reason_key] 오류 무시: %s", _e)
                        return ""

                    # rewind 쿨다운 — 60초 이내 재발이면 reset을 건너뛰고 이전 features 재사용
                    _now_epoch = float(time.time())
                    _elapsed = _now_epoch - float(self._adaptive_last_rewind_epoch)
                    _reason = str(reason or "")
                    _reason_key = _normalize_rewind_reason_key(_reason)
                    # 동일 rewind 사유 반복 시 reset/경고를 생략한다.
                    # (예: 같은 minute_df_rewind가 API 재전송으로 계속 들어오는 구간)
                    try:
                        _last_key = str(getattr(self, "_adaptive_last_rewind_key", "") or "")
                        if _reason_key and _reason_key == _last_key:
                            logger.debug("[ADAPT] duplicated rewind key skipped: %s (%s)", _reason_key, _reason)
                            return
                        if _reason and _reason == str(getattr(self, "_adaptive_last_rewind_reason", "")):
                            logger.debug("[ADAPT] duplicated rewind reason skipped: %s", _reason)
                            return
                    except Exception as _e:
                        logger.debug("[_normalize_rewind_reason_key] 오류 무시: %s", _e)
                    if (
                        "rewind" in _reason
                        and self._adaptive_warmed
                        and self._adaptive_last_features
                        and _elapsed < float(self._ADAPT_REWIND_COOLDOWN_SEC)
                    ):
                        try:
                            logger.debug(
                                "[ADAPT] rewind 쿨다운 중 (elapsed=%.0fs < %.0fs) — 직전 features 재사용: %s",
                                _elapsed, float(self._ADAPT_REWIND_COOLDOWN_SEC), _reason,
                            )
                        except Exception as _e:
                            logger.debug("오류 무시: %s", _e)
                        return   # reset 생략 — 직전 _adaptive_last_features 유지
                    try:
                        self._adaptive_last_rewind_epoch = _now_epoch
                        self._adaptive_last_rewind_reason = _reason
                        if _reason_key:
                            self._adaptive_last_rewind_key = _reason_key
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)
                    try:
                        logger.warning("[ADAPT] reset adaptive state: %s", _reason)
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)
                    try:
                        if self._adaptive_mgr is not None:
                            self._adaptive_mgr.reset()
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)
                    self._adaptive_warmed = False
                    self._adaptive_last_minute_ts = None
                    self._adaptive_last_features = {}
                    self._adaptive_last_context = ""
                    # 피봇 정보는 유지 (분봉 증가 시에도 확정된 피봇 정보를 보존)
                    # self._adaptive_last_zigzag_state = None  # 주석 처리: 피봇 정보 유지
                    self._adaptive_pending_flip = None  # FLIP-ACCUM: rewind 시 누적 flip 초기화

                try:
                    last_complete_ts = _pick_last_complete_ts(dfx)
                    if (
                        self._adaptive_warmed
                        and self._adaptive_last_minute_ts is not None
                        and last_complete_ts is not None
                        and last_complete_ts < self._adaptive_last_minute_ts
                    ):
                        # 1분 이하 차이는 데이터 소스의 일시적인 지연으로 간주하고 무시
                        time_diff = (self._adaptive_last_minute_ts - last_complete_ts).total_seconds()
                        if time_diff > 60:  # 1분 초과 차이만 rewind로 처리
                            _reset_adaptive_state(
                                f"minute_df_rewind last_complete_ts={last_complete_ts} prev={self._adaptive_last_minute_ts} diff={time_diff:.0f}s"
                            )
                        else:
                            logger.debug(
                                "[ADAPT] minute_df_rewind ignored (diff=%d <= 60s): last_complete_ts=%s prev=%s",
                                int(time_diff), last_complete_ts, self._adaptive_last_minute_ts
                            )
                except Exception as _e:
                    logger.debug("오류 무시: %s", _e)

                try:
                    _before_n = int(len(dfx.index))
                except Exception:
                    _before_n = 0
                dfx = _sanitize_adaptive_df(dfx)
                try:
                    _after_n = int(len(dfx.index))
                    if _before_n > _after_n:
                        logger.debug(
                            "[ADAPT_DF] sanitized rows: %d -> %d (dropped=%d)",
                            _before_n, _after_n, max(0, _before_n - _after_n),
                        )
                except Exception as _e:
                    logger.debug("오류 무시: %s", _e)

                if not self._adaptive_warmed:
                    tail_bars = dfx.tail(int(self._adaptive_warmup_bars or 45))
                    tail_bars = tail_bars.dropna(subset=["High", "Low", "Close"], how="any")
                    try:
                        self._adaptive_mgr.reset()
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)

                    try:
                        last_complete_ts = _pick_last_complete_ts(tail_bars)
                        if last_complete_ts is None:
                            raise RuntimeError("no_last_complete_bar")
                        if isinstance(tail_bars.index, pd.DatetimeIndex):
                            warm_bars = tail_bars[tail_bars.index <= pd.Timestamp(last_complete_ts)]
                        else:
                            warm_bars = tail_bars

                        res = None
                        for _wi, (_ts, row) in enumerate(warm_bars.iterrows()):
                            try:
                                # 완결봉 집합을 1회만 순회해 중복 update를 방지한다.
                                # [FIX] 시가 anchor를 활성화하되, 충분한 데이터 누적 후 anchor 심도록 수정
                                # 첫 번째 봉 대신 4번째 봉에 시가를 전달하여 anchor pivot 주입
                                _open_arg = float(row["Open"]) if _wi == 3 and "Open" in row.index else None
                                res = self._adaptive_mgr.update(
                                    float(row["High"]),
                                    float(row["Low"]),
                                    float(row["Close"]),
                                    open=_open_arg,
                                    bar_time=_ts,
                                )
                            except Exception:
                                continue

                        if res is None:
                            raise RuntimeError("warmup_no_rows_processed")
                        if isinstance(res, dict):
                            try:
                                if not bool(res.get("is_ready", True)):
                                    adaptive_features = None
                                    adaptive_context = ""
                                    raise RuntimeError("adaptive_not_ready")
                            except Exception as _e:
                                logger.debug("오류 무시: %s", _e)
                            tf = res.get("transformer")
                            if isinstance(tf, dict):
                                adaptive_features = {k: float(v) for k, v in tf.items() if v is not None}
                            ctx = res.get("llm_context")
                            if ctx:
                                adaptive_context = str(ctx)
                            adaptive_supertrend_state = res.get("supertrend_state")
                            adaptive_zigzag_state = res.get("zigzag_state")
                            # ── Step 1~3 워밍업: warm_bars 전체 순회 후 마지막 봉 피처 병합 ──
                            # 1-D 수정: 마지막 봉 1개만 전달하던 방식 → 전체 순회로 ATR/KF 상태 누적
                            if adaptive_features is not None:
                                try:
                                    # Step 1~3 지표 상태 리셋
                                    for _s123_inst in [
                                        getattr(self, "_aap", None),
                                        getattr(self, "_pap", None),
                                        getattr(self, "_msb", None),
                                        getattr(self, "_kf",  None),
                                    ]:
                                        if _s123_inst is not None:
                                            try: _s123_inst.reset()
                                            except Exception: pass

                                    # warm_bars 전체 순회하여 내부 상태 누적
                                    _dummy: Dict[str, float] = {}
                                    for _wts, _wrow in warm_bars.iterrows():
                                        try:
                                            # 4-A: .get() 대신 인덱스 접근 (Series 안전)
                                            _wh = float(_wrow["High"]  if "High"  in _wrow.index else 0) or 0.0
                                            _wl = float(_wrow["Low"]   if "Low"   in _wrow.index else 0) or 0.0
                                            _wc = float(_wrow["Close"] if "Close" in _wrow.index else 0) or 0.0
                                            if _wc > 0:  # 유효한 봉만 처리
                                                _merge_step123_features(
                                                    self, high=_wh, low=_wl, close=_wc,
                                                    bar_time=_wts, features=_dummy,
                                                )
                                        except Exception:
                                            continue

                                    # 마지막 봉 피처만 adaptive_features에 병합
                                    _last_wrow = warm_bars.iloc[-1]
                                    _last_wts  = warm_bars.index[-1]
                                    try:
                                        _wh = float(_last_wrow["High"]  if "High"  in _last_wrow.index else 0) or 0.0
                                        _wl = float(_last_wrow["Low"]   if "Low"   in _last_wrow.index else 0) or 0.0
                                        _wc = float(_last_wrow["Close"] if "Close" in _last_wrow.index else 0) or 0.0
                                    except Exception:
                                        _wh = _wl = _wc = 0.0
                                    if _wc > 0:
                                        _merge_step123_features(
                                            self, high=_wh, low=_wl, close=_wc,
                                            bar_time=_last_wts, features=adaptive_features,
                                        )
                                except Exception as _wm_ex:
                                    logger.debug("[STEP123][warmup] 전체 워밍업 실패: %s", _wm_ex)

                                # ── 신규 키 제로 패딩 (기존 모델 호환성) ──
                                try:
                                    _known_dim = 38  # 기존 모델 학습 차원
                                    from prediction.features.features import ADAPT_KEYS
                                    if len(ADAPT_KEYS) > _known_dim:
                                        for k in ADAPT_KEYS[_known_dim:]:
                                            if k not in adaptive_features:
                                                adaptive_features[k] = 0.0
                                except Exception as _pad_ex:
                                    logger.debug("[ZERO_PAD] 제로 패딩 실패: %s", _pad_ex)
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)
                    self._adaptive_warmed = True

                    try:
                        self._adaptive_last_minute_ts = _pick_last_complete_ts(tail_bars)
                    except Exception:
                        self._adaptive_last_minute_ts = None
                else:
                    try:
                        last_row = _pick_last_complete_bar(dfx)
                        last_ts = _pick_last_complete_ts(dfx)
                    except Exception:
                        last_row = None
                        last_ts = None

                    if last_row is not None and last_ts is not None:
                        if self._adaptive_last_minute_ts is None or last_ts > self._adaptive_last_minute_ts:
                            # ── FLIP-ACCUM: 직전 last_minute_ts 이후의 신규 봉을 전부 순서대로 처리 ──
                            # 예측 주기(5분) 사이에 발생한 SuperTrend flip을 놓치지 않기 위해
                            # 마지막 봉 1개만 update하는 기존 방식 → 신규 봉 전체 순회로 교체한다.
                            try:
                                if self._adaptive_last_minute_ts is not None and isinstance(dfx.index, pd.DatetimeIndex):
                                    new_bars = dfx[dfx.index > pd.Timestamp(self._adaptive_last_minute_ts)]
                                    new_bars = new_bars[new_bars.index <= pd.Timestamp(last_ts)]
                                else:
                                    # last_minute_ts 없음(첫 incremental) → 마지막 봉만
                                    new_bars = dfx.iloc[-1:]
                            except Exception:
                                new_bars = dfx.iloc[-1:]

                            _last_res = None
                            for _idx, _row in new_bars.iterrows():
                                try:
                                    _r = self._adaptive_mgr.update(
                                        float(_row["High"]),
                                        float(_row["Low"]),
                                        float(_row["Close"]),
                                        bar_time=_idx,
                                    )
                                    # flip 누적: ast_signal != 0이면 BUY/SELL 기록
                                    if isinstance(_r, dict):
                                        _tf = _r.get("transformer")
                                        if isinstance(_tf, dict):
                                            _sig = float(_tf.get("ast_signal") or 0.0)
                                            if _sig > 0.0:
                                                self._adaptive_pending_flip = "BUY"
                                                logger.debug(
                                                    "[ADAPT_FLIP] BUY flip 누적 ts=%s ast_signal=+1",
                                                    _idx,
                                                )
                                            elif _sig < 0.0:
                                                self._adaptive_pending_flip = "SELL"
                                                logger.debug(
                                                    "[ADAPT_FLIP] SELL flip 누적 ts=%s ast_signal=-1",
                                                    _idx,
                                                )
                                    _last_res = _r
                                except Exception:
                                    continue

                            self._adaptive_last_minute_ts = last_ts
                            res = _last_res  # 마지막 봉 결과를 features로 사용
                            if isinstance(res, dict):
                                try:
                                    if not bool(res.get("is_ready", True)):
                                        adaptive_features = None
                                        adaptive_context = ""
                                        raise RuntimeError("adaptive_not_ready")
                                except Exception as _e:
                                    logger.debug("오류 무시: %s", _e)
                                tf = res.get("transformer")
                                if isinstance(tf, dict):
                                    adaptive_features = {k: float(v) for k, v in tf.items() if v is not None}
                                ctx = res.get("llm_context")
                                if ctx:
                                    adaptive_context = str(ctx)
                                adaptive_supertrend_state = res.get("supertrend_state")
                                adaptive_zigzag_state = res.get("zigzag_state")
                                # ── Step 1~3 피처 병합 (incremental 마지막 봉 기준) ──
                                if adaptive_features is not None and last_row is not None:
                                    try:
                                        # 4-A: Series .get() → 인덱스 접근으로 교체
                                        try:
                                            _ih = float(last_row["High"]  if "High"  in last_row.index else 0) or 0.0
                                            _il = float(last_row["Low"]   if "Low"   in last_row.index else 0) or 0.0
                                            _ic = float(last_row["Close"] if "Close" in last_row.index else 0) or 0.0
                                        except Exception:
                                            _ih = _il = _ic = 0.0
                                        if _ic > 0:
                                            _merge_step123_features(
                                                self,
                                                high=_ih, low=_il, close=_ic,
                                                bar_time=last_ts,
                                                features=adaptive_features,
                                            )
                                    except Exception as _im_ex:
                                        logger.debug("[STEP123][incr] 병합 실패: %s", _im_ex)

                                    # ── 신규 키 제로 패딩 (기존 모델 호환성) ──
                                    try:
                                        _known_dim = 38  # 기존 모델 학습 차원
                                        from prediction.features.features import ADAPT_KEYS
                                        if len(ADAPT_KEYS) > _known_dim:
                                            for k in ADAPT_KEYS[_known_dim:]:
                                                if k not in adaptive_features:
                                                    adaptive_features[k] = 0.0
                                    except Exception as _pad_ex:
                                        logger.debug("[ZERO_PAD] 제로 패딩 실패: %s", _pad_ex)
                        elif last_ts < self._adaptive_last_minute_ts:
                            # 1분 이하 차이는 데이터 소스의 일시적인 지연으로 간주하고 무시
                            time_diff = (self._adaptive_last_minute_ts - last_ts).total_seconds()
                            if time_diff > 60:  # 1분 초과 차이만 rewind로 처리
                                _reset_adaptive_state(
                                    f"minute_df_rewind incremental last_ts={last_ts} prev={self._adaptive_last_minute_ts} diff={time_diff:.0f}s"
                                )
                            else:
                                logger.debug(
                                    "[ADAPT] minute_df_rewind incremental ignored (diff=%d <= 60s): last_ts=%s prev=%s",
                                    int(time_diff), last_ts, self._adaptive_last_minute_ts
                                )
                        else:
                            adaptive_features = dict(self._adaptive_last_features)
                            adaptive_context = str(self._adaptive_last_context or "")
                            adaptive_zigzag_state = getattr(
                                self, "_adaptive_last_zigzag_state", None
                            )

                self._adaptive_last_features = dict(adaptive_features or {})
                self._adaptive_last_context = str(adaptive_context or "")
                if adaptive_zigzag_state is not None:
                    self._adaptive_last_zigzag_state = adaptive_zigzag_state
                    # ZigZag 스윙 확정 신호를 prediction.log에 별도 기록
                    try:
                        _sig = str(getattr(adaptive_zigzag_state, "new_swing_signal", "none") or "none")
                        if _sig != "none":
                            _sym = str((getattr(self, "_adaptive_indicator", None) or {}).get("symbol", "") or "")
                            _px  = float(getattr(adaptive_zigzag_state,
                                         "last_swing_high" if _sig == "new_high"
                                         else "last_swing_low", 0.0) or 0.0)
                            _tm  = str(getattr(adaptive_zigzag_state,
                                        "last_swing_high_time" if _sig == "new_high"
                                        else "last_swing_low_time", "?") or "?")
                            _lag = int(getattr(adaptive_zigzag_state,
                                       "last_swing_high_lag_bars" if _sig == "new_high"
                                       else "last_swing_low_lag_bars", 0) or 0)
                            _struct = str(getattr(adaptive_zigzag_state, "structure", "?") or "?")
                            _conf_n = int(getattr(adaptive_zigzag_state,
                                          "confirmed_pivot_count", 0) or 0)
                            logger.info(
                                "[ZZ][스윙확정] %s | %s @%s=%.2f | lag=%d봉 | "
                                "구조=%s | 누적확정=%d개",
                                _sym,
                                "HIGH✦" if _sig == "new_high" else "LOW✦",
                                _tm, _px, _lag, _struct, _conf_n,
                            )
                    except Exception as _e:
                        logger.debug("[ZZ][swing_log_error] %s", _e)
        except Exception as _bundle_ex:
            logger.warning("[ADAPTIVE_BUNDLE] 예외 발생: %s", _bundle_ex, exc_info=True)
            adaptive_features = None
            adaptive_context = ""
            adaptive_supertrend_state = None
            # 예외 발생 시 이전에 저장된 zigzag_state를 유지 (None 덮어쓰기 금지)
            # → 피봇 정보가 라벨에서 "-"로 사라지는 현상 방지
            adaptive_zigzag_state = getattr(self, "_adaptive_last_zigzag_state", None)

        try:
            if self._adaptive_mgr is not None:
                a = None
                is_ready = bool(adaptive_features)
                reason = ""

                if isinstance(adaptive_features, dict) and adaptive_features:
                    # 휴리스틱 신호 결정: 피봇 확정(기본) + SuperTrend 방향(필터)
                    try:
                        _sym = str(getattr(getattr(self._adaptive_mgr, "config", None), "symbol", "") or "").strip()
                    except Exception:
                        _sym = ""
                    if not _sym:
                        _sym = "KP200 선물"
                    ast_dir, ast_signal_val, azz_swing = _parse_adaptive_heuristic_features(adaptive_features)

                    # ── 보조 feature 추출 (보완 규칙용) ────────────────────────
                    try:
                        _azz_hh  = float(adaptive_features.get("azz_higher_highs", 0.0) or 0.0) > 0.5
                        _azz_ll  = float(adaptive_features.get("azz_lower_lows",   0.0) or 0.0) > 0.5
                        _rng     = float(adaptive_features.get("azz_structure_ranging", 0.0) or 0.0) > 0.5
                        _s_up    = float(adaptive_features.get("azz_structure_up",   0.0) or 0.0) > 0.5
                        _s_dn    = float(adaptive_features.get("azz_structure_down", 0.0) or 0.0) > 0.5
                        _s_unk   = not (_rng or _s_up or _s_dn)  # structure==unknown
                    except Exception:
                        _azz_hh = _azz_ll = _rng = _s_up = _s_dn = False
                        _s_unk  = True

                    # [P5] SuperTrend 연계 강화: 피봇과 ST 라인 간격 분석
                    try:
                        # 현재 가격과 ST 라인 간격 (ATP 기준)
                        _ast_distance_pct = 0.0
                        try:
                            st_state = adaptive_supertrend_state
                            if st_state is not None:
                                current_price = float(adaptive_features.get("close", 0.0) or 0.0)
                                st_line = float(getattr(st_state, "st_line", 0.0) or 0.0)
                                if current_price > 0 and st_line > 0:
                                    _ast_distance_pct = abs(current_price - st_line) / current_price * 100
                        except Exception:
                            pass
                    except Exception:
                        _ast_distance_pct = 0.0

                    # ── 1단계: 피봇+ST 방향 판단 ──────────────────────────────
                    # L확정+ST상승→BUY, H확정+ST하락→SELL, 불일치→HOLD
                    if azz_swing == -1:
                        if ast_dir == 1:
                            a = "BUY"
                            reason = (
                                "zigzag_pivot_low(L)->BUY+ST_UP "
                                f"(symbol={_sym} azz_new_swing={azz_swing} ast_dir={ast_dir} ast_signal={ast_signal_val:+.0f})"
                            )
                        else:
                            a = "HOLD"
                            reason = (
                                "zigzag_pivot_low(L)->HOLD_ST_NOT_UP "
                                f"(symbol={_sym} azz_new_swing={azz_swing} ast_dir={ast_dir} ast_signal={ast_signal_val:+.0f})"
                            )
                    elif azz_swing == 1:
                        if ast_dir == -1:
                            a = "SELL"
                            reason = (
                                "zigzag_pivot_high(H)->SELL+ST_DOWN "
                                f"(symbol={_sym} azz_new_swing={azz_swing} ast_dir={ast_dir} ast_signal={ast_signal_val:+.0f})"
                            )
                        else:
                            a = "HOLD"
                            reason = (
                                "zigzag_pivot_high(H)->HOLD_ST_NOT_DOWN "
                                f"(symbol={_sym} azz_new_swing={azz_swing} ast_dir={ast_dir} ast_signal={ast_signal_val:+.0f})"
                            )
                    else:
                        a = "HOLD"
                        reason = (
                            "zigzag_no_pivot->HOLD "
                            f"(symbol={_sym} azz_new_swing={azz_swing} ast_dir={ast_dir} ast_signal={ast_signal_val:+.0f})"
                        )

                    # ── 2단계: 보완 규칙 적용 (BUY/SELL 신호에만) ────────────
                    _conf = "LOW"       # 기본값 (HOLD 경로 포함 항상 정의됨)
                    _qual_tags: list = []

                    if a in ("BUY", "SELL"):
                        _conf = "HIGH"  # BUY/SELL 기본은 HIGH, 보완 규칙이 강등

                        # [보완-1] structure=unknown → 장 초반 구조 미확정 → HOLD 억제
                        # ZigZag 피봇이 충분히 쌓이기 전에는 구조 판단 불가
                        if _s_unk:
                            a = "HOLD"
                            reason = (
                                reason.rstrip()
                                + " HOLD:structure=unknown(초기구조미확정)"
                            )
                            _conf = "LOW"
                            logger.info(
                                "[HEURISTIC] structure=unknown → HOLD 억제 (symbol=%s)", _sym
                            )

                        # [보완-2] LL/HH 구조 미확인 → confidence MEDIUM 강등
                        # BUY이나 LL=False: 저점이 낮아지지 않는 반등 (추세 약함)
                        # SELL이나 HH=False: 고점이 높아지지 않는 하락 (추세 약함)
                        elif (a == "BUY"  and not _azz_ll) or (a == "SELL" and not _azz_hh):
                            _conf = "MEDIUM"
                            _tag = "LL=False" if a == "BUY" else "HH=False"
                            reason = reason.rstrip() + f" MEDIUM:{_tag}(구조미확인)"
                            _qual_tags.append(_tag)
                            logger.info(
                                "[HEURISTIC] %s → MEDIUM confidence (%s) symbol=%s",
                                a, _tag, _sym,
                            )

                        if a in ("BUY", "SELL"):   # HOLD 억제 이후에도 살아남은 경우만
                            # [보완-3] ranging 구간 → confidence MEDIUM 강등
                            # ranging에서도 피봇+ST 일치 시 신호는 발행하되 강도를 낮춤
                            if _rng:
                                if _conf == "HIGH":   # 이미 MEDIUM이면 더 낮추지 않음
                                    _conf = "MEDIUM"
                                reason = reason.rstrip() + " MEDIUM:ranging구간"
                                _qual_tags.append("ranging")
                                logger.info(
                                    "[HEURISTIC] %s ranging → MEDIUM confidence (symbol=%s)",
                                    a, _sym,
                                )

                            # [P2] 시간대별 피봇 신호 빈도 제한
                            # 시간대별 최소 간격 체크 (모든 구간 적용)
                            if a in ("BUY", "SELL"):
                                try:
                                    current_bar_idx = len(df.index) - 1 if df is not None else 0
                                    last_signal_idx = getattr(self, "_last_pivot_signal_bar_idx", -999)

                                    # 시간대별 간격 가져오기
                                    min_interval = getattr(self, "_min_pivot_interval_bars") or 10
                                    session_table = getattr(self, "_session_min_pivot_interval_table", [])
                                    if session_table and df is not None and len(df.index) > 0:
                                        try:
                                            current_time = df.index[-1]
                                            if hasattr(current_time, "time"):
                                                current_time_str = current_time.strftime("%H:%M")
                                                for start, end, interval in session_table:
                                                    if isinstance(start, str) and isinstance(end, str):
                                                        if start <= current_time_str <= end:
                                                            min_interval = int(interval) if isinstance(interval, (int, float)) else min_interval
                                                            break
                                        except Exception:
                                            pass

                                    if current_bar_idx - last_signal_idx < min_interval:
                                        a = "HOLD"
                                        reason = reason.rstrip() + f" HOLD:too_frequent({current_bar_idx - last_signal_idx}bars<{min_interval})"
                                        _conf = "LOW"
                                        logger.info(
                                            "[HEURISTIC] 시간대별 신호 빈도 제한: {current_bar_idx - last_signal_idx}bars < {min_interval}bars → HOLD 억제 (symbol=%s)",
                                            _sym,
                                        )
                                    else:
                                        # 간격 충족 시 현재 인덱스 업데이트
                                        self._last_pivot_signal_bar_idx = current_bar_idx
                                except Exception:
                                    pass

                            # [보완-4] 추세와 역방향 반등/하락 → confidence MEDIUM 강등
                            # downtrend 내 L확정 BUY: 하락 추세 내 반등 매수 (단기 반등 가능하나 위험)
                            # uptrend 내 H확정 SELL: 상승 추세 내 하락 매도
                            if (a == "BUY"  and _s_dn) or (a == "SELL" and _s_up):
                                if _conf == "HIGH":
                                    _conf = "MEDIUM"
                                _tag = "downtrend내반등매수" if a == "BUY" else "uptrend내하락매도"
                                reason = reason.rstrip() + f" MEDIUM:{_tag}"
                                _qual_tags.append(_tag)
                                logger.info(
                                    "[HEURISTIC] %s trend-counter → MEDIUM confidence (%s) symbol=%s",
                                    a, _tag, _sym,
                                )

                            # [보완-5] wave_size_pct 하한 → 잡음 피봇 차단
                            # 파동 크기가 너무 작으면 잡음일 가능성이 높음
                            try:
                                _wave_size_pct = float(adaptive_features.get("azz_wave_size_pct", 0.0) or 0.0) * 100
                                if _wave_size_pct < 0.3:
                                    a = "HOLD"
                                    reason = reason.rstrip() + f" HOLD:wave_size_too_small({_wave_size_pct:.2f}%)"
                                    _conf = "LOW"
                                    logger.info(
                                        "[HEURISTIC] wave_size_pct {_wave_size_pct:.2f}% < 0.3% → HOLD 억제 (symbol=%s)",
                                        _sym,
                                    )
                            except Exception:
                                pass

                            # [보완-6] ST trend_duration 최소 → whipsaw 방지
                            # ST 방향 전환 직후 신호는 불안정함
                            try:
                                _trend_duration = float(adaptive_features.get("ast_trend_duration", 0.0) or 0.0) * 78
                                if _trend_duration < 3 and a in ("BUY", "SELL"):
                                    if _conf == "HIGH":
                                        _conf = "MEDIUM"
                                    reason = reason.rstrip() + f" MEDIUM:ST_trend_too_short({_trend_duration:.0f}bars)"
                                    _qual_tags.append("ST_trend_short")
                                    logger.info(
                                        "[HEURISTIC] ST trend_duration {_trend_duration:.0f}bars < 3 → MEDIUM confidence (symbol=%s)",
                                        _sym,
                                    )
                            except Exception:
                                pass

                            # [보완-7] bars_since_swing 최소 → 연속 피봇 억제
                            # 이전 피봇과 너무 가까우면 연속 피봇일 가능성
                            try:
                                _bars_since_swing = float(adaptive_features.get("azz_bars_since_swing", 0.0) or 0.0) * 50
                                if _bars_since_swing < 5 and a in ("BUY", "SELL"):
                                    a = "HOLD"
                                    reason = reason.rstrip() + f" HOLD:too_soon_after_last_pivot({_bars_since_swing:.0f}bars)"
                                    _conf = "LOW"
                                    logger.info(
                                        "[HEURISTIC] bars_since_swing {_bars_since_swing:.0f}bars < 5 → HOLD 억제 (symbol=%s)",
                                        _sym,
                                    )
                            except Exception:
                                pass

                            # [P8] 다중 타임프레임 통합: 상위 타임프레임 피봇 필터
                            # 상위 타임프레임(5분/15분) 피봇과 주추세 일치 시만 신호 허용
                            if getattr(self, "_higher_tf_pivot_filter", False) and a in ("BUY", "SELL"):
                                try:
                                    # 상위 타임프레임 피봇 상태 확인 (kospi_zigzag)
                                    higher_tf_state = None
                                    if hasattr(adaptive_supertrend_state, "kospi_zigzag_state"):
                                        higher_tf_state = getattr(adaptive_supertrend_state, "kospi_zigzag_state")
                                    
                                    if higher_tf_state is not None:
                                        # 상위 타임프레임 구조 확인
                                        higher_structure = getattr(higher_tf_state, "structure", "unknown")
                                        higher_swing = getattr(higher_tf_state, "last_swing_type", None)
                                        
                                        # BUY 신호: 상위 타임프레임이 uptrend 또는 저점 확정이어야 함
                                        if a == "BUY":
                                            if higher_structure == "downtrend":
                                                a = "HOLD"
                                                reason = reason.rstrip() + " HOLD:higher_tf_downtrend_filter"
                                                _conf = "LOW"
                                                logger.info(
                                                    "[HEURISTIC] 상위 타임프레임 downtrend → BUY 억제 (symbol=%s)",
                                                    _sym,
                                                )
                                        
                                        # SELL 신호: 상위 타임프레임이 downtrend 또는 고점 확정이어야 함
                                        elif a == "SELL":
                                            if higher_structure == "uptrend":
                                                a = "HOLD"
                                                reason = reason.rstrip() + " HOLD:higher_tf_uptrend_filter"
                                                _conf = "LOW"
                                                logger.info(
                                                    "[HEURISTIC] 상위 타임프레임 uptrend → SELL 억제 (symbol=%s)",
                                                    _sym,
                                                )
                                except Exception:
                                    pass

                            # [P10] ADX 기반 confidence 조정
                            # 추세 강도에 따라 confidence 차등 조정
                            if getattr(self, "_adx_confidence_filter_enabled", False) and a in ("BUY", "SELL"):
                                try:
                                    _adx_value = float(adaptive_features.get("ast_adx_norm", 0.0) or 0.0) * 100
                                    _hold_threshold = getattr(self, "_adx_hold_threshold", 15.0)
                                    _weak_threshold = getattr(self, "_adx_weak_threshold", 20.0)
                                    _strong_threshold = getattr(self, "_adx_strong_threshold", 35.0)
                                    
                                    # ADX < hold_threshold: 추세 너무 약함 → HOLD
                                    if _adx_value < _hold_threshold:
                                        a = "HOLD"
                                        reason = reason.rstrip() + f" HOLD:ADX_too_weak({_adx_value:.1f})"
                                        _conf = "LOW"
                                        logger.info(
                                            "[HEURISTIC] ADX {_adx_value:.1f} < {_hold_threshold:.1f} → HOLD 억제 (symbol=%s)",
                                            _sym,
                                        )
                                    
                                    # hold_threshold ≤ ADX < weak_threshold: 약한 추세 → confidence 강등
                                    elif _adx_value < _weak_threshold:
                                        if _conf == "HIGH":
                                            _conf = "MEDIUM"
                                            reason = reason.rstrip() + f" MEDIUM:ADX_weak({_adx_value:.1f})"
                                        elif _conf == "MEDIUM":
                                            _conf = "LOW"
                                            reason = reason.rstrip() + f" LOW:ADX_weak({_adx_value:.1f})"
                                        _qual_tags.append("ADX_weak")
                                        logger.info(
                                            "[HEURISTIC] ADX {_adx_value:.1f} < {_weak_threshold:.1f} → confidence 강등 (symbol=%s)",
                                            _sym,
                                        )
                                    
                                    # ADX ≥ strong_threshold: 강한 추세 → confidence 승격
                                    elif _adx_value >= _strong_threshold:
                                        if _conf == "MEDIUM":
                                            _conf = "HIGH"
                                            reason = reason.rstrip() + f" HIGH:ADX_strong({_adx_value:.1f})"
                                        elif _conf == "LOW":
                                            _conf = "MEDIUM"
                                            reason = reason.rstrip() + f" MEDIUM:ADX_strong({_adx_value:.1f})"
                                        _qual_tags.append("ADX_strong")
                                        logger.info(
                                            "[HEURISTIC] ADX {_adx_value:.1f} ≥ {_strong_threshold:.1f} → confidence 승격 (symbol=%s)",
                                            _sym,
                                        )
                                except Exception:
                                    pass
                else:
                    try:
                        bars_n = int(len(df.index)) if df is not None else 0
                    except Exception:
                        bars_n = 0
                    try:
                        last_ts_str = (
                            self._adaptive_last_minute_ts.isoformat()
                            if self._adaptive_last_minute_ts is not None
                            else ""
                        )
                    except Exception:
                        last_ts_str = ""
                    reason = (
                        "adaptive_not_ready_or_missing_features"
                        f" (bars={bars_n} warmup_bars={int(self._adaptive_warmup_bars or 0)} warmed={bool(self._adaptive_warmed)} last_minute_ts={last_ts_str})"
                    )

                if a is None:
                    a = "HOLD"

                # zigzag_pivot_* 단순화 규칙은 필터를 바이패스해 신호를 그대로 보낸다.
                if (
                    a in ("BUY", "SELL")
                    and isinstance(adaptive_features, dict)
                    and (not str(reason).startswith("zigzag_pivot_"))
                ):
                    a, reason = self._apply_ranging_filter(
                        action=a,
                        reason=reason,
                        features=adaptive_features,
                        zigzag_state=adaptive_zigzag_state,
                    )
                
                # BUY/SELL 신호 확정 시 거래 로깅
                if a in ("BUY", "SELL"):
                    self._log_trade_signal(a, reason, str(_conf), adaptive_features)

                model_outputs["heuristic"] = {
                    "action": str(a),
                    "provider": "adaptive_indicator",
                    "is_ready": bool(is_ready),
                    "reason": str(reason),
                    "confidence": str(_conf) if a in ("BUY", "SELL") else "LOW",
                    "supertrend_state": adaptive_supertrend_state,
                    "zigzag_state": adaptive_zigzag_state,
                }
        except Exception:
            model_outputs = {}

        return adaptive_features, adaptive_context, adaptive_supertrend_state, adaptive_zigzag_state, model_outputs

    def _apply_ranging_filter(
        self,
        *,
        action: str,
        reason: str,
        features: Dict[str, float],
        zigzag_state: Any,
    ) -> tuple:
        """횡보장 SuperTrend flip을 4단계 필터로 억제한다.

        ADX / ER / ZigZag structure / whipsaw 조건 중 하나라도 해당되면
        BUY/SELL을 HOLD로 바꾸고 억제 사유를 reason에 기록한다.

        Returns:
            (action, reason) — 억제 시 ('HOLD', 'ranging_suppressed[...] ...')
        """
        if not self._rf_enabled or action not in ("BUY", "SELL"):
            return action, reason

        suppressions: list = []

        # 1단계: ADX 필터
        # ast_adx_norm = adx / adx_norm_cap(100) 으로 정규화되어 있으므로 역산
        if self._rf_use_adx:
            try:
                adx_norm = float(features.get("ast_adx_norm") or 0.0)
                adx_raw = adx_norm * 100.0
                if adx_raw < float(self._rf_adx_min):
                    suppressions.append(f"ADX={adx_raw:.1f}<{self._rf_adx_min:.0f}")
            except Exception as _e:
                logger.debug("[_apply_ranging_filter] 오류 무시: %s", _e)

        # 2단계: ER 필터
        # ast_efficiency_ratio = raw ER (0~1)
        if self._rf_use_er:
            try:
                er = float(features.get("ast_efficiency_ratio") or 0.0)
                if er < float(self._rf_er_min):
                    suppressions.append(f"ER={er:.3f}<{self._rf_er_min:.2f}")
            except Exception as _e:
                logger.debug("오류 무시: %s", _e)

        # 3단계: ZigZag structure 필터
        # zigzag_state 객체 직접 참조를 우선, 없으면 transformer feature로 대체
        if self._rf_use_zigzag:
            try:
                zz_structure = getattr(zigzag_state, "structure", None)
                if zz_structure == "ranging":
                    suppressions.append("ZZ=ranging")
                elif zz_structure is None:
                    # zigzag_state 없을 때 features 폴백
                    is_ranging = float(features.get("azz_structure_ranging") or 0.0)
                    if is_ranging > 0.5:
                        suppressions.append("ZZ_feat=ranging")
            except Exception as _e:
                logger.debug("오류 무시: %s", _e)

        # 4단계: whipsaw 필터 (flip 직후 재flip 억제)
        # ast_trend_duration = bars_in_trend / trend_duration_cap_bars(78)
        # bars_in_trend < whipsaw_min_bars 이면 채찍질 구간으로 판단
        if self._rf_use_whipsaw:
            try:
                dur = float(features.get("ast_trend_duration") or 1.0)
                bars_approx = dur * 78.0
                if bars_approx < float(self._rf_whipsaw_min_bars):
                    suppressions.append(
                        f"bars≈{bars_approx:.0f}<{self._rf_whipsaw_min_bars}"
                    )
            except Exception as _e:
                logger.debug("오류 무시: %s", _e)

        if suppressions:
            suppressed_reason = (
                f"ranging_suppressed[{','.join(suppressions)}]"
                f" orig={action} {reason}"
            )
            logger.info(
                "[RANGING_FILTER] %s → HOLD  %s",
                action, ",".join(suppressions),
            )
            return "HOLD", suppressed_reason

        return action, reason
    
    def _log_trade_signal(
        self,
        action: str,
        reason: str,
        confidence: str,
        features: Dict[str, float]
    ) -> None:
        """거래 신호 로깅.
        
        Args:
            action: 액션 (BUY/SELL)
            reason: 사유
            confidence: 신뢰도
            features: 피쳐
        """
        if not TRADE_LOGGING_AVAILABLE:
            return
        
        try:
            # Pipeline의 trade_logger 인스턴스 사용
            trade_logger = getattr(self, "_trade_logger", None)
            if trade_logger is None:
                return
            
            # 현재 가격
            current_price = float(features.get("close", 0.0))
            if current_price == 0.0:
                return
            
            # ATR
            atr = float(features.get("atr", 0.0))
            
            # 포지션 사이즈 (기본 1.0)
            size = 1.0
            
            # 손절/이익실현 계산 (ATR 기반)
            stop_loss = None
            take_profit = None
            if atr > 0:
                if action == "BUY":
                    stop_loss = current_price - (atr * 2.0)
                    take_profit = current_price + (atr * 3.0)
                else:  # SELL
                    stop_loss = current_price + (atr * 2.0)
                    take_profit = current_price - (atr * 3.0)
            
            # 진입 로그 기록
            trade_logger.log_entry(
                action=action,
                price=current_price,
                size=size,
                confidence=confidence,
                signal_reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr=atr
            )
            
            logger.info(
                "[TRADE_LOGGING] 진입 신호 로그: %s @ %.2f (conf=%s, reason=%s)",
                action, current_price, confidence, reason
            )
            
        except Exception as e:
            logger.error("[TRADE_LOGGING] 진입 신호 로그 실패: %s", e)

    def _update_adaptive_multiscale_data(self, df: "pd.DataFrame") -> None:
        """Adaptive Indicator에 멀티스케일 데이터 업데이트.

        1분봉 DataFrame에서 5분봉/15분봉 ATR과 피봇 방향을 계산하여
        AdaptiveIndicatorManager에 전달.
        """
        if self._adaptive_mgr is None:
            return

        try:
            from prediction.features import calc_multiscale_features, calc_multiscale_features_15m
        except Exception:
            return

        try:
            # 원시 OHLCV 형태로 복원
            df_raw = df.copy()
            if not isinstance(df_raw.index, pd.DatetimeIndex):
                if "timestamp" in df_raw.columns:
                    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
                    df_raw = df_raw.dropna(subset=["timestamp"])
                    df_raw = df_raw.set_index("timestamp")

            # 5분봉 ATR 계산
            atr_5m = None
            pivot_direction_5m = None
            if self._multiscale_5m or 5 in self._multiscale_time_scales:
                try:
                    ms5_df = calc_multiscale_features(df_raw)
                    if not ms5_df.empty:
                        # ATR 계산 (High - Low 기반 근사)
                        atr_5m = float(ms5_df["ms5_range5_pct"].iloc[-1] * df_raw["Close"].iloc[-1])
                        # 피봇 방향 (종가 기울기 기반)
                        if len(ms5_df) >= 3:
                            slope = float(ms5_df["ms5_slope5"].iloc[-1])
                            pivot_direction_5m = 1 if slope > 0 else (-1 if slope < 0 else 0)
                except Exception:
                    pass

            # 15분봉 ATR 계산
            atr_15m = None
            pivot_direction_15m = None
            if self._multiscale_enabled and 15 in self._multiscale_time_scales:
                try:
                    ms15_df = calc_multiscale_features_15m(df_raw)
                    if not ms15_df.empty:
                        atr_15m = float(ms15_df["ms15_range15_pct"].iloc[-1] * df_raw["Close"].iloc[-1])
                        if len(ms15_df) >= 2:
                            slope = float(ms15_df["ms15_slope15"].iloc[-1])
                            pivot_direction_15m = 1 if slope > 0 else (-1 if slope < 0 else 0)
                except Exception:
                    pass

            # Adaptive Indicator Manager에 업데이트
            self._adaptive_mgr.update_multiscale_data(
                enabled=True,
                atr_5m=atr_5m,
                atr_15m=atr_15m,
                pivot_direction_5m=pivot_direction_5m,
                pivot_direction_15m=pivot_direction_15m,
            )

            logger.debug(
                "[MULTISCALE] ATR 5m=%.4f 15m=%.4f | Pivot 5m=%d 15m=%d",
                atr_5m or 0.0,
                atr_15m or 0.0,
                pivot_direction_5m or 0,
                pivot_direction_15m or 0,
            )
        except Exception as e:
            logger.debug("[_update_adaptive_multiscale_data] 실패: %s", e)

