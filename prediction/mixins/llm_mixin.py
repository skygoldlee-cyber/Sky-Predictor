"""Mixin extracted from prediction/pipeline.py.

이 파일은 PredictionPipeline의 일부를 Mixin으로 분리한 것입니다.
직접 인스턴스화하지 마십시오. PredictionPipeline을 통해 사용하세요.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from config import (
    LLM_COOLDOWN_SECONDS_ON_429,
    LLM_PROVIDER_COOLDOWN_ON_429,
    LLM_PROVIDER_COOLDOWN_ON_TIMEOUT,
    API_MAX_RETRIES,
    API_RETRY_DELAY_SECONDS,
    API_BACKOFF_MULTIPLIER,
)


class LLMMixin:
    """Mixin: LLMMixin methods extracted from PredictionPipeline."""

    def _llm_timeout_for_provider(self, provider: str) -> float:
        """Provider별 LLM timeout(초)을 반환한다."""
        try:
            p = str(provider or "").strip().lower()
            if p == "gemini":
                return max(0.1, float(getattr(self, "_gemini_timeout_sec", self._llm_timeout_sec)))
            return max(0.1, float(getattr(self, "_llm_timeout_sec", 8.0)))
        except Exception:
            return max(0.1, float(getattr(self, "_llm_timeout_sec", 8.0)))

    def _llm_failure_fallback_action(
        self,
        t_res: Any,
        merged_model_outputs: Dict[str, Any],
        *,
        llm_timed_out: bool,
    ) -> tuple[str, str, str]:
        """LLM 전부 실패 시 ``llm_action`` / ``llm_provider`` / ``rationale``.

        - ``heuristic_fallback`` False: 수치 신호만.
        - True: adaptive ``model_outputs.heuristic`` 이 BUY/SELL 이고 ready 이면 우선 사용.
        """
        base = str(getattr(t_res, "signal", "HOLD") or "HOLD")
        hf = bool(getattr(self, "_heuristic_fallback", True))
        if not hf:
            r = (
                "LLM 타임아웃 — 수치 신호만 사용 (heuristic_fallback=false)"
                if llm_timed_out
                else "LLM 실패 — 수치 신호만 사용 (heuristic_fallback=false)"
            )
            return base, ("timeout" if llm_timed_out else "error"), r
        h: Any = None
        try:
            h = (merged_model_outputs or {}).get("heuristic")
        except Exception:
            h = None
        if isinstance(h, dict) and bool(h.get("is_ready", True)):
            ha = str(h.get("action") or "").strip().upper()
            if ha in ("BUY", "SELL"):
                r = (
                    "LLM 타임아웃 — adaptive 휴리스틱 신호 사용"
                    if llm_timed_out
                    else "LLM 실패 — adaptive 휴리스틱 신호 사용"
                )
                return ha, "heuristic_fallback", r
        if llm_timed_out:
            return base, "timeout", "LLM 타임아웃으로 Transformer 결과를 사용"
        return base, "error", "LLM 실패로 Transformer 결과를 사용"

    def _judge_provider_direct(self, *, provider: str, system: str, user: str):
        if self.judge is None:
            return None
        try:
            prov = str(provider or "").strip().lower()
            try:
                prl_until = float((getattr(self, "_provider_rate_limited_until", {}) or {}).get(prov, 0.0) or 0.0)
                if prl_until > 0.0 and float(time.time()) < prl_until:
                    return None
            except Exception:
                pass
            return self.judge.judge_provider(
                str(provider),
                system,
                user,
                timeout=float(self._llm_timeout_for_provider(prov)),
            )
        except Exception as e:
            try:
                # [P2-FIX] _judge_provider_direct 예외에서도 429 감지 → provider 쿨다운 즉시 설정.
                # judge_provider 내부에서 예외가 LLMJudgment로 wrap되지 않고 직접 올라오는 경우를 커버.
                _s = str(e or "").lower()
                if "429" in _s or "too many requests" in _s or "insufficient_quota" in _s:
                    # insufficient_quota(크레딧 소진)는 재충전 전까지 회복 불가 → 세션 내 영구 비활성화
                    if "insufficient_quota" in _s:
                        try:
                            self._provider_rate_limited_until[prov] = float("inf")
                        except Exception:
                            pass
                        logger.warning("[LLM_QUOTA_EXHAUSTED] provider=%s insufficient_quota → disabled for session", prov)
                        try:
                            self._notify_quota_exhausted(prov)
                        except Exception:
                            pass
                    else:
                        _prov_cd = float(LLM_PROVIDER_COOLDOWN_ON_429 or 120.0)
                        try:
                            self._provider_rate_limited_until[prov] = float(time.time()) + _prov_cd
                        except Exception:
                            pass
                        logger.warning("[LLM_429] provider=%s (direct) prov_cooldown=%.0fs", prov, _prov_cd)
            except Exception:
                pass
            try:
                logger.debug("[LLM_DIRECT_FAIL] provider=%s err=%s", str(provider), str(e))
            except Exception:
                pass
            return None

    def _judge_with_timeout(self, *, system: str, user: str):
        """Run LLM judge with a timeout.

        Returns:
            (judgment, timed_out: bool, error: Optional[str])

        Note:
            [FIX] 이중 타임아웃 방지:
            SDK 내부 timeout과 fut.result(timeout=) 두 타임아웃이 직렬로 작용하면
            실질 최대 대기가 2×llm_timeout_sec에 수렴할 수 있다.
            fut.result()에 SDK 타임아웃 + 2s 여유를 주어 SDK 예외가
            Future 래퍼 안에서 정상 propagate되도록 한다.
        """
        if self.judge is None:
            return None, False, None
        if self._llm_executor is None:
            return None, False, None

        sdk_timeout = float(self._llm_timeout_sec)
        future_timeout = sdk_timeout + 2.0  # [FIX] fut.result 타임아웃 > SDK 타임아웃

        try:
            fut = self._llm_executor.submit(self.judge.judge, system, user, timeout=sdk_timeout)
            judgment = fut.result(timeout=future_timeout)
            return judgment, False, None
        except FuturesTimeoutError:
            try:
                fut.cancel()
            except Exception:
                pass
            try:
                self._reset_llm_executor("timeout")
            except Exception:
                pass
            return None, True, "timeout"
        except Exception as e:
            try:
                s = str(e or "").lower()
                if "429" in s or "too many requests" in s:
                    cd = float(LLM_COOLDOWN_SECONDS_ON_429 or 0.0)
                    if cd > 0.0:
                        self._llm_rate_limited_until_epoch = float(time.time()) + cd
                        try:
                            logger.warning("[LLM_429] single_provider cooldown=%.0fs", cd)
                        except Exception:
                            pass
            except Exception:
                pass
            return None, False, str(e)

    def _reset_llm_executor(self, reason: str) -> None:
        if not self._use_llm:
            return
        with self._llm_executor_lock:
            old = self._llm_executor
            self._llm_executor = None
            if old is not None:
                try:
                    old.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    try:
                        old.shutdown(wait=False)
                    except Exception:
                        pass
            workers = 2 if self._dual_llm else 1
            try:
                self._llm_executor = ThreadPoolExecutor(max_workers=int(workers))
            except Exception:
                self._llm_executor = ThreadPoolExecutor(max_workers=1)
        # FIX-RESET-BADMODELS: executor reset 시 bad_models 캐시를 초기화한다.
        # timeout/404 연쇄로 모든 fallback 모델이 bad_models에 등록되면
        # 이후 호출에서 모든 모델이 skip → resp=None → "failed: None" 반복.
        # executor를 새로 만든 시점에 bad_models도 비워 재시도 기회를 준다.
        try:
            j = getattr(self, "judge", None)
            if j is not None:
                try:
                    j._gemini_bad_models.clear()
                except Exception:
                    pass
                try:
                    j._openai_bad_models.clear()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            logger.warning("[LLM_EXECUTOR_RESET] reason=%s", str(reason))
        except Exception:
            pass

    def _judge_provider_with_timeout(self, *, provider: str, system: str, user: str):
        """Run LLM judge for a specific provider with a timeout.

        Returns:
            (judgment, timed_out: bool, error: Optional[str])
        """
        if self.judge is None:
            return None, False, None
        if self._llm_executor is None:
            return None, False, None

        prov = str(provider or "").strip().lower() or ""

        # provider별 429 쿨다운 체크 — rate-limited 상태면 즉시 skip
        try:
            prl_until = float((self._provider_rate_limited_until or {}).get(prov, 0.0) or 0.0)
            if prl_until > 0.0 and float(time.time()) < prl_until:
                remaining = prl_until - float(time.time())
                logger.debug("[LLM_SKIP] provider=%s cooldown %.0fs remaining", prov, remaining)
                return None, False, f"rate_limited:{remaining:.0f}s"
        except Exception:
            pass

        max_retries = 1
        delay = 0.0
        backoff = 1.0
        try:
            if prov in ("gemini", "gpt"):
                max_retries = max(1, int(API_MAX_RETRIES or 1))
                delay = float(API_RETRY_DELAY_SECONDS or 0.0)
                backoff = float(API_BACKOFF_MULTIPLIER or 1.0)
        except Exception:
            max_retries = 1
            delay = 0.0
            backoff = 1.0

        last_err: Optional[str] = None
        last_timed_out = False
        for attempt in range(int(max_retries)):
            fut = None
            try:
                _sdk_timeout = float(self._llm_timeout_for_provider(prov))
                _fut_timeout = _sdk_timeout + 2.0  # [FIX] fut.result > SDK timeout, SDK 예외가 Future 안에서 propagate되도록
                fut = self._llm_executor.submit(
                    getattr(self.judge, "judge_provider"),
                    str(provider),
                    system,
                    user,
                    timeout=_sdk_timeout,
                )
                judgment = fut.result(timeout=_fut_timeout)

                # Treat empty/blank raw output as a retryable failure (observed intermittently on Gemini).
                try:
                    raw = getattr(judgment, "raw", None)
                    if raw is None or str(raw).strip() == "":
                        raise RuntimeError("empty_output")
                except Exception as e:
                    last_err = str(e)
                    last_timed_out = False
                    judgment = None

                # Retry only when configured (e.g., gemini). No sleep after the last attempt.
                if judgment is not None:
                    return judgment, False, None

                if attempt < int(max_retries) - 1 and float(delay) > 0.0:
                    try:
                        time.sleep(float(delay))
                    except Exception:
                        pass
                    try:
                        delay = float(delay) * float(backoff)
                    except Exception:
                        pass
                continue
            except FuturesTimeoutError:
                try:
                    if fut is not None:
                        fut.cancel()
                except Exception:
                    pass
                try:
                    self._reset_llm_executor(f"provider_timeout:{prov}")
                except Exception:
                    pass
                # timeout 시 provider 쿨다운 등록 — 반복 timeout 방지
                try:
                    _prov_cd = float(
                        getattr(
                            self,
                            "_llm_provider_cooldown_on_timeout_sec",
                            LLM_PROVIDER_COOLDOWN_ON_TIMEOUT,
                        )
                        or LLM_PROVIDER_COOLDOWN_ON_TIMEOUT
                    )
                    self._provider_rate_limited_until[prov] = float(time.time()) + _prov_cd
                    logger.warning("[LLM_TIMEOUT] provider=%s timeout → prov_cooldown=%.0fs", prov, _prov_cd)
                except Exception:
                    pass
                last_err = "timeout"
                last_timed_out = True
                break
            except Exception as e:
                last_err = str(e)
                last_timed_out = False
                # [LLM-FIX-3] 429 감지: provider별 쿨다운과 전체 쿨다운을 분리.
                # provider별은 LLM_PROVIDER_COOLDOWN_ON_429(120s)로 차단하고
                # 전체는 LLM_COOLDOWN_SECONDS_ON_429(60s)로 짧게 유지.
                # 이를 통해 한 provider가 429 상태여도 다른 provider로 즉시 fallback 가능.
                try:
                    s = str(e or "").lower()
                    if "429" in s or "too many requests" in s or "insufficient_quota" in s:
                        now_t = float(time.time())
                        if "insufficient_quota" in s:
                            # 크레딧 소진 — 재충전 전까지 회복 불가 → 세션 내 영구 비활성화
                            try:
                                self._provider_rate_limited_until[prov] = float("inf")
                            except Exception:
                                pass
                            logger.warning(
                                "[LLM_QUOTA_EXHAUSTED] provider=%s insufficient_quota → disabled for session",
                                prov,
                            )
                            try:
                                self._notify_quota_exhausted(prov)
                            except Exception:
                                pass
                        else:
                            # 일반 속도 제한 — 쿨다운 후 재시도
                            # provider 개별 쿨다운 (더 길게)
                            prov_cd = float(LLM_PROVIDER_COOLDOWN_ON_429 or 120.0)
                            if prov_cd > 0.0:
                                until_prov = now_t + prov_cd
                                try:
                                    self._provider_rate_limited_until[prov] = until_prov
                                except Exception:
                                    pass
                            # 전체 쿨다운 (짧게 — dual_llm에서 다른 provider 허용)
                            global_cd = float(LLM_COOLDOWN_SECONDS_ON_429 or 60.0)
                            if global_cd > 0.0:
                                self._llm_rate_limited_until_epoch = now_t + global_cd
                            logger.warning(
                                "[LLM_429] provider=%s prov_cooldown=%.0fs global_cooldown=%.0fs",
                                prov, prov_cd, global_cd,
                            )
                        break  # 429/quota 는 재시도 무의미 — 즉시 중단
                except Exception:
                    pass
                if attempt < int(max_retries) - 1 and float(delay) > 0.0:
                    try:
                        time.sleep(float(delay))
                    except Exception:
                        pass
                    try:
                        delay = float(delay) * float(backoff)
                    except Exception:
                        pass
                continue

        return None, bool(last_timed_out), last_err

    def _run_llm_judgment(
        self,
        *,
        system: str,
        user: str,
        t_res: Any,
        model_outputs: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str, bool, str, str, str, str, Dict[str, Any]]:
        # ── LLM 비활성화 / 클라이언트 미설정 조기 반환 ─────────────────────────
        # NOTE: try/except로 raise→catch 패턴을 쓰지 않는다.
        #       AttributeError 등 실제 초기화 오류까지 "disabled" 경로로 삼켜버리기 때문.
        _llm_disabled = False
        try:
            _llm_disabled = (not bool(self._use_llm)) or (self.judge is None)
        except AttributeError:
            logger.error(
                "[LLM] _use_llm 또는 judge 속성 접근 실패 — 초기화를 확인하세요",
                exc_info=True,
            )
            raise

        if _llm_disabled:
            llm_action = str(getattr(t_res, "signal", "HOLD") or "HOLD")
            _mo: Dict[str, Any] = dict(model_outputs) if isinstance(model_outputs, dict) else {}
            return (
                llm_action, "disabled", False,
                "MEDIUM", "LLM disabled", "", "",
                _mo,
            )

        now_epoch = 0.0
        try:
            now_epoch = float(time.time())
        except Exception:
            now_epoch = 0.0

        # ── 429 쿨다운 체크 (dual_llm 포함 모든 경로) ─────────────────────────
        try:
            rl_until = float(self._llm_rate_limited_until_epoch or 0.0)
            if rl_until > 0.0 and now_epoch < rl_until:
                remaining = rl_until - now_epoch
                _action = str(getattr(t_res, "signal", "HOLD") or "HOLD")
                _mo: Dict[str, Any] = {}
                try:
                    if isinstance(model_outputs, dict):
                        _mo.update(model_outputs)
                except Exception:
                    pass
                logger.debug("[LLM_SKIP] 429 cooldown %.0fs remaining", remaining)
                return (
                    _action, "rate_limited", False,
                    "HIGH", f"LLM rate-limited (cooldown {remaining:.0f}s)",
                    "", "", _mo,
                )
        except Exception:
            pass

        cache_key = ""
        try:
            signal = str(getattr(t_res, "signal", "HOLD") or "HOLD")
        except Exception:
            signal = "HOLD"
        try:
            ensemble_method = str(getattr(t_res, "ensemble_method", "") or "")
        except Exception:
            ensemble_method = ""
        try:
            # [IMP-LLM-01] 캐시 키에 시장 상태 필드 추가.
            # 가격·IV·확신도가 바뀌면 별도 LLM 판단을 유도한다.

            # prob_band: 예측 확률 구간 (LOW/MID/HIGH)
            _prob = float(getattr(t_res, "prob", 0.5) or 0.5)
            _prob_band = "HIGH" if _prob > 0.70 else ("MID" if _prob > 0.55 else "LOW")

            # price_bucket: 현재가를 0.5pt 단위로 양자화
            try:
                _cur_px = float(
                    (getattr(self, "_last_price_for_cache", None))
                    or 0.0
                )
            except Exception:
                _cur_px = 0.0
            _price_bucket = str(int(round(_cur_px / 0.5))) if _cur_px > 0.0 else "0"

            # iv_band: ATM IV 구간 (LOW/MID/HIGH)
            try:
                _atm_iv = float(
                    (getattr(self, "_last_opt_snap", None) or {}).get("atm_iv") or 0.0
                )
            except Exception:
                _atm_iv = 0.0
            _iv_band = "HIGH" if _atm_iv > 0.25 else ("MID" if _atm_iv > 0.15 else "LOW")

            cache_key = "|".join(
                [
                    "dual" if bool(self._dual_llm) else "single",
                    str(self._dual_llm_primary_provider or ""),
                    str(signal),
                    str(ensemble_method),
                    str(_prob_band),
                    str(_price_bucket),
                    str(_iv_band),
                ]
            )
        except Exception:
            cache_key = ""

        try:
            # CON-02: _last_llm_* 변수는 dual_llm 모드에서 두 스레드가 동시에 접근 가능.
            # _llm_cache_lock으로 캐시 읽기를 원자적으로 수행한다.
            with self._llm_cache_lock:
                cache_hit = (
                    float(self._llm_min_interval_sec) > 0.0
                    and float(self._last_llm_call_epoch) > 0.0
                    and (now_epoch - float(self._last_llm_call_epoch)) < float(self._llm_min_interval_sec)
                    and self._last_llm_result is not None
                    and str(self._last_llm_cache_key or "") == str(cache_key or "")
                )
                cached_result = self._last_llm_result if cache_hit else None
            if cache_hit and cached_result is not None:
                return cached_result
        except Exception:
            pass

        def _cache_and_return(
            out: tuple[str, str, bool, str, str, str, str, Dict[str, Any]]
        ) -> tuple[str, str, bool, str, str, str, str, Dict[str, Any]]:
            try:
                # CON-02: 캐시 쓰기도 동일 Lock으로 보호한다.
                with self._llm_cache_lock:
                    self._last_llm_call_epoch = float(now_epoch)
                    self._last_llm_cache_key = str(cache_key or "")
                    self._last_llm_result = out
            except Exception:
                pass
            return out

        merged_model_outputs: Dict[str, Any] = {}
        try:
            if isinstance(model_outputs, dict) and model_outputs:
                merged_model_outputs.update(dict(model_outputs))
        except Exception:
            merged_model_outputs = {}
        llm_provider = ""
        llm_timed_out = False
        if self.judge is None:
            llm_action = str(getattr(t_res, "signal", "HOLD") or "HOLD")
            llm_provider = "no_client"
            risk_level = "MEDIUM"
            rationale = "LLM client unavailable"
            caution = ""
            llm_raw = ""
            out = (llm_action, llm_provider, llm_timed_out, risk_level, rationale, caution, llm_raw, merged_model_outputs)
            return _cache_and_return(out)

        if self._dual_llm:
            gpt_j = None
            gem_j = None
            gpt_to = False
            gem_to = False
            gpt_err = None
            gem_err = None
            try:
                if self._llm_executor is None:
                    raise RuntimeError("llm_executor_missing")

                # ── Gemini 우선 실행 보장 ────────────────────────────────────────────
                # Gemini를 먼저 submit하고 먼저 result를 대기한다.
                # GPT는 Gemini 대기 후 남은 슬롯으로 처리 (최소 2초 보장).
                # 429/timeout 쿨다운 체크는 양쪽 모두 독립적으로 수행한다.
                _now_t = float(time.time())
                _gpt_rate_until = float((self._provider_rate_limited_until or {}).get("gpt", 0.0) or 0.0)
                _gem_rate_until = float((self._provider_rate_limited_until or {}).get("gemini", 0.0) or 0.0)
                _gpt_skipped = _now_t < _gpt_rate_until
                _gem_skipped = _now_t < _gem_rate_until

                if _gem_skipped:
                    remaining_gem = _gem_rate_until - _now_t
                    logger.debug("[LLM_SKIP] provider=gemini rate_limited %.0fs remaining — dual submit skipped", remaining_gem)
                    gem_err = f"rate_limited:{remaining_gem:.0f}s"

                # Gemini 먼저 submit
                fut_gem = (
                    None if _gem_skipped
                    else self._llm_executor.submit(
                        self._judge_provider_direct,
                        provider="gemini",
                        system=system,
                        user=user,
                    )
                )

                if _gpt_skipped:
                    remaining_gpt = _gpt_rate_until - _now_t
                    logger.debug("[LLM_SKIP] provider=gpt rate_limited %.0fs remaining — dual submit skipped", remaining_gpt)
                    gpt_err = f"rate_limited:{remaining_gpt:.0f}s"

                # GPT 두 번째 submit
                fut_gpt = (
                    None if _gpt_skipped
                    else self._llm_executor.submit(
                        self._judge_provider_direct,
                        provider="gpt",
                        system=system,
                        user=user,
                    )
                )
                # ── /Gemini 우선 ─────────────────────────────────────────────────────

                # submit 완료 시각 기록
                _dual_submit_epoch = float(time.time())
                _llm_tmo = float(self._llm_timeout_sec)
                _gem_tmo = float(self._llm_timeout_for_provider("gemini"))

                # Gemini result 먼저 대기
                if fut_gem is not None:
                    try:
                        gem_j = fut_gem.result(timeout=_gem_tmo)
                    except FuturesTimeoutError:
                        gem_to = True
                        gem_err = "timeout"
                        try:
                            _prov_cd = float(
                                getattr(
                                    self,
                                    "_llm_provider_cooldown_on_timeout_sec",
                                    LLM_PROVIDER_COOLDOWN_ON_TIMEOUT,
                                )
                                or LLM_PROVIDER_COOLDOWN_ON_TIMEOUT
                            )
                            self._provider_rate_limited_until["gemini"] = _now_t + _prov_cd
                            logger.warning("[LLM_TIMEOUT] provider=gemini timeout → prov_cooldown=%.0fs", _prov_cd)
                        except Exception:
                            pass
                    except Exception as e:
                        gem_err = str(e)
                        try:
                            _s = str(e or "").lower()
                            if "429" in _s or "too many requests" in _s or "insufficient_quota" in _s:
                                if "insufficient_quota" in _s:
                                    self._provider_rate_limited_until["gemini"] = float("inf")
                                    logger.warning("[LLM_QUOTA_EXHAUSTED] provider=gemini → disabled for session")
                                    try:
                                        self._notify_quota_exhausted("gemini")
                                    except Exception:
                                        pass
                                else:
                                    _prov_cd = float(LLM_PROVIDER_COOLDOWN_ON_429 or 120.0)
                                    self._provider_rate_limited_until["gemini"] = _now_t + _prov_cd
                                    logger.warning("[LLM_429] provider=gemini (fut result) prov_cooldown=%.0fs", _prov_cd)
                        except Exception:
                            pass
                if fut_gpt is not None:
                    # GPT 대기 시간 = 전체 timeout - Gemini 대기에 소모된 시간. 최소 2초 보장.
                    _elapsed_since_submit = float(time.time()) - float(_dual_submit_epoch)
                    _gpt_wait = max(2.0, float(_llm_tmo) - float(_elapsed_since_submit))
                    try:
                        gpt_j = fut_gpt.result(timeout=_gpt_wait)
                    except FuturesTimeoutError:
                        gpt_to = True
                        gpt_err = "timeout"
                        try:
                            _prov_cd = float(
                                getattr(
                                    self,
                                    "_llm_provider_cooldown_on_timeout_sec",
                                    LLM_PROVIDER_COOLDOWN_ON_TIMEOUT,
                                )
                                or LLM_PROVIDER_COOLDOWN_ON_TIMEOUT
                            )
                            self._provider_rate_limited_until["gpt"] = _now_t + _prov_cd
                            logger.warning("[LLM_TIMEOUT] provider=gpt timeout → prov_cooldown=%.0fs", _prov_cd)
                        except Exception:
                            pass
                    except Exception as e:
                        gpt_err = str(e)
                        try:
                            _s = str(e or "").lower()
                            if "429" in _s or "too many requests" in _s or "insufficient_quota" in _s:
                                if "insufficient_quota" in _s:
                                    self._provider_rate_limited_until["gpt"] = float("inf")
                                    logger.warning("[LLM_QUOTA_EXHAUSTED] provider=gpt → disabled for session")
                                    try:
                                        self._notify_quota_exhausted("gpt")
                                    except Exception:
                                        pass
                                else:
                                    _prov_cd = float(LLM_PROVIDER_COOLDOWN_ON_429 or 120.0)
                                    self._provider_rate_limited_until["gpt"] = _now_t + _prov_cd
                                    logger.warning("[LLM_429] provider=gpt (fut result) prov_cooldown=%.0fs", _prov_cd)
                        except Exception:
                            pass
                try:
                    # [P2-FIX] 429-skipped provider는 timeout 카운트에서 제외한다.
                    # 실제 FuturesTimeout이 발생한 경우에만 Executor를 리셋한다.
                    _real_gpt_to = bool(gpt_to) and not bool(_gpt_skipped)
                    _real_gem_to = bool(gem_to) and not bool(_gem_skipped)
                    if _real_gpt_to or _real_gem_to:
                        self._reset_llm_executor("dual_timeout")
                except Exception:
                    pass
            except Exception:
                gpt_j, gpt_to, gpt_err = self._judge_provider_with_timeout(provider="gpt", system=system, user=user)
                gem_j, gem_to, gem_err = self._judge_provider_with_timeout(provider="gemini", system=system, user=user)

            def _judgment_has_response(j) -> bool:
                """judge_provider()는 실패 시 raw='' 인 LLMJudgment를 반환한다.
                raw 가 비어 있으면 실제 LLM 응답이 없는 실패 판단이다."""
                return j is not None and bool(str(getattr(j, "raw", "") or "").strip())

            if gpt_j is not None:
                merged_model_outputs["gpt"] = {
                    "action": getattr(gpt_j, "action", None),
                    "risk_level": getattr(gpt_j, "risk_level", None),
                    "rationale": getattr(gpt_j, "rationale", None),
                    "caution": getattr(gpt_j, "caution", None),
                    "pivot_candidate_probability": getattr(gpt_j, "pivot_candidate_probability", ""),
                    "pivot_candidate_reason": getattr(gpt_j, "pivot_candidate_reason", ""),
                    "raw": getattr(gpt_j, "raw", None),
                    "provider": getattr(gpt_j, "provider", "gpt"),
                }
                if gpt_to:
                    merged_model_outputs["gpt"]["timed_out"] = True
                if not _judgment_has_response(gpt_j):
                    merged_model_outputs["gpt"]["error"] = str(gpt_err or getattr(gpt_j, "rationale", ""))
            else:
                merged_model_outputs["gpt"] = {
                    "provider": "gpt",
                    "action": "HOLD",
                    "risk_level": "HIGH" if bool(gpt_to) else "MEDIUM",
                    "rationale": "LLM timeout" if bool(gpt_to) else "LLM error",
                    "caution": "",
                    "timed_out": bool(gpt_to),
                    "error": str(gpt_err or ""),
                }
            if gem_j is not None:
                merged_model_outputs["gemini"] = {
                    "action": getattr(gem_j, "action", None),
                    "risk_level": getattr(gem_j, "risk_level", None),
                    "rationale": getattr(gem_j, "rationale", None),
                    "caution": getattr(gem_j, "caution", None),
                    "pivot_candidate_probability": getattr(gem_j, "pivot_candidate_probability", ""),
                    "pivot_candidate_reason": getattr(gem_j, "pivot_candidate_reason", ""),
                    "raw": getattr(gem_j, "raw", None),
                    "provider": getattr(gem_j, "provider", "gemini"),
                }
                if gem_to:
                    merged_model_outputs["gemini"]["timed_out"] = True
                if not _judgment_has_response(gem_j):
                    merged_model_outputs["gemini"]["error"] = str(gem_err or getattr(gem_j, "rationale", ""))
            else:
                merged_model_outputs["gemini"] = {
                    "provider": "gemini",
                    "action": "HOLD",
                    "risk_level": "HIGH" if bool(gem_to) else "MEDIUM",
                    "rationale": "LLM timeout" if bool(gem_to) else "LLM error",
                    "caution": "",
                    "timed_out": bool(gem_to),
                    "error": str(gem_err or ""),
                }

            primary = self._dual_llm_primary_provider
            primary_j = gpt_j if primary == "gpt" else gem_j
            primary_to = bool(gpt_to) if primary == "gpt" else bool(gem_to)

            secondary_j = gem_j if primary == "gpt" else gpt_j
            secondary_to = bool(gem_to) if primary == "gpt" else bool(gpt_to)

            _primary_valid = _judgment_has_response(primary_j)
            _secondary_valid = _judgment_has_response(secondary_j)

            if not _primary_valid:
                if _secondary_valid:
                    llm_action = str(getattr(secondary_j, "action", "HOLD") or "HOLD")
                    risk_level = str(getattr(secondary_j, "risk_level", "MEDIUM") or "MEDIUM")
                    rationale = str(getattr(secondary_j, "rationale", "") or "")
                    caution = str(getattr(secondary_j, "caution", "") or "")
                    llm_raw = str(getattr(secondary_j, "raw", "") or "")
                    llm_provider = str(getattr(secondary_j, "provider", "") or ("gemini" if primary == "gpt" else "gpt"))
                    merged_model_outputs["pivot_candidate_probability"] = str(
                        getattr(secondary_j, "pivot_candidate_probability", "") or ""
                    )
                    merged_model_outputs["pivot_candidate_reason"] = str(
                        getattr(secondary_j, "pivot_candidate_reason", "") or ""
                    )
                    try:
                        meta = merged_model_outputs.setdefault("meta", {})
                        if isinstance(meta, dict):
                            meta["dual_llm_primary_failed_used_secondary"] = True
                            meta["dual_llm_primary"] = str(primary)
                            meta["dual_llm_primary_timed_out"] = bool(primary_to)
                            meta["dual_llm_secondary_timed_out"] = bool(secondary_to)
                    except Exception:
                        pass
                    out = (
                        llm_action,
                        llm_provider,
                        bool(primary_to) and bool(secondary_to),
                        risk_level,
                        rationale,
                        caution,
                        llm_raw,
                        merged_model_outputs,
                    )
                    return _cache_and_return(out)

                _pto = bool(primary_to) and bool(secondary_to)
                llm_action, llm_provider, rationale = self._llm_failure_fallback_action(
                    t_res, merged_model_outputs, llm_timed_out=_pto
                )
                risk_level = "MEDIUM"
                caution = ""
                llm_raw = ""
                out = (llm_action, llm_provider, llm_timed_out, risk_level, rationale, caution, llm_raw, merged_model_outputs)
                return _cache_and_return(out)

            gpt_action = str((merged_model_outputs.get("gpt") or {}).get("action") or "HOLD")
            if not _judgment_has_response(gpt_j):
                gpt_action = "HOLD"
            gem_action = str((merged_model_outputs.get("gemini") or {}).get("action") or "HOLD")
            if not _judgment_has_response(gem_j):
                gem_action = "HOLD"

            llm_action = str(getattr(primary_j, "action", "HOLD") or "HOLD")
            risk_level = str(getattr(primary_j, "risk_level", "MEDIUM") or "MEDIUM")
            rationale = str(getattr(primary_j, "rationale", "") or "")
            caution = str(getattr(primary_j, "caution", "") or "")
            llm_raw = str(getattr(primary_j, "raw", "") or "")
            llm_provider = str(getattr(primary_j, "provider", "") or primary)
            merged_model_outputs["pivot_candidate_probability"] = str(
                getattr(primary_j, "pivot_candidate_probability", "") or ""
            )
            merged_model_outputs["pivot_candidate_reason"] = str(
                getattr(primary_j, "pivot_candidate_reason", "") or ""
            )

            if (gpt_action != gem_action) and bool(self._disagreement_hold):
                # LLM 불일치 시 Transformer를 3자 tiebreaker로 활용한다.
                # - Transformer HIGH 확신(prob margin >= confidence_high_margin)이면 Transformer 신호 채택
                # - 그 외에는 기존대로 HOLD
                try:
                    def _act_to_p(a: str) -> float:
                        au = str(a or "").strip().upper()
                        if au == "BUY":  return 1.0
                        if au == "SELL": return 0.0
                        return 0.5

                    gpt_p = _act_to_p(gpt_action)
                    gem_p = _act_to_p(gem_action)
                    diff = abs(float(gpt_p) - float(gem_p))
                    thr = float(getattr(self, "_disagreement_hold_prob_diff_max", 0.3) or 0.3)

                    if diff <= thr:
                        # Transformer tiebreaker
                        _t_signal = str(getattr(t_res, "signal", "HOLD") or "HOLD")
                        _t_prob   = float(getattr(t_res, "prob", 0.5) or 0.5)
                        _t_margin = abs(_t_prob - 0.5)
                        _high_mg  = float(getattr(self, "_confidence_high_margin", 0.15) or 0.15)

                        if _t_margin >= _high_mg and _t_signal != "HOLD":
                            # Transformer 고확신 → 따라감
                            llm_action   = _t_signal
                            risk_level   = "MEDIUM"
                            llm_provider = "dual_disagreement_transformer_tiebreak"
                            rationale    = (
                                f"LLM 불일치(GPT={gpt_action}, Gemini={gem_action}) — "
                                f"Transformer 고확신({_t_prob:.2f}) 신호 채택"
                            )
                        else:
                            # Transformer 확신 부족 → HOLD
                            llm_action   = "HOLD"
                            risk_level   = "LOW"
                            llm_provider = "dual_disagreement_hold"
                            rationale    = (
                                f"LLM 불일치(GPT={gpt_action}, Gemini={gem_action}) — "
                                f"Transformer 확신 부족({_t_prob:.2f}), HOLD"
                            )
                except Exception:
                    llm_action   = "HOLD"
                    risk_level   = "LOW"
                    llm_provider = "dual_disagreement_hold"
                    rationale    = "dual_llm disagreement_hold (예외)"
                try:
                    meta = merged_model_outputs.setdefault("meta", {})
                    if isinstance(meta, dict):
                        meta["dual_llm_disagreement"]       = True
                        meta["gpt_action"]                  = gpt_action
                        meta["gemini_action"]               = gem_action
                        meta["tiebreak_transformer_signal"] = str(getattr(t_res, "signal", "HOLD") or "HOLD")
                        meta["tiebreak_transformer_prob"]   = float(getattr(t_res, "prob", 0.5) or 0.5)
                        meta["tiebreak_provider"]           = str(llm_provider)
                except Exception:
                    pass
            out = (llm_action, llm_provider, llm_timed_out, risk_level, rationale, caution, llm_raw, merged_model_outputs)
            return _cache_and_return(out)

        # Single-LLM 모드에서는 preferred_provider를 강제한다.
        # 예: preferred_provider=gemini 이면 실패해도 gpt/claude fallback을 타지 않는다.
        _preferred_single = ""
        try:
            _preferred_single = str(getattr(self.judge, "preferred_provider", "") or "").strip().lower()
            if _preferred_single in ("openai", "chatgpt"):
                _preferred_single = "gpt"
            if _preferred_single not in ("gemini", "gpt", "claude"):
                _preferred_single = ""
        except Exception:
            _preferred_single = ""

        if _preferred_single:
            judgment, llm_timed_out, err = self._judge_provider_with_timeout(
                provider=_preferred_single,
                system=system,
                user=user,
            )
        else:
            judgment, llm_timed_out, err = self._judge_with_timeout(system=system, user=user)
        if judgment is None:
            llm_action, llm_provider, rationale = self._llm_failure_fallback_action(
                t_res, merged_model_outputs, llm_timed_out=bool(llm_timed_out)
            )
            risk_level = "MEDIUM"
            caution = ""
            llm_raw = str(err or "")
            merged_model_outputs["pivot_candidate_probability"] = "LOW"
            merged_model_outputs["pivot_candidate_reason"] = "후보 없음"
            out = (llm_action, llm_provider, llm_timed_out, risk_level, rationale, caution, llm_raw, merged_model_outputs)
            return _cache_and_return(out)

        llm_action = str(judgment.action or "HOLD")
        risk_level = str(judgment.risk_level or "MEDIUM")
        rationale = str(judgment.rationale or "")
        caution = str(judgment.caution or "")
        llm_raw = str(judgment.raw or "")
        llm_provider = str(getattr(judgment, "provider", "") or "")
        merged_model_outputs["pivot_candidate_probability"] = str(
            getattr(judgment, "pivot_candidate_probability", "") or ""
        )
        merged_model_outputs["pivot_candidate_reason"] = str(
            getattr(judgment, "pivot_candidate_reason", "") or ""
        )
        out = (llm_action, llm_provider, llm_timed_out, risk_level, rationale, caution, llm_raw, merged_model_outputs)
        return _cache_and_return(out)

    def _notify_quota_exhausted(self, provider: str) -> None:
        """LLM provider 크레딧 소진(insufficient_quota) 감지 시 텔레그램 알람 전송.

        - 동일 provider에 대해 세션 내 1회만 전송 (중복 방지).
        - notifier가 없으면 조용히 무시.
        """
        try:
            notifier = getattr(self, "_notifier", None)
            if notifier is None:
                return

            # 세션 내 중복 전송 방지
            _sent_set = getattr(self, "_quota_alert_sent", None)
            if _sent_set is None:
                self._quota_alert_sent: set = set()
                _sent_set = self._quota_alert_sent
            prov = str(provider or "").strip().lower()
            if prov in _sent_set:
                return
            _sent_set.add(prov)

            prov_label = {"gpt": "GPT (OpenAI)", "gemini": "Gemini", "claude": "Claude"}.get(prov, prov.upper())
            msg = (
                f"🚨 <b>LLM 크레딧 소진 — {prov_label}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"<code>insufficient_quota</code> 에러 감지\n"
                f"해당 provider는 세션 종료까지 <b>비활성화</b>됩니다.\n\n"
                f"💳 크레딧 충전 후 재시작하세요.\n"
                f"충전 링크:\n"
                f"  • GPT: platform.openai.com/settings/organization/billing\n"
                f"  • Gemini: aistudio.google.com"
            )
            try:
                notifier.send_text(msg, parse_mode="HTML",
                                   debug_context={"kind": "quota_exhausted", "provider": prov})
            except Exception as _e:
                logger.debug("[QUOTA_ALERT] 텔레그램 전송 실패: %s", _e)
        except Exception as _e:
            logger.debug("[QUOTA_ALERT] 알람 처리 중 예외: %s", _e)

