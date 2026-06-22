"""LLM judgment module.

This module provides a small abstraction (`LLMJudge`) that:
- calls one of the supported providers (Claude/OpenAI/Gemini)
- parses the response as JSON
- normalizes fields into a stable `LLMJudgment` dataclass

All provider SDKs are treated as optional dependencies.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import (
    CLAUDE_FALLBACK_MODELS,
    CLAUDE_MODEL,
    GEMINI_FALLBACK_MODELS,
    GEMINI_MODEL,
    GPT_FALLBACK_MODELS,
    GPT_MODEL,
    LLM_EMPTY_OUTPUT_MAX_RETRIES,
)


logger = logging.getLogger(__name__)  # ARC-09: 하드코딩 "kp200_predictor" → __name__ (prediction.llm_judge)


def _show_sdk_missing_warning(
    provider: str,
    package: str,
    extra: str = "",
    notifier: Any = None,
) -> None:
    """SDK 미설치 시 GUI 메시지박스 + 텔레그램 알림을 전송한다.

    - PySide6 QApplication이 있을 때만 메시지박스 표시.
    - notifier(TelegramNotifier)가 주어지면 텔레그램으로도 전송.
    - 별도 스레드에서 호출돼도 안전하도록 QTimer.singleShot으로 메인 스레드에 위임.
    """
    msg_plain = (
        f"[{provider}] SDK가 설치되지 않아 LLM 판단을 사용할 수 없습니다.\n\n"
        f"터미널에서 아래 명령을 실행한 뒤 프로그램을 재시작하세요:\n\n"
        f"    pip install {package}\n"
    )
    if extra:
        msg_plain += f"\n※ {extra}"

    # ── 텔레그램 전송 ──────────────────────────────────────────────────────
    if notifier is not None:
        try:
            tg_msg = (
                f"⚠️ <b>LLM SDK 미설치 — {provider}</b>\n\n"
                f"<code>pip install {package}</code> 실행 후 재시작 필요."
            )
            if extra:
                tg_msg += f"\n\n※ {extra}"
            notifier.send_text(tg_msg, parse_mode="HTML",
                               debug_context={"kind": "sdk_missing", "provider": provider})
        except Exception as _te:
            logger.debug("[SDK_WARN] 텔레그램 전송 실패: %s", _te)

    # ── GUI 메시지박스 ────────────────────────────────────────────────────
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from PySide6.QtCore import QTimer

        app = QApplication.instance()
        if app is None:
            return  # GUI 없는 환경(CLI/스케줄러) — 로그만으로 충분

        def _show():
            try:
                box = QMessageBox()
                box.setWindowTitle("LLM SDK 미설치")
                box.setIcon(QMessageBox.Icon.Warning)
                box.setText(msg_plain)
                box.exec()
            except Exception as _e:
                logger.debug("[SDK_WARN] 메시지박스 표시 실패: %s", _e)

        QTimer.singleShot(0, _show)
    except Exception:
        pass  # PySide6 자체가 없는 환경 — 무시


@dataclass
class LLMJudgment:
    """Normalized LLM output.

    Attributes:
        action: BUY/SELL/HOLD.
        risk_level: LOW/MEDIUM/HIGH.
        rationale: Short explanation.
        caution: One-line risk note.
        raw: Raw response text.
        provider: Provider identifier used for this response.
    """
    action: str
    risk_level: str
    rationale: str
    caution: str
    raw: str
    provider: str = ""
    pivot_candidate_probability: str = ""
    pivot_candidate_reason: str = ""
    pivot_confirmed: str = ""


class LLMJudge:
    """Call an LLM provider and parse a strict JSON decision.

    Provider selection:
    - Uses whichever client(s) can be initialized from available keys.
    - Applies `preferred_provider` first if it is available.
    - Falls back to other providers on failure.
    """
    def __init__(
        self,
        *,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        anthropic_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        notifier: Any = None,
    ):
        """Initialize provider clients.

        Args:
            anthropic_key/openai_key/gemini_key: API keys.
            preferred_provider: Preferred provider name (claude|gpt|gemini).
            anthropic_model/openai_model/gemini_model: Override model names.
            notifier: TelegramNotifier 인스턴스. SDK 미설치 시 텔레그램 알림에 사용.
        """
        self._notifier = notifier
        self.anthropic_key = anthropic_key
        self.openai_key = openai_key
        self.gemini_key = gemini_key

        self.preferred_provider = str(preferred_provider or "").strip().lower() or None
        if self.preferred_provider in ("openai", "chatgpt"):
            self.preferred_provider = "gpt"
        if self.preferred_provider not in (None, "claude", "gpt", "gemini"):
            self.preferred_provider = None

        self.anthropic_model = str(anthropic_model or CLAUDE_MODEL)
        self.openai_model = str(openai_model or GPT_MODEL)
        self.gemini_model = str(gemini_model or GEMINI_MODEL)

        self._anthropic = None
        self._openai = None
        self._gemini = None

        self._gemini_bad_models: set[str] = set()
        self._openai_bad_models: set[str] = set()

        self._claude_fallback_models = [str(x) for x in (CLAUDE_FALLBACK_MODELS or ()) if str(x).strip()]
        self._gemini_fallback_models = [str(x) for x in (GEMINI_FALLBACK_MODELS or ()) if str(x).strip()]
        self._openai_fallback_models = [str(x) for x in (GPT_FALLBACK_MODELS or ()) if str(x).strip()]

        if self.anthropic_key:
            try:
                import anthropic  # type: ignore

                self._anthropic = anthropic.Anthropic(api_key=self.anthropic_key)
                logger.info("Anthropic client initialized")
                logger.info("Anthropic selected model: %s", self.anthropic_model)
            except ImportError as e:
                self._anthropic = None
                logger.error("Anthropic SDK import 실패 (pip install anthropic 필요): %s", e)
                logger.info("Anthropic selected model: %s", self.anthropic_model)
            except Exception as e:
                self._anthropic = None
                logger.error("Anthropic client 초기화 실패: %s", e, exc_info=True)
                logger.info("Anthropic selected model: %s", self.anthropic_model)
        else:
            logger.info("Anthropic client not initialized (missing ANTHROPIC_API_KEY)")
            logger.info("Anthropic selected model: %s", self.anthropic_model)

        if self.openai_key:
            try:
                import openai  # type: ignore

                # max_retries=0: SDK 내장 retry 비활성화. 429 처리는 pipeline.py에서 직접 수행.
                self._openai = openai.OpenAI(api_key=self.openai_key, max_retries=0)
                logger.info("OpenAI client initialized")
                self._log_openai_models()
                try:
                    self.openai_model = self._select_openai_model(self.openai_model)
                except Exception:
                    pass
                logger.info("OpenAI selected model: %s", self.openai_model)
            except ImportError as e:
                self._openai = None
                logger.error("OpenAI SDK import 실패 (pip install openai 필요): %s", e)
                _show_sdk_missing_warning("OpenAI", "openai", notifier=self._notifier)
                logger.info("OpenAI supported models (top 25): (skipped; client not available)")
                logger.info("OpenAI selected model: %s", self.openai_model)
            except Exception as e:
                self._openai = None
                logger.error("OpenAI client 초기화 실패: %s", e, exc_info=True)
                logger.info("OpenAI supported models (top 25): (skipped; client not available)")
                logger.info("OpenAI selected model: %s", self.openai_model)
        else:
            logger.info("OpenAI client not initialized (missing OPENAI_API_KEY)")
            logger.info("OpenAI supported models (top 25): (skipped; missing API key)")
            logger.info("OpenAI selected model: %s", self.openai_model)

        if self.gemini_key:
            try:
                from google import genai  # type: ignore

                self._gemini = genai.Client(api_key=self.gemini_key)
                logger.info("Gemini client initialized")
                self._log_gemini_models()
                try:
                    self.gemini_model = self._select_gemini_model(self.gemini_model)
                except Exception:
                    pass
                logger.info("Gemini selected model: %s", self.gemini_model)
            except ImportError as e:
                self._gemini = None
                logger.error(
                    "Gemini SDK import 실패 (pip install google-genai 필요; "
                    "google-generativeai 구버전과 혼용 불가): %s", e
                )
                _show_sdk_missing_warning(
                    "Gemini", "google-genai",
                    "google-generativeai 구버전과 혼용 불가 — 구버전은 먼저 제거하세요.",
                    notifier=self._notifier,
                )
                logger.info("Gemini supported models (top 25): (skipped; client not available)")
                logger.info("Gemini selected model: %s", self.gemini_model)
            except Exception as e:
                self._gemini = None
                logger.error("Gemini client 초기화 실패: %s", e, exc_info=True)
                logger.info("Gemini supported models (top 25): (skipped; client not available)")
                logger.info("Gemini selected model: %s", self.gemini_model)
        else:
            logger.info("Gemini client not initialized (missing GEMINI_API_KEY)")
            logger.info("Gemini supported models (top 25): (skipped; missing API key)")
            logger.info("Gemini selected model: %s", self.gemini_model)

    @staticmethod
    def _normalize_gemini_model_name(name: str) -> str:
        try:
            s = str(name or "").strip()
        except Exception:
            return ""
        if s.startswith("models/"):
            return s.split("/", 1)[1].strip()
        return s

    def _select_gemini_model(self, desired: str) -> str:
        """Pick a best-effort usable Gemini model name.

        - If `models.list()` works and includes the desired model, keep it.
        - Otherwise choose the first match from `GEMINI_FALLBACK_MODELS` that exists.
        - Fall back to the desired model as-is when listing is unavailable.
        """
        d = self._normalize_gemini_model_name(desired)
        if not self._gemini:
            return d

        try:
            models = self._gemini.models.list()
            names = [str(getattr(m, "name", "")).strip() for m in models]
            names = [m for m in names if m]
            norm_names = {self._normalize_gemini_model_name(n) for n in names}
            if d and d in norm_names:
                return d

            for cand in self._gemini_fallback_models:
                c = self._normalize_gemini_model_name(cand)
                if c and c in norm_names:
                    return c
        except Exception:
            return d

        return d

    @staticmethod
    def _is_gemini_model_error(err: Exception) -> bool:
        try:
            s = str(err or "").lower()
        except Exception:
            return False
        # NOTE:
        # - "model" 단어 단독 매칭은 과민하여 timeout/네트워크 오류도 bad_models로 오분류할 수 있다.
        # - 모델 자체 문제를 강하게 시사하는 패턴만 허용한다.
        strong_patterns = (
            "model not found",
            "not found",
            "does not exist",
            "unknown model",
            "invalid model",
            "unsupported model",
            "not supported for generatecontent",
            "permission denied",
            "access denied",
            "forbidden",
        )
        return any(p in s for p in strong_patterns)

    def _log_openai_models(self) -> None:
        """_log_openai_models.
"""
        if not self._openai:
            return
        try:
            models = self._openai.models.list()
            data = getattr(models, "data", None) or []
            ids = [str(getattr(m, "id", "")).strip() for m in data]
            ids = [m for m in ids if m]
            if not ids:
                return
            logger.info("OpenAI supported models (top 25): %s", ", ".join(ids[:25]))
        except Exception:
            return

    def _select_openai_model(self, desired: str) -> str:
        """Pick a best-effort usable OpenAI model name.

        - If `models.list()` works and includes the desired model, keep it.
        - Otherwise choose the first match from `GPT_FALLBACK_MODELS` that exists.
        - Fall back to the desired model as-is when listing is unavailable.
        """
        d = str(desired or "").strip()
        if not self._openai:
            return d

        try:
            models = self._openai.models.list()
            data = getattr(models, "data", None) or []
            ids = [str(getattr(m, "id", "")).strip() for m in data]
            ids = [m for m in ids if m]
            if d and d in ids:
                return d

            for cand in self._openai_fallback_models:
                c = str(cand or "").strip()
                if c and c in ids:
                    return c
        except Exception:
            return d

        return d

    @staticmethod
    def _is_openai_model_error(err: Exception) -> bool:
        try:
            s = str(err or "").lower()
        except Exception:
            return False
        keys = (
            "model",
            "not found",
            "does not exist",
            "invalid",
            "unknown",
            "unsupported",
        )
        return any(k in s for k in keys)

    def _log_gemini_models(self) -> None:
        """_log_gemini_models.
"""
        if not self._gemini:
            return
        try:
            models = self._gemini.models.list()
            names = [str(getattr(m, "name", "")).strip() for m in models]
            names = [m for m in names if m]
            if not names:
                return
            logger.info("Gemini supported models (top 25): %s", ", ".join(names[:25]))
        except Exception:
            return

    # ── 리소스 해제 (RES-02) ────────────────────────────────────────────────

    def close(self) -> None:
        """HTTP 클라이언트 연결 풀을 명시적으로 해제한다.

        anthropic, openai, gemini SDK 클라이언트가 내부적으로 유지하는
        세션/소켓을 닫아 리소스 누수를 방지한다.
        """
        for attr in ("_anthropic", "_openai", "_gemini"):
            client = getattr(self, attr, None)
            if client is None:
                continue
            fn = getattr(client, "close", None)
            if callable(fn):
                try:
                    fn()
                    logger.debug("[LLMJudge] %s 클라이언트 닫힘", attr)
                except Exception as e:
                    logger.debug("[LLMJudge] %s close 실패 (무시): %s", attr, e)
            setattr(self, attr, None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def judge(self, system: str, user: str, *, timeout: Optional[float] = None) -> LLMJudgment:
        """Run judgment using the first available provider (with fallback)."""
        providers = self._provider_order()
        if not providers:
            return LLMJudgment(
                action="HOLD",
                risk_level="MEDIUM",
                rationale="LLM disabled (missing key or dependency).",
                caution="",
                raw="",
                provider="",
            )

        last_err = None
        for prov in providers:
            try:
                raw = self._call_provider(provider=prov, system=system, user=user, timeout=timeout)
                data = self.parse_json(raw)

                action = self._norm_action(data.get("action"))
                risk = self._norm_risk(data.get("risk_level"))
                rationale = str(data.get("rationale") or "")
                caution = str(data.get("caution") or "")
                pivot_prob = str(data.get("pivot_candidate_probability") or "")
                pivot_reason = str(data.get("pivot_candidate_reason") or "")

                if not str(raw or "").strip():
                    action = action or "HOLD"
                    risk = risk or "HIGH"
                    if not rationale:
                        rationale = "LLM empty output"

                return LLMJudgment(
                    action=str(action or "HOLD"),
                    risk_level=str(risk or "MEDIUM"),
                    rationale=rationale,
                    caution=caution,
                    pivot_candidate_probability=pivot_prob,
                    pivot_candidate_reason=pivot_reason,
                    raw=str(raw or ""),
                    provider=str(prov),
                )
            except Exception as e:
                last_err = e
                continue

        return LLMJudgment(
            action="HOLD",
            risk_level="HIGH",
            rationale=f"LLM call failed: {last_err}",
            caution="",
            raw="",
            provider="",
        )

    def judge_provider(self, provider: str, system: str, user: str, *, timeout: Optional[float] = None) -> LLMJudgment:
        """Run judgment using a specific provider (no fallback).

        This is intended for debugging / dual-LLM mode so we can capture multiple
        providers' outputs in a single prediction round.
        """
        prov = str(provider or "").strip().lower()
        if prov in ("openai", "chatgpt"):
            prov = "gpt"
        if prov not in ("claude", "gpt", "gemini"):
            return LLMJudgment(
                action="HOLD",
                risk_level="HIGH",
                rationale=f"LLM provider invalid: {provider}",
                caution="",
                raw="",
                provider=str(provider or ""),
            )

        try:
            raw = self._call_provider(provider=prov, system=system, user=user, timeout=timeout)
            data = self.parse_json(raw)

            action = self._norm_action(data.get("action"))
            risk = self._norm_risk(data.get("risk_level"))
            rationale = str(data.get("rationale") or "")
            caution = str(data.get("caution") or "")
            pivot_prob = str(data.get("pivot_candidate_probability") or "")
            pivot_reason = str(data.get("pivot_candidate_reason") or "")

            if not str(raw or "").strip():
                action = action or "HOLD"
                risk = risk or "HIGH"
                if not rationale:
                    rationale = "LLM empty output"

            return LLMJudgment(
                action=str(action or "HOLD"),
                risk_level=str(risk or "MEDIUM"),
                rationale=rationale,
                caution=caution,
                pivot_candidate_probability=pivot_prob,
                pivot_candidate_reason=pivot_reason,
                raw=str(raw or ""),
                provider=str(prov),
            )
        except Exception as e:
            return LLMJudgment(
                action="HOLD",
                risk_level="HIGH",
                rationale=f"LLM call failed ({prov}): {e}",
                caution="",
                raw="",
                provider=str(prov),
            )

    def _provider_order(self) -> list[str]:
        """Return provider attempt order based on availability and preference.

        기본 우선순위: gemini → gpt → claude
        preferred_provider 지정 시 해당 provider가 최우선.
        """
        available: list[str] = []
        if self._gemini is not None:
            available.append("gemini")   # 우선
        if self._openai is not None:
            available.append("gpt")
        if self._anthropic is not None:
            available.append("claude")

        pref = self.preferred_provider
        if pref and pref in available:
            rest = [x for x in available if x != pref]
            return [pref] + rest
        return available

    @staticmethod
    def _is_claude_model_error(err: Exception) -> bool:
        try:
            s = str(err or "").lower()
        except Exception:
            return False
        keys = (
            "model",
            "not found",
            "does not exist",
            "invalid",
            "unknown",
            "unsupported",
        )
        return any(k in s for k in keys)

    def _call_provider(self, *, provider: str, system: str, user: str, timeout: Optional[float] = None) -> str:
        """Call a specific provider and return raw text.

        Raises:
            RuntimeError/ValueError when provider is unavailable or unknown.
        """
        p = str(provider).strip().lower()
        if p == "claude":
            if self._anthropic is None:
                raise RuntimeError("anthropic client unavailable")

            models_to_try = [str(self.anthropic_model)]
            for cand in self._claude_fallback_models:
                c = str(cand).strip()
                if c and c not in models_to_try:
                    models_to_try.append(c)

            last_err: Optional[Exception] = None
            resp = None
            for model_name in models_to_try:
                kwargs: Dict[str, Any] = {
                    "model": str(model_name),
                    "max_tokens": 512,
                    "temperature": 0.2,
                    "system": str(system),
                    "messages": [{"role": "user", "content": str(user)}],
                }
                if timeout is not None:
                    kwargs["timeout"] = float(timeout)
                try:
                    try:
                        resp = self._anthropic.messages.create(**kwargs)
                    except TypeError:
                        kwargs.pop("timeout", None)
                        resp = self._anthropic.messages.create(**kwargs)
                    if model_name != self.anthropic_model:
                        self.anthropic_model = str(model_name)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if not self._is_claude_model_error(e):
                        break

            if resp is None:
                raise RuntimeError(f"claude messages.create failed: {last_err}")

            try:
                return str(resp.content[0].text).strip()
            except Exception:
                return str(resp)

        if p in ("gpt", "openai"):
            if self._openai is None:
                raise RuntimeError("openai client unavailable")
            models_to_try = [str(self.openai_model)]
            for cand in self._openai_fallback_models:
                c = str(cand).strip()
                if c and c not in models_to_try:
                    models_to_try.append(c)

            last_err3: Optional[Exception] = None
            resp = None
            for model_name in models_to_try:
                try:
                    if str(model_name) in self._openai_bad_models:
                        continue
                except Exception:
                    pass

                kwargs = {
                    "model": str(model_name),
                    "temperature": 0.2,
                    "max_tokens": 512,
                    "messages": [
                        {"role": "system", "content": str(system)},
                        {"role": "user", "content": str(user)},
                    ],
                }
                if timeout is not None:
                    kwargs["timeout"] = float(timeout)
                try:
                    try:
                        resp = self._openai.chat.completions.create(**kwargs)
                    except TypeError:
                        kwargs.pop("timeout", None)
                        resp = self._openai.chat.completions.create(**kwargs)
                    if model_name != self.openai_model:
                        self.openai_model = str(model_name)
                    last_err3 = None
                    break
                except Exception as e:
                    last_err3 = e
                    if self._is_openai_model_error(e):
                        try:
                            self._openai_bad_models.add(str(model_name))
                        except Exception:
                            pass
                        continue
                    break

            if resp is None:
                raise RuntimeError(f"openai chat.completions.create failed: {last_err3}")
            try:
                return str(resp.choices[0].message.content or "").strip()
            except Exception:
                return str(resp)

        if p == "gemini":
            if self._gemini is None:
                raise RuntimeError("gemini client unavailable")

            # [LLM-FIX-6] Gemini는 system_instruction을 별도 파라미터로 지원한다.
            # 기존 concat 방식은 역할 혼동을 유발하여 JSON 포맷 불이행률을 높인다.
            # system_instruction 파라미터가 지원되지 않는 구버전 SDK는 concat으로 fallback한다.
            contents_concat = (str(system).strip() + "\n" + str(user).strip()).strip()

            models_to_try = [self._normalize_gemini_model_name(self.gemini_model)]
            for cand in self._gemini_fallback_models:
                c = self._normalize_gemini_model_name(cand)
                if c and c not in models_to_try:
                    models_to_try.append(c)

            last_err2: Optional[Exception] = None
            resp = None
            for model_name in models_to_try:
                try:
                    if str(model_name) in self._gemini_bad_models:
                        # FIX-BADMODEL-ERR: bad_models skip 시 last_err2를 설정한다.
                        # 모든 모델이 skip되면 last_err2=None인 채로 루프가 끝나
                        # RuntimeError("gemini generate_content failed: None") 가 발생하는 것을 방지.
                        if last_err2 is None:
                            last_err2 = RuntimeError(
                                f"model '{model_name}' skipped (bad_models)"
                            )
                        continue
                except Exception:
                    pass

                # [LLM-FIX-2] 빈 응답 재시도: Gemini는 간헐적으로 빈 텍스트를 반환한다.
                # LLM_EMPTY_OUTPUT_MAX_RETRIES 횟수만큼 재시도한다.
                _max_empty_retries = max(1, int(LLM_EMPTY_OUTPUT_MAX_RETRIES or 3))
                for _attempt in range(_max_empty_retries):
                    kwargs: Dict[str, Any] = {"model": str(model_name), "contents": str(user).strip()}
                    if timeout is not None:
                        kwargs["timeout"] = float(timeout)
                    try:
                        # system_instruction 지원 여부에 따라 분기
                        try:
                            from google.genai import types as _genai_types  # type: ignore
                            config_obj = _genai_types.GenerateContentConfig(
                                system_instruction=str(system).strip(),
                                temperature=0.2,
                            )
                            kwargs["config"] = config_obj
                            resp = self._gemini.models.generate_content(**kwargs)
                        except (ImportError, TypeError, AttributeError):
                            # SDK가 system_instruction을 지원하지 않으면 concat fallback
                            kwargs.pop("config", None)
                            kwargs["contents"] = contents_concat
                            try:
                                resp = self._gemini.models.generate_content(**kwargs)
                            except TypeError:
                                kwargs.pop("timeout", None)
                                resp = self._gemini.models.generate_content(**kwargs)
                        except Exception:
                            kwargs.pop("config", None)
                            kwargs["contents"] = contents_concat
                            resp = self._gemini.models.generate_content(**kwargs)

                        # [LLM-FIX-2] 빈 응답 감지 및 재시도
                        _text = None
                        try:
                            _text = getattr(resp, "text", None)
                            # candidates[0].content.parts[0].text 경로도 시도
                            if not _text:
                                _cands = getattr(resp, "candidates", None)
                                if _cands:
                                    _parts = getattr(getattr(_cands[0], "content", None), "parts", None)
                                    if _parts:
                                        _text = getattr(_parts[0], "text", None)
                        except Exception:
                            _text = None

                        if _text and str(_text).strip():
                            if model_name != self.gemini_model:
                                self.gemini_model = str(model_name)
                            last_err2 = None
                            resp = resp  # keep ref
                            break  # 빈 응답 retry loop 탈출

                        # 빈 응답 — 마지막 시도가 아니면 재시도
                        if _attempt < _max_empty_retries - 1:
                            logger.warning(
                                "[LLM_GEMINI_EMPTY] model=%s attempt=%d/%d — retrying",
                                model_name, _attempt + 1, _max_empty_retries,
                            )
                            resp = None
                            continue
                        else:
                            # 모든 재시도 소진 → empty_output 에러로 처리
                            raise RuntimeError("gemini_empty_output")

                    except RuntimeError as e:
                        if "gemini_empty_output" in str(e):
                            last_err2 = e
                            # model fallback 시도
                            break
                        raise
                    except Exception as e:
                        last_err2 = e
                        if self._is_gemini_model_error(e):
                            try:
                                self._gemini_bad_models.add(str(model_name))
                                logger.warning(
                                    "[LLM_GEMINI_BADMODEL_ADD] model=%s reason=%s bad_models=%s",
                                    str(model_name),
                                    str(e),
                                    ",".join(sorted(str(m) for m in self._gemini_bad_models)),
                                )
                            except Exception:
                                pass
                            resp = None
                            break  # 다음 model로
                        break  # 비-모델 에러는 즉시 중단

                if resp is not None and last_err2 is None:
                    break  # model loop 탈출

            if resp is None:
                # FIX-NONE-MSG: last_err2가 None이면 "all models exhausted" 메시지로 교체.
                # None 문자열이 rationale에 그대로 표시되어 원인 파악이 어려웠다.
                _err_msg = str(last_err2) if last_err2 is not None else "all fallback models exhausted or skipped"
                raise RuntimeError(f"gemini generate_content failed: {_err_msg}")

            try:
                t = getattr(resp, "text", None)
                if t and str(t).strip():
                    return str(t).strip()
            except Exception:
                pass
            # candidates 경로 fallback
            try:
                _cands = getattr(resp, "candidates", None)
                if _cands:
                    _parts = getattr(getattr(_cands[0], "content", None), "parts", None)
                    if _parts:
                        _t = getattr(_parts[0], "text", None)
                        if _t and str(_t).strip():
                            return str(_t).strip()
            except Exception:
                pass
            return str(resp)

        raise ValueError(f"unknown provider: {provider}")

    @staticmethod
    def _norm_action(v: Any) -> str:
        """Normalize action tokens into BUY/SELL/HOLD."""
        try:
            s = str(v or "").strip().upper()
        except Exception:
            s = ""
        if s in ("BUY", "SELL", "HOLD"):
            return s
        if s in ("LONG", "B"):
            return "BUY"
        if s in ("SHORT", "S"):
            return "SELL"
        if s in ("WAIT", "NEUTRAL", "NONE"):
            return "HOLD"
        return "HOLD"

    @staticmethod
    def _norm_risk(v: Any) -> str:
        """Normalize risk tokens into LOW/MEDIUM/HIGH."""
        try:
            s = str(v or "").strip().upper()
        except Exception:
            s = ""
        if s in ("LOW", "MEDIUM", "HIGH"):
            return s
        if s in ("MID", "M"):
            return "MEDIUM"
        if s in ("L",):
            return "LOW"
        if s in ("H",):
            return "HIGH"
        return "MEDIUM"

    @staticmethod
    def parse_json(raw: str) -> Dict[str, Any]:
        """Best-effort JSON object extraction from a provider response.

        Strategy:
        1) Prefer fenced code blocks when present.
        2) Try full-string JSON parsing.
        3) Try raw_decode from the first '{'.
        4) Balanced-brace extraction (provider may wrap JSON with prose).
        5) Simple brace slicing fallback.
        6) [LLM-FIX-4] 키워드 추출 최후 수단 — JSON 파싱이 모두 실패해도
           action/risk_level 키워드를 텍스트에서 직접 추출하여 partial dict 반환.
        """
        s = str(raw or "").strip()
        if not s:
            return {}

        # 1) fenced code block extraction
        if "```" in s:
            try:
                parts = s.split("```")
                for i in range(1, len(parts), 2):
                    block = parts[i].strip("\n")
                    lines = block.splitlines()
                    if lines and lines[0].strip().lower() in ("json", "javascript", "js"):
                        block = "\n".join(lines[1:]).strip()
                    if "{" in block and "}" in block:
                        s = block
                        break
            except Exception:
                pass

        # 2) direct json
        try:
            v = json.loads(s)
            return v if isinstance(v, dict) else {}
        except Exception:
            pass

        # 3) raw decode first dict in text
        try:
            decoder = json.JSONDecoder()
            for i, ch in enumerate(s):
                if ch != "{":
                    continue
                try:
                    obj, _end = decoder.raw_decode(s[i:])
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
        except Exception:
            pass

        # 4) regex/balanced-brace extraction (provider may wrap JSON with prose)
        try:
            s2 = s[:8192]
            start = s2.find("{")
            if start != -1:
                depth = 0
                end_pos = None
                for j in range(start, len(s2)):
                    ch = s2[j]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end_pos = j
                            break
                if end_pos is not None and end_pos > start:
                    cand = s2[start : end_pos + 1]
                    try:
                        v = json.loads(cand)
                        return v if isinstance(v, dict) else {}
                    except Exception:
                        pass

            # non-greedy match (may fail on nested braces)
            m = re.search(r"\{.*?\}", s2, flags=re.DOTALL)
            if m:
                v = json.loads(m.group(0))
                return v if isinstance(v, dict) else {}
        except Exception:
            pass

        # 5) fallback: simple brace slicing (length-limited; no regex)
        try:
            s2 = s[:4096]
            a = s2.find("{")
            b = s2.rfind("}")
            if a != -1 and b != -1 and b > a:
                return json.loads(s2[a : b + 1])
        except Exception:
            pass

        # 6) [LLM-FIX-4] 키워드 추출 최후 수단.
        # LLM이 JSON 형식을 어기고 자연어로 응답할 때 action/risk_level을 최소한 추출한다.
        # 이 경로가 자주 실행된다면 프롬프트 강화 또는 모델 교체가 필요하다.
        try:
            su = s.upper()
            result: Dict[str, Any] = {}

            # action 추출
            for act in ("BUY", "SELL", "HOLD"):
                if re.search(rf"\b{act}\b", su):
                    result["action"] = act
                    break

            # risk_level 추출
            for risk in ("HIGH", "MEDIUM", "LOW"):
                if re.search(rf"\b{risk}\b", su):
                    result["risk_level"] = risk
                    break

            if result:
                # rationale/caution은 원본 텍스트 앞부분 요약으로 채움
                result.setdefault("action", "HOLD")
                result.setdefault("risk_level", "MEDIUM")
                result["rationale"] = s[:200].strip()
                result["caution"] = ""
                logger.warning(
                    "[LLM_JSON_FALLBACK] JSON 파싱 실패, 키워드 추출로 대체: action=%s risk=%s raw_len=%d",
                    result["action"], result["risk_level"], len(s),
                )
                return result
        except Exception:
            pass

        return {}
