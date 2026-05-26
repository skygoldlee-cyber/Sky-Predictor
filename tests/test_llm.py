"""test_llm.py — LLM 초기화 진단 + 실제 API 호출 테스트

사용법:
    # 초기화 진단만 (기존 동작)
    python test_llm.py --config config.json

    # GPT + Gemini 실제 호출
    python test_llm.py --config config.json --call gpt gemini

    # Claude 포함 전체 호출
    python test_llm.py --config config.json --call gpt gemini claude

    # 타임아웃 지정 (기본 30초)
    python test_llm.py --config config.json --call gemini --timeout 60

    # secrets 파일 별도 지정
    python test_llm.py --config config.json --secrets config.secrets.json --call gpt gemini
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── 임의 테스트 프롬프트 ─────────────────────────────────────────────────────
_TEST_SYSTEM = """당신은 KP200 선물 방향 예측 전문가입니다.
아래 시장 데이터를 분석하고, 반드시 다음 JSON 형식으로만 응답하십시오:
{
  "action": "BUY" | "SELL" | "HOLD",
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "rationale": "판단 근거 (1~3문장)",
  "caution": "주의사항 (없으면 빈 문자열)"
}
JSON 외의 텍스트는 절대 포함하지 마십시오."""

_TEST_USER = """[시장 데이터 — 테스트 입력]
- 현재가: 385.25
- 전일 종가: 383.10
- 등락률: +0.56%
- 거래량: 125,430계약 (평균 대비 +12%)
- ATM IV: 14.2%
- Transformer 예측: BUY (confidence 0.72)
- ADX: 28.4 (추세 강도 보통)
- ZigZag 마지막 방향: UP

위 데이터를 기반으로 선물 방향을 판단하십시오."""


def _try_import(name: str) -> Dict[str, Any]:
    try:
        __import__(name)
        return {"ok": True, "error": ""}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _safe_get(d: Any, path: str) -> Any:
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _call_provider(
    judge: Any,
    provider: str,
    timeout: Optional[float],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """단일 provider에 대해 judge_provider() 호출 후 결과를 dict로 반환."""
    logger.info("[%s] API 호출 시작 (timeout=%.0fs) ...", provider.upper(), timeout or 30)
    t0 = time.monotonic()
    try:
        result = judge.judge_provider(
            provider,
            _TEST_SYSTEM,
            _TEST_USER,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        out = {
            "ok": True,
            "elapsed_s": round(elapsed, 2),
            "judgment": {
                "action": result.action,
                "risk_level": result.risk_level,
                "rationale": result.rationale,
                "caution": result.caution,
                "provider": result.provider,
            },
            "raw": result.raw,
        }
        # 실패 판단: rationale에 "LLM call failed" 포함 시
        if "LLM call failed" in (result.rationale or "") or "LLM provider invalid" in (result.rationale or ""):
            out["ok"] = False
            out["error"] = result.rationale
            logger.error("[%s] 호출 실패 (%.2fs): %s", provider.upper(), elapsed, result.rationale)
        else:
            logger.info(
                "[%s] 호출 성공 (%.2fs) → action=%s risk=%s",
                provider.upper(), elapsed, result.action, result.risk_level,
            )
    except Exception as e:
        elapsed = time.monotonic() - t0
        out = {
            "ok": False,
            "elapsed_s": round(elapsed, 2),
            "error": f"{type(e).__name__}: {e}",
            "judgment": None,
            "raw": "",
        }
        logger.error("[%s] 예외 발생 (%.2fs): %s", provider.upper(), elapsed, e)
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("test_llm")

    p = argparse.ArgumentParser(description="LLM 초기화 진단 + 실제 API 호출 테스트")
    p.add_argument("--config", default="config.json", help="메인 설정 파일 경로")
    p.add_argument("--secrets", default="", help="시크릿 설정 파일 경로 (선택)")
    p.add_argument(
        "--call",
        nargs="*",
        choices=["gpt", "gemini", "claude"],
        metavar="PROVIDER",
        help="실제 API를 호출할 provider 목록 (gpt / gemini / claude). 미지정 시 호출 생략.",
    )
    p.add_argument("--timeout", type=float, default=30.0, help="API 호출 타임아웃(초), 기본 30")
    args = p.parse_args()

    providers_to_call: List[str] = list(args.call or [])

    secrets_path = str(getattr(args, "secrets", "") or "").strip()
    if secrets_path:
        os.environ["APP_SECRETS_CONFIG"] = secrets_path

    logger.info("LLM 테스트 시작")
    logger.info("Python: %s", sys.executable)
    logger.info("Config: %s", args.config)
    if secrets_path:
        logger.info("Secrets (arg): %s", secrets_path)
    if providers_to_call:
        logger.info("호출 대상: %s  (timeout=%.0fs)", ", ".join(providers_to_call), args.timeout)
    else:
        logger.info("호출 대상: 없음 (초기화 진단만 실행)")

    result: Dict[str, Any] = {
        "python": sys.executable,
        "version": sys.version,
        "cwd": os.getcwd(),
        "config_path": str(args.config),
        "env": {
            "ANTHROPIC_API_KEY_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "OPENAI_API_KEY_present": bool(os.environ.get("OPENAI_API_KEY")),
            "GEMINI_API_KEY_present": bool(os.environ.get("GEMINI_API_KEY")),
            "APP_SECRETS_CONFIG": str(os.environ.get("APP_SECRETS_CONFIG") or ""),
        },
        "secrets": {
            "arg": secrets_path,
            "env": str(os.environ.get("APP_SECRETS_CONFIG") or ""),
        },
        "imports": {
            "anthropic": _try_import("anthropic"),
            "openai": _try_import("openai"),
            "google.genai": _try_import("google.genai"),
        },
        "config": {},
        "llm_judge": {},
        "api_calls": {},   # ← 실제 호출 결과
    }

    # ── secrets 파일 탐지 ───────────────────────────────────────────────────
    try:
        sp = str(os.environ.get("APP_SECRETS_CONFIG") or "").strip()
        if sp:
            result["secrets"]["exists"] = os.path.exists(sp)
            if result["secrets"]["exists"]:
                logger.info("Secrets 파일 발견: %s", sp)
            else:
                logger.error("Secrets 파일 없음: %s", sp)
        else:
            default_sp = os.path.join(
                os.path.dirname(os.path.abspath(str(args.config))), "config.secrets.json"
            )
            result["secrets"]["default"] = default_sp
            result["secrets"]["default_exists"] = os.path.exists(default_sp)
            if result["secrets"]["default_exists"]:
                logger.info("Secrets 파일 발견 (기본): %s", default_sp)
            else:
                logger.warning("Secrets 파일 없음 (기본): %s", default_sp)
    except Exception as e:
        result["secrets"]["probe_error"] = f"{type(e).__name__}: {e}"
        logger.error("Secrets 탐지 실패: %s", e)

    # ── SDK import 로그 ─────────────────────────────────────────────────────
    for pkg, info in (result.get("imports") or {}).items():
        if (info or {}).get("ok"):
            logger.info("SDK import OK: %s", pkg)
        else:
            logger.error("SDK import FAIL: %s (%s)", pkg, (info or {}).get("error") or "")

    # ── config 로드 ─────────────────────────────────────────────────────────
    cfg = None
    try:
        from config import load_config

        cfg = load_config(str(args.config))
        ai = getattr(cfg, "ai_providers", None)
        pred = getattr(cfg, "prediction", None)
        result["config"] = {
            "loaded": True,
            "ai": {
                "anthropic_key_present": bool(getattr(ai, "anthropic_key", None)),
                "openai_key_present": bool(getattr(ai, "openai_key", None)),
                "gemini_key_present": bool(getattr(ai, "gemini_key", None)),
            },
            "prediction": {
                "use_llm": bool(getattr(pred, "use_llm", True)),
                "dual_llm": bool(getattr(pred, "dual_llm", False)),
                "preferred_provider": str(getattr(pred, "preferred_provider", "") or ""),
            },
        }
        ai_s = result["config"]["ai"]
        logger.info(
            "Config 로드 OK — anthropic=%s openai=%s gemini=%s",
            str(bool(ai_s.get("anthropic_key_present"))).lower(),
            str(bool(ai_s.get("openai_key_present"))).lower(),
            str(bool(ai_s.get("gemini_key_present"))).lower(),
        )
    except Exception as e:
        result["config"] = {"loaded": False, "error": f"{type(e).__name__}: {e}"}
        logger.error("Config 로드 실패: %s", e)

    # ── LLMJudge 초기화 ─────────────────────────────────────────────────────
    judge = None
    try:
        from prediction.llm_judge import LLMJudge

        anthropic_key = openai_key = gemini_key = preferred_provider = None
        if cfg is not None:
            ai = getattr(cfg, "ai_providers", None)
            pred = getattr(cfg, "prediction", None)
            if ai:
                anthropic_key = getattr(ai, "anthropic_key", None)
                openai_key = getattr(ai, "openai_key", None)
                gemini_key = getattr(ai, "gemini_key", None)
            if pred:
                preferred_provider = getattr(pred, "preferred_provider", None)

        judge = LLMJudge(
            anthropic_key=str(anthropic_key or "") or None,
            openai_key=str(openai_key or "") or None,
            gemini_key=str(gemini_key or "") or None,
            preferred_provider=str(preferred_provider or "") or None,
        )
        clients = {
            "anthropic": bool(getattr(judge, "_anthropic", None)),
            "openai": bool(getattr(judge, "_openai", None)),
            "gemini": bool(getattr(judge, "_gemini", None)),
        }
        result["llm_judge"] = {
            "initialized": True,
            "clients": clients,
            "models": {
                "anthropic_model": str(getattr(judge, "anthropic_model", "") or ""),
                "openai_model": str(getattr(judge, "openai_model", "") or ""),
                "gemini_model": str(getattr(judge, "gemini_model", "") or ""),
            },
        }
        logger.info(
            "LLMJudge 초기화 OK — anthropic=%s openai=%s gemini=%s",
            str(clients["anthropic"]).lower(),
            str(clients["openai"]).lower(),
            str(clients["gemini"]).lower(),
        )
    except Exception as e:
        result["llm_judge"] = {"initialized": False, "error": f"{type(e).__name__}: {e}"}
        logger.error("LLMJudge 초기화 실패: %s", e)

    # ── 실제 API 호출 ───────────────────────────────────────────────────────
    if providers_to_call:
        if judge is None:
            logger.error("LLMJudge 초기화 실패로 API 호출을 건너뜁니다.")
            result["api_calls"]["_error"] = "LLMJudge not initialized"
        else:
            logger.info("=" * 60)
            logger.info("테스트 프롬프트:")
            logger.info("  [SYSTEM] %s", _TEST_SYSTEM[:80].replace("\n", " "))
            logger.info("  [USER]   %s", _TEST_USER[:120].replace("\n", " "))
            logger.info("=" * 60)

            for prov in providers_to_call:
                result["api_calls"][prov] = _call_provider(judge, prov, args.timeout, logger)

            # 요약
            logger.info("=" * 60)
            total = len(providers_to_call)
            success = sum(1 for v in result["api_calls"].values() if isinstance(v, dict) and v.get("ok"))
            logger.info("API 호출 결과: %d/%d 성공", success, total)
            for prov in providers_to_call:
                call_res = result["api_calls"].get(prov, {})
                if call_res.get("ok"):
                    jdg = call_res.get("judgment") or {}
                    logger.info(
                        "  ✓ %-8s  action=%-4s  risk=%-6s  elapsed=%.2fs",
                        prov, jdg.get("action", "?"), jdg.get("risk_level", "?"), call_res.get("elapsed_s", 0),
                    )
                else:
                    logger.error(
                        "  ✗ %-8s  error=%s  elapsed=%.2fs",
                        prov, str(call_res.get("error", ""))[:80], call_res.get("elapsed_s", 0),
                    )
    else:
        logger.info("--call 미지정: API 호출 생략 (초기화 진단만 완료)")

    # ── 전체 결과 JSON 출력 ─────────────────────────────────────────────────
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # ── 최종 상태 ───────────────────────────────────────────────────────────
    clients = (result.get("llm_judge") or {}).get("clients") or {}
    any_client = any(bool(v) for v in clients.values()) if isinstance(clients, dict) else False

    if providers_to_call:
        any_success = any(
            isinstance(v, dict) and v.get("ok")
            for v in result.get("api_calls", {}).values()
        )
        if any_success:
            logger.info("RESULT: SUCCESS — API 호출 성공")
        else:
            logger.error("RESULT: FAIL — 모든 API 호출 실패")
        return 0 if any_success else 1
    else:
        if any_client:
            logger.info("RESULT: SUCCESS — 최소 1개 provider 클라이언트 초기화됨")
        else:
            logger.error("RESULT: FAIL — provider 클라이언트 없음")
        return 0 if any_client else 1


if __name__ == "__main__":
    raise SystemExit(main())
