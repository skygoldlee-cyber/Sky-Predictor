"""LLM timeout / rate-limit fallback 테스트.

NW-TST-01: 운영 중 가장 빈번한 시나리오인 LLM 타임아웃 및 429 rate-limit 처리를
검증한다. 모든 테스트는 실제 LLM 호출 없이 monkeypatch로 대체한다.
"""

from __future__ import annotations

import time
from datetime import datetime


def _make_pipeline(**kwargs):
    from prediction.pipeline import PredictionPipeline
    defaults = dict(use_llm=False, numeric_predictor="ensemble")
    defaults.update(kwargs)
    return PredictionPipeline(**defaults)


def _seed_price(p, price: float = 400.0) -> None:
    import time as _t
    try:
        p._fc0_ticks.append({"price": price, "chetime": "090000", "_ts_epoch": _t.time()})
    except Exception:
        pass


def _dummy_df():
    import pandas as pd
    return pd.DataFrame(
        {"Open": [1.0]*50, "High": [1.0]*50,
         "Low": [1.0]*50, "Close": [1.0]*50, "Volume": [1.0]*50}
    )


class _DummyRes:
    prob = 0.65
    signal = "BUY"
    confidence = "HIGH"
    transformer_prob = 0.65
    tft_prob = 0.65
    ensemble_method = "weighted_avg"
    agreement = True
    feature_snapshot: dict = {}


# ──────────────────────────────────────────────────────────────────
# 테스트 1: LLM 타임아웃 → transformer 신호 그대로 반환
# ──────────────────────────────────────────────────────────────────

def test_llm_timeout_falls_back_to_transformer_signal(monkeypatch) -> None:
    """LLM 타임아웃 시 transformer 신호가 결과에 반영되는지 검증."""
    p = _make_pipeline(use_llm=True)
    _seed_price(p)

    monkeypatch.setattr(p, "_get_current_price_or_error", lambda: 400.0)
    monkeypatch.setattr(p, "_get_minute_df_or_error", lambda warmup_bars=0: _dummy_df())
    monkeypatch.setattr(p, "_compute_adaptive_bundle",
                        lambda df, now_dt: ({}, "", None, None, {"heuristic": {"action": "BUY"}}))
    monkeypatch.setattr(p, "_build_llm_prompt", lambda **k: ("sys", "user"))
    monkeypatch.setattr(p, "_run_llm_judgment",
                        lambda **k: ("BUY", "timeout", True, "LOW", "", "", "", {}))
    monkeypatch.setattr(p, "_build_and_predict_numeric",
                        lambda **k: (_DummyRes(), None, None, None, [], {}, 0))

    out = p.get_prediction(_now=datetime.now())
    assert isinstance(out, dict)
    assert "error" not in out, f"예상치 못한 error: {out.get('error')}"
    assert out.get("signal") in ("BUY", "SELL", "HOLD")
    provider = str(out.get("llm_provider", ""))
    assert "timeout" in provider or out.get("signal") == "BUY", \
        f"timeout fallback 미반영: llm_provider={provider!r}"


# ──────────────────────────────────────────────────────────────────
# 테스트 2: LLM 429 → _llm_rate_limited_until_epoch 설정 확인
# ──────────────────────────────────────────────────────────────────

def test_llm_429_sets_rate_limit_cooldown() -> None:
    """LLM 429 응답 시 _llm_rate_limited_until_epoch 가 미래 시각으로 설정되는지 검증."""
    from concurrent.futures import Future
    from prediction.pipeline import PredictionPipeline
    from config import LLM_COOLDOWN_SECONDS_ON_429

    p = PredictionPipeline(use_llm=True, numeric_predictor="ensemble")
    # _ensure_llm_executor는 _reset_llm_executor로 대체됨.
    # executor를 None에서 새 인스턴스로 초기화하는 역할이므로
    # 테스트에서는 아래처럼 _Fake429Executor를 직접 주입한다.

    class _Fake429Executor:
        def submit(self, fn, *a, **kw):
            f = Future()
            f.set_exception(Exception("HTTP 429 Too Many Requests"))
            return f

    before = time.time()
    p._llm_executor = _Fake429Executor()

    judgment, timed_out, err = p._judge_with_timeout(system="sys", user="user")

    assert judgment is None
    assert timed_out is False
    assert err is not None
    assert p._llm_rate_limited_until_epoch > before, \
        "_llm_rate_limited_until_epoch 가 설정되지 않음"
    expected_min = before + float(LLM_COOLDOWN_SECONDS_ON_429) - 5
    assert p._llm_rate_limited_until_epoch >= expected_min, \
        f"cooldown 값 부족: {p._llm_rate_limited_until_epoch - before:.0f}s"


# ──────────────────────────────────────────────────────────────────
# 테스트 3: dual_llm — LLM 최소 1회 호출
# ──────────────────────────────────────────────────────────────────

def test_dual_llm_calls_llm_at_least_once(monkeypatch) -> None:
    """dual_llm 모드에서 LLM이 최소 1회 이상 호출되는지 검증."""
    p = _make_pipeline(use_llm=True, dual_llm=True)
    _seed_price(p)

    class _HoldRes:
        prob = 0.55
        signal = "HOLD"
        confidence = "LOW"
        transformer_prob = 0.55
        tft_prob = 0.55
        ensemble_method = "weighted_avg"
        agreement = False
        feature_snapshot: dict = {}

    monkeypatch.setattr(p, "_get_current_price_or_error", lambda: 400.0)
    monkeypatch.setattr(p, "_get_minute_df_or_error", lambda warmup_bars=0: _dummy_df())
    monkeypatch.setattr(p, "_compute_adaptive_bundle",
                        lambda df, now_dt: ({}, "", None, None, {}))
    monkeypatch.setattr(p, "_build_llm_prompt", lambda **k: ("sys", "user"))
    monkeypatch.setattr(p, "_build_and_predict_numeric",
                        lambda **k: (_HoldRes(), None, None, None, [], {}, 0))

    calls: list = []

    def _fake_run_llm(**kwargs):
        calls.append(1)
        return ("HOLD", "secondary_ok", False, "LOW", "ok", "", "", {})

    monkeypatch.setattr(p, "_run_llm_judgment", _fake_run_llm)

    out = p.get_prediction(_now=datetime.now())
    assert isinstance(out, dict)
    assert len(calls) >= 1, "dual_llm 모드에서 LLM 미호출"


# ──────────────────────────────────────────────────────────────────
# 테스트 4: off_boundary=True + rate_limited → LLM skip
# ──────────────────────────────────────────────────────────────────

def test_off_boundary_skips_when_rate_limited(monkeypatch) -> None:
    """rate_limited 상태에서 off_boundary=True 호출이 LLM을 건너뛰는지 검증."""
    p = _make_pipeline(use_llm=True)
    _seed_price(p)

    monkeypatch.setattr(p, "_get_current_price_or_error", lambda: 400.0)
    monkeypatch.setattr(p, "_get_minute_df_or_error", lambda warmup_bars=0: _dummy_df())
    monkeypatch.setattr(p, "_compute_adaptive_bundle",
                        lambda df, now_dt: ({}, "", None, None, {"heuristic": {"action": "BUY"}}))
    monkeypatch.setattr(p, "_build_and_predict_numeric",
                        lambda **k: (_DummyRes(), None, None, None, [], {}, 0))
    monkeypatch.setattr(p, "_build_llm_prompt", lambda **k: ("sys", "user"))

    judge_called = {"n": 0}

    def _spy(**kwargs):
        judge_called["n"] += 1
        return ("BUY", "should_not_be_called", False, "LOW", "", "", "", {})

    monkeypatch.setattr(p, "_run_llm_judgment", _spy)

    # rate-limit 강제 설정
    p._llm_rate_limited_until_epoch = time.time() + 300.0

    out = p.get_prediction(_now=datetime.now(), off_boundary=True)

    assert isinstance(out, dict)
    assert "error" not in out
    assert judge_called["n"] == 0, \
        f"rate-limited 상태에서 LLM이 {judge_called['n']}회 호출됨"
    assert out.get("signal") in ("BUY", "SELL", "HOLD")
