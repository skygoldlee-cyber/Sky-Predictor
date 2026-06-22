from concurrent.futures import TimeoutError as FuturesTimeoutError


def test_pipeline_llm_timeout_fallback(monkeypatch) -> None:
    from prediction.pipeline import PredictionPipeline

    p = PredictionPipeline(use_llm=True)

    class _DummyRes:
        def __init__(self) -> None:
            self.prob = 0.7
            self.signal = "BUY"
            self.confidence = "HIGH"
            self.transformer_prob = 0.7
            self.tft_prob = None
            self.ensemble_method = "transformer_only"
            self.agreement = True

    def _raise_timeout(*args, **kwargs):
        raise FuturesTimeoutError()

    monkeypatch.setattr(p, "_judge_with_timeout", lambda *a, **k: (None, True, "timeout"))

    # Force numeric output without relying on tick buffers.
    monkeypatch.setattr(p, "_get_current_price_or_error", lambda: 400.0)
    monkeypatch.setattr(
        p,
        "_get_minute_df_or_error",
        lambda warmup_bars=0: __import__("pandas").DataFrame(
            {
                "Open": [1.0] * 50,
                "High": [1.0] * 50,
                "Low": [1.0] * 50,
                "Close": [1.0] * 50,
                "Volume": [1.0] * 50,
            }
        ),
    )
    monkeypatch.setattr(
        p,
        "_compute_adaptive_bundle",
        lambda df, now_dt: ({}, "", None, None, {"heuristic": {"action": "BUY"}}),
    )
    monkeypatch.setattr(p, "_build_and_predict_numeric", lambda **k: (_DummyRes(), None, None, None, [], {}, 0))
    monkeypatch.setattr(p, "_build_llm_prompt", lambda **k: ("sys", "user"))

    out = p.get_prediction(_now=__import__("datetime").datetime.now())

    assert out.get("signal") == "BUY"
    assert out.get("llm_timed_out") is True
    # llm_provider: LLM 타임아웃 시 heuristic fallback으로 변경됨
    # 구 동작: "timeout" | "error" | ""
    # 신 동작: "heuristic_fallback" (adaptive heuristic 결과 사용)
    assert str(out.get("llm_provider")) in ("timeout", "error", "", "heuristic_fallback")
    # rationale에 LLM 관련 메시지 포함 확인
    assert "LLM" in str(out.get("rationale") or "") or "휴리스틱" in str(out.get("rationale") or "")
