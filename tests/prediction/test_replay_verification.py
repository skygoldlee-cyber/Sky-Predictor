from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def _dummy_minute_df(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [1.0] * n,
            "High": [1.0] * n,
            "Low": [1.0] * n,
            "Close": [1.0] * n,
            "Volume": [1.0] * n,
        }
    )


def test_feedback_snapshot_required_uses_price_at(monkeypatch) -> None:
    from prediction.pipeline import PredictionPipeline

    dt0 = datetime(2026, 3, 2, 9, 0, 0)

    p = PredictionPipeline(
        use_llm=False,
        numeric_predictor="ensemble",
        prediction_minutes=5,
        feedback_use_price_snapshot=True,
        feedback_snapshot_required=True,
        feedback_snapshot_tolerance_sec=0.0,
        feedback_skip_hold_ticks=0,
    )

    class _DummyRes:
        def __init__(self) -> None:
            self.prob = 0.7
            self.signal = "BUY"
            self.confidence = "HIGH"
            self.transformer_prob = 0.9
            self.tft_prob = 0.9
            self.ensemble_method = "weighted_avg"
            self.agreement = True

    monkeypatch.setattr(p, "_get_current_price_or_error", lambda: 999.0)
    monkeypatch.setattr(p, "_get_minute_df_or_error", lambda warmup_bars=0: _dummy_minute_df())
    monkeypatch.setattr(p, "_compute_adaptive_bundle", lambda df, now_dt: ({}, "", None, None, {"heuristic": {"action": "BUY"}}))
    monkeypatch.setattr(p, "_build_llm_prompt", lambda **k: ("sys", "user"))
    monkeypatch.setattr(
        p,
        "_run_llm_judgment",
        lambda **k: (
            "BUY",
            "disabled",
            False,
            "LOW",
            "",
            "",
            "",
            k.get("model_outputs") or {},
        ),
    )
    monkeypatch.setattr(
        p,
        "_build_and_predict_numeric",
        lambda **k: (_DummyRes(), None, None, None, [], {}, 0),
    )

    calls = {"n": 0}

    def _spy_update(*, transformer_correct: bool, tft_correct: bool, **_kw) -> None:
        calls["n"] += 1

    monkeypatch.setattr(p.numeric_predictor, "update_adaptive_weights", _spy_update, raising=False)

    target_dt = dt0
    entry_dt = dt0 - timedelta(minutes=6)

    # Provide a minute-bucketed FC0 snapshot at target time.
    minute_key = target_dt.replace(second=0, microsecond=0)
    p.tick_processor.futures_minute_data[minute_key] = [
        {"timestamp": target_dt, "price": 400.6, "volume": 1, "cvolume": 1}
    ]

    p._feedback_queue.append(
        {
            "ts_epoch": float(entry_dt.timestamp()),
            "target_ts_epoch": float(target_dt.timestamp()),
            "price": 400.0,
            "transformer_prob": 0.9,
            "tft_prob": 0.9,
        }
    )

    out = p.get_prediction(_now=dt0)
    assert isinstance(out, dict)
    assert calls["n"] >= 1

    m = p.get_metrics()
    assert int(m.get("feedback_snapshot_used") or 0) >= 1


def test_feedback_snapshot_required_uses_near_tick_with_tolerance(monkeypatch) -> None:
    from prediction.pipeline import PredictionPipeline

    dt0 = datetime(2026, 3, 2, 9, 0, 0)

    p = PredictionPipeline(
        use_llm=False,
        numeric_predictor="ensemble",
        prediction_minutes=5,
        feedback_use_price_snapshot=True,
        feedback_snapshot_required=True,
        feedback_snapshot_tolerance_sec=30.0,
        feedback_skip_hold_ticks=0,
    )

    class _DummyRes:
        def __init__(self) -> None:
            self.prob = 0.7
            self.signal = "BUY"
            self.confidence = "HIGH"
            self.transformer_prob = 0.9
            self.tft_prob = 0.9
            self.ensemble_method = "weighted_avg"
            self.agreement = True

    monkeypatch.setattr(p, "_get_current_price_or_error", lambda: 999.0)
    monkeypatch.setattr(p, "_get_minute_df_or_error", lambda warmup_bars=0: _dummy_minute_df())
    monkeypatch.setattr(p, "_compute_adaptive_bundle", lambda df, now_dt: ({}, "", None, None, {"heuristic": {"action": "BUY"}}))
    monkeypatch.setattr(p, "_build_llm_prompt", lambda **k: ("sys", "user"))
    monkeypatch.setattr(
        p,
        "_run_llm_judgment",
        lambda **k: (
            "BUY",
            "disabled",
            False,
            "LOW",
            "",
            "",
            "",
            k.get("model_outputs") or {},
        ),
    )
    monkeypatch.setattr(
        p,
        "_build_and_predict_numeric",
        lambda **k: (_DummyRes(), None, None, None, [], {}, 0),
    )

    calls = {"n": 0}

    def _spy_update(*, transformer_correct: bool, tft_correct: bool, **_kw) -> None:
        calls["n"] += 1

    monkeypatch.setattr(p.numeric_predictor, "update_adaptive_weights", _spy_update, raising=False)

    target_dt = dt0
    entry_dt = dt0 - timedelta(minutes=6)

    # Ensure get_price_at misses by leaving futures_minute_data empty, but provide a near tick.
    p.tick_processor.futures_ticks.append(
        {"timestamp": target_dt + timedelta(seconds=10), "price": 400.6, "volume": 1, "cvolume": 1}
    )

    p._feedback_queue.append(
        {
            "ts_epoch": float(entry_dt.timestamp()),
            "target_ts_epoch": float(target_dt.timestamp()),
            "price": 400.0,
            "transformer_prob": 0.9,
            "tft_prob": 0.9,
        }
    )

    out = p.get_prediction(_now=dt0)
    assert isinstance(out, dict)
    assert calls["n"] >= 1

    m = p.get_metrics()
    assert int(m.get("feedback_snapshot_used") or 0) >= 1
