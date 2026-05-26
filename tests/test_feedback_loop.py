from __future__ import annotations


def test_pipeline_feedback_updates_weights(monkeypatch) -> None:
    from datetime import datetime, timedelta

    from prediction.pipeline import PredictionPipeline

    p = PredictionPipeline(
        use_llm=False,
        numeric_predictor="ensemble",
        feedback_skip_hold_ticks=0,
        feedback_use_price_snapshot=False,
        feedback_snapshot_required=False,
    )

    # Force numeric output without relying on tick buffers.
    class _DummyRes:
        def __init__(self, *, prob: float, transformer_prob: float, tft_prob: float) -> None:
            self.prob = float(prob)
            self.signal = "BUY"
            self.confidence = "HIGH"
            self.transformer_prob = float(transformer_prob)
            self.tft_prob = float(tft_prob)
            self.ensemble_method = "weighted_avg"
            self.agreement = True

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
    monkeypatch.setattr(p, "_build_llm_prompt", lambda **k: ("sys", "user"))
    monkeypatch.setattr(
        p,
        "_run_llm_judgment",
        lambda **k: ("BUY", "disabled", False, "LOW", "", "", "", k.get("model_outputs") or {}),
    )

    # Track whether update_adaptive_weights gets called.
    calls = {"n": 0}

    def _spy_update(*, transformer_correct: bool, tft_correct: bool, **_kw) -> None:
        calls["n"] += 1

    monkeypatch.setattr(p.numeric_predictor, "update_adaptive_weights", _spy_update, raising=False)

    # Seed a matured feedback record: 6 minutes ago.
    now = datetime.now()
    entry = {
        "ts_epoch": float((now - timedelta(minutes=6)).timestamp()),
        "price": 400.0,
        "transformer_prob": 0.9,
        "tft_prob": 0.9,
    }
    p._feedback_queue.append(entry)

    # Current price moves +10 ticks (0.05 * 10 = 0.5)
    monkeypatch.setattr(p, "_get_current_price_or_error", lambda: 400.6)
    monkeypatch.setattr(
        p,
        "_build_and_predict_numeric",
        lambda **k: (_DummyRes(prob=0.7, transformer_prob=0.9, tft_prob=0.9), None, None, None, [], {}, 0),
    )

    out = p.get_prediction(_now=now)
    assert isinstance(out, dict)
    assert calls["n"] >= 1



def test_pipeline_reset_adaptive_weights_best_effort() -> None:
    from prediction.pipeline import PredictionPipeline

    p = PredictionPipeline(use_llm=False, numeric_predictor="ensemble")
    ok = p.reset_adaptive_weights()
    assert isinstance(ok, bool)


# ──────────────────────────────────────────────────────────────────
# NW-TST-02: 추가 경계 케이스
# ──────────────────────────────────────────────────────────────────

def test_feedback_snapshot_required_skips_without_snapshot(monkeypatch) -> None:
    """feedback_snapshot_required=True 이고 스냅샷 없을 때 가중치 업데이트 미발생 검증."""
    from datetime import datetime, timedelta
    from prediction.pipeline import PredictionPipeline

    p = PredictionPipeline(
        use_llm=False,
        numeric_predictor="ensemble",
        feedback_use_price_snapshot=True,   # 스냅샷 사용
        feedback_snapshot_required=True,    # 스냅샷 없으면 skip
        feedback_skip_hold_ticks=0,
    )

    calls = {"n": 0}

    def _spy(**kw):
        calls["n"] += 1

    monkeypatch.setattr(p.numeric_predictor, "update_adaptive_weights", _spy, raising=False)

    # get_price_at / get_price_near 를 None 반환으로 mock
    monkeypatch.setattr(p.tick_processor, "get_price_at", lambda dt: None, raising=False)
    monkeypatch.setattr(p.tick_processor, "get_price_near", lambda dt, **kw: None, raising=False)

    now = datetime.now()
    entry = {
        "ts_epoch": float((now - timedelta(minutes=6)).timestamp()),
        "target_ts_epoch": float((now - timedelta(minutes=1)).timestamp()),
        "price": 400.0,
        "transformer_prob": 0.8,
        "tft_prob": 0.8,
    }
    p._feedback_queue.append(entry)

    # _maybe_process_feedback 직접 호출
    p._maybe_process_feedback(now_dt=now, current_price=400.5)

    assert calls["n"] == 0, \
        f"snapshot_required=True인데 스냅샷 없이 가중치 업데이트 {calls['n']}회 발생"


def test_feedback_skip_hold_small_move(monkeypatch) -> None:
    """skip_hold_ticks 기준 이하 소폭 움직임은 피드백 평가를 건너뛰는지 검증."""
    from datetime import datetime, timedelta
    from prediction.pipeline import PredictionPipeline

    # tick_size=0.05, skip_hold_ticks=3 → skip 임계값 = 0.15
    p = PredictionPipeline(
        use_llm=False,
        numeric_predictor="ensemble",
        feedback_skip_hold_ticks=3,
        feedback_snapshot_required=False,
        feedback_use_price_snapshot=False,
    )

    calls = {"n": 0}

    def _spy(**kw):
        calls["n"] += 1

    monkeypatch.setattr(p.numeric_predictor, "update_adaptive_weights", _spy, raising=False)

    now = datetime.now()
    entry = {
        "ts_epoch": float((now - timedelta(minutes=6)).timestamp()),
        "price": 400.0,
        "transformer_prob": 0.8,
        "tft_prob": 0.8,
    }
    p._feedback_queue.append(entry)

    # 소폭 변동: +0.05 (1틱) — skip_thr(0.15) 미만이므로 건너뛰어야 함
    p._maybe_process_feedback(now_dt=now, current_price=400.05)

    assert calls["n"] == 0, \
        f"소폭 HOLD 이동인데 가중치 업데이트 {calls['n']}회 발생"


def test_feedback_queue_maturation_time(monkeypatch) -> None:
    """prediction_minutes 미경과 레코드는 평가되지 않는지 검증."""
    from datetime import datetime, timedelta
    from prediction.pipeline import PredictionPipeline

    p = PredictionPipeline(
        use_llm=False,
        numeric_predictor="ensemble",
        prediction_minutes=5,
        feedback_snapshot_required=False,
        feedback_use_price_snapshot=False,
    )

    calls = {"n": 0}

    def _spy(**kw):
        calls["n"] += 1

    monkeypatch.setattr(p.numeric_predictor, "update_adaptive_weights", _spy, raising=False)

    now = datetime.now()
    # 2분 전 레코드 — 아직 5분 horizon 미도달
    entry = {
        "ts_epoch": float((now - timedelta(minutes=2)).timestamp()),
        "price": 400.0,
        "transformer_prob": 0.8,
        "tft_prob": 0.8,
    }
    p._feedback_queue.append(entry)

    p._maybe_process_feedback(now_dt=now, current_price=401.0)

    assert calls["n"] == 0, \
        f"미성숙 레코드인데 가중치 업데이트 {calls['n']}회 발생"
    # 큐에 레코드가 아직 남아 있어야 함
    assert len(p._feedback_queue) == 1, \
        "미성숙 레코드가 큐에서 제거됨"


def test_feedback_weight_bounds_after_many_correct(monkeypatch) -> None:
    """transformer 연속 정답 시 가중치가 [0.0, 1.0] 범위를 벗어나지 않는지 검증."""
    from prediction.pipeline import PredictionPipeline

    p = PredictionPipeline(use_llm=False, numeric_predictor="ensemble")

    tracker = p.numeric_predictor._adaptive_weight_tracker  # AdaptiveEnsembleWeightTracker

    # transformer 100번 연속 정답, TFT 100번 연속 오답
    for _ in range(100):
        tracker.update(transformer_correct=True, tft_correct=False)

    t_w, f_w = tracker.get_weights()

    assert 0.0 <= t_w <= 1.0, f"transformer 가중치 범위 초과: {t_w}"
    assert 0.0 <= f_w <= 1.0, f"tft 가중치 범위 초과: {f_w}"
    assert abs(t_w + f_w - 1.0) < 1e-6, f"가중치 합 != 1.0: {t_w + f_w}"
    # transformer가 압도적으로 우세해야 함
    assert t_w > 0.8, f"transformer 연속 정답인데 가중치 낮음: {t_w}"
