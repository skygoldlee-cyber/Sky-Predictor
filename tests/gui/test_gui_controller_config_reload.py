"""gui_controller_config_reload 순수 로직."""

from __future__ import annotations

from types import SimpleNamespace

from gui.controller_config_reload import merge_prediction_effective_from_loaded_config


def test_merge_returns_false_when_no_prediction() -> None:
    cfg = SimpleNamespace(prediction=None)
    target: dict[str, object] = {"tick_size": 0.99}
    assert merge_prediction_effective_from_loaded_config(cfg, target) is False
    assert target["tick_size"] == 0.99


def test_merge_updates_target_from_prediction() -> None:
    p = SimpleNamespace(
        llm_min_interval_sec=60.0,
        tick_size=0.1,
        feedback_threshold_ticks=20,
        feedback_skip_hold_ticks=3,
        feedback_weight_high=1.0,
        feedback_weight_mid=0.6,
        feedback_weight_low=0.3,
        feedback_use_price_snapshot=False,
        feedback_snapshot_tolerance_sec=45.0,
        feedback_snapshot_required=True,
        fc0_stale_threshold_sec=12.0,
        fc0_stale_cooldown_sec=90.0,
    )
    cfg = SimpleNamespace(prediction=p)
    target: dict[str, object] = {}
    assert merge_prediction_effective_from_loaded_config(cfg, target) is True
    assert target["llm_min_interval_sec"] == 60.0
    assert target["tick_size"] == 0.1
    assert target["feedback_threshold_ticks"] == 20
    assert target["feedback_use_price_snapshot"] is False
    assert target["fc0_stale_cooldown_sec"] == 90.0
