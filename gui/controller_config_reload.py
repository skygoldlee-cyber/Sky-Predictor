"""config.json에서 예측 effective 기본값 병합 (`gui_controller` 12단계 분리)."""

from __future__ import annotations

from typing import Any, MutableMapping

__all__ = ["merge_prediction_effective_from_loaded_config"]


def merge_prediction_effective_from_loaded_config(
    config: Any,
    effective_target: MutableMapping[str, Any],
) -> bool:
    """``config.prediction`` 필드를 ``effective_target``에 반영한다.

    Returns:
        True: 병합 수행.
        False: ``prediction`` 이 없어 변경 없음 (호출부에서 조용히 return).
    """
    p = getattr(config, "prediction", None)
    if p is None:
        return False
    effective_target.update(
        {
            "llm_min_interval_sec": float(getattr(p, "llm_min_interval_sec", 30.0) or 0.0),
            "tick_size": float(getattr(p, "tick_size", 0.05) or 0.0),
            "feedback_threshold_ticks": int(getattr(p, "feedback_threshold_ticks", 10) or 10),
            "feedback_skip_hold_ticks": int(getattr(p, "feedback_skip_hold_ticks", 2) or 0),
            "feedback_weight_high": float(getattr(p, "feedback_weight_high", 1.0) or 0.0),
            "feedback_weight_mid": float(getattr(p, "feedback_weight_mid", 0.5) or 0.0),
            "feedback_weight_low": float(getattr(p, "feedback_weight_low", 0.25) or 0.0),
            "feedback_use_price_snapshot": bool(getattr(p, "feedback_use_price_snapshot", True)),
            "feedback_snapshot_tolerance_sec": float(
                getattr(p, "feedback_snapshot_tolerance_sec", 30.0) or 0.0
            ),
            "feedback_snapshot_required": bool(getattr(p, "feedback_snapshot_required", False)),
            "fc0_stale_threshold_sec": float(getattr(p, "fc0_stale_threshold_sec", 10.0) or 0.0),
            "fc0_stale_cooldown_sec": float(getattr(p, "fc0_stale_cooldown_sec", 60.0) or 0.0),
        }
    )
    return True
