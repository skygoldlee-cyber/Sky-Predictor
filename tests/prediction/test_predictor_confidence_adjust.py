"""predictor: Conformal 신뢰도 보정·연속 불일치 헬퍼."""

from __future__ import annotations

from prediction.predictor import (
    _max_pairwise_abs_diff,
    adjust_confidence_by_conformal_interval_width,
)


def test_adjust_confidence_no_interval() -> None:
    assert adjust_confidence_by_conformal_interval_width("HIGH", None, None) == "HIGH"


def test_adjust_confidence_wide_forces_low() -> None:
    # width = 0.6 >= medium max 0.55
    assert (
        adjust_confidence_by_conformal_interval_width(
            "HIGH",
            0.2,
            0.8,
            width_max_for_high=0.35,
            width_max_for_medium=0.55,
        )
        == "LOW"
    )


def test_adjust_confidence_high_downgrade_to_medium() -> None:
    # width 0.4: between 0.35 and 0.55 → HIGH → MEDIUM
    assert (
        adjust_confidence_by_conformal_interval_width(
            "HIGH",
            0.3,
            0.7,
            width_max_for_high=0.35,
            width_max_for_medium=0.55,
        )
        == "MEDIUM"
    )


def test_max_pairwise_abs_diff() -> None:
    assert _max_pairwise_abs_diff([0.5]) == 0.0
    assert abs(_max_pairwise_abs_diff([0.0, 1.0]) - 1.0) < 1e-9
    assert abs(_max_pairwise_abs_diff([0.0, 0.5, 1.0]) - 1.0) < 1e-9
