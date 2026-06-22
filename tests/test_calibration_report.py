"""calibration_report / calibration_thresholds 스모크."""

from __future__ import annotations

import numpy as np

from prediction.calibration_report import build_validation_report
from prediction.calibration_thresholds import format_tunable_keys_reference


def test_build_validation_report_smoke() -> None:
    rng = np.random.default_rng(1)
    n = 100
    p = rng.uniform(0.2, 0.8, size=n)
    y = (rng.uniform(0, 1, size=n) < p).astype(np.float64)
    s = build_validation_report(p, y, include_tunable_reference=False)
    assert "Brier" in s
    assert "ECE" in s


def test_tunable_reference_contains_keys() -> None:
    t = format_tunable_keys_reference()
    assert "disagreement_hold_prob_diff_max" in t
    assert "buy_threshold" in t
