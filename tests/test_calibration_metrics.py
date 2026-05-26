"""prediction.calibration_metrics 단위 테스트."""

from __future__ import annotations

import numpy as np

from prediction.calibration_metrics import expected_calibration_error, mean_brier_score


def test_mean_brier_perfect() -> None:
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 1.0])
    assert mean_brier_score(p, y) == 0.0


def test_mean_brier_wrong() -> None:
    y = np.array([1.0, 0.0])
    p = np.array([0.0, 1.0])
    assert abs(mean_brier_score(p, y) - 1.0) < 1e-9


def test_ece_finite_well_behaved() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < p).astype(np.float64)
    ece = expected_calibration_error(p, y, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_ece_miscalibrated_high() -> None:
    # Always predict 0.9, half positives
    p = np.full(1000, 0.9)
    y = np.array([1.0] * 500 + [0.0] * 500)
    ece = expected_calibration_error(p, y, n_bins=10)
    assert ece > 0.05
