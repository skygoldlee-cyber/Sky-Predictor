"""검증 세트용 Brier/ECE 요약 리포트 (오프라인)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from prediction.calibration_metrics import expected_calibration_error, mean_brier_score
from prediction.calibration_thresholds import format_tunable_keys_reference

__all__ = ["build_validation_report", "format_tunable_keys_reference"]


def build_validation_report(
    probs: Any,
    labels: Any,
    *,
    n_bins: int = 10,
    include_tunable_reference: bool = True,
) -> str:
    """이진 라벨 ``labels ∈ {0,1}`` 과 예측 확률 ``probs`` 에 대한 요약 문자열.

    Args:
        probs: shape (N,) 예측 확률 [0,1]
        labels: shape (N,) 실제 라벨 (상승=1, 하락=0 등)
        n_bins: ECE 빈 개수
        include_tunable_reference: True면 튜닝 키 목록을 하단에 덧붙임
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.size != y.size or p.size == 0:
        raise ValueError("probs and labels must be same non-empty length")

    brier = mean_brier_score(p, y)
    ece = expected_calibration_error(p, y, n_bins=int(n_bins))

    lines = [
        "=== Validation calibration report ===",
        f"samples: {int(p.size)}",
        f"mean Brier score: {brier:.6f}",
        f"ECE ({n_bins} bins): {ece:.6f}",
        "",
    ]
    if include_tunable_reference:
        lines.append(format_tunable_keys_reference())
    return "\n".join(lines)
