"""오프라인 검증용 캘리브레이션 지표 (Brier, ECE).

검증 세트에서 `confidence_*` / Conformal 임계값을 조정할 때 사용한다.
런타임 파이프라인 의존성 없음 (numpy만 필요).
"""

from __future__ import annotations

import numpy as np

__all__ = ["mean_brier_score", "expected_calibration_error"]


def mean_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """이진 라벨 ``labels ∈ {0,1}`` 에 대한 평균 Brier 점수."""
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 0.0, 1.0)
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.size != y.size or p.size == 0:
        raise ValueError("probs and labels must have the same non-empty shape")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """[0,1] 구간 등폭 빈에 대한 기대 캘리브레이션 오차 (ECE).

    각 빈에서 평균 확률(신뢰)과 실제 양성 비율의 차이를 가중 평균한다.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 0.0, 1.0)
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.size != y.size or p.size == 0:
        raise ValueError("probs and labels must have the same non-empty shape")
    nb = max(2, int(n_bins))
    edges = np.linspace(0.0, 1.0, nb + 1)
    ece = 0.0
    n = float(len(p))
    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        if i < nb - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)
