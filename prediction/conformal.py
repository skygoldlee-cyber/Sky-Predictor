"""Conformal Prediction 래퍼.

Split Conformal Prediction (Venn-ABERS 간략판) 구현.
추가 모델 훈련 없이 기존 predictor 의 출력값과 검증 레이블만으로
커버리지가 보장된 예측 구간 [lower, upper] 를 생성한다.

사용 방법:
    1. 훈련 후 별도 검증셋(칼리브레이션셋)에서 calibrate() 호출
    2. 이후 predict_interval(prob) 으로 구간 조회
    3. 체크포인트에 quantile 값을 함께 저장하면 재훈련 없이 재사용 가능

수학적 보장:
    α = conformal_alpha 로 설정할 때
    P(y ∈ [lower, upper]) ≥ 1 - α (marginal coverage)
    단, 칼리브레이션셋이 i.i.d. 조건을 만족할 때.

KP200 선물 적용 시 주의:
    - 장중 레짐 전환이 잦으면 i.i.d. 가정이 약화된다.
    - 권장: 당일 개장 직후 30분 분을 칼리브레이션에 포함해 재보정.
    - conformal_alpha = 0.1 → 90% 커버리지 구간 (기본값)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """Split Conformal Prediction 래퍼.

    핵심 아이디어:
        비적합 점수(nonconformity score) = |prob - label| 를 검증셋에서 계산하고
        (1 - α) 분위수 q 를 저장한다.
        예측 시 [prob - q, prob + q] 를 구간으로 반환한다.

    Args:
        alpha: 오차율. 0.1 → 90% 커버리지 (기본값).
    """

    _VERSION = 1  # 체크포인트 호환성 관리용

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(max(0.01, min(0.5, alpha)))
        self._quantile: Optional[float] = None  # calibrate() 후 설정
        self._n_cal: int = 0                    # 칼리브레이션 샘플 수 (품질 지표)

    # ── 칼리브레이션 ───────────────────────────────────────────────────────────

    def calibrate(
        self,
        probs: "np.ndarray",
        labels: "np.ndarray",
    ) -> float:
        """검증셋에서 비적합 점수 분위수를 계산해 저장한다.

        Args:
            probs:  모델 출력 확률 (N,) float, 범위 [0, 1].
            labels: 실제 레이블 (N,) int/float, 값 0 또는 1.

        Returns:
            계산된 분위수 q (저장됨).
        """
        p = np.asarray(probs, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()

        if len(p) != len(y) or len(p) == 0:
            raise ValueError(
                f"probs 와 labels 의 길이가 같아야 합니다: got {len(p)} vs {len(y)}"
            )

        # 비적합 점수: 예측 확률과 실제 레이블 간 절댓값 차이
        scores = np.abs(p - y)

        # Conformal 분위수: ceil((n+1)(1-α))/n 번째 순서통계량
        n = len(scores)
        level = float(math.ceil((n + 1) * (1.0 - self.alpha))) / float(n)
        level = min(1.0, max(0.0, level))

        q = float(np.quantile(scores, level))
        self._quantile = q
        self._n_cal = n

        logger.info(
            "[Conformal] calibrated: n=%d alpha=%.2f q=%.4f (coverage≥%.0f%%)",
            n, self.alpha, q, (1.0 - self.alpha) * 100,
        )
        return q

    # ── 예측 구간 ──────────────────────────────────────────────────────────────

    def predict_interval(self, prob: float) -> Tuple[float, float]:
        """prob 에 대한 커버리지 보장 예측 구간을 반환한다.

        Args:
            prob: 모델 출력 확률 [0, 1].

        Returns:
            (lower, upper): 클리핑된 구간 [0, 1] 내.

        Raises:
            RuntimeError: calibrate() 가 호출되지 않은 경우.
        """
        if self._quantile is None:
            raise RuntimeError(
                "ConformalPredictor.calibrate() 를 먼저 호출해야 합니다. "
                "또는 load() 로 저장된 분위수를 불러오세요."
            )
        p = float(prob)
        q = float(self._quantile)
        lower = float(np.clip(p - q, 0.0, 1.0))
        upper = float(np.clip(p + q, 0.0, 1.0))
        return lower, upper

    def interval_width(self) -> float:
        """현재 분위수 기준 구간 폭 (2q). 칼리브레이션 전이면 1.0 반환."""
        if self._quantile is None:
            return 1.0
        return float(min(1.0, 2.0 * self._quantile))

    def is_calibrated(self) -> bool:
        return self._quantile is not None

    # ── 저장 / 불러오기 ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """분위수와 메타정보를 npz 로 저장한다."""
        Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            version=np.array([self._VERSION], dtype=np.int32),
            alpha=np.array([self.alpha], dtype=np.float64),
            quantile=np.array([self._quantile if self._quantile is not None else np.nan]),
            n_cal=np.array([self._n_cal], dtype=np.int64),
        )
        logger.info("[Conformal] saved to %s (q=%.4f, n=%d)", path, self._quantile or 0, self._n_cal)

    @classmethod
    def load(cls, path: str) -> "ConformalPredictor":
        """저장된 분위수를 불러온다."""
        data = np.load(str(path))
        alpha = float(data["alpha"][0])
        obj = cls(alpha=alpha)
        q = float(data["quantile"][0])
        if not math.isnan(q):
            obj._quantile = q
        obj._n_cal = int(data.get("n_cal", np.array([0]))[0])
        logger.info("[Conformal] loaded from %s (q=%.4f, n=%d)", path, obj._quantile or 0, obj._n_cal)
        return obj

    @classmethod
    def load_or_create(cls, path: str, alpha: float = 0.1) -> "ConformalPredictor":
        """파일이 있으면 load, 없으면 빈 인스턴스를 반환한다."""
        try:
            if Path(str(path)).exists():
                return cls.load(path)
        except Exception as e:
            logger.warning("[Conformal] load 실패 (%s) — 새 인스턴스 반환: %s", path, e)
        return cls(alpha=alpha)

    def __repr__(self) -> str:
        q_str = f"{self._quantile:.4f}" if self._quantile is not None else "미보정"
        return (
            f"ConformalPredictor(alpha={self.alpha:.2f}, q={q_str}, "
            f"n_cal={self._n_cal}, width={self.interval_width():.4f})"
        )
