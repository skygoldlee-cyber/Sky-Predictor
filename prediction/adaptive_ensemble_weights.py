"""Adaptive Ensemble Weight Tracker.

최근 예측 결과를 기반으로 모델별 가중치를 동적 조정합니다.
"""
from __future__ import annotations

from collections import deque
from typing import Optional


class AdaptiveEnsembleWeightTracker:
    """최근 예측 결과를 기반으로 모델별 가중치를 동적 조정.
    
    Transformer와 TFT의 방향 적중률을 추적하여 동적으로 가중치를 조정합니다.
    """
    
    def __init__(self, window: int = 20, decay: float = 0.95):
        """초기화.
        
        Args:
            window: 추적할 최근 예측 수
            decay: 가중치 감소 계수 (과거 예측의 영향을 줄임)
        """
        self._transformer_hits = deque(maxlen=window)
        self._tft_hits = deque(maxlen=window)
        self._decay = float(decay)
    
    def update(self, transformer_correct: bool, tft_correct: bool) -> None:
        """예측 결과 업데이트.
        
        Args:
            transformer_correct: Transformer 예측이 맞았는지
            tft_correct: TFT 예측이 맞았는지
        """
        self._transformer_hits.append(float(transformer_correct))
        self._tft_hits.append(float(tft_correct))
    
    def get_weights(self) -> tuple[float, float]:
        """현재 가중치 반환.
        
        Returns:
            (transformer_weight, tft_weight): 합이 1.0인 가중치 튜플
        """
        if not self._transformer_hits or not self._tft_hits:
            return 0.5, 0.5
        
        t_acc = sum(self._transformer_hits) / max(len(self._transformer_hits), 1)
        tft_acc = sum(self._tft_hits) / max(len(self._tft_hits), 1)
        
        total = t_acc + tft_acc
        if total < 1e-9:
            return 0.5, 0.5
        
        return t_acc / total, tft_acc / total
    
    def reset(self) -> None:
        """추적 데이터 초기화."""
        self._transformer_hits.clear()
        self._tft_hits.clear()
