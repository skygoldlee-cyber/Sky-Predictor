"""
피봇 확정 확률 계산기

과거 피봇 데이터와 기술적 조건을 기반으로 피봇 확정 확률을 계산합니다.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class HistoricalPivot:
    """과거 피봇 데이터."""
    idx: int
    price: float
    pivot_type: str  # "H" or "L"
    confirmed: bool
    confirmation_bars: int = 0
    price_deviation_pct: float = 0.0
    timestamp: Optional[pd.Timestamp] = None


class PivotProbabilityCalculator:
    """피봇 확정 확률 계산기 (통계 + 기술적 조건 조합)."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.historical_pivots: List[HistoricalPivot] = []
        self.stat_weight = 0.4  # 통계 기반 가중치
        self.tech_weight = 0.6  # 기술적 조건 가중치
    
    def add_pivot(self, pivot: HistoricalPivot) -> None:
        """피봇 데이터 추가."""
        self.historical_pivots.append(pivot)
        if len(self.historical_pivots) > self.max_history:
            self.historical_pivots.pop(0)
    
    def find_similar_pivots(self, candidate: HistoricalPivot, max_count: int = 50) -> List[HistoricalPivot]:
        """유사한 과거 피봇 찾기."""
        similar = []
        for p in self.historical_pivots:
            # 피봇 유형 일치
            if p.pivot_type != candidate.pivot_type:
                continue
            
            # 가격 범위 유사 (±5%)
            price_diff_pct = abs(p.price - candidate.price) / candidate.price * 100
            if price_diff_pct > 5.0:
                continue
            
            similar.append(p)
            if len(similar) >= max_count:
                break
        
        return similar
    
    def calculate_statistical_probability(self, candidate: HistoricalPivot) -> float:
        """과거 통계 기반 확률."""
        similar = self.find_similar_pivots(candidate)
        if not similar:
            return 0.5  # 데이터 부족 시 기본 확률
        
        confirmed_count = sum(1 for p in similar if p.confirmed)
        return confirmed_count / len(similar)
    
    def calculate_technical_probability(self, candidate: HistoricalPivot, 
                                      current_price: float, 
                                      confirmation_bars_required: int = 3) -> float:
        """기술적 조건 기반 확률."""
        probability = 0.5
        
        # 1. confirmation_bars 진행 정도 (최대 30% 가중)
        if confirmation_bars_required > 0:
            progress = min(candidate.confirmation_bars / confirmation_bars_required, 1.0)
            probability += progress * 0.3
        
        # 2. 가격이 피봇에서 벗어난 정도 (최대 40% 가중)
        price_deviation = abs(current_price - candidate.price) / candidate.price * 100
        # 가격이 피봇에서 멀어질수록 확정 확률 감소
        if price_deviation < 0.5:
            probability += 0.4  # 가격이 피봇 근처에 있음
        elif price_deviation < 1.0:
            probability += 0.2
        elif price_deviation < 2.0:
            probability += 0.1
        
        # 3. 확률 범위 제한
        return max(0.0, min(1.0, probability))
    
    def calculate_combined_probability(self, candidate: HistoricalPivot,
                                      current_price: float,
                                      confirmation_bars_required: int = 3) -> float:
        """조합 확률 계산."""
        stat_prob = self.calculate_statistical_probability(candidate)
        tech_prob = self.calculate_technical_probability(
            candidate, current_price, confirmation_bars_required
        )
        
        combined = (stat_prob * self.stat_weight) + (tech_prob * self.tech_weight)
        return max(0.0, min(1.0, combined))
