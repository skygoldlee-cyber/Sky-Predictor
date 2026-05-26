"""다중 시간프레임 지그재그 결합 모듈.

여러 시간프레임(1분봉, 5분봉, 15분봉)에서 독립적으로 피봇을 감지하고,
이를 결합하여 신뢰도 높은 피봇을 식별합니다.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

_logger = logging.getLogger(__name__)


class MultiTimeframeZigZag:
    """다중 시간프레임 피봇 확인 클래스.
    
    기존 AdaptiveZigZag에서 확정된 피봇에 대해
    다른 시간프레임에서도 피봇이 있는지 확인하여
    신뢰도를 평가합니다.
    """
    
    def __init__(
        self,
        scales: List[int] = [5, 15],
        consensus_threshold: int = 2,
        price_tolerance_pct: float = 1.0,
        index_tolerance_multiplier: float = 2.0
    ):
        """초기화.
        
        Args:
            scales: 확인할 상위 시간프레임 목록 (분 단위)
            consensus_threshold: 합의도 임계값 (이상일 때만 신호 통과)
            price_tolerance_pct: 가격 허용 오차 (%)
            index_tolerance_multiplier: 인덱스 허용 오차 배수 (시간프레임 × 배수)
        """
        self.scales = scales
        self.consensus_threshold = consensus_threshold
        self.price_tolerance_pct = price_tolerance_pct
        self.index_tolerance_multiplier = index_tolerance_multiplier
        
        # 각 시간프레임별 피봇 캐시
        self.pivot_cache: Dict[int, List[Dict[str, Any]]] = {}
        
        # 캐시 시그니처 (중복 업데이트 방지)
        self._cache_signatures: Dict[int, str] = {}
        
        # 성능 카운터
        self._check_count = 0
        self._consensus_match_count = 0  # [FIX] 합의도 매칭 성공 횟수
        
        _logger.info("[MultiTF] 초기화: scales=%s, threshold=%d, price_tol=%.2f%%, index_tol=%.1fx", 
                    scales, consensus_threshold, price_tolerance_pct, index_tolerance_multiplier)
    
    def update_pivot_cache(self, scale: int, pivots: List[Dict[str, Any]]):
        """시간프레임별 피봇 캐시 업데이트.
        
        Args:
            scale: 시간프레임 (분 단위)
            pivots: 피봇 목록
        """
        # 캐시 시그니처 계산 (중복 업데이트 방지)
        signature = self._calculate_pivot_signature(pivots)
        
        if self._cache_signatures.get(scale) == signature:
            _logger.debug("[MultiTF] %d분봉 피봇 캐시 무시 (동일 시그니처)", scale)
            return
        
        self.pivot_cache[scale] = pivots
        self._cache_signatures[scale] = signature
        _logger.debug("[MultiTF] %d분봉 피봇 캐시 업데이트: %d개", scale, len(pivots))
    
    def _calculate_pivot_signature(self, pivots: List[Dict[str, Any]]) -> str:
        """피봇 목록 시그니처 계산 (캐싱용).
        
        Args:
            pivots: 피봇 목록
            
        Returns:
            시그니처 문자열
        """
        if not pivots:
            return "empty"
        
        # [FIX] 피봇 수를 시그니처에 포함 (중간 피봇 변경 감지)
        last_pivot = pivots[-1]
        signature = f"{len(pivots)}_{last_pivot.get('index', 0)}_{last_pivot.get('price', 0)}"
        return signature
    
    def check_consensus(
        self,
        pivot_index: int,
        pivot_price: float,
        pivot_type: str,
        pivot_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """피봇에 대한 다중 시간프레임 합의도 확인.
        
        Args:
            pivot_index: 피봇 인덱스
            pivot_price: 피봇 가격
            pivot_type: 피봇 타입 ('H' or 'L')
            pivot_time: 피봇 시간 (선택적)
            
        Returns:
            {
                'consensus': int,  # 합의도 (일치하는 시간프레임 수)
                'total_scales': int,  # 전체 시간프레임 수
                'consensus_ratio': float,  # 합의도 비율
                'details': List[Dict],  # 각 시간프레임별 상세 정보
                'passed': bool  # 합의도 임계값 통과 여부
            }
        """
        self._check_count += 1
        consensus = 0
        details = []
        
        for scale in self.scales:
            scale_pivots = self.pivot_cache.get(scale, [])
            matched = self._find_matching_pivot(
                pivot_index, pivot_price, pivot_type, pivot_time, scale_pivots, scale
            )
            
            if matched:
                consensus += 1
                self._consensus_match_count += 1
                details.append({
                    'scale': scale,
                    'matched': True,
                    'pivot': matched
                })
            else:
                details.append({
                    'scale': scale,
                    'matched': False
                })
        
        total_scales = len(self.scales)
        consensus_ratio = consensus / total_scales if total_scales > 0 else 0
        passed = consensus >= self.consensus_threshold
        
        result = {
            'consensus': consensus,
            'total_scales': total_scales,
            'consensus_ratio': consensus_ratio,
            'details': details,
            'passed': passed
        }
        
        _logger.info(
            "[MultiTF] 피봇 확인: index=%d, price=%.2f, type=%s, consensus=%d/%d (%.1f%%), passed=%s",
            pivot_index, pivot_price, pivot_type, consensus, total_scales, consensus_ratio * 100, passed
        )
        
        return result
    
    def _find_matching_pivot(
        self,
        base_index: int,
        base_price: float,
        base_type: str,
        base_time: Optional[datetime],
        scale_pivots: List[Dict[str, Any]],
        scale: int
    ) -> Optional[Dict[str, Any]]:
        """기준 피봇과 일치하는 상위 시간프레임 피봇 찾기.
        
        Args:
            base_index: 기준 피봇 인덱스
            base_price: 기준 피봇 가격
            base_type: 기준 피봇 타입
            base_time: 기준 피봇 시간
            scale_pivots: 상위 시간프레임 피봇 목록
            scale: 상위 시간프레임
            
        Returns:
            일치하는 피봇 또는 None
        """
        for pivot in scale_pivots:
            # 피봇 타입 일치 확인
            if pivot.get('pivot_type') != base_type:
                continue
            
            # 가격 유사도 확인 (파라미터 사용)
            pivot_price = pivot.get('price', 0)
            if pivot_price <= 0:
                continue
                
            price_diff = abs(pivot_price - base_price) / base_price
            if price_diff > (self.price_tolerance_pct / 100.0):
                continue
            
            # 인덱스 범위 확인 (파라미터 사용)
            pivot_index = pivot.get('index', 0)
            index_diff = abs(pivot_index - base_index)
            max_index_diff = scale * self.index_tolerance_multiplier
            
            if index_diff > max_index_diff:
                continue
            
            _logger.debug(
                "[MultiTF] 피봇 매칭: base_idx=%d, scale_idx=%d, diff=%d (max=%d), price_diff=%.2f%%",
                base_index, pivot_index, index_diff, max_index_diff, price_diff * 100
            )
            
            return pivot
        
        return None
    
    def reset(self):
        """상태 초기화."""
        self.pivot_cache.clear()
        self._cache_signatures.clear()
        self._check_count = 0
        self._consensus_match_count = 0
        _logger.info("[MultiTF] 상태 초기화 완료")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환.
        
        Returns:
            {
                'check_count': 합의도 확인 총 횟수,
                'consensus_match_count': 합의도 매칭 성공 횟수,
                'consensus_rate': 합의도 성공률 (%)
            }
        """
        consensus_rate = (self._consensus_match_count / self._check_count * 100) if self._check_count > 0 else 0
        return {
            'check_count': self._check_count,
            'consensus_match_count': self._consensus_match_count,
            'consensus_rate': consensus_rate
        }
