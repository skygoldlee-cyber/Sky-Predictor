"""캐시 관리 모듈"""

import logging
import time
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class CacheManager:
    """데이터 캐시 관리자."""
    
    def __init__(self, cache_ttl: float = 5.0):
        """
        Args:
            cache_ttl: 캐시 유효 시간 (초)
        """
        self._cache: Dict[str, Tuple[Any, Any, float]] = {}
        self._cache_ttl = cache_ttl
    
    def get_cache_key(self, selected_plot: str, minutes: int) -> str:
        """캐시 키 생성.
        
        Args:
            selected_plot: 선택된 플롯 타입 ("futures" 또는 "kospi")
            minutes: 가져올 분봉 수
            
        Returns:
            캐시 키
        """
        return f"{selected_plot}_{minutes}"
    
    def is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 검사.
        
        Args:
            cache_key: 캐시 키
            
        Returns:
            캐시가 유효하면 True
        """
        if cache_key not in self._cache:
            return False
        _, _, timestamp = self._cache[cache_key]
        return (time.monotonic() - timestamp) < self._cache_ttl
    
    def get(self, cache_key: str) -> Optional[Tuple[Any, Any]]:
        """캐시에서 데이터 가져오기.
        
        Args:
            cache_key: 캐시 키
            
        Returns:
            (df, pm) 튜플 또는 None
        """
        if cache_key not in self._cache:
            return None
        df, pm, _ = self._cache[cache_key]
        return df, pm
    
    def set(self, cache_key: str, df: Any, pm: Any) -> None:
        """캐시에 데이터 저장.
        
        Args:
            cache_key: 캐시 키
            df: 데이터프레임
            pm: 피봇 메타데이터
        """
        self._cache[cache_key] = (df, pm, time.monotonic())
        logger.debug("[CacheManager] 캐시 저장: key=%s", cache_key)
    
    def clear(self) -> None:
        """캐시 전체 삭제."""
        self._cache.clear()
        logger.debug("[CacheManager] 캐시 삭제")
    
    def set_ttl(self, ttl: float) -> None:
        """캐시 유효 시간 설정.
        
        Args:
            ttl: 캐시 유효 시간 (초)
        """
        self._cache_ttl = ttl
