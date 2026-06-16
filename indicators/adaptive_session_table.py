"""Adaptive Session Table - 시간대별 파라미터 테이블 동적 학습
================================================================
실제 시장 데이터 기반 시간대별 파라미터 테이블 자동 학습 시스템.

기능
---------
- 시간대별 피봇 품질 통계 수집
- 품질 기반 파라미터 배율 동적 계산
- 학습된 테이블을 AdaptiveZigZagConfig에 적용
- 테이블 롤백 및 초기화 지원

사용 예시
---------
::

    from indicators import AdaptiveSessionTable, AdaptiveZigZagConfig

    base_config = AdaptiveZigZagConfig()
    session_table = AdaptiveSessionTable(base_config)
    
    # 매 봉마다 피봇 품질 업데이트
    session_table.update(bar_time, pivot_quality=0.75, pivot_count=1)
    
    # 학습된 테이블 가져오기
    learned_table = session_table.get_learned_table()
    
    # 설정에 적용
    base_config.session_min_wave_atr_ratio_table = learned_table
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 시간대별 통계 데이터 클래스
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HourlyStats:
    """시간대별 통계."""
    pivot_count: int = 0
    quality_sum: float = 0.0
    quality_avg: float = 0.0
    last_updated: Optional[str] = None


@dataclass
class SessionTableState:
    """세션 테이블 상태."""
    total_updates: int = 0
    is_enabled: bool = True
    min_samples_per_hour: int = 10  # 학습 최소 샘플 수
    learned_table: List[Tuple[str, str, float]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 클래스
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveSessionTable:
    """시간대별 파라미터 테이블 동적 학습 시스템.
    
    학습 로직:
    1. 시간대별 피봇 품질(0~1) 수집
    2. 품질이 낮은 시간대 → 배율 상향 (노이즈 차단)
    3. 품질이 높은 시간대 → 배율 하향 (민감도 회복)
    4. 최소 샘플 수 미만 시 기본값 사용
    
    배율 계산:
    multiplier = 1.0 + (0.5 - quality) * 0.5
    - quality = 1.0 → multiplier = 0.75 (민감도 회복)
    - quality = 0.5 → multiplier = 1.0 (기본)
    - quality = 0.0 → multiplier = 1.25 (노이즈 차단)
    """
    
    # 기본 시간대별 테이블 (초기값)
    DEFAULT_TABLE: List[Tuple[str, str, float]] = [
        ("09:00", "09:30", 0.8),   # 장 시작: 빠른 반응
        ("09:30", "10:30", 1.2),   # 오전: 안정적
        ("10:30", "13:00", 1.8),   # 점심: 노이즈 필터링
        ("13:00", "14:30", 1.2),   # 오후: 안정적
        ("14:30", "15:20", 0.8),   # 마감 전: 빠른 반응
        ("15:20", "15:30", 0.5),   # 마감: 최고 민감도
    ]
    
    def __init__(self, base_config: Any, now_fn: Optional[Callable[[], datetime]] = None) -> None:
        """세션 테이블 초기화.

        Args:
            base_config: AdaptiveZigZagConfig 인스턴스
            now_fn: 시간 함수 (테스트/백테스트용 주입 가능)
        """
        self._base_config = base_config
        self._state = SessionTableState()
        self._now_fn = now_fn if now_fn is not None else datetime.now

        # 시간대별 통계 (hour → HourlyStats)
        self._hourly_stats: Dict[int, HourlyStats] = defaultdict(
            lambda: HourlyStats()
        )

        logger.info("[AdaptiveSessionTable] 초기화 완료")
    
    def update(
        self,
        bar_time: Any,
        pivot_quality: float,
        pivot_count: int = 1,
    ) -> None:
        """시간대별 통계 업데이트.
        
        Args:
            bar_time: 봉 시각 (pandas Timestamp, "HH:MM:SS", "HH:MM" 등)
            pivot_quality: 피봇 품질 (0~1)
            pivot_count: 이번 업데이트의 피봇 수 (기본 1)
        """
        if not self._state.is_enabled:
            return
        
        hour = self._extract_hour(bar_time)
        if hour is None:
            return
        
        stats = self._hourly_stats[hour]
        stats.pivot_count += pivot_count
        stats.quality_sum += pivot_quality * pivot_count
        stats.quality_avg = stats.quality_sum / stats.pivot_count if stats.pivot_count > 0 else 0.0
        stats.last_updated = self._now_fn().strftime("%H:%M:%S")
        
        self._state.total_updates += 1
        
        logger.debug(
            "[AdaptiveSessionTable] hour=%02d quality=%.2f count=%d",
            hour, pivot_quality, stats.pivot_count
        )
    
    def get_learned_table(self) -> List[Tuple[str, str, float]]:
        """학습된 시간대별 테이블 반환.
        
        Returns:
            List[(start_time, end_time, multiplier)]
        """
        learned = []
        
        for hour in range(9, 16):  # 09:00 ~ 15:00
            stats = self._hourly_stats[hour]
            
            # 샘플 부족 시 기본값 사용
            if stats.pivot_count < self._state.min_samples_per_hour:
                # 기본 테이블에서 해당 시간대 찾기
                default_mult = self._get_default_multiplier(hour)
                multiplier = default_mult
            else:
                # 품질 기반 배율 계산
                multiplier = self._calc_multiplier_from_quality(stats.quality_avg)
            
            # 시간대 범위 설정
            start_time = f"{hour:02d}:00"
            end_time = f"{hour+1:02d}:00" if hour < 15 else "15:30"
            
            learned.append((start_time, end_time, multiplier))
        
        self._state.learned_table = learned
        return learned
    
    def _extract_hour(self, bar_time: Any) -> Optional[int]:
        """봉 시각에서 시간 추출.
        
        Returns:
            시간 (0-23), 실패 시 None
        """
        try:
            if bar_time is None:
                return None
            
            # pandas Timestamp
            if hasattr(bar_time, 'hour'):
                return bar_time.hour
            
            # 문자열 "HH:MM:SS" 또는 "HH:MM"
            s = str(bar_time).strip()
            if ':' in s:
                return int(s.split(':')[0])
            
            return None
        except Exception:
            return None
    
    def _get_default_multiplier(self, hour: int) -> float:
        """기본 테이블에서 해당 시간대의 배율 찾기."""
        for start, end, mult in self.DEFAULT_TABLE:
            start_h = int(start.split(':')[0])
            end_h = int(end.split(':')[0])
            if start_h <= hour < end_h:
                return mult
        return 1.0  # 기본값
    
    def _calc_multiplier_from_quality(self, quality: float) -> float:
        """품질 기반 배율 계산.
        
        multiplier = 1.0 + (0.5 - quality) * 0.5
        - quality = 1.0 → multiplier = 0.75 (민감도 회복)
        - quality = 0.5 → multiplier = 1.0 (기본)
        - quality = 0.0 → multiplier = 1.25 (노이즈 차단)
        """
        multiplier = 1.0 + (0.5 - quality) * 0.5
        return float(max(0.5, min(2.0, multiplier)))  # 0.5~2.0 범위 클램핑
    
    def reset(self) -> None:
        """통계 초기화."""
        self._hourly_stats.clear()
        self._state = SessionTableState()
        logger.info("[AdaptiveSessionTable] 통계 초기화")
    
    def enable(self) -> None:
        """학습 활성화."""
        self._state.is_enabled = True
        logger.info("[AdaptiveSessionTable] 활성화")
    
    def disable(self) -> None:
        """학습 비활성화."""
        self._state.is_enabled = False
        logger.info("[AdaptiveSessionTable] 비활성화")
    
    def apply_to_config(self, config: Any) -> None:
        """학습된 테이블을 설정에 적용.
        
        Args:
            config: AdaptiveZigZagConfig 인스턴스
        """
        learned = self.get_learned_table()
        
        if hasattr(config, 'session_min_wave_atr_ratio_table'):
            config.session_min_wave_atr_ratio_table = learned
            logger.info(
                "[AdaptiveSessionTable] 테이블 적용 완료 (%d 시간대)",
                len(learned)
            )
    
    @property
    def state(self) -> SessionTableState:
        """현재 상태."""
        return self._state
    
    @property
    def hourly_stats(self) -> Dict[int, HourlyStats]:
        """시간대별 통계."""
        return dict(self._hourly_stats)
    
    def get_summary(self) -> str:
        """학습 요약 문자열 반환."""
        lines = [
            f"총 업데이트: {self._state.total_updates}",
            f"활성화: {self._state.is_enabled}",
            f"최소 샘플: {self._state.min_samples_per_hour}",
            "시간대별 통계:"
        ]
        
        for hour in sorted(self._hourly_stats.keys()):
            stats = self._hourly_stats[hour]
            lines.append(
                f"  {hour:02d}:00 - count={stats.pivot_count}, "
                f"quality={stats.quality_avg:.2f}, "
                f"updated={stats.last_updated or 'N/A'}"
            )
        
        return "\n".join(lines)
