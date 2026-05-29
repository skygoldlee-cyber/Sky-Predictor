"""Real-Time Parameter Tuner
================================
피봇 품질 메트릭 기반 실시간 파라미터 자동 튜닝 시스템.

기능
---------
- 취소율 기반 confirmation_bars 동적 조정
- 밀도 기반 pivot_threshold_min_pct 동적 조정
- 정확도 기반 use_atr_based_filtering 자동 활성화
- 튜닝 이력 추적 및 롤백 지원

사용 예시
---------
::

    from indicators import RealTimeParameterTuner, AdaptiveZigZagConfig

    tuner = RealTimeParameterTuner(base_config)
    
    # 매 봉마다 품질 메트릭 업데이트 후 튜닝
    metrics = quality_analyzer.compute(zz, cfg)
    tuned_config = tuner.tune(metrics)
    
    # 튜닝된 설정 적용
    zz.config = tuned_config
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 튜닝 이력 데이터 클래스
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TuningAction:
    """튜닝 액션 기록."""
    bar_idx: int
    metric_name: str
    old_value: float
    new_value: float
    reason: str


@dataclass
class TuningState:
    """튜너 상태."""
    total_tunings: int = 0
    last_tuning_bar: int = -1
    tuning_history: List[TuningAction] = field(default_factory=list)
    is_enabled: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# 메인 튜너 클래스
# ─────────────────────────────────────────────────────────────────────────────

class RealTimeParameterTuner:
    """피봇 품질 메트릭 기반 실시간 파라미터 튜너.
    
    튜닝 규칙:
    1. 취소율 > 40% → confirmation_bars +1
    2. 취소율 < 20% → confirmation_bars -1 (최소 1)
    3. 밀도 > 25/h → pivot_threshold_min_pct × 1.1
    4. 밀도 < 5/h → pivot_threshold_min_pct × 0.9
    5. 정확도 < 50% → use_atr_based_filtering = True
    """
    
    # 튜닝 임계값
    CANCEL_RATE_HIGH = 0.40
    CANCEL_RATE_LOW = 0.20
    DENSITY_HIGH = 25.0
    DENSITY_LOW = 5.0
    ACCURACY_LOW = 0.50
    
    # 파라미터 범위
    CONFIRMATION_BARS_MIN = 1
    CONFIRMATION_BARS_MAX = 5
    THRESHOLD_MIN_PCT_MIN = 0.1
    THRESHOLD_MIN_PCT_MAX = 5.0
    
    # 튜닝 간격 (봉 수) - 너무 잦은 튜닝 방지
    MIN_TUNING_INTERVAL = 30
    
    def __init__(self, base_config: Any) -> None:
        """튜너 초기화.
        
        Args:
            base_config: AdaptiveZigZagConfig 인스턴스 (복사본 사용)
        """
        self._base_config = self._deep_copy_config(base_config)
        self._state = TuningState()
        self._performance_history = deque(maxlen=100)
        
        logger.info("[RealTimeParameterTuner] 초기화 완료")
    
    def _deep_copy_config(self, config: Any) -> Any:
        """설정 깊은 복사."""
        try:
            import copy
            return copy.deepcopy(config)
        except Exception:
            # deepcopy 실패 시 얕은 복사
            return config
    
    def tune(self, metrics: Any, bar_idx: int) -> Any:
        """품질 메트릭 기반 파라미터 튜닝.
        
        Args:
            metrics: PivotQualityMetrics 인스턴스
            bar_idx: 현재 봉 인덱스
            
        Returns:
            튜닝된 설정 (AdaptiveZigZagConfig)
        """
        if not self._state.is_enabled:
            return self._base_config
        
        # 튜닝 간격 체크
        if bar_idx - self._state.last_tuning_bar < self.MIN_TUNING_INTERVAL:
            return self._base_config
        
        # 튜닝 수행
        tuned = self._deep_copy_config(self._base_config)
        actions = []
        
        # 1. 취소율 기반 confirmation_bars 튜닝
        if hasattr(metrics, 'cancel_rate'):
            if metrics.cancel_rate > self.CANCEL_RATE_HIGH:
                old_val = getattr(tuned, 'confirmation_bars', 2)
                new_val = min(old_val + 1, self.CONFIRMATION_BARS_MAX)
                if new_val != old_val:
                    setattr(tuned, 'confirmation_bars', new_val)
                    actions.append(TuningAction(
                        bar_idx=bar_idx,
                        metric_name='cancel_rate',
                        old_value=float(old_val),
                        new_value=float(new_val),
                        reason=f'취소율 {metrics.cancel_rate:.0%} > {self.CANCEL_RATE_HIGH:.0%}'
                    ))
            
            elif metrics.cancel_rate < self.CANCEL_RATE_LOW:
                old_val = getattr(tuned, 'confirmation_bars', 2)
                new_val = max(old_val - 1, self.CONFIRMATION_BARS_MIN)
                if new_val != old_val:
                    setattr(tuned, 'confirmation_bars', new_val)
                    actions.append(TuningAction(
                        bar_idx=bar_idx,
                        metric_name='cancel_rate',
                        old_value=float(old_val),
                        new_value=float(new_val),
                        reason=f'취소율 {metrics.cancel_rate:.0%} < {self.CANCEL_RATE_LOW:.0%}'
                    ))
        
        # 2. 밀도 기반 pivot_threshold_min_pct 튜닝
        if hasattr(metrics, 'pivots_per_hour'):
            if metrics.pivots_per_hour > self.DENSITY_HIGH:
                old_val = getattr(tuned, 'pivot_threshold_min_pct', 0.3)
                new_val = min(old_val * 1.1, self.THRESHOLD_MIN_PCT_MAX)
                if abs(new_val - old_val) > 0.01:
                    setattr(tuned, 'pivot_threshold_min_pct', new_val)
                    actions.append(TuningAction(
                        bar_idx=bar_idx,
                        metric_name='pivots_per_hour',
                        old_value=old_val,
                        new_value=new_val,
                        reason=f'밀도 {metrics.pivots_per_hour:.0f}/h > {self.DENSITY_HIGH:.0f}'
                    ))
            
            elif metrics.pivots_per_hour < self.DENSITY_LOW and metrics.total_confirmed > 5:
                old_val = getattr(tuned, 'pivot_threshold_min_pct', 0.3)
                new_val = max(old_val * 0.9, self.THRESHOLD_MIN_PCT_MIN)
                if abs(new_val - old_val) > 0.01:
                    setattr(tuned, 'pivot_threshold_min_pct', new_val)
                    actions.append(TuningAction(
                        bar_idx=bar_idx,
                        metric_name='pivots_per_hour',
                        old_value=old_val,
                        new_value=new_val,
                        reason=f'밀도 {metrics.pivots_per_hour:.1f}/h < {self.DENSITY_LOW:.0f}'
                    ))
        
        # 3. 정확도 기반 use_atr_based_filtering 튜닝
        if hasattr(metrics, 'accuracy_score') and hasattr(metrics, 'accuracy_sample'):
            if metrics.accuracy_sample >= 5 and metrics.accuracy_score < self.ACCURACY_LOW:
                old_val = getattr(tuned, 'use_atr_based_filtering', False)
                if not old_val:
                    setattr(tuned, 'use_atr_based_filtering', True)
                    actions.append(TuningAction(
                        bar_idx=bar_idx,
                        metric_name='accuracy_score',
                        old_value=0.0,
                        new_value=1.0,
                        reason=f'정확도 {metrics.accuracy_score:.0%} < {self.ACCURACY_LOW:.0%}'
                    ))
        
        # 튜닝 기록
        if actions:
            self._state.total_tunings += len(actions)
            self._state.last_tuning_bar = bar_idx
            self._state.tuning_history.extend(actions)
            
            # 기록 제한
            if len(self._state.tuning_history) > 50:
                self._state.tuning_history = self._state.tuning_history[-50:]
            
            # 기본 설정 업데이트
            self._base_config = tuned
            
            logger.info(
                "[RealTimeParameterTuner] 튜닝 %d건 수행 (bar=%d)",
                len(actions), bar_idx
            )
        
        return tuned
    
    def reset(self) -> None:
        """튜너 상태 초기화."""
        self._state = TuningState()
        self._performance_history.clear()
        logger.info("[RealTimeParameterTuner] 상태 초기화")
    
    def enable(self) -> None:
        """튜너 활성화."""
        self._state.is_enabled = True
        logger.info("[RealTimeParameterTuner] 활성화")
    
    def disable(self) -> None:
        """튜너 비활성화."""
        self._state.is_enabled = False
        logger.info("[RealTimeParameterTuner] 비활성화")
    
    def rollback(self, steps: int = 1) -> Optional[Any]:
        """튜닝 롤백.
        
        Args:
            steps: 롤백할 튜닝 액션 수
            
        Returns:
            롤백된 설정 (롤백 불가 시 None)
        """
        if not self._state.tuning_history:
            logger.warning("[RealTimeParameterTuner] 롤백할 이력 없음")
            return None
        
        if steps > len(self._state.tuning_history):
            steps = len(self._state.tuning_history)
        
        # 역순으로 롤백
        for action in reversed(self._state.tuning_history[-steps:]):
            if action.metric_name == 'cancel_rate':
                setattr(self._base_config, 'confirmation_bars', int(action.old_value))
            elif action.metric_name == 'pivots_per_hour':
                setattr(self._base_config, 'pivot_threshold_min_pct', action.old_value)
            elif action.metric_name == 'accuracy_score':
                setattr(self._base_config, 'use_atr_based_filtering', bool(action.old_value == 1.0))
        
        # 이력 제거
        self._state.tuning_history = self._state.tuning_history[:-steps]
        
        logger.info("[RealTimeParameterTuner] %d단계 롤백 완료", steps)
        return self._base_config
    
    @property
    def state(self) -> TuningState:
        """현재 튜너 상태."""
        return self._state
    
    @property
    def config(self) -> Any:
        """현재 설정."""
        return self._base_config
    
    def get_tuning_summary(self) -> str:
        """튜닝 요약 문자열 반환."""
        if not self._state.tuning_history:
            return "튜닝 이력 없음"
        
        summary_lines = [
            f"총 튜닝: {self._state.total_tunings}건",
            f"마지막 튜닝: bar {self._state.last_tuning_bar}",
            "최근 튜닝 이력:"
        ]
        
        for action in self._state.tuning_history[-5:]:
            summary_lines.append(
                f"  bar {action.bar_idx}: {action.metric_name} "
                f"{action.old_value:.3f} → {action.new_value:.3f} "
                f"({action.reason})"
            )
        
        return "\n".join(summary_lines)
