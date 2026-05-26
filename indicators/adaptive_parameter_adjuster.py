"""adaptive_parameter_adjuster.py — 장상황 기반 ZigZag 파라미터 동적 조정
==================================================================

장상황(변동성, 거래량, 추세, 시간대, 원웨이 추세)에 따라 ZigZag 파라미터를 동적으로 조정하여
피봇 지연시간을 최소화하고 정확도를 높이는 시스템.

지원 전략:
- VolatilityStrategy: 변동성 기반 조정
- TrendStrategy: 추세 기반 조정
- VolumeStrategy: 거래량 기반 조정
- TimeStrategy: 시간대 기반 조정
- OneWayStrategy: 원웨이 추세 무력화 방지 조정
"""

from dataclasses import dataclass
from collections import deque
from typing import Optional, Dict, Any, List, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AdjustmentStrategy(ABC):
    """조정 전략 추상 클래스 (확장성 확보)"""
    name: str = "base"  # 전략 식별자 (인덱스 하드코딩 방지)
    
    @abstractmethod
    def adjust(self, df: pd.DataFrame, base_params: dict, indicators: Optional[dict] = None,
              current_time: Optional[datetime] = None) -> dict:
        """파라미터 조정
        
        Args:
            df: OHLCV 데이터
            base_params: 기준 파라미터
            indicators: 미리 계산된 공통 지표 (선택 사항)
            current_time: 현재 시간 (선택 사항)
            
        Returns:
            조정된 파라미터
        """
        pass


class VolatilityStrategy(AdjustmentStrategy):
    """변동성 기반 조정 전략"""
    name = "volatility"
    
    def __init__(self, adjuster: 'AdaptiveParameterAdjuster'):
        self.adjuster = adjuster
    
    def adjust(self, df: pd.DataFrame, base_params: dict, indicators: Optional[dict] = None,
              current_time: Optional[datetime] = None) -> dict:
        return self.adjuster.adjust_for_volatility(df, indicators)


class TrendStrategy(AdjustmentStrategy):
    """추세 기반 조정 전략"""
    name = "trend"
    
    def __init__(self, adjuster: 'AdaptiveParameterAdjuster'):
        self.adjuster = adjuster
    
    def adjust(self, df: pd.DataFrame, base_params: dict, indicators: Optional[dict] = None,
              current_time: Optional[datetime] = None) -> dict:
        return self.adjuster.adjust_for_trend(df, indicators)


class VolumeStrategy(AdjustmentStrategy):
    """거래량 기반 조정 전략"""
    name = "volume"
    
    def __init__(self, adjuster: 'AdaptiveParameterAdjuster'):
        self.adjuster = adjuster
    
    def adjust(self, df: pd.DataFrame, base_params: dict, indicators: Optional[dict] = None,
              current_time: Optional[datetime] = None) -> dict:
        return self.adjuster.adjust_for_volume(df, indicators)


class TimeStrategy(AdjustmentStrategy):
    """시간대 기반 조정 전략"""
    name = "time"
    
    def __init__(self, adjuster: 'AdaptiveParameterAdjuster'):
        self.adjuster = adjuster
    
    def adjust(self, df: pd.DataFrame, base_params: dict, indicators: Optional[dict] = None,
              current_time: Optional[datetime] = None) -> dict:
        return self.adjuster.adjust_for_time_of_day(current_time)


class OneWayStrategy(AdjustmentStrategy):
    """원웨이 추세 무력화 방지 전략"""
    name = "oneway"
    
    def __init__(self, adjuster: 'AdaptiveParameterAdjuster'):
        self.adjuster = adjuster
    
    def adjust(self, df: pd.DataFrame, base_params: dict, indicators: Optional[dict] = None,
              current_time: Optional[datetime] = None) -> dict:
        return self.adjuster.adjust_for_oneway(df, indicators)


@dataclass
class AdaptiveParams:
    """동적 조정된 파라미터"""
    atr_multiplier: float = 0.5
    pivot_threshold_min_pct: float = 0.01
    confirmation_bars: int = 2
    min_wave_atr_ratio: float = 1.0
    
    def __post_init__(self):
        """파라미터 유효성 검사
        
        물리적 불가능 값만 차단 (> 0)
        실제 운영 범위는 PARAM_RANGES에서 클램핑
        """
        if self.atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be > 0, got {self.atr_multiplier}")
        if self.pivot_threshold_min_pct <= 0:
            raise ValueError(f"pivot_threshold_min_pct must be > 0, got {self.pivot_threshold_min_pct}")
        # PARAM_RANGES['confirmation_bars'] = (1, 5)와 통일
        if not (1 <= self.confirmation_bars <= 5):
            raise ValueError(f"confirmation_bars out of range [1, 5]: {self.confirmation_bars}")
        if self.min_wave_atr_ratio <= 0:
            raise ValueError(f"min_wave_atr_ratio must be > 0, got {self.min_wave_atr_ratio}")


class AdaptiveParameterAdjuster:
    """장상황에 따라 ZigZag 파라미터 동적 조정"""
    
    # 심볼별 프로필 (클래스 변수, 메모리 절약)
    SYMBOL_PROFILES = {
        "futures": {"vol_weight": 0.50, "base_atr_mult": 0.8, "base_confirm": 2},
        "kospi": {"vol_weight": 0.35, "base_atr_mult": 0.5, "base_confirm": 3}
    }
    
    # 파라미터 허용 범위
    # KP200 선물 config 기준 상향 조정 (pivot_threshold_min_pct=0.5%, min_wave_atr_ratio=2.0~4.0)
    # NEWS_EVENT, OPENING_EVENT 레짐의 atr_multiplier 5~8을 수용하기 위해 상한을 8.0으로 확장
    PARAM_RANGES = {
        'atr_multiplier': (0.1, 8.0),  # 상한 3.0 → 8.0 (NEWS/OPENING 레짐 수용)
        'pivot_threshold_min_pct': (0.10, 0.60),  # 상한 0.3% → 0.6% (KP200 config 0.5% 수용)
        'confirmation_bars': (1, 5),
        'min_wave_atr_ratio': (1.0, 5.0)   # 하한 0.3→1.0, 상한 3.5→5.0 (KP200 session table 2.0~4.0 수용)
    }
    
    # 조정 폭 제한 (최대 ±30%)
    MAX_ADJUSTMENT_RATIO = 0.3
    
    def __init__(self, base_params: Optional[AdaptiveParams] = None, config: Optional[Dict[str, Any]] = None, symbol: str = "futures"):
        """초기화
        
        Args:
            base_params: 기준 파라미터 (None이면 config에서 읽음)
            config: config.json 딕셔너리 (None이면 기본값 사용)
            symbol: 심볼 ("futures" 또는 "kospi")
        """
        self._symbol = symbol
        
        # [SSOT] config에서 기본 파라미터 읽기 — zigzag_settings_from_dict 경유
        if base_params is None and config is not None:
            try:
                from config import zigzag_settings_from_dict as _zz_from_dict
                adaptive_cfg = config.get("adaptive_indicator", {})
                zz_base = _zz_from_dict(adaptive_cfg.get("zigzag") or {})
                if symbol == "kospi":
                    zz_s = _zz_from_dict(adaptive_cfg.get("kospi_zigzag") or {}, base=zz_base)
                else:
                    zz_s = _zz_from_dict(adaptive_cfg.get("futures_zigzag") or {}, base=zz_base)

                base_params = AdaptiveParams(
                    atr_multiplier=zz_s.atr_multiplier,
                    pivot_threshold_min_pct=zz_s.pivot_threshold_min_pct,
                    confirmation_bars=zz_s.confirmation_bars,
                    min_wave_atr_ratio=zz_s.min_wave_atr_ratio,
                )
                logger.info("[AdaptiveParameterAdjuster] config에서 초기값 로드 (SSOT): symbol=%s, params=%s", symbol, base_params)
            except Exception as e:
                logger.warning("[AdaptiveParameterAdjuster] config 로드 실패: %s, 기본값 사용", e)
        
        # 심볼별 프로필 (설정 기반 최적화, 클래스 변수로 메모리 절약)
        self.profile = self.SYMBOL_PROFILES.get(symbol, self.SYMBOL_PROFILES["futures"])
        
        # 기준 파라미터 설정 (프로필 기반 기본값 적용)
        if base_params is None:
            self.base_params = AdaptiveParams(
                atr_multiplier=self.profile["base_atr_mult"],
                confirmation_bars=self.profile["base_confirm"]
            )
        else:
            self.base_params = base_params
        
        # 최근 N봉 ATR 통계 (변동성 측정)
        self.atr_window = 20
        self.atr_history = deque(maxlen=self.atr_window)
        self.atr_ma: Optional[float] = None
        self.atr_std: Optional[float] = None
        
        # 피드백 루프 상수 (외부 조정 가능)
        self.MAX_LAG_SECONDS = 5.0
        self.FEEDBACK_THRESHOLD_AGGRESSIVE = 0.6
        self.FEEDBACK_THRESHOLD_CONSERVATIVE = 0.3
        
        # ATR 계산 캐싱 (성능 최적화, 메모리 누수 방지를 위해 최신값만 유지)
        self._last_atr: Optional[float] = None
        self._last_df_len = 0
        
        # 조정 이력 모니터링
        self.adjustment_history = deque(maxlen=10)
        
        # 플러그인 전략 (확장성 확보)
        self.strategies: List[AdjustmentStrategy] = []
        self._init_strategies()
        
        # constants 모듈 임포트 시도 (초기화 시 1회)
        self._is_market_closed = None
        try:
            from config import is_market_closed
            self._is_market_closed = is_market_closed
        except (ImportError, AttributeError) as e:
            logger.warning("[AdaptiveParameterAdjuster] config 모듈 로드 실패: %s", e)
        
        logger.info("[AdaptiveParameterAdjuster] 초기화 완료 (base_params=%s, symbol=%s)", self.base_params, self._symbol)
    
    def _init_strategies(self) -> None:
        """조정 전략 초기화 (플러그인 구조)"""
        self.strategies = [
            VolatilityStrategy(self),
            TrendStrategy(self),
            VolumeStrategy(self),
            TimeStrategy(self),
            OneWayStrategy(self)
        ]
        logger.debug("[AdaptiveParameterAdjuster] 전략 초기화 완료: %d개 전략", len(self.strategies))
    
    def _collect_via_methods(self, df: pd.DataFrame, indicators: dict,
                            current_time: Optional[datetime]) -> dict:
        """기본 메서드로 파라미터 수집 (SRP 준수)
        
        Args:
            df: OHLCV 데이터
            indicators: 공통 지표
            current_time: 현재 시간
            
        Returns:
            이름 기반 파라미터 딕셔너리
        """
        return {
            'volatility': self.adjust_for_volatility(df, indicators),
            'trend': self.adjust_for_trend(df, indicators),
            'volume': self.adjust_for_volume(df, indicators),
            'time': self.adjust_for_time_of_day(current_time),
            'oneway': self.adjust_for_oneway(df, indicators)
        }
    
    def _collect_adjusted_params(self, df: pd.DataFrame, base: dict, indicators: dict,
                                current_time: Optional[datetime], use_strategies: bool) -> dict:
        """조정 결과를 이름 기반 딕셔너리로 수집
        
        Args:
            df: OHLCV 데이터
            base: 기준 파라미터
            indicators: 공통 지표
            current_time: 현재 시간
            use_strategies: 플러그인 전략 사용 여부
            
        Returns:
            이름 기반 파라미터 딕셔너리
        """
        # 전략 사용 안 하거나 전략 리스트 비어있으면 기본 메서드 사용
        if not use_strategies or not self.strategies:
            if use_strategies and not self.strategies:
                logger.warning("[AdaptiveParameterAdjuster] 전략 리스트 비어있음, 기본 메서드로 폴백")
            return self._collect_via_methods(df, indicators, current_time)
        
        # 전략 사용
        results = {}
        for strategy in self.strategies:
            try:
                results[strategy.name] = strategy.adjust(df, base, indicators, current_time)
            except Exception as e:
                logger.warning("[AdaptiveParameterAdjuster] 전략 실행 실패: %s", e)
                results[strategy.name] = base
        return results
    
    def _apply_weights(self, param_map: dict, weights: dict, base: dict) -> dict:
        """가중치 적용
        
        Args:
            param_map: 이름 기반 파라미터 딕셔너리
            weights: 가중치 딕셔너리
            base: 기준 파라미터
            
        Returns:
            가중 평균된 파라미터
        """
        # 1. 기본 전략들 가중 평균
        weighted_params = self._weighted_average([
            (param_map.get('volatility', base), weights.get('volatility', 0.25)),
            (param_map.get('trend', base), weights.get('trend', 0.25)),
            (param_map.get('volume', base), weights.get('volume', 0.25)),
            (param_map.get('time', base), weights.get('time', 0.25))
        ])
        
        # 2. 원웨이 상황이면 'oneway' 파라미터를 강하게 결합 (50% 비중으로 믹스)
        # [FIX-ONEWAY-CMP] dict equality는 float 오차로 오판 가능 → 실제 값 차이로 판단
        oneway_params = param_map.get('oneway', base)
        oneway_differs = any(
            abs(float(oneway_params.get(k, 0)) - float(base.get(k, 0))) > 1e-6
            for k in base
        )
        if oneway_differs:
            weighted_params = self._weighted_average([
                (weighted_params, 0.5),
                (oneway_params, 0.5)
            ])
            logger.debug("[AdaptiveParameterAdjuster] 원웨이 파라미터 50% 결합 적용")
        
        return weighted_params
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """ATR 계산
        
        Args:
            df: OHLCV 데이터
            period: ATR 기간
            
        Returns:
            ATR 값
        """
        if len(df) < period + 1:
            return 0.0
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(period).mean().iloc[-1]
        
        return float(atr) if not np.isnan(atr) else 0.0
    
    def update_atr_stats(self, df: pd.DataFrame, force: bool = False) -> None:
        """ATR 통계 업데이트 (캐싱 적용, 메모리 누수 방지)
        
        Args:
            df: OHLCV 데이터
            force: 길이가 같아도 강제 갱신 (실시간 봉 갱신 대응)
        """
        current_len = len(df)
        # 길이가 줄어든 경우 (데이터 재로드) → 캐시 리셋
        if current_len < self._last_df_len:
            logger.warning("[AdaptiveParameterAdjuster] df 길이 감소 감지, 캐시 리셋: %d→%d",
                           self._last_df_len, current_len)
            self._last_df_len = 0
        
        # 데이터 길이가 변경된 경우 또는 강제 갱신 시 ATR 재계산
        if current_len > self._last_df_len:
            # 새 봉 확정 시 정상 추가
            atr = self._calculate_atr(df)
            self._last_atr = atr  # 최신값만 유지 (메모리 누수 방지)
            # ATR=0인 경우(데이터 부족) history에 추가하지 않음 (편향 방지)
            if atr > 0:
                self.atr_history.append(atr)
            self._last_df_len = current_len
            
            # ATR 통계 업데이트
            if len(self.atr_history) >= 10:
                self.atr_ma = float(np.mean(self.atr_history))
                self.atr_std = float(np.std(self.atr_history))
                logger.debug("[AdaptiveParameterAdjuster] ATR 통계 업데이트: MA=%.4f, STD=%.4f",
                            self.atr_ma, self.atr_std)
        elif force:
            # force=True이더라도 새 봉이 확정된 경우(current_len > _last_df_len)는
            # 정상 경로(if 블록)에서 처리되므로 elif는 같은 봉 갱신 시에만 실행됨
            atr = self._calculate_atr(df)
            self._last_atr = atr
            # 정상 경로와 동일하게 ATR=0 필터링 (편향 방지)
            if atr > 0:
                if self.atr_history:
                    self.atr_history[-1] = atr  # 마지막 값 교체
                else:
                    self.atr_history.append(atr)
            # _last_df_len은 변경하지 않음 (같은 봉 갱신임)
            
            # force 시에도 ATR 통계 업데이트
            if len(self.atr_history) >= 10:
                self.atr_ma = float(np.mean(self.atr_history))
                self.atr_std = float(np.std(self.atr_history))
                logger.debug("[AdaptiveParameterAdjuster] ATR 통계 업데이트 (force): MA=%.4f, STD=%.4f", 
                            self.atr_ma, self.atr_std)
    
    def _compute_common_indicators(self, df: pd.DataFrame) -> dict:
        """공통 지표 미리 계산 (DRY 원칙, 성능 최적화)
        
        Args:
            df: OHLCV 데이터
            
        Returns:
            공통 지표 딕셔너리
        """
        indicators = {}
        
        # 이동평균 (NaN 전파 방지)
        if len(df) >= 30:
            ma_short = df['close'].rolling(10).mean().iloc[-1]
            ma_long = df['close'].rolling(30).mean().iloc[-1]
            
            if pd.isna(ma_short) or pd.isna(ma_long):
                # NaN이면 이동평균 지표 제외
                pass
            else:
                indicators['ma_short'] = float(ma_short)
                indicators['ma_long'] = float(ma_long)
        
        # 거래량 (NaN 전파 방지, 독립 처리)
        if len(df) >= 20:
            vol_avg = df['volume'].rolling(20).mean().iloc[-1]
            vol_current = df['volume'].iloc[-1]
            if not pd.isna(vol_avg):
                indicators['vol_avg'] = float(vol_avg)
            if not pd.isna(vol_current):
                indicators['vol_current'] = float(vol_current)
        
        # 추세 강도
        if 'ma_short' in indicators and 'ma_long' in indicators:
            ma_long = indicators['ma_long']
            if ma_long != 0:
                indicators['trend_strength'] = float(abs(indicators['ma_short'] - ma_long) / ma_long)
        
        return indicators
    
    def _get_base_params(self) -> dict:
        """기준 파라미터 반환
        
        Returns:
            기준 파라미터 딕셔너리
        """
        return {
            'atr_multiplier': self.base_params.atr_multiplier,
            'pivot_threshold_min_pct': self.base_params.pivot_threshold_min_pct,
            'confirmation_bars': self.base_params.confirmation_bars,
            'min_wave_atr_ratio': self.base_params.min_wave_atr_ratio
        }
    
    def _clamp_params(self, params: dict) -> dict:
        """파라미터 클램핑 (허용 범위 내로 제한)
        
        Args:
            params: 조정된 파라미터
            
        Returns:
            클램핑된 파라미터
        """
        clamped = params.copy()
        
        clamped['atr_multiplier'] = float(np.clip(
            params['atr_multiplier'],
            self.PARAM_RANGES['atr_multiplier'][0],
            self.PARAM_RANGES['atr_multiplier'][1]
        ))
        clamped['pivot_threshold_min_pct'] = float(np.clip(
            params['pivot_threshold_min_pct'],
            self.PARAM_RANGES['pivot_threshold_min_pct'][0],
            self.PARAM_RANGES['pivot_threshold_min_pct'][1]
        ))
        # confirmation_bars는 최종적으로 int로 반환 (타입 일관성)
        clamped['confirmation_bars'] = int(round(np.clip(
            params['confirmation_bars'],
            self.PARAM_RANGES['confirmation_bars'][0],
            self.PARAM_RANGES['confirmation_bars'][1]
        )))
        clamped['min_wave_atr_ratio'] = float(np.clip(
            params['min_wave_atr_ratio'],
            self.PARAM_RANGES['min_wave_atr_ratio'][0],
            self.PARAM_RANGES['min_wave_atr_ratio'][1]
        ))
        
        return clamped
    
    def _limit_adjustment(self, base: float, adjusted: float) -> float:
        """조정 폭 제한 (최대 ±30%)
        
        Args:
            base: 기준값
            adjusted: 조정된 값
            
        Returns:
            조정 폭이 제한된 값
        """
        if base == 0:
            # base가 0이면 조정 폭 제한 불가, 그대로 반환
            return adjusted
        max_change = abs(base) * self.MAX_ADJUSTMENT_RATIO
        if abs(adjusted - base) > max_change:
            return base + max_change if adjusted > base else base - max_change
        return adjusted
    
    def _calc_atr_percentile(self) -> float:
        """당일 ATR 누적 분포에서 현재 ATR의 백분위 (0.0~1.0).
        
        [Layer B] 절대값 임계값 없이 상대적 변동성 판단.
        ATR이 3pt든 0.5pt든 항상 "오늘 기준 상위 25%인지"로 판단.
        """
        if len(self.atr_history) < 10:
            return 0.5  # warm-up 미완료 → 중간값
        arr = np.array(list(self.atr_history))
        current = arr[-1]
        return float(np.mean(arr < current))  # 현재값보다 낮은 비율
    
    def get_vol_ratio(self) -> float:
        """Layer B용: ATR 백분위 기반 변동성 비율 반환.
        
        Returns:
            0.85 (저변동, 하위 25%), 1.0 (중간), 1.25 (고변동, 상위 25%)
        """
        pct = self._calc_atr_percentile()
        if pct >= 0.75:
            return 1.25
        elif pct <= 0.25:
            return 0.85
        return 1.0
    
    def adjust_for_volatility(self, df: pd.DataFrame, indicators: Optional[dict] = None) -> dict:
        """[Layer B] ATR 백분위 기반 파라미터 조정
        
        절대값 임계값(vol_high_threshold) 없이 상대적 변동성 판단.
        당일 ATR 분포 기준 백분위로 조정.
        
        Args:
            df: OHLCV 데이터
            indicators: 미리 계산된 공통 지표 (확장성, 현재 미사용)
            
        Returns:
            조정된 파라미터
        """
        pct = self._calc_atr_percentile()
        adjusted = self._get_base_params()
        
        if pct >= 0.75:       # 상위 25% → 고변동
            adjusted['atr_multiplier'] = self._limit_adjustment(
                adjusted['atr_multiplier'],
                adjusted['atr_multiplier'] * 1.25
            )
            adjusted['confirmation_bars'] = max(1, adjusted['confirmation_bars'] - 1)
            logger.debug("[AdaptiveParameterAdjuster] 고변동성 조정: pct=%.2f", pct)
            
        elif pct <= 0.25:     # 하위 25% → 저변동
            adjusted['atr_multiplier'] = self._limit_adjustment(
                adjusted['atr_multiplier'],
                adjusted['atr_multiplier'] * 0.85
            )
            adjusted['confirmation_bars'] = min(4, adjusted['confirmation_bars'] + 1)
            logger.debug("[AdaptiveParameterAdjuster] 저변동성 조정: pct=%.2f", pct)
        # 나머지(중간 50%)는 base 그대로
        
        return self._clamp_params(adjusted)
    
    def adjust_for_volume(self, df: pd.DataFrame, indicators: Optional[dict] = None) -> dict:
        """거래량 기반 파라미터 조정
        
        Args:
            df: OHLCV 데이터
            indicators: 미리 계산된 공통 지표 (성능 최적화)
            
        Returns:
            조정된 파라미터
        """
        if indicators is None:
            indicators = self._compute_common_indicators(df)
        
        if 'vol_avg' not in indicators or 'vol_current' not in indicators:
            return self._get_base_params()
        
        vol_avg = indicators['vol_avg']
        vol_current = indicators['vol_current']
        
        if vol_avg <= 0 or pd.isna(vol_avg) or pd.isna(vol_current):
            return self._get_base_params()
        
        vol_ratio = vol_current / vol_avg
        
        # vol_ratio 비정상값 방어 (inf, NaN)
        if not np.isfinite(vol_ratio):
            logger.debug("[AdaptiveParameterAdjuster] vol_ratio 비정상값 감지: %s", vol_ratio)
            return self._get_base_params()
        
        adjusted = self._get_base_params()
        
        if vol_ratio > 2.0:  # 거래량 급증
            adjusted['min_wave_atr_ratio'] = self._limit_adjustment(
                adjusted['min_wave_atr_ratio'],
                adjusted['min_wave_atr_ratio'] * 1.5
            )
            adjusted['pivot_threshold_min_pct'] = self._limit_adjustment(
                adjusted['pivot_threshold_min_pct'],
                adjusted['pivot_threshold_min_pct'] * 0.8
            )
            logger.debug("[AdaptiveParameterAdjuster] 거래량 급증 조정: vol_ratio=%.2f", vol_ratio)
            
        elif vol_ratio < 0.5:  # 거래량 부진
            adjusted['min_wave_atr_ratio'] = self._limit_adjustment(
                adjusted['min_wave_atr_ratio'],
                adjusted['min_wave_atr_ratio'] * 0.7
            )
            logger.debug("[AdaptiveParameterAdjuster] 거래량 부진 조정: vol_ratio=%.2f", vol_ratio)
        
        return self._clamp_params(adjusted)
    
    def adjust_for_trend(self, df: pd.DataFrame, indicators: Optional[dict] = None) -> dict:
        """추세 강도 기반 파라미터 조정
        
        Args:
            df: OHLCV 데이터
            indicators: 미리 계산된 공통 지표 (성능 최적화)
            
        Returns:
            조정된 파라미터
        """
        if indicators is None:
            indicators = self._compute_common_indicators(df)
        
        if 'ma_short' not in indicators or 'ma_long' not in indicators:
            return self._get_base_params()
        
        trend_strength = indicators.get('trend_strength', 0.0)
        if pd.isna(trend_strength):
            trend_strength = 0.0
        
        adjusted = self._get_base_params()
        
        if trend_strength > 0.02:  # 강한 추세
            adjusted['confirmation_bars'] = max(1, adjusted['confirmation_bars'] - 1)
            adjusted['min_wave_atr_ratio'] = self._limit_adjustment(
                adjusted['min_wave_atr_ratio'],
                adjusted['min_wave_atr_ratio'] * 0.8
            )
            logger.debug("[AdaptiveParameterAdjuster] 강한 추세 조정: trend_strength=%.4f", trend_strength)
            
        elif trend_strength < 0.005:  # 횡보
            adjusted['confirmation_bars'] = min(5, adjusted['confirmation_bars'] + 1)
            adjusted['min_wave_atr_ratio'] = self._limit_adjustment(
                adjusted['min_wave_atr_ratio'],
                adjusted['min_wave_atr_ratio'] * 1.2
            )
            logger.debug("[AdaptiveParameterAdjuster] 횡보 조정: trend_strength=%.4f", trend_strength)
        
        return self._clamp_params(adjusted)
    
    def adjust_for_time_of_day(self, current_time: Optional[datetime] = None) -> dict:
        """시간대 기반 파라미터 조정
        
        Args:
            current_time: 현재 시간 (None이면 현재 시간 사용)
            
        Returns:
            조정된 파라미터
        """
        # timezone-naive datetime 사용 (한국 로컬 시간 기준)
        # timezone-aware datetime이 전달되어도 .hour/.minute 접근은 정상
        if current_time is None:
            current_time = datetime.now()
        
        hour = current_time.hour
        minute = current_time.minute
        time_minutes = hour * 60 + minute
        
        adjusted = self._get_base_params()
        
        # JIF 기반 장 상태 판단 (초기화 시 임포트한 constants 사용)
        # [FIX-TIME-FALLBACK] constants 없을 때 시간 범위로 장중 여부 대략 판단
        if self._is_market_closed is None:
            # 09:00~15:30 범위이면 장 중으로 간주 (폴백)
            is_closed = not (540 <= time_minutes < 930)
            logger.debug("[AdaptiveParameterAdjuster] constants 없음, 시간 기반 폴백: time_min=%d, is_closed=%s",
                         time_minutes, is_closed)
        else:
            try:
                is_closed = self._is_market_closed(use_jif=True)
            except Exception:
                # 안전 폴백: 시간 범위로 판단
                is_closed = not (540 <= time_minutes < 930)
        
        # 장 중인 경우에만 시간대 기반 조정 적용
        if not is_closed:
            # 장 초반 (09:00-10:00): 변동성 높음 → 빠른 확정
            if 540 <= time_minutes < 600:
                # confirmation_bars는 정수형이므로 _limit_adjustment 대신 직접 ±1 조정
                # (PARAM_RANGES 클램핑으로 범위 보장)
                adjusted['confirmation_bars'] = max(1, adjusted['confirmation_bars'] - 1)
                adjusted['atr_multiplier'] = self._limit_adjustment(
                    adjusted['atr_multiplier'],
                    adjusted['atr_multiplier'] * 1.2
                )
                logger.debug("[AdaptiveParameterAdjuster] 장 초반 조정")
                
            # 점심시간 (12:00-13:00): 변동성 낮음 → 느린 확정
            elif 720 <= time_minutes < 780:
                adjusted['confirmation_bars'] = min(5, adjusted['confirmation_bars'] + 1)
                adjusted['atr_multiplier'] = self._limit_adjustment(
                    adjusted['atr_multiplier'],
                    adjusted['atr_multiplier'] * 0.8
                )
                logger.debug("[AdaptiveParameterAdjuster] 점심시간 조정")
                
            # 장 마감 직전 (15:00-15:30): 변동성 높음
            elif 900 <= time_minutes < 930:
                adjusted['confirmation_bars'] = max(1, adjusted['confirmation_bars'] - 1)
                logger.debug("[AdaptiveParameterAdjuster] 장 마감 직전 조정")
        
        return self._clamp_params(adjusted)
    
    def adjust_for_oneway(self, df: pd.DataFrame, indicators: Optional[dict] = None) -> dict:
        """원웨이 추세 무력화 방지 파라미터 조정
        
        강한 원웨이(One-way) 추세에서 ZigZag가 반대편 피봇을 발생시키지 못해
        지표가 무력화되는 현상을 방지하기 위해 파라미터를 동적으로 조정합니다.
        
        Args:
            df: OHLCV 데이터
            indicators: 미리 계산된 공통 지표 (선택 사항)
            
        Returns:
            조정된 파라미터
        """
        if len(df) < 5:
            return self._get_base_params()
        
        # indicators가 없으면 계산
        if indicators is None:
            indicators = self._compute_common_indicators(df)
        
        adjusted = self._get_base_params()
        oneway_detected = False

        # [FIX-ONEWAY-ACCUM] 각 조건이 직전 단계의 adjusted 값을 기반으로 누적 적용.
        # 기존 코드는 각 조건이 항상 _get_base_params() 기준으로 독립 적용하여
        # 조건 1에서 0.5배 줄인 threshold가 조건 2에서는 base 기준 0.6배로 덮어써지는
        # 의도하지 않은 비누적 동작이 발생했다.

        # 1. 추세 가속도 기반 조정 (ROC - Rate of Change)
        try:
            # 최근 3봉의 기울기(ROC) 계산
            returns = df['close'].pct_change(3).iloc[-1]

            # 원웨이 폭등/폭락 상황 (0.5% 이상 급변)
            if abs(returns) > 0.005:
                # 추세 방향으로 피봇이 계속 밀리는 것을 방지하기 위해
                # 오히려 임계치를 낮추어 작은 눌림목에도 피봇이 찍히게 유도
                adjusted['confirmation_bars'] = 1
                adjusted['pivot_threshold_min_pct'] = self._limit_adjustment(
                    adjusted['pivot_threshold_min_pct'],
                    adjusted['pivot_threshold_min_pct'] * 0.5
                )
                oneway_detected = True
                logger.debug("[AdaptiveParameterAdjuster] 원웨이 가속 감지: ROC=%.4f → 극단적 공격성 부여", returns)
        except (IndexError, KeyError) as e:
            logger.debug("[AdaptiveParameterAdjuster] ROC 계산 실패: %s", e)

        # 2. 가격 이격도(Disparity) 기반 조정 — 조건 1 결과에 누적 적용
        try:
            if 'ma_long' in indicators:
                current_price = df['close'].iloc[-1]
                ma_long = indicators['ma_long']

                if ma_long > 0:
                    disparity = abs(current_price - ma_long) / ma_long

                    # 이격도가 과도하게 벌어진 경우 (강한 원웨이)
                    if disparity > 0.03:  # 3% 이상 이격
                        # 역설적으로 threshold를 낮추어 '작은 반등'에도 피봇이 확정되게 함
                        # [FIX-ONEWAY-ACCUM] 조건 1에서 이미 줄어든 adjusted 값 기반으로 추가 감소
                        adjusted['pivot_threshold_min_pct'] = self._limit_adjustment(
                            adjusted['pivot_threshold_min_pct'],
                            adjusted['pivot_threshold_min_pct'] * 0.6
                        )
                        adjusted['min_wave_atr_ratio'] = self._limit_adjustment(
                            adjusted['min_wave_atr_ratio'],
                            adjusted['min_wave_atr_ratio'] * 0.5
                        )
                        # confirmation_bars=1 중복 set 제거 (조건 1에서 이미 설정)
                        if adjusted['confirmation_bars'] > 1:
                            adjusted['confirmation_bars'] = 1
                        oneway_detected = True
                        logger.debug("[AdaptiveParameterAdjuster] 원웨이 이격 과다 감지: disparity=%.4f → 민감도 극대화", disparity)
        except (KeyError, IndexError) as e:
            logger.debug("[AdaptiveParameterAdjuster] 이격도 계산 실패: %s", e)

        # 3. 추세 지속 시간 기반 조정 — 조건 1·2 결과에 누적 적용
        try:
            # 최근 20봉의 추세 방향 확인
            recent_closes = df['close'].tail(20)
            if len(recent_closes) >= 10:
                # 단순화된 기울기 계산 (첫 봉과 끝 봉의 차이)
                # np.polyfit 대신 단순 차이 비율 사용 (성능 최적화)
                first_close = recent_closes.iloc[0]
                last_close = recent_closes.iloc[-1]

                if first_close > 0:
                    slope = (last_close - first_close) / first_close

                    # 기울기가 일정 이상이고 방향이 일관성 있는 경우
                    if abs(slope) > 0.01:  # 1% 이상 변화
                        # 추세가 길어질수록 작은 되돌림에도 민감하게 반응
                        # [FIX-ONEWAY-ACCUM] 직전 단계 adjusted 기반으로 추가 감소
                        adjusted['min_wave_atr_ratio'] = self._limit_adjustment(
                            adjusted['min_wave_atr_ratio'],
                            adjusted['min_wave_atr_ratio'] * 0.7
                        )
                        oneway_detected = True
                        logger.debug("[AdaptiveParameterAdjuster] 원웨이 추세 지속 감지: slope=%.4f", slope)
        except (IndexError, ZeroDivisionError) as e:
            logger.debug("[AdaptiveParameterAdjuster] 추세 지속 계산 실패: %s", e)
        
        if oneway_detected:
            logger.info("[AdaptiveParameterAdjuster] 원웨이 추세 감지 → 파라미터 조정 완료")
        
        return self._clamp_params(adjusted)
    
    def adjust_for_lag_feedback(self, recent_lags: List[Union[int, float]],
                            success_rates: Optional[List[float]] = None) -> dict:
        """최근 피봇 지연시간 기반 파라미터 조정 (성공률 고려)
        
        Args:
            recent_lags: 최근 지연시간 리스트
            success_rates: 최근 피봇 확정 성공률 리스트 (선택 사항, 0~1)
            
        Returns:
            조정된 파라미터
        """
        # [FIX-LAG-NONE] None/비정상 값 방어 후 클리핑
        cleaned_lags = [float(x) for x in recent_lags if x is not None and np.isfinite(float(x))]
        if len(cleaned_lags) < 5:
            return self._get_base_params()

        # 음수 지연시간 클리핑
        avg_lag = float(np.mean(np.clip(cleaned_lags, 0, None)))
        # success_rate=0.5: 중립 가정 (보수적/공격적 조정 없음 의도)
        # feedback_score = 0*0.4 + 0.5*0.6 = 0.30 → 정확히 CONSERVATIVE 경계
        # avg_lag=0이면 경계값 → 중간 상태 (조정 없음) ← 의도된 동작
        success_rate = float(np.clip(np.mean(success_rates), 0.0, 1.0)) \
            if success_rates and len(success_rates) >= 5 else 0.5
        
        # 지연 + 정확도 복합 스코어 (정규화 후 합산)
        lag_score = min(avg_lag / self.MAX_LAG_SECONDS, 1.0)  # 0~1
        accuracy_score = 1.0 - success_rate               # 0~1
        feedback_score = lag_score * 0.4 + accuracy_score * 0.6  # 0~1
        
        adjusted = self._get_base_params()
        
        if feedback_score > self.FEEDBACK_THRESHOLD_AGGRESSIVE:  # 지연 길거나 정확도 낮음 - 공격적 조정
            adjusted['confirmation_bars'] = max(1, adjusted['confirmation_bars'] - 1)
            adjusted['pivot_threshold_min_pct'] = self._limit_adjustment(
                adjusted['pivot_threshold_min_pct'],
                adjusted['pivot_threshold_min_pct'] * 0.85  # 더 낮게
            )
            adjusted['atr_multiplier'] = self._limit_adjustment(
                adjusted['atr_multiplier'],
                adjusted['atr_multiplier'] * 1.2  # 더 높게
            )
            logger.debug("[AdaptiveParameterAdjuster] 피드백 공격적 조정: score=%.2f (lag=%.2f, success=%.2f)", 
                        feedback_score, avg_lag, success_rate)
            
        elif feedback_score < self.FEEDBACK_THRESHOLD_CONSERVATIVE:  # 지연 짧고 정확도 높음 - 보수적 조정
            adjusted['confirmation_bars'] = min(5, adjusted['confirmation_bars'] + 1)
            adjusted['min_wave_atr_ratio'] = self._limit_adjustment(
                adjusted['min_wave_atr_ratio'],
                adjusted['min_wave_atr_ratio'] * 1.2
            )
            logger.debug("[AdaptiveParameterAdjuster] 피드백 보수적 조정: score=%.2f (lag=%.2f, success=%.2f)", 
                        feedback_score, avg_lag, success_rate)
            
        else:  # 중간 상태: 조정 없음 (안정 유지)
            logger.debug("[AdaptiveParameterAdjuster] 피드백 중간 상태, 조정 없음: score=%.2f", feedback_score)
        
        return self._clamp_params(adjusted)
    
    def _compute_dynamic_weights(self, df: pd.DataFrame, indicators: Optional[dict] = None) -> dict:
        """장상황별 가중치 동적 계산
        
        Args:
            df: OHLCV 데이터
            indicators: 미리 계산된 공통 지표 (선택 사항, 중복 계산 방지)
            
        Returns:
            동적으로 조정된 가중치 딕셔너리
        """
        if indicators is None:
            indicators = self._compute_common_indicators(df)
        
        weights = {
            'volatility': 0.40,
            'trend': 0.30, 
            'volume': 0.20,
            'time': 0.10
        }
        
        # 고변동성 시 변동성 가중치 ↑
        if self.atr_ma and self.atr_history:
            current_atr = self.atr_history[-1]
            if current_atr > 0:  # ATR=0 방어
                vol_ratio = current_atr / self.atr_ma
                if vol_ratio > 1.5:
                    weights['volatility'] += 0.15
                    weights['time'] -= 0.10
                elif vol_ratio < 0.7:
                    weights['volatility'] -= 0.10
                    weights['trend'] += 0.10
        
        # 강한 추세 시 추세 가중치 ↑
        if 'trend_strength' in indicators:
            trend_strength = indicators['trend_strength']
            if pd.isna(trend_strength):
                trend_strength = 0.0
            if trend_strength > 0.02:
                weights['trend'] += 0.15
                weights['volume'] -= 0.10
            elif trend_strength < 0.005:
                weights['volume'] += 0.10
                weights['trend'] -= 0.10
        
        # 원웨이 해제 조건: 추세가 꺾일 때 횡보 가중치 ↑ (confirmation_bars 복구)
        try:
            if len(df) >= 5:
                # 최근 3봉 ROC 계산
                recent_roc = df['close'].pct_change(3).iloc[-1]
                # 이전 3봉 ROC 계산
                if len(df) >= 8:
                    prev_roc = df['close'].pct_change(3).iloc[-4]
                    
                    # ROC가 급격히 줄어들면 추세 꺾임으로 판단
                    if abs(recent_roc) < 0.002 and abs(prev_roc) > 0.005:
                        # 횡보 가중치 ↑, 추세 가중치 ↓
                        weights['trend'] -= 0.10
                        weights['volume'] += 0.10
                        logger.debug("[AdaptiveParameterAdjuster] 원웨이 해제 감지: ROC 급감 → 횡보 가중치 증가")
        except (IndexError, KeyError) as e:
            logger.debug("[AdaptiveParameterAdjuster] 원웨이 해제 감지 실패: %s", e)
        
        # [FIX-MIN-WEIGHT] 정규화 전에 MIN_WEIGHT 클램프 → 재정규화 1회로 충분
        # 기존: 정규화 → 재정규화 후 min 검사 → 2차 재정규화 (재정규화 후 또 min 미달 가능)
        # 수정: min 클램프 → 정규화 1회 (수학적으로 항상 수렴 보장)
        MIN_WEIGHT = 0.05
        weights = {k: max(0.0, v) for k, v in weights.items()}
        weights = {k: max(MIN_WEIGHT, v) for k, v in weights.items()}  # min 보장 선적용
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(weights) for k in weights}  # 균등 분배
        
        logger.debug("[AdaptiveParameterAdjuster] 동적 가중치: %s", weights)
        return weights
    
    def _weighted_average(self, params_with_weights: List[Tuple[dict, float]]) -> dict:
        """가중 평균 계산

        Args:
            params_with_weights: [(params_dict, weight), ...] 리스트

        Returns:
            가중 평균된 파라미터 (confirmation_bars는 float 유지, _clamp_params에서 int 변환)
        """
        # 빈 리스트 입력 방어 (초기화 전에 체크)
        if not params_with_weights:
            logger.warning("[AdaptiveParameterAdjuster] 빈 가중치 리스트, 기본값 반환")
            return self._get_base_params()
        
        # result 초기화 (base_params 키 기반)
        result = {key: 0.0 for key in self._get_base_params().keys()}
        
        # 폴백용 기본 파라미터 (루프 외부에서 1회 계산)
        base_fallback = self._get_base_params()
        
        total_weight = 0.0
        for params, weight in params_with_weights:
            # 키 누락 감지 및 방어
            missing = set(result.keys()) - set(params.keys())
            if missing:
                logger.warning("[AdaptiveParameterAdjuster] 파라미터 키 누락: %s", missing)
            total_weight += weight
            for key in result.keys():
                result[key] += params.get(key, base_fallback.get(key, 0.0)) * weight
        
        if total_weight > 0:
            for key in result.keys():
                result[key] /= total_weight
        else:
            # 총 가중치가 0인 경우 기본값 반환
            logger.warning("[AdaptiveParameterAdjuster] 총 가중치가 0, 기본값 반환")
            return self._get_base_params()
        
        return result
    
    def get_adaptive_params(self, df: pd.DataFrame, 
                           recent_lags: Optional[List[Union[int, float]]] = None,
                           success_rates: Optional[List[float]] = None,
                           current_time: Optional[datetime] = None,
                           use_dynamic_weights: bool = False,
                           use_strategies: bool = False) -> dict:
        """모든 요소 통합하여 파라미터 조정

        [FIX-RETURN-TYPE] 반환 타입을 AdaptiveParams → dict 로 변경.
        adaptive_zigzag.py 호출부가 adjusted_params.get('key') 형태로 접근하므로
        dataclass를 반환하면 AttributeError가 발생한다. dict를 반환해야 호환된다.

        참고: 각 adjust_for_* 메서드는 독립적으로 base_params에서 조정하고,
        _weighted_average로 합산하는 구조입니다. 이 방식은 각 조정이 기준값 대비
        ±30% 이내로 제한되므로, 복합 상황(고변동성 + 강한 추세 동시 발생)에서는
        조정 폭이 실제로는 희석될 수 있습니다. 이는 안정성과 민감도 간의 트레이드오프입니다.

        Args:
            df: OHLCV 데이터
            recent_lags: 최근 지연시간 리스트 (선택 사항)
            success_rates: 최근 피봇 확정 성공률 리스트 (선택 사항, 0~1)
            current_time: 현재 시간 (선택 사항)
            use_dynamic_weights: 동적 가중치 사용 여부 (기본 False)
            use_strategies: 플러그인 전략 사용 여부 (기본 False)

        Returns:
            조정된 파라미터 dict (키: atr_multiplier, pivot_threshold_min_pct,
                                       confirmation_bars, min_wave_atr_ratio)
        """
        # 진입점 유효성 검사 (타입 검사 포함)
        required_cols = {'high', 'low', 'close', 'volume'}
        if not isinstance(df, pd.DataFrame) or df.empty or not required_cols.issubset(df.columns):
            logger.warning("[AdaptiveParameterAdjuster] df 유효성 검사 실패, 기본값 반환")
            return self._get_base_params()  # [FIX-RETURN-TYPE] dict 반환
        
        # ATR 통계 업데이트 (캐싱 적용)
        self.update_atr_stats(df)
        
        # 공통 지표 미리 계산 (DRY 원칙, 성능 최적화)
        indicators = self._compute_common_indicators(df)
        base = self._get_base_params()
        
        # 조정 결과 수집 (분리된 메서드)
        param_map = self._collect_adjusted_params(df, base, indicators, current_time, use_strategies)
        
        # 가중치 계산 (use_strategies와 무관하게 프로필 기반 통일)
        if use_dynamic_weights:
            weights = self._compute_dynamic_weights(df, indicators)
        else:
            # 항상 프로필 기반 고정 가중치 (use_strategies와 무관)
            # trend:volume:time = 3:2:1 비율
            vol_w = self.profile['vol_weight']
            remaining = 1.0 - vol_w
            weights = {
                'volatility': vol_w,
                'trend': remaining / 2,      # 3/6
                'volume': remaining / 3,      # 2/6
                'time': remaining / 6       # 1/6
            }
        
        combined = self._apply_weights(param_map, weights, base)
        
        # 피드백 루프 적용 (최우선) - 성공률 고려
        if recent_lags is not None and len(recent_lags) >= 5:
            feedback_params = self.adjust_for_lag_feedback(recent_lags, success_rates)
            combined = self._weighted_average([
                (combined, 0.70),
                (feedback_params, 0.30)
            ])
        
        # 클램핑 (_clamp_params 내부에서 int 변환 수행)
        clamped = self._clamp_params(combined)
        
        # 조정 이력 저장 (최종 적용값 저장)
        self.adjustment_history.append(clamped)
        
        # 메트릭 수집 (모니터링 강화)
        self._log_adjustment_metrics(base, clamped, indicators)
        
        logger.debug("[AdaptiveParameterAdjuster] 최종 파라미터: %s", clamped)

        return clamped  # [FIX-RETURN-TYPE] dict 반환 (AdaptiveParams 아님)
    
    def _log_adjustment_metrics(self, before: dict, after: dict, indicators: dict) -> None:
        """조정 메트릭 로깅 (모니터링 강화)
        
        Args:
            before: 조정 전 파라미터
            after: 조정 후 파라미터
            indicators: 공통 지표
        """
        try:
            metrics = {
                'adjustment_ratio': {
                    k: float(after[k]) / float(before[k]) if abs(before[k]) > 1e-9 else 1.0
                    for k in after.keys()
                },
                'situation': {
                    'vol_ratio': (self.atr_history[-1] / self.atr_ma)
                               if (self.atr_ma and self.atr_history) else 1.0,
                    'trend_strength': (lambda ts: float(ts) if not pd.isna(ts) else 0.0)(
                        indicators.get('trend_strength', 0.0)
                    ),
                }
            }
            logger.debug("[AdaptiveParameterAdjuster] 조정 메트릭: ratio=%s, situation=%s",
                    metrics['adjustment_ratio'], metrics['situation'])
        except Exception as e:
            logger.debug("[AdaptiveParameterAdjuster] 메트릭 로깅 실패: %s", e)
