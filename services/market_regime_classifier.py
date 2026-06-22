"""
시장 레짐(Market Regime) 분류기

시장 상태를 변동성과 방향성 기준으로 분류하여 적합한 전략을 선택하도록 지원합니다.

주요 기능:
- 변동성 상태 판단 (ATR, 표준편차)
- 방향성 판단 (ADX, 추세 강도)
- 시장 상태 분류 (고변동/저변동 + 방향성/무방향)
- 장초반 이벤트장 감지
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VolatilityState(Enum):
    """변동성 상태"""
    HIGH = "high"      # 고변동
    LOW = "low"        # 저변동
    NORMAL = "normal"  # 정상


class TrendDirection(Enum):
    """추세 방향"""
    UP = "up"          # 상승
    DOWN = "down"      # 하락
    NEUTRAL = "neutral"  # 횡보/무방향


class MarketRegime(Enum):
    """시장 레짐"""
    HIGH_VOL_NO_DIRECTION = "high_vol_no_direction"  # 고변동+무방향 (흔들기)
    HIGH_VOL_UP = "high_vol_up"                      # 고변동+상승 (강한 상승 추세)
    HIGH_VOL_DOWN = "high_vol_down"                  # 고변동+하락 (강한 하락 추세)
    NORMAL_VOL_NO_DIRECTION = "normal_vol_no_direction"  # 정상 변동성+무방향
    NORMAL_VOL_UP = "normal_vol_up"                      # 정상 변동성+상승
    NORMAL_VOL_DOWN = "normal_vol_down"                  # 정상 변동성+하락
    LOW_VOL_NO_DIRECTION = "low_vol_no_direction"    # 저변동+무방향 (횡보)
    LOW_VOL_UP = "low_vol_up"                        # 저변동+상승 (느린 상승)
    LOW_VOL_DOWN = "low_vol_down"                    # 저변동+하락 (느린 하락)
    OPENING_EVENT = "opening_event"                  # 장초반 이벤트장
    NEWS_EVENT = "news_event"                        # 뉴스 급등락장


@dataclass
class MarketState:
    """시장 상태 정보"""
    regime: MarketRegime
    volatility_state: VolatilityState
    trend_direction: TrendDirection
    atr: float
    atr_ratio: float  # ATR / 가격
    adx: float
    std_dev: float
    std_ratio: float  # 표준편차 / 가격
    price: float
    timestamp: pd.Timestamp
    is_opening_session: bool  # 장초반 여부
    confidence: float  # 분류 신뢰도 (0~1)


class MarketRegimeClassifier:
    """
    시장 레짐 분류기
    
    변동성과 방향성 지표를 사용하여 시장 상태를 분류합니다.
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        adx_period: int = 14,
        std_period: int = 20,
        vol_high_threshold: float = 0.005,  # ATR 비율 기준 0.5% (KP200 현실에 맞게 조정)
        vol_low_threshold: float = 0.0015,  # ATR 비율 기준 0.15% (KP200 현실에 맞게 조정)
        adx_trend_threshold: float = 20,   # ADX 기준 20 (KP200 현실에 맞게 조정)
        adx_weak_threshold: float = 12,    # ADX 기준 12 (KP200 현실에 맞게 조정)
        opening_minutes: int = 30,         # 장초반 기준 (분)
        market_open_hour: int = 8,        # 장 시작 시간 (시) - KP200 선물 08:45
        market_open_minute: int = 45,      # 장 시작 시간 (분) - KP200 선물 08:45
        enable_option_sentiment: bool = False,  # 옵션 센티먼트 활성화 여부
        sentiment_confidence_boost: float = 0.2,  # 센티먼트 일치 시 신뢰도 상향 폭
        sentiment_confidence_penalty: float = 0.2,  # 센티먼트 불일치 시 신뢰도 하향 폭
        ma_short_period: int = 20,         # 단기 이동평균 기간
        ma_long_period: int = 60,          # 장기 이동평균 기간
        enable_enhanced_trend: bool = True, # 향상된 추세 분석 활성화 여부
    ):
        """
        Args:
            atr_period: ATR 계산 기간
            adx_period: ADX 계산 기간
            std_period: 표준편차 계산 기간
            vol_high_threshold: 고변동 기준 (ATR/가격 비율)
            vol_low_threshold: 저변동 기준 (ATR/가격 비율)
            adx_trend_threshold: 추세 기준 (ADX)
            adx_weak_threshold: 횡보 기준 (ADX)
            opening_minutes: 장초반 기준 시간 (분)
            market_open_hour: 장 시작 시간 (시) - KP200 선물 08:45
            market_open_minute: 장 시작 시간 (분) - KP200 선물 08:45
            enable_option_sentiment: 옵션 센티먼트 활성화 여부
            sentiment_confidence_boost: 센티먼트 일치 시 신뢰도 상향 폭
            sentiment_confidence_penalty: 센티먼트 불일치 시 신뢰도 하향 폭
            ma_short_period: 단기 이동평균 기간 (기본값: 20)
            ma_long_period: 장기 이동평균 기간 (기본값: 60)
            enable_enhanced_trend: 향상된 추세 분석 활성화 여부 (기본값: True)
        """
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.std_period = std_period
        self.vol_high_threshold = vol_high_threshold
        self.vol_low_threshold = vol_low_threshold
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_weak_threshold = adx_weak_threshold
        self.opening_minutes = opening_minutes
        self.market_open_hour = market_open_hour
        self.market_open_minute = market_open_minute
        self.enable_option_sentiment = enable_option_sentiment
        self.sentiment_confidence_boost = sentiment_confidence_boost
        self.sentiment_confidence_penalty = sentiment_confidence_penalty
        self.ma_short_period = ma_short_period
        self.ma_long_period = ma_long_period
        self.enable_enhanced_trend = enable_enhanced_trend
    
    def classify(
        self, 
        df: pd.DataFrame, 
        current_idx: int = -1,
        skew: Optional[float] = None,
        volume_pcr: Optional[float] = None,
        oi_pcr: Optional[float] = None,
    ) -> Optional[MarketState]:
        """
        시장 상태 분류

        Args:
            df: OHLCV 데이터 (High, Low, Close, Volume 컬럼 필요)
            current_idx: 분석할 인덱스 (기본값: 최신 데이터)
            skew: 옵션 Skew (call_iv - put_iv), 양수=강세, 음수=약세
            volume_pcr: 옵션 Volume PCR (put_volume / call_volume), 낮을수록 강세
            oi_pcr: 옵션 OI PCR (put_OI / call_OI), 낮을수록 강세

        Returns:
            MarketState 객체 또는 None (데이터 부족 시)
        """
        # 데이터 검증
        required_cols = {'High', 'Low', 'Close'}
        if not required_cols.issubset(df.columns):
            logger.error("[MarketRegimeClassifier] 필수 컬럼 누락: %s", required_cols - set(df.columns))
            return None

        if len(df) < max(self.atr_period, self.adx_period, self.std_period) + 1:
            logger.warning("[MarketRegimeClassifier] 데이터 부족 (필요: %d, 실제: %d)",
                          max(self.atr_period, self.adx_period, self.std_period) + 1, len(df))
            return None
        
        try:
            # 현재 데이터 추출
            if current_idx < 0:
                current_idx = len(df) + current_idx

            # NaN 데이터 검사 (current_idx 결정 후)
            if df[list(required_cols)].iloc[:current_idx+1].isnull().any().any():
                logger.warning("[MarketRegimeClassifier] NaN 데이터 포함")
            
            # current_idx=0 엣지 케이스: TR 배열이 df[1:] 기반이므로 유효하지 않음
            if current_idx < 1:
                logger.warning("[MarketRegimeClassifier] current_idx=%d: 최소 1 이상 필요 (TR 배열 오프셋)", current_idx)
                return None
            
            current_data = df.iloc[current_idx]
            price = current_data['Close']
            timestamp = current_data.name if hasattr(current_data, 'name') else pd.Timestamp.now()
            
            # 장초반 여부 판단
            is_opening_session = self._is_opening_session(timestamp)
            
            # 변동성 지표 계산
            atr, atr_ratio = self._calculate_atr(df, current_idx, price)
            std_dev, std_ratio = self._calculate_std(df, current_idx, price)
            
            # 방향성 지표 계산
            adx, plus_di, minus_di = self._calculate_adx(df, current_idx)
            
            # 변동성 상태 판단
            vol_state = self._classify_volatility(atr_ratio, std_ratio)
            
            # 추세 방향 판단
            if self.enable_enhanced_trend:
                trend_dir, trend_debug = self._classify_trend_enhanced(df, current_idx, adx, plus_di, minus_di)
                logger.debug(
                    "[MarketRegime] 향상된 추세 분석: %s (DI: %s, MA: %s, Structure: %s, Votes: %s)",
                    trend_dir.value,
                    trend_debug.get("di_direction"),
                    trend_debug.get("ma_direction"),
                    trend_debug.get("market_structure"),
                    trend_debug.get("votes")
                )
            else:
                trend_dir = self._classify_trend(adx, plus_di, minus_di)
            
            # 시장 레짐 결정
            regime = self._determine_regime(
                vol_state, trend_dir, is_opening_session, atr_ratio, adx
            )
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(atr_ratio, adx, vol_state, trend_dir)
            
            # 옵션 센티먼트 필터링 (다단계 필터링: 신뢰도 보정)
            if self.enable_option_sentiment and all(v is not None for v in [skew, volume_pcr, oi_pcr]):
                confidence = self._apply_option_sentiment_filter(
                    confidence, trend_dir, skew, volume_pcr, oi_pcr
                )
            
            return MarketState(
                regime=regime,
                volatility_state=vol_state,
                trend_direction=trend_dir,
                atr=atr,
                atr_ratio=atr_ratio,
                adx=adx,
                std_dev=std_dev,
                std_ratio=std_ratio,
                price=price,
                timestamp=timestamp,
                is_opening_session=is_opening_session,
                confidence=confidence,
            )
            
        except Exception as e:
            logger.error("[MarketRegimeClassifier] 분류 실패: %s", e, exc_info=True)
            return None
    
    def _is_opening_session(self, timestamp: pd.Timestamp) -> bool:
        """장초반 여부 판단
        
        날짜 경계를 고려하여 계산합니다.
        """
        market_open = timestamp.replace(
            hour=self.market_open_hour,
            minute=self.market_open_minute,
            second=0, microsecond=0
        )
        delta = (timestamp - market_open).total_seconds() / 60
        return 0 <= delta < self.opening_minutes
    
    def _calculate_atr(self, df: pd.DataFrame, current_idx: int, price: float) -> Tuple[float, float]:
        """ATR 계산"""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        # True Range 계산
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR 계산 (RMA 방식)
        atr = self._calculate_rma(tr, self.atr_period)
        
        # TR 배열은 df[1:] 기반이라 길이가 len(df) - 1
        # 인덱스 오프셋 보정 필요
        if current_idx >= len(atr):
            atr_value = atr[-1]
        else:
            tr_idx = min(current_idx - 1, len(atr) - 1)
            atr_value = atr[max(tr_idx, 0)]
        
        # NaN 체크 (warm-up 구간)
        if np.isnan(atr_value):
            atr_value = atr[-1] if not np.isnan(atr[-1]) else 0.0
        
        atr_ratio = atr_value / price if price > 0 else 0
        
        return atr_value, atr_ratio
    
    def _calculate_rma(self, data: np.ndarray, period: int) -> np.ndarray:
        """RMA (Wilder's Smoothing) 계산
        
        Wilder's RMA는 초기 수십 개 값이 부정확합니다.
        첫 period개는 단순 평균으로 시드(seed)하는 것이 표준입니다.
        warm-up 구간은 NaN으로 마킹하여 유효하지 않음을 표시합니다.
        """
        alpha = 1.0 / period
        rma = np.full_like(data, np.nan, dtype=float)  # NaN으로 초기화
        
        if len(data) >= period:
            rma[period - 1] = np.mean(data[:period])  # 시드값
            for i in range(period, len(data)):
                rma[i] = alpha * data[i] + (1 - alpha) * rma[i - 1]
        else:
            # 데이터가 period보다 적으면 단순 누적 평균
            if len(data) > 0:
                rma[0] = data[0]  # 먼저 설정
            for i in range(1, len(data)):
                rma[i] = alpha * data[i] + (1 - alpha) * rma[i - 1]
        
        return rma
    
    def _calculate_std(self, df: pd.DataFrame, current_idx: int, price: float) -> Tuple[float, float]:
        """표준편차 계산"""
        close = df['Close'].values
        
        if current_idx >= self.std_period:
            window = close[current_idx - self.std_period + 1:current_idx + 1]
        else:
            window = close[:current_idx + 1]
        
        if len(window) < 2:
            std_value = 0.0
        else:
            std_value = np.std(window, ddof=1)
        
        std_ratio = std_value / price if price > 0 else 0
        
        return std_value, std_ratio
    
    def _calculate_ma_slope(self, df: pd.DataFrame, period: int, current_idx: int) -> float:
        """이동평균 기울기 계산
        
        Args:
            df: OHLCV 데이터
            period: 이동평균 기간
            current_idx: 현재 인덱스
            
        Returns:
            기울기 (양수: 상승, 음수: 하락)
        """
        close = df['Close'].values
        
        if len(close) < period + 1:
            return 0.0
        
        # 최근 기간의 MA와 이전 기간의 MA 계산
        if current_idx >= len(close):
            current_ma = close[-period:].mean()
            prev_ma = close[-period-1:-1].mean()
        else:
            end_idx = min(current_idx + 1, len(close))
            start_idx = max(end_idx - period, 0)
            prev_end_idx = max(end_idx - 1, period)
            prev_start_idx = max(prev_end_idx - period, 0)
            
            current_ma = close[start_idx:end_idx].mean()
            prev_ma = close[prev_start_idx:prev_end_idx].mean()
        
        # 기울기 계산
        if prev_ma > 0:
            slope = (current_ma - prev_ma) / prev_ma
        else:
            slope = 0.0
        
        return slope
    
    def _classify_market_structure(self, df: pd.DataFrame, current_idx: int, lookback_swings: int = 5) -> str:
        """가격 구조 분류 (Higher High/Lower Low 등)
        
        피벗 구조를 분석하여:
        - Higher High + Higher Low → 상승 구조
        - Lower High + Lower Low → 하락 구조
        - 그 외 → 횡보 구조
        
        Args:
            df: OHLCV 데이터
            current_idx: 현재 인덱스
            lookback_swings: 분석할 피벗 개수
            
        Returns:
            "uptrend", "downtrend", "ranging"
        """
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # 데이터 부족 시 횡보 반환
        if len(close) < lookback_swings * 3:
            return "ranging"
        
        # 간단한 피벗 감지 (로컬 고점/저점)
        # 실전에서는 ZigZag 등의 피벗 알고리즘 사용 권장
        pivots_high = []
        pivots_low = []
        
        window = 3  # 피벗 감지 윈도우
        
        for i in range(window, len(close) - window):
            # 로컬 고점
            if high[i] == high[i-window:i+window+1].max():
                pivots_high.append((i, high[i]))
            # 로컬 저점
            if low[i] == low[i-window:i+window+1].min():
                pivots_low.append((i, low[i]))
        
        # 최근 피벗만 분석
        recent_highs = pivots_high[-lookback_swings:] if len(pivots_high) >= lookback_swings else pivots_high
        recent_lows = pivots_low[-lookback_swings:] if len(pivots_low) >= lookback_swings else pivots_low
        
        # 피벗 부족 시 횡보 반환
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return "ranging"
        
        # Higher High/Lower Low 판단
        higher_highs = all(recent_highs[i][1] > recent_highs[i-1][1] for i in range(1, len(recent_highs)))
        lower_highs = all(recent_highs[i][1] < recent_highs[i-1][1] for i in range(1, len(recent_highs)))
        higher_lows = all(recent_lows[i][1] > recent_lows[i-1][1] for i in range(1, len(recent_lows)))
        lower_lows = all(recent_lows[i][1] < recent_lows[i-1][1] for i in range(1, len(recent_lows)))
        
        # 구조 판단
        if higher_highs and higher_lows:
            return "uptrend"
        elif lower_highs and lower_lows:
            return "downtrend"
        else:
            return "ranging"
    
    def _calculate_adx(self, df: pd.DataFrame, current_idx: int) -> Tuple[float, float, float]:
        """ADX, +DI, -DI 계산"""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        n = self.adx_period
        
        # +DM, -DM 계산
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # +DI, -DI 계산
        atr = self._calculate_rma(tr, n)
        # atr 0 나눗셈 보호
        safe_atr = np.where(atr > 1e-10, atr, 1e-10)
        plus_di = 100 * self._calculate_rma(plus_dm, n) / safe_atr
        minus_di = 100 * self._calculate_rma(minus_dm, n) / safe_atr
        
        # DX 계산
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        # ADX 계산
        adx = self._calculate_rma(dx, n)
        
        # ADX, DI 배열은 df[1:] 기반이라 길이가 len(df) - 1
        # 인덱스 오프셋 보정 필요
        if current_idx >= len(adx):
            adx_value = adx[-1]
            plus_di_value = plus_di[-1]
            minus_di_value = minus_di[-1]
        else:
            idx = min(current_idx - 1, len(adx) - 1) if current_idx > 0 else 0
            adx_value = adx[idx]
            plus_di_value = plus_di[idx]
            minus_di_value = minus_di[idx]
        
        # NaN 체크 (warm-up 구간)
        if np.isnan(adx_value):
            adx_value = adx[-1] if not np.isnan(adx[-1]) else 0.0
        if np.isnan(plus_di_value):
            plus_di_value = plus_di[-1] if not np.isnan(plus_di[-1]) else 0.0
        if np.isnan(minus_di_value):
            minus_di_value = minus_di[-1] if not np.isnan(minus_di[-1]) else 0.0
        
        return adx_value, plus_di_value, minus_di_value
    
    def _classify_volatility(self, atr_ratio: float, std_ratio: float) -> VolatilityState:
        """변동성 상태 분류
        
        ATR 비율과 표준편차 비율을 각각 임계값과 비교 후 투표 방식으로 결정합니다.
        두 지표의 스케일이 달라 단순 평균 시 왜곡이 발생할 수 있기 때문입니다.
        """
        atr_high = atr_ratio >= self.vol_high_threshold
        std_high = std_ratio >= self.vol_high_threshold
        atr_low = atr_ratio <= self.vol_low_threshold
        std_low = std_ratio <= self.vol_low_threshold

        if atr_high and std_high:
            return VolatilityState.HIGH
        if atr_low and std_low:
            return VolatilityState.LOW
        return VolatilityState.NORMAL
    
    def _classify_trend(self, adx: float, plus_di: float, minus_di: float) -> TrendDirection:
        """추세 방향 분류 (기본 버전)"""
        if adx < self.adx_weak_threshold:
            return TrendDirection.NEUTRAL
        
        if plus_di > minus_di:
            return TrendDirection.UP
        else:
            return TrendDirection.DOWN
    
    def _classify_trend_enhanced(
        self,
        df: pd.DataFrame,
        current_idx: int,
        adx: float,
        plus_di: float,
        minus_di: float,
    ) -> Tuple[TrendDirection, Dict[str, Any]]:
        """향상된 추세 방향 분류
        
        3축 조합으로 방향성 판단:
        1. ADX + DI+/DI- (추세 강도 + 방향)
        2. 이동평균 기울기 (MA20, MA60)
        3. Market Structure (Higher High/Lower Low)
        
        Args:
            df: OHLCV 데이터
            current_idx: 현재 인덱스
            adx: ADX 값
            plus_di: +DI 값
            minus_di: -DI 값
            
        Returns:
            (추세 방향, 디버그 정보 딕셔너리)
        """
        debug_info = {}
        
        # 1축: ADX 강도 체크
        if adx < self.adx_weak_threshold:
            debug_info["adx_signal"] = "weak"
            return TrendDirection.NEUTRAL, debug_info
        
        # 2축: DI 방향
        di_direction = TrendDirection.UP if plus_di > minus_di else TrendDirection.DOWN
        debug_info["di_direction"] = di_direction.value
        
        # 3축: 이동평균 기울기
        ma_short_slope = self._calculate_ma_slope(df, self.ma_short_period, current_idx)
        ma_long_slope = self._calculate_ma_slope(df, self.ma_long_period, current_idx)
        
        # MA 기울기 방향 판단
        if ma_short_slope > 0 and ma_long_slope > 0:
            ma_direction = TrendDirection.UP
        elif ma_short_slope < 0 and ma_long_slope < 0:
            ma_direction = TrendDirection.DOWN
        else:
            ma_direction = TrendDirection.NEUTRAL
        
        debug_info["ma_short_slope"] = ma_short_slope
        debug_info["ma_long_slope"] = ma_long_slope
        debug_info["ma_direction"] = ma_direction.value
        
        # 4축: Market Structure
        structure = self._classify_market_structure(df, current_idx)
        structure_direction = TrendDirection.UP if structure == "uptrend" else (TrendDirection.DOWN if structure == "downtrend" else TrendDirection.NEUTRAL)
        debug_info["market_structure"] = structure
        debug_info["structure_direction"] = structure_direction.value
        
        # 투표 방식으로 결정 (DI + MA + Structure)
        votes = [di_direction, ma_direction, structure_direction]
        up_votes = sum(1 for v in votes if v == TrendDirection.UP)
        down_votes = sum(1 for v in votes if v == TrendDirection.DOWN)
        
        debug_info["votes"] = [v.value for v in votes]
        debug_info["up_votes"] = up_votes
        debug_info["down_votes"] = down_votes
        
        # 다수결
        if up_votes > down_votes:
            return TrendDirection.UP, debug_info
        elif down_votes > up_votes:
            return TrendDirection.DOWN, debug_info
        else:
            # 동수일 경우 DI 우선
            return di_direction, debug_info
    
    def _determine_regime(
        self,
        vol_state: VolatilityState,
        trend_dir: TrendDirection,
        is_opening_session: bool,
        atr_ratio: float,
        adx: float,
    ) -> MarketRegime:
        """시장 레짐 결정"""
        # 장초반 이벤트장 우선
        if is_opening_session and atr_ratio > self.vol_high_threshold:
            return MarketRegime.OPENING_EVENT
        
        # 뉴스 급등락장 (매우 높은 변동성)
        if atr_ratio > self.vol_high_threshold * 2:
            return MarketRegime.NEWS_EVENT
        
        # 변동성 상태와 추세 방향 조합
        if vol_state == VolatilityState.HIGH:
            if trend_dir == TrendDirection.UP:
                return MarketRegime.HIGH_VOL_UP
            elif trend_dir == TrendDirection.DOWN:
                return MarketRegime.HIGH_VOL_DOWN
            else:
                return MarketRegime.HIGH_VOL_NO_DIRECTION
        elif vol_state == VolatilityState.LOW:
            if trend_dir == TrendDirection.UP:
                return MarketRegime.LOW_VOL_UP
            elif trend_dir == TrendDirection.DOWN:
                return MarketRegime.LOW_VOL_DOWN
            else:
                return MarketRegime.LOW_VOL_NO_DIRECTION
        else:  # NORMAL
            # 정상 변동성: 방향성에 따라 분류
            # [FIX] NORMAL_VOL_* 레짐을 추가하여 중간 파라미터 적용
            if trend_dir == TrendDirection.UP:
                return MarketRegime.NORMAL_VOL_UP
            elif trend_dir == TrendDirection.DOWN:
                return MarketRegime.NORMAL_VOL_DOWN
            else:
                return MarketRegime.NORMAL_VOL_NO_DIRECTION
    
    def _calculate_confidence(
        self,
        atr_ratio: float,
        adx: float,
        vol_state: VolatilityState,
        trend_dir: TrendDirection,
    ) -> float:
        """분류 신뢰도 계산 (0~1)"""
        confidence = 0.5  # 기본 신뢰도
        
        # 변동성이 명확할 때 신뢰도 증가
        if vol_state == VolatilityState.HIGH and atr_ratio > self.vol_high_threshold * 1.5:
            confidence += 0.2
        elif vol_state == VolatilityState.LOW and atr_ratio < self.vol_low_threshold * 0.5:
            confidence += 0.2
        elif vol_state == VolatilityState.NORMAL:
            # 경계 상태는 신뢰도 소폭 감소
            confidence -= 0.1
        
        # 추세가 명확할 때 신뢰도 증가
        if trend_dir != TrendDirection.NEUTRAL:
            if adx > self.adx_trend_threshold * 1.5:
                confidence += 0.3
            elif adx > self.adx_trend_threshold:
                confidence += 0.15
        else:
            # 횡보 레짐은 ADX가 낮을수록 신뢰도가 높아야 함
            if adx < self.adx_weak_threshold:
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _apply_option_sentiment_filter(
        self,
        confidence: float,
        trend_dir: TrendDirection,
        skew: float,
        volume_pcr: float,
        oi_pcr: float,
    ) -> float:
        """
        옵션 센티먼트 필터링: 기술적 신호와 옵션 센티먼트 일치 시 신뢰도 보정
        
        Args:
            confidence: 기술적 분류 신뢰도
            trend_dir: 기술적 추세 방향
            skew: 옵션 Skew (양수=강세, 음수=약세)
            volume_pcr: 옵션 Volume PCR (낮을수록 강세)
            oi_pcr: 옵션 OI PCR (낮을수록 강세)
            
        Returns:
            보정된 신뢰도
        """
        try:
            # 옵션 센티먼트 방향 판단 (간단 규칙)
            # Skew 양수 + PCR 낮음 = 강세
            # Skew 음수 + PCR 높음 = 약세
            sentiment_bullish = (skew > 0) and (volume_pcr < 1.0) and (oi_pcr < 1.0)
            sentiment_bearish = (skew < 0) and (volume_pcr > 1.0) and (oi_pcr > 1.0)
            
            # 기술적 상승 + 옵션 강세 → 신뢰도 상향
            if trend_dir == TrendDirection.UP and sentiment_bullish:
                adjusted_confidence = min(1.0, confidence + self.sentiment_confidence_boost)
                logger.info(
                    "[MarketRegime] 기술적 상승 + 옵션 강세: 신뢰도 %.2f → %.2f",
                    confidence, adjusted_confidence
                )
                return adjusted_confidence
            
            # 기술적 하락 + 옵션 약세 → 신뢰도 상향
            elif trend_dir == TrendDirection.DOWN and sentiment_bearish:
                adjusted_confidence = min(1.0, confidence + self.sentiment_confidence_boost)
                logger.info(
                    "[MarketRegime] 기술적 하락 + 옵션 약세: 신뢰도 %.2f → %.2f",
                    confidence, adjusted_confidence
                )
                return adjusted_confidence
            
            # 기술적 상승 + 옵션 약세 → 신뢰도 하향
            elif trend_dir == TrendDirection.UP and sentiment_bearish:
                adjusted_confidence = max(0.0, confidence - self.sentiment_confidence_penalty)
                logger.info(
                    "[MarketRegime] 기술적 상승 + 옵션 약세: 신뢰도 %.2f → %.2f (보정)",
                    confidence, adjusted_confidence
                )
                return adjusted_confidence
            
            # 기술적 하락 + 옵션 강세 → 신뢰도 하향
            elif trend_dir == TrendDirection.DOWN and sentiment_bullish:
                adjusted_confidence = max(0.0, confidence - self.sentiment_confidence_penalty)
                logger.info(
                    "[MarketRegime] 기술적 하락 + 옵션 강세: 신뢰도 %.2f → %.2f (보정)",
                    confidence, adjusted_confidence
                )
                return adjusted_confidence
            
            # 기타 경우 (중립, 혼합 신호) → 신뢰도 유지
            return confidence
            
        except Exception as e:
            logger.warning("[MarketRegime] 옵션 센티먼트 필터링 실패: %s", e)
            return confidence
    
    def get_suitable_strategy(self, regime: MarketRegime) -> str:
        """레짐에 적합한 전략 반환"""
        strategy_map = {
            MarketRegime.HIGH_VOL_NO_DIRECTION: "짧은 스캘핑 (Short Scalping)",
            MarketRegime.HIGH_VOL_UP: "돌파추종 (Breakout Following)",
            MarketRegime.HIGH_VOL_DOWN: "공매도 돌파추종 (Short Breakout Following)",
            MarketRegime.LOW_VOL_NO_DIRECTION: "Mean Reversion",
            MarketRegime.LOW_VOL_UP: "스윙 트레이딩 (Swing Trading)",
            MarketRegime.LOW_VOL_DOWN: "스윙 숏 (Swing Short)",
            MarketRegime.OPENING_EVENT: "장초반 스캘핑 (Opening Scalping)",
            MarketRegime.NEWS_EVENT: "뉴스 트레이딩 (News Trading) 또는 관망",
        }
        return strategy_map.get(regime, "기본 전략")
    
    def get_regime_description(self, regime: MarketRegime) -> str:
        """레짐 설명 반환"""
        descriptions = {
            MarketRegime.HIGH_VOL_NO_DIRECTION: "고변동 횡보 - 흔들기 심함, 짧은 스캘핑 적합",
            MarketRegime.HIGH_VOL_UP: "고변동 상승 - 강한 상승 추세, 돌파추종 적합",
            MarketRegime.HIGH_VOL_DOWN: "고변동 하락 - 강한 하락 추세, 공매도 돌파추종 적합",
            MarketRegime.LOW_VOL_NO_DIRECTION: "저변동 횡보 - 조용한 횡보, Mean Reversion 적합",
            MarketRegime.LOW_VOL_UP: "저변동 상승 - 느린 상승, 스윙 트레이딩 적합",
            MarketRegime.LOW_VOL_DOWN: "저변동 하락 - 느린 하락, 스윙 숏 적합",
            MarketRegime.OPENING_EVENT: "장초반 이벤트장 - 변동성 확대, 장초반 스캘핑 적합",
            MarketRegime.NEWS_EVENT: "뉴스 급등락장 - 급격한 변동, 뉴스 트레이딩 또는 관망",
        }
        return descriptions.get(regime, "알 수 없는 레짐")


# 테스트 코드
if __name__ == "__main__":
    # 더미 데이터 생성
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2024-01-01 09:00", periods=n, freq="1min")
    
    # 횡보 데이터
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.random.rand(n) * 0.2
    low = close - np.random.rand(n) * 0.2
    volume = np.random.randint(1000, 10000, n)
    
    df = pd.DataFrame({
        'Open': close,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)
    
    # 분류기 생성
    classifier = MarketRegimeClassifier()
    
    # 분류
    state = classifier.classify(df)
    
    if state:
        print(f"시장 상태: {state.regime.value}")
        print(f"변동성: {state.volatility_state.value} (ATR: {state.atr:.2f}, 비율: {state.atr_ratio:.4f})")
        print(f"추세: {state.trend_direction.value} (ADX: {state.adx:.2f})")
        print(f"설명: {classifier.get_regime_description(state.regime)}")
        print(f"적합 전략: {classifier.get_suitable_strategy(state.regime)}")
        print(f"신뢰도: {state.confidence:.2f}")
