"""regime_param_mapper.py — 시장 레짐 기반 ZigZag 파라미터 매핑
=================================================================

MarketRegimeClassifier 출력(MarketState)을 받아 레짐에 최적화된
ZigZag 파라미터 프로파일을 AdaptiveParameterAdjuster에 주입한다.

설계 원칙
---------
1. 레짐 프로파일이 base_params를 재정의  →  AdaptiveParameterAdjuster가 ±30% 미세조정
2. 신뢰도(confidence) 낮으면 프로파일 전환 억제  →  히스테리시스로 깜빡임 방지
3. 레짐 전환 이력 3-of-5 다수결  →  일시적 오분류 무시
4. 피드백 루프: 최근 피봇 품질(lag, success_rate)로 프로파일 내 미세조정 강도 제어

통합 방법
---------
AdaptiveZigZag.update() 내부의 파라미터 조정 블록을 아래처럼 교체한다.

    adjusted_params = self._param_adjuster.get_adaptive_params(
        recent_df, current_time=bar_time
    )
    ↓
    market_state = self._regime_mapper.get_adaptive_params_for_regime(
        recent_df,
        current_time=bar_time,
        recent_lags=self._pivot_lag_history,
        success_rates=self._pivot_success_history,
    )
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# 로컬 임포트 (같은 패키지)
# ---------------------------------------------------------------------------
try:
    from .market_regime_classifier import (
        MarketRegime,
        MarketState,
        MarketRegimeClassifier,
        VolatilityState,
        TrendDirection,
    )
    from .adaptive_parameter_adjuster import (
        AdaptiveParameterAdjuster,
        AdaptiveParams,
    )
except ImportError:
    try:
        from services.market_regime_classifier import (
            MarketRegime,
            MarketState,
            MarketRegimeClassifier,
            VolatilityState,
            TrendDirection,
        )
        from indicators.adaptive_parameter_adjuster import (
            AdaptiveParameterAdjuster,
            AdaptiveParams,
        )
    except ImportError:
        from market_regime_classifier import (
            MarketRegime,
            MarketState,
            MarketRegimeClassifier,
            VolatilityState,
            TrendDirection,
        )
        from adaptive_parameter_adjuster import (
            AdaptiveParameterAdjuster,
            AdaptiveParams,
        )

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. 레짐별 파라미터 프로파일
# ===========================================================================

@dataclass
class RegimeProfile:
    """레짐에 최적화된 ZigZag 파라미터 범위.

    min/max 값을 갖는 이유:
        - confidence 수준에 따라 [min, max] 내에서 선형 보간
        - confidence 1.0 → max(가장 공격적/보수적), 0.0 → base_params 유지
    """
    # ATR 배수 (임계값 폭 결정)
    atr_multiplier_min: float
    atr_multiplier_max: float

    # 확정 대기 봉수 (허위 피봇 억제)
    confirmation_bars_min: int
    confirmation_bars_max: int

    # 최소 파동 크기 (ATR 배수)
    min_wave_atr_ratio_min: float
    min_wave_atr_ratio_max: float

    # 최소 임계값 (%)
    pivot_threshold_min_pct_min: float
    pivot_threshold_min_pct_max: float

    # 설명 (로깅용)
    description: str = ""

    def interpolate(self, confidence: float) -> Dict[str, float]:
        """confidence [0, 1]에 따라 파라미터를 선형 보간하여 반환."""
        t = float(np.clip(confidence, 0.0, 1.0))
        return {
            "atr_multiplier": self.atr_multiplier_min + t * (
                self.atr_multiplier_max - self.atr_multiplier_min
            ),
            "confirmation_bars": int(round(
                self.confirmation_bars_min + t * (
                    self.confirmation_bars_max - self.confirmation_bars_min
                )
            )),
            "min_wave_atr_ratio": self.min_wave_atr_ratio_min + t * (
                self.min_wave_atr_ratio_max - self.min_wave_atr_ratio_min
            ),
            "pivot_threshold_min_pct": self.pivot_threshold_min_pct_min + t * (
                self.pivot_threshold_min_pct_max - self.pivot_threshold_min_pct_min
            ),
        }


# ---------------------------------------------------------------------------
# 레짐별 프로파일 테이블
# ---------------------------------------------------------------------------
REGIME_PROFILES: Dict[MarketRegime, RegimeProfile] = {
    # -------------------------------------------------------------------
    # 고변동 추세 레짐: 빠른 확정, 중간 임계값 → 돌파추종
    # KP200 선물 기준 파라미터 상향 조정 (피봇 급증 방지)
    # -------------------------------------------------------------------
    MarketRegime.HIGH_VOL_UP: RegimeProfile(
        atr_multiplier_min=2.0, atr_multiplier_max=3.0,
        confirmation_bars_min=1, confirmation_bars_max=1,
        min_wave_atr_ratio_min=2.0, min_wave_atr_ratio_max=3.0,   # 0.8~1.2 → 2.0~3.0 (KP200 session table 2.0~4.0)
        pivot_threshold_min_pct_min=0.30, pivot_threshold_min_pct_max=0.40,  # 0.20 → 0.30~0.40
        description="고변동 상승: 돌파추종 — 빠른 확정, 중간 임계값",
    ),
    MarketRegime.HIGH_VOL_DOWN: RegimeProfile(
        atr_multiplier_min=2.0, atr_multiplier_max=3.0,
        confirmation_bars_min=1, confirmation_bars_max=1,
        min_wave_atr_ratio_min=2.0, min_wave_atr_ratio_max=3.0,   # 0.8~1.2 → 2.0~3.0
        pivot_threshold_min_pct_min=0.30, pivot_threshold_min_pct_max=0.40,  # 0.20 → 0.30~0.40
        description="고변동 하락: 공매도 돌파추종 — 빠른 확정",
    ),

    # -------------------------------------------------------------------
    # 정상 변동성 추세 레짐: 중간 파라미터 → 균형 잡힌 피봇 탐색
    # HIGH_VOL과 LOW_VOL의 중간값
    # -------------------------------------------------------------------
    MarketRegime.NORMAL_VOL_UP: RegimeProfile(
        atr_multiplier_min=1.5, atr_multiplier_max=2.0,
        confirmation_bars_min=1, confirmation_bars_max=2,
        min_wave_atr_ratio_min=1.5, min_wave_atr_ratio_max=2.0,
        pivot_threshold_min_pct_min=0.25, pivot_threshold_min_pct_max=0.35,
        description="정상 변동성 상승: 균형 잡힌 피봇 탐색",
    ),
    MarketRegime.NORMAL_VOL_DOWN: RegimeProfile(
        atr_multiplier_min=1.5, atr_multiplier_max=2.0,
        confirmation_bars_min=1, confirmation_bars_max=2,
        min_wave_atr_ratio_min=1.5, min_wave_atr_ratio_max=2.0,
        pivot_threshold_min_pct_min=0.25, pivot_threshold_min_pct_max=0.35,
        description="정상 변동성 하락: 균형 잡힌 피봇 탐색",
    ),
    MarketRegime.NORMAL_VOL_NO_DIRECTION: RegimeProfile(
        atr_multiplier_min=1.5, atr_multiplier_max=2.5,
        confirmation_bars_min=2, confirmation_bars_max=3,
        min_wave_atr_ratio_min=1.5, min_wave_atr_ratio_max=2.5,
        pivot_threshold_min_pct_min=0.25, pivot_threshold_min_pct_max=0.40,
        description="정상 변동성 횡보: 중간 수준의 임계값",
    ),

    # -------------------------------------------------------------------
    # 고변동 무방향: 흔들기 노이즈 최대 억제
    # confirmation_bars 3~4 → 2~3 (실시간 확정 지연 방지)
    # -------------------------------------------------------------------
    MarketRegime.HIGH_VOL_NO_DIRECTION: RegimeProfile(
        atr_multiplier_min=3.0, atr_multiplier_max=4.5,
        confirmation_bars_min=2, confirmation_bars_max=3,   # 3~4 → 2~3
        min_wave_atr_ratio_min=2.5, min_wave_atr_ratio_max=3.5,   # 1.5~2.5 → 2.5~3.5
        pivot_threshold_min_pct_min=0.40, pivot_threshold_min_pct_max=0.50,  # 0.35 → 0.40~0.50
        description="고변동 횡보: 흔들기 — 임계값·대기봉 최대화",
    ),

    # -------------------------------------------------------------------
    # 저변동 횡보: Mean Reversion — KP200 선물에 맞는 파라미터
    # KOSPI 현금(ATR≈8pt)와 달리 KP200 선물(ATR≈1.5pt)에서는 0.4~0.7×ATR=0.6~1.0pt로 틱 노이즈 구분 불가
    # KP200 기준으로 상향 조정: 지지/저항 밴드 반전 매매용
    # -------------------------------------------------------------------
    MarketRegime.LOW_VOL_NO_DIRECTION: RegimeProfile(
        atr_multiplier_min=0.8, atr_multiplier_max=1.2,
        confirmation_bars_min=2, confirmation_bars_max=3,   # 3~4 → 2~3
        min_wave_atr_ratio_min=1.5, min_wave_atr_ratio_max=2.5,   # 0.4~0.7 → 1.5~2.5 (KP200 기준)
        pivot_threshold_min_pct_min=0.25, pivot_threshold_min_pct_max=0.35,  # 0.10 → 0.25~0.35
        description="저변동 횡보: Mean Reversion — KP200 선물 기준 반전 포착",
    ),

    # -------------------------------------------------------------------
    # 저변동 추세: 스윙 트레이딩 — 중간 민감도
    # KP200 선물 기준 파라미터 상향 조정
    # -------------------------------------------------------------------
    MarketRegime.LOW_VOL_UP: RegimeProfile(
        atr_multiplier_min=1.0, atr_multiplier_max=1.8,
        confirmation_bars_min=2, confirmation_bars_max=3,
        min_wave_atr_ratio_min=1.5, min_wave_atr_ratio_max=2.5,   # 0.5~0.8 → 1.5~2.5
        pivot_threshold_min_pct_min=0.25, pivot_threshold_min_pct_max=0.35,  # 0.12 → 0.25~0.35
        description="저변동 상승: 스윙 트레이딩 — 중간 감도",
    ),
    MarketRegime.LOW_VOL_DOWN: RegimeProfile(
        atr_multiplier_min=1.0, atr_multiplier_max=1.8,
        confirmation_bars_min=2, confirmation_bars_max=3,
        min_wave_atr_ratio_min=1.5, min_wave_atr_ratio_max=2.5,   # 0.5~0.8 → 1.5~2.5
        pivot_threshold_min_pct_min=0.25, pivot_threshold_min_pct_max=0.35,  # 0.12 → 0.25~0.35
        description="저변동 하락: 스윙 숏 — 중간 감도",
    ),

    # -------------------------------------------------------------------
    # 장초반 이벤트: 최대 보수적 — 노이즈 억제 우선
    # KP200 선물 기준 파라미터 상향 조정
    # -------------------------------------------------------------------
    MarketRegime.OPENING_EVENT: RegimeProfile(
        atr_multiplier_min=4.0, atr_multiplier_max=8.0,
        confirmation_bars_min=1, confirmation_bars_max=2,
        min_wave_atr_ratio_min=2.5, min_wave_atr_ratio_max=4.0,   # 1.5~3.0 → 2.5~4.0
        pivot_threshold_min_pct_min=0.40, pivot_threshold_min_pct_max=0.50,  # 0.40 → 0.40~0.50
        description="장초반 이벤트: 임계값 최대화, 허위 피봇 차단",
    ),

    # -------------------------------------------------------------------
    # 뉴스 급등락: 가장 보수적 — 관망 모드
    # KP200 선물 기준 파라미터 상향 조정
    # -------------------------------------------------------------------
    MarketRegime.NEWS_EVENT: RegimeProfile(
        atr_multiplier_min=5.0, atr_multiplier_max=8.0,
        confirmation_bars_min=1, confirmation_bars_max=1,
        min_wave_atr_ratio_min=3.0, min_wave_atr_ratio_max=4.5,   # 2.0~4.0 → 3.0~4.5
        pivot_threshold_min_pct_min=0.50, pivot_threshold_min_pct_max=0.60,  # 0.50 → 0.50~0.60
        description="뉴스 급등락: 최대 보수적 — 관망·보호",
    ),
}


# ===========================================================================
# 2. RegimeParamMapper
# ===========================================================================

class RegimeParamMapper:
    """MarketState → ZigZag 파라미터 변환기.

    AdaptiveZigZag.update() 내부에서 기존 AdaptiveParameterAdjuster를
    대체(혹은 래핑)하는 핵심 컴포넌트.

    파라미터 결정 흐름
    -----------------
    1. MarketRegimeClassifier.classify(df) → MarketState
    2. 레짐 이력 다수결 (3-of-5) → 안정된 레짐 결정
    3. REGIME_PROFILES[regime].interpolate(confidence) → 프로파일 파라미터
    4. AdaptiveParameterAdjuster.get_adaptive_params(df, ...) → 실시간 미세조정
    5. 두 결과를 가중 합산 (profile_weight : adjuster_weight)
    6. 클램핑 후 반환
    """

    # 히스테리시스: 신뢰도가 이 값 미만이면 이전 레짐 유지
    CONFIDENCE_THRESHOLD: float = 0.55

    # 레짐 전환 이력 다수결 윈도우
    REGIME_HISTORY_LEN: int = 5
    REGIME_MAJORITY: int = 3   # 5봉 중 3번 이상 같은 레짐 → 전환

    # 프로파일 vs 실시간 조정기 가중치
    PROFILE_WEIGHT: float = 0.65
    ADJUSTER_WEIGHT: float = 0.35

    def __init__(
        self,
        classifier: Optional[MarketRegimeClassifier] = None,
        adjuster: Optional[AdaptiveParameterAdjuster] = None,
        base_params: Optional[AdaptiveParams] = None,
        config: Optional[Dict] = None,
        symbol: str = "futures",
        classify_interval_bars: int = 10,   # 분류 주기 (봉)
    ):
        """
        Args:
            classifier: MarketRegimeClassifier 인스턴스 (None이면 내부 생성)
            adjuster:   AdaptiveParameterAdjuster 인스턴스 (None이면 내부 생성)
            base_params: 기준 파라미터 (None이면 심볼 기본값 사용)
            config:      설정 딕셔너리
            symbol:      "futures" 또는 "kospi"
            classify_interval_bars: 레짐 재분류 주기 (성능 최적화)
        """
        self._symbol = symbol
        self._classify_interval = classify_interval_bars

        # --- Classifier ---
        self._classifier = classifier or MarketRegimeClassifier()

        # --- Base params ---
        if base_params is None:
            _defaults = {"futures": (0.8, 2), "kospi": (0.5, 3)}
            atr_m, conf_b = _defaults.get(symbol, (0.8, 2))
            base_params = AdaptiveParams(
                atr_multiplier=atr_m,
                confirmation_bars=conf_b,
            )
        self._base_params = base_params

        # --- Adjuster ---
        if adjuster is None:
            _cfg = config or {"adaptive_indicator": {"zigzag": {}, "kospi_zigzag": {}, "futures_zigzag": {}}}
            adjuster = AdaptiveParameterAdjuster(
                base_params=base_params,
                config=_cfg,
                symbol=symbol,
            )
        self._adjuster = adjuster

        # --- 상태 ---
        self._current_regime: MarketRegime = MarketRegime.LOW_VOL_NO_DIRECTION
        self._current_state: Optional[MarketState] = None
        self._regime_history: Deque[MarketRegime] = deque(maxlen=self.REGIME_HISTORY_LEN)
        self._stable_regime: MarketRegime = MarketRegime.LOW_VOL_NO_DIRECTION
        self._last_classify_bar: int = -1
        self._bar_counter: int = 0

        # 레짐 전환 로그 (텔레그램 송출용)
        self._last_logged_regime: Optional[MarketRegime] = None

        logger.info(
            "[RegimeParamMapper] 초기화 완료: symbol=%s, classify_interval=%d",
            symbol, classify_interval_bars,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def classify(self, df: pd.DataFrame, current_time: Optional[datetime] = None) -> Optional[MarketState]:
        """레짐 분류 후 MarketState 반환 (읽기 전용 정보).
        
        [Layer C] 레짐 레이블만 제공. ZigZag 파라미터에는 관여하지 않는다.
        
        Args:
            df: 최근 N봉 OHLCV DataFrame
            current_time: 현재 봉 시각 (None이면 현재 시각)
            
        Returns:
            MarketState (레짐 분류 결과)
        """
        self._bar_counter += 1

        # 레짐 분류 (호출 횟수 기준 주기 제한)
        if self._should_classify():
            self._classify(df)

        return self._current_state

    @property
    def current_state(self) -> Optional[MarketState]:
        """최신 MarketState 반환 (외부 참조용)."""
        return self._current_state

    @property
    def stable_regime(self) -> MarketRegime:
        """히스테리시스 적용 후 안정된 레짐."""
        return self._stable_regime

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _should_classify(self) -> bool:
        """레짐 재분류 타이밍 결정.
        
        [FIX] ATR spike 감지 시 즉시 재분류 (변동성 급변 대응)
        """
        if self._last_classify_bar < 0:
            return True
        
        # ATR spike 감지 시 즉시 재분류
        if self._current_state is not None and hasattr(self._current_state, 'atr_spike_detected'):
            if self._current_state.atr_spike_detected:
                logger.debug("[RegimeParamMapper] ATR spike 감지, 즉시 재분류")
                return True
        
        return (self._bar_counter - self._last_classify_bar) >= self._classify_interval

    def _classify(self, df: pd.DataFrame) -> None:
        """레짐 분류 및 히스테리시스 적용."""
        # DataFrame 컬럼 정규화 (대소문자 통일)
        col_map = {c.lower(): c for c in df.columns}
        renamed = df.rename(columns={v: k for k, v in col_map.items()})

        # High/Low/Close 컬럼명 매핑
        for src, dst in [("high", "High"), ("low", "Low"), ("close", "Close")]:
            if src in renamed.columns and dst not in renamed.columns:
                renamed[dst] = renamed[src]

        state = self._classifier.classify(renamed)
        self._last_classify_bar = self._bar_counter

        if state is None:
            return

        self._current_state = state
        self._current_regime = state.regime
        self._regime_history.append(state.regime)

        # 히스테리시스: confidence 낮거나 다수결 미충족 시 이전 레짐 유지
        if state.confidence >= self.CONFIDENCE_THRESHOLD:
            if self._majority_vote():
                if self._stable_regime != state.regime:
                    logger.info(
                        "[RegimeParamMapper] 레짐 전환: %s → %s (confidence=%.2f)",
                        self._stable_regime.value, state.regime.value, state.confidence,
                    )
                self._stable_regime = state.regime
        else:
            logger.debug(
                "[RegimeParamMapper] 신뢰도 낮음 (%.2f), 레짐 유지: %s",
                state.confidence, self._stable_regime.value,
            )

    def _majority_vote(self) -> bool:
        """최근 이력에서 현재 레짐이 다수결을 충족하는지 확인."""
        if len(self._regime_history) < self.REGIME_MAJORITY:
            return True  # 이력 부족 → 즉시 전환 허용
        count = sum(1 for r in self._regime_history if r == self._current_regime)
        return count >= self.REGIME_MAJORITY

    def _get_profile_params(self) -> Dict[str, float]:
        """안정된 레짐의 프로파일에서 파라미터 보간.

        [FIX] NORMAL 변동성이 귀속되는 HIGH_VOL_UP/DOWN 레짐 프로파일이 없을 때
        LOW_VOL_NO_DIRECTION을 안전한 fallback으로 사용한다.
        """
        profile = REGIME_PROFILES.get(self._stable_regime)
        if profile is None:
            # 알 수 없는 레짐(예: NORMAL 귀속 케이스) → 안전한 보수적 fallback
            # [FIX] 기존 base_params 반환 대신 LOW_VOL_NO_DIRECTION 프로파일 사용
            fallback_profile = REGIME_PROFILES.get(MarketRegime.LOW_VOL_NO_DIRECTION)
            if fallback_profile is not None:
                logger.debug(
                    "[RegimeParamMapper] 프로파일 없음(regime=%s), LOW_VOL_NO_DIRECTION fallback 사용",
                    self._stable_regime.value,
                )
                return fallback_profile.interpolate(0.5)
            return {
                "atr_multiplier": self._base_params.atr_multiplier,
                "confirmation_bars": float(self._base_params.confirmation_bars),
                "min_wave_atr_ratio": self._base_params.min_wave_atr_ratio,
                "pivot_threshold_min_pct": self._base_params.pivot_threshold_min_pct,
            }

        confidence = (
            self._current_state.confidence
            if self._current_state is not None
            else 0.5
        )
        return profile.interpolate(confidence)

    def _get_adjuster_ratio(
        self,
        df: pd.DataFrame,
        recent_lags: Optional[List[float]],
        success_rates: Optional[List[float]],
        current_time: Optional[datetime],
    ) -> Dict[str, float]:
        """AdaptiveParameterAdjuster의 조정 비율을 추출한다.

        [FIX] base_params를 직접 뮤테이션하지 않고, 원래 base_params 기준으로
        조정기를 실행한 뒤 (조정값 / base값) 비율만 반환한다.
        레짐 전환 시 atr_history 기준점이 바뀌지 않으므로 vol_ratio 오염이 없다.

        Returns:
            각 파라미터의 조정 비율 딕셔너리 (1.0 = 변화 없음, 0.8 = 20% 감소 등)
        """
        base_dict = {
            "atr_multiplier": self._base_params.atr_multiplier,
            "confirmation_bars": float(self._base_params.confirmation_bars),
            "min_wave_atr_ratio": self._base_params.min_wave_atr_ratio,
            "pivot_threshold_min_pct": self._base_params.pivot_threshold_min_pct,
        }
        try:
            adj = self._adjuster.get_adaptive_params(
                df,
                recent_lags=recent_lags,
                success_rates=success_rates,
                current_time=current_time,
            )
            ratios: Dict[str, float] = {}
            for key, base_val in base_dict.items():
                adj_val = float(adj.get(key, base_val))
                if abs(base_val) > 1e-9:
                    ratio = adj_val / base_val
                    # 조정 비율을 ±30% 이내로 클램핑 (AdaptiveParameterAdjuster 보장값과 일치)
                    ratio = float(np.clip(ratio, 0.7, 1.3))
                else:
                    ratio = 1.0
                ratios[key] = ratio
            return ratios
        except Exception as e:
            logger.debug("[RegimeParamMapper] adjuster 실행 실패, ratio=1.0 사용: %s", e)
            return {k: 1.0 for k in base_dict}

    def _apply_ratio(
        self,
        profile_params: Dict[str, float],
        ratios: Dict[str, float],
    ) -> Dict[str, float]:
        """프로파일 파라미터에 조정 비율을 곱하여 최종값을 생성한다."""
        result: Dict[str, float] = {}
        for key, p_val in profile_params.items():
            ratio = ratios.get(key, 1.0)
            result[key] = float(p_val) * ratio
        return result

    def _clamp(self, params: Dict[str, float]) -> Dict[str, float]:
        """파라미터 허용 범위 클램핑."""
        ranges = AdaptiveParameterAdjuster.PARAM_RANGES
        clamped = {}
        for key, val in params.items():
            lo, hi = ranges.get(key, (val, val))
            clamped[key] = float(np.clip(val, lo, hi))
        # confirmation_bars는 반드시 int
        clamped["confirmation_bars"] = int(round(clamped["confirmation_bars"]))
        return clamped


# [Layer C] patch_zigzag_with_regime 제거 - 레짐은 읽기 전용으로만 사용
# 레짐 레이블은 pipeline.py의 컨텍스트 빌드와 텔레그램 알림에서만 참조


# ===========================================================================
# 4. 독립 사용 예시 (테스트용)
# ===========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    # --- 더미 OHLCV 생성 ---
    rng = np.random.default_rng(42)
    n = 200

    # 고변동 상승 시뮬레이션
    price = 360.0
    prices = [price]
    for _ in range(n - 1):
        price += rng.normal(0.3, 1.5)
        prices.append(max(price, 300.0))

    close = np.array(prices)
    high  = close + rng.uniform(0.2, 1.5, n)
    low   = close - rng.uniform(0.2, 1.5, n)

    dates = pd.date_range("2024-01-15 09:00", periods=n, freq="1min")
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": rng.integers(1000, 9000, n)},
        index=dates,
    )

    # --- RegimeParamMapper 단독 테스트 ---
    mapper = RegimeParamMapper(symbol="futures", classify_interval_bars=5)

    print("\n=== RegimeParamMapper 파라미터 출력 (최근 50봉 기준) ===")
    for i in [50, 100, 150, 199]:
        params = mapper.get_adaptive_params(
            df.iloc[max(0, i - 50): i + 1].copy(),
            current_time=dates[i].to_pydatetime(),
        )
        state = mapper.current_state
        regime_str = mapper.stable_regime.value if state else "미분류"
        conf_str = f"{state.confidence:.2f}" if state else "N/A"
        print(
            f"  봉={i:3d}  regime={regime_str:25s}  conf={conf_str}"
            f"  → atr_mult={params['atr_multiplier']:.3f}"
            f"  conf_bars={params['confirmation_bars']}"
            f"  min_wave={params['min_wave_atr_ratio']:.3f}"
            f"  thr_min={params['pivot_threshold_min_pct']:.4f}%"
        )
