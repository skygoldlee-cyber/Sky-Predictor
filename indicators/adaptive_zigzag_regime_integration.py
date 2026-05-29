"""adaptive_zigzag_regime_integration.py
=============================================
AdaptiveZigZag ↔ RegimeParamMapper 통합 패치.

기존 코드 변경 최소화 원칙:
- adaptive_zigzag.py 수정 없이 외부에서 주입
- 기존 _param_adjuster 인터페이스를 유지하므로 update() 내부 로직 무변경

사용법:
    from adaptive_zigzag import AdaptiveZigZag, AdaptiveZigZagConfig
    from adaptive_zigzag_regime_integration import build_regime_aware_zigzag

    zz, mapper = build_regime_aware_zigzag(symbol="futures")
    for bar in stream:
        state = zz.update(bar.high, bar.low, bar.close, bar_time=bar.time)
        # mapper.current_state 로 현재 레짐 조회 가능
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .adaptive_zigzag import AdaptiveZigZag, AdaptiveZigZagConfig
    from .market_regime_classifier import MarketRegimeClassifier
    from .adaptive_parameter_adjuster import AdaptiveParams
    from .regime_param_mapper import RegimeParamMapper, patch_zigzag_with_regime
except ImportError:
    try:
        from indicators.adaptive_zigzag import AdaptiveZigZag, AdaptiveZigZagConfig
        from services.market_regime_classifier import MarketRegimeClassifier
        from indicators.adaptive_parameter_adjuster import AdaptiveParams
        from indicators.regime_param_mapper import RegimeParamMapper, patch_zigzag_with_regime
    except ImportError:
        from adaptive_zigzag import AdaptiveZigZag, AdaptiveZigZagConfig
        from market_regime_classifier import MarketRegimeClassifier
        from regime_param_mapper import RegimeParamMapper, patch_zigzag_with_regime

logger = logging.getLogger(__name__)


def _get_market_regime_class():
    """파일 상단 import에서 이미 로드된 MarketRegime 클래스를 반환한다.
    per-call import 없이 sys.modules에서 탐색하여 단일 해석 경로를 보장한다.
    """
    import sys
    for mod_name in (
        "services.market_regime_classifier",
        "indicators.market_regime_classifier",
        "market_regime_classifier",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "MarketRegime"):
            return mod.MarketRegime
    # 최후 폴백: regime_param_mapper가 이미 가져온 MarketRegime 재사용
    try:
        from regime_param_mapper import REGIME_PROFILES
        return type(next(iter(REGIME_PROFILES)))
    except Exception:
        pass
    # 재귀적 import (이 시점엔 반드시 한 경로가 동작함)
    try:
        from .market_regime_classifier import MarketRegime
        return MarketRegime
    except ImportError:
        from market_regime_classifier import MarketRegime
        return MarketRegime


# ===========================================================================
# 팩토리 함수
# ===========================================================================

def build_regime_aware_zigzag(
    symbol: str = "futures",
    zz_config: Optional[AdaptiveZigZagConfig] = None,
    classifier_kwargs: Optional[Dict] = None,
    classify_interval_bars: int = 10,
    config: Optional[Dict] = None,
) -> Tuple[AdaptiveZigZag, RegimeParamMapper]:
    """레짐 기반 파라미터 조정이 내장된 AdaptiveZigZag 인스턴스를 생성한다.

    Args:
        symbol:                  "futures" 또는 "kospi"
        zz_config:               AdaptiveZigZagConfig (None이면 심볼 기본값)
        classifier_kwargs:       MarketRegimeClassifier 생성자 kwargs
        classify_interval_bars:  레짐 재분류 주기 (봉)
        config:                  설정 딕셔너리 (None이면 기본값)

    Returns:
        (AdaptiveZigZag 인스턴스, RegimeParamMapper 인스턴스)
    """
    # 1. ZigZag 설정 (심볼별 기본값)
    if zz_config is None:
        if symbol == "futures":
            zz_config = AdaptiveZigZagConfig(
                atr_multiplier=1.5,
                atr_period=14,
                confirmation_bars=2,
                min_wave_bars=5,
                pivot_threshold_min_pct=0.3,
                pivot_threshold_max_pct=3.0,
                use_atr_based_filtering=True,
                min_wave_atr_ratio=1.0,
                pivot_lifecycle_log=True,
                pivot_lifecycle_log_prefix="KP200선물",
            )
        else:  # kospi
            zz_config = AdaptiveZigZagConfig(
                atr_multiplier=1.0,
                atr_period=14,
                confirmation_bars=3,
                min_wave_bars=7,
                pivot_threshold_min_pct=0.2,
                pivot_threshold_max_pct=2.0,
                use_atr_based_filtering=True,
                min_wave_atr_ratio=0.7,
                pivot_lifecycle_log=True,
                pivot_lifecycle_log_prefix="KOSPI200",
            )

    # 2. AdaptiveZigZag 생성
    zz = AdaptiveZigZag(config=zz_config)
    zz.set_symbol("KP200 선물" if symbol == "futures" else "KOSPI200")

    # 3. Classifier 생성
    ckw = classifier_kwargs or {}
    classifier = MarketRegimeClassifier(**ckw)

    # 4. RegimeParamMapper 주입
    mapper = patch_zigzag_with_regime(
        zz,
        classifier=classifier,
        config=config,
        symbol=symbol,
        classify_interval_bars=classify_interval_bars,
    )

    logger.info("[build_regime_aware_zigzag] 완료: symbol=%s", symbol)
    return zz, mapper


# ===========================================================================
# RegimeAwareZigZagRunner: 실시간 스트림 처리 헬퍼
# ===========================================================================

class RegimeAwareZigZagRunner:
    """봉 데이터를 스트리밍하며 레짐 기반 파라미터 조정을 자동으로 처리한다.

    사용 예:
        runner = RegimeAwareZigZagRunner(symbol="futures")
        runner.seed(open_price=360.0, swing_type="low")  # 장 시작 앵커

        for bar in live_feed:
            result = runner.on_bar(bar.high, bar.low, bar.close, bar.time, bar.open)
            if result.new_swing_signal != "none":
                send_signal(result)
    """

    # 피봇 품질 피드백 윈도우
    LAG_HISTORY_LEN: int = 20
    SUCCESS_HISTORY_LEN: int = 20

    def __init__(
        self,
        symbol: str = "futures",
        zz_config: Optional[AdaptiveZigZagConfig] = None,
        classify_interval_bars: int = 10,
    ):
        self._zz, self._mapper = build_regime_aware_zigzag(
            symbol=symbol,
            zz_config=zz_config,
            classify_interval_bars=classify_interval_bars,
        )
        self._symbol = symbol

        # 피봇 품질 피드백 버퍼
        self._lag_history: list = []
        self._success_history: list = []
        self._prev_swing_bar: int = -1
        self._bar_idx: int = 0

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def seed(self, open_price: float, swing_type: str = "low") -> None:
        """장 시작 앵커 피봇 주입 (seed_anchor 래퍼).

        [FIX] 파일 상단 import 블록과 동일한 다단계 폴백 방식으로 SwingType을 가져온다.
        기존 'from adaptive_zigzag import SwingType' 절대 경로는 패키지 환경에서 실패.
        """
        # SwingType은 파일 상단에서 이미 AdaptiveZigZag와 함께 import됨 — 재사용
        try:
            st = AdaptiveZigZag.__module__  # 실제 임포트 경로 확인용 (부작용 없음)
        except Exception:
            pass
        # AdaptiveZigZagConfig와 같은 모듈에서 SwingType을 가져옴
        zz_module = __import__(AdaptiveZigZag.__module__, fromlist=["SwingType"])
        SwingType = getattr(zz_module, "SwingType")
        st = SwingType.LOW if swing_type.lower() == "low" else SwingType.HIGH
        self._zz.seed_anchor(price=open_price, swing_type=st)
        logger.info("[RegimeAwareZigZagRunner] seed_anchor: price=%.2f type=%s", open_price, swing_type)

    def reset_session(self) -> None:
        """장 재시작(새 세션) 시 호출."""
        self._zz.reset_for_new_session()
        self._bar_idx = 0
        self._prev_swing_bar = -1
        self._lag_history.clear()
        self._success_history.clear()

    def on_bar(
        self,
        high: float,
        low: float,
        close: float,
        bar_time=None,
        open_: float = 0.0,
        volume: float = 1.0,
    ):
        """새 봉 처리.

        Args:
            high, low, close: 봉 OHLC
            bar_time:         봉 시각 (pandas Timestamp 또는 str)
            open_:            봉 시가 (없으면 close로 대체)
            volume:           봉 거래량

        Returns:
            ZigZagState (피봇·구조·피보나치 등 포함)

        Note:
            [FIX] OHLCV 버퍼(_push/_build_df)를 제거했다.
            AdaptiveParameterAdjuster(RegimeParamMapper 경유)는 ZigZag.update()
            내부에서 자체 수집한 최근 50봉을 사용하므로 Runner 측 버퍼는 불필요했다.
        """
        self._bar_idx += 1

        # ZigZag 업데이트 (내부에서 _param_adjuster = mapper가 호출됨)
        state = self._zz.update(high, low, close, bar_time=bar_time, open=open_, volume=volume)

        # 피봇 품질 피드백 수집
        self._collect_feedback(state)

        return state

    @property
    def current_regime(self):
        """현재 안정 레짐 반환."""
        return self._mapper.stable_regime

    @property
    def current_market_state(self):
        """최신 MarketState 반환."""
        return self._mapper.current_state

    @property
    def zigzag(self) -> AdaptiveZigZag:
        return self._zz

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _collect_feedback(self, state) -> None:
        """피봇 확정 시 지연봉수·성공 여부를 피드백 버퍼에 추가."""
        if state.new_swing_signal == "none":
            return

        # lag 봉수 계산
        lag = state.last_swing_high_lag_bars if state.new_swing_signal == "new_high" \
            else state.last_swing_low_lag_bars
        self._lag_history.append(float(lag))
        if len(self._lag_history) > self.LAG_HISTORY_LEN:
            self._lag_history.pop(0)

        # 성공률: 레짐 방향과 피봇 방향 일치 여부
        # [FIX] _collect_feedback 내 per-call import → 파일 상단 import 재사용
        MR = _get_market_regime_class()
        regime = self._mapper.stable_regime
        if regime in (MR.HIGH_VOL_UP, MR.LOW_VOL_UP):
            is_success = state.new_swing_signal == "new_high"
        elif regime in (MR.HIGH_VOL_DOWN, MR.LOW_VOL_DOWN):
            is_success = state.new_swing_signal == "new_low"
        else:
            is_success = True  # 횡보·이벤트 레짐에선 어느 방향이든 OK

        self._success_history.append(1.0 if is_success else 0.0)
        if len(self._success_history) > self.SUCCESS_HISTORY_LEN:
            self._success_history.pop(0)


# ===========================================================================
# 빠른 테스트
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    rng = np.random.default_rng(0)
    n = 300
    price = 360.0
    prices = [price]
    for _ in range(n - 1):
        price += rng.normal(0.2, 1.2)
        prices.append(max(price, 300.0))

    close = np.array(prices)
    high = close + rng.uniform(0.1, 1.2, n)
    low  = close - rng.uniform(0.1, 1.2, n)
    vol  = rng.integers(1000, 8000, n).astype(float)
    dates = pd.date_range("2024-03-01 09:01", periods=n, freq="1min")

    runner = RegimeAwareZigZagRunner(symbol="futures", classify_interval_bars=10)
    runner.seed(open_price=close[0], swing_type="low")

    pivot_count = 0
    for i in range(n):
        state = runner.on_bar(
            high[i], low[i], close[i],
            bar_time=dates[i], open_=close[i], volume=float(vol[i]),
        )
        if state.new_swing_signal != "none":
            pivot_count += 1
            print(
                f"  봉={i:3d} [{dates[i].strftime('%H:%M')}] "
                f"signal={state.new_swing_signal:8s} "
                f"regime={runner.current_regime.value:25s} "
                f"conf_bars={runner.zigzag.config.confirmation_bars}  "
                f"atr_mult={runner.zigzag.config.atr_multiplier:.2f}"
            )

    print(f"\n총 확정 피봇: {pivot_count}개")
