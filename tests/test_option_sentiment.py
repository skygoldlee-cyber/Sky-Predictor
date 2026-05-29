"""옵션 센티먼트 분석기 단위 테스트.

Skew, Volume PCR, OI PCR을 조합하여 장의 방향성을 진단하는 로직을 검증한다.
"""

from __future__ import annotations

from typing import Dict, Any


# ──────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────

def _default_config() -> Dict[str, Any]:
    """기본 설정을 반환한다."""
    return {
        "skew_bullish_threshold": 0.05,
        "skew_bearish_threshold": -0.05,
        "volume_pcr_bullish_threshold": 0.8,
        "volume_pcr_bearish_threshold": 1.2,
        "oi_pcr_bullish_threshold": 0.9,
        "oi_pcr_bearish_threshold": 1.1,
        "skew_weight": 0.3,
        "volume_pcr_weight": 0.35,
        "oi_pcr_weight": 0.35,
        "min_confidence": 0.6,
        "direction_change_alert": True,
        "confidence_spike_alert": True,
        "confidence_spike_threshold": 0.8,
    }


# ──────────────────────────────────────────────────────────────────
# 테스트 1: 기본 분석 로직
# ──────────────────────────────────────────────────────────────────

def test_basic_bullish_analysis() -> None:
    """모든 지표가 강세 신호일 때 상승 방향성이 결정되는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 모든 지표 강세
    signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)

    assert signal.direction.value == "bullish"
    assert signal.confidence > 0.6
    assert signal.skew_signal == "bullish"
    assert signal.volume_pcr_signal == "bullish"
    assert signal.oi_pcr_signal == "bullish"


def test_basic_bearish_analysis() -> None:
    """모든 지표가 약세 신호일 때 하락 방향성이 결정되는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 모든 지표 약세
    signal = analyzer.analyze(skew=-0.06, volume_pcr=1.25, oi_pcr=1.15)

    assert signal.direction.value == "bearish"
    assert signal.confidence > 0.6
    assert signal.skew_signal == "bearish"
    assert signal.volume_pcr_signal == "bearish"
    assert signal.oi_pcr_signal == "bearish"


def test_mixed_signals_neutral() -> None:
    """지표가 섞여 있을 때 중립 방향성이 결정되는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 지표 섞임 (skew 강세, volume_pcr 약세, oi_pcr 중립)
    signal = analyzer.analyze(skew=0.06, volume_pcr=1.25, oi_pcr=1.0)

    assert signal.direction.value == "neutral"


def test_low_confidence_neutral() -> None:
    """신뢰도가 낮을 때 중립 방향성이 결정되는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 약한 신호
    signal = analyzer.analyze(skew=0.02, volume_pcr=0.95, oi_pcr=0.95)

    assert signal.direction.value == "neutral"
    assert signal.confidence < config.min_confidence


# ──────────────────────────────────────────────────────────────────
# 테스트 2: 개별 지표 분류
# ──────────────────────────────────────────────────────────────────

def test_skew_classification() -> None:
    """Skew 분류 로직 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 강세
    signal = analyzer.analyze(skew=0.06, volume_pcr=1.0, oi_pcr=1.0)
    assert signal.skew_signal == "bullish"

    # 약세
    signal = analyzer.analyze(skew=-0.06, volume_pcr=1.0, oi_pcr=1.0)
    assert signal.skew_signal == "bearish"

    # 중립
    signal = analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.0)
    assert signal.skew_signal == "neutral"


def test_volume_pcr_classification() -> None:
    """Volume PCR 분류 로직 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 강세 (낮을수록 강세)
    signal = analyzer.analyze(skew=0.0, volume_pcr=0.75, oi_pcr=1.0)
    assert signal.volume_pcr_signal == "bullish"

    # 약세 (높을수록 약세)
    signal = analyzer.analyze(skew=0.0, volume_pcr=1.25, oi_pcr=1.0)
    assert signal.volume_pcr_signal == "bearish"

    # 중립
    signal = analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.0)
    assert signal.volume_pcr_signal == "neutral"


def test_oi_pcr_classification() -> None:
    """OI PCR 분류 로직 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 강세 (낮을수록 강세)
    signal = analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=0.85)
    assert signal.oi_pcr_signal == "bullish"

    # 약세 (높을수록 약세)
    signal = analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.15)
    assert signal.oi_pcr_signal == "bearish"

    # 중립
    signal = analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.0)
    assert signal.oi_pcr_signal == "neutral"


# ──────────────────────────────────────────────────────────────────
# 테스트 3: 이벤트 감지
# ──────────────────────────────────────────────────────────────────

def test_direction_change_event() -> None:
    """방향성 전환 이벤트 감지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 첫 분석 (중립)
    signal = analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.0)
    assert signal.event_type == "none"

    # 상승으로 전환
    signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    assert signal.event_type == "direction_change"
    assert signal.direction.value == "bullish"
    assert signal.prev_direction.value == "neutral"


def test_confidence_spike_event() -> None:
    """신뢰도 급증 이벤트 감지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 낮은 신뢰도 (약세 - 낮은 신뢰도)
    signal = analyzer.analyze(skew=-0.02, volume_pcr=1.15, oi_pcr=1.05)
    assert signal.event_type == "none"
    assert signal.confidence < config.confidence_spike_threshold

    # 신뢰도 급증 (강세로 전환)
    signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    # 방향성 전환이 우선되므로 direction_change가 감지됨
    assert signal.event_type in ("direction_change", "confidence_spike")
    assert signal.confidence >= config.confidence_spike_threshold


def test_no_event_on_neutral_to_neutral() -> None:
    """중립에서 중립으로 변경 시 이벤트가 발생하지 않는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    signal = analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.0)
    assert signal.event_type == "none"

    signal = analyzer.analyze(skew=0.01, volume_pcr=0.99, oi_pcr=0.99)
    assert signal.event_type == "none"


def test_direction_change_alert_disabled() -> None:
    """방향성 전환 알림 비활성화 시 direction_change 이벤트가 감지되지 않는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config_dict = _default_config()
    config_dict["direction_change_alert"] = False
    config = OptionSentimentConfig(**config_dict)
    analyzer = OptionSentimentAnalyzer(config)

    # 첫 분석
    signal = analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.0)
    assert signal.event_type == "none"

    # 상승으로 전환 (알림 비활성화 - confidence_spike는 감지될 수 있음)
    signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    assert signal.event_type != "direction_change"  # 방향성 전환 이벤트는 감지되지 않음


def test_confidence_spike_alert_disabled() -> None:
    """신뢰도 급증 알림 비활성화 시 confidence_spike 이벤트가 감지되지 않는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config_dict = _default_config()
    config_dict["confidence_spike_alert"] = False
    config = OptionSentimentConfig(**config_dict)
    analyzer = OptionSentimentAnalyzer(config)

    # 낮은 신뢰도 (중립)
    signal = analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.0)
    assert signal.event_type == "none"

    # 신뢰도 급증 (알림 비활성화 - direction_change는 감지될 수 있음)
    signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    assert signal.event_type != "confidence_spike"  # 신뢰도 급증 이벤트는 감지되지 않음


# ──────────────────────────────────────────────────────────────────
# 테스트 4: 콜백 호출
# ──────────────────────────────────────────────────────────────────

def test_callback_on_direction_change() -> None:
    """방향성 전환 시 콜백이 호출되는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    callback_called = []
    callback_signal = []

    def callback(signal):
        callback_called.append(True)
        callback_signal.append(signal)

    analyzer = OptionSentimentAnalyzer(config, event_callback=callback)

    # 첫 분석
    signal = analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.0)
    assert len(callback_called) == 0

    # 상승으로 전환
    signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    assert len(callback_called) == 1
    assert callback_signal[0].event_type == "direction_change"


def test_callback_on_confidence_spike() -> None:
    """신뢰도 급증 시 콜백이 호출되는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    callback_called = []

    def callback(signal):
        callback_called.append(True)

    analyzer = OptionSentimentAnalyzer(config, event_callback=callback)

    # 낮은 신뢰도
    signal = analyzer.analyze(skew=0.02, volume_pcr=0.95, oi_pcr=0.95)
    assert len(callback_called) == 0

    # 신뢰도 급증
    signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    assert len(callback_called) == 1


def test_callback_not_called_on_no_event() -> None:
    """이벤트 없을 때 콜백이 호출되지 않는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    callback_called = []

    def callback(signal):
        callback_called.append(True)

    analyzer = OptionSentimentAnalyzer(config, event_callback=callback)

    # 이벤트 없는 분석
    signal = analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.0)
    assert len(callback_called) == 0


def test_callback_exception_handling() -> None:
    """콜백 예외가 분석기 동작에 영향을 주지 않는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())

    def failing_callback(signal):
        raise RuntimeError("Callback error")

    analyzer = OptionSentimentAnalyzer(config, event_callback=failing_callback)

    # 콜백 예외 발생해도 분석은 정상 수행
    signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    assert signal.direction.value == "bullish"
    assert signal.confidence > 0.0


# ──────────────────────────────────────────────────────────────────
# 테스트 5: 설정 로드
# ──────────────────────────────────────────────────────────────────

def test_load_config_from_dict() -> None:
    """딕셔너리에서 설정 로드 검증."""
    from indicators.option_sentiment import load_config_from_dict, OptionSentimentConfig

    config_dict = {
        "option_sentiment": _default_config()
    }

    config = load_config_from_dict(config_dict)

    assert isinstance(config, OptionSentimentConfig)
    assert config.skew_bullish_threshold == 0.05
    assert config.skew_bearish_threshold == -0.05
    assert config.volume_pcr_bullish_threshold == 0.8
    assert config.volume_pcr_bearish_threshold == 1.2
    assert config.oi_pcr_bullish_threshold == 0.9
    assert config.oi_pcr_bearish_threshold == 1.1
    assert abs(config.skew_weight - 0.3) < 1e-9  # 부동소수점 근사값 비교
    assert abs(config.volume_pcr_weight - 0.35) < 1e-9
    assert abs(config.oi_pcr_weight - 0.35) < 1e-9
    assert config.min_confidence == 0.6
    assert config.direction_change_alert is True
    assert config.confidence_spike_alert is True
    assert config.confidence_spike_threshold == 0.8


def test_load_config_with_defaults() -> None:
    """설정이 없을 때 기본값 사용 검증."""
    from indicators.option_sentiment import load_config_from_dict, OptionSentimentConfig

    config = load_config_from_dict({})

    assert isinstance(config, OptionSentimentConfig)
    assert config.skew_bullish_threshold == 0.05  # 기본값
    # min_confidence 기본값: 연속 점수 방식 도입으로 0.6 → 0.5로 조정
    assert config.min_confidence == 0.5  # 기본값


def test_load_config_with_overrides() -> None:
    """설정 오버라이드 검증."""
    from indicators.option_sentiment import load_config_from_dict

    config_dict = {
        "option_sentiment": {
            "skew_bullish_threshold": 0.10,
            "min_confidence": 0.7,
        }
    }

    config = load_config_from_dict(config_dict)

    assert config.skew_bullish_threshold == 0.10
    assert config.min_confidence == 0.7
    assert config.volume_pcr_bullish_threshold == 0.8  # 기본값 유지


# ──────────────────────────────────────────────────────────────────
# 테스트 6: LLM 컨텍스트
# ──────────────────────────────────────────────────────────────────

def test_get_llm_context() -> None:
    """LLM 컨텍스트 생성 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    context = analyzer.get_llm_context(signal)

    assert "[옵션 센티먼트]" in context
    assert "상승" in context
    assert "신뢰도" in context
    assert "Skew" in context
    assert "Volume PCR" in context
    assert "OI PCR" in context


def test_get_llm_context_with_event() -> None:
    """이벤트 포함 LLM 컨텍스트 생성 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 첫 분석
    analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.0)

    # 상승으로 전환
    signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    context = analyzer.get_llm_context(signal)

    assert "⚠️ 이벤트: 방향성 전환" in context


def test_get_llm_context_confidence_spike() -> None:
    """신뢰도 급증 이벤트 포함 LLM 컨텍스트 생성 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 낮은 신뢰도 (약세)
    analyzer.analyze(skew=-0.02, volume_pcr=1.15, oi_pcr=1.05)

    # 신뢰도 급증 (강세로 전환)
    signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    context = analyzer.get_llm_context(signal)

    # 방향성 전환 또는 신뢰도 급증 이벤트 포함
    assert "⚠️ 이벤트:" in context


# ──────────────────────────────────────────────────────────────────
# 테스트 7: 가중치 정규화
# ──────────────────────────────────────────────────────────────────

def test_weight_normalization() -> None:
    """가중치 정규화 검증."""
    from indicators.option_sentiment import OptionSentimentConfig

    config = OptionSentimentConfig(
        skew_weight=0.3,
        volume_pcr_weight=0.35,
        oi_pcr_weight=0.35,
    )

    total = config.skew_weight + config.volume_pcr_weight + config.oi_pcr_weight
    assert abs(total - 1.0) < 1e-9


def test_weight_normalization_with_invalid_weights() -> None:
    """잘못된 가중치 정규화 검증."""
    from indicators.option_sentiment import OptionSentimentConfig

    config = OptionSentimentConfig(
        skew_weight=0.5,
        volume_pcr_weight=0.5,
        oi_pcr_weight=0.5,
    )

    total = config.skew_weight + config.volume_pcr_weight + config.oi_pcr_weight
    assert abs(total - 1.0) < 1e-9


# ──────────────────────────────────────────────────────────────────
# 테스트 8: 상태 추적
# ──────────────────────────────────────────────────────────────────

def test_state_tracking() -> None:
    """상태 추적 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 첫 분석
    signal1 = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    assert signal1.prev_direction is None
    assert signal1.prev_confidence is None

    # 두 번째 분석
    signal2 = analyzer.analyze(skew=0.04, volume_pcr=0.80, oi_pcr=0.90)
    assert signal2.prev_direction.value == "bullish"
    assert signal2.prev_confidence == signal1.confidence


def test_state_persistence() -> None:
    """상태 지속성 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 연속 분석
    for i in range(5):
        signal = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
        if i == 0:
            assert signal.prev_direction is None
        else:
            assert signal.prev_direction.value == "bullish"


# ──────────────────────────────────────────────────────────────────
# 테스트 9: 엣지 케이스
# ──────────────────────────────────────────────────────────────────

def test_zero_values() -> None:
    """0 값 처리 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    signal = analyzer.analyze(skew=0.0, volume_pcr=0.0, oi_pcr=0.0)
    # 0은 bullish_threshold(0.8) 이하이므로 강세로 판단됨
    assert signal.direction.value == "bullish"


def test_extreme_values() -> None:
    """극단 값 처리 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 매우 강세
    signal = analyzer.analyze(skew=1.0, volume_pcr=0.1, oi_pcr=0.1)
    assert signal.direction.value == "bullish"
    assert signal.confidence > 0.8

    # 매우 약세
    signal = analyzer.analyze(skew=-1.0, volume_pcr=10.0, oi_pcr=10.0)
    assert signal.direction.value == "bearish"
    assert signal.confidence > 0.8


def test_negative_pcr_values() -> None:
    """음수 PCR 값 처리 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # 음수 PCR (비정상적이지만 처리 가능해야 함)
    signal = analyzer.analyze(skew=0.0, volume_pcr=-1.0, oi_pcr=1.0)
    # 음수 PCR은 강세로 해석될 수 있음
    assert signal.volume_pcr_signal == "bullish"


# ──────────────────────────────────────────────────────────────────
# 테스트 10: MarketDirection Enum
# ──────────────────────────────────────────────────────────────────

def test_market_direction_enum() -> None:
    """MarketDirection Enum 검증."""
    from indicators.option_sentiment import MarketDirection

    assert MarketDirection.BULLISH.value == "bullish"
    assert MarketDirection.BEARISH.value == "bearish"
    assert MarketDirection.NEUTRAL.value == "neutral"


# ──────────────────────────────────────────────────────────────────
# 테스트 11: SentimentSignal 데이터클래스
# ──────────────────────────────────────────────────────────────────

def test_sentiment_signal_dataclass() -> None:
    """SentimentSignal 데이터클래스 필드 검증."""
    from indicators.option_sentiment import SentimentSignal, MarketDirection

    signal = SentimentSignal(
        direction=MarketDirection.BULLISH,
        confidence=0.75,
        skew=0.06,
        volume_pcr=0.75,
        oi_pcr=0.85,
        skew_signal="bullish",
        volume_pcr_signal="bullish",
        oi_pcr_signal="bullish",
        event_type="none",
        prev_direction=None,
        prev_confidence=None,
    )

    assert signal.direction == MarketDirection.BULLISH
    assert signal.confidence == 0.75
    assert signal.skew == 0.06
    assert signal.volume_pcr == 0.75
    assert signal.oi_pcr == 0.85
    assert signal.event_type == "none"


# ──────────────────────────────────────────────────────────────────
# 테스트 12: 연속 점수 방식 검증
# ──────────────────────────────────────────────────────────────────

def test_continuous_score_gradual_change() -> None:
    """임계값 경계 부근에서 신뢰도가 점진적으로 변하는지 검증.
    (구 양자화 방식에서는 동일한 {-1,0,+1} 점수를 받았을 값들)
    """
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # skew만 변화시키고, volume/oi는 중립(mid=1.0)에 고정
    sig_weak = analyzer.analyze(skew=0.03, volume_pcr=1.0, oi_pcr=1.0)
    sig_strong = analyzer.analyze(skew=0.10, volume_pcr=1.0, oi_pcr=1.0)

    # 더 강한 skew → 더 높은 신뢰도
    assert sig_strong.details["skew_score"] > sig_weak.details["skew_score"]


def test_continuous_score_at_threshold() -> None:
    """임계값 경계값(threshold)에서 점수가 정확히 1.0이 되는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    sig = analyzer.analyze(skew=0.05, volume_pcr=1.0, oi_pcr=1.0)
    assert abs(sig.details["skew_score"] - 1.0) < 1e-9


def test_continuous_score_bearish_symmetry() -> None:
    """약세 skew에서 점수가 음수 방향으로 대칭적으로 산출되는지 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    sig = analyzer.analyze(skew=-0.05, volume_pcr=1.0, oi_pcr=1.0)
    assert abs(sig.details["skew_score"] - (-1.0)) < 1e-9


def test_two_thirds_signals_produce_signal() -> None:
    """지표 2개 강세 + 1개 중립 시 신호가 발생하는지 검증 (min_confidence=0.5).
    구 양자화 방식(0.65 신뢰도 → 임계값 0.6 통과)에서 발생하던 문제:
    연속 점수 방식에서도 동일하게 신호가 발생해야 한다.
    """
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())
    analyzer = OptionSentimentAnalyzer(config)

    # skew=bullish(0.3w), volume_pcr=bullish(0.35w), oi_pcr=neutral(0w)
    # 연속 점수: skew=1.0, volume_pcr=약간 양수, oi_pcr=0
    sig = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=1.0)
    assert sig.direction.value == "bullish"
    assert sig.confidence >= config.min_confidence


# ──────────────────────────────────────────────────────────────────
# 테스트 13: require_neutral_transit 옵션 검증
# ──────────────────────────────────────────────────────────────────

def test_require_neutral_transit_blocks_direct_flip() -> None:
    """require_neutral_transit=True 시 bullish→bearish 직전환 이벤트 억제 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config_dict = _default_config()
    config_dict["require_neutral_transit"] = True
    config = OptionSentimentConfig(**config_dict)
    analyzer = OptionSentimentAnalyzer(config)

    # bullish 설정
    analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    # bearish로 직전환 — 이벤트 억제되어야 함
    sig = analyzer.analyze(skew=-0.06, volume_pcr=1.25, oi_pcr=1.15)
    assert sig.event_type != "direction_change"


def test_require_neutral_transit_allows_via_neutral() -> None:
    """require_neutral_transit=True 시 중립 경유 후 전환은 이벤트 발생 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config_dict = _default_config()
    config_dict["require_neutral_transit"] = True
    config = OptionSentimentConfig(**config_dict)
    analyzer = OptionSentimentAnalyzer(config)

    # neutral 상태로 시작
    analyzer.analyze(skew=0.0, volume_pcr=1.0, oi_pcr=1.0)
    # neutral → bullish : 이벤트 발생해야 함
    sig = analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    assert sig.event_type == "direction_change"


def test_require_neutral_transit_false_allows_direct_flip() -> None:
    """require_neutral_transit=False(기본) 시 직전환 이벤트 발생 검증."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, OptionSentimentConfig

    config = OptionSentimentConfig(**_default_config())  # require_neutral_transit=False
    analyzer = OptionSentimentAnalyzer(config)

    analyzer.analyze(skew=0.06, volume_pcr=0.75, oi_pcr=0.85)
    sig = analyzer.analyze(skew=-0.06, volume_pcr=1.25, oi_pcr=1.15)
    assert sig.event_type == "direction_change"


# ──────────────────────────────────────────────────────────────────
# 테스트 14: iv_skew → skew 변환 패턴 (OptionMixin 관련 주석 검증)
# ──────────────────────────────────────────────────────────────────

def test_iv_skew_none_vs_zero_distinction() -> None:
    """iv_skew=None 과 iv_skew=0.0 을 구분해서 처리하는 패턴 검증.
    0.0은 유효한 값(call_iv=put_iv)이므로 fallback 없이 사용해야 한다.
    """
    # 권장 변환 패턴
    def convert_iv_skew(iv_skew_raw):
        iv_skew = float(iv_skew_raw) if iv_skew_raw is not None else 1.0
        return 1.0 - iv_skew

    # None → fallback 1.0 → skew=0.0 (중립)
    assert convert_iv_skew(None) == 0.0

    # 0.0 (put_iv=0, call_iv=0 비율) → skew=1.0 (강세)
    assert convert_iv_skew(0.0) == 1.0

    # 정상값 1.2 (put_iv > call_iv) → skew=-0.2 (약세)
    assert abs(convert_iv_skew(1.2) - (-0.2)) < 1e-9


# ──────────────────────────────────────────────────────────────────
# 테스트 15: load_config require_neutral_transit
# ──────────────────────────────────────────────────────────────────

def test_load_config_require_neutral_transit() -> None:
    """require_neutral_transit 설정 로드 검증."""
    from indicators.option_sentiment import load_config_from_dict

    # 기본값: False
    config = load_config_from_dict({})
    assert config.require_neutral_transit is False

    # True 설정
    config = load_config_from_dict({"option_sentiment": {"require_neutral_transit": True}})
    assert config.require_neutral_transit is True
