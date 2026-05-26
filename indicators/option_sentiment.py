"""옵션 센티먼트 분석기
===========================
Skew, Volume PCR, OI PCR을 조합하여 장의 방향성을 진단합니다.

[Skew 부호 규약]
    skew = call_iv - put_iv
    양수(+): call_iv > put_iv → 콜 프리미엄이 비쌈 → 상방 투기 수요 → 강세
    음수(-): put_iv  > call_iv → 풋 프리미엄이 비쌈 → 하방 헷지 수요 → 약세
    ※ KOSPI200 옵션은 통상 풋 skew(음수 방향)가 형성되므로, 음수 범위가 정상 구간.
       opt_snap의 iv_skew(= put_iv / call_iv)를 받을 때는 외부에서 변환 필요:
       skew = 1.0 - iv_skew  단, iv_skew=None/0 → fallback=1.0 처리 시
       None 체크와 0 체크를 분리할 것 (0도 유효한 값).

[연속 점수 방식]
    각 지표를 {-1, 0, +1}로 단순 양자화하지 않고,
    임계값 대비 비율로 연속 점수(-1.0 ~ +1.0)를 산출합니다.
    → 경계 부근의 미세한 변화도 신뢰도에 반영됩니다.

Config 설정 (config.json의 option_sentiment 섹션):
    skew_bullish_threshold      : skew >= 이 값이면 강세 (call_iv - put_iv 기준)
    skew_bearish_threshold      : skew <= 이 값이면 약세 (음수)
    volume_pcr_bullish_threshold: volume PCR <= 이 값이면 강세
    volume_pcr_bearish_threshold: volume PCR >= 이 값이면 약세
    oi_pcr_bullish_threshold    : OI PCR <= 이 값이면 강세
    oi_pcr_bearish_threshold    : OI PCR >= 이 값이면 약세
    skew_weight                 : skew 가중치
    volume_pcr_weight           : volume PCR 가중치
    oi_pcr_weight               : OI PCR 가중치
    min_confidence              : 최소 신뢰도 (이 값 미만이면 NEUTRAL로 강등)
    require_neutral_transit     : True이면 bullish↔bearish 직전환 시 이벤트 억제
    direction_change_alert      : 방향성 전환 알림 활성화
    confidence_spike_alert      : 신뢰도 급증 알림 활성화
    confidence_spike_threshold  : 신뢰도 급증 임계값
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

_logger = logging.getLogger(__name__)


class MarketDirection(Enum):
    """장 방향성

    Note:
        Enum 값은 소문자 문자열("bullish", "bearish", "neutral")로 저장됩니다.
        SentimentSignal.skew_signal 등의 문자열 필드와 비교 시 .value를 사용하거나,
        문자열 비교 시 Enum.value를 변환해야 합니다.
        예: MarketDirection.BULLISH.value == "bullish"
    """
    BULLISH = "bullish"   # 상승
    BEARISH = "bearish"   # 하락
    NEUTRAL = "neutral"   # 중립


@dataclass
class OptionSentimentConfig:
    """옵션 센티먼트 분석 설정"""

    # ── Skew 임계값 ────────────────────────────────────────────────
    # skew = call_iv - put_iv (근사치: 1 - put_iv/call_iv)
    #   양수: 콜 프리미엄 비쌈 → 상방 투기 → 강세
    #   음수: 풋 프리미엄 비쌈 → 하방 헷지 → 약세
    # 주의: 현재 구현은 skew = 1 - (put_iv/call_iv) 비율 차이를 사용하므로
    #       임계값 0.05는 퍼센트가 아니라 비율 기준임
    #       예: call_iv=20%, put_iv=18% → skew=+0.10 (강세)
    #           call_iv=18%, put_iv=20% → skew=-0.11 (약세)
    skew_bullish_threshold: float = 0.05   # 비율 +0.05 이상이면 강세
    skew_bearish_threshold: float = -0.05  # 비율 -0.05 이하이면 약세

    # ── Volume PCR 임계값 ──────────────────────────────────────────
    # put_volume / call_volume  낮을수록 강세, 높을수록 약세
    volume_pcr_bullish_threshold: float = 0.8   # 이하이면 강세
    volume_pcr_bearish_threshold: float = 1.2   # 이상이면 약세

    # ── OI PCR 임계값 ─────────────────────────────────────────────
    # put_OI / call_OI  낮을수록 강세, 높을수록 약세
    oi_pcr_bullish_threshold: float = 0.9   # 이하이면 강세
    oi_pcr_bearish_threshold: float = 1.1   # 이상이면 약세

    # ── 가중치 (합계 1.0이어야 함 — __post_init__에서 자동 정규화) ──
    skew_weight: float = 0.3
    volume_pcr_weight: float = 0.35
    oi_pcr_weight: float = 0.35

    # ── 신뢰도 설정 ────────────────────────────────────────────────
    # 연속 점수 방식 도입으로 신뢰도 범위가 세밀해졌으므로
    # 기본값을 0.5로 낮춰 2/3 지표 합의 시에도 신호가 발생하도록 함.
    min_confidence: float = 0.5

    # ── 이벤트 알림 설정 ───────────────────────────────────────────
    direction_change_alert: bool = True
    confidence_spike_alert: bool = True
    confidence_spike_threshold: float = 0.8

    # bullish ↔ bearish 직전환을 이벤트로 허용할지 여부.
    # True 이면 반드시 NEUTRAL을 경유한 경우만 direction_change 발생.
    # 즉, BULLISH → BEARISH 또는 BEARISH → BULLISH 직전환은 억제됨.
    # KOSPI200처럼 빠른 방향 전환이 잦은 시장에서 허위 알림 억제에 유용.
    require_neutral_transit: bool = False

    def __post_init__(self) -> None:
        """가중치 합이 0이 아닌 경우 자동 정규화."""
        # 임계값 대소 관계 검증
        if self.volume_pcr_bullish_threshold >= self.volume_pcr_bearish_threshold:
            raise ValueError(
                f"volume_pcr_bullish_threshold ({self.volume_pcr_bullish_threshold}) "
                f"must be < volume_pcr_bearish_threshold ({self.volume_pcr_bearish_threshold})"
            )
        if self.oi_pcr_bullish_threshold >= self.oi_pcr_bearish_threshold:
            raise ValueError(
                f"oi_pcr_bullish_threshold ({self.oi_pcr_bullish_threshold}) "
                f"must be < oi_pcr_bearish_threshold ({self.oi_pcr_bearish_threshold})"
            )

        # 가중치 정규화
        total = self.skew_weight + self.volume_pcr_weight + self.oi_pcr_weight
        if total > 0:
            if abs(total - 1.0) > 1e-6:
                _logger.warning(
                    "[OptionSentiment] 가중치 합계 %.3f ≠ 1.0, 자동 정규화 적용 (skew=%.2f, volume=%.2f, oi=%.2f)",
                    total, self.skew_weight, self.volume_pcr_weight, self.oi_pcr_weight
                )
                self.skew_weight /= total
                self.volume_pcr_weight /= total
                self.oi_pcr_weight /= total


@dataclass
class SentimentSignal:
    """센티먼트 신호"""
    direction: MarketDirection
    confidence: float           # 0.0 ~ 1.0
    skew: float
    volume_pcr: float
    oi_pcr: float
    skew_signal: str            # "bullish" | "bearish" | "neutral"
    volume_pcr_signal: str
    oi_pcr_signal: str
    event_type: str = "none"    # "direction_change" | "confidence_spike" | "none"
    prev_direction: Optional[MarketDirection] = None
    prev_confidence: Optional[float] = None
    event_timestamp: Optional[float] = None  # 의미 발생 시각 (Unix timestamp)
    details: Dict[str, Any] = field(default_factory=dict)


class OptionSentimentAnalyzer:
    """옵션 센티먼트 분석기

    Skew / Volume PCR / OI PCR 세 지표를 연속 점수로 환산한 뒤
    가중 평균하여 장의 방향성과 신뢰도를 산출합니다.
    """

    def __init__(
        self,
        config: Optional[OptionSentimentConfig] = None,
        event_callback: Optional[Callable[[SentimentSignal], None]] = None,
    ) -> None:
        self.config = config or OptionSentimentConfig()
        self._event_callback = event_callback

        # 이전 틱 상태 추적
        self._last_direction: Optional[MarketDirection] = None
        self._last_confidence: Optional[float] = None
        # require_neutral_transit용: 마지막 NEUTRAL이 아닌 방향 추적
        self._last_non_neutral_direction: Optional[MarketDirection] = None

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def analyze(
        self,
        skew: float,
        volume_pcr: float,
        oi_pcr: float,
    ) -> SentimentSignal:
        """센티먼트 분석.

        Args:
            skew:       call_iv - put_iv
                        양수 → 콜 프리미엄 비쌈 → 강세
                        음수 → 풋 프리미엄 비쌈 → 약세
            volume_pcr: put_volume / call_volume  낮을수록 강세
            oi_pcr:     put_OI / call_OI          낮을수록 강세

        Returns:
            SentimentSignal: 방향성, 신뢰도, 개별 신호, 이벤트 정보 포함.
        """
        # 입력값 유효성 검사 (nan/inf 체크)
        if math.isnan(skew) or math.isinf(skew):
            _logger.warning("[OptionSentiment] 유효하지 않은 skew: %s", skew)
            skew = 0.0
        if math.isnan(volume_pcr) or math.isinf(volume_pcr):
            _logger.warning("[OptionSentiment] 유효하지 않은 volume_pcr: %s", volume_pcr)
            volume_pcr = 1.0
        if math.isnan(oi_pcr) or math.isinf(oi_pcr):
            _logger.warning("[OptionSentiment] 유효하지 않은 oi_pcr: %s", oi_pcr)
            oi_pcr = 1.0

        cfg = self.config

        # 1. 개별 지표 분류 (레이블)
        skew_signal = self._classify_skew(skew, cfg)
        volume_pcr_signal = self._classify_volume_pcr(volume_pcr, cfg)
        oi_pcr_signal = self._classify_oi_pcr(oi_pcr, cfg)

        # 2. 연속 점수 산출 (-1.0 ~ +1.0)
        skew_score = self._skew_continuous_score(skew, cfg)
        volume_pcr_score = self._volume_pcr_continuous_score(volume_pcr, cfg)
        oi_pcr_score = self._oi_pcr_continuous_score(oi_pcr, cfg)

        # 3. 가중 평균 점수
        weighted_score = (
            skew_score * cfg.skew_weight
            + volume_pcr_score * cfg.volume_pcr_weight
            + oi_pcr_score * cfg.oi_pcr_weight
        )

        # 4. 신뢰도 (0.0 ~ 1.0)
        confidence = abs(weighted_score)

        # 5. 방향성 결정 — min_confidence 미만이면 NEUTRAL
        if confidence < cfg.min_confidence:
            direction = MarketDirection.NEUTRAL
        elif weighted_score > 0:
            direction = MarketDirection.BULLISH
        else:
            direction = MarketDirection.BEARISH

        # 6. 이벤트 감지
        prev_direction = self._last_direction
        prev_confidence = self._last_confidence
        event_type = self._detect_event(
            direction, confidence, prev_direction, prev_confidence, cfg
        )

        # [TIME-FIX] 의미 발생 시각 설정 (이벤트 발생 시점의 현재 시간)
        event_timestamp = time.time() if event_type != "none" else None

        # 7. 상태 업데이트
        self._last_direction = direction
        self._last_confidence = confidence
        # require_neutral_transit용: NEUTRAL이 아닌 방향 추적
        if direction != MarketDirection.NEUTRAL:
            self._last_non_neutral_direction = direction

        # 8. 신호 객체 생성
        signal = SentimentSignal(
            direction=direction,
            confidence=confidence,
            skew=skew,
            volume_pcr=volume_pcr,
            oi_pcr=oi_pcr,
            skew_signal=skew_signal,
            volume_pcr_signal=volume_pcr_signal,
            oi_pcr_signal=oi_pcr_signal,
            event_type=event_type,
            prev_direction=prev_direction,
            prev_confidence=prev_confidence,
            event_timestamp=event_timestamp,
            details={
                "skew_score": skew_score,
                "volume_pcr_score": volume_pcr_score,
                "oi_pcr_score": oi_pcr_score,
                "weighted_score": weighted_score,
            },
        )

        # 9. 이벤트 콜백
        if event_type != "none" and self._event_callback:
            try:
                self._event_callback(signal)
            except Exception as exc:
                _logger.error("[OptionSentiment] 콜백 호출 실패: %s", exc)

        return signal

    def get_llm_context(self, signal: SentimentSignal) -> str:
        """LLM 프롬프트에 삽입할 센티먼트 요약 문자열 반환.

        Args:
            signal: SentimentSignal 객체

        Returns:
            str: 센티먼트 요약 문자열 (한국어)

        Example:
            >>> signal = analyzer.analyze(skew=0.1, volume_pcr=0.85, oi_pcr=0.9)
            >>> context = analyzer.get_llm_context(signal)
            >>> print(context)
            [옵션 센티먼트]
            종합 방향: 상승 (신뢰도: 65.0%)
              - Skew: +10.00% (강세)
              - Volume PCR: 0.85 (중립)
              - OI PCR: 0.90 (강세)
        """
        _DIR_KOR = {
            MarketDirection.BULLISH: "상승",
            MarketDirection.BEARISH: "하락",
            MarketDirection.NEUTRAL: "중립",
        }
        _SIG_KOR = {"bullish": "강세", "bearish": "약세", "neutral": "중립"}

        direction_kor = _DIR_KOR[signal.direction]
        lines = [
            "[옵션 센티먼트]",
            f"종합 방향: {direction_kor} (신뢰도: {signal.confidence * 100:.1f}%)",
            f"  - Skew: {signal.skew * 100:+.2f}% ({_SIG_KOR[signal.skew_signal]})",
            f"  - Volume PCR: {signal.volume_pcr:.2f} ({_SIG_KOR[signal.volume_pcr_signal]})",
            f"  - OI PCR: {signal.oi_pcr:.2f} ({_SIG_KOR[signal.oi_pcr_signal]})",
        ]

        if "direction_change" in signal.event_type:
            prev_kor = _DIR_KOR.get(signal.prev_direction, "알 수 없음")
            lines.append(f"⚠️ 이벤트: 방향성 전환 ({prev_kor} → {direction_kor})")
        if "confidence_spike" in signal.event_type and signal.prev_confidence is not None:
            lines.append(
                f"⚠️ 이벤트: 신뢰도 급증"
                f" ({signal.prev_confidence * 100:.1f}%"
                f" → {signal.confidence * 100:.1f}%)"
            )

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    # 분류 (레이블) — 임계값 기반 3단계
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_skew(skew: float, cfg: OptionSentimentConfig) -> str:
        if skew >= cfg.skew_bullish_threshold:
            return "bullish"
        if skew <= cfg.skew_bearish_threshold:
            return "bearish"
        return "neutral"

    @staticmethod
    def _classify_volume_pcr(volume_pcr: float, cfg: OptionSentimentConfig) -> str:
        if volume_pcr <= cfg.volume_pcr_bullish_threshold:
            return "bullish"
        if volume_pcr >= cfg.volume_pcr_bearish_threshold:
            return "bearish"
        return "neutral"

    @staticmethod
    def _classify_oi_pcr(oi_pcr: float, cfg: OptionSentimentConfig) -> str:
        if oi_pcr <= cfg.oi_pcr_bullish_threshold:
            return "bullish"
        if oi_pcr >= cfg.oi_pcr_bearish_threshold:
            return "bearish"
        return "neutral"

    # ──────────────────────────────────────────────────────────────
    # 연속 점수 산출 (-1.0 ~ +1.0)
    # 임계값을 기준으로 정규화하여 경계 부근 변화도 반영.
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _skew_continuous_score(skew: float, cfg: OptionSentimentConfig) -> float:
        """skew(call_iv - put_iv) → [-1.0, +1.0]."""
        if skew >= 0:
            denom = cfg.skew_bullish_threshold
            if denom <= 0:
                return 1.0 if skew > 0 else 0.0
            return min(1.0, skew / denom)
        else:
            denom = abs(cfg.skew_bearish_threshold)
            if denom <= 0:
                return -1.0  # skew < 0이 보장됨
            return max(-1.0, skew / denom)

    @staticmethod
    def _volume_pcr_continuous_score(volume_pcr: float, cfg: OptionSentimentConfig) -> float:
        """Volume PCR → [-1.0, +1.0].  낮을수록 +, 높을수록 -."""
        mid = (cfg.volume_pcr_bullish_threshold + cfg.volume_pcr_bearish_threshold) / 2.0
        if volume_pcr <= mid:
            span = mid - cfg.volume_pcr_bullish_threshold
            if span <= 0:
                return 1.0
            return min(1.0, (mid - volume_pcr) / span)
        else:
            span = cfg.volume_pcr_bearish_threshold - mid
            if span <= 0:
                return -1.0
            return max(-1.0, -(volume_pcr - mid) / span)

    @staticmethod
    def _oi_pcr_continuous_score(oi_pcr: float, cfg: OptionSentimentConfig) -> float:
        """OI PCR → [-1.0, +1.0].  낮을수록 +, 높을수록 -."""
        mid = (cfg.oi_pcr_bullish_threshold + cfg.oi_pcr_bearish_threshold) / 2.0
        if oi_pcr <= mid:
            span = mid - cfg.oi_pcr_bullish_threshold
            if span <= 0:
                return 1.0
            return min(1.0, (mid - oi_pcr) / span)
        else:
            span = cfg.oi_pcr_bearish_threshold - mid
            if span <= 0:
                return -1.0
            return max(-1.0, -(oi_pcr - mid) / span)

    # ──────────────────────────────────────────────────────────────
    # 이벤트 감지
    # ──────────────────────────────────────────────────────────────

    def _detect_event(
        self,
        direction: MarketDirection,
        confidence: float,
        prev_direction: Optional[MarketDirection],
        prev_confidence: Optional[float],
        cfg: OptionSentimentConfig,
    ) -> str:
        """이벤트 유형을 반환한다.

        우선순위: direction_change > confidence_spike > none
        두 이벤트가 동시에 성립하면 "+"로 연결하여 반환 (예: "direction_change+confidence_spike").

        Returns:
            str: 이벤트 타입 ("none", "direction_change", "confidence_spike", "direction_change+confidence_spike")
        """
        events = []

        # ── 방향성 전환 감지 ──────────────────────────────────────
        if cfg.direction_change_alert and prev_direction is not None:
            changed = prev_direction != direction
            not_neutral = direction != MarketDirection.NEUTRAL

            if cfg.require_neutral_transit:
                # 중립을 경유한 전환만 인정
                # BULLISH ↔ BEARISH 직전환 억제: 이전에 반대 방향이었음을 기억
                via_neutral = prev_direction == MarketDirection.NEUTRAL
                last_non_neutral = self._last_non_neutral_direction
                # 직전환 감지: 이전 상태와 현재 상태가 반대 방향
                direct_opposite = (
                    (prev_direction == MarketDirection.BULLISH and direction == MarketDirection.BEARISH) or
                    (prev_direction == MarketDirection.BEARISH and direction == MarketDirection.BULLISH)
                )
                # 이벤트 발생 조건:
                # 1. 방향 변경됨
                # 2. 현재 상태가 NEUTRAL 아님
                # 3. 직전 상태가 NEUTRAL이거나 (via_neutral)
                # 4. 이전에 반대 방향이었음 (last_non_neutral != direction)
                if changed and not_neutral and (via_neutral or (last_non_neutral is not None and last_non_neutral != direction)):
                    events.append("direction_change")
            else:
                # 직전환도 허용 (기본)
                if changed and not_neutral:
                    events.append("direction_change")

            if "direction_change" in events:
                _logger.info(
                    "[OptionSentiment] 방향성 전환: %s → %s (신뢰도: %.2f)",
                    prev_direction.value, direction.value, confidence,
                )

        # ── 신뢰도 급증 감지 ───────────────────────────────────────
        if (
            cfg.confidence_spike_alert
            and prev_confidence is not None
            and confidence >= cfg.confidence_spike_threshold
            and prev_confidence < cfg.confidence_spike_threshold
        ):
            events.append("confidence_spike")
            _logger.info(
                "[OptionSentiment] 신뢰도 급증: %.2f → %.2f (임계값: %.2f)",
                prev_confidence, confidence, cfg.confidence_spike_threshold,
            )

        # 이벤트 결합 (없으면 "none")
        if not events:
            return "none"
        elif len(events) == 1:
            return events[0]
        else:
            return "+".join(events)


# ──────────────────────────────────────────────────────────────────
# 설정 로드 헬퍼
# ──────────────────────────────────────────────────────────────────

def load_config_from_dict(config_dict: Dict[str, Any]) -> OptionSentimentConfig:
    """config.json 딕셔너리의 ``option_sentiment`` 섹션에서 설정을 로드한다.

    키가 없으면 OptionSentimentConfig 기본값을 사용한다.

    Note:
        iv_skew 변환(put_iv/call_iv → call_iv - put_iv)은 호출 측(OptionMixin 등)
        에서 수행해야 합니다. iv_skew 값이 None일 때와 0.0일 때를 반드시 구분하세요:
            iv_skew_raw = opt_snap.get("iv_skew")
            iv_skew = float(iv_skew_raw) if iv_skew_raw is not None else 1.0
            skew = 1.0 - iv_skew
    """
    import dataclasses

    cfg_raw = config_dict.get("option_sentiment") or {}
    # 기본값이 있는 필드만 처리 (MISSING 필드 방어)
    defaults = {
        f.name: f.default
        for f in dataclasses.fields(OptionSentimentConfig)
        if f.default is not dataclasses.MISSING
    }
    kwargs = {k: cfg_raw.get(k, v) for k, v in defaults.items()}
    return OptionSentimentConfig(**kwargs)
