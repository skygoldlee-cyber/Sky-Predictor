"""Pivot Score Integrator — 6-Layer 통합 변곡점 강도 계산기
================================================================
Step 1 (ATR Adaptive), Step 2 (Percent Adaptive), Step 3 (Hybrid Adaptive),
Step 4 (MSB), Step 5 (OI), Step 6 (Kalman) 의 점수를 하나의 종합
``PivotScore`` 로 통합합니다.

아키텍처
--------
::

    ┌─────────────────────────────────────────────────────────┐
    │                 PivotScoreIntegrator                    │
    │                                                         │
    │  Layer 1: ATR Adaptive Pivot    (aap_score) × w1=0.20  │
    │  Layer 2: Percent Adaptive Pivot (pap_score) × w2=0.20  │
    │  Layer 3: Hybrid Adaptive Pivot  (hap_score) × w3=0.15  │
    │  Layer 4: Market Structure Break (msb_score)× w4=0.20  │
    │  Layer 5: OI × MSB cross        (oi_score)  × w5=0.10  │
    │  Layer 6: Kalman Turning Point  (kf_score)  × w6=0.15  │
    │                                                         │
    │  PivotScore = Σ(layer_i × w_i)  ∈ [0, 1]              │
    │  SignalStrength = regime_boost(PivotScore, regime)       │
    └─────────────────────────────────────────────────────────┘

각 레이어 독립성
---------------
- 모든 레이어는 선택적 (None 전달 시 해당 레이어 가중치를 나머지에 재분배)
- 최소 1개 레이어만 있어도 동작

TradeExecutionGate 통합 예시
----------------------------
::

    from indicators import (
        ATRAdaptivePivot, PercentAdaptivePivot, HybridAdaptivePivot,
        MarketStructureBreak, OIStructureGate,
        KalmanTurningPoint, PivotScoreIntegrator,
    )

    aap = ATRAdaptivePivot()
    pap = PercentAdaptivePivot()
    hap = HybridAdaptivePivot()
    msb  = MarketStructureBreak()
    kf   = KalmanTurningPoint()
    oi_gate = OIStructureGate()
    integrator = PivotScoreIntegrator()

    for bar in stream:
        aaps = aap.update(bar.high, bar.low, bar.close, bar_time=bar.time)
        paps = pap.update(bar.high, bar.low, bar.close, bar_time=bar.time)
        haps = hap.update(bar.high, bar.low, bar.close, bar_time=bar.time)
        ms   = msb.update(bar.high, bar.low, bar.close,
                         bar_time=bar.time,
                         pivot_points=aap.confirmed_pivots)
        kfs  = kf.update(bar.close, bar.high, bar.low, bar_time=bar.time)
        oi_score = oi_gate.score(ms, bar.close, oi_levels=current_oi)

        result = integrator.compute(
            aap_score    = aaps.pivot_score,
            pap_score    = paps.pivot_score,
            hap_score    = haps.pivot_score,
            msb_score    = ms.msb_score,
            oi_score     = oi_score,
            kalman_score = kfs.kalman_score,
            aap_signal   = aaps.new_pivot_signal,
            pap_signal   = paps.new_pivot_signal,
            hap_signal   = haps.new_pivot_signal,
            regime       = haps.structure,
        )

        # TradeExecutionGate 진입 조건
        if result.total_score > 0.60 and result.signal != "none":
            direction = result.signal   # "long" | "short"

피처 통합 예시 (PriceTransformer)
----------------------------------
::

    features = {}
    features.update(pivot.get_transformer_features(close))   # azz_* + aap_*
    features.update(msb.get_transformer_features(close))      # msb_*
    features.update(kf.get_transformer_features(close))       # kf_*
    features.update(oi_gate.get_transformer_features(ms, close, oi))  # oi_*
    features.update(integrator.get_transformer_features(result))       # ps_*
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntegratorConfig:
    """PivotScoreIntegrator 설정.

    Parameters
    ----------
    w_aap, w_pap, w_hap, w_msb, w_oi, w_kf:
        각 레이어 가중치. 합계가 1.0 이 아니어도 되며 내부에서 정규화합니다.
    entry_threshold:
        진입 신호 기준 total_score 임계값.
    strong_threshold:
        강한 신호(STRONG) 판정 임계값.
    regime_boost:
        추세 레짐 시 score 부스트 배수. (기본 1.15)
    regime_suppress:
        횡보 레짐 시 score 억제 배수. (기본 0.85)
    decay_half_life:
        신호 강도 지수 감쇠 반감기 (봉 수).
        0 이면 감쇠 없음.
    """
    w_aap:             float = 0.20
    w_pap:             float = 0.20
    w_hap:             float = 0.15
    w_msb:             float = 0.20
    w_oi:              float = 0.10
    w_kf:              float = 0.15
    entry_threshold:   float = 0.55
    strong_threshold:  float = 0.72
    regime_boost:      float = 1.15
    regime_suppress:   float = 0.85
    decay_half_life:   int   = 5    # 봉


# ─────────────────────────────────────────────────────────────────────────────
# 결과 / 상태
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntegratorResult:
    """PivotScoreIntegrator.compute() 반환값."""
    # 개별 레이어 점수
    aap_score:     float = 0.0
    pap_score:     float = 0.0
    msb_score:     float = 0.0
    oi_score:      float = 0.0
    kalman_score:  float = 0.0

    # 통합 점수
    total_score:   float = 0.0   # 가중합 (레짐 부스트 적용 전)
    adjusted_score: float = 0.0  # 레짐 부스트/억제 적용 후

    # 신호
    signal:        str   = "none"    # "long" | "short" | "none"
    signal_strength: str = "none"    # "STRONG" | "MODERATE" | "WEAK" | "none"

    # 신호 방향 투표 (각 레이어의 방향 의견)
    direction_votes: Dict[str, str] = field(default_factory=dict)

    # 유효 레이어 수 (None 제외)
    active_layers: int = 0

    # 레짐 정보
    regime_applied: str = "none"   # "boost" | "suppress" | "none"


# ─────────────────────────────────────────────────────────────────────────────
# 통합기
# ─────────────────────────────────────────────────────────────────────────────

class PivotScoreIntegrator:
    """5-Layer 통합 변곡점 강도 계산기.

    각 레이어는 독립적이며 None 전달 시 자동 제외됩니다.
    가중치는 활성 레이어 합계로 정규화됩니다.
    """

    def __init__(self, config: Optional[IntegratorConfig] = None) -> None:
        self.config = config or IntegratorConfig()
        self._last_result = IntegratorResult()
        self._bar_since_signal: int = 0  # 마지막 신호 이후 경과 봉

    def compute(
        self,
        aap_score:    Optional[float] = None,  # ATRAdaptivePivot.pivot_score
        pap_score:    Optional[float] = None,  # PercentAdaptivePivot.pivot_score
        hap_score:    Optional[float] = None,  # HybridAdaptivePivot.pivot_score
        msb_score:    Optional[float] = None,  # MarketStructureBreak msb_score
        oi_score:     Optional[float] = None,  # OIStructureGate.score()
        kalman_score: Optional[float] = None,  # KalmanTurningPoint kalman_score
        *,
        # 방향 정보 (각 레이어에서 선택적 전달)
        aap_signal:    str = "none",   # "new_high" | "new_low" | "none"
        pap_signal:    str = "none",   # "new_high" | "new_low" | "none"
        hap_signal:    str = "none",   # "new_high" | "new_low" | "none"
        msb_signal:    str = "none",   # BOSType.value
        kalman_signal: str = "none",   # "up" | "down" | "none"
        # 레짐 (선택적)
        regime:        str = "unknown", # "uptrend" | "downtrend" | "ranging" | "unknown"
    ) -> IntegratorResult:
        """통합 점수 계산.

        Parameters
        ----------
        aap_score, pap_score, hap_score, msb_score, oi_score, kalman_score:
            각 레이어 점수 [0, 1]. None = 해당 레이어 비활성.
        aap_signal:
            ATRAdaptivePivot 신호. "new_high" → 고점 변곡, "new_low" → 저점 변곡.
        pap_signal:
            PercentAdaptivePivot 신호. "new_high" → 고점 변곡, "new_low" → 저점 변곡.
        hap_signal:
            HybridAdaptivePivot 신호. "new_high" → 고점 변곡, "new_low" → 저점 변곡.
        msb_signal:
            MSB BOS 신호. "bos_up"/"choch_up" → 상승 구조, "bos_down"/"choch_down" → 하락.
        kalman_signal:
            Kalman 방향. "up" → 저점 반전, "down" → 고점 반전.
        regime:
            시장 레짐. "uptrend"/"downtrend" → 부스트, "ranging" → 억제.
        """
        cfg = self.config

        # ── 레이어 수집 ──────────────────────────────────────────────────────
        layers = [
            ("aap", aap_score,    cfg.w_aap),
            ("pap", pap_score,    cfg.w_pap),
            ("hap", hap_score,    cfg.w_hap),
            ("msb", msb_score,    cfg.w_msb),
            ("oi",  oi_score,     cfg.w_oi),
            ("kf",  kalman_score, cfg.w_kf),
        ]

        active   = [(name, sc, w) for name, sc, w in layers if sc is not None]
        inactive = [(name, sc, w) for name, sc, w in layers if sc is None]

        if not active:
            return IntegratorResult()

        # 가중치 정규화 (비활성 레이어 가중치를 활성 레이어에 비례 재분배)
        total_w_active   = sum(w for _, _, w in active)
        total_w_inactive = sum(w for _, _, w in inactive)
        redistrib = total_w_inactive / total_w_active if total_w_active > 0 else 0.0

        norm_weights = {
            name: w * (1.0 + redistrib)
            for name, _, w in active
        }

        # ── 가중 합산 ────────────────────────────────────────────────────────
        total = sum(
            float(sc) * norm_weights[name]
            for name, sc, _ in active
        )
        total = float(np.clip(total, 0.0, 1.0))

        # ── 레짐 조정 ────────────────────────────────────────────────────────
        regime_applied = "none"
        if regime in ("uptrend", "downtrend"):
            adjusted = float(np.clip(total * cfg.regime_boost, 0.0, 1.0))
            regime_applied = "boost"
        elif regime == "ranging":
            adjusted = float(np.clip(total * cfg.regime_suppress, 0.0, 1.0))
            regime_applied = "suppress"
        else:
            adjusted = total

        # ── 방향 투표 ────────────────────────────────────────────────────────
        votes: Dict[str, str] = {}

        # AAP
        if aap_signal == "new_high":
            votes["aap"] = "short"   # 고점 변곡 → 하락 시작 가능
        elif aap_signal == "new_low":
            votes["aap"] = "long"

        # PAP
        if pap_signal == "new_high":
            votes["pap"] = "short"   # 고점 변곡 → 하락 시작 가능
        elif pap_signal == "new_low":
            votes["pap"] = "long"

        # HAP
        if hap_signal == "new_high":
            votes["hap"] = "short"   # 고점 변곡 → 하락 시작 가능
        elif hap_signal == "new_low":
            votes["hap"] = "long"

        # MSB
        msb_long_signals  = {"bos_up", "choch_up"}
        msb_short_signals = {"bos_down", "choch_down"}
        if msb_signal in msb_long_signals:
            votes["msb"] = "long"
        elif msb_signal in msb_short_signals:
            votes["msb"] = "short"

        # Kalman
        if kalman_signal == "up":
            votes["kf"] = "long"
        elif kalman_signal == "down":
            votes["kf"] = "short"

        # ── 방향 다수결 + 일관성 가중 ────────────────────────────────────────
        signal  = self._majority_vote(votes)
        strength = self._signal_strength(adjusted, cfg)

        # 방향 일관성 계수 적용: 만장일치일수록 점수 상향
        if votes:
            n_votes   = len(votes)
            long_cnt  = sum(1 for v in votes.values() if v == "long")
            short_cnt = sum(1 for v in votes.values() if v == "short")
            majority  = max(long_cnt, short_cnt)
            # 일관성: 만장일치=1.0, 2:1≈0.33, 동점=0.0
            consistency = (majority / n_votes - 0.5) * 2.0
            consistency = float(np.clip(consistency, 0.0, 1.0))
            # total 점수에 일관성 반영 (최대 ±15%)
            adjusted = float(np.clip(adjusted * (0.85 + 0.15 * consistency), 0.0, 1.0))
            # 점수 재계산 후 strength 갱신
            strength = self._signal_strength(adjusted, cfg)

        # 임계값 미달 시 신호 억제
        if adjusted < cfg.entry_threshold:
            signal = "none"
            strength = "none"

        # ── 신호 감쇠 카운터 ─────────────────────────────────────────────────
        if signal != "none":
            self._bar_since_signal = 0
        else:
            self._bar_since_signal += 1

        # ── 결과 ─────────────────────────────────────────────────────────────
        result = IntegratorResult(
            aap_score      = float(aap_score)    if aap_score    is not None else 0.0,
            pap_score      = float(pap_score)    if pap_score    is not None else 0.0,
            msb_score      = float(msb_score)    if msb_score    is not None else 0.0,
            oi_score       = float(oi_score)     if oi_score     is not None else 0.0,
            kalman_score   = float(kalman_score) if kalman_score is not None else 0.0,
            total_score    = total,
            adjusted_score = adjusted,
            signal         = signal,
            signal_strength= strength,
            direction_votes= votes,
            active_layers  = len(active),
            regime_applied = regime_applied,
        )
        self._last_result = result
        return result

    def get_transformer_features(
        self,
        result: Optional["IntegratorResult"] = None,
    ) -> Dict[str, float]:
        """ps_* 피처 반환 (PriceTransformer 주입용)."""
        r = result or self._last_result

        signal_map    = {"long": 1.0, "short": -1.0, "none": 0.0}
        strength_map  = {"STRONG": 1.0, "MODERATE": 0.6, "WEAK": 0.3, "none": 0.0}

        return {
            "ps_total_score":    float(r.total_score),
            "ps_adjusted_score": float(r.adjusted_score),
            "ps_signal":         signal_map.get(r.signal, 0.0),
            "ps_strength":       strength_map.get(r.signal_strength, 0.0),
            "ps_aap_score":      float(r.aap_score),
            "ps_pap_score":      float(r.pap_score),
            "ps_msb_score":      float(r.msb_score),
            "ps_oi_score":       float(r.oi_score),
            "ps_kf_score":       float(r.kalman_score),
            "ps_active_layers":  float(r.active_layers) / 5.0,
            "ps_regime_boost":   float(r.regime_applied == "boost"),
            "ps_regime_suppress":float(r.regime_applied == "suppress"),
            "ps_long":           float(r.signal == "long"),
            "ps_short":          float(r.signal == "short"),
        }

    def get_all_features(
        self,
        result:    Optional["IntegratorResult"] = None,
        pivot_feats: Optional[Dict[str, float]] = None,
        msb_feats:   Optional[Dict[str, float]] = None,
        kf_feats:    Optional[Dict[str, float]] = None,
        oi_feats:    Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """모든 레이어 피처를 하나의 dict 로 합산 반환.

        PriceTransformer 에 주입할 완전한 피처 집합을 만들 때 사용합니다.
        """
        combined: Dict[str, float] = {}
        for d in [pivot_feats, msb_feats, kf_feats, oi_feats]:
            if d:
                combined.update(d)
        combined.update(self.get_transformer_features(result))
        return combined

    # ── 내부 유틸 ────────────────────────────────────────────────────────────

    @staticmethod
    def _majority_vote(votes: Dict[str, str]) -> str:
        """방향 다수결."""
        if not votes:
            return "none"
        long_cnt  = sum(1 for v in votes.values() if v == "long")
        short_cnt = sum(1 for v in votes.values() if v == "short")
        if long_cnt > short_cnt:
            return "long"
        if short_cnt > long_cnt:
            return "short"
        return "none"  # 동점 → 신호 없음

    @staticmethod
    def _signal_strength(score: float, cfg: IntegratorConfig) -> str:
        if score >= cfg.strong_threshold:
            return "STRONG"
        if score >= cfg.entry_threshold:
            return "MODERATE"
        if score >= cfg.entry_threshold * 0.7:
            return "WEAK"
        return "none"
