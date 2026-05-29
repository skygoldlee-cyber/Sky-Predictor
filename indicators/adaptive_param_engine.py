"""indicators/adaptive_param_engine.py
====================================================
장중 자기완결형 적응형 파라미터 조정 엔진 (AdaptiveParamEngine)

외부 레짐 분류기나 MarketRegimeClassifier 없이,
AdaptiveZigZag 내부 버퍼(_highs/_lows/_atr_values/_all_swings 등)만으로
장중 피봇 탐색 파라미터를 실시간 자동 조정한다.

통합 방법 (adaptive_zigzag.py 수정 2곳):
─────────────────────────────────────────
[수정 1] AdaptiveZigZag.__init__() 말미에 엔진 초기화 추가:

    # ── [자기완결형 적응 엔진] ─────────────────────────────────────
    from .adaptive_param_engine import AdaptiveParamEngine
    self._adaptive_engine = AdaptiveParamEngine(self.config)

[수정 2] _get_runtime_params() 내 Layer B 블록 교체:

    # 기존 Layer B (삭제):
    b_mult = 1.0
    if hasattr(self, '_param_adjuster') and ...:
        b_mult = self._param_adjuster.get_vol_ratio()

    # 새 Layer C (추가):
    adjusted = self._adaptive_engine.compute(
        atr_values  = list(self._atr_values),
        all_swings  = self._all_swings,
        bar_idx     = self._bar_idx,
        er          = float(self._calc_er()),
        der         = float(self._calc_der()),
        direction   = self._current_direction,
        last_confirmed_bar_idx = self._last_confirmed_bar_idx,
    )
    # 결합: config 수정 없이 런타임 dict로만
    return {
        "atr_multiplier": float(np.clip(
            cfg.atr_multiplier * adjusted["mult"],
            cfg.atr_multiplier_min, cfg.atr_multiplier_max,
        )),
        "confirmation_bars": adjusted["confirmation_bars"],
        "min_wave_atr_ratio": float(np.clip(
            a_atr_ratio * adjusted["wave_ratio_mult"],
            0.5, 5.0,
        )),
        "min_wave_bars": a_wave_bars,
        "pivot_threshold_min_pct": float(np.clip(
            cfg.pivot_threshold_min_pct * adjusted["thr_mult"],
            cfg.pivot_threshold_min_pct * 0.5,
            cfg.pivot_threshold_max_pct,
        )),
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 출력 데이터클래스
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveAdjustment:
    """compute()의 반환값. 각 값은 config 기준값에 곱해지는 배율."""

    mult: float = 1.0               # atr_multiplier 배율
    wave_ratio_mult: float = 1.0    # min_wave_atr_ratio 배율
    thr_mult: float = 1.0           # pivot_threshold_min_pct 배율
    confirmation_bars: int = 2      # 확정 대기 봉수 (절대값)

    # 진단용 (GUI 패널 표시)
    er: float = 0.5
    atr_pct: float = 50.0           # ATR 백분위 (0~100)
    density_signal: str = "normal"  # "sparse" | "normal" | "dense"
    regime_label: str = "unknown"   # "trend_up" | "trend_dn" | "chop" | "volatile"


# ──────────────────────────────────────────────────────────────────────────────
# 핵심 엔진
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveParamEngine:
    """ZigZag 내부 버퍼만으로 장중 파라미터를 자기완결적으로 조정한다.

    3개 신호를 합성:
      Signal-A  ER (Efficiency Ratio)       : 추세 강도 0~1
      Signal-B  ATR 백분위                  : 현재 변동성 상대 위치
      Signal-C  피봇 밀도 피드백             : 최근 피봇 수 과다/부족

    파라미터 결정 흐름:
      1. 신호 계산
      2. 레짐 라벨 결정 (4종)
      3. 레짐별 기준 배율 조회
      4. 밀도 피드백으로 미세 보정
      5. EMA 스무딩 (깜빡임 방지)
      6. 클램핑 후 반환
    """

    # ── 레짐별 기준 배율 테이블 ──────────────────────────────────────────────
    #
    # KP200 선물 1분봉 기준으로 설정:
    #   ATR ≈ 0.3~0.8pt, 일간 범위 ≈ 5~12pt
    #   주요 변곡점 파동 ≈ 2~5pt (일간 범위의 25~40%)
    #
    # mult: atr_multiplier 배율 (config 기준값에 곱함)
    # wave: min_wave_atr_ratio 배율
    # thr:  pivot_threshold_min_pct 배율
    # cb:   confirmation_bars (절대값)
    #
    REGIME_TABLE = {
        #                     mult  wave   thr    cb
        "trend_strong_up":   (1.30, 1.20,  0.90,  1),  # 강한 상승 추세: 큰 되돌림만 피봇
        "trend_strong_dn":   (1.30, 1.20,  0.90,  1),  # 강한 하락 추세: 대칭
        "trend_weak_up":     (1.10, 1.05,  1.00,  2),  # 약한 상승: 중간 민감도
        "trend_weak_dn":     (1.10, 1.05,  1.00,  2),  # 약한 하락: 대칭
        "chop_low_vol":      (0.80, 0.85,  0.85,  2),  # 저변동 횡보: 작은 반전 포착
        "chop_high_vol":     (1.40, 1.40,  1.15,  3),  # 고변동 횡보: 흔들기 억제
        "volatile":          (1.50, 1.50,  1.20,  2),  # 급변동(뉴스): 임계값 최대
        "unknown":           (1.00, 1.00,  1.00,  2),  # 웜업/미분류: 기본값
    }

    # ── EMA 스무딩 계수 (깜빡임 방지) ────────────────────────────────────────
    EMA_ALPHA = 0.15   # 작을수록 천천히 반응 (0.1~0.3 권장)

    # ── 밀도 기준 ────────────────────────────────────────────────────────────
    DENSITY_WINDOW_BARS = 30       # 밀도 측정 윈도우 (봉)
    DENSITY_HIGH_THRESH = 4        # 30봉 내 4개 이상 → 과다
    DENSITY_LOW_THRESH  = 1        # 30봉 내 1개 미만 → 부족
    DENSITY_HIGH_BOOST  = 1.15     # 과다 시 임계값 배율 상향
    DENSITY_LOW_REDUCE  = 0.90     # 부족 시 임계값 배율 하향

    # ── ATR 백분위 기준 ──────────────────────────────────────────────────────
    ATR_PCT_WINDOW = 60            # 백분위 계산 윈도우 (봉)
    ATR_HIGH_THRESH = 75.0         # 상위 25% → 고변동
    ATR_LOW_THRESH  = 25.0         # 하위 25% → 저변동

    # ── ER 기준 ──────────────────────────────────────────────────────────────
    ER_STRONG = 0.60               # 강한 추세
    ER_WEAK   = 0.30               # 횡보 진입

    def __init__(self, config: Any) -> None:
        self._cfg = config

        # EMA 상태 (스무딩)
        self._ema_mult:        float = 1.0
        self._ema_wave:        float = 1.0
        self._ema_thr:         float = 1.0
        self._ema_cb:          float = float(getattr(config, "confirmation_bars", 2))
        self._initialized:     bool  = False

        # 진단용 마지막 계산 결과
        self._last_adj: Optional[AdaptiveAdjustment] = None

        logger.info("[AdaptiveParamEngine] 초기화 완료")

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        atr_values:             List[float],
        all_swings:             Any,          # List[SwingPoint]
        bar_idx:                int,
        er:                     float,
        der:                    float,
        direction:              int,
        last_confirmed_bar_idx: int,
        structure:              str = "unknown",
    ) -> AdaptiveAdjustment:
        """매 봉마다 호출. 조정된 파라미터 배율을 반환한다.

        Args:
            atr_values:             ZigZag._atr_values deque (리스트로 전달)
            all_swings:             ZigZag._all_swings 리스트
            bar_idx:                ZigZag._bar_idx
            er:                     _calc_er() 결과 (0~1)
            der:                    _calc_der() 결과 (-1~1)
            direction:              _current_direction (-1/0/1)
            last_confirmed_bar_idx: ZigZag._last_confirmed_bar_idx
            structure:              시장 구조 (uptrend/downtrend/ranging/unknown)
        """
        try:
            # ① ATR 백분위 계산
            atr_pct = self._calc_atr_percentile(atr_values)

            # ② 피봇 밀도 계산
            density, density_signal = self._calc_density(all_swings, bar_idx)

            # ③ 레짐 라벨 결정 (시장 구조 정보 추가)
            regime = self._classify_regime(er, der, atr_pct, direction, structure)

            # ④ 기준 배율 조회
            base_mult, base_wave, base_thr, base_cb = self.REGIME_TABLE[regime]

            # ⑤ 밀도 피드백 보정
            if density_signal == "dense":
                base_mult *= self.DENSITY_HIGH_BOOST
                base_wave *= self.DENSITY_HIGH_BOOST
                base_thr  *= self.DENSITY_HIGH_BOOST
            elif density_signal == "sparse":
                base_mult *= self.DENSITY_LOW_REDUCE
                base_wave *= self.DENSITY_LOW_REDUCE
                base_thr  *= self.DENSITY_LOW_REDUCE

            # ⑥ EMA 스무딩 (급격한 파라미터 변동 방지)
            if not self._initialized:
                self._ema_mult = base_mult
                self._ema_wave = base_wave
                self._ema_thr  = base_thr
                self._ema_cb   = float(base_cb)
                self._initialized = True
            else:
                a = self.EMA_ALPHA
                self._ema_mult = a * base_mult + (1 - a) * self._ema_mult
                self._ema_wave = a * base_wave + (1 - a) * self._ema_wave
                self._ema_thr  = a * base_thr  + (1 - a) * self._ema_thr
                self._ema_cb   = a * base_cb   + (1 - a) * self._ema_cb

            # ⑦ 클램핑
            final_mult = float(np.clip(self._ema_mult, 0.6, 2.0))
            final_wave = float(np.clip(self._ema_wave, 0.6, 2.0))
            final_thr  = float(np.clip(self._ema_thr,  0.6, 1.5))
            final_cb   = int(np.clip(round(self._ema_cb), 1, 4))

            adj = AdaptiveAdjustment(
                mult              = final_mult,
                wave_ratio_mult   = final_wave,
                thr_mult          = final_thr,
                confirmation_bars = final_cb,
                er                = er,
                atr_pct           = atr_pct,
                density_signal    = density_signal,
                regime_label      = regime,
            )
            self._last_adj = adj

            logger.debug(
                "[AdaptiveParamEngine] regime=%s er=%.2f atr_pct=%.0f%% "
                "density=%s mult=%.2f wave=%.2f thr=%.2f cb=%d",
                regime, er, atr_pct, density_signal,
                final_mult, final_wave, final_thr, final_cb,
            )
            return adj

        except Exception as e:
            logger.warning("[AdaptiveParamEngine] compute 실패, 기본값 사용: %s", e)
            return AdaptiveAdjustment()

    @property
    def last_adjustment(self) -> Optional[AdaptiveAdjustment]:
        """마지막 계산 결과 (GUI 패널 표시용)."""
        return self._last_adj

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    def _calc_atr_percentile(self, atr_values: List[float]) -> float:
        """현재 ATR이 최근 N봉 ATR 분포에서 몇 퍼센타일에 위치하는지 반환 (0~100)."""
        if len(atr_values) < 5:
            return 50.0
        window = atr_values[-self.ATR_PCT_WINDOW:]
        current = window[-1]
        if current <= 0:
            return 50.0
        pct = float(np.mean([v <= current for v in window]) * 100.0)
        return float(np.clip(pct, 0.0, 100.0))

    def _calc_density(
        self, all_swings: Any, bar_idx: int
    ) -> tuple[int, str]:
        """최근 DENSITY_WINDOW_BARS 봉 내 확정 피봇 수와 밀도 신호 반환."""
        try:
            recent_count = sum(
                1 for s in all_swings
                if getattr(s, "confirmed", False) and
                   getattr(s, "confirmed_at_idx", -1) >= bar_idx - self.DENSITY_WINDOW_BARS
            )
        except Exception:
            return 0, "normal"

        if recent_count >= self.DENSITY_HIGH_THRESH:
            return recent_count, "dense"
        elif recent_count < self.DENSITY_LOW_THRESH:
            return recent_count, "sparse"
        return recent_count, "normal"

    def _classify_regime(
        self,
        er:        float,
        der:       float,
        atr_pct:   float,
        direction: int,
        structure: str = "unknown",
    ) -> str:
        """ER + DER + ATR 백분위 + 시장 구조 → 레짐 라벨 결정.

        레짐 결정 트리:
          ATR 백분위 > 75% + ER < 0.35  → volatile (급변동)
          ER > 0.60                     → trend_strong_* (방향에 따라)
          ER > 0.35                     → trend_weak_*
          ER ≤ 0.35 + ATR > 75%        → chop_high_vol
          ER ≤ 0.35 + ATR ≤ 75%        → chop_low_vol
          
        [개선] 시장 구조 보정:
          structure="uptrend" + ER ≥ 0.35 → 강한 상승 추세로 강화
          structure="downtrend" + ER ≥ 0.35 → 강한 하락 추세로 강화
        """
        # 급변동 우선 (ATR 급등 + ER 낮음 = 방향 없는 폭발적 변동)
        if atr_pct > self.ATR_HIGH_THRESH and er < self.ER_WEAK:
            return "volatile"

        # 시장 구조 보정: 명확한 추세 구조인 경우 레짐 강화
        if structure == "uptrend" and er >= self.ER_WEAK:
            # 상승 구조 + 추세 ER → 강한 상승 추세로 강화
            if er >= self.ER_STRONG:
                return "trend_strong_up"
            return "trend_weak_up"
        
        if structure == "downtrend" and er >= self.ER_WEAK:
            # 하락 구조 + 추세 ER → 강한 하락 추세로 강화
            if er >= self.ER_STRONG:
                return "trend_strong_dn"
            return "trend_weak_dn"

        # 기존 ER 기반 분류
        if er >= self.ER_STRONG:
            # DER 부호로 방향 보정 (DER 양수=상승추세, 음수=하락추세)
            if der >= 0.0:
                return "trend_strong_up"
            else:
                return "trend_strong_dn"

        if er >= self.ER_WEAK:
            if der >= 0.0:
                return "trend_weak_up"
            else:
                return "trend_weak_dn"

        # 횡보 구간
        if atr_pct > self.ATR_HIGH_THRESH:
            return "chop_high_vol"
        return "chop_low_vol"
