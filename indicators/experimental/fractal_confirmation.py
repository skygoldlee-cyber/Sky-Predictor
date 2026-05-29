"""Fractal Confirmation Layer
=================================
빌 윌리엄스 프랙탈 확장형.

핵심 특징
---------
- 좌우 N봉 극값 기반 Fractal 탐지 (기본 2봉 좌우)
- 거래량 spike 필터로 fake pivot 억제
- ATRAdaptivePivot 과 독립적으로 사용하거나,
  ``FractalConfirmedPivot`` 으로 조합하여 확증 레이어로 활용 가능

사용 예시
---------
::

    from indicators import FractalConfirmation, FractalConfig

    frac = FractalConfirmation(FractalConfig(lookback=2, volume_spike_ratio=1.5))

    for bar in bars:
        result = frac.update(bar.high, bar.low, bar.close, bar.volume)
        if result.fractal_high:
            # N봉 전 고점이 프랙탈 고점으로 확정됨
            print(f"Fractal HIGH @ {result.fractal_high_price}")

조합 예시 (ATRAdaptivePivot + Fractal 확증)
------------------------------------------
::

    from indicators import ATRAdaptivePivot, FractalConfirmation, FractalConfig

    pivot = ATRAdaptivePivot()
    frac  = FractalConfirmation()

    for h, l, c, v, t in bars:
        ps = pivot.update(h, l, c, bar_time=t)
        fs = frac.update(h, l, c, v)

        # ATR Pivot 신호 + Fractal 확증 → 더 높은 신뢰도
        if ps.new_pivot_signal != "none" and fs.fractal_confirmed:
            ...
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FractalConfig:
    """Fractal Confirmation 설정.

    Parameters
    ----------
    lookback:
        좌우 비교 봉 수 (기본 2). 고점 프랙탈: high[i] > high[i±1..lookback].
    volume_spike_ratio:
        거래량 급증 배율. 직전 N봉 평균 × 이 비율 이상이어야 프랙탈 유효 판정.
        1.0 = 거래량 필터 없음.
    volume_lookback:
        거래량 평균 계산 구간 (봉 수).
    min_bar_gap:
        연속 프랙탈 최소 봉 간격. 너무 가까운 프랙탈은 noise 처리.
    max_fractals:
        보관할 최대 프랙탈 수.
    """
    lookback:            int   = 2
    volume_spike_ratio:  float = 1.3
    volume_lookback:     int   = 10
    min_bar_gap:         int   = 3
    max_fractals:        int   = 30


@dataclass
class FractalPoint:
    """확정된 프랙탈 피봇."""
    index:      int
    price:      float
    ftype:      str          # "high" | "low"
    volume:     float = 0.0
    vol_ratio:  float = 0.0  # volume / avg_volume
    bar_time:   Optional[str] = None


@dataclass
class FractalState:
    """매 봉 업데이트 후 반환 상태."""
    # 이번 봉에서 새로 확정된 프랙탈 (lookback 봉 지연)
    fractal_high:         bool  = False
    fractal_high_price:   float = 0.0
    fractal_high_time:    Optional[str] = None
    fractal_high_bar:     int   = -1

    fractal_low:          bool  = False
    fractal_low_price:    float = 0.0
    fractal_low_time:     Optional[str] = None
    fractal_low_bar:      int   = -1

    # 이번 봉에 프랙탈 확정이 있었는지 (high or low)
    fractal_confirmed:    bool  = False

    # 거래량 정보
    current_vol_ratio:    float = 0.0  # 현재 봉 volume / 평균

    # 최근 프랙탈 목록
    recent_fractals:      List[FractalPoint] = field(default_factory=list)


class FractalConfirmation:
    """빌 윌리엄스 프랙탈 확장형 변곡점 탐지기.

    ``lookback`` 봉 이후에 프랙탈을 확정하므로, 실시간에서는
    lookback 봉의 지연이 발생합니다.
    (기본 2봉 → 매 업데이트 시 현재 봉 기준 2봉 전의 프랙탈을 판단)
    """

    def __init__(self, config: Optional[FractalConfig] = None) -> None:
        self.config = config or FractalConfig()
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        buf = max(cfg.lookback * 4 + 20, 60)

        self._highs:   deque = deque(maxlen=buf)
        self._lows:    deque = deque(maxlen=buf)
        self._volumes: deque = deque(maxlen=buf)
        self._times:   deque = deque(maxlen=buf)   # "HH:MM" or None

        self._bar_idx: int = 0
        self._fractals: List[FractalPoint] = []
        self._last_fractal_bar: Dict[str, int] = {"high": -1, "low": -1}
        self._state = FractalState()

    def update(
        self,
        high:     float,
        low:      float,
        close:    float,
        volume:   float = 1.0,
        bar_time: Any   = None,
    ) -> FractalState:
        """1분봉 입력 → 프랙탈 상태 반환."""
        self._highs.append(high)
        self._lows.append(low)
        self._volumes.append(max(volume, 1.0))
        self._times.append(self._fmt(bar_time))

        n = len(self._highs)
        lb = self.config.lookback

        # 이번 봉에서는 lookback 봉 전 피봇을 판단
        # 최소 (2*lookback+1) 봉 이상이어야 탐지 가능
        new_high = False
        new_low  = False
        fh_price = 0.0
        fh_time  = None
        fh_bar   = -1
        fl_price = 0.0
        fl_time  = None
        fl_bar   = -1

        if n >= lb * 2 + 1:
            # 판단 대상 봉: 버퍼 중앙 = index (n-1-lb)
            ci = n - 1 - lb   # center index in deque

            h_list = list(self._highs)
            l_list = list(self._lows)
            v_list = list(self._volumes)
            t_list = list(self._times)

            ch = h_list[ci]
            cl = l_list[ci]

            left_highs  = h_list[ci - lb : ci]
            right_highs = h_list[ci + 1 : ci + lb + 1]
            left_lows   = l_list[ci - lb : ci]
            right_lows  = l_list[ci + 1 : ci + lb + 1]

            # ── 프랙탈 고점 판단 ──────────────────────────────────────────
            is_frac_high = (
                len(left_highs)  == lb and
                len(right_highs) == lb and
                all(ch > x for x in left_highs) and
                all(ch > x for x in right_highs)
            )

            # ── 프랙탈 저점 판단 ──────────────────────────────────────────
            is_frac_low = (
                len(left_lows)  == lb and
                len(right_lows) == lb and
                all(cl < x for x in left_lows) and
                all(cl < x for x in right_lows)
            )

            # ── 거래량 spike 필터 ─────────────────────────────────────────
            vol_avg = self._avg_volume(ci, v_list)
            cv      = v_list[ci]
            vol_ratio = cv / vol_avg if vol_avg > 0 else 1.0
            spike   = vol_ratio >= self.config.volume_spike_ratio

            abs_bar_idx = self._bar_idx - lb  # 실제 봉 인덱스

            # ── 고점 확정 ─────────────────────────────────────────────────
            if is_frac_high and spike:
                gap = abs_bar_idx - self._last_fractal_bar.get("high", -999)
                if gap >= self.config.min_bar_gap:
                    fp = FractalPoint(
                        index=abs_bar_idx,
                        price=ch,
                        ftype="high",
                        volume=cv,
                        vol_ratio=vol_ratio,
                        bar_time=t_list[ci],
                    )
                    self._fractals.append(fp)
                    self._last_fractal_bar["high"] = abs_bar_idx
                    new_high = True
                    fh_price = ch
                    fh_time  = t_list[ci]
                    fh_bar   = abs_bar_idx
                    logger.warning(
                        "[Fractal][HIGH] %.2f | time=%s | vol_ratio=%.2f | bar=%d",
                        ch, t_list[ci], vol_ratio, self._bar_idx,
                    )

            # ── 저점 확정 ─────────────────────────────────────────────────
            if is_frac_low and spike:
                gap = abs_bar_idx - self._last_fractal_bar.get("low", -999)
                if gap >= self.config.min_bar_gap:
                    fp = FractalPoint(
                        index=abs_bar_idx,
                        price=cl,
                        ftype="low",
                        volume=cv,
                        vol_ratio=vol_ratio,
                        bar_time=t_list[ci],
                    )
                    self._fractals.append(fp)
                    self._last_fractal_bar["low"] = abs_bar_idx
                    new_low  = True
                    fl_price = cl
                    fl_time  = t_list[ci]
                    fl_bar   = abs_bar_idx
                    logger.warning(
                        "[Fractal][LOW]  %.2f | time=%s | vol_ratio=%.2f | bar=%d",
                        cl, t_list[ci], vol_ratio, self._bar_idx,
                    )

            if len(self._fractals) > self.config.max_fractals * 2:
                self._fractals = self._fractals[-self.config.max_fractals:]

            # 상태 갱신
            self._state.fractal_high         = new_high
            self._state.fractal_high_price   = fh_price
            self._state.fractal_high_time    = fh_time
            self._state.fractal_high_bar     = fh_bar
            self._state.fractal_low          = new_low
            self._state.fractal_low_price    = fl_price
            self._state.fractal_low_time     = fl_time
            self._state.fractal_low_bar      = fl_bar
            self._state.fractal_confirmed    = new_high or new_low
            self._state.current_vol_ratio    = vol_ratio
            self._state.recent_fractals      = list(self._fractals[-self.config.max_fractals:])
        else:
            # 버퍼 부족
            self._state = FractalState()

        self._bar_idx += 1
        return self._state

    def get_transformer_features(self) -> Dict[str, float]:
        """프랙탈 지표를 PriceTransformer feature dict 로 반환."""
        s = self._state
        return {
            "frac_confirmed":     float(s.fractal_confirmed),
            "frac_high":          float(s.fractal_high),
            "frac_low":           float(s.fractal_low),
            "frac_vol_ratio":     float(min(s.current_vol_ratio / 3.0, 1.0)),
        }

    @property
    def state(self) -> FractalState:
        return self._state

    @property
    def confirmed_fractals(self) -> List[FractalPoint]:
        return list(self._fractals)

    # ── 내부 유틸 ────────────────────────────────────────────────────────────

    def _avg_volume(self, ci: int, v_list: List[float]) -> float:
        """ci 봉 이전 volume_lookback 봉 평균."""
        lb = self.config.volume_lookback
        start = max(0, ci - lb)
        vals  = v_list[start:ci]
        if not vals:
            return 1.0
        return sum(vals) / len(vals)

    def _fmt(self, bar_time: Any) -> Optional[str]:
        if bar_time is None:
            return None
        try:
            import pandas as pd
            return pd.Timestamp(bar_time).strftime("%H:%M")
        except Exception:
            s = str(bar_time).strip()
            return s[:5] if len(s) >= 5 and ":" in s else None
