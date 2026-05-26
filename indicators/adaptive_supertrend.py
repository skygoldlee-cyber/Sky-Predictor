"""Adaptive SuperTrend Indicator
================================
ATR 배수(multiplier)와 기간(period)을 시장 변동성/추세 강도에 따라
동적으로 조정하는 Adaptive SuperTrend 구현.

수정된 버그
-----------
[FIX-1] bars_in_trend 플립 봉 누적 버그
    방향 전환 봉에서 bars_in_trend = 0 리셋 후 즉시 +1 되는 문제.
    just_flipped 플래그로 플립 봉에서는 증가를 건너뜀.

[FIX-2] ATR 재초기화 임계값: 비율 기준으로 통일
    절댓값(period_change > atr_max * 0.3) 대신
    비율(period_change_ratio > 0.5) 사용으로 설정값 무관하게 일관성 확보.

[FIX-3] LLM advice 딕셔너리 키 오류
    s.trend_strength('weak'/'neutral'/'strong') 와 매칭되지 않는
    'uptrend'/'downtrend' 키를 s.direction 기반 조건으로 교체.

[FIX-4] _prev_adx 초기값 0.0 -> 25.0
    워밍업 중 fallback 경로와 초기값 통일.
"""

import logging
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List

try:
    from .wilder_smooth import WilderRMA
except ImportError:
    from wilder_smooth import WilderRMA

_logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSuperTrendConfig:
    atr_min_period: int = 7
    atr_max_period: int = 21
    multiplier_min: float = 1.5
    multiplier_max: float = 4.0
    er_period: int = 10
    adx_period: int = 14
    use_bb_correction: bool = True
    bb_period: int = 20
    bb_std: float = 2.0
    smooth_period: int = 3
    use_smooth_for_features: bool = False
    trend_duration_cap_bars: int = 78
    adx_norm_cap: float = 100.0
    adx_mult_norm_cap: float = 60.0
    bb_correction_floor: float = 0.7
    bb_correction_ref_pct: float = 0.05


@dataclass
class SuperTrendState:
    value: float = 0.0
    direction: int = 1
    upper_band: float = 0.0
    lower_band: float = 0.0
    atr: float = 0.0
    adaptive_atr_period: float = 14.0
    adaptive_multiplier: float = 3.0
    efficiency_ratio: float = 0.5
    adx: float = 25.0
    trend_strength: str = "neutral"
    signal: str = "hold"
    bars_in_trend: int = 0
    last_flip_price: float = 0.0


class AdaptiveSuperTrend:
    """Adaptive SuperTrend 지표 (모든 알려진 버그 수정 완료)."""

    def __init__(self, config: Optional[AdaptiveSuperTrendConfig] = None) -> None:
        self.config = config or AdaptiveSuperTrendConfig()
        self._reset_buffers()

    # ──────────────────── 공개 API ────────────────────────

    def reset(self) -> None:
        """상태를 완전 초기화한다. 데이터 소스 변경 또는 강제 재계산 시 사용."""
        self._reset_buffers()

    def update(self, high: float, low: float, close: float) -> SuperTrendState:
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._bars_since_init += 1

        n = len(self._closes)
        cfg = self.config

        # 1. True Range
        if n >= 2:
            pc = float(self._closes[-2])
            tr = max(high - low, abs(high - pc), abs(low - pc))
        else:
            tr = high - low
        self._tr.append(tr)

        # 2. Efficiency Ratio
        er = self._calc_er()
        self._er_values.append(er)

        # 3. 적응형 ATR 기간 (ER 높을수록 짧은 기간)
        ap = cfg.atr_max_period - er * (cfg.atr_max_period - cfg.atr_min_period)
        ap = max(cfg.atr_min_period, int(round(ap)))

        # 4. ATR  [FIX-2: 비율 기준 재초기화]
        if len(self._tr) >= ap:
            prev_ap = int(self._prev_adaptive_period or ap)
            ratio = abs(ap - prev_ap) / max(float(prev_ap), 1.0)
            if (not self._atr_initialized) or (ratio > 0.5):
                atr = float(np.mean(list(self._tr)[-ap:]))
                self._atr_initialized = True
            else:
                alpha = 1.0 / max(float(ap), 1.0)
                atr = float(self._prev_atr) * (1.0 - alpha) + float(tr) * alpha
        else:
            atr = float(np.mean(list(self._tr))) if self._tr else float(tr)

        self._atr_values.append(atr)
        self._prev_atr = float(atr)
        self._prev_adaptive_period = int(ap)

        # 5. ADX
        adx = self._calc_adx(high, low)
        self._adx_values.append(adx)

        # 6. 적응형 멀티플라이어 (ADX 높을수록 타이트)
        adx_norm = min(adx / max(float(cfg.adx_mult_norm_cap), 1e-8), 1.0)
        mult = cfg.multiplier_max - adx_norm * (cfg.multiplier_max - cfg.multiplier_min)

        if cfg.use_bb_correction and len(self._closes) >= cfg.bb_period:
            bb_w = self._calc_bb_width()
            bb_norm = min(bb_w / (close * float(cfg.bb_correction_ref_pct)), 1.0) if close > 0 else 0.0
            mult *= float(cfg.bb_correction_floor) + (1.0 - float(cfg.bb_correction_floor)) * bb_norm

        mult = float(np.clip(mult, cfg.multiplier_min, cfg.multiplier_max))

        # 7. 밴드 계산
        hl2 = (high + low) / 2.0
        upper_band = hl2 + mult * atr
        lower_band = hl2 - mult * atr

        # 8. 밴드 연속성
        prev_dir: Optional[int] = None
        if len(self._super_trend) >= 1:
            prev_dir = int(self._direction[-1])
            pc2 = float(self._closes[-2]) if len(self._closes) >= 2 else close
            if pc2 <= float(self._state.upper_band):
                upper_band = min(upper_band, float(self._state.upper_band))
            if pc2 >= float(self._state.lower_band):
                lower_band = max(lower_band, float(self._state.lower_band))

        if prev_dir is None:
            direction = 1 if close > hl2 else -1
        elif prev_dir == 1:
            direction = -1 if close < lower_band else 1
        else:
            direction = 1 if close > upper_band else -1

        st_value = lower_band if direction == 1 else upper_band

        # 스무딩
        if cfg.smooth_period > 1 and len(self._super_trend) >= cfg.smooth_period:
            alpha_s = 2.0 / (cfg.smooth_period + 1)
            st_value = alpha_s * st_value + (1.0 - alpha_s) * float(self._super_trend[-1])

        self._super_trend.append(st_value)
        self._direction.append(direction)

        # 9. 신호 & bars_in_trend  [FIX-1: just_flipped 패턴]
        signal = "hold"
        just_flipped = False
        if prev_dir is not None:
            if prev_dir == -1 and direction == 1:
                signal = "buy"
                self._state.last_flip_price = float(close)
                self._state.bars_in_trend = 0
                just_flipped = True
            elif prev_dir == 1 and direction == -1:
                signal = "sell"
                self._state.last_flip_price = float(close)
                self._state.bars_in_trend = 0
                just_flipped = True

        if not just_flipped:
            self._state.bars_in_trend += 1

        # 10. 추세 강도
        if adx < 20:
            trend_strength = "weak"
        elif adx < 40:
            trend_strength = "neutral"
        else:
            trend_strength = "strong"

        self._state.value = float(st_value)
        self._state.direction = int(direction)
        self._state.upper_band = float(upper_band)
        self._state.lower_band = float(lower_band)
        self._state.atr = float(atr)
        self._state.adaptive_atr_period = float(ap)
        self._state.adaptive_multiplier = float(mult)
        self._state.efficiency_ratio = float(er)
        self._state.adx = float(adx)
        self._state.trend_strength = trend_strength
        self._state.signal = signal

        return self._state

    def compute_from_df(
        self,
        df: pd.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """DataFrame 전체 처리. 컬럼명 대소문자 자동 탐지."""
        hc = _resolve_col(df, high_col)
        lc = _resolve_col(df, low_col)
        cc = _resolve_col(df, close_col)
        self._reset_buffers()
        rows: List[dict] = []
        for row in df.itertuples(index=False):
            h = float(getattr(row, hc))
            lo = float(getattr(row, lc))
            c = float(getattr(row, cc))
            s = self.update(h, lo, c)
            bfd = float(s.lower_band if s.direction == 1 else s.upper_band) \
                if not self.config.use_smooth_for_features else float(s.value)
            rows.append({
                "ast_value":      float(s.value),
                "ast_direction":  float(s.direction),
                "ast_upper":      float(s.upper_band),
                "ast_lower":      float(s.lower_band),
                "ast_atr":        float(s.atr),
                "ast_er":         float(s.efficiency_ratio),
                "ast_adx":        float(s.adx),
                "ast_mult":       float(s.adaptive_multiplier),
                "ast_atr_period": float(s.adaptive_atr_period),
                "ast_signal":     float(1 if s.signal == "buy" else (-1 if s.signal == "sell" else 0)),
                "ast_bars_trend": float(s.bars_in_trend),
                "ast_dist_pct":   float((c - bfd) / bfd) if bfd != 0.0 else 0.0,
            })
        return df.assign(**pd.DataFrame(rows, index=df.index))

    def get_transformer_features(self, close: float) -> Dict[str, float]:
        s = self._state
        cfg = self.config
        bfd = float(s.lower_band if s.direction == 1 else s.upper_band) \
            if not cfg.use_smooth_for_features else float(s.value)
        dist = (close - bfd) / bfd if bfd != 0.0 else 0.0
        return {
            "ast_direction":        float(s.direction),
            "ast_dist_pct":         float(dist),
            "ast_atr_pct":          float(s.atr / close) if close != 0 else 0.0,
            "ast_efficiency_ratio": float(s.efficiency_ratio),
            "ast_adx_norm":         float(min(s.adx / max(float(cfg.adx_norm_cap), 1e-8), 1.0)),
            "ast_mult_norm":        float(
                (s.adaptive_multiplier - cfg.multiplier_min)
                / max(cfg.multiplier_max - cfg.multiplier_min, 1e-8)
            ),
            "ast_trend_duration":   float(min(s.bars_in_trend / max(float(cfg.trend_duration_cap_bars), 1.0), 1.0)),
            "ast_signal":           float(1 if s.signal == "buy" else (-1 if s.signal == "sell" else 0)),
            "ast_band_width_pct":   float((s.upper_band - s.lower_band) / close) if close > 0 else 0.0,
        }

    def get_llm_context(self, close: float, symbol: str = "KP200 선물") -> str:
        s = self._state
        cfg = self.config
        bfd = float(s.lower_band if s.direction == 1 else s.upper_band) \
            if not cfg.use_smooth_for_features else float(s.value)
        dist_pct = (close - bfd) / bfd * 100 if bfd != 0.0 else 0.0
        dist_abs = close - bfd
        atr_pct = s.atr / close * 100 if close > 0 else 0.0

        sig_txt = (
            f"매수 신호 발생 (전환가: {s.last_flip_price:.2f})" if s.signal == "buy" else
            f"매도 신호 발생 (전환가: {s.last_flip_price:.2f})" if s.signal == "sell" else
            f"신호 없음 ({s.bars_in_trend}봉 지속)"
        )
        # [FIX-3] s.direction 기반 advice
        advice = {
            1:  "상승 추세 유지 중 — 매도 신호에 신중하세요.",
            -1: "하락 추세 유지 중 — 매수 신호에 신중하세요.",
        }.get(s.direction, "횡보 구조 — 지지/저항 범위 매매가 유리합니다.")

        str_kor = {"weak": "약한(ADX<20)", "neutral": "중간(ADX 20-40)", "strong": "강한(ADX>40)"}[s.trend_strength]
        return (
            f"[Adaptive SuperTrend - {symbol}]\n"
            f"현재가: {close:.2f}  ST: {s.value:.2f} ({'+' if dist_abs >= 0 else ''}{dist_abs:.2f}, "
            f"{'+' if dist_pct >= 0 else ''}{dist_pct:.1f}%)\n"
            f"방향: {'상승' if s.direction == 1 else '하락'}  강도: {str_kor} (ADX={s.adx:.1f})\n"
            f"신호: {sig_txt}\n"
            f"ER={s.efficiency_ratio:.3f}  ATR기간={s.adaptive_atr_period:.0f}봉  "
            f"ATR={s.atr:.2f}({atr_pct:.2f}%)  Mult={s.adaptive_multiplier:.2f}\n"
            f"밴드: {s.lower_band:.2f}~{s.upper_band:.2f}\n{advice}"
        )

    @property
    def state(self) -> SuperTrendState:
        return self._state

    # ────────────────── 내부 메서드 ───────────────────────

    def _reset_buffers(self) -> None:
        cfg = self.config
        max_buf = max(cfg.atr_max_period, cfg.adx_period, cfg.bb_period, cfg.er_period) * 3
        self._highs:       deque = deque(maxlen=max_buf)
        self._lows:        deque = deque(maxlen=max_buf)
        self._closes:      deque = deque(maxlen=max_buf)
        self._atr_values:  deque = deque(maxlen=max_buf)
        self._er_values:   deque = deque(maxlen=max_buf)
        self._adx_values:  deque = deque(maxlen=max_buf)
        self._super_trend: deque = deque(maxlen=max_buf)
        self._direction:   deque = deque(maxlen=max_buf)
        self._tr:          deque = deque(maxlen=max_buf)

        self._rma_tr       = WilderRMA(period=int(cfg.adx_period))
        self._rma_plus_dm  = WilderRMA(period=int(cfg.adx_period))
        self._rma_minus_dm = WilderRMA(period=int(cfg.adx_period))
        self._rma_adx      = WilderRMA(period=int(cfg.adx_period))

        self._prev_adx:            float = 25.0   # [FIX-4]
        self._adx_initialized:     bool  = False
        self._bars_since_init:     int   = 0
        self._prev_atr:            float = 0.0
        self._atr_initialized:     bool  = False
        self._prev_adaptive_period: int  = int(cfg.atr_max_period)
        self._state = SuperTrendState()

    def _calc_er(self) -> float:
        cfg = self.config
        n = len(self._closes)
        if n < cfg.er_period + 1:
            return 0.5
        cs = list(self._closes)[-(cfg.er_period + 1):]
        direction = abs(cs[-1] - cs[0])
        volatility = sum(abs(cs[i] - cs[i - 1]) for i in range(1, len(cs)))
        if volatility < 1e-10:
            return 0.0
        return float(np.clip(direction / volatility, 0.0, 1.0))

    def _calc_adx(self, high: float, low: float) -> float:
        if len(self._closes) < 2:
            return 25.0
        ph = float(self._highs[-2])
        pl = float(self._lows[-2])
        up = high - ph;  dn = pl - low
        pdm = max(up, 0.0) if up > dn else 0.0
        mdm = max(dn, 0.0) if dn > up else 0.0
        if pdm == mdm:
            pdm = mdm = 0.0
        tr = float(self._tr[-1])
        try:
            st = float(self._rma_tr.update(float(tr)))
            sp = float(self._rma_plus_dm.update(float(pdm)))
            sm = float(self._rma_minus_dm.update(float(mdm)))
        except Exception as exc:
            _logger.warning("ADX RMA error at bar %d: %s", self._bars_since_init, exc)
            return float(self._prev_adx)
        if not (self._rma_tr.ready and self._rma_plus_dm.ready and self._rma_minus_dm.ready):
            return 25.0
        if st < 1e-10:
            return float(self._prev_adx)
        pdi = 100.0 * sp / st;  mdi = 100.0 * sm / st
        di_sum = pdi + mdi
        dx = 100.0 * abs(pdi - mdi) / di_sum if di_sum > 1e-10 else 0.0
        try:
            adx = float(self._rma_adx.update(float(dx)))
        except Exception:
            adx = float(self._prev_adx)
        self._prev_adx = float(adx)
        return float(np.clip(adx, 0.0, 100.0))

    def _calc_bb_width(self) -> float:
        cfg = self.config
        cs = np.array(list(self._closes)[-cfg.bb_period:], dtype=float)
        std = np.std(cs, ddof=1)
        return float(2.0 * cfg.bb_std * std)


    # ── SkyEbest 래퍼 호환 메서드 ────────────────────────

    def get_super_trend(
        self,
        df: pd.DataFrame,
        lookback: int,
        multiplier: float,
        smooth_period: int = 3,
    ):
        """SkyEbest MyTechnicalAnalysis.get_super_trend() 호환 래퍼.

        Returns (st_array, upper_array, lower_array)  — numpy float arrays.

        파라미터
        --------
        smooth_period : 1 = 스무딩 없음, 3(기본) = EMA(3) 스무딩.
            두 메서드를 통일하려면 동일 값을 전달하세요.
        """
        length = 0 if df is None else len(df)
        nan_a = np.full(length, np.nan, dtype=float)
        if df is None or df.empty or length <= max(2, int(lookback)):
            return nan_a, nan_a.copy(), nan_a.copy()

        lb = int(lookback); mt = float(multiplier)
        ast = AdaptiveSuperTrend(AdaptiveSuperTrendConfig(
            atr_min_period=max(2, lb // 2), atr_max_period=lb,
            multiplier_min=max(0.5, mt * 0.5), multiplier_max=mt * 1.5,
            adx_period=lb, use_bb_correction=False,
            smooth_period=smooth_period,
        ))
        hc = _resolve_col(df, "high");  lc = _resolve_col(df, "low");  cc = _resolve_col(df, "close")
        ha = pd.to_numeric(df[hc], errors="coerce").to_numpy(float)
        la = pd.to_numeric(df[lc], errors="coerce").to_numpy(float)
        ca = pd.to_numeric(df[cc], errors="coerce").to_numpy(float)
        st = nan_a.copy(); ub = nan_a.copy(); lb_ = nan_a.copy()
        for i in range(length):
            if any(np.isnan([ha[i], la[i], ca[i]])):
                continue
            s = ast.update(ha[i], la[i], ca[i])
            if i >= lb:
                st[i] = s.value; ub[i] = s.upper_band; lb_[i] = s.lower_band
        return st, ub, lb_


def _resolve_col(df: pd.DataFrame, name: str) -> str:
    for col in df.columns:
        if str(col).lower() == name.lower():
            return col
    return name
