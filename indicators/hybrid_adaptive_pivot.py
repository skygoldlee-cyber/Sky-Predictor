"""Hybrid Adaptive Pivot Detector
================================
ATR 기반 변동성 적응 + 퍼센트 기반 직관적 설정을 결합한 하이브리드 변곡점 탐지기.

설계 원칙
---------
- ATRAdaptivePivot의 변동성 적응성 유지
- PercentAdaptivePivot의 직관적 퍼센트 설정 유지
- 두 클래스의 장점 결합: ATR 가중치 + 퍼센트 가중치 혼합
- cancel_ratio 파라미터화 (PercentAdaptivePivot 동일)
- 즉시 확정 지원 (confirmation_bars=0)
- 명시적 방향 복귀 (상태 일치성 보장)
- 이중 파동 필터 (퍼센트 + ATR)

하이브리드 임계값 계산
---------------------
    thr_pct = close × base_pct/100 × er_multiplier × session_scale
    thr_atr = atr × base_multiplier × er_multiplier × session_scale
    thr_hybrid = (1 - atr_weight) × thr_pct + atr_weight × thr_atr

    atr_weight = 0 → 퍼센트만 사용
    atr_weight = 1 → ATR만 사용
    atr_weight = 0.5 → 둘 다 혼합 (기본)

장점 결합
---------
- 변동성 적응 (ATR 기반)
- 직관적 설정 (퍼센트 기반)
- 가격 수준 독립성 (ATR 기반)
- 이중 필터 (퍼센트 + ATR)
- cancel_ratio 파라미터화
- 즉시 확정 지원
- 명시적 방향 복귀

사용 예시
---------
::

    from indicators import HybridAdaptivePivot, HybridAdaptivePivotConfig

    cfg = HybridAdaptivePivotConfig(
        base_pct=0.3,
        base_multiplier=2.0,
        atr_weight=0.5,  # 50% ATR, 50% 퍼센트 혼합
    )
    pivot = HybridAdaptivePivot(cfg)

    for h, l, c, t in bars:
        state = pivot.update(h, l, c, bar_time=t)
        features = pivot.get_transformer_features(c)
        score = pivot.pivot_score   # 0.0 ~ 1.0

    if pivot.pivot_score > 0.6 and state.new_pivot_signal in ("new_high", "new_low"):
        ...
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .wilder_smooth import WilderRMA
except ImportError:
    from wilder_smooth import WilderRMA  # type: ignore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WARMUP_BARS = 20
DEFAULT_CONFIRMATION_BARS = 0  # 즉시 확정으로 변경
DEFAULT_MIN_BAR_GAP = 1  # 최소 봉 간격 감소
DEFAULT_CANCEL_RATIO = 0.1  # 취소 비율 감소 (더 민감하게)
MULTIPLIER_CLIP_MIN_FACTOR = 0.5
MULTIPLIER_CLIP_MAX_FACTOR = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 열거형 / 데이터 클래스
# ─────────────────────────────────────────────────────────────────────────────

class PivotType(Enum):
    HIGH = "high"
    LOW  = "low"


@dataclass
class PivotPoint:
    """확정된 변곡점 정보."""
    index:      int
    price:      float
    pivot_type: PivotType
    pct:        float            # 파동 퍼센트 크기 (직전 확정 피봇 대비)
    atr:        float            # 해당 시점의 ATR
    bar_time:   Optional[str] = None   # "HH:MM"
    is_major:   bool = False


@dataclass
class HybridAdaptivePivotConfig:
    """Hybrid Adaptive Pivot 설정.

    Parameters
    ----------
    base_pct:
        기본 퍼센트 임계값 (%).
        퍼센트 기반 임계값: thr_pct = close × base_pct/100 × er_multiplier × session_scale.
    base_multiplier:
        기본 ATR 배수.
        ATR 기반 임계값: thr_atr = atr × base_multiplier × er_multiplier × session_scale.
    atr_weight:
        ATR 가중치 (0~1).
        0: 퍼센트만 사용, 1: ATR만 사용, 0.5: 둘 다 혼합.
        기본 0.5 (균형 혼합).
    atr_period:
        ATR 계산 주기 (WilderRMA). 기본 14.
    multiplier_min / multiplier_max:
        ER 기반 동적 배수 하한/상한.
        ER ↑ 추세 강함 → 배수 크게(노이즈 차단).
        ER ↓ 횡보 → 배수 작게(민감도 회복).
    er_period:
        표준 Kaufman ER 계산 구간.
    confirmation_bars:
        후보 등록 후 N봉 유지 확인 후 확정.
        0 = 즉시 확정(실시간 반응 최대).
    min_wave_pct:
        파동 크기가 이 퍼센트(%) 미만이면 후보 등록 차단 (소파동 필터).
    min_wave_atr_ratio:
        파동 크기가 ATR × 이 비율 미만이면 후보 등록 차단 (소파동 필터).
    max_pivots:
        보관할 최대 확정 피봇 수.
    session_multiplier_table:
        시간대별 배율 테이블.
        형식: List[("HH:MM", "HH:MM", multiplier_scale)]
        예) [("09:00","09:30", 1.5), ("14:30","15:20", 0.8)]
    warmup_bars:
        ATR/ER 안정화에 필요한 최소 봉 수. 미만이면 신호 미출력.
    cancel_ratio:
        pending 취소 판단 비율. 후보 대비 되돌림이 threshold × 이 비율 미만이면 취소.
        기본 0.3.
    """
    base_pct:             float = 0.3
    base_multiplier:      float = 2.0
    atr_weight:          float = 0.5
    atr_period:          int   = 14
    multiplier_min:       float = 0.8
    multiplier_max:       float = 2.0
    er_period:           int   = 10
    confirmation_bars:   int   = DEFAULT_CONFIRMATION_BARS
    min_wave_pct:        float = 0.05  # 최소 파동 퍼센트 감소
    min_wave_atr_ratio:  float = 0.2  # ATR 비율 감소
    max_pivots:          int   = 30
    session_multiplier_table: List[Tuple[str, str, float]] = field(
        default_factory=list
    )
    warmup_bars:         int   = DEFAULT_WARMUP_BARS
    cancel_ratio:        float = DEFAULT_CANCEL_RATIO

    # Layer C: Fractal 교차 확증
    use_fractal_confirmation: bool = False
    fractal_lookback: int = 2
    fractal_volume_spike: float = 1.3
    fractal_price_tolerance_pct: float = 0.3
    fractal_bonus: float = 0.15

    # Layer B: AdaptiveParamEngine 연결
    use_adaptive_engine: bool = False
    regime_atr_weight_table: dict = field(default_factory=lambda: {
        "trend_strong_up":  0.75,
        "trend_strong_dn":  0.75,
        "trend_weak_up":    0.55,
        "trend_weak_dn":    0.55,
        "chop_low_vol":     0.35,
        "chop_high_vol":    0.85,
        "volatile":         0.90,
        "unknown":          0.50,
    })

    def __post_init__(self) -> None:
        if self.multiplier_max < self.multiplier_min:
            self.multiplier_min, self.multiplier_max = (
                self.multiplier_max, self.multiplier_min
            )
        if not 0.0 <= self.atr_weight <= 1.0:
            raise ValueError("atr_weight must be between 0.0 and 1.0")


@dataclass
class HybridAdaptivePivotState:
    """매 봉 update() 후 반환되는 상태 객체."""
    # 최근 확정 피봇 (미확정 시 NaN)
    last_high:        float = float("nan")
    last_low:         float = float("nan")
    last_high_idx:    int   = -1
    last_low_idx:     int   = -1
    last_high_time:   Optional[str] = None
    last_low_time:    Optional[str] = None

    # 신호 : "new_high" | "new_low" | "none"
    new_pivot_signal: str   = "none"

    # 시장 구조
    structure:        str   = "unknown"   # uptrend / downtrend / ranging / unknown
    direction:        int   = 0           # +1 상승탐색, -1 하락탐색, 0 미결정

    # 지표값
    atr:              float = 0.0
    threshold_abs:    float = 0.0        # 현재 봉 reversal threshold (pt)
    threshold_pct:    float = 0.0        # threshold / close × 100
    efficiency_ratio: float = 0.0

    # 후보 피봇 (미확정)
    pending_type:     Optional[str] = None   # "high" | "low"
    pending_price:    float = 0.0
    pending_time:     Optional[str] = None
    pending_remaining: int   = 0

    # 피봇 목록 (최근 N개)
    recent_pivots:    List[PivotPoint] = field(default_factory=list)

    # Pivot Score (0~1)
    pivot_score:      float = 0.0

    # 파동 / 경과 봉
    wave_size_pct:    float = 0.0
    bars_since_pivot: int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# 메인 클래스
# ─────────────────────────────────────────────────────────────────────────────

class HybridAdaptivePivot:
    """하이브리드 적응형 피봇 탐지기 (ATR + 퍼센트 혼합).

    ATR 기반 변동성 적응 + 퍼센트 기반 직관적 설정을 결합.
    """

    def __init__(self, config: Optional[HybridAdaptivePivotConfig] = None) -> None:
        self.config = config or HybridAdaptivePivotConfig()
        self._symbol: str = "KP200 선물"
        
        # Layer B: AdaptiveParamEngine (선택적)
        self._adaptive_engine = None
        self._last_regime = "unknown"
        self._last_eff_atr_weight = self.config.atr_weight
        if self.config.use_adaptive_engine:
            try:
                from .adaptive_param_engine import AdaptiveParamEngine
                self._adaptive_engine = AdaptiveParamEngine(self.config)
            except ImportError:
                pass
        
        # Layer C: Fractal 교차 확증 (선택적)
        self._fractal = None
        self._last_fractal = None
        if self.config.use_fractal_confirmation:
            try:
                from .experimental.fractal_confirmation import FractalConfirmation, FractalConfig
                self._fractal = FractalConfirmation(FractalConfig(
                    lookback=self.config.fractal_lookback,
                    volume_spike_ratio=self.config.fractal_volume_spike,
                ))
            except ImportError:
                pass
        
        self.reset()

    # ── 공개 API ────────────────────────────────────────────────────────────

    def set_symbol(self, symbol: str) -> None:
        self._symbol = symbol

    def reset(self) -> None:
        """완전 초기화."""
        cfg = self.config
        _buf = max(cfg.atr_period * 10, 200)

        # OHLC 버퍼
        self._highs:  deque = deque(maxlen=_buf)
        self._lows:   deque = deque(maxlen=_buf)
        self._closes: deque = deque(maxlen=_buf)
        self._tr:     deque = deque(maxlen=_buf)

        # ATR
        self._atr_rma   = WilderRMA(period=cfg.atr_period)
        self._atr_values: deque = deque(maxlen=_buf)

        # 상태
        self._bar_idx:   int   = 0
        self._direction: int   = 0      # 0 미결정, +1 상승탐색, -1 하락탐색
        self._pending_high:     float = 0.0
        self._pending_low:      float = float("inf")
        self._pending_high_idx: int   = -1
        self._pending_low_idx:  int   = -1

        # 후보 확인 창
        self._pending_confirm: Optional[Dict[str, Any]] = None
        self._pending_confirm_bar: int = -1

        # 확정 피봇
        self._pivots:             List[PivotPoint] = []
        self._last_confirmed_bar: int = -1

        # 시각 맵 (bar_idx → "HH:MM")
        self._hhmm_map: OrderedDict = OrderedDict()

        # 상태 객체
        self._state = HybridAdaptivePivotState()

    # ── 메인 업데이트 ────────────────────────────────────────────────────────

    def update(
        self,
        high:     float,
        low:      float,
        close:    float,
        bar_time: Any = None,
        open:     float = 0.0,
        volume:   float = 1.0,
    ) -> HybridAdaptivePivotState:
        """1분봉 데이터 입력 → 상태 갱신.

        Parameters
        ----------
        high, low, close : float
            OHLC 가격 데이터.
        bar_time : Any
            봉 시각. pandas Timestamp, "HH:MM:SS", "HH:MM" 모두 허용.
        open, volume : float
            현재 미사용. 인터페이스 호환용.

        Returns
        -------
        HybridAdaptivePivotState
        """
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._remember_time(bar_time)
        n = len(self._closes)

        # ── 1. True Range / ATR ─────────────────────────────────────────────
        if n >= 2:
            pc = self._closes[-2]
            tr = max(high - low, abs(high - pc), abs(low - pc))
        else:
            tr = high - low
        self._tr.append(tr)
        atr = self._atr_rma.update(tr)
        self._atr_values.append(atr)

        # ── 2. ER 계산 ─────────────────────────────────────────────────────
        er = self._calc_er()

        # ── Layer B: 레짐 기반 atr_weight 동적 조정 ──────────────────────────
        effective_atr_weight = self.config.atr_weight
        if self._adaptive_engine is not None and len(self._closes) >= self.config.warmup_bars:
            # PivotPoint를 SwingPoint 인터페이스로 어댑팅
            class _PivotAdapter:
                """PivotPoint를 SwingPoint 인터페이스로 어댑팅."""
                def __init__(self, p):
                    self._p = p
                confirmed = True
                @property
                def confirmed_at_idx(self):
                    return self._p.index
                @property
                def price(self):
                    return self._p.price
                @property
                def swing_type(self):
                    return "high" if self._p.pivot_type == PivotType.HIGH else "low"

            adj = self._adaptive_engine.compute(
                atr_values=list(self._atr_values),
                all_swings=[_PivotAdapter(p) for p in self._pivots],
                bar_idx=self._bar_idx,
                er=er,
                der=0.0,
                direction=self._direction,
                last_confirmed_bar_idx=self._last_confirmed_bar,
                structure=self._state.structure,
            )
            table = self.config.regime_atr_weight_table
            effective_atr_weight = table.get(adj.regime_label, self.config.atr_weight)
            self._last_regime = adj.regime_label
        self._last_eff_atr_weight = effective_atr_weight

        # ── Layer C: Fractal 업데이트 (volume=1.0 기본, volume 파라미터 사용 시 전달)
        fractal_result = None
        if self._fractal is not None:
            fractal_result = self._fractal.update(high, low, close, volume)
        self._last_fractal = fractal_result

        # ── 3. 하이브리드 동적 threshold ───────────────────────────────────────
        thr_abs, thr_pct = self._calc_threshold(close, atr, er, bar_time, atr_weight_override=effective_atr_weight)

        # ── 4. 웜업 중이면 신호 미출력 ─────────────────────────────────────
        signal = "none"
        if n >= self.config.warmup_bars:
            signal = self._run_logic(high, low, close, atr, thr_abs)

        # ── 5. 상태 갱신 ────────────────────────────────────────────────────
        self._bar_idx += 1
        self._update_state(signal, atr, thr_abs, thr_pct, er, close)
        return self._state

    # ── Transformer / LLM 출력 ───────────────────────────────────────────────

    def get_transformer_features(self, close: float) -> Dict[str, float]:
        """ATRAdaptivePivot azz_* 키 완전 호환 + hap_* 고유 키."""
        s = self._state

        def _fin(v: float, fb: float = 0.0) -> float:
            try:
                x = float(v)
                return x if math.isfinite(x) else fb
            except Exception:
                return fb

        # ── 방향 / 구조 ──────────────────────────────────────────────────
        dir_f  = float(s.direction)
        str_up = float(s.structure == "uptrend")
        str_dn = float(s.structure == "downtrend")
        str_rg = float(s.structure == "ranging")

        # ── 최근 피봇 거리 정규화 (±5% clip → /0.05) ─────────────────────
        def _dist(price: float) -> float:
            if close <= 0 or not math.isfinite(price) or price <= 0:
                return 0.0
            raw = (price - close) / close
            return float(np.clip(raw, -0.05, 0.05) / 0.05)

        # ── 파동 크기 ─────────────────────────────────────────────────────
        wave_norm = _fin(min(s.wave_size_pct / 10.0, 1.0))

        # ── pending 피처 ─────────────────────────────────────────────────
        pc = self._pending_confirm
        if isinstance(pc, dict) and pc:
            pt           = pc.get("type", "")
            pp           = float(pc.get("price") or 0.0)
            rem          = int(pc.get("remaining") or 0)
            cb           = max(float(self.config.confirmation_bars), 1.0)
            pend_type    = 1.0 if pt == "high" else -1.0
            pend_dist    = _dist(pp)
            pend_urgency = float(np.clip(1.0 - rem / cb, 0.0, 1.0))
            waited       = (
                self._bar_idx - self._pending_confirm_bar
                if self._pending_confirm_bar >= 0 else 0
            )
            pend_age     = float(math.exp(-waited / 5.0))
        else:
            pend_type = pend_dist = pend_urgency = pend_age = 0.0

        # ── bars_since_pivot ─────────────────────────────────────────────
        bsp_norm = _fin(min(s.bars_since_pivot / 50.0, 1.0))

        # ── support / resistance 거리 ─────────────────────────────────────
        highs  = [p.price for p in self._pivots if p.pivot_type == PivotType.HIGH and p.price > close]
        lows   = [p.price for p in self._pivots if p.pivot_type == PivotType.LOW  and p.price < close]
        resist = min(highs) if highs else 0.0
        supprt = max(lows)  if lows  else 0.0
        res_dist = _fin(min(abs(_dist(resist)), 1.0)) if resist > 0 else 0.0
        sup_dist = _fin(min(abs(_dist(supprt)), 1.0)) if supprt > 0 else 0.0

        # ── 피봇 스코어 ──────────────────────────────────────────────────
        pivot_score = _fin(s.pivot_score)

        return {
            # ── AdaptiveZigZag / ATRAdaptivePivot 호환 키 ─────────────────
            "azz_direction":         dir_f,
            "azz_last_high":         _dist(s.last_high),
            "azz_last_low":          _dist(s.last_low),
            "azz_wave_size_pct":     wave_norm,
            "azz_support_dist_pct":  sup_dist,
            "azz_res_dist_pct":      res_dist,
            "azz_bars_since_swing":  bsp_norm,
            "azz_higher_highs":      float(s.structure == "uptrend"),
            "azz_lower_lows":        float(s.structure == "downtrend"),
            "azz_new_swing":         float(
                1  if s.new_pivot_signal == "new_high" else
                -1 if s.new_pivot_signal == "new_low"  else 0
            ),
            "azz_swing_recency":     _fin(math.exp(-s.bars_since_pivot / 5.0)),
            "azz_threshold_pct":     _fin(float(np.clip(s.threshold_pct / 3.0, 0.0, 1.0))),
            "azz_structure_up":      str_up,
            "azz_structure_down":    str_dn,
            "azz_structure_ranging": str_rg,
            "azz_micro_up":          str_up,
            "azz_micro_down":        str_dn,
            "azz_micro_ranging":     str_rg,
            "azz_structure_conf":    _fin(0.7 if s.structure != "unknown" else 0.0),
            "azz_pend_sr_dist":      pend_dist,
            "azz_pending_type":      pend_type,
            "azz_pending_dist":      pend_dist,
            "azz_pending_urgency":   pend_urgency,
            "azz_pending_age":       pend_age,
            "azz_pending_prob":      pivot_score,
            # ── HybridAdaptivePivot 고유 키 ─────────────────────────────
            "hap_atr":               _fin(float(np.clip(s.atr / (close * 0.05), 0.0, 1.0))),
            "hap_atr_weight":        float(self.config.atr_weight),
            "hap_threshold_pct":     _fin(float(np.clip(s.threshold_pct / 3.0, 0.0, 1.0))),
            "hap_pivot_score":       pivot_score,
            # ── Layer B 피처 ─────────────────────────────────────────────
            "hap_regime":            float(self._regime_to_encoding(getattr(self, "_last_regime", "unknown"))),
            "hap_effective_w":       float(getattr(self, "_last_eff_atr_weight", self.config.atr_weight)),
            # ── Layer C 피처 ─────────────────────────────────────────────
            "hap_fractal_bonus":     float(np.clip(self._calc_fractal_bonus() / 0.15, 0.0, 1.0)),
            # ── Layer D 피처 ─────────────────────────────────────────────
            "hap_wave_quality":      float(np.clip(
                s.wave_size_pct / (s.threshold_pct * 3.0), 0.0, 1.0
            ) if s.threshold_pct > 0 else 0.0),
        }

    def _regime_to_encoding(self, regime: str) -> float:
        """레짐 라벨을 숫자로 인코딩."""
        regime_enc = {
            "trend_strong_up": 1.0, "trend_strong_dn": -1.0,
            "trend_weak_up": 0.5,   "trend_weak_dn": -0.5,
            "chop_low_vol": 0.1,    "chop_high_vol": 0.2,
            "volatile": 0.8,        "unknown": 0.0,
        }
        return regime_enc.get(regime, 0.0)

    def get_llm_context(self, close: float) -> str:
        """LLM 프롬프트 삽입용 컨텍스트 텍스트."""
        s = self._state
        dir_kor = {1: "상승", -1: "하락", 0: "미결정"}.get(s.direction, "미결정")
        str_kor = {
            "uptrend":  "상승 구조",
            "downtrend":"하락 구조",
            "ranging":  "횡보 구조",
            "unknown":  "구조 미확정",
        }.get(s.structure, "구조 미확정")

        pivot_list = []
        for p in self._pivots[-6:]:
            t   = "H" if p.pivot_type == PivotType.HIGH else "L"
            maj = "★" if p.is_major else " "
            pivot_list.append(f"  {maj}{p.bar_time or '?'} {t} {p.price:.2f}")

        pc_txt = "피봇후보 없음"
        if isinstance(self._pending_confirm, dict) and self._pending_confirm:
            pt  = "고점" if self._pending_confirm.get("type") == "high" else "저점"
            pp  = float(self._pending_confirm.get("price") or 0.0)
            rem = int(self._pending_confirm.get("remaining") or 0)
            pc_txt = f"{pt} 후보 {pp:.2f} | 확정까지 {rem}봉"

        lh = s.last_high if math.isfinite(s.last_high) else 0.0
        ll = s.last_low  if math.isfinite(s.last_low)  else 0.0

        return (
            f"[HybridAdaptivePivot - {self._symbol}]\n"
            f"현재가: {close:.2f}  방향: {dir_kor}  구조: {str_kor}\n"
            f"신호: {s.new_pivot_signal}\n"
            f"후보: {pc_txt}\n"
            f"최근 고점: {lh:.2f}  저점: {ll:.2f}  파동: {s.wave_size_pct:.2f}%\n"
            f"피봇 목록:\n" + "\n".join(pivot_list or ["  (없음)"]) + "\n"
            f"ATR: {s.atr:.2f}  threshold: {s.threshold_abs:.2f}pt ({s.threshold_pct:.2f}%)  "
            f"ATR가중치: {self.config.atr_weight:.2f}\n"
            f"Pivot Score: {s.pivot_score:.3f}"
        )

    # ── 프로퍼티 ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> HybridAdaptivePivotState:
        return self._state

    @property
    def pivot_score(self) -> float:
        """현재 Pivot Score (0~1)."""
        return self._state.pivot_score

    @property
    def confirmed_pivots(self) -> List[PivotPoint]:
        return list(self._pivots)

    # ── 내부 로직 ────────────────────────────────────────────────────────────

    def _calc_threshold(
        self,
        close:    float,
        atr:      float,
        er:       float,
        bar_time: Any,
        atr_weight_override: Optional[float] = None,
    ) -> Tuple[float, float]:
        """하이브리드 임계값 계산 (퍼센트 + ATR 혼합)."""
        cfg = self.config
        w = atr_weight_override if atr_weight_override is not None else cfg.atr_weight

        # ER 기반 배수 계산
        mult = self._calc_multiplier(er, bar_time)

        # 퍼센트 기반 임계값
        thr_pct = close * cfg.base_pct / 100.0 * mult

        # ATR 기반 임계값
        thr_atr = atr * cfg.base_multiplier * mult

        # 하이브리드 혼합
        thr_hybrid = (1 - w) * thr_pct + w * thr_atr

        # 퍼센트 변환
        thr_pct_value = thr_hybrid / close * 100.0 if close > 0 else 0.0

        return thr_hybrid, thr_pct_value

    def _run_logic(
        self,
        high:  float,
        low:   float,
        close: float,
        atr:   float,
        thr:   float,
    ) -> str:
        """ZigZag 스타일 변곡점 탐지 코어.

        Returns
        -------
        "new_high" | "new_low" | "none"
        """
        signal = "none"

        # ── Step A: 후보 확인 창 처리 ────────────────────────────────────
        if self._pending_confirm is not None:
            signal = self._process_pending(high, low, close, atr, thr)
            if signal != "none":
                return signal

        # ── Step B: 방향 미결정 초기화 ───────────────────────────────────
        if self._direction == 0:
            return self._init_direction(high, low, close, atr, thr)

        # ── Step C: 상승 탐색 중 (고점 갱신 또는 저점 반전 감지) ─────────
        if self._direction == 1:
            if high > self._pending_high:
                self._pending_high     = high
                self._pending_high_idx = self._bar_idx
            if self._pending_low == float("inf") or low < self._pending_low:
                self._pending_low     = low
                self._pending_low_idx = self._bar_idx

            reversal = self._pending_high - low
            if reversal >= thr and self._wave_ok(thr, low, high, close, atr, self._pending_high_idx):
                if (self._pending_confirm is None
                        or self._pending_confirm.get("type") != "high"):
                    self._register_candidate("high", self._pending_high, self._pending_high_idx)

        # ── Step D: 하락 탐색 중 (저점 갱신 또는 고점 반전 감지) ─────────
        elif self._direction == -1:
            if low < self._pending_low:
                self._pending_low     = low
                self._pending_low_idx = self._bar_idx
            if self._pending_high == 0.0 or high > self._pending_high:
                self._pending_high     = high
                self._pending_high_idx = self._bar_idx

            reversal = high - self._pending_low
            if reversal >= thr and self._wave_ok(thr, low, high, close, atr, self._pending_low_idx):
                if (self._pending_confirm is None
                        or self._pending_confirm.get("type") != "low"):
                    self._register_candidate("low", self._pending_low, self._pending_low_idx)

        return signal

    def _process_pending(
        self,
        high:  float,
        low:   float,
        close: float,
        atr:   float,
        thr:   float,
    ) -> str:
        """후보 확인 창 처리.

        freeze=False 방식: 극단값 갱신 허용.
        되돌림이 thr×cancel_ratio 미만이면 후보 취소 + 방향 복귀.

        Returns
        -------
        "new_high" | "new_low" | "none"
        """
        pc   = self._pending_confirm
        pt   = pc.get("type")
        pp   = float(pc.get("price") or 0.0)
        pidx = int(pc.get("idx") or -1)
        rem  = int(pc.get("remaining") or 0)
        cr   = self.config.cancel_ratio

        # 후보 갱신 (극단값 갱신)
        updated = False
        if pt == "high" and high > pp:
            pp = high; pidx = self._bar_idx; updated = True
        elif pt == "low" and low < pp:
            pp = low;  pidx = self._bar_idx; updated = True
        if updated:
            pc.update(price=pp, idx=pidx)

        # confirmation_bars=0인 경우 즉시 확정 (rem 감소 전에 확인)
        if rem == 0:
            return self._confirm_pivot(pt, pp, pidx, atr, close)

        # 되돌림 초과 → 후보 취소 + 방향 복귀
        if pt == "high" and (pp - low) < thr * cr:
            logger.debug(
                "[HAP][취소] H@%s=%.2f | 되돌림 thr=%.4f | bar=%d",
                pc.get("bar_time"), pp, thr, self._bar_idx,
            )
            self._pending_confirm     = None
            self._pending_confirm_bar = -1
            self._pending_high        = 0.0
            self._pending_high_idx    = -1
            self._pending_low         = float("inf")
            self._pending_low_idx     = -1
            # 방향 복귀: HIGH 취소 시 하락 탐색(-1) 유지
            self._direction = -1
            return "none"

        if pt == "low" and (high - pp) < thr * cr:
            logger.debug(
                "[HAP][취소] L@%s=%.2f | 되돌림 thr=%.4f | bar=%d",
                pc.get("bar_time"), pp, thr, self._bar_idx,
            )
            self._pending_confirm     = None
            self._pending_confirm_bar = -1
            self._pending_low         = low
            self._pending_low_idx     = self._bar_idx
            self._pending_high        = 0.0
            self._pending_high_idx    = -1
            # 방향 복귀: LOW 취소 시 상승 탐색(1) 유지
            self._direction = 1
            return "none"

        rem -= 1
        pc["remaining"] = rem

        if rem <= 0:
            return self._confirm_pivot(pt, pp, pidx, atr, close)

        return "none"

    def _init_direction(
        self,
        high:  float,
        low:   float,
        close: float,
        atr:   float,
        thr:   float,
    ) -> str:
        """direction=0 초기 구간 처리.

        HIGH/LOW 인덱스 시간순 비교로 첫 앵커 결정.
        """
        if self._pending_high == 0.0:
            self._pending_high     = high
            self._pending_high_idx = self._bar_idx
        if self._pending_low == float("inf"):
            self._pending_low     = low
            self._pending_low_idx = self._bar_idx

        if high > self._pending_high:
            self._pending_high     = high
            self._pending_high_idx = self._bar_idx
        if low < self._pending_low:
            self._pending_low     = low
            self._pending_low_idx = self._bar_idx

        rng = self._pending_high - self._pending_low
        if rng < thr:
            return "none"

        # 고점이 나중에 형성 → LOW 앵커 먼저 확정, +1 방향(상승탐색) 시작
        if self._pending_high_idx >= self._pending_low_idx:
            self._direction              = 1
            self._state.last_low         = self._pending_low
            self._state.last_low_idx     = self._pending_low_idx
            self._state.last_low_time    = self.hhmm(self._pending_low_idx)
            # 절대 포인트를 퍼센트로 변환
            wave_pct = (rng / self._pending_low) * 100.0 if self._pending_low > 0 else 0.0
            pivot = self._add_pivot(
                self._pending_low_idx, self._pending_low, PivotType.LOW, wave_pct, atr
            )
            self._last_confirmed_bar     = self._bar_idx
            self._pending_high           = high
            self._pending_high_idx       = self._bar_idx
            self._pending_low            = float("inf")
            self._pending_low_idx        = -1
            logger.debug(
                "[HAP][초기확정] L@%s=%.2f | bar=%d",
                pivot.bar_time, pivot.price, self._bar_idx,
            )
            return "new_low"

        # 저점이 나중에 형성 → HIGH 앵커 먼저 확정, -1 방향(하락탐색) 시작
        else:
            self._direction              = -1
            self._state.last_high        = self._pending_high
            self._state.last_high_idx    = self._pending_high_idx
            self._state.last_high_time   = self.hhmm(self._pending_high_idx)
            # 절대 포인트를 퍼센트로 변환
            wave_pct = (rng / self._pending_high) * 100.0 if self._pending_high > 0 else 0.0
            pivot = self._add_pivot(
                self._pending_high_idx, self._pending_high, PivotType.HIGH, wave_pct, atr
            )
            self._last_confirmed_bar     = self._bar_idx
            self._pending_low            = low
            self._pending_low_idx        = self._bar_idx
            self._pending_high           = 0.0
            self._pending_high_idx       = -1
            logger.debug(
                "[HAP][초기확정] H@%s=%.2f | bar=%d",
                pivot.bar_time, pivot.price, self._bar_idx,
            )
            return "new_high"

    def _register_candidate(self, pt: str, price: float, idx: int) -> None:
        """후보 피봇 등록."""
        rem = max(0, self.config.confirmation_bars)
        self._pending_confirm = dict(
            type=pt,
            price=price,
            idx=idx,
            remaining=rem,
            bar_time=self.hhmm(idx),
        )
        self._pending_confirm_bar = self._bar_idx
        logger.debug(
            "[HAP][후보등록] %s@%s=%.2f | 대기=%d봉 | bar=%d",
            pt.upper(), self.hhmm(idx), price, rem, self._bar_idx,
        )

    def _confirm_pivot(
        self,
        pt:    str,
        price: float,
        idx:   int,
        atr:   float,
        close: float,
    ) -> str:
        """피봇 확정 처리."""
        self._pending_confirm      = None
        self._pending_confirm_bar  = -1
        self._last_confirmed_bar   = self._bar_idx

        # 파동 크기 계산 (직전 확정 피봇 대비 퍼센트)
        wave_pct = 0.0
        if self._pivots and self._pivots[-1].price > 0:
            wave_pct = abs(price - self._pivots[-1].price) / self._pivots[-1].price * 100.0

        ptype = PivotType.HIGH if pt == "high" else PivotType.LOW
        pivot = self._add_pivot(idx, price, ptype, wave_pct, atr)

        if pt == "high":
            self._state.last_high      = price
            self._state.last_high_idx  = idx
            self._state.last_high_time = pivot.bar_time
            self._direction            = -1
            self._pending_high         = 0.0
            self._pending_high_idx     = -1
            self._pending_low          = float("inf")
            self._pending_low_idx      = -1
            signal = "new_high"
        else:
            self._state.last_low       = price
            self._state.last_low_idx   = idx
            self._state.last_low_time  = pivot.bar_time
            self._direction            = 1
            self._pending_low          = float("inf")
            self._pending_low_idx      = -1
            self._pending_high         = 0.0
            self._pending_high_idx     = -1
            signal = "new_low"

        logger.warning(
            "[HAP][확정] %s | %s@%s=%.2f | 확정봉=%s | bar=%d",
            self._symbol, pt.upper(), pivot.bar_time, price,
            self.hhmm(self._bar_idx), self._bar_idx,
        )
        return signal

    # ── 피봇 추가 ────────────────────────────────────────────────────────────

    def _add_pivot(
        self,
        idx:      int,
        price:    float,
        ptype:    PivotType,
        wave_pct: float,
        atr:      float,
    ) -> PivotPoint:
        """PivotPoint 추가 후 반환."""
        prev_same = next(
            (p for p in reversed(self._pivots) if p.pivot_type == ptype),
            None,
        )
        avg_wave = self._avg_wave_pct(n=3)
        is_major = (
            True if prev_same is None
            else abs(price - prev_same.price) / prev_same.price * 100.0 >= avg_wave * 1.5
            if avg_wave > 0 and prev_same.price > 0
            else False
        )

        pivot = PivotPoint(
            index=idx,
            price=price,
            pivot_type=ptype,
            pct=wave_pct,
            atr=atr,
            bar_time=self.hhmm(idx),
            is_major=is_major,
        )
        self._pivots.append(pivot)
        # SR 탐색용으로 max_pivots * 2 보관 후 trim
        max_keep = self.config.max_pivots * 2
        if len(self._pivots) > max_keep:
            self._pivots = self._pivots[-max_keep:]
        return pivot

    # ── 필터 / 파라미터 계산 ─────────────────────────────────────────────────

    def _wave_ok(self, thr: float, low: float, high: float, close: float, atr: float, candidate_idx: int) -> bool:
        """이중 필터: 퍼센트 + ATR.
        
        reversal 계산과 동일한 기준점(low/high)으로 wave 크기를 측정한다.
        """
        cfg = self.config

        # 최소 봉 간격: 직전 확정 피봇으로부터 최소 1봉 이상
        if self._last_confirmed_bar >= 0:
            gap = candidate_idx - self._last_confirmed_bar
            if gap < 1:
                return False

        # 반전 기준점: direction=1이면 low, direction=-1이면 high
        if self._direction == 1:
            wave_ref = low
            wave_abs = abs(self._pending_high - wave_ref)
        else:
            wave_ref = high
            wave_abs = abs(wave_ref - self._pending_low)

        # 퍼센트 기반 최소 파동
        wave_pct = wave_abs / wave_ref * 100.0 if wave_ref > 0 else 0.0
        if wave_pct < cfg.min_wave_pct:
            return False

        # ATR 기반 최소 파동 (선택적)
        if atr > 0 and cfg.min_wave_atr_ratio > 0:
            if wave_abs < atr * cfg.min_wave_atr_ratio:
                return False

        return True

    def _calc_multiplier(self, er: float, bar_time: Any) -> float:
        """ER + 세션 시간대 테이블 → 배수 결정."""
        cfg  = self.config
        mmin = cfg.multiplier_min
        mmax = cfg.multiplier_max

        # ER 기반: ER↑(추세) → 배수 크게(노이즈 차단)
        n = len(self._closes)
        if n < cfg.er_period + 5:
            mult = (mmin + mmax) / 2.0
        else:
            mult = mmin + er * (mmax - mmin)

        # 세션 시간대 배율 적용
        table = cfg.session_multiplier_table
        if table:
            hhmm = self._format_hhmm(bar_time)
            if hhmm:
                for start, end, scale in table:
                    if str(start) <= hhmm < str(end):
                        mult *= float(scale)
                        break

        return float(np.clip(mult, mmin * 0.5, mmax * 2.0))

    def _calc_er(self) -> float:
        """Efficiency Ratio (0~1)."""
        cfg = self.config
        n   = len(self._closes)
        if n < cfg.er_period + 1:
            return 0.5
        cs = list(self._closes)[-(cfg.er_period + 1):-1]
        if len(cs) < cfg.er_period:
            return 0.5
        try:
            direction  = abs(float(cs[-1]) - float(cs[0]))
            volatility = sum(abs(float(cs[i]) - float(cs[i-1])) for i in range(1, len(cs)))
            if volatility < 1e-10:
                return 0.0
            return float(np.clip(direction / volatility, 0.0, 1.0))
        except Exception:
            return 0.5

    def _avg_wave_pct(self, n: int = 3) -> float:
        """평균 파동 퍼센트 계산."""
        if len(self._pivots) < 2:
            return 0.0
        sizes = [
            abs(self._pivots[-i].price - self._pivots[-i-1].price) / self._pivots[-i-1].price * 100.0
            for i in range(1, min(n + 1, len(self._pivots)))
        ]
        return float(sum(sizes) / len(sizes)) if sizes else 0.0

    # ── 구조 분석 ────────────────────────────────────────────────────────────

    def _analyze_structure(self) -> str:
        """시장 구조 분석."""
        if len(self._pivots) < 4:
            return "unknown"

        highs = [p.price for p in self._pivots[-8:] if p.pivot_type == PivotType.HIGH][-3:]
        lows  = [p.price for p in self._pivots[-8:] if p.pivot_type == PivotType.LOW][-3:]

        if len(highs) < 2 or len(lows) < 2:
            return "unknown"

        # 상승 구조: 고점 상승, 저점 상승
        if highs[0] < highs[-1] and lows[0] < lows[-1]:
            return "uptrend"
        # 하락 구조: 고점 하락, 저점 하락
        elif highs[0] > highs[-1] and lows[0] > lows[-1]:
            return "downtrend"
        # 횡보 구조
        else:
            return "ranging"

    def _calc_pivot_score(self, close: float = 0.0) -> float:
        """Pivot Score 계산 (0~1) — 5요소 가중합."""
        cfg = self.config

        # ① ATR 백분위 기반 변동성 점수 (25%)
        # 현재 ATR이 최근 60봉 분포에서 상위일수록 변곡 신뢰도 높음
        if len(self._atr_values) >= 10:
            window = list(self._atr_values)[-60:]
            cur_atr = window[-1]
            if cur_atr > 1e-10:
                atr_pct = float(np.mean([v <= cur_atr for v in window]))
                vol_score = atr_pct * 0.25
            else:
                vol_score = 0.12
        else:
            vol_score = 0.12

        # ② ER 점수 (25%)
        er = self._calc_er()
        er_score = er * 0.25

        # ③ 후보 임박도 (20%)
        urgency_score = 0.0
        if self._pending_confirm:
            cb = max(float(cfg.confirmation_bars), 1.0)
            rem = int(self._pending_confirm.get("remaining", cb))
            urgency_score = (1.0 - rem / cb) * 0.20

        # ④ 파동 품질 점수 (20%)
        # 파동이 임계값의 몇 배인지 (클수록 신뢰도 높음, 최대 3배에서 포화)
        wave_score = 0.0
        if self._pivots and len(self._atr_values) > 0:
            cur_thr_abs = self._state.threshold_abs
            if cur_thr_abs > 1e-10 and self._pivots:
                ref_price = close if close > 0 else (list(self._closes)[-1] if self._closes else 0.0)
                wave_abs = abs(ref_price - self._pivots[-1].price) if ref_price > 0 else 0.0
                ratio = wave_abs / cur_thr_abs
                wave_score = float(np.clip(ratio / 3.0, 0.0, 1.0)) * 0.20

        # ⑤ Fractal 교차 확증 보너스 (최대 10%)
        fractal_bonus = min(self._calc_fractal_bonus(), 0.10)

        raw = vol_score + er_score + urgency_score + wave_score + fractal_bonus
        return float(np.clip(raw, 0.0, 1.0))

    def _calc_fractal_bonus(self) -> float:
        """후보 피봇 방향과 Fractal 방향 교차 확증 → bonus 반환."""
        if self._fractal is None or self._pending_confirm is None:
            return 0.0
        
        fr = self._last_fractal
        if fr is None:
            return 0.0
        
        pc = self._pending_confirm
        pt = pc.get("type")
        pp = float(pc.get("price") or 0.0)
        tol = pp * self.config.fractal_price_tolerance_pct / 100.0
        
        if pt == "high" and fr.fractal_high and fr.fractal_high_price > 0:
            if abs(fr.fractal_high_price - pp) <= tol:
                return self.config.fractal_bonus
        elif pt == "low" and fr.fractal_low and fr.fractal_low_price > 0:
            if abs(fr.fractal_low_price - pp) <= tol:
                return self.config.fractal_bonus
        
        return 0.0

    # ── 상태 갱신 ────────────────────────────────────────────────────────────

    def _update_state(
        self,
        signal: str,
        atr:    float,
        thr_abs: float,
        thr_pct: float,
        er:     float,
        close:  float,
    ) -> None:
        """상태 객체 갱신."""
        s = self._state

        s.new_pivot_signal = signal
        s.atr = atr
        s.threshold_abs = thr_abs
        s.threshold_pct = thr_pct
        s.efficiency_ratio = er
        s.structure = self._analyze_structure()
        s.pivot_score = self._calc_pivot_score(close)

        # 파동 크기
        if self._pivots and self._pivots[-1].price > 0:
            s.wave_size_pct = abs(close - self._pivots[-1].price) / self._pivots[-1].price * 100.0

        # 경과 봉
        if self._last_confirmed_bar >= 0:
            s.bars_since_pivot = self._bar_idx - self._last_confirmed_bar

        # 후보 정보
        if self._pending_confirm:
            s.pending_type = self._pending_confirm.get("type")
            s.pending_price = float(self._pending_confirm.get("price") or 0.0)
            s.pending_time = self._pending_confirm.get("bar_time")
            s.pending_remaining = int(self._pending_confirm.get("remaining", 0))
        else:
            s.pending_type = None
            s.pending_price = 0.0
            s.pending_time = None
            s.pending_remaining = 0

        # 최근 피봇
        s.recent_pivots = self._pivots[-6:]

    # ── 유틸리티 ─────────────────────────────────────────────────────────────

    def _remember_time(self, bar_time: Any) -> None:
        """봉 시각 기억."""
        if bar_time is None:
            return
        hhmm = self._format_hhmm(bar_time)
        if hhmm:
            self._hhmm_map[self._bar_idx] = hhmm

    def _format_hhmm(self, bar_time: Any) -> str:
        """봉 시각을 HH:MM 형식으로 변환."""
        if bar_time is None:
            return ""
        if isinstance(bar_time, str):
            if ":" in bar_time:
                parts = bar_time.split(":")
                return f"{parts[0]}:{parts[1]}"
            return bar_time
        if hasattr(bar_time, "strftime"):
            return bar_time.strftime("%H:%M")
        return ""

    def hhmm(self, bar_idx: int) -> Optional[str]:
        """bar_idx에 해당하는 HH:MM 반환."""
        return self._hhmm_map.get(bar_idx)
