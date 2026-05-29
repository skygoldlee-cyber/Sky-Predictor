"""Percent Adaptive Pivot Detector
================================
ZigZag 의 고정 threshold 를 퍼센트 기반 동적 threshold 로 대체한 변곡점 탐지기.
ATRAdaptivePivot 의 설계 원칙과 기능 수준을 유지하면서 ATR 의존성을 제거.

설계 원칙
---------
- WilderRMA / ATR 의존성 제거 — 추가 패키지 없이 indicators 내부만 사용
- ATRAdaptivePivot 과 동일한 ``update(high, low, close, bar_time)`` 인터페이스
- ``get_transformer_features()`` 는 ATRAdaptivePivot azz_* 키 완전 호환
  → PriceTransformer / pipeline 무수정 주입 가능
- direction +1/−1 상태머신으로 HIGH↔LOW 교번 강제
- 후보 취소 로직 (되돌림 thr×0.3 미만 시 취소 + 방향 복귀)
- ``get_llm_context()`` 제공

퍼센트 threshold 계산
---------------------
    thr_abs = close × base_pct/100 × er_multiplier × session_scale

ER (Efficiency Ratio)
---------------------
    ER = |close[t] − close[t−n]| / Σ|close[i] − close[i−1]|
    (Kaufman 표준 정의 — 방향성 / 총 변동성)

ER ↑ 추세 강함 → multiplier 크게 (threshold 확대, 노이즈 차단)
ER ↓ 횡보      → multiplier 작게 (threshold 축소, 민감도 회복)

주요 개선 이력 (vs 구 PercentAdaptivePivot)
--------------------------------------------
- ER 계산 방식 → 표준 Kaufman ER 로 교체 (비표준 방향변경 횟수 방식 폐기)
- direction 상태머신 도입 (init_direction, +1/-1 교번 강제)
- _process_pending(): 후보 취소 + freeze=False 극단값 갱신
- _wave_ok(): 봉 간격(최소 3봉) + 퍼센트 파동 크기 이중 필터
- is_major: 직전 동일 타입 피봇 대비 avg_wave × 1.5 기준
- PAPState: ATRAdaptivePivotState 에 대응하는 22개 필드 상태 객체
- _analyze_structure(): uptrend/downtrend/ranging/unknown
- _calc_pivot_score(): 변동성 변화(40%) + ER(30%) + 후보 임박도(30%)
- get_transformer_features(): azz_* 25키 + pap_* 2키 = 27키
- get_llm_context(): LLM 프롬프트용 텍스트 제공
- reset() 완전 초기화 메서드
- AttributeError 버그 수정 (_confirm_pending_pivot 후 None 접근)

사용 예시
---------

    from indicators import PercentAdaptivePivot, PercentAdaptivePivotConfig

    cfg = PercentAdaptivePivotConfig(base_pct=0.3, er_period=10)
    pivot = PercentAdaptivePivot(cfg)

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
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WARMUP_BARS = 20
DEFAULT_CONFIRMATION_BARS = 1
DEFAULT_MIN_BAR_GAP = 3
DEFAULT_CANCEL_RATIO = 0.3
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
    bar_time:   Optional[str] = None   # "HH:MM"
    is_major:   bool = False


@dataclass
class PercentAdaptivePivotConfig:
    """Percent Adaptive Pivot 설정.

    Parameters
    ----------
    base_pct:
        기본 퍼센트 임계값 (%).
        reversal_threshold = close × base_pct/100 × er_multiplier × session_scale.
        값이 클수록 주요 변곡점만 탐지 (노이즈 차단).
    multiplier_min / multiplier_max:
        ER 기반 동적 배수 하한/상한.
        ER ↑ 추세 → 배수 크게, ER ↓ 횡보 → 배수 작게.
    er_period:
        표준 Kaufman ER 계산 구간.
    confirmation_bars:
        후보 등록 후 N봉 유지 확인 후 확정.
        1 이상 권장 (0 = 즉시 확정).
    min_wave_pct:
        파동 크기가 이 퍼센트(%) 미만이면 후보 등록 차단 (소파동 필터).
    min_bar_gap:
        직전 확정 피봇 이후 최소 봉 간격. 기본 3봉.
    max_pivots:
        보관할 최대 확정 피봇 수.
    session_multiplier_table:
        시간대별 퍼센트 배율 테이블.
        형식: List[("HH:MM", "HH:MM", multiplier_scale)]
        예) [("09:00","09:30", 1.5), ("14:30","15:20", 0.8)]
    warmup_bars:
        ER 안정화에 필요한 최소 봉 수. 미만이면 신호 미출력.
    cancel_ratio:
        pending 취소 판단 비율. 후보 대비 되돌림이 threshold × 이 비율 미만이면 취소.
        기본 0.3 (ATRAdaptivePivot 동일).
    """
    base_pct:             float = 0.3
    multiplier_min:       float = 0.8
    multiplier_max:       float = 2.0
    er_period:            int   = 10
    confirmation_bars:    int   = DEFAULT_CONFIRMATION_BARS
    min_wave_pct:         float = 0.15
    min_bar_gap:          int   = DEFAULT_MIN_BAR_GAP
    max_pivots:           int   = 30
    session_multiplier_table: List[Tuple[str, str, float]] = field(
        default_factory=list
    )
    warmup_bars:          int   = DEFAULT_WARMUP_BARS
    cancel_ratio:         float = DEFAULT_CANCEL_RATIO

    def __post_init__(self) -> None:
        if self.multiplier_max < self.multiplier_min:
            self.multiplier_min, self.multiplier_max = (
                self.multiplier_max, self.multiplier_min
            )


@dataclass
class PAPState:
    """매 봉 update() 후 반환되는 상태 객체.

    ATRAdaptivePivotState 에 대응; atr/threshold_abs 대신 threshold_pct 직접 사용.
    """
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

    # 임계값 (퍼센트)
    threshold_pct:    float = 0.0         # 현재 봉 reversal threshold (%)
    efficiency_ratio: float = 0.0

    # 후보 피봇 (미확정)
    pending_type:      Optional[str] = None   # "high" | "low"
    pending_price:     float = 0.0
    pending_time:      Optional[str] = None
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

class PercentAdaptivePivot:
    """퍼센트 기반 적응형 피봇 탐지기 (ATRAdaptivePivot 기능 동등).

    ATR / WilderRMA 의존성 없이 ATRAdaptivePivot 수준의 탐지 품질 제공.
    TickDataProvider 레이어에서 ATRAdaptivePivot 의 drop-in 대안.
    """

    def __init__(self, config: Optional[PercentAdaptivePivotConfig] = None) -> None:
        self.config = config or PercentAdaptivePivotConfig()
        self._symbol: str = "KP200 선물"
        self.reset()

    # ── 공개 API ────────────────────────────────────────────────────────────

    def set_symbol(self, symbol: str) -> None:
        self._symbol = symbol

    def reset(self) -> None:
        """완전 초기화."""
        cfg  = self.config
        _buf = max(cfg.er_period * 10, 200)

        # OHLC 버퍼
        self._closes: deque = deque(maxlen=_buf)
        self._pct_buf: deque = deque(maxlen=_buf)   # threshold_pct 이력 (score용)

        # 상태
        self._bar_idx:   int   = 0
        self._direction: int   = 0      # 0 미결정, +1 상승탐색, -1 하락탐색
        self._pending_high:     float = 0.0
        self._pending_low:      float = float("inf")
        self._pending_high_idx: int   = -1
        self._pending_low_idx:  int   = -1

        # 후보 확인 창
        self._pending_confirm:     Optional[Dict[str, Any]] = None
        self._pending_confirm_bar: int = -1

        # 확정 피봇
        self._pivots:             List[PivotPoint] = []
        self._last_confirmed_bar: int = -1

        # 시각 맵 (bar_idx → "HH:MM")
        self._hhmm_map: OrderedDict = OrderedDict()

        # 상태 객체
        self._state = PAPState()

    # ── 메인 업데이트 ────────────────────────────────────────────────────────

    def update(
        self,
        high:     float,
        low:      float,
        close:    float,
        bar_time: Any = None,
        open:     float = 0.0,
        volume:   float = 1.0,
    ) -> PAPState:
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
        PAPState
        """
        self._closes.append(close)
        self._remember_time(bar_time)
        n = len(self._closes)

        # ── 1. ER 계산 ──────────────────────────────────────────────────────
        er = self._calc_er()

        # ── 2. 동적 threshold ───────────────────────────────────────────────
        mult    = self._calc_multiplier(er, bar_time)
        thr_pct = self.config.base_pct * mult             # % 단위
        thr_abs = close * thr_pct / 100.0                 # 절대 포인트

        self._pct_buf.append(thr_pct)

        # ── 3. 웜업 중이면 신호 미출력 ─────────────────────────────────────
        signal = "none"
        if n >= self.config.warmup_bars:
            signal = self._run_logic(high, low, close, thr_abs, thr_pct)

        # ── 4. 상태 갱신 ────────────────────────────────────────────────────
        self._bar_idx += 1
        self._update_state(signal, er, thr_pct, close)
        return self._state

    # ── Transformer / LLM 출력 ───────────────────────────────────────────────

    def get_transformer_features(self, close: float) -> Dict[str, float]:
        """ATRAdaptivePivot azz_* 키 완전 호환 + pap_* 고유 키.

        azz_* 25키를 그대로 사용하므로 pipeline / features.py 수정 없이 주입 가능.
        """
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
            # ── PercentAdaptivePivot 고유 키 ─────────────────────────────
            "pap_threshold_pct":     _fin(float(np.clip(s.threshold_pct / 3.0, 0.0, 1.0))),
            "pap_pivot_score":       pivot_score,
        }

    def get_llm_context(self, close: float) -> str:
        """LLM 프롬프트 삽입용 컨텍스트 텍스트."""
        s       = self._state
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
            f"[PercentAdaptivePivot - {self._symbol}]\n"
            f"현재가: {close:.2f}  방향: {dir_kor}  구조: {str_kor}\n"
            f"신호: {s.new_pivot_signal}\n"
            f"후보: {pc_txt}\n"
            f"최근 고점: {lh:.2f}  저점: {ll:.2f}  파동: {s.wave_size_pct:.2f}%\n"
            f"피봇 목록:\n" + "\n".join(pivot_list or ["  (없음)"]) + "\n"
            f"threshold: {s.threshold_pct:.3f}%  ER: {s.efficiency_ratio:.3f}\n"
            f"Pivot Score: {s.pivot_score:.3f}"
        )

    # ── 프로퍼티 ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> PAPState:
        return self._state

    @property
    def pivot_score(self) -> float:
        """현재 Pivot Score (0~1). MSB·Kalman·OI 통합 슬롯."""
        return self._state.pivot_score

    @property
    def confirmed_pivots(self) -> List[PivotPoint]:
        return list(self._pivots)

    # ── 내부 로직 ────────────────────────────────────────────────────────────

    def _run_logic(
        self,
        high:    float,
        low:     float,
        close:   float,
        thr_abs: float,
        thr_pct: float,  # 미사용 (향후 확장용 보존)
    ) -> str:
        """ZigZag 스타일 변곡점 탐지 코어.

        Returns
        -------
        "new_high" | "new_low" | "none"
        """
        signal = "none"

        # ── Step A: 후보 확인 창 처리 ────────────────────────────────────
        if self._pending_confirm is not None:
            signal = self._process_pending(high, low, close, thr_abs)
            if signal != "none":
                return signal

        # ── Step B: 방향 미결정 초기화 ───────────────────────────────────
        if self._direction == 0:
            return self._init_direction(high, low, close, thr_abs)

        # ── Step C: 상승 탐색 중 (고점 갱신 또는 저점 반전 감지) ─────────
        if self._direction == 1:
            if high > self._pending_high:
                self._pending_high     = high
                self._pending_high_idx = self._bar_idx
            if self._pending_low == float("inf") or low < self._pending_low:
                self._pending_low     = low
                self._pending_low_idx = self._bar_idx

            reversal = self._pending_high - low
            if reversal >= thr_abs and self._wave_ok(thr_abs, close, self._pending_high_idx):
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
            if reversal >= thr_abs and self._wave_ok(thr_abs, close, self._pending_low_idx):
                if (self._pending_confirm is None
                        or self._pending_confirm.get("type") != "low"):
                    self._register_candidate("low", self._pending_low, self._pending_low_idx)

        return signal

    def _process_pending(
        self,
        high:    float,
        low:     float,
        close:   float,
        thr_abs: float,
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
            return self._confirm_pivot(pt, pp, pidx, close)

        # 되돌림 초과 → 후보 취소 + 방향 복귀
        if pt == "high" and (pp - low) < thr_abs * cr:
            logger.debug(
                "[PAP][취소] H@%s=%.2f | 되돌림 thr=%.4f | bar=%d",
                pc.get("bar_time"), pp, thr_abs, self._bar_idx,
            )
            self._pending_confirm     = None
            self._pending_confirm_bar = -1
            self._pending_high        = high
            self._pending_high_idx    = self._bar_idx
            self._pending_low         = float("inf")
            self._pending_low_idx     = -1
            # 방향 복귀: HIGH 취소 시 하락 탐색(-1) 유지
            self._direction = -1
            return "none"

        if pt == "low" and (high - pp) < thr_abs * cr:
            logger.debug(
                "[PAP][취소] L@%s=%.2f | 되돌림 thr=%.4f | bar=%d",
                pc.get("bar_time"), pp, thr_abs, self._bar_idx,
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
            return self._confirm_pivot(pt, pp, pidx, close)

        return "none"

    def _init_direction(
        self,
        high:    float,
        low:     float,
        close:   float,
        thr_abs: float,
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
        if rng < thr_abs:
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
                self._pending_low_idx, self._pending_low, PivotType.LOW, wave_pct
            )
            self._last_confirmed_bar     = self._bar_idx
            self._pending_high           = high
            self._pending_high_idx       = self._bar_idx
            self._pending_low            = float("inf")
            self._pending_low_idx        = -1
            logger.debug(
                "[PAP][초기확정] L@%s=%.2f | bar=%d",
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
                self._pending_high_idx, self._pending_high, PivotType.HIGH, wave_pct
            )
            self._last_confirmed_bar     = self._bar_idx
            self._pending_low            = low
            self._pending_low_idx        = self._bar_idx
            self._pending_high           = 0.0
            self._pending_high_idx       = -1
            logger.debug(
                "[PAP][초기확정] H@%s=%.2f | bar=%d",
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
            "[PAP][후보등록] %s@%s=%.2f | 대기=%d봉 | bar=%d",
            pt.upper(), self.hhmm(idx), price, rem, self._bar_idx,
        )

    def _confirm_pivot(
        self,
        pt:    str,
        price: float,
        idx:   int,
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
        pivot = self._add_pivot(idx, price, ptype, wave_pct)

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
            "[PAP][확정] %s | %s@%s=%.2f | 확정봉=%s | bar=%d",
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

    def _wave_ok(self, thr_abs: float, close: float, candidate_idx: int) -> bool:
        """최소 파동 크기 + 봉 간격 이중 필터."""
        cfg = self.config

        # 최소 봉 간격
        if self._last_confirmed_bar >= 0:
            if (candidate_idx - self._last_confirmed_bar) < cfg.min_bar_gap:
                return False

        # 퍼센트 파동 크기 필터
        if close > 0:
            wave_abs = (
                abs(self._pending_high - close)
                if self._direction == 1
                else abs(close - self._pending_low)
            )
            wave_pct = wave_abs / close * 100.0
            if wave_pct < cfg.min_wave_pct:
                return False

        return True

    def _calc_er(self) -> float:
        """표준 Kaufman Efficiency Ratio (0~1).

        ER = |close[t] − close[t−n]| / Σ|close[i] − close[i−1]|
        """
        cfg = self.config
        n   = len(self._closes)
        if n < cfg.er_period + 1:
            return 0.5

        cs = list(self._closes)[-(cfg.er_period + 1):]
        try:
            direction  = abs(float(cs[-1]) - float(cs[0]))
            volatility = sum(abs(float(cs[i]) - float(cs[i - 1])) for i in range(1, len(cs)))
            if volatility < 1e-10:
                return 0.0
            return float(np.clip(direction / volatility, 0.0, 1.0))
        except Exception:
            return 0.5

    def _calc_multiplier(self, er: float, bar_time: Any) -> float:
        """ER + 세션 시간대 → threshold 배수 결정."""
        cfg  = self.config
        mmin = cfg.multiplier_min
        mmax = cfg.multiplier_max

        n = len(self._closes)
        if n < cfg.er_period + 5:
            mult = (mmin + mmax) / 2.0
        else:
            mult = mmin + er * (mmax - mmin)

        # 세션 시간대 배율
        table = cfg.session_multiplier_table
        if table:
            hhmm = self._format_hhmm(bar_time)
            if hhmm:
                for start, end, scale in table:
                    if str(start) <= hhmm < str(end):
                        mult *= float(scale)
                        break

        return float(np.clip(mult, mmin * MULTIPLIER_CLIP_MIN_FACTOR, mmax * MULTIPLIER_CLIP_MAX_FACTOR))

    def _avg_wave_pct(self, n: int = 3) -> float:
        """최근 n파동 퍼센트 평균."""
        if len(self._pivots) < 2:
            return 0.0
        sizes = [
            self._pivots[-i].pct
            for i in range(1, min(n + 1, len(self._pivots)))
            if self._pivots[-i].pct > 0
        ]
        return float(sum(sizes) / len(sizes)) if sizes else 0.0

    # ── 구조 분석 ────────────────────────────────────────────────────────────

    def _analyze_structure(self) -> str:
        """HH·HL / LH·LL 70% 기준으로 시장 구조 분류."""
        if len(self._pivots) < 4:
            return "unknown"
        rh = [p.price for p in self._pivots[-8:] if p.pivot_type == PivotType.HIGH][-3:]
        rl = [p.price for p in self._pivots[-8:] if p.pivot_type == PivotType.LOW][-3:]
        if len(rh) < 2 or len(rl) < 2:
            return "unknown"
        hh = sum(1 for i in range(1, len(rh)) if rh[i] > rh[i - 1])
        hl = sum(1 for i in range(1, len(rl)) if rl[i] > rl[i - 1])
        lh = sum(1 for i in range(1, len(rh)) if rh[i] < rh[i - 1])
        ll = sum(1 for i in range(1, len(rl)) if rl[i] < rl[i - 1])
        n  = len(rh) - 1
        if n > 0:
            if hh / n >= 0.7 and hl / n >= 0.7:
                return "uptrend"
            if lh / n >= 0.7 and ll / n >= 0.7:
                return "downtrend"
        return "ranging"

    # ── Pivot Score ───────────────────────────────────────────────────────────

    def _calc_pivot_score(self, close: float, er: float) -> float:
        """Pivot Score [0, 1].

        구성:
        - 변동성 변화 (40%): 최근 20봉 threshold_pct 평균 대비 현재값
        - ER 추세 강도  (30%): 표준 Kaufman ER
        - 후보 임박도   (30%): remaining / confirmation_bars
        """
        score = 0.0

        # 변동성 변화 (기준창 20봉)
        _WIN = 20
        if len(self._pct_buf) >= _WIN + 1:
            recent = float(self._pct_buf[-1])
            ma_pct = float(np.mean(list(self._pct_buf)[-(_WIN + 1):-1]))
            if ma_pct > 0:
                shift = abs(recent - ma_pct) / ma_pct
                score += float(np.clip(shift * 2.0, 0.0, 1.0)) * 0.4

        # ER 강도 (update()에서 계산된 값 재사용)
        score += er * 0.3

        # 후보 임박도
        pc = self._pending_confirm
        if isinstance(pc, dict) and pc:
            cb  = max(float(self.config.confirmation_bars), 1.0)
            rem = float(pc.get("remaining") or 0)
            score += float(np.clip(1.0 - rem / cb, 0.0, 1.0)) * 0.3

        return float(np.clip(score, 0.0, 1.0))

    # ── 상태 업데이트 ─────────────────────────────────────────────────────────

    def _update_state(
        self,
        signal:  str,
        er:      float,
        thr_pct: float,
        close:   float,
    ) -> None:
        s = self._state
        s.new_pivot_signal = signal
        s.direction        = self._direction
        s.threshold_pct    = thr_pct
        s.efficiency_ratio = er
        s.structure        = self._analyze_structure()
        s.pivot_score      = self._calc_pivot_score(close, er)
        s.recent_pivots    = list(self._pivots[-self.config.max_pivots:])

        # 파동 크기
        if math.isfinite(s.last_high) and math.isfinite(s.last_low) and s.last_high > 0 and s.last_low > 0:
            mid = (s.last_high + s.last_low) / 2.0
            s.wave_size_pct = (s.last_high - s.last_low) / mid * 100.0 if mid > 0 else 0.0
        else:
            s.wave_size_pct = 0.0

        # 마지막 피봇 이후 경과 봉
        s.bars_since_pivot = (
            max(0, self._bar_idx - self._last_confirmed_bar)
            if self._last_confirmed_bar >= 0 else 0
        )

        # 후보 상태
        pc = self._pending_confirm
        if isinstance(pc, dict) and pc:
            s.pending_type      = pc.get("type")
            s.pending_price     = float(pc.get("price") or 0.0)
            s.pending_time      = pc.get("bar_time")
            s.pending_remaining = int(pc.get("remaining") or 0)
        else:
            s.pending_type      = None
            s.pending_price     = 0.0
            s.pending_time      = None
            s.pending_remaining = 0

    # ── 시각 유틸 ─────────────────────────────────────────────────────────────

    def _remember_time(self, bar_time: Any) -> None:
        hhmm = self._format_hhmm(bar_time)
        if hhmm:
            self._hhmm_map[self._bar_idx] = hhmm
            while len(self._hhmm_map) > 4096:
                self._hhmm_map.popitem(last=False)

    def _format_hhmm(self, bar_time: Any) -> Optional[str]:
        if bar_time is None:
            return None
        try:
            # pandas 없이 datetime 사용
            dt = datetime.fromisoformat(str(bar_time))
            return dt.strftime("%H:%M")
        except Exception:
            s = str(bar_time).strip()
            return s[:5] if len(s) >= 5 and ":" in s else None

    def hhmm(self, idx: int) -> Optional[str]:
        """bar_idx → "HH:MM" 공개 접근자."""
        return self._hhmm_map.get(idx)
