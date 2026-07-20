"""ATR Adaptive Pivot Detector
================================
ZigZag 의 고정 threshold 를 ATR 기반 동적 threshold 로 대체한 변곡점 탐지기.

설계 원칙
---------
- AdaptiveZigZag 와 동일한 ``update(high, low, close, bar_time)`` 인터페이스 제공
  → TickDataProvider 레이어에서 drop-in 교체 가능
- 추가 의존성 없이 indicators 패키지 내부만 사용 (WilderRMA)
- ``get_transformer_features()`` 는 AdaptiveZigZag 와 동일한 키 집합 반환
  → PriceTransformer 에 그대로 주입 가능
- ``pivot_score`` 프로퍼티 : 이후 MSB / Kalman / OI 레이어가 추가될 통합 점수 슬롯

ZigZag 대비 개선점
------------------
1. **동적 threshold** : reversal = k × ATR(n)
   - 변동성 상승 → threshold 확대(노이즈 차단)
   - 변동성 하락 → threshold 축소(민감도 회복)
2. **세션 시간대 배율** : session_multiplier_table 로 장초반/점심/마감 등 구분
3. **확정 지연 없음** : pending_confirm 대신 반전 즉시 후보 등록 + N봉 유지 확인
4. **fake pivot 필터** : 최소 wave_atr_ratio 기준으로 소파동 차단
5. **Pivot Score** : 향후 MSB·Kalman·OI 점수와 가중합할 기반 슬롯

사용 예시
---------
::

    from indicators import ATRAdaptivePivot, ATRAdaptivePivotConfig

    cfg = ATRAdaptivePivotConfig(atr_period=14, base_multiplier=2.0)
    pivot = ATRAdaptivePivot(cfg)

    for h, l, c, t in bars:
        state = pivot.update(h, l, c, bar_time=t)
        features = pivot.get_transformer_features(c)
        score = pivot.pivot_score   # 0.0 ~ 1.0

    # TradeExecutionGate 조건 예시
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
    from ..wilder_smooth import WilderRMA
except ImportError:
    from wilder_smooth import WilderRMA  # type: ignore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 열거형 / 데이터 클래스
# ─────────────────────────────────────────────────────────────────────────────

class PivotType(Enum):
    HIGH = "high"
    LOW  = "low"


@dataclass
class PivotPoint:
    """확정된 변곡점 정보."""
    index:     int
    price:     float
    pivot_type: PivotType
    atr:       float
    bar_time:  Optional[str] = None   # "HH:MM"
    is_major:  bool = False


@dataclass
class ATRAdaptivePivotConfig:
    """ATR Adaptive Pivot 설정.

    Parameters
    ----------
    atr_period:
        ATR 계산 주기 (WilderRMA). 기본 14.
    base_multiplier:
        기본 ATR 배수. reversal_threshold = base_multiplier × ATR.
        값이 클수록 주요 변곡점만 탐지 (노이즈 차단).
    multiplier_min / multiplier_max:
        ER(Efficiency Ratio) 기반 동적 배수 하한/상한.
        ER ↑ 추세 강함 → 배수 크게(노이즈 차단).
        ER ↓ 횡보 → 배수 작게(민감도 회복).
    er_period:
        ER 계산 구간.
    confirmation_bars:
        후보 등록 후 N봉 연속 방향 유지를 확인해야 확정.
        0 = 즉시 확정(실시간 반응 최대).
    min_wave_atr_ratio:
        파동 크기가 ATR × 이 비율 미만이면 후보 등록 차단 (소파동 필터).
    max_pivots:
        보관할 최대 확정 피봇 수.
    session_multiplier_table:
        시간대별 ATR 배수 배율 테이블.
        형식: List[("HH:MM", "HH:MM", multiplier_scale)]
        예) [("09:00","09:30", 1.5), ("14:30","15:20", 0.8)]
        빈 리스트 → 단일 base_multiplier 사용.
    warmup_bars:
        ATR 안정화에 필요한 최소 봉 수. 미만이면 신호 미출력.
    cancel_ratio:
        pending 취소 판단 비율. 후보 대비 되돌림이 threshold × 이 비율 미만이면 취소.
        기본 0.3 (PercentAdaptivePivot 동일).
    """
    atr_period:            int   = 14
    base_multiplier:       float = 2.0
    multiplier_min:        float = 1.2
    multiplier_max:        float = 3.5
    er_period:             int   = 10
    confirmation_bars:     int   = 1
    min_wave_atr_ratio:    float = 0.5
    max_pivots:            int   = 30
    session_multiplier_table: List[Tuple[str, str, float]] = field(
        default_factory=list
    )
    warmup_bars:           int   = 20
    cancel_ratio:         float = 0.3

    def __post_init__(self) -> None:
        if self.multiplier_max < self.multiplier_min:
            self.multiplier_min, self.multiplier_max = (
                self.multiplier_max, self.multiplier_min
            )


@dataclass
class ATRAdaptivePivotState:
    """매 봉 업데이트 후 반환되는 상태."""
    # 최근 확정 피봇 (미확정 시 NaN — 0.0 과 구분)
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
    direction:        int   = 0           # +1 상승 탐색 중, -1 하락 탐색 중

    # 지표값
    atr:              float = 0.0
    threshold_abs:    float = 0.0        # 현재 봉 reversal threshold (pt)
    threshold_pct:    float = 0.0        # threshold / close × 100

    # 후보 피봇 (아직 미확정)
    pending_type:     Optional[str] = None   # "high" | "low"
    pending_price:    float = 0.0
    pending_time:     Optional[str] = None
    pending_remaining: int  = 0

    # 피봇 목록 (최근 N개)
    recent_pivots:    List[PivotPoint] = field(default_factory=list)

    # Pivot Score (0~1 : 이후 MSB·Kalman·OI 와 통합될 슬롯)
    pivot_score:      float = 0.0

    # 파동
    wave_size_pct:    float = 0.0
    bars_since_pivot: int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# 메인 클래스
# ─────────────────────────────────────────────────────────────────────────────

class ATRAdaptivePivot:
    """ATR 기반 동적 threshold 변곡점 탐지기.

    AdaptiveZigZag 와 동일한 인터페이스를 제공하므로
    TickDataProvider 레이어에서 drop-in 교체가 가능합니다.
    """

    def __init__(self, config: Optional[ATRAdaptivePivotConfig] = None) -> None:
        self.config = config or ATRAdaptivePivotConfig()
        self._symbol: str = "KP200 선물"
        self.reset()

    # ── 공개 API ────────────────────────────────────────────────────────────

    def set_symbol(self, symbol: str) -> None:
        self._symbol = symbol

    def reset(self) -> None:
        """완전 초기화."""
        cfg = self.config
        _buf = max(cfg.atr_period * 5, 100)

        # OHLC 버퍼
        self._highs:  deque = deque(maxlen=_buf)
        self._lows:   deque = deque(maxlen=_buf)
        self._closes: deque = deque(maxlen=_buf)
        self._tr:     deque = deque(maxlen=_buf)

        # ATR
        self._atr_rma   = WilderRMA(period=cfg.atr_period)
        self._atr_values: deque = deque(maxlen=_buf)

        # ER
        self._er_values:  deque = deque(maxlen=_buf)

        # 상태
        self._bar_idx:   int   = 0
        self._direction: int   = 0      # 0=미결정, +1=상승탐색, -1=하락탐색
        self._pending_high: float = 0.0
        self._pending_low:  float = float("inf")
        self._pending_high_idx: int = -1
        self._pending_low_idx:  int = -1

        # 후보 확인 창
        self._pending_confirm: Optional[Dict[str, Any]] = None
        self._pending_confirm_bar: int = -1

        # 확정 피봇
        self._pivots:     List[PivotPoint] = []
        self._last_confirmed_bar: int = -1

        # 시각 맵 (bar_idx → "HH:MM")
        self._hhmm_map: OrderedDict = OrderedDict()

        # 상태 객체
        self._state = ATRAdaptivePivotState()

    def update(
        self,
        high:     float,
        low:      float,
        close:    float,
        bar_time: Any = None,
        open:     float = 0.0,
        volume:   float = 1.0,
    ) -> ATRAdaptivePivotState:
        """1분봉 데이터 입력 → 상태 갱신."""
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

        # ── 2. ER (Efficiency Ratio) ────────────────────────────────────────
        er = self._calc_er()

        # ── 3. 동적 threshold ───────────────────────────────────────────────
        mult   = self._calc_multiplier(er, bar_time)
        thr    = atr * mult
        thr_pct = thr / close * 100.0 if close > 0 else 0.0

        # ── 4. 웜업 중이면 신호 미출력 ──────────────────────────────────────
        signal = "none"
        if n >= self.config.warmup_bars:
            signal = self._run_logic(high, low, close, atr, thr)

        # ── 5. 상태 갱신 ────────────────────────────────────────────────────
        self._bar_idx += 1
        self._update_state(signal, atr, thr, thr_pct, close)
        return self._state

    def get_transformer_features(self, close: float) -> Dict[str, float]:
        """AdaptiveZigZag 와 동일한 키 반환 (PriceTransformer 호환).

        azz_* 키를 그대로 사용하므로 features.py 수정 없이 주입 가능.
        신규 키 ``aap_*`` 는 이 지표 고유 정보를 담습니다.
        """
        s = self._state

        def _fin(v: float, fb: float = 0.0) -> float:
            try:
                x = float(v)
                return x if math.isfinite(x) else fb
            except Exception:
                return fb

        # ── 방향 / 구조 ───────────────────────────────────────────────────
        dir_f  = float(s.direction)
        str_up = float(s.structure == "uptrend")
        str_dn = float(s.structure == "downtrend")
        str_rg = float(s.structure == "ranging")

        # ── 최근 피봇 거리 정규화 (±5% clip → /0.05) ─────────────────────
        def _dist(price: float) -> float:
            # NaN(미확정) 또는 0 이하이면 0 반환
            if close <= 0 or not math.isfinite(price) or price <= 0:
                return 0.0
            raw = (price - close) / close
            return float(np.clip(raw, -0.05, 0.05) / 0.05)

        # ── 파동 크기 ─────────────────────────────────────────────────────
        wave_norm = _fin(min(s.wave_size_pct / 10.0, 1.0))

        # ── pending 피처 ──────────────────────────────────────────────────
        pc = self._pending_confirm
        if isinstance(pc, dict) and pc:
            pt       = pc.get("type", "")
            pp       = float(pc.get("price") or 0.0)
            rem      = int(pc.get("remaining") or 0)
            cb       = max(float(self.config.confirmation_bars), 1.0)
            pend_type     = 1.0 if pt == "high" else -1.0
            pend_dist     = _dist(pp)
            pend_urgency  = float(np.clip(1.0 - rem / cb, 0.0, 1.0))
            waited = self._bar_idx - self._pending_confirm_bar if self._pending_confirm_bar >= 0 else 0
            pend_age = float(math.exp(-waited / 5.0))
        else:
            pend_type = pend_dist = pend_urgency = pend_age = 0.0

        # ── bars_since_pivot ─────────────────────────────────────────────
        bsp_norm = _fin(min(s.bars_since_pivot / 50.0, 1.0))

        # ── support / resistance 거리 ─────────────────────────────────────
        highs = [p.price for p in self._pivots if p.pivot_type == PivotType.HIGH and p.price > close]
        lows  = [p.price for p in self._pivots if p.pivot_type == PivotType.LOW  and p.price < close]
        resist = min(highs) if highs else 0.0
        supprt = max(lows)  if lows  else 0.0

        res_dist = _fin(min(abs(_dist(resist)), 1.0)) if resist > 0 else 0.0
        sup_dist = _fin(min(abs(_dist(supprt)), 1.0)) if supprt > 0 else 0.0

        # ── 피봇 스코어 ───────────────────────────────────────────────────
        pivot_score = _fin(s.pivot_score)

        return {
            # AdaptiveZigZag 호환 키 (기존 pipeline 무수정 주입)
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
            # ATRAdaptivePivot 고유 키
            "aap_atr":               _fin(float(np.clip(s.atr / (close * 0.05), 0.0, 1.0))),
            "aap_threshold_pct":     _fin(float(np.clip(s.threshold_pct / 3.0, 0.0, 1.0))),
            "aap_pivot_score":       pivot_score,
        }

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
            t = "H" if p.pivot_type == PivotType.HIGH else "L"
            maj = "★" if p.is_major else " "
            pivot_list.append(f"  {maj}{p.bar_time or '?'} {t} {p.price:.2f}")

        pc_txt = "피봇후보 없음"
        if isinstance(self._pending_confirm, dict) and self._pending_confirm:
            pt  = "고점" if self._pending_confirm.get("type") == "high" else "저점"
            pp  = float(self._pending_confirm.get("price") or 0.0)
            rem = int(self._pending_confirm.get("remaining") or 0)
            pc_txt = f"{pt} 후보 {pp:.2f} | 확정까지 {rem}봉"

        return (
            f"[ATRAdaptivePivot - {self._symbol}]\n"
            f"현재가: {close:.2f}  방향: {dir_kor}  구조: {str_kor}\n"
            f"신호: {s.new_pivot_signal}\n"
            f"후보: {pc_txt}\n"
            f"최근 고점: {s.last_high:.2f}  저점: {s.last_low:.2f}  파동: {s.wave_size_pct:.2f}%\n"
            f"피봇 목록:\n" + "\n".join(pivot_list or ["  (없음)"]) + "\n"
            f"ATR: {s.atr:.2f}  threshold: {s.threshold_abs:.2f}pt ({s.threshold_pct:.2f}%)\n"
            f"Pivot Score: {s.pivot_score:.3f}"
        )

    @property
    def state(self) -> ATRAdaptivePivotState:
        return self._state

    @property
    def pivot_score(self) -> float:
        """현재 Pivot Score (0~1). 이후 MSB·Kalman·OI 와 통합될 슬롯."""
        return self._state.pivot_score

    @property
    def confirmed_pivots(self) -> List[PivotPoint]:
        return list(self._pivots)

    # ── 내부 로직 ────────────────────────────────────────────────────────────

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
            signal = self._init_direction(high, low, close, atr, thr)
            return signal

        # ── Step C: 상승 탐색 중 (고점 갱신 또는 저점 반전) ─────────────
        if self._direction == 1:
            if high > self._pending_high:
                self._pending_high     = high
                self._pending_high_idx = self._bar_idx
            # 현재 봉의 저가가 지금까지 최저가보다 낮으면 갱신 (반전 threshold 계산용)
            if self._pending_low == float("inf") or low < self._pending_low:
                self._pending_low     = low
                self._pending_low_idx = self._bar_idx

            reversal = self._pending_high - low
            if reversal >= thr and self._wave_ok(thr, close, self._pending_high_idx):
                if self._pending_confirm is None or self._pending_confirm.get("type") != "high":
                    self._register_candidate("high", self._pending_high, self._pending_high_idx)

        # ── Step D: 하락 탐색 중 (저점 갱신 또는 고점 반전) ─────────────
        elif self._direction == -1:
            if low < self._pending_low:
                self._pending_low     = low
                self._pending_low_idx = self._bar_idx
            # 현재 봉의 고가가 지금까지 최고가보다 높으면 갱신
            if self._pending_high == 0.0 or high > self._pending_high:
                self._pending_high     = high
                self._pending_high_idx = self._bar_idx

            reversal = high - self._pending_low
            if reversal >= thr and self._wave_ok(thr, close, self._pending_low_idx):
                if self._pending_confirm is None or self._pending_confirm.get("type") != "low":
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
        """후보 확인 창 처리. 확정되면 "new_high"/"new_low", 그렇지 않으면 "none"."""
        pc   = self._pending_confirm
        pt   = pc.get("type")
        pp   = float(pc.get("price") or 0.0)
        pidx = int(pc.get("idx") or -1)
        rem  = int(pc.get("remaining") or 0)
        cr   = self.config.cancel_ratio

        # 후보 갱신 (freeze=False 방식: 더 극단적이면 갱신)
        updated = False
        if pt == "high" and high > pp:
            pp = high; pidx = self._bar_idx; updated = True
        elif pt == "low" and low < pp:
            pp = low; pidx = self._bar_idx; updated = True

        if updated:
            pc.update(price=pp, idx=pidx)

        # confirmation_bars=0인 경우 즉시 확정 (rem 감소 전에 확인)
        if rem == 0:
            return self._confirm_pivot(pt, pp, pidx, atr, close)

        # 반대 방향 돌파 → 후보 취소, 방향 복귀
        if pt == "high" and (pp - low) < thr * cr:
            # 너무 많이 되돌아옴 → 취소
            logger.debug(
                "[AAP][취소] H@%s=%.2f | 되돌림 thr=%.2f | bar=%d",
                pc.get("bar_time"), pp, thr, self._bar_idx,
            )
            self._pending_confirm = None
            self._pending_confirm_bar = -1
            # direction=1 복귀 — pending_high를 현재 봉 고가로, pending_low 리셋
            self._pending_high     = high
            self._pending_high_idx = self._bar_idx
            self._pending_low      = float("inf")
            self._pending_low_idx  = -1
            # 명시적 방향 복귀: HIGH 취소 시 하락 탐색(-1) 유지
            self._direction = -1
            return "none"
        if pt == "low" and (high - pp) < thr * cr:
            logger.debug(
                "[AAP][취소] L@%s=%.2f | 되돌림 thr=%.2f | bar=%d",
                pc.get("bar_time"), pp, thr, self._bar_idx,
            )
            self._pending_confirm = None
            self._pending_confirm_bar = -1
            # direction=-1 복귀 — pending_low를 현재 봉 저가로, pending_high 리셋
            self._pending_low      = low
            self._pending_low_idx  = self._bar_idx
            self._pending_high     = 0.0
            self._pending_high_idx = -1
            # 명시적 방향 복귀: LOW 취소 시 상승 탐색(1) 유지
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
        """direction=0 초기 구간 처리."""
        if self._pending_high == 0.0:
            self._pending_high = high; self._pending_high_idx = self._bar_idx
        if self._pending_low == float("inf"):
            self._pending_low = low; self._pending_low_idx = self._bar_idx

        if high > self._pending_high:
            self._pending_high = high; self._pending_high_idx = self._bar_idx
        if low < self._pending_low:
            self._pending_low  = low;  self._pending_low_idx  = self._bar_idx

        rng = self._pending_high - self._pending_low
        if rng < thr:
            return "none"

        if self._pending_high_idx >= self._pending_low_idx:
            # 고점이 나중 → 하락 방향 먼저 확정 (LOW 앵커)
            self._direction = 1
            self._state.last_low     = self._pending_low
            self._state.last_low_idx = self._pending_low_idx
            self._state.last_low_time= self._hhmm(self._pending_low_idx)
            pivot = self._add_pivot(self._pending_low_idx, self._pending_low, PivotType.LOW, atr)
            self._last_confirmed_bar = self._bar_idx
            # LOW 앵커 확정 → 이후 HIGH 탐색. pending_high 를 현재까지의 실제 최고가로 설정
            self._pending_high     = high
            self._pending_high_idx = self._bar_idx
            self._pending_low      = float("inf")
            self._pending_low_idx  = -1
            logger.debug("[AAP][초기확정] L@%s=%.2f | bar=%d", pivot.bar_time, pivot.price, self._bar_idx)
            return "new_low"
        else:
            # 저점이 나중 → 상승 방향 먼저 확정 (HIGH 앵커)
            self._direction = -1
            self._state.last_high     = self._pending_high
            self._state.last_high_idx = self._pending_high_idx
            self._state.last_high_time= self._hhmm(self._pending_high_idx)
            pivot = self._add_pivot(self._pending_high_idx, self._pending_high, PivotType.HIGH, atr)
            self._last_confirmed_bar = self._bar_idx
            # HIGH 앵커 확정 → 이후 LOW 탐색. pending_low 를 현재까지의 실제 최저가로 설정
            self._pending_low      = low
            self._pending_low_idx  = self._bar_idx
            self._pending_high     = 0.0
            self._pending_high_idx = -1
            logger.debug("[AAP][초기확정] H@%s=%.2f | bar=%d", pivot.bar_time, pivot.price, self._bar_idx)
            return "new_high"

    def _register_candidate(
        self,
        pt:    str,
        price: float,
        idx:   int,
    ) -> None:
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
            "[AAP][후보등록] %s@%s=%.2f | 대기=%d봉 | bar=%d",
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

        ptype = PivotType.HIGH if pt == "high" else PivotType.LOW
        pivot = self._add_pivot(idx, price, ptype, atr)

        if pt == "high":
            self._state.last_high      = price
            self._state.last_high_idx  = idx
            self._state.last_high_time = pivot.bar_time
            self._direction       = -1
            # HIGH 확정 → LOW 탐색 시작: 양방향 pending 모두 초기화
            self._pending_high     = 0.0
            self._pending_high_idx = -1
            self._pending_low      = float("inf")
            self._pending_low_idx  = -1
            signal = "new_high"
        else:
            self._state.last_low      = price
            self._state.last_low_idx  = idx
            self._state.last_low_time = pivot.bar_time
            self._direction       = 1
            # LOW 확정 → HIGH 탐색 시작: 양방향 pending 모두 초기화
            self._pending_low      = float("inf")
            self._pending_low_idx  = -1
            self._pending_high     = 0.0
            self._pending_high_idx = -1
            signal = "new_low"

        logger.warning(
            "[AAP][확정] %s | %s@%s=%.2f 확정✓ | 확정봉=%s | bar=%d",
            self._symbol, pt.upper(), pivot.bar_time, price,
            self.hhmm(self._bar_idx), self._bar_idx,
        )
        return signal

    def _add_pivot(
        self,
        idx:        int,
        price:      float,
        ptype:      PivotType,
        atr:        float,
    ) -> PivotPoint:
        """PivotPoint 추가 후 반환."""
        # is_major: 직전 동일 타입 피봇 대비 파동 크기
        prev_same = next(
            (p for p in reversed(self._pivots) if p.pivot_type == ptype),
            None,
        )
        avg_wave = self._avg_wave_size(n=3)
        is_major = (
            True if prev_same is None
            else abs(price - prev_same.price) >= avg_wave * 1.5 if avg_wave > 0
            else abs(price - prev_same.price) >= atr * 2.0
        )

        pivot = PivotPoint(
            index=idx,
            price=price,
            pivot_type=ptype,
            atr=atr,
            bar_time=self.hhmm(idx),
            is_major=is_major,
        )
        self._pivots.append(pivot)
        if len(self._pivots) > self.config.max_pivots * 2:
            self._pivots = self._pivots[-self.config.max_pivots:]
        return pivot

    # ── 임계값 / 파라미터 계산 ───────────────────────────────────────────────

    def _calc_multiplier(self, er: float, bar_time: Any) -> float:
        """ER + 세션 시간대 테이블 → ATR 배수 결정."""
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

    def _wave_ok(self, thr: float, close: float, candidate_idx: int) -> bool:
        """최소 파동 크기 / 봉 간격 필터."""
        cfg = self.config

        # 최소 봉 간격: 직전 확정 피봇으로부터 최소 3봉 이상
        if self._last_confirmed_bar >= 0:
            gap = candidate_idx - self._last_confirmed_bar
            if gap < 3:
                return False

        # ATR 기반 최소 파동
        if self._atr_values and close > 0:
            atr  = float(self._atr_values[-1])
            ratio = cfg.min_wave_atr_ratio
            if atr > 0:
                wave = (
                    abs(self._pending_high - close)
                    if self._direction == 1
                    else abs(close - self._pending_low)
                )
                if wave < atr * ratio:
                    return False

        return True

    def _avg_wave_size(self, n: int = 3) -> float:
        if len(self._pivots) < 2:
            return 0.0
        sizes = [
            abs(self._pivots[-i].price - self._pivots[-i-1].price)
            for i in range(1, min(n + 1, len(self._pivots)))
        ]
        return float(sum(sizes) / len(sizes)) if sizes else 0.0

    # ── 구조 분석 ────────────────────────────────────────────────────────────

    def _analyze_structure(self) -> str:
        if len(self._pivots) < 4:
            return "unknown"
        rh = [p.price for p in self._pivots[-8:] if p.pivot_type == PivotType.HIGH][-3:]
        rl = [p.price for p in self._pivots[-8:] if p.pivot_type == PivotType.LOW][-3:]
        if len(rh) < 2 or len(rl) < 2:
            return "unknown"
        hh = sum(1 for i in range(1, len(rh)) if rh[i] > rh[i-1])
        hl = sum(1 for i in range(1, len(rl)) if rl[i] > rl[i-1])
        lh = sum(1 for i in range(1, len(rh)) if rh[i] < rh[i-1])
        ll = sum(1 for i in range(1, len(rl)) if rl[i] < rl[i-1])
        n  = len(rh) - 1
        if n > 0:
            if hh / n >= 0.7 and hl / n >= 0.7:
                return "uptrend"
            if lh / n >= 0.7 and ll / n >= 0.7:
                return "downtrend"
        return "ranging"

    # ── Pivot Score 계산 ─────────────────────────────────────────────────────

    def _calc_pivot_score(self, close: float) -> float:
        """현재 봉의 Pivot Score [0, 1].

        구성 요소:
        - volatility_shift : ATR 변화율 (최근 20봉 평균 대비) — 과민 반응 방지
        - trend_curvature   : ER 기반 추세 강도
        - pending_urgency   : 후보 확정 임박도
        """
        score = 0.0

        # 변동성 변화 — 기준창 5봉(과민) → 20봉(안정)
        _ATR_WINDOW = 20
        if len(self._atr_values) >= _ATR_WINDOW + 1:
            recent_atr = float(self._atr_values[-1])
            ma_atr     = float(np.mean(list(self._atr_values)[-(_ATR_WINDOW + 1):-1]))
            if ma_atr > 0:
                shift = abs(recent_atr - ma_atr) / ma_atr
                # 2배 증폭 후 clip — 기준창이 길어진 만큼 민감도 보정
                score += float(np.clip(shift * 2.0, 0.0, 1.0)) * 0.4

        # ER 기반 추세 강도
        er = self._calc_er()
        score += er * 0.3

        # 후보 임박도
        pc = self._pending_confirm
        if isinstance(pc, dict) and pc:
            cb  = max(float(self.config.confirmation_bars), 1.0)
            rem = float(pc.get("remaining") or 0)
            score += float(np.clip(1.0 - rem / cb, 0.0, 1.0)) * 0.3

        return float(np.clip(score, 0.0, 1.0))

    # ── 상태 업데이트 ────────────────────────────────────────────────────────

    def _update_state(
        self,
        signal:  str,
        atr:     float,
        thr:     float,
        thr_pct: float,
        close:   float,
    ) -> None:
        s = self._state
        s.new_pivot_signal = signal
        s.direction        = self._direction
        s.atr              = atr
        s.threshold_abs    = thr
        s.threshold_pct    = thr_pct
        s.structure        = self._analyze_structure()
        s.pivot_score      = self._calc_pivot_score(close)
        s.recent_pivots    = list(self._pivots[-self.config.max_pivots:])

        # 파동 크기
        if s.last_high > 0 and s.last_low > 0:
            mid = (s.last_high + s.last_low) / 2.0
            s.wave_size_pct = (s.last_high - s.last_low) / mid * 100.0 if mid > 0 else 0.0
        else:
            s.wave_size_pct = 0.0

        # 마지막 피봇 이후 경과 봉
        if self._last_confirmed_bar >= 0:
            s.bars_since_pivot = max(0, self._bar_idx - self._last_confirmed_bar)
        else:
            s.bars_since_pivot = 0

        # 후보 상태
        pc = self._pending_confirm
        if isinstance(pc, dict) and pc:
            s.pending_type      = pc.get("type")
            s.pending_price     = float(pc.get("price") or 0.0)
            s.pending_time      = pc.get("bar_time")
            s.pending_remaining = int(pc.get("remaining") or 0)
        else:
            s.pending_type = None
            s.pending_price = 0.0
            s.pending_time = None
            s.pending_remaining = 0

    # ── 시각 유틸 ────────────────────────────────────────────────────────────

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
            import pandas as pd
            return pd.Timestamp(bar_time).strftime("%H:%M")
        except Exception:
            s = str(bar_time).strip()
            return s[:5] if len(s) >= 5 and ":" in s else None

    def _hhmm(self, idx: int) -> Optional[str]:
        return self._hhmm_map.get(idx)

    def hhmm(self, idx: int) -> Optional[str]:
        """공개 접근자."""
        return self._hhmm_map.get(idx)
