"""Market Structure Break (MSB / BOS) Detector
================================================
프로 트레이더 접근법 — 가격 구조 자체를 변곡점으로 정의.

핵심 개념
---------
- **Swing High / Swing Low** : N봉 lookback 기준으로 확정된 극값
- **Break of Structure (BOS)** : 이전 swing high/low 를 돌파 → 구조 전환
- **Change of Character (CHoCH)** : 추세 내 최초 반전 구조 신호
- **OI 레벨 연동** : ATRAdaptivePivot 또는 calc_oi_levels 의 지지/저항과 교차

Step 2 구현 범위
----------------
1. ``MarketStructureBreak`` — 실시간 BOS/CHoCH 탐지 (상태머신)
2. ``OIStructureGate`` — OI 레벨과 구조 붕괴 교차 분석
   - call/put OI peak 근처에서 BOS 발생 시 신호 강화
3. ``get_transformer_features()`` — msb_* 피처 반환 (PriceTransformer 주입용)
4. TradeExecutionGate 조건 통합 예시 (docstring)

설계 원칙
---------
- ATRAdaptivePivot 의 ``PivotPoint`` 목록을 직접 소비 (중복 계산 없음)
- 독립 운용도 가능 (high/low/close 스트림만으로 내부 swing 탐지)
- ``pivot_score`` 에 msb_score 를 가중 누적 → Step 3 Kalman 과 최종 통합

TradeExecutionGate 통합 예시
----------------------------
::

    from indicators import ATRAdaptivePivot, MarketStructureBreak, OIStructureGate

    pivot = ATRAdaptivePivot()
    msb   = MarketStructureBreak()
    gate  = OIStructureGate()

    for bar in stream:
        ps = pivot.update(bar.high, bar.low, bar.close, bar_time=bar.time)
        ms = msb.update(bar.high, bar.low, bar.close,
                        pivot_points=pivot.confirmed_pivots)
        oi_score = gate.score(ms, oi_levels=current_oi_levels, close=bar.close)

        # TradeExecutionGate 진입 조건
        total_score = (
            ps.pivot_score * 0.4 +
            ms.msb_score   * 0.4 +
            oi_score       * 0.2
        )
        if total_score > 0.65 and ms.bos_signal != "none":
            gate_decision = "진입"
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .atr_adaptive_pivot import PivotPoint, PivotType
except ImportError:
    from atr_adaptive_pivot import PivotPoint, PivotType  # type: ignore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 열거형
# ─────────────────────────────────────────────────────────────────────────────

class StructureType(Enum):
    UPTREND   = "uptrend"
    DOWNTREND = "downtrend"
    RANGING   = "ranging"
    UNKNOWN   = "unknown"


class BOSType(Enum):
    NONE  = "none"
    BOS_UP   = "bos_up"    # 이전 swing high 상향 돌파 → 상승 구조 확인
    BOS_DOWN = "bos_down"  # 이전 swing low 하향 돌파 → 하락 구조 확인
    CHOCH_UP   = "choch_up"   # 하락 추세 중 swing high 돌파 → 반전 신호
    CHOCH_DOWN = "choch_down" # 상승 추세 중 swing low 돌파 → 반전 신호


# ─────────────────────────────────────────────────────────────────────────────
# 설정 / 상태
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MSBConfig:
    """Market Structure Break 설정.

    Parameters
    ----------
    swing_lookback:
        내부 swing 탐지 lookback (봉 수). ATRAdaptivePivot 피봇이 주입되면
        이 값은 보조 역할을 합니다.
    bos_buffer_pct:
        BOS 판정 버퍼 (%). 이 비율 이상 돌파해야 BOS 확정.
        KOSPI200 1분봉 노이즈 대비 0.20% 이상 권장.
    bos_cooldown_bars:
        동일 ref 레벨 BOS 재발생 억제 봉 수.
        0이면 쿨다운 없음 (매 봉 BOS 가능 — 비권장).
    structure_lookback_pivots:
        구조 분석에 사용할 최근 피봇 수.
    choch_enabled:
        CHoCH (Change of Character) 탐지 활성화.
    min_swing_gap_bars:
        유효 스윙 간 최소 봉 간격.
    max_swings:
        보관할 최대 스윙 수.
    """
    swing_lookback:              int   = 3
    bos_buffer_pct:              float = 0.20   # 0.05 → 0.20: KOSPI200 틱노이즈 대응
    bos_cooldown_bars:           int   = 5      # 신규: 동일 레벨 BOS 중복 억제
    structure_lookback_pivots:   int   = 4      # 6 → 4: 최근 구조 우선
    choch_enabled:               bool  = True
    min_swing_gap_bars:          int   = 5      # 3 → 5: 스윙 간격 확대
    max_swings:                  int   = 30


@dataclass
class MSBState:
    """매 봉 MSB 업데이트 결과."""
    # BOS 신호
    bos_signal:     BOSType = BOSType.NONE
    bos_price:      float   = 0.0    # 돌파된 스윙 레벨 가격
    bos_time:       Optional[str] = None

    # 시장 구조
    structure:      StructureType = StructureType.UNKNOWN
    prev_structure: StructureType = StructureType.UNKNOWN

    # 구조 강도 지표
    hh_count:       int   = 0    # 최근 Higher High 횟수
    ll_count:       int   = 0    # 최근 Lower Low 횟수
    hl_count:       int   = 0    # 최근 Higher Low 횟수
    lh_count:       int   = 0    # 최근 Lower High 횟수

    # 스윙 레벨
    last_swing_high: float = 0.0
    last_swing_low:  float = 0.0
    last_sh_time:    Optional[str] = None
    last_sl_time:    Optional[str] = None

    # Pivot Score 기여분 (0~1)
    msb_score:      float = 0.0

    # 최근 BOS 이력
    recent_bos:     List[Dict] = field(default_factory=list)

    # 스윙 목록 (디버그용)
    swing_highs:    List[float] = field(default_factory=list)
    swing_lows:     List[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Market Structure Break 탐지기
# ─────────────────────────────────────────────────────────────────────────────

class MarketStructureBreak:
    """실시간 BOS / CHoCH 탐지기.

    두 가지 피봇 소스를 지원합니다.

    1. **외부 주입** : ``update(..., pivot_points=pivot.confirmed_pivots)``
       → ATRAdaptivePivot 확정 피봇을 그대로 소비 (권장)
    2. **내부 탐지** : ``pivot_points`` 미전달 시 lookback 기반 자체 탐지
       → ATRAdaptivePivot 없이 독립 사용 가능
    """

    def __init__(self, config: Optional[MSBConfig] = None) -> None:
        self.config = config or MSBConfig()
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        buf = max(cfg.swing_lookback * 4 + 20, 60)

        self._highs:  deque = deque(maxlen=buf)
        self._lows:   deque = deque(maxlen=buf)
        self._closes: deque = deque(maxlen=buf)
        self._times:  deque = deque(maxlen=buf)

        self._bar_idx: int = 0

        # 확정 스윙 목록 (PivotPoint 또는 내부 생성)
        self._swing_highs: List[Tuple[int, float, Optional[str]]] = []  # (idx, price, time)
        self._swing_lows:  List[Tuple[int, float, Optional[str]]] = []

        # 구조 상태
        self._structure:      StructureType = StructureType.UNKNOWN
        self._prev_structure: StructureType = StructureType.UNKNOWN

        # 마지막 외부 피봇 처리 인덱스 (중복 방지)
        self._last_ext_pivot_idx: int = -1

        # BOS 이력 + 쿨다운 추적
        self._recent_bos: List[Dict] = []
        self._last_bos_bar:   int   = -999   # 마지막 BOS 발생 봉
        self._last_bos_level: float = 0.0    # 마지막 BOS ref 레벨

        self._state = MSBState()

    def update(
        self,
        high:         float,
        low:          float,
        close:        float,
        bar_time:     Any = None,
        pivot_points: Optional[List[PivotPoint]] = None,
    ) -> MSBState:
        """매 봉 호출. pivot_points 는 ATRAdaptivePivot.confirmed_pivots 전달."""
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._times.append(self._fmt(bar_time))

        # ── 1. 스윙 업데이트 ─────────────────────────────────────────────
        if pivot_points is not None:
            self._ingest_external_pivots(pivot_points)
        else:
            self._detect_internal_swings()

        # ── 2. BOS / CHoCH 탐지 ──────────────────────────────────────────
        bos = self._check_bos(high, low, close)

        # ── 3. 구조 업데이트 ─────────────────────────────────────────────
        self._prev_structure = self._structure
        self._structure      = self._analyze_structure()

        # ── 4. 상태 갱신 ─────────────────────────────────────────────────
        self._update_state(bos, close)
        self._bar_idx += 1
        return self._state

    def get_transformer_features(self, close: float) -> Dict[str, float]:
        """msb_* 피처 반환 (PriceTransformer 주입용)."""
        s = self._state

        def _dist(price: float) -> float:
            if close <= 0 or price <= 0:
                return 0.0
            return float(np.clip((price - close) / close, -0.05, 0.05) / 0.05)

        bos_map = {
            BOSType.NONE:       0.0,
            BOSType.BOS_UP:     0.5,
            BOSType.BOS_DOWN:  -0.5,
            BOSType.CHOCH_UP:   1.0,
            BOSType.CHOCH_DOWN:-1.0,
        }
        str_map = {
            StructureType.UPTREND:   1.0,
            StructureType.DOWNTREND:-1.0,
            StructureType.RANGING:   0.0,
            StructureType.UNKNOWN:   0.0,
        }

        return {
            "msb_bos_signal":    bos_map.get(s.bos_signal, 0.0),
            "msb_structure":     str_map.get(s.structure, 0.0),
            "msb_hh_ratio":      float(np.clip(s.hh_count / 3.0, 0.0, 1.0)),
            "msb_ll_ratio":      float(np.clip(s.ll_count / 3.0, 0.0, 1.0)),
            "msb_sh_dist":       _dist(s.last_swing_high),
            "msb_sl_dist":       _dist(s.last_swing_low),
            "msb_score":         float(s.msb_score),
            "msb_choch":         float(s.bos_signal in (BOSType.CHOCH_UP, BOSType.CHOCH_DOWN)),
        }

    @property
    def state(self) -> MSBState:
        return self._state

    # ── 내부 스윙 관리 ────────────────────────────────────────────────────────

    def _ingest_external_pivots(self, pivots: List[PivotPoint]) -> None:
        """ATRAdaptivePivot 확정 피봇을 스윙 목록에 반영.

        개선: 인접 봉(min_swing_gap_bars 미만)의 동일 방향 피봇은
        더 극단적인 가격으로 갱신한다 (단순 중복 제거보다 정확).
        """
        for p in pivots:
            if p.index <= self._last_ext_pivot_idx:
                continue
            entry = (p.index, p.price, p.bar_time)
            gap = self.config.min_swing_gap_bars

            if p.pivot_type == PivotType.HIGH:
                if (self._swing_highs and
                        abs(p.index - self._swing_highs[-1][0]) < gap):
                    # 인접 HIGH: 더 높은 값으로 갱신
                    if p.price > self._swing_highs[-1][1]:
                        self._swing_highs[-1] = entry
                elif not self._swing_highs or p.price != self._swing_highs[-1][1]:
                    self._swing_highs.append(entry)
            else:
                if (self._swing_lows and
                        abs(p.index - self._swing_lows[-1][0]) < gap):
                    # 인접 LOW: 더 낮은 값으로 갱신
                    if p.price < self._swing_lows[-1][1]:
                        self._swing_lows[-1] = entry
                elif not self._swing_lows or p.price != self._swing_lows[-1][1]:
                    self._swing_lows.append(entry)

            self._last_ext_pivot_idx = max(self._last_ext_pivot_idx, p.index)

        # 최대 크기 유지
        cfg = self.config
        if len(self._swing_highs) > cfg.max_swings:
            self._swing_highs = self._swing_highs[-cfg.max_swings:]
        if len(self._swing_lows) > cfg.max_swings:
            self._swing_lows = self._swing_lows[-cfg.max_swings:]

    def _detect_internal_swings(self) -> None:
        """lookback 기반 내부 스윙 탐지 (ATRAdaptivePivot 없이 독립 사용 시)."""
        cfg = self.config
        lb  = cfg.swing_lookback
        n   = len(self._highs)
        if n < lb * 2 + 1:
            return

        h_list = list(self._highs)
        l_list = list(self._lows)
        t_list = list(self._times)
        ci     = n - 1 - lb  # center index

        ch = h_list[ci]
        cl = l_list[ci]
        abs_idx = self._bar_idx - lb

        # 내부 고점 프랙탈
        if (all(ch > h_list[ci - i] for i in range(1, lb + 1)) and
                all(ch > h_list[ci + i] for i in range(1, lb + 1))):
            if not self._swing_highs or abs_idx - self._swing_highs[-1][0] >= cfg.min_swing_gap_bars:
                self._swing_highs.append((abs_idx, ch, t_list[ci]))
                if len(self._swing_highs) > cfg.max_swings:
                    self._swing_highs = self._swing_highs[-cfg.max_swings:]

        # 내부 저점 프랙탈
        if (all(cl < l_list[ci - i] for i in range(1, lb + 1)) and
                all(cl < l_list[ci + i] for i in range(1, lb + 1))):
            if not self._swing_lows or abs_idx - self._swing_lows[-1][0] >= cfg.min_swing_gap_bars:
                self._swing_lows.append((abs_idx, cl, t_list[ci]))
                if len(self._swing_lows) > cfg.max_swings:
                    self._swing_lows = self._swing_lows[-cfg.max_swings:]

    # ── BOS / CHoCH 탐지 ─────────────────────────────────────────────────────

    def _check_bos(self, high: float, low: float, close: float) -> BOSType:
        """현재 봉이 swing high/low 를 돌파했는지 판정.

        개선 사항:
        - bos_cooldown_bars: 동일 레벨 BOS 쿨다운으로 중복 신호 억제
        - ref_sh/sl: [-2] 아닌 [-1] 사용 — BOS 확정 즉시 ref 갱신
        """
        cfg = self.config

        if not self._swing_highs and not self._swing_lows:
            return BOSType.NONE

        # 참조 스윙: 가장 최신 확정 스윙 레벨 사용
        # (이전 [-2] 방식은 BOS 후 ref가 갱신되지 않아 매 봉 BOS 발생)
        ref_sh = self._swing_highs[-1][1] if self._swing_highs else 0.0
        ref_sl = self._swing_lows[-1][1]  if self._swing_lows  else float("inf")

        if ref_sh <= 0 or ref_sl >= float("inf"):
            return BOSType.NONE

        buf = close * cfg.bos_buffer_pct / 100.0

        # ── 쿨다운 체크: 동일 레벨 BOS 반복 억제 ────────────────────────
        cooldown = cfg.bos_cooldown_bars
        if cooldown > 0 and self._last_bos_bar >= 0:
            bars_since = self._bar_idx - self._last_bos_bar
            if bars_since < cooldown:
                return BOSType.NONE

        # ── 상향 돌파 ────────────────────────────────────────────────────
        if high > ref_sh + buf:
            if self._structure == StructureType.DOWNTREND and cfg.choch_enabled:
                bos = BOSType.CHOCH_UP
            else:
                bos = BOSType.BOS_UP
            self._last_bos_bar   = self._bar_idx
            self._last_bos_level = ref_sh
            self._log_bos(bos, ref_sh, close)
            return bos

        # ── 하향 돌파 ────────────────────────────────────────────────────
        if low < ref_sl - buf:
            if self._structure == StructureType.UPTREND and cfg.choch_enabled:
                bos = BOSType.CHOCH_DOWN
            else:
                bos = BOSType.BOS_DOWN
            self._last_bos_bar   = self._bar_idx
            self._last_bos_level = ref_sl
            self._log_bos(bos, ref_sl, close)
            return bos

        return BOSType.NONE

    def _log_bos(self, bos: BOSType, level: float, close: float) -> None:
        entry = {
            "bar":   self._bar_idx,
            "bos":   bos.value,
            "level": level,
            "close": close,
            "time":  list(self._times)[-1] if self._times else None,
        }
        self._recent_bos.append(entry)
        if len(self._recent_bos) > 20:
            self._recent_bos = self._recent_bos[-20:]
        logger.warning(
            "[MSB][%s] level=%.2f | close=%.2f | bar=%d",
            bos.value, level, close, self._bar_idx,
        )

    # ── 구조 분석 ────────────────────────────────────────────────────────────

    def _analyze_structure(self) -> StructureType:
        """최근 스윙 목록으로 시장 구조 판정.

        개선: 피봇 수가 적을수록 임계값을 완화해
        초기 수렴을 빠르게 하고 경직성을 줄인다.
        """
        cfg = self.config
        n   = cfg.structure_lookback_pivots

        sh = [p[1] for p in self._swing_highs[-n:]]
        sl = [p[1] for p in self._swing_lows[-n:]]

        if len(sh) < 2 or len(sl) < 2:
            return StructureType.UNKNOWN

        hh = sum(1 for i in range(1, len(sh)) if sh[i] > sh[i-1])
        lh = sum(1 for i in range(1, len(sh)) if sh[i] < sh[i-1])
        hl = sum(1 for i in range(1, len(sl)) if sl[i] > sl[i-1])
        ll = sum(1 for i in range(1, len(sl)) if sl[i] < sl[i-1])

        total = len(sh) - 1
        if total <= 0:
            return StructureType.UNKNOWN

        # 피봇 수에 따른 동적 임계값: total=1→0.65, total=3→0.60, total=5→0.55
        threshold = max(0.55, 0.70 - 0.05 * total)

        if hh / total >= threshold and hl / total >= threshold:
            return StructureType.UPTREND
        if lh / total >= threshold and ll / total >= threshold:
            return StructureType.DOWNTREND
        return StructureType.RANGING

    # ── 상태 업데이트 ────────────────────────────────────────────────────────

    def _update_state(self, bos: BOSType, close: float) -> None:
        s = self._state
        s.bos_signal     = bos
        s.structure      = self._structure
        s.prev_structure = self._prev_structure
        s.bos_price      = (
            self._recent_bos[-1]["level"] if self._recent_bos and bos != BOSType.NONE
            else s.bos_price
        )
        s.bos_time = (
            self._recent_bos[-1]["time"] if self._recent_bos and bos != BOSType.NONE
            else s.bos_time
        )
        s.recent_bos     = list(self._recent_bos[-5:])

        # 최근 스윙 레벨
        s.last_swing_high = self._swing_highs[-1][1] if self._swing_highs else 0.0
        s.last_swing_low  = self._swing_lows[-1][1]  if self._swing_lows  else 0.0
        s.last_sh_time    = self._swing_highs[-1][2] if self._swing_highs else None
        s.last_sl_time    = self._swing_lows[-1][2]  if self._swing_lows  else None

        # HH / LL / HL / LH 카운트 (최근 N개)
        n  = self.config.structure_lookback_pivots
        sh = [p[1] for p in self._swing_highs[-n:]]
        sl = [p[1] for p in self._swing_lows[-n:]]
        s.hh_count = sum(1 for i in range(1, len(sh)) if sh[i] > sh[i-1])
        s.lh_count = sum(1 for i in range(1, len(sh)) if sh[i] < sh[i-1])
        s.hl_count = sum(1 for i in range(1, len(sl)) if sl[i] > sl[i-1])
        s.ll_count = sum(1 for i in range(1, len(sl)) if sl[i] < sl[i-1])

        s.swing_highs = [p[1] for p in self._swing_highs[-n:]]
        s.swing_lows  = [p[1] for p in self._swing_lows[-n:]]

        # MSB Score 계산
        s.msb_score = self._calc_msb_score(bos)

    def _calc_msb_score(self, bos: BOSType) -> float:
        """MSB 기여 점수 [0, 1].

        구성:
        - BOS/CHoCH 발생 여부 (0.4)
        - 구조 일관성 (0.35)
        - HH/LL 연속성 (0.25)
        """
        score = 0.0

        # BOS 신호 강도
        bos_weight = {
            BOSType.NONE:       0.0,
            BOSType.BOS_UP:     0.3,
            BOSType.BOS_DOWN:   0.3,
            BOSType.CHOCH_UP:   0.4,
            BOSType.CHOCH_DOWN: 0.4,
        }
        score += bos_weight.get(bos, 0.0)

        # 구조 일관성
        if self._structure != StructureType.UNKNOWN:
            if self._structure == self._prev_structure:
                score += 0.25  # 구조 유지
            else:
                score += 0.10  # 구조 전환 중

        # HH/LL 연속성
        n  = self.config.structure_lookback_pivots
        sh = [p[1] for p in self._swing_highs[-n:]]
        sl = [p[1] for p in self._swing_lows[-n:]]
        total = max(len(sh) - 1, 1)

        if self._structure == StructureType.UPTREND:
            hh = sum(1 for i in range(1, len(sh)) if sh[i] > sh[i-1])
            hl = sum(1 for i in range(1, len(sl)) if sl[i] > sl[i-1])
            score += min((hh + hl) / (total * 2), 1.0) * 0.35
        elif self._structure == StructureType.DOWNTREND:
            lh = sum(1 for i in range(1, len(sh)) if sh[i] < sh[i-1])
            ll = sum(1 for i in range(1, len(sl)) if sl[i] < sl[i-1])
            score += min((lh + ll) / (total * 2), 1.0) * 0.35

        return float(np.clip(score, 0.0, 1.0))

    # ── 유틸 ─────────────────────────────────────────────────────────────────

    def _fmt(self, bar_time: Any) -> Optional[str]:
        if bar_time is None:
            return None
        try:
            import pandas as pd
            return pd.Timestamp(bar_time).strftime("%H:%M")
        except Exception:
            s = str(bar_time).strip()
            return s[:5] if len(s) >= 5 and ":" in s else None


# ─────────────────────────────────────────────────────────────────────────────
# OI Structure Gate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OIStructureConfig:
    """OI 구조 게이트 설정.

    Parameters
    ----------
    oi_proximity_pct:
        OI peak 근처 판정 거리 (%). 현재가가 OI peak 에서 이 비율 이내면 근접.
    bos_oi_boost:
        BOS + OI 근접 시 점수 부스트 배수.
    choch_oi_boost:
        CHoCH + OI 근접 시 점수 부스트 배수 (BOS 보다 큰 가중치).
    """
    oi_proximity_pct: float = 0.3
    bos_oi_boost:     float = 1.4
    choch_oi_boost:   float = 1.7


class OIStructureGate:
    """OI 레벨 × 구조 붕괴 교차 분석.

    ``calc_oi_levels()`` 결과 dict 를 받아,
    BOS/CHoCH 발생 위치가 OI peak 근처인지 판단하고
    종합 점수를 반환합니다.

    단독으로도, MSB + ATRAdaptivePivot 조합으로도 사용 가능합니다.
    """

    def __init__(self, config: Optional[OIStructureConfig] = None) -> None:
        self.config = config or OIStructureConfig()

    def score(
        self,
        msb_state:  MSBState,
        close:      float,
        oi_levels:  Optional[Dict[str, float]] = None,
    ) -> float:
        """OI 교차 가중 점수 [0, 1] 반환.

        Parameters
        ----------
        msb_state:
            MarketStructureBreak.update() 결과.
        close:
            현재가.
        oi_levels:
            calc_oi_levels() 반환값.
            None 이면 OI 부스트 없이 msb_score 그대로 반환.
        """
        base = float(msb_state.msb_score)

        if oi_levels is None or close <= 0:
            return base

        call_peak = float(oi_levels.get("call_oi_peak") or 0.0)
        put_peak  = float(oi_levels.get("put_oi_peak")  or 0.0)

        prox  = self.config.oi_proximity_pct / 100.0
        near_call = call_peak > 0 and abs(close - call_peak) / close <= prox
        near_put  = put_peak  > 0 and abs(close - put_peak)  / close <= prox

        bos = msb_state.bos_signal
        boost = 1.0

        if bos in (BOSType.CHOCH_UP, BOSType.CHOCH_DOWN):
            # CHoCH + OI 근접 → 가장 강한 신호
            if (bos == BOSType.CHOCH_UP   and near_call) or \
               (bos == BOSType.CHOCH_DOWN and near_put):
                boost = self.config.choch_oi_boost
        elif bos in (BOSType.BOS_UP, BOSType.BOS_DOWN):
            if (bos == BOSType.BOS_UP   and near_call) or \
               (bos == BOSType.BOS_DOWN and near_put):
                boost = self.config.bos_oi_boost

        return float(np.clip(base * boost, 0.0, 1.0))

    def get_transformer_features(
        self,
        msb_state:  MSBState,
        close:      float,
        oi_levels:  Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """OI×MSB 교차 피처 반환."""
        oi_score = self.score(msb_state, close, oi_levels)

        call_peak = float((oi_levels or {}).get("call_oi_peak") or 0.0)
        put_peak  = float((oi_levels or {}).get("put_oi_peak")  or 0.0)
        prox = self.config.oi_proximity_pct / 100.0

        near_call = float(call_peak > 0 and abs(close - call_peak) / close <= prox) \
                    if close > 0 else 0.0
        near_put  = float(put_peak  > 0 and abs(close - put_peak)  / close <= prox) \
                    if close > 0 else 0.0

        def _dist(price: float) -> float:
            if close <= 0 or price <= 0:
                return 0.0
            return float(np.clip((price - close) / close, -0.05, 0.05) / 0.05)

        return {
            "oi_msb_score":     oi_score,
            "oi_near_call":     near_call,
            "oi_near_put":      near_put,
            "oi_call_dist":     _dist(call_peak),
            "oi_put_dist":      _dist(put_peak),
            "oi_bos_boosted":   float(oi_score > float(msb_state.msb_score) + 0.05),
        }
