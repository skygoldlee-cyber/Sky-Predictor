"""Kalman Filter Turning Point Detector
=========================================
가격을 "noisy 관측값"으로 보고 실제 추세(hidden state)와 노이즈를 분리.
Slope(기울기) 변화 시점을 변곡점으로 판단.

설계 원칙
---------
- **Constant-velocity Kalman Filter** (2-state: position + velocity)
  - state  : [price_est, slope_est]
  - obs    : [close]
  - Q (process noise)   : 추세 변화 허용도 조절
  - R (observation noise): 가격 데이터 신뢰도 조절
- **EMA 보다 lag 작음** — slope 가 EMA derivative 보다 빠르게 반응
- **AdaptiveZigZag 상위호환 느낌** 문서에서 언급된 접근

변곡점 판정 기준
----------------
1. slope 부호 전환 (양→음, 음→양)  → ``slope_flip``
2. slope 크기 급변 (전 N봉 평균 대비 k배 이상 급증) → ``slope_surge``
3. slope 가중 변곡 점수 (0~1) → ``kalman_score``

인터페이스
----------
``get_transformer_features()`` 반환 키는 ``kf_*`` 접두사.
Step 1(aap_*) / Step 2(msb_*) 와 충돌 없이 병렬 주입 가능.

사용 예시
---------
::

    from indicators import KalmanTurningPoint, KalmanConfig

    kf = KalmanTurningPoint(KalmanConfig(q=0.01, r=1.0))

    for h, l, c, t in bars:
        state = kf.update(c, bar_time=t)
        if state.slope_flip:
            print(f"Kalman 변곡점: slope {state.prev_slope:.4f} → {state.slope:.4f}")

통합 예시 (4-layer)
-------------------
::

    from indicators import PivotScoreIntegrator

    integrator = PivotScoreIntegrator()
    total = integrator.compute(
        aap_score  = pivot.pivot_score,
        msb_score  = msb.state.msb_score,
        oi_score   = oi_gate.score(msb.state, close, oi_levels),
        kalman_score = kf.state.kalman_score,
    )
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KalmanConfig:
    """Kalman Filter 변곡점 탐지기 설정.

    Parameters
    ----------
    q : float
        Process noise covariance (상태 전이 불확실성).
        클수록 slope 가 빠르게 변함 → 민감도 ↑ / 노이즈 ↑.
        KOSPI200 선물 1분봉 권장: 0.005 ~ 0.05
    r : float
        Observation noise covariance (측정 불확실성).
        클수록 관측값을 덜 신뢰 → 추세선 매끄러움 ↑ / 반응 속도 ↓.
        KOSPI200 선물 1분봉 권장: 0.5 ~ 5.0
    warmup_bars : int
        Kalman 안정화에 필요한 최소 봉 수.
    slope_flip_min : float
        slope 부호 전환으로 인정할 최소 slope 절대값.
        너무 작으면 노이즈 flip 증가.
    slope_surge_k : float
        slope 급변 배수 기준. 최근 N봉 slope 평균 × k 이상 → surge.
    slope_history_n : int
        slope 평균 계산에 사용할 봉 수.
    adaptive_q : bool
        True 이면 ATR 기반으로 Q 를 동적 조정.
        변동성 증가 → Q 증가 → slope 빠르게 추적.
    adaptive_q_atr_period : int
        adaptive Q 계산용 ATR 주기.
    """
    q:                    float = 0.05    # 0.01 → 0.05: slope 반응 속도 향상
    r:                    float = 1.0     # 2.0 → 1.0: 관측값 신뢰도 상향
    warmup_bars:          int   = 15
    slope_flip_min:       float = 0.003   # 0.005 → 0.003: 약한 반전도 탐지
    slope_surge_k:        float = 2.5
    slope_history_n:      int   = 10
    adaptive_q:           bool  = True
    adaptive_q_atr_period: int  = 10     # 14 → 10: 더 빠른 변동성 반영


# ─────────────────────────────────────────────────────────────────────────────
# 상태
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KalmanState:
    """매 봉 업데이트 후 반환 상태."""
    # 추정값
    price_est:    float = 0.0   # Kalman 가격 추정 (노이즈 제거됨)
    slope:        float = 0.0   # 현재 추세 기울기 (pt/봉)
    prev_slope:   float = 0.0   # 직전 봉 기울기

    # 변곡점 신호
    slope_flip:   bool  = False  # slope 부호 전환 (변곡점)
    slope_surge:  bool  = False  # slope 급변
    turning_signal: str = "none" # "up" | "down" | "none"

    # 점수
    kalman_score: float = 0.0   # 0~1

    # Kalman 내부 진단
    innovation:   float = 0.0   # 관측 - 예측 (잔차)
    kalman_gain:  float = 0.0   # Kalman gain (K)
    q_adaptive:   float = 0.0   # 현재 사용 중인 Q 값

    # 이력
    slope_history: List[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filter 핵심
# ─────────────────────────────────────────────────────────────────────────────

class KalmanTurningPoint:
    """Constant-velocity Kalman Filter 기반 변곡점 탐지기.

    State vector: x = [price, slope]
    Transition  : x_k = F · x_{k-1} + w_k
    Observation : z_k = H · x_k + v_k

    F = [[1, 1],    H = [1, 0]
         [0, 1]]
    """

    def __init__(self, config: Optional[KalmanConfig] = None) -> None:
        self.config = config or KalmanConfig()
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        buf = max(cfg.warmup_bars * 3, 60)

        # Kalman state: column vector [[price_est], [slope_est]]
        self._x = np.array([[0.0], [0.0]])    # 2×1 column vector
        self._P = np.eye(2) * 100.0           # 2×2 error covariance

        # System matrices
        self._F = np.array([[1.0, 1.0],
                            [0.0, 1.0]])       # 2×2 transition
        self._H = np.array([[1.0, 0.0]])       # 1×2 observation
        self._Q = np.eye(2) * cfg.q            # 2×2 process noise
        self._R = np.array([[cfg.r]])          # 1×1 observation noise

        # 버퍼
        self._closes:       deque = deque(maxlen=buf)
        self._slope_hist:   deque = deque(maxlen=cfg.slope_history_n * 3)
        self._tr_buf:       deque = deque(maxlen=cfg.adaptive_q_atr_period * 3)

        self._bar_idx:      int   = 0
        self._initialized:  bool  = False
        self._prev_slope:   float = 0.0
        self._hhmm_map:     Dict[int, str] = {}

        self._last_innovation: float = 0.0
        self._last_gain:       float = 0.0

        self._state = KalmanState()

    def update(
        self,
        close:    float,
        high:     float = 0.0,
        low:      float = 0.0,
        bar_time: Any   = None,
    ) -> KalmanState:
        """1봉 입력 → Kalman 추정 갱신."""
        self._closes.append(close)
        self._remember_time(bar_time)
        n = len(self._closes)

        # True Range (adaptive Q 용)
        if high > 0 and low > 0 and n >= 2:
            pc = list(self._closes)[-2]
            tr = max(high - low, abs(high - pc), abs(low - pc))
        else:
            tr = abs(high - low) if high > 0 else 0.0
        self._tr_buf.append(tr)

        # 초기화: 첫 봉에서 state 를 관측값으로 설정
        if not self._initialized:
            self._x = np.array([[close], [0.0]])   # column vector
            self._initialized = True
            self._bar_idx += 1
            self._state = KalmanState(price_est=close)
            return self._state

        # 웜업 중
        if n < self.config.warmup_bars:
            self._run_filter(close)
            self._bar_idx += 1
            self._state = KalmanState(
                price_est=float(self._x[0, 0]),
                slope=float(self._x[1, 0]),
                kalman_score=0.0,
            )
            return self._state

        # 본 추정
        self._run_filter(close)
        slope     = float(self._x[1, 0])
        prev_slope = self._prev_slope

        # 변곡점 판정
        flip  = self._detect_flip(slope, prev_slope)
        surge = self._detect_surge(slope)
        signal = self._determine_signal(slope, prev_slope, flip)

        # 점수
        score = self._calc_kalman_score(slope, prev_slope, flip, surge)

        self._slope_hist.append(slope)
        self._prev_slope = slope

        # 상태 갱신
        self._state = KalmanState(
            price_est     = float(self._x[0, 0]),
            slope         = slope,
            prev_slope    = prev_slope,
            slope_flip    = flip,
            slope_surge   = surge,
            turning_signal= signal,
            kalman_score  = score,
            innovation    = float(getattr(self, "_last_innovation", 0.0)),
            kalman_gain   = float(getattr(self, "_last_gain", 0.0)),
            q_adaptive    = float(self._Q[0, 0]),
            slope_history = list(self._slope_hist)[-self.config.slope_history_n:],
        )

        self._bar_idx += 1
        return self._state

    def get_transformer_features(self, close: float) -> Dict[str, float]:
        """kf_* 피처 반환 (PriceTransformer 주입 / PivotScoreIntegrator 입력)."""
        s = self._state

        def _fin(v: float) -> float:
            return v if math.isfinite(v) else 0.0

        # slope 정규화: 최근 ATR 기반 동적 기준 (5-A: 고정 1.0pt 기준 → 변동성 반영)
        _atr_period = self.config.adaptive_q_atr_period
        if len(self._tr_buf) >= _atr_period:
            _ref_slope = float(np.mean(list(self._tr_buf)[-_atr_period:])) + 1e-9
        else:
            _ref_slope = 1.0  # 웜업 미완료 시 고정값 fallback
        slope_norm = _fin(float(np.clip(s.slope / _ref_slope, -1.0, 1.0)))

        # price_est 와 close 의 괴리 (필터 lag 지표)
        dev = (float(s.price_est) - close) / close if close > 0 else 0.0
        dev_norm = _fin(float(np.clip(dev, -0.05, 0.05) / 0.05))

        # innovation 정규화: ATR 기반 (고정 /2.0 → 변동성 적응)
        innov_norm = _fin(float(np.clip(s.innovation / (_ref_slope + 1e-9), -1.0, 1.0)))

        signal_map = {"up": 1.0, "down": -1.0, "none": 0.0}

        return {
            "kf_slope_norm":     slope_norm,
            "kf_slope_flip":     float(s.slope_flip),
            "kf_slope_surge":    float(s.slope_surge),
            "kf_turning_signal": signal_map.get(s.turning_signal, 0.0),
            "kf_score":          _fin(s.kalman_score),
            "kf_dev_norm":       dev_norm,
            "kf_innovation":     innov_norm,
            "kf_gain":           _fin(float(np.clip(s.kalman_gain, 0.0, 1.0))),
        }

    @property
    def state(self) -> KalmanState:
        return self._state

    # ── Kalman Filter 코어 ────────────────────────────────────────────────────

    def _run_filter(self, close: float) -> None:
        """Predict + Update 1 step (column vector 방식)."""
        cfg = self.config

        # Adaptive Q (ATR 기반)
        if cfg.adaptive_q and len(self._tr_buf) >= cfg.adaptive_q_atr_period:
            atr = float(np.mean(list(self._tr_buf)[-cfg.adaptive_q_atr_period:]))
            q_adaptive = float(np.clip(
                cfg.q * (atr ** 2), cfg.q * 0.1, cfg.q * 10.0
            ))
        else:
            q_adaptive = cfg.q

        self._Q = np.array([[q_adaptive,         0.0],
                            [0.0,        q_adaptive * 0.1]])

        # Predict
        x_pred = self._F @ self._x           # 2×1
        P_pred = self._F @ self._P @ self._F.T + self._Q  # 2×2

        # Innovation (scalar)
        z         = np.array([[close]])      # 1×1
        innovation = float((z - self._H @ x_pred)[0, 0])
        self._last_innovation = innovation

        # Kalman Gain  K = P_pred H^T (H P_pred H^T + R)^-1   → 2×1
        S = self._H @ P_pred @ self._H.T + self._R   # 1×1
        K = P_pred @ self._H.T / float(S[0, 0])      # 2×1
        self._last_gain = float(K[0, 0])

        # Update
        self._x = x_pred + K * innovation            # 2×1
        self._P = (np.eye(2) - K @ self._H) @ P_pred # 2×2

    # ── 변곡점 판정 ──────────────────────────────────────────────────────────

    def _detect_flip(self, slope: float, prev_slope: float) -> bool:
        """slope 부호 전환 탐지."""
        cfg = self.config
        if abs(slope) < cfg.slope_flip_min or abs(prev_slope) < cfg.slope_flip_min:
            return False
        return (slope > 0) != (prev_slope > 0)

    def _detect_surge(self, slope: float) -> bool:
        """slope 급변 탐지."""
        cfg = self.config
        hist = list(self._slope_hist)
        if len(hist) < 3:
            return False
        avg = float(np.mean([abs(s) for s in hist[-cfg.slope_history_n:]]))
        if avg < 1e-6:
            return False
        return abs(slope) >= avg * cfg.slope_surge_k

    def _determine_signal(
        self,
        slope:      float,
        prev_slope: float,
        flip:       bool,
    ) -> str:
        """변곡 신호 방향 결정.

        flip=True 전제:
          prev_slope > 0, slope < 0 → 고점 반전 → "down"
          prev_slope < 0, slope > 0 → 저점 반전 → "up"
        """
        if not flip:
            return "none"
        # prev_slope 부호로 이전 추세 방향 판단
        return "up" if prev_slope < 0 else "down"

    # ── Kalman Score ──────────────────────────────────────────────────────────

    def _calc_kalman_score(
        self,
        slope:      float,
        prev_slope: float,
        flip:       bool,
        surge:      bool,
    ) -> float:
        """Kalman 변곡 점수 [0, 1].

        구성:
        - slope flip       : 0.45 (핵심 신호)
        - slope surge      : 0.20 (강도 보조)
        - slope 절대값 크기: 0.20 (추세 강도)
        - innovation 크기  : 0.15 (관측 이탈도)
        """
        score = 0.0

        if flip:
            score += 0.45

        if surge:
            score += 0.20

        # slope 절대값 → 추세 강도 (최대 1pt/봉 기준 정규화)
        slope_strength = float(np.clip(abs(slope) / 1.0, 0.0, 1.0))
        score += slope_strength * 0.20

        # innovation 크기 → 가격의 필터 이탈도
        innov_strength = float(np.clip(
            abs(getattr(self, "_last_innovation", 0.0)) / 2.0, 0.0, 1.0
        ))
        score += innov_strength * 0.15

        return float(np.clip(score, 0.0, 1.0))

    # ── 시각 유틸 ────────────────────────────────────────────────────────────

    def _remember_time(self, bar_time: Any) -> None:
        hhmm = self._fmt(bar_time)
        if hhmm:
            self._hhmm_map[self._bar_idx] = hhmm
            if len(self._hhmm_map) > 4096:
                oldest = min(self._hhmm_map)
                del self._hhmm_map[oldest]

    def _fmt(self, bar_time: Any) -> Optional[str]:
        if bar_time is None:
            return None
        try:
            import pandas as pd
            return pd.Timestamp(bar_time).strftime("%H:%M")
        except Exception:
            s = str(bar_time).strip()
            return s[:5] if len(s) >= 5 and ":" in s else None

    def hhmm(self, idx: int) -> Optional[str]:
        return self._hhmm_map.get(idx)
