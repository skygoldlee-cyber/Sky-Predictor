# -*- coding: utf-8 -*-
"""
pivot_optuna_v2.py
==================

`48. 피봇탐색_성능검증.py` 의 Optuna 최적화 파이프라인 리팩토링 버전.

PIVOT_OPTUNA_REVIEW.md 의 개선항목 1~9 를 모두 반영한다.

    1. 목적함수를 '승률'에서 '비용차감 후 위험조정 수익(Sharpe/Expectancy/PF)'으로 교체
    2. 거래비용·슬리피지·계약승수 모델 도입
    3. 청크 in-sample 평균 → Purged/Embargo Walk-Forward 평가
    4. 지표(ATR/ADX/SuperTrend)는 전체 시계열에서 1회 계산 후 슬라이스, 검출기 상태 연속
    5. 피봇을 실제 극점(extreme)으로 기록, 진입은 '확정봉 다음봉 시가'
    6. 전역변수 제거 → FilterConfig / BacktestConfig 주입 (n_jobs 병렬 가능)
    7. Optuna: seed + multivariate TPE + MedianPruner + 제약(min_trades) + 중요도
    8. 백테스트 O(N×P) → O(P) 이벤트 드리븐 벡터화
    9. 데이터 로더: row-count 휴리스틱 → 날짜 범위 쿼리

원본 `indicators.hybrid_adaptive_pivot` 의 HybridAdaptivePivot / Config 를 그대로 재사용한다.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

# 검출기는 원본 구현을 재사용
from indicators.hybrid_adaptive_pivot import (
    HybridAdaptivePivot,
    HybridAdaptivePivotConfig,
)

try:
    import optuna
except ImportError:  # 최적화를 돌리지 않는 환경(백테스트만)에서도 import 가능하게
    optuna = None


# ════════════════════════════════════════════════════════════════════════════
# 설정 객체 (개선 #6: 전역변수 제거)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class FilterConfig:
    """피봇 후처리 필터 파라미터. 전역변수 대신 명시적으로 주입한다."""
    enabled: bool = True
    min_wave_pct: float = 0.3              # P1: 직전 '실제 극점' 대비 최소 파동 %
    min_pivot_interval_bars: int = 10      # P2: 피봇 간 최소 봉 간격
    st_distance_threshold: float = 0.1     # P5: SuperTrend 와의 최소 거리 %
    adx_hold_threshold: float = 15.0       # P10: 최소 ADX


@dataclass
class TrendConfig:
    """트렌드 팔로우 전략 파라미터."""
    method: str = "ma_crossover"           # 'ma_crossover' | 'breakout' | 'adx_trend'
    short_ma: int = 20                     # 단기 이동평균 기간
    long_ma: int = 60                      # 장기 이동평균 기간
    adx_threshold: float = 25.0           # ADX 기준 (추세 강도)
    breakout_period: int = 20             # 브레이크아웃 기간
    atr_multiplier: float = 2.0           # ATR 기반 손절매 배수


@dataclass
class BacktestConfig:
    """백테스트 비용/체결 모델 (개선 #2)."""
    multiplier: float = 250_000.0          # KOSPI200 선물 1pt 가치 (원)
    commission_pct_per_side: float = 0.00003   # 편도 수수료율 (체결금액 대비)
    slippage_ticks_per_side: float = 1.0       # 편도 슬리피지(틱)
    tick_size: float = 0.05                     # KOSPI200 선물 호가단위(pt)
    entry_on: str = "next_open"            # 'next_open'(권장) | 'confirm_close'
    annualization: float = 252.0           # 일 단위 Sharpe 연율화

    # ── 당일 청산(인트라데이) 설정 ───────────────────────────────────────
    intraday_only: bool = True             # True 면 오버나잇 보유 금지(거래일 종료 시 강제청산)
    session_boundary_hour: int = 8         # 이 시각 이전 봉(예: 00:00~07:59 야간세션 꼬리)은
                                            # '전일 거래일'로 귀속. KOSPI200 야간세션(18:00~익일05:00)
                                            # 이 자정을 넘기는 것을 처리. (주간 단독 데이터면 영향 없음)

    # ── 방향성 모드 설정 ───────────────────────────────────────────────────
    direction_mode: str = "both"           # 'both'(롱+숏) | 'long_only'(롱만) | 'short_only'(숏만)

    # ── 리스크 관리 설정 ───────────────────────────────────────────────────
    stop_loss_pct: float = 0.0             # 진입가 대비 손절 % (0이면 미사용)
    take_profit_pct: float = 0.0           # 진입가 대비 익절 % (0이면 미사용)
    trailing_stop_pct: float = 0.0       # 고점/저점 대비 트레일링 스탑 % (0이면 미사용)
    daily_loss_limit_krw: float = 0.0      # 일별 손실 한도 (원). 0이면 미사용
    position_size_mode: str = "fixed"      # 'fixed' | 'atr'
    atr_sizing_period: int = 14            # ATR 기반 사이징 기간
    atr_sizing_target_pts: float = 0.0     # legacy: 목표 ATR(points). 0이면 미사용
    atr_sizing_target_krw: float = 0.0     # 거래당 목표 위험금액 (원). 0이면 미사용
    max_position_size_factor: float = 3.0  # 기본 승수 대비 최대 증감 비율

    def round_trip_cost_pts(self, entry_px: float, exit_px: float) -> float:
        """한 거래(진입+청산)의 총 비용을 '포인트' 단위로 환산."""
        comm = self.commission_pct_per_side * (entry_px + exit_px)
        slip = 2.0 * self.slippage_ticks_per_side * self.tick_size
        return comm + slip


def trading_day_key(index: pd.DatetimeIndex, boundary_hour: int = 8) -> np.ndarray:
    """각 봉을 '거래일'로 매핑한 정수 키 배열을 만든다.

    KOSPI200 거래일 = 주간세션(09:00~15:45) + 그날 저녁 시작해 익일 새벽까지 가는
    야간세션(18:00~익일 05:00). 자정을 넘긴 새벽 봉(시각 < boundary_hour)은
    '전일' 거래일에 귀속시켜 하나의 야간세션이 두 날짜로 쪼개지지 않게 한다.
    """
    dates = index.normalize()
    is_early = index.hour < boundary_hour
    tday = dates - pd.to_timedelta(is_early.astype(int), unit="D")
    return tday.asi8  # 정수 키 (groupby/비교용)


@dataclass
class BacktestResult:
    n_trades: int = 0
    win_rate: float = 0.0           # 참고용으로만 보관 (최적화 목표 아님)
    total_pnl_pts: float = 0.0
    total_pnl_krw: float = 0.0
    expectancy_pts: float = 0.0     # 거래당 평균 순손익(pt)
    expectancy_krw: float = 0.0       # 거래당 평균 순손익(KRW)
    profit_factor: float = 0.0
    sharpe_daily: float = 0.0       # 일별 순손익 기준 연율화 Sharpe
    max_drawdown_krw: float = 0.0
    trades: Optional[pd.DataFrame] = None

    def as_dict(self) -> Dict:
        d = asdict(self)
        d.pop("trades", None)
        return d


# ════════════════════════════════════════════════════════════════════════════
# 지표 계산 — 인과적(causal), 전체 시계열에서 1회 (개선 #4)
# ════════════════════════════════════════════════════════════════════════════
def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["CLOSE"].shift(1)
    tr = pd.concat(
        [df["HIGH"] - df["LOW"],
         (df["HIGH"] - prev_close).abs(),
         (df["LOW"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["HIGH"], df["LOW"], df["CLOSE"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index)
                     .ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index)
                      .ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _supertrend(df: pd.DataFrame, period: int, mult: float) -> pd.Series:
    atr = _atr(df, period).to_numpy()
    hl2 = ((df["HIGH"] + df["LOW"]) / 2.0).to_numpy()
    close = df["CLOSE"].to_numpy()
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    n = len(df)
    st = np.full(n, np.nan)
    direction = 1
    for i in range(1, n):
        if np.isnan(atr[i]):
            continue
        prev = st[i - 1] if not np.isnan(st[i - 1]) else lower[i]
        if direction == 1:
            if close[i] <= lower[i]:
                direction, st[i] = -1, lower[i]
            else:
                st[i] = max(upper[i], prev)
        else:
            if close[i] >= upper[i]:
                direction, st[i] = 1, upper[i]
            else:
                st[i] = min(lower[i], prev)
    return pd.Series(st, index=df.index)


def compute_indicators(
    df: pd.DataFrame,
    atr_period: int = 14,
    adx_period: int = 14,
    st_period: int = 10,
    st_mult: float = 1.5,
) -> pd.DataFrame:
    """전체 시계열에서 지표를 1회 계산해 컬럼으로 붙여 반환 (개선 #4).

    Walk-Forward 시 이 결과를 슬라이스하면 fold 경계의 워밍업 NaN/재계산이 사라진다.
    """
    out = df.copy()
    out["ATR"] = _atr(out, atr_period)
    out["ADX"] = _adx(out, adx_period)
    out["ST"] = _supertrend(out, st_period, st_mult)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 피봇 검출 — 실제 극점 기록 + 연속 상태 + 필터 주입 (개선 #4,5,6)
# ════════════════════════════════════════════════════════════════════════════
def detect_pivots(
    df: pd.DataFrame,
    pivot_cfg: HybridAdaptivePivotConfig,
    filter_cfg: FilterConfig,
) -> pd.DataFrame:
    """피봇 검출.

    - 검출기 인스턴스는 df 전 구간에 대해 단 하나 (상태 연속, 개선 #4)
    - 피봇 가격/시각은 확정봉 종가가 아니라 '실제 극점'을 기록 (개선 #5)
    - 필터 파라미터는 filter_cfg 로 주입 (전역변수 제거, 개선 #6)
    - 진입 타이밍 계산용으로 confirm_time(신호 확정봉)도 함께 기록 (개선 #5)

    df 에는 compute_indicators 로 ATR/ADX/ST 컬럼이 이미 있어야 한다.
    """
    need = {"HIGH", "LOW", "CLOSE"}
    if not need.issubset(df.columns):
        raise ValueError(f"df must contain {need}")
    has_ind = {"ATR", "ADX", "ST"}.issubset(df.columns)

    detector = HybridAdaptivePivot(pivot_cfg)
    times = list(df.index)

    records: List[Dict] = []
    last_extreme_price: Optional[float] = None
    last_pivot_pos = -10**9

    highs = df["HIGH"].to_numpy()
    lows = df["LOW"].to_numpy()
    closes = df["CLOSE"].to_numpy()
    st = df["ST"].to_numpy() if has_ind else None
    adx = df["ADX"].to_numpy() if has_ind else None

    for i, ts in enumerate(times):
        state = detector.update(
            high=highs[i], low=lows[i], close=closes[i], bar_time=ts
        )
        sig = state.new_pivot_signal
        if sig not in ("new_high", "new_low"):
            continue

        is_high = sig == "new_high"
        # 개선 #5: 확정봉 종가가 아니라 검출기가 들고 있는 '실제 극점' 사용
        ext_price = state.last_high if is_high else state.last_low
        ext_idx = state.last_high_idx if is_high else state.last_low_idx
        if ext_price is None or (isinstance(ext_price, float) and math.isnan(ext_price)):
            ext_price = closes[i]
            ext_idx = i
        ext_idx = int(ext_idx)
        ext_time = times[ext_idx] if 0 <= ext_idx < len(times) else ts

        # ── 필터 (실제 극점 기준으로 계산) ──────────────────────────────
        if filter_cfg.enabled:
            # P1: 직전 극점 대비 파동 %
            if last_extreme_price is not None and last_extreme_price != 0:
                wave_pct = abs(ext_price - last_extreme_price) / abs(last_extreme_price) * 100
                if wave_pct < filter_cfg.min_wave_pct:
                    continue
            # P2: 피봇 간 최소 간격 (확정 위치 기준)
            if i - last_pivot_pos < filter_cfg.min_pivot_interval_bars:
                continue
            # P5: SuperTrend 거리
            if st is not None and not math.isnan(st[i]) and closes[i] != 0:
                st_dist = abs(closes[i] - st[i]) / abs(closes[i]) * 100
                if st_dist < filter_cfg.st_distance_threshold:
                    continue
            # P10: ADX
            if adx is not None and not math.isnan(adx[i]):
                if adx[i] < filter_cfg.adx_hold_threshold:
                    continue

        records.append({
            "pivot_time": ext_time,        # 실제 극점 시각 (분석용)
            "pivot_price": float(ext_price),  # 실제 극점 가격 (필터/분석용)
            "is_high": bool(is_high),
            "confirm_time": ts,            # 신호가 확정된 봉 (진입 가능 시점)
            "confirm_pos": i,
        })
        last_extreme_price = ext_price
        last_pivot_pos = i

    return pd.DataFrame(records)


def detect_pivots_daily(
    df: pd.DataFrame,
    pivot_cfg: HybridAdaptivePivotConfig,
    filter_cfg: FilterConfig,
    session_boundary_hour: int = 8,
) -> pd.DataFrame:
    """일자(거래일)별로 검출기 상태를 리셋하여 피봇을 검출한 뒤 합친다.

    `detect_pivots` 는 df 전 구간에 검출기 1개를 연속 적용하므로 전일 15:45 → 당일
    08:45 갭을 가로질러 ZigZag 방향/pending 상태가 이어진다. 이 함수는 거래일마다
    검출기(및 필터의 last_extreme/last_pivot 상태)를 새로 시작해 날 간 누수를 끊는다.

    주의:
    - 지표(ATR/ADX/ST)는 전체 시계열에서 1회 계산된 컬럼을 '그대로 슬라이스'해서 쓴다.
      (일자별로 지표를 재계산하면 매일 앞부분에 워밍업 NaN 이 생겨 P5/P10 필터가
       날마다 무력화되는 [C-3] 문제가 재발하므로 의도적으로 재계산하지 않는다.)
    - 반환되는 confirm_pos 는 '전체 df 기준' 위치로 보정되어 backtest 와 호환된다.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=["pivot_time", "pivot_price", "is_high", "confirm_time", "confirm_pos"]
        )

    tday = trading_day_key(df.index, session_boundary_hour)
    positions = np.arange(len(df))
    parts: List[pd.DataFrame] = []

    for day_val in pd.unique(tday):           # 시간순(등장순) 보존
        mask = tday == day_val
        day_df = df.iloc[mask]                 # 거래일 블록은 연속
        global_start = int(positions[mask][0])
        piv = detect_pivots(day_df, pivot_cfg, filter_cfg)   # 검출기 새로 시작
        if len(piv):
            piv = piv.copy()
            piv["confirm_pos"] = piv["confirm_pos"].astype(int) + global_start
            parts.append(piv)

    if not parts:
        return pd.DataFrame(
            columns=["pivot_time", "pivot_price", "is_high", "confirm_time", "confirm_pos"]
        )
    return pd.concat(parts, ignore_index=True)


# ════════════════════════════════════════════════════════════════════════════
# 트렌드 팔로우 신호 검출
# ════════════════════════════════════════════════════════════════════════════
def detect_trend_signals(
    df: pd.DataFrame,
    trend_cfg: TrendConfig,
) -> pd.DataFrame:
    """트렌드 팔로우 신호 검출.

    이동평균 크로스오버, 브레이크아웃, ADX 기반 추세 감지.
    피봇 검출과 동일한 포맷으로 반환하여 백테스트와 호환.
    """
    need = {"HIGH", "LOW", "CLOSE"}
    if not need.issubset(df.columns):
        raise ValueError(f"df must contain {need}")

    idx = df.index
    n = len(df)
    close = df["CLOSE"].to_numpy()
    high = df["HIGH"].to_numpy()
    low = df["LOW"].to_numpy()

    records: List[Dict] = []

    if trend_cfg.method == "ma_crossover":
        # 이동평균 크로스오버
        short_ma = pd.Series(close).rolling(trend_cfg.short_ma).mean().to_numpy()
        long_ma = pd.Series(close).rolling(trend_cfg.long_ma).mean().to_numpy()

        current_position = 0  # 0: neutral, 1: long, -1: short

        for i in range(trend_cfg.long_ma, n):
            if short_ma[i] > long_ma[i] and current_position != 1:
                # 롱 신호
                records.append({
                    "pivot_time": idx[i],
                    "pivot_price": close[i],
                    "is_high": False,  # 저점에서 롱 (피봇 포맷 호환)
                    "confirm_time": idx[i],
                    "confirm_pos": i,
                })
                current_position = 1
            elif short_ma[i] < long_ma[i] and current_position != -1:
                # 숏 신호
                records.append({
                    "pivot_time": idx[i],
                    "pivot_price": close[i],
                    "is_high": True,  # 고점에서 숏 (피봇 포맷 호환)
                    "confirm_time": idx[i],
                    "confirm_pos": i,
                })
                current_position = -1

    elif trend_cfg.method == "breakout":
        # 브레이크아웃
        period = trend_cfg.breakout_period
        highest = pd.Series(high).rolling(period).max().to_numpy()
        lowest = pd.Series(low).rolling(period).min().to_numpy()

        current_position = 0

        for i in range(period, n):
            if close[i] > highest[i-1] and current_position != 1:
                # 고점 돌파 → 롱
                records.append({
                    "pivot_time": idx[i],
                    "pivot_price": close[i],
                    "is_high": False,
                    "confirm_time": idx[i],
                    "confirm_pos": i,
                })
                current_position = 1
            elif close[i] < lowest[i-1] and current_position != -1:
                # 저점 이탈 → 숏
                records.append({
                    "pivot_time": idx[i],
                    "pivot_price": close[i],
                    "is_high": True,
                    "confirm_time": idx[i],
                    "confirm_pos": i,
                })
                current_position = -1

    elif trend_cfg.method == "adx_trend":
        # ADX 기반 추세
        if "ADX" not in df.columns:
            raise ValueError("ADX column required for adx_trend method")

        adx = df["ADX"].to_numpy()
        short_ma = pd.Series(close).rolling(trend_cfg.short_ma).mean().to_numpy()
        long_ma = pd.Series(close).rolling(trend_cfg.long_ma).mean().to_numpy()

        current_position = 0

        for i in range(trend_cfg.long_ma, n):
            # ADX가 충분히 높을 때만 신호
            if adx[i] >= trend_cfg.adx_threshold:
                if short_ma[i] > long_ma[i] and current_position != 1:
                    records.append({
                        "pivot_time": idx[i],
                        "pivot_price": close[i],
                        "is_high": False,
                        "confirm_time": idx[i],
                        "confirm_pos": i,
                    })
                    current_position = 1
                elif short_ma[i] < long_ma[i] and current_position != -1:
                    records.append({
                        "pivot_time": idx[i],
                        "pivot_price": close[i],
                        "is_high": True,
                        "confirm_time": idx[i],
                        "confirm_pos": i,
                    })
                    current_position = -1
            else:
                # ADX가 낮으면 중립
                current_position = 0

    else:
        raise ValueError(f"Unknown trend method: {trend_cfg.method}")

    if not records:
        return pd.DataFrame(
            columns=["pivot_time", "pivot_price", "is_high", "confirm_time", "confirm_pos"]
        )

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# 백테스트 — 이벤트 드리븐 O(P), 비용/승수 반영, 위험지표 산출 (개선 #1,2,5,8)
# ════════════════════════════════════════════════════════════════════════════
def backtest(df: pd.DataFrame, pivots: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    """반전(reversal) 시스템 백테스트.

    각 확정 피봇에서 포지션을 뒤집고 다음 확정 피봇까지 보유한다.
    진입은 '확정봉 다음봉 시가'(개선 #5), 손익은 비용·승수 반영(개선 #2),
    피봇 수와 무관하게 O(피봇수)로 계산(개선 #8).

    cfg.intraday_only=True 면 오버나잇 보유를 금지한다: 다음 피봇이 다음 거래일이면
    당일 마지막 봉(장마감) 종가로 강제청산하고, 그날 마지막 봉에서의 신규 진입은 스킵한다.
    trades['exit_reason'] 로 청산 사유('pivot'|'eod'|'final')를 확인할 수 있다.
    """
    empty = BacktestResult(trades=pd.DataFrame())
    if pivots is None or len(pivots) < 1:
        return empty

    idx = df.index
    has_open = "OPEN" in df.columns
    px_open = df["OPEN"].to_numpy() if has_open else df["CLOSE"].to_numpy()
    px_high = df["HIGH"].to_numpy()
    px_low = df["LOW"].to_numpy()
    px_close = df["CLOSE"].to_numpy()
    n = len(df)

    piv = pivots.sort_values("confirm_pos").reset_index(drop=True)

    # 당일청산(인트라데이): 거래일별 마지막 봉 인덱스 = 그날 장마감(≈15:45) 봉
    intraday = cfg.intraday_only
    if intraday:
        tday = trading_day_key(idx, cfg.session_boundary_hour)
        day_last = (
            pd.Series(np.arange(n)).groupby(tday).transform("max").to_numpy()
        )
    else:
        tday = np.array([])

    # ── 리스크 관리 사전 준비 ───────────────────────────────────────────────
    use_sizing = cfg.position_size_mode == "atr" and cfg.atr_sizing_target_krw > 0
    if use_sizing:
        atr = _atr(df, cfg.atr_sizing_period).to_numpy()
    else:
        atr = np.array([])

    daily_loss_used: Dict[int, float] = {}
    has_loss_limit = cfg.daily_loss_limit_krw > 0

    # 각 피봇의 '진입 가능 위치'
    events: List[Tuple[int, int]] = []   # (entry_pos, direction)
    for _, p in piv.iterrows():
        cpos = int(p["confirm_pos"])
        epos = cpos + 1 if cfg.entry_on == "next_open" else cpos
        if epos >= n:
            continue
        # 당일청산: 진입봉이 거래일 마지막 봉이면 같은 날 청산 불가 → 진입 스킵
        if intraday and epos >= int(day_last[epos]):
            continue
        direction = -1 if p["is_high"] else 1   # 고점→숏, 저점→롱

        # 방향성 모드 필터링
        if cfg.direction_mode == "long_only" and direction != 1:
            continue
        if cfg.direction_mode == "short_only" and direction != -1:
            continue

        events.append((epos, direction))

    if not events:
        return empty

    rows = []
    for k in range(len(events)):
        e_pos, d = events[k]
        e_day = int(tday[e_pos]) if intraday else idx[e_pos].date()
        if has_loss_limit and daily_loss_used.get(e_day, 0.0) <= -cfg.daily_loss_limit_krw:
            continue

        e_px = px_open[e_pos] if cfg.entry_on == "next_open" else px_close[e_pos]
        nxt = events[k + 1][0] if k + 1 < len(events) else None

        if intraday:
            eod = int(day_last[e_pos])
            if nxt is not None and nxt <= eod:
                # 같은 거래일 안에서 다음 피봇 발생 → 거기서 반전 청산
                x_pos = nxt
                x_px = px_open[x_pos] if cfg.entry_on == "next_open" else px_close[x_pos]
                reason = "pivot"
            else:
                # 다음 피봇이 다음 거래일 → 당일 마지막 봉(장마감) 종가로 강제청산
                x_pos = eod
                x_px = px_close[x_pos]
                reason = "eod"
        else:
            if nxt is not None:
                x_pos = nxt
                x_px = px_open[x_pos] if cfg.entry_on == "next_open" else px_close[x_pos]
                reason = "pivot"
            else:
                x_pos = n - 1
                x_px = px_close[x_pos]
                reason = "final"

        if x_pos <= e_pos:
            continue

        # ── 포지션 사이징 (ATR 기반) ─────────────────────────────────────
        sl_pct = cfg.stop_loss_pct
        tp_pct = cfg.take_profit_pct
        trail_pct = cfg.trailing_stop_pct
        if d == 1:
            sl_price = e_px * (1 - sl_pct) if sl_pct > 0 else 0.0
            tp_price = e_px * (1 + tp_pct) if tp_pct > 0 else 0.0
        else:
            sl_price = e_px * (1 + sl_pct) if sl_pct > 0 else 0.0
            tp_price = e_px * (1 - tp_pct) if tp_pct > 0 else 0.0

        if use_sizing:
            if sl_pct > 0:
                risk_pts = abs(e_px - sl_price)
            else:
                risk_pts = float(atr[e_pos]) if len(atr) > e_pos else 0.0
            risk_pts = max(risk_pts, 1e-9)
            size_factor = cfg.atr_sizing_target_krw / (risk_pts * cfg.multiplier)
            size_factor = max(1.0 / cfg.max_position_size_factor, min(size_factor, cfg.max_position_size_factor))
        else:
            size_factor = 1.0
        effective_multiplier = cfg.multiplier * size_factor

        # ── 인트라-트레이드 손절/익절/트레일링 스캔 ─────────────────────
        exit_pos = x_pos
        exit_px = x_px
        exit_reason = reason
        if sl_pct > 0 or tp_pct > 0 or trail_pct > 0:
            if d == 1:
                high_watermark = e_px
                for i in range(e_pos + 1, x_pos + 1):
                    if px_high[i] > high_watermark:
                        high_watermark = px_high[i]
                    eff_sl = sl_price
                    if trail_pct > 0:
                        trail = high_watermark * (1 - trail_pct)
                        if eff_sl == 0.0:
                            eff_sl = trail
                        else:
                            eff_sl = max(eff_sl, trail)
                    if eff_sl > 0 and px_low[i] <= eff_sl:
                        exit_pos = i
                        exit_px = eff_sl
                        exit_reason = (
                            "stop"
                            if (sl_pct > 0 and abs(eff_sl - sl_price) < 1e-9)
                            else "trail"
                        )
                        break
                    if tp_pct > 0 and px_high[i] >= tp_price:
                        exit_pos = i
                        exit_px = tp_price
                        exit_reason = "tp"
                        break
            else:
                low_watermark = e_px
                for i in range(e_pos + 1, x_pos + 1):
                    if px_low[i] < low_watermark:
                        low_watermark = px_low[i]
                    eff_sl = sl_price
                    if trail_pct > 0:
                        trail = low_watermark * (1 + trail_pct)
                        if eff_sl == 0.0:
                            eff_sl = trail
                        else:
                            eff_sl = min(eff_sl, trail)
                    if eff_sl > 0 and px_high[i] >= eff_sl:
                        exit_pos = i
                        exit_px = eff_sl
                        exit_reason = (
                            "stop"
                            if (sl_pct > 0 and abs(eff_sl - sl_price) < 1e-9)
                            else "trail"
                        )
                        break
                    if tp_pct > 0 and px_low[i] <= tp_price:
                        exit_pos = i
                        exit_px = tp_price
                        exit_reason = "tp"
                        break

        gross_pts = d * (exit_px - e_px)
        cost_pts = cfg.round_trip_cost_pts(e_px, exit_px)
        net_krw = (gross_pts - cost_pts) * effective_multiplier
        net_pts = net_krw / cfg.multiplier

        rows.append({
            "entry_time": idx[e_pos], "exit_time": idx[exit_pos],
            "direction": d, "entry_px": e_px, "exit_px": exit_px,
            "exit_reason": exit_reason,
            "gross_pts": gross_pts, "cost_pts": cost_pts,
            "net_pts": net_pts, "net_krw": net_krw,
            "size_factor": size_factor,
        })

        if has_loss_limit:
            daily_loss_used[e_day] = daily_loss_used.get(e_day, 0.0) + net_krw

    if not rows:
        return empty

    tdf = pd.DataFrame(rows)
    net = tdf["net_pts"]
    wins = net[net > 0]
    losses = net[net < 0]

    total_pts = float(net.sum())
    total_krw = float(tdf["net_krw"].sum())
    n_trades = int(len(tdf))
    win_rate = float((net > 0).mean() * 100)
    expectancy = float(net.mean())
    expectancy_krw = float(tdf["net_krw"].mean())
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    # 일별 순손익 → 연율화 Sharpe (개선 #1)
    tdf["exit_date"] = pd.to_datetime(tdf["exit_time"]).dt.date
    daily = tdf.groupby("exit_date")["net_krw"].sum()
    if len(daily) >= 2 and daily.std(ddof=1) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(cfg.annualization))
    else:
        sharpe = 0.0

    # 최대낙폭 (원화 누적)
    equity = tdf["net_krw"].cumsum()
    running_max = equity.cummax()
    max_dd = float((equity - running_max).min())

    return BacktestResult(
        n_trades=n_trades,
        win_rate=win_rate,
        total_pnl_pts=total_pts,
        total_pnl_krw=total_krw,
        expectancy_pts=expectancy,
        expectancy_krw=expectancy_krw,
        profit_factor=profit_factor,
        sharpe_daily=sharpe,
        max_drawdown_krw=max_dd,
        trades=tdf,
    )


# ════════════════════════════════════════════════════════════════════════════
# Purged / Embargo Walk-Forward 평가 (개선 #3)
# ════════════════════════════════════════════════════════════════════════════
def purged_walkforward_folds(
    df: pd.DataFrame, n_splits: int, embargo_bars: int
) -> Iterator[pd.DataFrame]:
    """전체 구간을 연속 n_splits 개로 나누되, fold 사이에 embargo_bars 만큼
    버퍼를 둬서 피봇/지표가 경계를 넘어 새어나가지 않게 한다 (개선 #3).

    각 fold 는 '서로 다른 시장 구간' 이며, 목적함수는 fold 별 위험지표의
    평균/안정성을 보므로 특정 구간 과적합을 억제한다.
    """
    n = len(df)
    fold = n // n_splits
    for i in range(n_splits):
        start = i * fold
        end = (i + 1) * fold if i < n_splits - 1 else n
        # 앞쪽 embargo 만큼은 이전 fold 와의 경계 오염 방지를 위해 제외
        s = start + (embargo_bars if i > 0 else 0)
        if end - s < 100:
            continue
        yield df.iloc[s:end]


# ════════════════════════════════════════════════════════════════════════════
# Optuna 목적함수 / 최적화 (개선 #1,3,7)
# ════════════════════════════════════════════════════════════════════════════
def make_objective(
    df_train_ind: pd.DataFrame,
    bt_cfg: BacktestConfig,
    n_splits: int = 4,
    embargo_bars: int = 30,
    min_total_trades: int = 40,
    metric: str = "sharpe",        # 'sharpe' | 'expectancy' | 'profit_factor'
    robustness_lambda: float = 0.5,  # 목적 = mean(metric) - lambda * std(metric)
    daily_reset: bool = False,       # True 면 일자별 검출기 리셋(detect_pivots_daily)
) -> Callable:
    """위험조정 수익 기반 + Walk-Forward + 제약 목적함수를 생성한다.

    df_train_ind 는 compute_indicators 가 이미 적용된 학습 구간.
    daily_reset=True 면 fold 내 피봇 검출을 거래일별로 끊어서 수행한다.
    """
    def _metric(res: BacktestResult) -> float:
        if metric == "sharpe":
            return res.sharpe_daily
        if metric == "expectancy":
            return res.expectancy_pts
        if metric == "profit_factor":
            pf = res.profit_factor
            return min(pf, 10.0) if math.isfinite(pf) else 10.0
        raise ValueError(metric)

    def objective(trial) -> float:
        pivot_cfg = HybridAdaptivePivotConfig(
            base_pct=trial.suggest_float("base_pct", 0.05, 2.0, log=True),
            base_multiplier=trial.suggest_float("base_multiplier", 0.5, 10.0),
            atr_weight=trial.suggest_float("atr_weight", 0.0, 1.0),
            confirmation_bars=trial.suggest_int("confirmation_bars", 0, 10),
        )
        filter_cfg = FilterConfig(
            enabled=True,
            min_wave_pct=trial.suggest_float("min_wave_pct", 0.05, 2.0, log=True),
            min_pivot_interval_bars=trial.suggest_int("min_pivot_interval_bars", 1, 30),
            st_distance_threshold=trial.suggest_float("st_distance_threshold", 0.01, 1.0, log=True),
            adx_hold_threshold=trial.suggest_float("adx_hold_threshold", 5.0, 50.0),
        )

        scores: List[float] = []
        total_trades = 0
        MIN_FOLD_TRADES = 3          # fold 당 최소 거래수
        NO_TRADE_PENALTY = -1.0      # 무거래/소량 fold 페널티 (0 동률화 방지 + 페널티)
        for step, fold in enumerate(purged_walkforward_folds(df_train_ind, n_splits, embargo_bars)):
            if daily_reset:
                pivots = detect_pivots_daily(
                    fold, pivot_cfg, filter_cfg, bt_cfg.session_boundary_hour
                )
            else:
                pivots = detect_pivots(fold, pivot_cfg, filter_cfg)
            res = backtest(fold, pivots, bt_cfg)
            total_trades += res.n_trades
            # 거래가 거의 없는 fold 를 '중립 0' 이 아니라 음수로 처리:
            #  (1) 무거래 파라미터를 올바르게 페널티  (2) 목적값 분산 확보 → 중요도 계산 가능
            if res.n_trades < MIN_FOLD_TRADES:
                scores.append(NO_TRADE_PENALTY)
            else:
                scores.append(_metric(res))
            # 개선 #7: 중간 결과 보고 → MedianPruner 가 가망없는 trial 조기 중단
            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # 개선 #7: 거래수 제약 (constraints_func 가 읽는다). <=0 이면 feasible
        constraint = float(min_total_trades - total_trades)
        trial.set_user_attr("constraints", (constraint,))
        trial.set_user_attr("total_trades", total_trades)

        if not scores:
            return -1e9
        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores)) if len(scores) > 1 else 0.0
        # 구간 간 일관성 보상 (개선 #3): 평균은 높고 편차는 낮은 파라미터 선호
        return mean_s - robustness_lambda * std_s

    return objective


def _constraints(trial) -> Tuple[float, ...]:
    return trial.user_attrs.get("constraints", (0.0,))


def _safe_param_importances(study) -> Tuple[Dict[str, float], Optional[str]]:
    """파라미터 중요도를 안전하게 계산한다.

    fanova 는 (1) 완료 trial < 2 (2) 목적값 분산 ≈ 0 (대부분 동일 점수) 일 때 예외를
    던진다. 이 경우 조용히 {} 로 삼키지 말고 '왜 비었는지'를 사유 문자열로 돌려준다.
    fanova 실패 시 PedAnova → MDI 로 폴백한다.
    """
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if len(complete) < 2:
        return {}, f"완료 trial {len(complete)}개 (≥2 필요). pruner 가 과한지 확인."
    vals = np.array([t.value for t in complete], dtype=float)
    if float(np.nanstd(vals)) < 1e-12:
        return {}, ("완료 trial 목적값 분산≈0 (대부분 동일 점수). "
                    "거래 0건 fold 가 0점으로 동률화된 경우가 흔함 → 탐색범위/페널티 재설계 필요.")

    evaluators = [None]
    for name in ("PedAnovaImportanceEvaluator", "MeanDecreaseImpurityImportanceEvaluator"):
        ev = getattr(optuna.importance, name, None)
        if ev is not None:
            evaluators.append(ev)

    last_err = None
    for ev in evaluators:
        try:
            imp = (optuna.importance.get_param_importances(study) if ev is None
                   else optuna.importance.get_param_importances(study, evaluator=ev()))
            return {k: float(v) for k, v in imp.items()}, None
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
    return {}, last_err


def optimize(
    df_train_ind: pd.DataFrame,
    n_trials: int = 200,
    seed: int = 42,
    bt_cfg: Optional[BacktestConfig] = None,
    output_dir: Optional[Path] = None,
    **obj_kwargs,
) -> Dict:
    """seed 고정 + multivariate TPE + MedianPruner + 제약 + 중요도 (개선 #7)."""
    if optuna is None:
        raise RuntimeError("optuna 가 설치되어 있지 않습니다.")
    bt_cfg = bt_cfg or BacktestConfig()

    sampler = optuna.samplers.TPESampler(
        multivariate=True, group=True, seed=seed, constraints_func=_constraints
    )
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    objective = make_objective(df_train_ind, bt_cfg, **obj_kwargs)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 제약(min_total_trades)을 만족하는 trial 이 하나도 없으면 best_trial 이 예외를 던진다.
    # 이 경우 '거래수 하한 미충족'을 알리고, 완료된 trial 중 값이 가장 큰 것으로 폴백한다.
    feasible = True
    try:
        best = study.best_trial
    except ValueError:
        feasible = False
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if not completed:
            raise RuntimeError(
                "완료된 trial 이 없습니다. n_trials/필터범위/데이터구간을 확인하세요."
            )
        best = max(completed, key=lambda t: t.value)

    importance, imp_err = _safe_param_importances(study)

    from collections import Counter
    states = dict(Counter(t.state.name for t in study.trials))

    result = {
        "best_params": best.params,
        "best_value": best.value,
        "best_metric": obj_kwargs.get("metric", "sharpe"),
        "best_total_trades": best.user_attrs.get("total_trades"),
        "constraint_satisfied": feasible,
        "n_trials": len(study.trials),
        "trial_states": states,
        "param_importances": importance,
        "param_importance_error": imp_err,
        "backtest_cfg": asdict(bt_cfg),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "optuna_optimization_v2.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    return result


# ════════════════════════════════════════════════════════════════════════════
# 데이터 로더 — 날짜 범위 쿼리 (개선 #9)
# ════════════════════════════════════════════════════════════════════════════
def load_data_by_date(
    db_path: str,
    table_name: str,
    start: Optional[str] = None,   # 'YYYY-MM-DD'
    end: Optional[str] = None,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """row-count 휴리스틱(days*500) 대신 timestamp 범위로 정확히 로드 (개선 #9).

    timestamp 컬럼이 VARCHAR('YYYYMMDD HHMM', 원본 형식)이든 진짜 TIMESTAMP
    타입이든 모두 안전하게 처리한다.
    """
    import duckdb
    con = duckdb.connect(db_path, read_only=True)

    # 컬럼 타입 판별 → 문자열/타임스탬프에 맞는 WHERE·파싱 선택
    info = con.execute(f"PRAGMA table_info('{table_name}')").df()
    row = info[info["name"].str.lower() == ts_col.lower()]
    col_type = (row["type"].iloc[0] if len(row) else "VARCHAR").upper()
    is_str = any(k in col_type for k in ("CHAR", "STRING", "TEXT"))

    where = []
    if is_str:
        if start:
            where.append(f"{ts_col} >= '{start.replace('-', '')} 0000'")
        if end:
            where.append(f"{ts_col} <= '{end.replace('-', '')} 2359'")
    else:  # TIMESTAMP / DATE
        if start:
            where.append(f"{ts_col} >= TIMESTAMP '{start} 00:00:00'")
        if end:
            where.append(f"{ts_col} <= TIMESTAMP '{end} 23:59:59'")
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    df = con.execute(
        f"SELECT * FROM {table_name}{clause} ORDER BY {ts_col}"
    ).df()
    con.close()

    if is_str:
        df[ts_col] = pd.to_datetime(df[ts_col], format="%Y%m%d %H%M")
    else:
        df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col)
    df.columns = df.columns.str.upper()
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.75) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """시간 순서 단일 홀드아웃 분할 (과거→학습, 미래→테스트). 누수 없음."""
    cut = int(len(df) * train_frac)
    return df.iloc[:cut], df.iloc[cut:]


def filter_day_session(
    df: pd.DataFrame, start: str = "08:45", end: str = "15:45"
) -> pd.DataFrame:
    """주간세션 시간대(기본 08:45~15:45)만 남긴다. 야간세션 봉 제거.

    지표/피봇/백테스트 전 단계에서 1회 적용하면 파이프라인 전체가 주간세션만 다룬다.
    구간은 양끝 포함(inclusive).
    """
    return df.between_time(start, end, inclusive="both")


def intraday_long_baseline(df: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    """벤치마크: 매 거래일 시가 진입 → 종가 청산하는 '장중 상시 롱'.

    전략의 롱 leg 가 *알파*(피봇 타이밍의 가치)인지 단순 *베타*(그냥 롱 노출)인지
    가른다. 이 베이스라인의 Sharpe/PnL 이 전략 롱 leg 와 비슷하거나 더 좋으면,
    롱 수익은 피봇이 만든 게 아니라 그 구간이 올랐기 때문일 뿐이다.
    """
    idx = df.index
    px_open = df["OPEN"].to_numpy() if "OPEN" in df.columns else df["CLOSE"].to_numpy()
    px_close = df["CLOSE"].to_numpy()
    tday = trading_day_key(idx, cfg.session_boundary_hour)
    pos = np.arange(len(df))
    rows = []
    for day_val in pd.unique(tday):
        mask = tday == day_val
        first = int(pos[mask][0]); last = int(pos[mask][-1])
        if last <= first:
            continue
        e_px, x_px = px_open[first], px_close[last]
        net = (x_px - e_px) - cfg.round_trip_cost_pts(e_px, x_px)
        rows.append({"exit_time": idx[last], "net_pts": net, "net_krw": net * cfg.multiplier})
    if not rows:
        return BacktestResult(trades=pd.DataFrame())
    tdf = pd.DataFrame(rows)
    daily = tdf.groupby(pd.to_datetime(tdf["exit_time"]).dt.date)["net_krw"].sum()
    sharpe = (float(daily.mean() / daily.std(ddof=1) * math.sqrt(cfg.annualization))
              if len(daily) >= 2 and daily.std(ddof=1) > 0 else 0.0)
    equity = tdf["net_krw"].cumsum()
    return BacktestResult(
        n_trades=len(tdf),
        win_rate=float((tdf["net_pts"] > 0).mean() * 100),
        total_pnl_pts=float(tdf["net_pts"].sum()),
        total_pnl_krw=float(tdf["net_krw"].sum()),
        expectancy_pts=float(tdf["net_pts"].mean()),
        sharpe_daily=sharpe,
        max_drawdown_krw=float((equity - equity.cummax()).min()),
        trades=tdf,
    )


def intraday_long_with_stoploss(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    stoploss_pct: float = 0.02,  # 손절매 비율 (2%)
    volatility_filter: bool = True,
    atr_threshold: float = 0.5,  # ATR 기반 변동성 필터
) -> BacktestResult:
    """장중 상시 롱 + 손절매 + 변동성 필터.

    손절매:
    - 진입 후 stoploss_pct 손실률 도달 시 강제 청산
    - 당일 종가까지 손절매 발생하지 않으면 종가 청산

    변동성 필터:
    - ATR 기반 변동성이 너무 낮으면 진입 스킵
    - 고변동 구간에서 포지션 축소 가능
    """
    idx = df.index
    px_open = df["OPEN"].to_numpy() if "OPEN" in df.columns else df["CLOSE"].to_numpy()
    px_close = df["CLOSE"].to_numpy()
    px_high = df["HIGH"].to_numpy() if "HIGH" in df.columns else px_close
    px_low = df["LOW"].to_numpy() if "LOW" in df.columns else px_close

    tday = trading_day_key(idx, cfg.session_boundary_hour)
    pos = np.arange(len(df))
    rows = []

    # 전일 데이터 저장
    prev_atr = None

    for day_val in pd.unique(tday):
        mask = tday == day_val
        first = int(pos[mask][0])
        last = int(pos[mask][-1])
        if last <= first:
            continue

        # 진입 필터 체크
        entry = True

        # 변동성 필터: ATR 기반
        if volatility_filter and "ATR" in df.columns and prev_atr is not None:
            current_atr = df["ATR"].iloc[first]
            if current_atr < prev_atr * atr_threshold:  # 변동성 너무 낮음
                entry = False

        if entry:
            e_px = px_open[first]
            stoploss_px = e_px * (1 - stoploss_pct)  # 손절매 가격

            # 당일 중간 손절매 체크
            stopped = False
            exit_px = px_close[last]  # 기본: 종가 청산
            exit_time = idx[last]

            for i in range(first, last + 1):
                if px_low[i] <= stoploss_px:  # 손절매 발생
                    stopped = True
                    exit_px = stoploss_px
                    exit_time = idx[i]
                    break

            net = (exit_px - e_px) - cfg.round_trip_cost_pts(e_px, exit_px)
            rows.append({"exit_time": exit_time, "net_pts": net, "net_krw": net * cfg.multiplier})

        # 전일 데이터 업데이트
        if "ATR" in df.columns:
            prev_atr = df["ATR"].iloc[last]

    if not rows:
        return BacktestResult(trades=pd.DataFrame())

    tdf = pd.DataFrame(rows)
    daily = tdf.groupby(pd.to_datetime(tdf["exit_time"]).dt.date)["net_krw"].sum()
    sharpe = (float(daily.mean() / daily.std(ddof=1) * math.sqrt(cfg.annualization))
              if len(daily) >= 2 and daily.std(ddof=1) > 0 else 0.0)
    equity = tdf["net_krw"].cumsum()
    return BacktestResult(
        n_trades=len(tdf),
        win_rate=float((tdf["net_pts"] > 0).mean() * 100),
        total_pnl_pts=float(tdf["net_pts"].sum()),
        total_pnl_krw=float(tdf["net_krw"].sum()),
        expectancy_pts=float(tdf["net_pts"].mean()),
        sharpe_daily=sharpe,
        max_drawdown_krw=float((equity - equity.cummax()).min()),
        trades=tdf,
    )


def enhanced_intraday_long(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    filter_previous_candle: bool = True,
    filter_atr_ratio: float = 0.5,
) -> BacktestResult:
    """장중 상시 롱 + 간단한 필터.

    필터:
    - filter_previous_candle: 전일 음봉 시 진입 (반등 기대)
    - filter_atr_ratio: ATR 기반 변동성 필터 (변동성 너무 낮으면 진입 스킵)
    """
    idx = df.index
    px_open = df["OPEN"].to_numpy() if "OPEN" in df.columns else df["CLOSE"].to_numpy()
    px_close = df["CLOSE"].to_numpy()
    px_high = df["HIGH"].to_numpy() if "HIGH" in df.columns else px_close
    px_low = df["LOW"].to_numpy() if "LOW" in df.columns else px_close

    tday = trading_day_key(idx, cfg.session_boundary_hour)
    pos = np.arange(len(df))
    rows = []

    # 전일 데이터 저장
    prev_close = None
    prev_atr = None

    for day_val in pd.unique(tday):
        mask = tday == day_val
        first = int(pos[mask][0])
        last = int(pos[mask][-1])
        if last <= first:
            continue

        # 진입 필터 체크
        entry = True

        # 필터 1: 전일 음봉 시 진입 (반등 기대)
        if filter_previous_candle and prev_close is not None:
            if px_close[first-1] >= px_open[first-1]:  # 전일 양봉
                entry = False

        # 필터 2: ATR 기반 변동성 필터
        if filter_atr_ratio > 0 and "ATR" in df.columns and prev_atr is not None:
            current_atr = df["ATR"].iloc[first]
            if current_atr < prev_atr * filter_atr_ratio:  # 변동성 너무 낮음
                entry = False

        if entry:
            e_px, x_px = px_open[first], px_close[last]
            net = (x_px - e_px) - cfg.round_trip_cost_pts(e_px, x_px)
            rows.append({"exit_time": idx[last], "net_pts": net, "net_krw": net * cfg.multiplier})

        # 전일 데이터 업데이트
        prev_close = px_close[last]
        if "ATR" in df.columns:
            prev_atr = df["ATR"].iloc[last]

    if not rows:
        return BacktestResult(trades=pd.DataFrame())

    tdf = pd.DataFrame(rows)
    daily = tdf.groupby(pd.to_datetime(tdf["exit_time"]).dt.date)["net_krw"].sum()
    sharpe = (float(daily.mean() / daily.std(ddof=1) * math.sqrt(cfg.annualization))
              if len(daily) >= 2 and daily.std(ddof=1) > 0 else 0.0)
    equity = tdf["net_krw"].cumsum()
    return BacktestResult(
        n_trades=len(tdf),
        win_rate=float((tdf["net_pts"] > 0).mean() * 100),
        total_pnl_pts=float(tdf["net_pts"].sum()),
        total_pnl_krw=float(tdf["net_krw"].sum()),
        expectancy_pts=float(tdf["net_pts"].mean()),
        sharpe_daily=sharpe,
        max_drawdown_krw=float((equity - equity.cummax()).min()),
        trades=tdf,
    )


def regime_based_intraday(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    regime_method: str = "ma",
    ma_short: int = 20,
    ma_long: int = 60,
    adx_threshold: float = 25.0,
    filter_atr_ratio: float = 0.0,
    gap_threshold: float = 0.01,
    intraday_reversal: bool = False,
) -> BacktestResult:
    """레짐 기반 당일 매매: 상승장 롱, 하락장 숏, 횡보장 스킵.

    레짐 감지:
    - ma: 이동평균 크로스오버 (MA 단기 > 장기 = 상승장)
    - adx: ADX 기반 추세 강도 (ADX > threshold + MA 방향)

    필터:
    - filter_atr_ratio: ATR 기반 변동성 필터 (변동성 너무 낮으면 진입 스킵)
    - gap_threshold: 갭 크기 필터 (갭이 threshold 이상이면 진입 스킵)
    - intraday_reversal: 당일 중간 레짐 역전 감지 및 포지션 전환
    """
    idx = df.index
    px_open = df["OPEN"].to_numpy() if "OPEN" in df.columns else df["CLOSE"].to_numpy()
    px_close = df["CLOSE"].to_numpy()
    n = len(df)

    tday = trading_day_key(idx, cfg.session_boundary_hour)
    pos = np.arange(len(df))
    rows = []

    # 레짐 감지용 이동평균 계산
    if regime_method == "ma" or regime_method == "adx":
        short_ma = pd.Series(px_close).rolling(ma_short).mean().to_numpy()
        long_ma = pd.Series(px_close).rolling(ma_long).mean().to_numpy()

    # ADX 계산 (필요시)
    if regime_method == "adx":
        if "ADX" not in df.columns:
            raise ValueError("ADX column required for adx regime method")
        adx = df["ADX"].to_numpy()

    # ATR 계산 (필터용)
    if filter_atr_ratio > 0:
        if "ATR" not in df.columns:
            raise ValueError("ATR column required for ATR filter")
        atr = df["ATR"].to_numpy()
        prev_atr = None

    # 전일 종가 저장 (갭 계산용)
    prev_close = None

    for day_val in pd.unique(tday):
        mask = tday == day_val
        first = int(pos[mask][0])
        last = int(pos[mask][-1])
        if last <= first:
            continue

        # 갭 필터
        gap_filter_ok = True
        if gap_threshold > 0 and prev_close is not None:
            gap = (px_open[first] - prev_close) / prev_close
            if abs(gap) > gap_threshold:
                gap_filter_ok = False

        # ATR 필터
        atr_filter_ok = True
        if filter_atr_ratio > 0 and prev_atr is not None:
            if atr[first] < prev_atr * filter_atr_ratio:
                atr_filter_ok = False
            prev_atr = atr[last]

        if intraday_reversal:
            # 당일 중간 레짐 역전 감지 및 포지션 전환
            current_position = None  # 'long' or 'short' or None
            entry_px = None

            for i in range(first, last + 1):
                # 레짐 감지
                regime = "neutral"
                if regime_method == "ma":
                    if i >= ma_long:
                        if short_ma[i] > long_ma[i]:
                            regime = "bull"
                        elif short_ma[i] < long_ma[i]:
                            regime = "bear"
                elif regime_method == "adx":
                    if i >= ma_long:
                        if adx[i] >= adx_threshold:
                            if short_ma[i] > long_ma[i]:
                                regime = "bull"
                            elif short_ma[i] < long_ma[i]:
                                regime = "bear"

                # 포지션 전환 로직
                if regime == "bull" and atr_filter_ok and gap_filter_ok:
                    if current_position == "short":
                        # 숏 → 롤 전환: 숏 청산
                        net = (entry_px - px_close[i]) - cfg.round_trip_cost_pts(entry_px, px_close[i])
                        rows.append({"exit_time": idx[i], "net_pts": net, "net_krw": net * cfg.multiplier})
                        # 롱 진입
                        entry_px = px_close[i]
                        current_position = "long"
                    elif current_position is None:
                        # 롱 진입
                        entry_px = px_close[i]
                        current_position = "long"
                elif regime == "bear" and atr_filter_ok and gap_filter_ok:
                    if current_position == "long":
                        # 롱 → 숏 전환: 롱 청산
                        net = (px_close[i] - entry_px) - cfg.round_trip_cost_pts(entry_px, px_close[i])
                        rows.append({"exit_time": idx[i], "net_pts": net, "net_krw": net * cfg.multiplier})
                        # 숏 진입
                        entry_px = px_close[i]
                        current_position = "short"
                    elif current_position is None:
                        # 숏 진입
                        entry_px = px_close[i]
                        current_position = "short"
                else:
                    # 횡보장: 포지션 청산
                    if current_position == "long":
                        net = (px_close[i] - entry_px) - cfg.round_trip_cost_pts(entry_px, px_close[i])
                        rows.append({"exit_time": idx[i], "net_pts": net, "net_krw": net * cfg.multiplier})
                        current_position = None
                    elif current_position == "short":
                        net = (entry_px - px_close[i]) - cfg.round_trip_cost_pts(entry_px, px_close[i])
                        rows.append({"exit_time": idx[i], "net_pts": net, "net_krw": net * cfg.multiplier})
                        current_position = None

            # 장종료: 남은 포지션 청산
            if current_position == "long":
                net = (px_close[last] - entry_px) - cfg.round_trip_cost_pts(entry_px, px_close[last])
                rows.append({"exit_time": idx[last], "net_pts": net, "net_krw": net * cfg.multiplier})
            elif current_position == "short":
                net = (entry_px - px_close[last]) - cfg.round_trip_cost_pts(entry_px, px_close[last])
                rows.append({"exit_time": idx[last], "net_pts": net, "net_krw": net * cfg.multiplier})
        else:
            # 기본: 당일 시가 진입 → 종가 청산
            # 레짐 감지
            regime = "neutral"
            if regime_method == "ma":
                if first >= ma_long:
                    if short_ma[first] > long_ma[first]:
                        regime = "bull"
                    elif short_ma[first] < long_ma[first]:
                        regime = "bear"
            elif regime_method == "adx":
                if first >= ma_long:
                    if adx[first] >= adx_threshold:
                        if short_ma[first] > long_ma[first]:
                            regime = "bull"
                        elif short_ma[first] < long_ma[first]:
                            regime = "bear"

            # 진입/청산
            if regime == "bull" and atr_filter_ok and gap_filter_ok:
                # 상승장: 롱 진입
                e_px, x_px = px_open[first], px_close[last]
                net = (x_px - e_px) - cfg.round_trip_cost_pts(e_px, x_px)
                rows.append({"exit_time": idx[last], "net_pts": net, "net_krw": net * cfg.multiplier})
            elif regime == "bear" and atr_filter_ok and gap_filter_ok:
                # 하락장: 숏 진입
                e_px, x_px = px_open[first], px_close[last]
                net = (e_px - x_px) - cfg.round_trip_cost_pts(e_px, x_px)
                rows.append({"exit_time": idx[last], "net_pts": net, "net_krw": net * cfg.multiplier})
            # 횡보장: 진입 스킵

        # 전일 종가 업데이트
        prev_close = px_close[last]

    if not rows:
        return BacktestResult(trades=pd.DataFrame())

    tdf = pd.DataFrame(rows)
    daily = tdf.groupby(pd.to_datetime(tdf["exit_time"]).dt.date)["net_krw"].sum()
    sharpe = (float(daily.mean() / daily.std(ddof=1) * math.sqrt(cfg.annualization))
              if len(daily) >= 2 and daily.std(ddof=1) > 0 else 0.0)
    equity = tdf["net_krw"].cumsum()
    return BacktestResult(
        n_trades=len(tdf),
        win_rate=float((tdf["net_pts"] > 0).mean() * 100),
        total_pnl_pts=float(tdf["net_pts"].sum()),
        total_pnl_krw=float(tdf["net_krw"].sum()),
        expectancy_pts=float(tdf["net_pts"].mean()),
        sharpe_daily=sharpe,
        max_drawdown_krw=float((equity - equity.cummax()).min()),
        trades=tdf,
    )


def diagnose_windows(
    df_ind: pd.DataFrame,
    pivot_cfg: HybridAdaptivePivotConfig,
    filter_cfg: FilterConfig,
    bt_cfg: BacktestConfig,
    n_windows: int = 6,
    daily_reset: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """고정 파라미터로 전체 구간을 n_windows 등분해 구간별 성과를 본다.

    단일 홀드아웃(마지막 25~30%)이 '운 좋은 레짐'인지 판단하는 용도.
    특히 롱/숏 손익을 분리 출력하므로, 숏이 *모든 구간*에서 죽어 있으면 구조적 문제,
    *일부 구간*에서만 0이면 레짐(상승장에서 숏이 안 먹힘) 문제로 구분할 수 있다.

    df_ind 는 compute_indicators 가 적용된 (주간세션) 전체 구간.
    """
    n = len(df_ind)
    w = n // n_windows
    rows = []
    for i in range(n_windows):
        s = i * w
        e = (i + 1) * w if i < n_windows - 1 else n
        seg = df_ind.iloc[s:e]
        piv = (detect_pivots_daily(seg, pivot_cfg, filter_cfg, bt_cfg.session_boundary_hour)
               if daily_reset else detect_pivots(seg, pivot_cfg, filter_cfg))
        res = backtest(seg, piv, bt_cfg)
        t = res.trades
        l_pnl = s_pnl = 0.0
        l_n = s_n = 0
        if t is not None and len(t):
            lm = t["direction"] == 1
            sm = t["direction"] == -1
            l_pnl = float(t.loc[lm, "net_krw"].sum()); l_n = int(lm.sum())
            s_pnl = float(t.loc[sm, "net_krw"].sum()); s_n = int(sm.sum())
        rows.append({
            "window": i + 1,
            "start": seg.index[0], "end": seg.index[-1],
            "n_trades": res.n_trades,
            "sharpe": round(res.sharpe_daily, 3),
            "pnl_krw": round(res.total_pnl_krw),
            "long_pnl": round(l_pnl), "long_n": l_n,
            "short_pnl": round(s_pnl), "short_n": s_n,
        })
    out = pd.DataFrame(rows)
    if verbose:
        print(f"{'win':>3} {'기간':<23}{'거래':>5}{'Sharpe':>9}{'PnL(원)':>14}"
              f"{'롱PnL':>13}{'숏PnL':>13}")
        for _, r in out.iterrows():
            period = f"{pd.Timestamp(r['start']).date()}~{pd.Timestamp(r['end']).date()}"
            print(f"{int(r['window']):>3} {period:<23}{int(r['n_trades']):>5}"
                  f"{r['sharpe']:>9.3f}{int(r['pnl_krw']):>14,}"
                  f"{int(r['long_pnl']):>13,}{int(r['short_pnl']):>13,}")
        pos = (out["pnl_krw"] > 0).sum()
        s_pos = (out["short_pnl"] > 0).sum()
        both = int(out["pnl_krw"].sum())
        long_only = int(out["long_pnl"].sum())
        short_leg = int(out["short_pnl"].sum())
        print(f"  → 수익 구간 {pos}/{len(out)} | 숏 수익 구간 {s_pos}/{len(out)}")
        print(f"  → 합계  both={both:,}  롱only={long_only:,}  숏leg={short_leg:,}")
        if short_leg < 0:
            print(f"     숏 leg 가 전체적으로 음수(드래그). 숏 제거 시 {both:,} → {long_only:,} "
                  f"({long_only - both:+,}) 로 개선 가능성.")
        print("     (숏이 일부 구간만 수익이면 레짐 의존, 전 구간 음수면 구조적 결함)")
    return out
