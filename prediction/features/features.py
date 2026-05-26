"""Feature engineering helpers for the prediction pipeline.

This module converts raw market snapshots into numerical features.

Key responsibilities:
- Parse FH0 orderbook/quote snapshots into stable numeric features (with alias support).
- Compute candle/minute-bar features from OHLCV DataFrames.
- Build fixed-length sequences that combine orderbook + candle features.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime

from config import FUTURE_KNOWN_DIM
from .time_features import build_time_features


logger = logging.getLogger(__name__)


OB_KEYS = [
    "obi",
    "obi_delta1",
    "obi_delta5",
    "obi_ema5",
    "spread",
    "level1_ratio",
    "bid_slope",
    "offer_slope",
    "totbidrem",
    "totofferrem",
]

CD_KEYS = [
    "ret1", "ret3", "slope3", "vol_accel", "range_pct",
    # [IMP-CD-01] 거래량 구조 피처
    "vol_ratio_ma5",   # 현재 거래량 / 5봉 평균 거래량. 거래량 급증 감지.
    "cvd_slope3",      # 3봉 누적 델타 거래량(Close>Open → +Vol, else -Vol) 기울기.
    "body_ratio",      # |Close-Open| / (High-Low+ε). 캔들 실체 비율 [0,1].
]

OPT_KEYS_V1 = [
    "pcr_volume",
    "iv_skew",
    "max_pain_dist_pct",
    "atm_iv",
    "atm_spread_pct",
    "atm_orderbook_imb",
    "atm_liquidity_log",
]

# Extended option micro-movement features derived from OH0 + option_minute_ohlcv.
OPT_KEYS_V2 = OPT_KEYS_V1 + [
    "optm_call_ret",
    "optm_put_ret",
    "optm_straddle_ret",
    "optm_call_range_pct",
    "optm_put_range_pct",
    "optm_straddle_range_pct",
    "optm_call_vol",
    "optm_put_vol",
    "optm_straddle_vol",
]

# v3: v2 피처에 만기주 콜-풋 패리티 이탈 피처 추가.
# ordering은 모델 입력 차원에 영향을 주므로 변경 금지.
OPT_KEYS_V3 = OPT_KEYS_V2 + [
    "parity_spread_pct",        # 패리티 이탈 비율 (%). 이론 대비 C-P 가격차 이탈.
    "call_delta_proxy",         # C/(C+P) 델타 근사값 [0, 1]. ATM 이론값 = 0.5.
    "straddle_price",           # ATM C+P. 내재 변동성 크기의 직접 지표.
    "straddle_vs_fut_move",     # straddle_price / |F-K|. 선물 이동 대비 스트래들 배율.
    "call_vs_fut_ret_diff",     # 콜 수익률 - (0.5 * 선물 수익률). 추종 이탈.
    "dte_weight_norm",          # 만기 근접도 [0, 1]. 당일 = 1.0, 10일 전 ≈ 0.4.
    "parity_divergence_score",  # 종합 이탈 스코어 [-1, 1]. DTE 가중 적용.
]

# v4: v3 피처에 선물 상승 중 옵션 프리미엄 수축(Premium Bleed) 피처 추가.
# 원인: Theta 급가속, IV Crush, MM 롤오버. 만기 당일~2일 전 유효.
# ordering은 모델 입력 차원에 영향을 주므로 변경 금지.
OPT_KEYS_V4 = OPT_KEYS_V3 + [
    "straddle_decay_vs_fut",    # 스트래들 수익률 - |선물 수익률|*0.5. 음수 = 비정상 수축.
    "iv_crush_proxy",           # ATM IV 방향 근사 변화율. 음수 = IV 감소.
    "fut_ret",                  # 직전 틱 선물 수익률. 방향 확인용.
    "straddle_now",             # 현재 ATM 스트래들 가격 (C+P).
    "straddle_prev",            # 직전 틱 ATM 스트래들 가격.
    "premium_bleed_score",      # 종합 프리미엄 수축 스코어 [-1, 1]. -1 = 강한 수축.
]

# v5: v4 피처에 OI 기반 지지저항 레벨 및 딜러 감마 구조 피처 추가.
# Dealer Gamma Hedge Flow(Zero Gamma, Volatility Trigger)를 수치화.
# ordering은 모델 입력 차원에 영향을 주므로 변경 금지.
# 모델 재학습 필요: prediction/weights/tft_v5.pt, transformer_v5.pt
OPT_KEYS_V5 = OPT_KEYS_V4 + [
    "dist_to_call_peak",    # 현재가 → Call OI Peak 거리(%). 양수 = 저항이 현재가 위.
    "dist_to_put_peak",     # 현재가 → Put OI Peak 거리(%). 양수 = 지지가 현재가 아래.
    "oi_center_dist_pct",   # 현재가 → OI 박스 중심가격 거리(%). 0이면 정중앙.
    "oi_range_pct",         # OI 박스 폭(%). 크면 레인지가 넓음.
    "call_oi_peak_norm",    # Call OI 집중도 [0, 1]. 클수록 상단 저항이 강함.
    "put_oi_peak_norm",     # Put OI 집중도 [0, 1]. 클수록 하단 지지가 강함.
    "above_vol_trigger",    # 1.0 = Vol Trigger 위(Dealer Long Gamma, 안정), 0.0 = 아래(불안정).
    "zero_gamma_dist_pct",  # 현재가 → Zero Gamma Level 거리(%). 0 근처 = 레짐 전환 임박.
]

# Backward-compat alias (defaults to v1).
OPT_KEYS = OPT_KEYS_V1

# ─────────────────────────────────────────────────────────────────────────────
# 멀티스케일 피처 (5분봉 기준) — MS5_KEYS
# ─────────────────────────────────────────────────────────────────────────────
# 1분봉 CD_KEYS와 개념은 같되 5분봉 집계에서 파생한 중기 추세 피처.
# 활성화: config.json prediction.multiscale_5m = true (기본 false)
# PAST_UNKNOWN_DIM: 47 → 55 (+8) 로 확장됨.
# ordering은 모델 입력 차원에 영향을 주므로 변경 금지.
MS5_KEYS: list[str] = [
    "ms5_ret5",           # 5분봉 수익률
    "ms5_slope5",         # 5분봉 종가 선형기울기 (3봉 정규화)
    "ms5_vol_ratio_ma20", # 현재 5분봉 거래량 / 20봉 MA
    "ms5_range5_pct",     # 5분봉 (High-Low)/Close
    "ms5_body5_ratio",    # 5분봉 캔들 실체 비율
    "ms5_ret5_3bar",      # 3개 5분봉 누적 수익률 (15분 방향)
    "ms5_cvd5_slope3",    # 5분봉 CVD 3봉 기울기
    "ms5_vol_accel",      # 5분봉 거래량 가속도 (이전 봉 대비)
]

# 15분 스케일 피처 — MS15_KEYS
# ─────────────────────────────────────────────────────────────────────────────
# 장기 추세와 변동성 감지용 15분봉 기반 피처
MS15_KEYS: list[str] = [
    "ms15_ret15",         # 15분봉 수익률
    "ms15_slope15",       # 15분봉 종가 선형기울기 (3봉 정규화)
    "ms15_vol_ratio_ma12",# 현재 15분봉 거래량 / 12봉 MA
    "ms15_range15_pct",   # 15분봉 (High-Low)/Close
    "ms15_body15_ratio",  # 15분봉 캔들 실체 비율
    "ms15_ret15_2bar",    # 2개 15분봉 누적 수익률 (30분 방향)
    "ms15_cvd15_slope2",  # 15분봉 CVD 2봉 기울기
    "ms15_vol_accel",     # 15분봉 거래량 가속도
]


def get_opt_keys(feature_set: str = "v1") -> list[str]:
    s = str(feature_set or "v1").strip().lower()
    if s == "v5":
        return list(OPT_KEYS_V5)
    if s == "v4":
        return list(OPT_KEYS_V4)
    if s == "v3":
        return list(OPT_KEYS_V3)
    if s == "v2":
        return list(OPT_KEYS_V2)
    return list(OPT_KEYS_V1)


def calc_multiscale_features(df_1m: pd.DataFrame, resample_rule: str = "5min") -> pd.DataFrame:
    """1분봉 OHLCV를 5분봉으로 리샘플링해 MS5_KEYS 피처를 계산한다.

    Args:
        df_1m: calc_candle_features() 이전의 원시 1분봉 OHLCV DataFrame.
               index: DatetimeIndex, 컬럼: Open/High/Low/Close/Volume.
        resample_rule: pandas resample 규칙 (기본 "5min").

    Returns:
        ms5_df: DatetimeIndex(5분봉 기준), 컬럼 = MS5_KEYS.
                각 행은 해당 5분 구간 종료 시점의 피처 값.
                index는 1분봉 index와 병합(reindex)에 사용된다.
    """
    try:
        if df_1m is None or df_1m.empty:
            return pd.DataFrame(columns=MS5_KEYS)

        # ── 5분봉 OHLCV 집계 ─────────────────────────────────────────────
        rule = str(resample_rule or "5min")
        df5 = df_1m.resample(rule, closed="left", label="right").agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna(subset=["Close"])

        if df5.empty:
            return pd.DataFrame(columns=MS5_KEYS)

        out = pd.DataFrame(index=df5.index)

        # ms5_ret5: 5분봉 종가 수익률
        out["ms5_ret5"] = df5["Close"].pct_change(1).fillna(0.0)

        # ms5_slope5: 3봉 종가 선형 기울기 / 평균종가 (정규화)
        slope_raw = df5["Close"].diff(1) / 3.0
        price_mean = df5["Close"].rolling(3).mean().replace(0, np.nan)
        out["ms5_slope5"] = (slope_raw / price_mean).fillna(0.0)

        # ms5_vol_ratio_ma20: 현재 거래량 / 20봉 평균
        ma20 = df5["Volume"].rolling(20).mean()
        out["ms5_vol_ratio_ma20"] = (df5["Volume"] / (ma20 + 1e-9)).fillna(1.0)

        # ms5_range5_pct: (High-Low)/Close
        out["ms5_range5_pct"] = (
            (df5["High"] - df5["Low"]) / (df5["Close"].abs() + 1e-9)
        ).fillna(0.0)

        # ms5_body5_ratio: 캔들 실체 비율
        out["ms5_body5_ratio"] = (
            (df5["Close"] - df5["Open"]).abs()
            / (df5["High"] - df5["Low"] + 1e-9)
        ).fillna(0.0)

        # ms5_ret5_3bar: 3봉 누적 수익률 (15분 방향)
        out["ms5_ret5_3bar"] = df5["Close"].pct_change(3).fillna(0.0)

        # ms5_cvd5_slope3: 5분봉 CVD 3봉 기울기
        _delta5 = df5["Volume"] * (
            (df5["Close"] > df5["Open"]).astype(float) * 2 - 1
        )
        out["ms5_cvd5_slope3"] = (
            _delta5.rolling(3).sum().diff(1)
            / (df5["Close"].abs() + 1e-9)
        ).fillna(0.0)

        # ms5_vol_accel: 거래량 가속도 (이전 봉 대비 비율)
        vol_prev = df5["Volume"].shift(1).replace(0, np.nan)
        out["ms5_vol_accel"] = (df5["Volume"] / (vol_prev + 1e-9)).fillna(1.0)

        return out[MS5_KEYS].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    except Exception:
        logger.debug("calc_multiscale_features 실패", exc_info=True)
        return pd.DataFrame(columns=MS5_KEYS)


def calc_multiscale_features_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """1분봉 OHLCV를 15분봉으로 리샘플링해 MS15_KEYS 피처를 계산한다.

    Args:
        df_1m: 원시 1분봉 OHLCV DataFrame (index: DatetimeIndex).

    Returns:
        ms15_df: 15분봉 기준 MS15_KEYS 피처.
                 closed="left", label="right" — 5분봉(calc_multiscale_features)과 동일한
                 바 경계 기준을 사용하므로 08:45 장 시작봉도 일관되게 집계된다.
                 (08:45~08:59 → 09:00, 08:45~08:49 → 08:50)
    """
    try:
        if df_1m is None or df_1m.empty:
            return pd.DataFrame(columns=MS15_KEYS)

        # ── 15분봉 OHLCV 집계 (5분봉과 동일하게 closed="left", label="right") ──
        df15 = df_1m.resample("15min", closed="left", label="right").agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna(subset=["Close"])

        if df15.empty:
            return pd.DataFrame(columns=MS15_KEYS)

        out = pd.DataFrame(index=df15.index)

        # ms15_ret15: 15분봉 종가 수익률
        out["ms15_ret15"] = df15["Close"].pct_change(1).fillna(0.0)

        # ms15_slope15: 3봉 종가 선형 기울기 / 평균종가 (5분봉 ms5_slope5 와 동일 방식)
        slope_raw = df15["Close"].diff(1) / 3.0
        price_mean = df15["Close"].rolling(3).mean().replace(0, np.nan)
        out["ms15_slope15"] = (slope_raw / price_mean).fillna(0.0)

        # ms15_vol_ratio_ma12: 현재 거래량 / 12봉 평균
        ma12 = df15["Volume"].rolling(12).mean()
        out["ms15_vol_ratio_ma12"] = (df15["Volume"] / (ma12 + 1e-9)).fillna(1.0)

        # ms15_range15_pct: (High-Low)/Close
        out["ms15_range15_pct"] = (
            (df15["High"] - df15["Low"]) / (df15["Close"].abs() + 1e-9)
        ).fillna(0.0)

        # ms15_body15_ratio: 캔들 실체 비율
        out["ms15_body15_ratio"] = (
            (df15["Close"] - df15["Open"]).abs()
            / (df15["High"] - df15["Low"] + 1e-9)
        ).fillna(0.0)

        # ms15_ret15_2bar: 2봉 누적 수익률 (30분 방향)
        out["ms15_ret15_2bar"] = df15["Close"].pct_change(2).fillna(0.0)

        # ms15_cvd15_slope2: 15분봉 CVD 2봉 기울기
        _delta15 = df15["Volume"] * (
            (df15["Close"] > df15["Open"]).astype(float) * 2 - 1
        )
        out["ms15_cvd15_slope2"] = (
            _delta15.rolling(2).sum().diff(1)
            / (df15["Close"].abs() + 1e-9)
        ).fillna(0.0)

        # ms15_vol_accel: 거래량 가속도 (이전 봉 대비 비율)
        vol_prev = df15["Volume"].shift(1).replace(0, np.nan)
        out["ms15_vol_accel"] = (df15["Volume"] / (vol_prev + 1e-9)).fillna(1.0)

        return out[MS15_KEYS].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    except Exception:
        logger.debug("calc_multiscale_features_15m 실패", exc_info=True)
        return pd.DataFrame(columns=MS15_KEYS)


def calc_all_multiscale_features(df_1m: pd.DataFrame, time_scales: list[int]) -> dict[str, pd.DataFrame]:
    """설정된 시간 스케일들에 대한 멀티스케일 피처를 모두 계산한다.
    
    Args:
        df_1m: 원시 1분봉 OHLCV DataFrame
        time_scales: 시간 스케일 리스트 (예: [1, 5, 15])
        
    Returns:
        dict: 스케일별 피처 DataFrame {"ms1": df, "ms5": df, "ms15": df}
    """
    results = {}
    
    try:
        for scale in time_scales:
            if scale == 1:
                # 1분봉은 기존 candle_features 사용
                continue
            elif scale == 5:
                results["ms5"] = calc_multiscale_features(df_1m, "5min")
            elif scale == 15:
                results["ms15"] = calc_multiscale_features_15m(df_1m)
            else:
                logger.warning(f"지원하지 않는 시간 스케일: {scale}분")
                
        logger.info(f"멀티스케일 피처 계산 완료: {list(results.keys())}")
        return results
        
    except Exception as e:
        logger.error(f"calc_all_multiscale_features 실패: {e}", exc_info=True)
        return {}


# Adaptive indicators (adaptive_supertrend + adaptive_zigzag + cross features)
# Ordering must remain stable because it affects model input dimensions.
ADAPT_KEYS = [
    # Adaptive SuperTrend
    "ast_direction",
    "ast_dist_pct",
    "ast_atr_pct",
    "ast_efficiency_ratio",
    "ast_adx_norm",
    "ast_mult_norm",
    "ast_trend_duration",
    "ast_signal",
    "ast_band_width_pct",
    # Adaptive ZigZag
    "azz_direction",
    "azz_wave_size_pct",
    "azz_support_dist_pct",
    "azz_res_dist_pct",
    "azz_bars_since_swing",
    "azz_fib618_dist",
    "azz_fib382_dist",
    "azz_higher_highs",
    "azz_lower_lows",
    "azz_new_swing",
    "azz_swing_recency",
    "azz_threshold_pct",
    "azz_structure_up",
    "azz_structure_down",
    "azz_structure_ranging",
    # 보완-3: micro_structure / structure_confidence
    "azz_micro_up",
    "azz_micro_down",
    "azz_micro_ranging",
    "azz_structure_conf",
    # 보완-6: pending 잠정 S/R 거리
    "azz_pend_sr_dist",
    # 후보 확정 확률
    "azz_pending_type",
    "azz_pending_dist",
    "azz_pending_urgency",
    "azz_pending_age",
    "azz_pending_prob",
    # Cross
    "cross_trend_agreement",
    "cross_at_support",
    "cross_at_resistance",
    "cross_breakout_potential",
    # ── Step 1: ATR Adaptive Pivot (aap_*) ───────────────────────────────
    # Ordering must remain stable; append-only.
    "aap_atr",               # ATR 정규화 [0,1]
    "aap_threshold_pct",     # 동적 threshold %
    "aap_pivot_score",       # ATR Pivot 점수 [0,1]
    # ── Step 2: Market Structure Break (msb_*) ──────────────────────────
    "msb_bos_signal",        # BOS 신호 방향 {-1,0,+0.5,+1}
    "msb_structure",         # 시장 구조 {-1,0,+1}
    "msb_hh_ratio",          # Higher-High 비율 [0,1]
    "msb_ll_ratio",          # Lower-Low 비율 [0,1]
    "msb_sh_dist",           # 최근 swing high 거리 [-1,1]
    "msb_sl_dist",           # 최근 swing low 거리 [-1,1]
    "msb_score",             # MSB 강도 [0,1]
    "msb_choch",             # CHoCH 발생 여부 {0,1}
    # ── Step 3a: Kalman Turning Point (kf_*) ───────────────────────────
    "kf_slope_norm",         # 정규화 slope [-1,1]
    "kf_slope_flip",         # slope 부호 전환 {0,1}
    "kf_slope_surge",        # slope 급변 {0,1}
    "kf_turning_signal",     # 전환 방향 {-1,0,+1}
    "kf_score",              # Kalman 변곡 점수 [0,1]
    "kf_dev_norm",           # 가격-필터 괴리 [-1,1]
    "kf_innovation",         # 혁신(잔차) 정규화 [-1,1]
    "kf_gain",               # Kalman gain [0,1]
    # ── Step 3b: OI Structure Gate (oi_*) ────────────────────────────────
    # OIStructureGate.get_transformer_features() 실제 반환 키와 정확히 일치
    "oi_msb_score",      # OI 교차 가중 MSB 점수 [0,1]
    "oi_near_call",      # Call OI peak 근접 {0,1}
    "oi_near_put",       # Put OI peak 근접 {0,1}
    "oi_call_dist",      # Call peak 까지 거리 [-1,1]
    "oi_put_dist",       # Put peak 까지 거리 [-1,1]
    "oi_bos_boosted",    # BOS+OI 부스트 발생 {0,1}
    # ── Step 3c: 통합 PivotScore (ps_*) ────────────────────────────────
    "ps_total_score",        # 4-Layer 가중합 [0,1]
    "ps_adjusted_score",     # 레짐 조정 후 점수 [0,1]
    "ps_signal",             # 진입 방향 {-1,0,+1}
    "ps_strength",           # 신호 강도 {0,0.3,0.6,1}
    "ps_long",               # long 신호 {0,1}
    "ps_short",              # short 신호 {0,1}
]


def calc_orderbook_features(quote: dict) -> dict:
    """Compute orderbook features from a single FH0-like quote dict.

    The live FH0 schema may vary by environment; this function supports:
    - L1 keys: `offerho`, `bidho` (per FH0_OH0_SCHEMA.md)
    - Optional quantity keys: `offerrem`, `bidrem` (when provided by some feeds)
    - Optional depth keys: `offerho1~5`, `bidho1~5`, `offerrem1~5`, `bidrem1~5`

    Returns a dict including:
    - `obi`, `spread`, `level1_ratio`
    - `bid_slope`, `offer_slope`
    - `totbidrem`, `totofferrem`

    If required quotes are missing, returns a dict with `_invalid=True`.
    """
    q = quote if isinstance(quote, dict) else {}

    def _as_float(v) -> float:
        """Best-effort conversion of a value to float; returns 0.0 on failure."""
        try:
            if v is None:
                return 0.0
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip()
            return float(s) if s != "" else 0.0
        except Exception:
            return 0.0

    def _get_any(keys) -> float:
        """Return the first non-null value found for any key in `keys` (as float)."""
        for k in keys:
            if k in q and q.get(k) is not None:
                return _as_float(q.get(k))
        return 0.0

    offer_l1_qty = _get_any(
        [
            "offerrem",
            "offerrem1",
            "offer_rem",
            "askrem",
            "ask_rem",
            "offer_qty",
            "ask_qty",
        ]
    )
    bid_l1_qty = _get_any(
        [
            "bidrem",
            "bidrem1",
            "bid_rem",
            "bidqty",
            "bid_qty",
        ]
    )

    total_offer = _get_any(["totofferrem", "tot_offer_rem", "total_offer_rem", "total_offer", "totoffer"])
    total_bid = _get_any(["totbidrem", "tot_bid_rem", "total_bid_rem", "total_bid", "totbid"])

    # eBest FH0/OH0 use offerho/bidho for best quotes.
    offer1 = _get_any(["offerho", "offerho1", "ask1", "ask", "offer1"])
    bid1 = _get_any(["bidho", "bidho1", "bid1", "bid"])

    # Some feeds provide full depth as offerho1..5/bidho1..5 but omit offerho/bidho.
    if offer1 <= 0.0:
        offer1 = _get_any(["offerho1", "ask1"])
    if bid1 <= 0.0:
        bid1 = _get_any(["bidho1", "bid1"])

    # Basic validation: if bid/ask are missing, downstream features become noisy.
    # Return zeros (skip-like behavior) so caller can decide to drop.
    if offer1 <= 0.0 or bid1 <= 0.0:
        return {
            "obi": 0.0,
            "spread": 0.0,
            "level1_ratio": 0.0,
            "bid_slope": 0.0,
            "offer_slope": 0.0,
            "totbidrem": float(total_bid),
            "totofferrem": float(total_offer),
            "_invalid": True,
        }

    spread = float(offer1) - float(bid1)
    if not np.isfinite(float(spread)):
        spread = 0.0
    if spread < 0.0:
        # Some feeds can momentarily invert due to update ordering.
        spread = abs(float(spread))

    bid_rems = [_get_any([f"bidrem{i}", f"bid_rem{i}", f"bidrem_{i}", f"bid_qty{i}", f"bidqty{i}"]) for i in range(1, 6)]
    offer_rems = [_get_any([f"offerrem{i}", f"offer_rem{i}", f"offerrem_{i}", f"ask_qty{i}", f"askqty{i}"]) for i in range(1, 6)]

    # Some FH0 feeds may only provide L1 qty.
    if bid_rems[0] <= 0.0 and bid_l1_qty > 0.0:
        bid_rems[0] = float(bid_l1_qty)
    if offer_rems[0] <= 0.0 and offer_l1_qty > 0.0:
        offer_rems[0] = float(offer_l1_qty)

    # If full depth is present, use it to derive totals when tot* fields are missing.
    if total_offer <= 0.0 and any(v > 0.0 for v in offer_rems):
        total_offer = float(sum(offer_rems))
    if total_bid <= 0.0 and any(v > 0.0 for v in bid_rems):
        total_bid = float(sum(bid_rems))

    # Final fallback: if totals are still missing, use L1 totals to keep OBI meaningful.
    if total_offer <= 0.0 and offer_rems[0] > 0.0:
        total_offer = float(offer_rems[0])
    if total_bid <= 0.0 and bid_rems[0] > 0.0:
        total_bid = float(bid_rems[0])

    total = float(total_offer) + float(total_bid)
    denom = float(max(total, 1e-9))
    try:
        obi = float(np.clip((float(total_bid) - float(total_offer)) / denom, -1.0, 1.0))
    except Exception:
        obi = 0.0

    level1_total = float(bid_rems[0]) + float(offer_rems[0])
    level1_denom = float(max(level1_total, 1e-9))
    try:
        level1_ratio = float(np.clip((float(bid_rems[0]) - float(offer_rems[0])) / level1_denom, -1.0, 1.0))
    except Exception:
        level1_ratio = 0.0

    bid_slope = 0.0
    offer_slope = 0.0
    try:
        x = np.arange(5, dtype=np.float32)
        bid_arr = np.array(bid_rems, dtype=np.float32)
        offer_arr = np.array(offer_rems, dtype=np.float32)

        if float(np.nansum(bid_arr)) > 0.0:
            b0 = float(bid_arr[0])
            b0 = b0 if abs(b0) > 1e-9 else 1.0
            bid_slope = float(np.polyfit(x, bid_arr, 1)[0]) / float(b0)

        if float(np.nansum(offer_arr)) > 0.0:
            o0 = float(offer_arr[0])
            o0 = o0 if abs(o0) > 1e-9 else 1.0
            offer_slope = float(np.polyfit(x, offer_arr, 1)[0]) / float(o0)
    except Exception:
        bid_slope = 0.0
        offer_slope = 0.0

    return {
        "obi": obi,
        "spread": spread,
        "level1_ratio": level1_ratio,
        "bid_slope": bid_slope,
        "offer_slope": offer_slope,
        "totbidrem": total_bid,
        "totofferrem": total_offer,
    }


def calc_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candle features from an OHLCV DataFrame.

    Expected columns: `Open`, `High`, `Low`, `Close`, `Volume`.
    Returns a DataFrame with normalized/filled numeric features.
    """
    out = pd.DataFrame(index=df.index)
    out["ret1"] = df["Close"].pct_change(1)
    out["ret3"] = df["Close"].pct_change(3)
    out["slope3"] = df["Close"].diff(3) / 3.0
    out["vol_accel"] = df["Volume"] / (df["Volume"].rolling(5).mean() + 1e-9)
    out["range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)

    # [IMP-CD-01] 거래량 구조 피처
    # vol_ratio_ma5: 현재 거래량 / 5봉 평균 — 거래량 급증 감지 (vol_accel과 다른 스케일)
    out["vol_ratio_ma5"] = df["Volume"] / (df["Volume"].rolling(5).mean() + 1e-9)

    # cvd_slope3: 3봉 누적 델타 거래량 기울기.
    # 델타 = Close > Open이면 +Volume, else -Volume (방향성 거래량)
    _delta = df["Volume"] * ((df["Close"] > df["Open"]).astype(float) * 2 - 1)
    out["cvd_slope3"] = _delta.rolling(3).sum().diff(1) / (df["Close"].abs() + 1e-9)

    # body_ratio: 캔들 실체 비율 [0,1]. 1에 가까울수록 강한 방향성 봉.
    out["body_ratio"] = (df["Close"] - df["Open"]).abs() / (df["High"] - df["Low"] + 1e-9)

    return out.fillna(0.0)


def build_sequence(
    ob_records: list[dict],
    candle_df: "pd.DataFrame | None",
    seq_len: int = 60,
    opt_features: "dict | None" = None,
    adaptive_features: "dict | None" = None,
    opt_keys_override: "list[str] | None" = None,
    ms5_features: "dict | None" = None,
    multiscale_features: "dict[str, pd.DataFrame] | None" = None,
) -> np.ndarray:
    """Build a fixed-length 2D feature sequence.

    Output shape: (seq_len, N)
    - orderbook(len(OB_KEYS)) + candle(len(CD_KEYS)) + option(len(OPT_KEYS))
      + [ms5(len(MS5_KEYS))] + [multiscale(len(MS5_KEYS)+MS15_KEYS)] + [adaptive(len(ADAPT_KEYS))] + time(FUTURE_KNOWN_DIM)

    Args:
        ms5_features: MS5_KEYS 기준의 5분봉 피처 딕셔너리 (레거시).
                      None이면 멀티스케일 블록을 포함하지 않는다.
        multiscale_features: 멀티스케일 피처 딕셔너리 {"ms5": df, "ms15": df}.
                             None이면 멀티스케일 블록을 포함하지 않는다.
    """

    ob_keys = OB_KEYS
    cd_keys = CD_KEYS
    opt_keys = opt_keys_override if opt_keys_override is not None else OPT_KEYS
    adapt_keys = ADAPT_KEYS

    ob_arr = np.zeros((seq_len, len(ob_keys)), dtype=np.float32)
    tail = ob_records[-seq_len:] if ob_records else []
    start = seq_len - len(tail)
    for i, rec in enumerate(tail):
        ob_arr[start + i] = [float(rec.get(k, 0.0) or 0.0) for k in ob_keys]

    cd_arr = np.zeros((seq_len, len(cd_keys)), dtype=np.float32)
    if candle_df is not None and (not candle_df.empty):
        logged_cd_err = False
        try:
            # Preferred mapping: align each orderbook record by its timestamp.
            # `_ts_epoch` is attached by the pipeline when buffering FH0 (1Hz).
            use_ts = bool(tail) and all(isinstance(r, dict) and r.get("_ts_epoch") is not None for r in tail)
            if use_ts:
                # Normalize candle_df index to minute timestamps.
                cdf = candle_df
                if not isinstance(cdf.index, pd.DatetimeIndex):
                    try:
                        cdf = cdf.copy()
                        cdf.index = pd.to_datetime(cdf.index)
                    except Exception:
                        cdf = candle_df

                if isinstance(cdf.index, pd.DatetimeIndex) and len(cdf.index) >= 1:
                    last_min = cdf.index.max()
                    last_complete_min = last_min
                    try:
                        now_min = datetime.fromtimestamp(float(tail[-1].get("_ts_epoch")) ).replace(second=0, microsecond=0)
                        if last_min.replace(second=0, microsecond=0) == now_min and len(cdf.index) >= 2:
                            last_complete_min = cdf.index[-2]
                    except Exception:
                        last_complete_min = last_min

                    for i, rec in enumerate(tail):
                        try:
                            ts = float(rec.get("_ts_epoch"))
                            minute = datetime.fromtimestamp(ts).replace(second=0, microsecond=0)
                            if minute > last_complete_min:
                                minute = last_complete_min
                            if minute in cdf.index:
                                cd_arr[start + i] = cdf.loc[minute, cd_keys].values.astype(np.float32)
                            else:
                                # best-effort: nearest previous candle
                                try:
                                    pos = int(cdf.index.searchsorted(minute, side="right")) - 1
                                    if pos >= 0:
                                        cd_arr[start + i] = cdf.iloc[pos][cd_keys].values.astype(np.float32)
                                except Exception:
                                    if (not logged_cd_err) and logger.isEnabledFor(logging.DEBUG):
                                        logged_cd_err = True
                                        logger.debug(
                                            "build_sequence: candle nearest mapping failed (i=%s minute=%s)",
                                            int(i),
                                            str(minute),
                                            exc_info=True,
                                        )
                        except Exception:
                            if (not logged_cd_err) and logger.isEnabledFor(logging.DEBUG):
                                logged_cd_err = True
                                logger.debug(
                                    "build_sequence: candle ts mapping failed (i=%s)",
                                    int(i),
                                    exc_info=True,
                                )
                            continue
                else:
                    use_ts = False

            # Fallback: linear mapping (legacy behavior)
            if not use_ts:
                cd_vals = candle_df[cd_keys].values.astype(np.float32)
                bars = int(len(cd_vals))
                if bars > 0:
                    for row in range(int(seq_len)):
                        bar_idx = min(bars - 1, int(row * bars / int(seq_len)))
                        cd_arr[row] = cd_vals[bar_idx]
        except Exception:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("build_sequence: candle feature mapping failed", exc_info=True)

    opt_row = np.zeros((len(opt_keys),), dtype=np.float32)
    if opt_features:
        for j, k in enumerate(opt_keys):
            try:
                v = opt_features.get(k)
                if v is not None:
                    opt_row[j] = float(v)
            except Exception:
                pass

    opt_arr = np.zeros((seq_len, len(opt_keys)), dtype=np.float32)
    try:
        for i, rec in enumerate(tail):
            try:
                rf = (rec or {}).get("_opt_features")
                if isinstance(rf, dict) and rf:
                    row = opt_arr[start + i]
                    row.fill(0.0)
                    for j, k in enumerate(opt_keys):
                        try:
                            v = rf.get(k)
                            if v is not None:
                                row[j] = float(v)
                        except Exception:
                            pass
                    continue
            except Exception:
                rf = None
            opt_arr[start + i] = opt_row
    except Exception:
        opt_arr[:] = np.tile(opt_row, (seq_len, 1))

    time_arr = np.zeros((seq_len, int(FUTURE_KNOWN_DIM)), dtype=np.float32)
    try:
        last_dt = datetime.now()
        for i, rec in enumerate(tail):
            try:
                ts = float((rec or {}).get("_ts_epoch") or 0.0)
                if ts > 0:
                    dt = datetime.fromtimestamp(ts)
                    last_dt = dt
                else:
                    dt = last_dt
                time_arr[start + i] = np.array(build_time_features(dt), dtype=np.float32)
            except Exception:
                pass
    except Exception:
        time_arr = np.zeros((seq_len, int(FUTURE_KNOWN_DIM)), dtype=np.float32)

    adapt_row = np.zeros((len(adapt_keys),), dtype=np.float32)
    if adaptive_features:
        for j, k in enumerate(adapt_keys):
            try:
                v = adaptive_features.get(k)
                if v is not None:
                    adapt_row[j] = float(v)
            except Exception:
                pass

    adapt_arr = np.zeros((seq_len, len(adapt_keys)), dtype=np.float32)
    try:
        for i, rec in enumerate(tail):
            try:
                rf = (rec or {}).get("_adaptive_features")
                if isinstance(rf, dict) and rf:
                    row = adapt_arr[start + i]
                    row.fill(0.0)
                    for j, k in enumerate(adapt_keys):
                        try:
                            v = rf.get(k)
                            if v is not None:
                                row[j] = float(v)
                        except Exception:
                            pass
                    continue
            except Exception:
                rf = None
            adapt_arr[start + i] = adapt_row
    except Exception:
        adapt_arr[:] = np.tile(adapt_row, (seq_len, 1))

    # ── 멀티스케일 블록 조립 (헬퍼) ──────────────────────────────────────────
    # 중복 제거: adaptive 유무와 무관하게 동일 로직으로 멀티스케일 배열 구성.
    def _build_multiscale_arrs() -> list:
        """multiscale_features(새 경로) 또는 ms5_features(레거시)로 배열 목록 반환."""
        arrs: list = []
        if multiscale_features is not None:
            for scale_key, scale_keys in (("ms5", MS5_KEYS), ("ms15", MS15_KEYS)):
                df = multiscale_features.get(scale_key)
                if df is not None and not df.empty:
                    row = np.zeros((len(scale_keys),), dtype=np.float32)
                    last = df.iloc[-1]
                    for j, k in enumerate(scale_keys):
                        try:
                            row[j] = float(last.get(k, 0.0))
                        except Exception:
                            pass
                    arrs.append(np.tile(row, (seq_len, 1)))
        elif ms5_features is not None:
            # 레거시: ms5_features 딕셔너리 직접 전달
            row = np.zeros((len(MS5_KEYS),), dtype=np.float32)
            for j, k in enumerate(MS5_KEYS):
                try:
                    v = ms5_features.get(k)
                    if v is not None:
                        row[j] = float(v)
                except Exception:
                    pass
            arrs.append(np.tile(row, (seq_len, 1)))
        return arrs

    # adaptive 피처가 없는 경로 (no-adaptive 또는 레코드에도 없음)
    if adaptive_features is None:
        has_any = False
        try:
            if isinstance(tail, list) and tail:
                has_any = any(
                    isinstance((r or {}).get("_adaptive_features"), dict)
                    and (r or {}).get("_adaptive_features")
                    for r in tail
                )
        except Exception:
            has_any = False
        if not has_any:
            ms_arrs = _build_multiscale_arrs()
            if ms_arrs:
                all_blocks = [ob_arr, cd_arr, opt_arr] + ms_arrs + [time_arr]
            else:
                all_blocks = [ob_arr, cd_arr, opt_arr, time_arr]
            result = np.concatenate(all_blocks, axis=1)
            if not np.isfinite(result).all():
                logger.warning("[build_sequence] NaN/Inf 검출 — 0으로 치환")
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return result

    # adaptive 포함 경로
    ms_arrs = _build_multiscale_arrs()
    if ms_arrs:
        all_blocks = [ob_arr, cd_arr, opt_arr] + ms_arrs + [adapt_arr, time_arr]
    else:
        all_blocks = [ob_arr, cd_arr, opt_arr, adapt_arr, time_arr]
    result = np.concatenate(all_blocks, axis=1)
    if not np.isfinite(result).all():
        logger.warning("[build_sequence] NaN/Inf 검출 — 0으로 치환")
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result
