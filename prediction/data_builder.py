"""Dataset builder: ticks_replay_*.jsonl -> (X, y) npz.

This script converts replay logs produced by `ebest_live.py --out-ticks` into a
simple supervised dataset for the Transformer model.

Assumptions (best-effort):
- FC0 records provide `tick.price` and `tick.chetime` (HHMMSS)
- FH0 records provide orderbook snapshot keys consumed by `calc_orderbook_features`

The generated dataset is intentionally minimal:
- X is built from replay logs best-effort and aims to match the runtime feature
  composition:
  - orderbook(7) + candle(5) + option(7)
  - plus adaptive_indicator(28) when enabled via config.json

Usage:
  python -m prediction.data_builder \
    --files ticks_replay_20250210.jsonl ticks_replay_20250211.jsonl \
    --out dataset_5m.npz --seq-len 60 --horizon 5
"""

from __future__ import annotations

import argparse
import json
import logging
import gzip
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import load_config, FUTURE_KNOWN_DIM, HORIZON_SEC, TRCode  # NW-QUA-02: TRCode 추가
# [SSOT] config.py 헬퍼 — AdaptiveZigZagSettings 경유 AdaptiveZigZagConfig 생성
try:
    from config import zigzag_settings_from_dict as _zz_settings_from_dict
except ImportError:
    _zz_settings_from_dict = None  # type: ignore[assignment]

from prediction.features import ADAPT_KEYS, CD_KEYS, MS5_KEYS, OB_KEYS, calc_candle_features, calc_multiscale_features, calc_orderbook_features, get_opt_keys
from prediction.features.option_features import build_option_snapshot
from prediction.features.time_features import build_time_features
from data.tick_processor import RealTimeTickProcessor
from core.utils import normalize_adaptive_indicator_symbol, parse_chetime, safe_float

logger = logging.getLogger(__name__)

# [IMP-1-3] 전역 심볼 재정의를 로컬 별칭으로 변경하여 모듈 오염 방지.
# features.py의 원본 리스트를 변경하지 않고 이 파일 내부에서만 사용하는 복사본을 만든다.
_OB_KEYS: list = list(OB_KEYS)
_CD_KEYS: list = list(CD_KEYS)
_ADAPT_KEYS: list = list(ADAPT_KEYS)
# 하위 코드 호환성을 위해 로컬 별칭 유지 (모듈 export는 하지 않음)
OB_KEYS = _OB_KEYS  # noqa: F811 — intentional local shadow
CD_KEYS = _CD_KEYS  # noqa: F811
ADAPT_KEYS = _ADAPT_KEYS  # noqa: F811

TICK_SIZE = 0.05


def _validate_ohlcv_bar(bar: Dict[str, Any]) -> bool:
    try:
        o = float(bar.get("Open") or 0.0)
        h = float(bar.get("High") or 0.0)
        l = float(bar.get("Low") or 0.0)
        c = float(bar.get("Close") or 0.0)
        v = float(bar.get("Volume") or 0.0)
    except Exception:
        return False

    if o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
        return False
    if v < 0.0:
        return False
    if h < l:
        return False
    if h < o or h < c:
        return False
    if l > o or l > c:
        return False
    return True


def _load_jsonl(path: str) -> Iterable[dict]:
    """_load_jsonl.

Args:
    path:
"""
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _restore_compact_prices(tick: Any) -> Any:
    """Restore compact price representation used in disk logs.

    Live saver may store offerho*/bidho* as integer cents (x100). For training
    feature extraction, convert them back to float prices.
    """

    if not isinstance(tick, dict):
        return tick

    out: Dict[str, Any] = {}
    for k, v in tick.items():
        ks = str(k)
        if (ks.startswith("offerho") or ks.startswith("bidho")) and isinstance(v, int):
            # Only convert when it looks like a compact x100 integer.
            # Avoid converting already-normal integer prices (e.g., 340) into 3.4.
            try:
                iv = int(v)
                if 1000 <= abs(iv) <= 100000000:
                    out[ks] = float(iv) / 100.0
                    continue
            except Exception:
                pass
        out[ks] = v
    return out


def _hhmm_to_minutes(hhmm: str) -> int:
    """_hhmm_to_minutes.

Args:
    hhmm:
"""
    return int(hhmm[:2]) * 60 + int(hhmm[2:4])


def _to_dt_minute(chetime: str) -> datetime:
    """Parse chetime/hotime and floor to minute."""

    dt = parse_chetime(chetime)
    return dt.replace(second=0, microsecond=0)


def _extract_ts_epoch(chetime: str) -> int:
    """Convert chetime/hotime to epoch seconds (best-effort)."""

    try:
        dt = parse_chetime(chetime)
        return int(dt.replace(microsecond=0).timestamp())
    except Exception:
        return int(pd.Timestamp.utcnow().timestamp())


def build_dataset(
    files: List[str],
    *,
    seq_len: int = 60,
    horizon_min: int = 5,
    tft: bool = False,
    tft_horizon_sec: int = HORIZON_SEC,
    config_path: str = "config.json",
    min_profit_ticks: float = 1.5,
    multiscale_5m: bool = False,
    now_fn: Optional[Callable[[], datetime]] = None,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y) from jsonl files.

    Returns:
        X: (N, seq_len, feature_dim) float32
        y: (N,) int64 (0=down, 1=up)

        When `tft=True`:
            past_known: (N, seq_len, FUTURE_KNOWN_DIM) float32
            future_known: (N, tft_horizon_sec, FUTURE_KNOWN_DIM) float32
    """

    seq_len = int(seq_len)
    horizon_min = int(horizon_min)
    _now = now_fn if now_fn is not None else datetime.now

    cfg = None
    try:
        cfg = load_config(str(config_path or "config.json"))
    except Exception:
        cfg = None

    adaptive_enabled = False
    adaptive_warmup_bars = 15  # fallback: 최소 지표 윈도우
    adaptive_symbol = "KOSPI 지수"  # dual_mode는 항상 true
    option_feature_set = "v1"
    pcr_atm_strikes_each_side = 5
    if cfg is not None:
        try:
            adaptive_enabled = bool(getattr(cfg, "adaptive_indicator", None) and cfg.adaptive_indicator.enabled)
        except Exception:
            adaptive_enabled = False
        try:
            adaptive_warmup_bars = max(
                15, int(getattr(getattr(cfg, "adaptive_indicator", None), "warmup_bars", 15) or 15)
            )
        except Exception:
            adaptive_warmup_bars = 15
        try:
            adaptive_symbol = str(getattr(getattr(cfg, "adaptive_indicator", None), "kospi_symbol", "KOSPI 지수") or "KOSPI 지수")
        except Exception:
            adaptive_symbol = "KP200 선물"

        try:
            option_feature_set = str(getattr(getattr(cfg, "prediction", None), "option_feature_set", "v1") or "v1")
        except Exception:
            option_feature_set = "v1"
        try:
            pcr_atm_strikes_each_side = max(
                0,
                min(50, int(getattr(getattr(cfg, "prediction", None), "pcr_atm_strikes_each_side", 5) or 5)),
            )
        except Exception:
            pcr_atm_strikes_each_side = 5

    OPT_KEYS = list(get_opt_keys(str(option_feature_set or "v1")))

    # ── 1패스: 1분봉 OHLCV 재구성 (파일 스트리밍, 메모리 절감) ───────────────
    # Reconstruct minute OHLCV from FC0 ticks.
    minute_ohlcv: dict[datetime, dict] = {}  # KP200 선물
    minute_ohlcv_kospi: dict[datetime, dict] = {}  # KOSPI 지수
    
    # 1패스: FC0만 스트리밍하여 1분봉 구성
    def _iter_fc0():
        for fpath in files:
            logger.info("pass1 %s", fpath)
            yield from (r for r in _load_jsonl(str(fpath))
                        if str(r.get("trcode") or "").upper() == TRCode.FUTURES.value)
    
    # KOSPI 지수용 반복자 (symbol 필드로 구분)
    def _iter_kospi():
        for fpath in files:
            logger.info("pass1_kospi %s", fpath)
            yield from (r for r in _load_jsonl(str(fpath))
                        if str(r.get("symbol") or "").strip() == "001")  # KOSPI 지수 코드

    for rec in _iter_fc0():
        tick = rec.get("tick") or {}
        if not isinstance(tick, dict):
            continue

        che = str(tick.get("chetime") or "")
        if len(che) < 6:
            continue
        price = safe_float(tick.get("price"), 0.0)
        if price <= 0.0:
            continue

        minute = _to_dt_minute(che)
        bar = minute_ohlcv.get(minute)
        if bar is None:
            bar = {
                "Open": float(price),
                "High": float(price),
                "Low": float(price),
                "Close": float(price),
                "_cum_volume_max": float(safe_float(tick.get("volume"), 0.0)),
                "_cvol_sum": float(safe_float(tick.get("cvolume"), 0.0)),
            }
            minute_ohlcv[minute] = bar
        else:
            bar["High"] = max(float(bar["High"]), float(price))
            bar["Low"] = min(float(bar["Low"]), float(price))
            bar["Close"] = float(price)
            bar["_cum_volume_max"] = max(float(bar.get("_cum_volume_max") or 0.0), float(safe_float(tick.get("volume"), 0.0)))
            bar["_cvol_sum"] = float(bar.get("_cvol_sum") or 0.0) + float(safe_float(tick.get("cvolume"), 0.0))
    
    # KOSPI 지수 OHLCV 변환
    for rec in _iter_kospi():
        tick = rec.get("tick") or {}
        if not isinstance(tick, dict):
            continue

        che = str(tick.get("chetime") or "")
        if len(che) < 6:
            continue
        price = safe_float(tick.get("price"), 0.0)
        if price <= 0.0:
            continue

        minute = _to_dt_minute(che)
        bar = minute_ohlcv_kospi.get(minute)
        if bar is None:
            bar = {
                "Open": float(price),
                "High": float(price),
                "Low": float(price),
                "Close": float(price),
                "_cum_volume_max": float(safe_float(tick.get("volume") or tick.get("cvolume"), 0.0)),
                "_cvol_sum": float(safe_float(tick.get("cvolume") or tick.get("volume"), 0.0)),
            }
            minute_ohlcv_kospi[minute] = bar
        else:
            bar["High"] = max(float(bar["High"]), float(price))
            bar["Low"] = min(float(bar["Low"]), float(price))
            bar["Close"] = float(price)
            vol = safe_float(tick.get("volume") or tick.get("cvolume"), 0.0)
            bar["_cum_volume_max"] = max(float(bar.get("_cum_volume_max") or 0.0), float(vol))
            bar["_cvol_sum"] = float(bar.get("_cvol_sum") or 0.0) + float(vol)

    candle_df = None
    minute_close: dict[datetime, float] = {}
    if minute_ohlcv:
        keys = sorted(minute_ohlcv.keys())
        rows = []
        prev_cum: float | None = None
        invalid_ohlcv = 0
        for k in keys:
            bar = minute_ohlcv[k]
            cum = float(bar.get("_cum_volume_max") or 0.0)
            cvol_sum = float(bar.get("_cvol_sum") or 0.0)
            if cum > 0.0:
                if prev_cum is None:
                    vol = cum
                else:
                    delta = cum - float(prev_cum)
                    vol = delta if delta >= 0.0 else cum
                prev_cum = cum
            else:
                vol = cvol_sum

            row = {
                "timestamp": k,
                "Open": float(bar["Open"]),
                "High": float(bar["High"]),
                "Low": float(bar["Low"]),
                "Close": float(bar["Close"]),
                "Volume": float(vol),
            }
            if not _validate_ohlcv_bar(row):
                invalid_ohlcv += 1
                continue

            rows.append(row)
            minute_close[k] = float(bar["Close"])

        try:
            if invalid_ohlcv > 0:
                logger.warning("invalid OHLCV bars skipped: %d", int(invalid_ohlcv))
        except Exception:
            pass

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        candle_df = calc_candle_features(df)

    # ── KOSPI 지수 OHLCV DataFrame 생성 ───────────────────────────────────────
    candle_df_kospi = None
    if minute_ohlcv_kospi:
        keys = sorted(minute_ohlcv_kospi.keys())
        rows = []
        prev_cum: float | None = None
        invalid_ohlcv = 0
        for k in keys:
            bar = minute_ohlcv_kospi[k]
            cum = float(bar.get("_cum_volume_max") or 0.0)
            cvol_sum = float(bar.get("_cvol_sum") or 0.0)
            if cum > 0.0:
                if prev_cum is None:
                    vol = cum
                else:
                    delta = cum - float(prev_cum)
                    vol = delta if delta >= 0.0 else cum
                prev_cum = cum
            else:
                vol = cvol_sum

            row = {
                "timestamp": k,
                "Open": float(bar["Open"]),
                "High": float(bar["High"]),
                "Low": float(bar["Low"]),
                "Close": float(bar["Close"]),
                "Volume": float(vol),
            }
            if not _validate_ohlcv_bar(row):
                invalid_ohlcv += 1
                continue

            rows.append(row)

        try:
            if invalid_ohlcv > 0:
                logger.warning("[KOSPI] invalid OHLCV bars skipped: %d", int(invalid_ohlcv))
        except Exception:
            pass

        df_kospi = pd.DataFrame(rows)
        df_kospi["timestamp"] = pd.to_datetime(df_kospi["timestamp"])
        df_kospi = df_kospi.set_index("timestamp")
        candle_df_kospi = calc_candle_features(df_kospi)
        logger.info("[KOSPI] OHLCV 생성 완료: %d rows", len(candle_df_kospi))
    else:
        logger.info("[KOSPI] OHLCV 데이터 없음")

    # ── 5분봉 멀티스케일 피처 df 생성 (multiscale_5m=True 일 때만) ──────────
    multiscale_5m_df: "pd.DataFrame | None" = None
    if multiscale_5m and minute_ohlcv:
        try:
            raw_rows = []
            for t, v in sorted(minute_ohlcv.items()):
                cum = float(v.get("_cum_volume_max") or 0.0)
                cvol = float(v.get("_cvol_sum") or 0.0)
                vol = cum if cum > 0.0 else cvol
                raw_rows.append({
                    "timestamp": t,
                    "Open":   float(v.get("Open",  0.0)),
                    "High":   float(v.get("High",  0.0)),
                    "Low":    float(v.get("Low",   0.0)),
                    "Close":  float(v.get("Close", 0.0)),
                    "Volume": float(vol),
                })
            raw_1m_df = pd.DataFrame(raw_rows).set_index("timestamp")
            raw_1m_df.index = pd.to_datetime(raw_1m_df.index)
            multiscale_5m_df = calc_multiscale_features(raw_1m_df)
            logger.info("multiscale_5m_df: %d rows", len(multiscale_5m_df))
        except Exception:
            logger.debug("multiscale_5m_df 생성 실패", exc_info=True)
            multiscale_5m_df = None

    adaptive_features_per_minute: Dict[datetime, Dict[str, Any]] = {}
    adaptive_context_per_minute: Dict[datetime, str] = {}
    if adaptive_enabled and minute_ohlcv:
        try:
            from indicators import (
                AdaptiveIndicatorManager,
                IndicatorManagerConfig,
                AdaptiveSuperTrendConfig,
            )
        except Exception:
            adaptive_enabled = False
            adaptive_features_per_minute = {}
            adaptive_context_per_minute = {}

        st = {}
        try:
            st = getattr(getattr(cfg, "adaptive_indicator", None), "supertrend", None)
            st = st.__dict__ if hasattr(st, "__dict__") else {}
        except Exception:
            st = {}
        try:
            zz = getattr(getattr(cfg, "adaptive_indicator", None), "zigzag", None)
            zz = zz.__dict__ if hasattr(zz, "__dict__") else {}
        except Exception:
            zz = {}
        try:
            adaptive_dict = getattr(cfg, "adaptive_indicator", None)
            adaptive_dict = adaptive_dict.__dict__ if hasattr(adaptive_dict, "__dict__") else {}
        except Exception:
            adaptive_dict = {}

        try:
            # ── [SSOT] AdaptiveZigZagConfig 는 AdaptiveZigZagSettings.to_zigzag_config() 경유 ──
            if _zz_settings_from_dict is None:
                raise ImportError("zigzag_settings_from_dict를 config.py에서 로드하지 못했습니다.")
            _fn = _zz_settings_from_dict

            _sym_n  = normalize_adaptive_indicator_symbol(adaptive_symbol)
            _prefix = f"[{_sym_n}]" if _sym_n else ""

            st_cfg = AdaptiveSuperTrendConfig(
                atr_min_period=int(st.get("atr_min_period", 7) or 7),
                atr_max_period=int(st.get("atr_max_period", 21) or 21),
                multiplier_min=float(st.get("multiplier_min", 1.5) or 1.5),
                multiplier_max=float(st.get("multiplier_max", 4.0) or 4.0),
                er_period=int(st.get("er_period", 10) or 10),
                adx_period=int(st.get("adx_period", 14) or 14),
                use_bb_correction=bool(st.get("use_bb_correction", True)),
                bb_period=int(st.get("bb_period", 20) or 20),
                bb_std=float(st.get("bb_std", 2.0) or 2.0),
                smooth_period=int(st.get("smooth_period", 3) or 3),
            )

            # ZigZag: dict → AdaptiveZigZagSettings → AdaptiveZigZagConfig (SSOT 경로)
            zz_s         = _fn(zz)
            kospi_zz_s   = _fn(adaptive_dict.get("kospi_zigzag")  or {}, base=zz_s)
            futures_zz_s = _fn(adaptive_dict.get("futures_zigzag") or {}, base=zz_s)

            zz_cfg         = zz_s.to_zigzag_config(
                pivot_lifecycle_log=True, pivot_lifecycle_log_prefix=_prefix)
            kospi_zz_cfg   = kospi_zz_s.to_zigzag_config(
                pivot_lifecycle_log=True, pivot_lifecycle_log_prefix="[KOSPI]")
            futures_zz_cfg = futures_zz_s.to_zigzag_config(
                pivot_lifecycle_log=True, pivot_lifecycle_log_prefix="[KP200]")

            dual_mode      = bool(adaptive_dict.get("dual_mode", False) or False)
            kospi_symbol   = str(adaptive_dict.get("kospi_symbol",   "KOSPI 지수") or "KOSPI 지수")
            futures_symbol = str(adaptive_dict.get("futures_symbol", "KP200 선물") or "KP200 선물")

            # 피봇 근접 알림 설정
            pivot_proximity_alert = adaptive_dict.get("pivot_proximity_alert") or {}
            pivot_proximity_alert_enabled     = bool(pivot_proximity_alert.get("enabled", True) or True)
            pivot_proximity_max_bars_diff     = int(pivot_proximity_alert.get("max_bars_diff", 1) or 1)
            pivot_proximity_telegram_enabled  = bool(pivot_proximity_alert.get("telegram_enabled", True) or True)

            kospi_st_cfg   = AdaptiveSuperTrendConfig(**{k: v for k, v in st_cfg.__dict__.items()})
            futures_st_cfg = AdaptiveSuperTrendConfig(**{k: v for k, v in st_cfg.__dict__.items()})

            mgr = AdaptiveIndicatorManager(
                config=IndicatorManagerConfig(
                    supertrend=st_cfg, 
                    zigzag=zz_cfg,
                    kospi_supertrend=kospi_st_cfg,
                    kospi_zigzag=kospi_zz_cfg,
                    futures_supertrend=futures_st_cfg,
                    futures_zigzag=futures_zz_cfg,
                    symbol=str(adaptive_symbol),
                    kospi_symbol=kospi_symbol,
                    futures_symbol=futures_symbol,
                    dual_mode=dual_mode,
                    pivot_proximity_alert_enabled=pivot_proximity_alert_enabled,
                    pivot_proximity_max_bars_diff=pivot_proximity_max_bars_diff,
                    pivot_proximity_telegram_enabled=False,  # 백테스트에서는 텔레그램 비활성화
                )
            )

            keys = sorted(minute_ohlcv.keys())
            warm_n = max(45, int(adaptive_warmup_bars or 45))

            for idx, k in enumerate(keys):
                bar = minute_ohlcv.get(k) or {}
                try:
                    # [FIX] 시가 anchor를 활성화하되, 충분한 데이터 누적 후 anchor 심도록 수정
                    # 첫 번째 봉 대신 4번째 봉에 시가를 전달하여 anchor pivot 주입
                    _open_arg = float(bar.get("Open")) if idx == 3 and bar.get("Open") is not None else None
                    res = mgr.update(
                        float(bar.get("High")),
                        float(bar.get("Low")),
                        float(bar.get("Close")),
                        open=_open_arg,
                        bar_time=k,
                    )
                except Exception:
                    continue

                # Store only after warmup so early bars don't contain unstable initialization artifacts.
                if idx >= int(warm_n) - 1 and isinstance(res, dict):
                    tf = res.get("transformer")
                    if isinstance(tf, dict):
                        adaptive_per_minute[k] = {kk: float(tf.get(kk) or 0.0) for kk in ADAPT_KEYS}
                    ctx = res.get("llm_context")
                    if ctx:
                        adaptive_context_per_minute[k] = str(ctx)
        except Exception:
            adaptive_per_minute = {}
            adaptive_context_per_minute = {}
        
        # 피봇 통계 로그 출력
        if mgr is not None:
            try:
                kospi_zz = mgr.kospi_zigzag if hasattr(mgr, 'kospi_zigzag') else None
                futures_zz = mgr.futures_zigzag if hasattr(mgr, 'futures_zigzag') else None
                
                if kospi_zz is not None:
                    logger.info(
                        "[KOSPI 피봇 통계] 등록=%d 갱신=%d 취소=%d 확정=%d",
                        kospi_zz._candidate_registered_count,
                        kospi_zz._candidate_updated_count,
                        kospi_zz._candidate_cancelled_count,
                        kospi_zz.state.confirmed_pivot_count if hasattr(kospi_zz.state, 'confirmed_pivot_count') else 0
                    )
                if futures_zz is not None:
                    logger.info(
                        "[KP200 피봇 통계] 등록=%d 갱신=%d 취소=%d 확정=%d",
                        futures_zz._candidate_registered_count,
                        futures_zz._candidate_updated_count,
                        futures_zz._candidate_cancelled_count,
                        futures_zz.state.confirmed_pivot_count if hasattr(futures_zz.state, 'confirmed_pivot_count') else 0
                    )
            except Exception:
                logger.debug("[백테스트] 피봇 통계 출력 실패", exc_info=True)
        
        # 백테스트 시뮬레이션 (선물 OHLCV 사용)
        if mgr is not None and minute_ohlcv:
            try:
                # KOSPI 피봇 사용
                zz = mgr.kospi_zigzag if hasattr(mgr, 'kospi_zigzag') else mgr.zigzag
                if zz is not None:
                    all_swings = list(getattr(zz, "_all_swings", None) or [])
                    # anchor pivot 제외 (초기화용, 실시간 신호 아님) - index==0인 swing는 anchor
                    all_swings = [s for s in all_swings if s.index != 0]
                    if all_swings:
                        logger.info("[백테스트] KOSPI 피봇 기반 선물 매매 시뮬레이션 시작 (진입/청산: 피봇 확정 봉 종가, 지연: confirmation_bars 이상)")
                        
                        total_trades = win_trades = loss_trades = stop_loss_trades = 0
                        total_profit = 0.0
                        trade_details = []  # 개별 거래 내역 저장
                        
                        for i in range(len(all_swings) - 1):
                            cur = all_swings[i]
                            nxt = all_swings[i + 1]
                            ct = str(getattr(cur, "swing_type", "")).upper()
                            nt = str(getattr(nxt, "swing_type", "")).upper()
                            
                            # 피봇 확정 시간 (confirmed_at_idx 사용)
                            # 참고: confirmation_bars만큼 지연이 최소이며, 보통은 그 이상 지연되어야 피봇 확정 가능
                            # 현실적으로 다음 피봇 확정 시점에 청산 가능
                            cur_idx = int(getattr(cur, "confirmed_at_idx", -1) or -1)
                            nxt_idx = int(getattr(nxt, "confirmed_at_idx", -1) or -1)
                            
                            if cur_idx < 0 or nxt_idx < 0:
                                continue
                            
                            # 해당 시점의 선물 가격 찾기
                            keys = sorted(minute_ohlcv.keys())
                            if cur_idx >= len(keys) or nxt_idx >= len(keys):
                                continue
                            
                            cur_time = keys[cur_idx]
                            nxt_time = keys[nxt_idx]
                            
                            cur_bar = minute_ohlcv.get(cur_time) or {}
                            nxt_bar = minute_ohlcv.get(nxt_time) or {}
                            
                            # 진입가/청산가: 피봇 확정 봉 종가 (선물 가격)
                            # 현실적으로 다음 피봇 확정 시점에 청산 가능
                            c_entry = float(cur_bar.get("Close", 0.0) or 0.0)
                            n_exit = float(nxt_bar.get("Close", 0.0) or 0.0)
                            
                            if c_entry <= 0 or n_exit <= 0:
                                continue
                            
                            # 손절 기준: 이전 피봇 확정 봉 종가 (선물 가격)
                            stop_loss_price = None
                            if i > 0:
                                prev = all_swings[i - 1]
                                prev_idx = int(getattr(prev, "confirmed_at_idx", -1) or -1)
                                if prev_idx >= 0 and prev_idx < len(keys):
                                    prev_time = keys[prev_idx]
                                    prev_bar = minute_ohlcv.get(prev_time) or {}
                                    prev_price = float(prev_bar.get("Close", 0.0) or 0.0)
                                    if prev_price > 0:
                                        if "LOW" in ct:
                                            # 저점 매수: 이전 피봇 확정 봉 종가를 손절 기준으로 설정
                                            stop_loss_price = prev_price
                                        elif "HIGH" in ct:
                                            # 고점 매도: 이전 피봇 확정 봉 종가를 손절 기준으로 설정
                                            stop_loss_price = prev_price
                            
                            if "LOW" in ct and "HIGH" in nt:
                                profit = n_exit - c_entry
                                total_trades += 1
                                total_profit += profit
                                
                                if stop_loss_price is not None and c_entry < stop_loss_price:
                                    loss = stop_loss_price - c_entry
                                    total_profit -= loss
                                    stop_loss_trades += 1
                                    logger.info("[백테스트] ⚠️ 매수 %.2f → 손절 %.2f | 손실 %.2fpt", c_entry, stop_loss_price, loss)
                                    trade_details.append({
                                        "type": "BUY",
                                        "entry_price": c_entry,
                                        "exit_price": stop_loss_price,
                                        "profit": -loss,
                                        "result": "stop_loss",
                                        "entry_time": str(cur_time),
                                        "exit_time": str(keys[prev_idx]) if i > 0 else ""
                                    })
                                else:
                                    if profit > 0:
                                        win_trades += 1
                                        logger.info("[백테스트] ✅ 매수 %.2f → 청산 %.2f | 수익 +%.2fpt", c_entry, n_exit, profit)
                                        trade_details.append({
                                            "type": "BUY",
                                            "entry_price": c_entry,
                                            "exit_price": n_exit,
                                            "profit": profit,
                                            "result": "win",
                                            "entry_time": str(cur_time),
                                            "exit_time": str(nxt_time)
                                        })
                                    else:
                                        loss_trades += 1
                                        logger.info("[백테스트] ❌ 매수 %.2f → 청산 %.2f | 손실 %.2fpt", c_entry, n_exit, profit)
                                        trade_details.append({
                                            "type": "BUY",
                                            "entry_price": c_entry,
                                            "exit_price": n_exit,
                                            "profit": profit,
                                            "result": "loss",
                                            "entry_time": str(cur_time),
                                            "exit_time": str(nxt_time)
                                        })
                            
                            elif "HIGH" in ct and "LOW" in nt:
                                profit = c_entry - n_exit
                                total_trades += 1
                                total_profit += profit
                                
                                if stop_loss_price is not None and c_entry > stop_loss_price:
                                    loss = c_entry - stop_loss_price
                                    total_profit -= loss
                                    stop_loss_trades += 1
                                    logger.info("[백테스트] ⚠️ 매도 %.2f → 손절 %.2f | 손실 %.2fpt", c_entry, stop_loss_price, loss)
                                    trade_details.append({
                                        "type": "SELL",
                                        "entry_price": c_entry,
                                        "exit_price": stop_loss_price,
                                        "profit": -loss,
                                        "result": "stop_loss",
                                        "entry_time": str(cur_time),
                                        "exit_time": str(keys[prev_idx]) if i > 0 else ""
                                    })
                                else:
                                    if profit > 0:
                                        win_trades += 1
                                        logger.info("[백테스트] ✅ 매도 %.2f → 청산 %.2f | 수익 +%.2fpt", c_entry, n_exit, profit)
                                        trade_details.append({
                                            "type": "SELL",
                                            "entry_price": c_entry,
                                            "exit_price": n_exit,
                                            "profit": profit,
                                            "result": "win",
                                            "entry_time": str(cur_time),
                                            "exit_time": str(nxt_time)
                                        })
                                    else:
                                        loss_trades += 1
                                        logger.info("[백테스트] ❌ 매도 %.2f → 청산 %.2f | 손실 %.2fpt", c_entry, n_exit, profit)
                                        trade_details.append({
                                            "type": "SELL",
                                            "entry_price": c_entry,
                                            "exit_price": n_exit,
                                            "profit": profit,
                                            "result": "loss",
                                            "entry_time": str(cur_time),
                                            "exit_time": str(nxt_time)
                                        })
                        
                        if total_trades > 0:
                            logger.info("[백테스트] 시뮬레이션 결과 - 총 거래: %d건, 승리: %d건, 패배: %d건, 손절: %d건, 승률: %.1f%%, 총 수익: %+.2fpt (KP200 선물)", 
                                total_trades, win_trades, loss_trades, stop_loss_trades, win_trades/total_trades*100, total_profit)
                        else:
                            logger.info("[백테스트] 시뮬레이션 결과 - 거래 없음")
                        
                        # 백테스트 결과 파일 저장
                        try:
                            import os
                            import json
                            from datetime import datetime
                            
                            # 백테스트 결과 디렉토리 생성
                            backtest_dir = "backtest_results"
                            os.makedirs(backtest_dir, exist_ok=True)
                            
                            # 날짜 추출 (minute_ohlcv의 첫 번째 키에서)
                            if minute_ohlcv:
                                first_date = sorted(minute_ohlcv.keys())[0]
                                date_str = first_date.strftime("%Y-%m-%d")
                            else:
                                date_str = _now().strftime("%Y-%m-%d")
                            
                            # 결과 데이터 구성
                            result = {
                                "date": date_str,
                                "total_trades": total_trades,
                                "win_trades": win_trades,
                                "loss_trades": loss_trades,
                                "stop_loss_trades": stop_loss_trades,
                                "win_rate": win_trades / total_trades * 100 if total_trades > 0 else 0.0,
                                "total_profit": total_profit,
                                "avg_profit": total_profit / total_trades if total_trades > 0 else 0.0,
                                "trades": trade_details
                            }
                            
                            # 파일 저장
                            filepath = os.path.join(backtest_dir, f"{date_str}.json")
                            with open(filepath, "w", encoding="utf-8") as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                            
                            logger.info("[백테스트] 결과 저장 완료: %s", filepath)
                        except Exception as e:
                            logger.error("[백테스트] 결과 저장 실패: %s", e)
            except Exception:
                logger.debug("[백테스트] 시뮬레이션 실패", exc_info=True)

    ob_buf: deque = deque(maxlen=int(seq_len))
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    PK_list: List[np.ndarray] = []
    FK_list: List[np.ndarray] = []

    # Option snapshots from OC0.
    opt_parser = RealTimeTickProcessor(default_futures_minutes=120, default_options_minutes=120)
    try:
        om = {}
        try:
            om = getattr(cfg, "option_minute_ohlcv", None)
            om = om.__dict__ if hasattr(om, "__dict__") else (om if isinstance(om, dict) else {})
        except Exception:
            om = {}
        opt_parser.configure_option_minute_ohlcv(
            enabled=bool(om.get("enabled", False)),
            atm_window=int(om.get("atm_window", 2) or 2),
        )
    except Exception:
        pass
    # Keep references to the processor's stores so OC0/OH0 update the same snapshots.
    calls: Dict[str, Dict[str, Any]] = opt_parser.call_options
    puts: Dict[str, Dict[str, Any]] = opt_parser.put_options
    last_fc0_price = 0.0
    _skipped_rows = 0  # NW-MNT-02: 무음 예외 누적 카운터
    # build_option_snapshot 캐싱: 분 단위로 1회만 계산 (FH0 107k→336번으로 축소)
    _opt_snap_cache: dict = {}
    _opt_snap_cache_minute: str = ""

    # 2패스: 파일 스트리밍 (all_records 대신 제너레이터 재스캔으로 메모리 절감)
    def _iter_records():
        for fpath in files:
            logger.info("pass2 %s", fpath)
            yield from _load_jsonl(str(fpath))

    for rec in _iter_records():
        trcode = str(rec.get("trcode") or "").upper()

        tick = rec.get("tick") or {}
        if not isinstance(tick, dict):
            tick = {}

        # FH0 -> buffer
        if trcode == TRCode.FUTURES_BOOK.value:
            ob = calc_orderbook_features(tick)
            try:
                che = str(tick.get("hotime") or tick.get("chetime") or "")
                if len(che) >= 6:
                    ob.setdefault("_ts_epoch", int(_extract_ts_epoch(che)))
            except Exception:
                pass

            try:
                # OBI delta/EMA features (computed from buffered history).
                sec_key = None
                try:
                    sec_key = int(ob.get("_ts_epoch") or 0)
                except Exception:
                    sec_key = None

                cur_obi = float(ob.get("obi") or 0.0)
                prev_obi = None
                prev_ema = None
                obi_5s_ago = None

                hist = list(ob_buf)
                if hist:
                    try:
                        prev_obi = float(hist[-1].get("obi") or 0.0)
                    except Exception:
                        prev_obi = None
                    try:
                        prev_ema = float(hist[-1].get("obi_ema5") or hist[-1].get("obi") or 0.0)
                    except Exception:
                        prev_ema = None

                    if sec_key is not None and int(sec_key) > 0:
                        tgt_ts = int(sec_key) - 5
                        for r in reversed(hist):
                            try:
                                ts = int((r or {}).get("_ts_epoch") or 0)
                            except Exception:
                                ts = 0
                            if ts <= int(tgt_ts) and ts > 0:
                                try:
                                    obi_5s_ago = float((r or {}).get("obi") or 0.0)
                                except Exception:
                                    obi_5s_ago = None
                                break

                if prev_obi is None:
                    prev_obi = float(cur_obi)
                if obi_5s_ago is None:
                    obi_5s_ago = float(prev_obi)
                if prev_ema is None:
                    prev_ema = float(prev_obi)

                alpha = 2.0 / (5.0 + 1.0)
                ob["obi_delta1"] = float(cur_obi) - float(prev_obi)
                ob["obi_delta5"] = float(cur_obi) - float(obi_5s_ago)
                ob["obi_ema5"] = float(alpha) * float(cur_obi) + (1.0 - float(alpha)) * float(prev_ema)
            except Exception:
                pass

            try:
                cur_px = 0.0
                try:
                    cur_px = float(opt_parser.get_current_price() or 0.0)
                except Exception:
                    cur_px = 0.0
                if cur_px <= 0.0:
                    try:
                        cur_px = float(last_fc0_price or 0.0)
                    except Exception:
                        cur_px = 0.0

                if cur_px > 0.0:
                    # 분 단위 캐싱: 같은 분에 반복 호출 생략
                    _cur_minute_str = str(tick.get("hotime") or tick.get("chetime") or "")[:4]
                    if _cur_minute_str != _opt_snap_cache_minute or not _opt_snap_cache:
                        _opt_snap_cache = build_option_snapshot(
                            calls,
                            puts,
                            float(cur_px),
                            tick_processor=opt_parser,
                            option_feature_set=str(option_feature_set or "v1"),
                            pcr_atm_strikes_each_side=int(pcr_atm_strikes_each_side),
                        )
                        _opt_snap_cache_minute = _cur_minute_str
                    opt_snap = _opt_snap_cache
                    if isinstance(opt_snap, dict) and opt_snap:
                        ob["_opt_features"] = {k: opt_snap.get(k) for k in list(OPT_KEYS)}
            except Exception:
                pass

            try:
                if adaptive_enabled:
                    feats = adaptive_per_minute.get(_to_dt_minute(str(tick.get("hotime") or tick.get("chetime") or "")))
                    if feats is None:
                        pos = bisect_right(sorted(adaptive_per_minute.keys()), _to_dt_minute(str(tick.get("hotime") or tick.get("chetime") or ""))) - 1
                        if pos >= 0:
                            feats = adaptive_per_minute.get(sorted(adaptive_per_minute.keys())[pos])
                    if isinstance(feats, dict) and feats:
                        ob["_adaptive_features"] = {k: float(feats.get(k) or 0.0) for k in ADAPT_KEYS}
            except Exception:
                pass

            try:
                if not bool(ob.get("_invalid")):
                    ob_buf.append(ob)
            except Exception:
                pass
            continue

        # OC0 -> update option trade snapshot (price/volume/OI/IV)
        if trcode == TRCode.OPTIONS.value:
            try:
                opt_parser.process_option_tick({"trcode": TRCode.OPTIONS.value, "symbol": str(rec.get("symbol") or ""), "tick": tick})
            except Exception:
                pass
            continue

        # OH0 -> enrich option quote snapshot (bid/ask/depth/qty)
        if trcode == TRCode.OPTIONS_QUOTE.value:
            try:
                opt_parser.process_option_quote_tick({"trcode": TRCode.OPTIONS_QUOTE.value, "symbol": str(rec.get("symbol") or ""), "tick": tick})
            except Exception:
                pass
            continue

        # FC0 -> sample
        if trcode != TRCode.FUTURES.value:
            continue

        che = str(tick.get("chetime") or "")
        sample_dt = None
        sample_epoch = None
        try:
            if len(che) >= 6:
                sample_dt = parse_chetime(che)
                sample_epoch = int(sample_dt.replace(microsecond=0).timestamp())
        except Exception:
            sample_dt = None
            sample_epoch = None
        try:
            price = float(tick.get("price") or 0.0)
        except Exception:
            price = 0.0

        try:
            last_fc0_price = float(price)
        except Exception:
            pass

        if price <= 0.0 or len(che) < 4:
            continue
        ob_tail_all = list(ob_buf)
        if sample_epoch is not None:
            try:
                ob_tail_all = [r for r in ob_tail_all if float((r or {}).get("_ts_epoch") or 0.0) < float(sample_epoch)]
            except Exception:
                ob_tail_all = list(ob_buf)

        if len(ob_tail_all) < max(1, int(seq_len) // 2):
            continue

        try:
            cur_minute = _to_dt_minute(che)
        except Exception:
            continue

        tgt_minute = cur_minute + timedelta(minutes=int(horizon_min))
        if tgt_minute not in minute_close:
            continue

        try:
            future_price = float(minute_close[tgt_minute])
        except Exception:
            continue

        future_ret = float(future_price) - float(price)
        thr = float(TICK_SIZE) * float(min_profit_ticks)
        if float(abs(future_ret)) <= float(thr):
            continue

        label = 1 if float(future_ret) > 0.0 else 0

        ob_arr = np.zeros((seq_len, len(OB_KEYS)), dtype=np.float32)
        tail = ob_tail_all[-seq_len:]
        start = seq_len - len(tail)
        for i, r in enumerate(tail):
            ob_arr[start + i] = [float(r.get(k, 0.0) or 0.0) for k in OB_KEYS]

        cd_arr = np.zeros((seq_len, len(CD_KEYS)), dtype=np.float32)
        try:
            if candle_df is not None and (not candle_df.empty) and isinstance(candle_df.index, pd.DatetimeIndex):
                # Use only the last *completed* candle at sampling time (avoid look-ahead).
                # Conservative: pick previous minute bar.
                minute = cur_minute
                try:
                    minute = (cur_minute - timedelta(minutes=1))
                except Exception:
                    minute = cur_minute

                if minute in candle_df.index:
                    row = candle_df.loc[minute, CD_KEYS].values.astype(np.float32)
                    cd_arr[:] = row
                else:
                    pos = int(candle_df.index.searchsorted(minute, side="right")) - 1
                    if pos >= 0:
                        cd_arr[:] = candle_df.iloc[pos][CD_KEYS].values.astype(np.float32)
        except Exception:
            pass

        time_arr = np.zeros((seq_len, int(FUTURE_KNOWN_DIM)), dtype=np.float32)
        try:
            last_dt = cur_minute
            for i, r in enumerate(tail):
                try:
                    ts = float((r or {}).get("_ts_epoch") or 0.0)
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

        opt_arr = np.zeros((seq_len, len(OPT_KEYS)), dtype=np.float32)
        try:
            for i, r in enumerate(tail):
                rf = (r or {}).get("_opt_features")
                if isinstance(rf, dict) and rf:
                    opt_arr[start + i] = [float(rf.get(k) or 0.0) for k in OPT_KEYS]
        except Exception:
            pass

        adapt_arr = np.zeros((seq_len, len(ADAPT_KEYS)), dtype=np.float32)
        try:
            for i, r in enumerate(tail):
                rf = (r or {}).get("_adaptive_features")
                if isinstance(rf, dict) and rf:
                    adapt_arr[start + i] = [float(rf.get(k) or 0.0) for k in ADAPT_KEYS]
        except Exception:
            pass

        # ── 5분봉 멀티스케일 피처 배열 ─────────────────────────────────────
        ms5_arr = np.zeros((seq_len, len(MS5_KEYS)), dtype=np.float32)
        if multiscale_5m and multiscale_5m_df is not None and not multiscale_5m_df.empty:
            try:
                # 현재 분봉을 5분봉 기간으로 매핑 (해당 5분 구간의 마지막 완성봉 사용)
                min5 = cur_minute.replace(
                    minute=(cur_minute.minute // 5) * 5, second=0, microsecond=0
                )
                pos5 = int(multiscale_5m_df.index.searchsorted(min5, side="right")) - 1
                if pos5 >= 0:
                    row5 = multiscale_5m_df.iloc[pos5][MS5_KEYS].values.astype(np.float32)
                    ms5_arr[:] = row5  # 시퀀스 전체에 동일 값 broadcast
            except Exception:
                pass

        # ── 최종 피처 벡터 조합 ──────────────────────────────────────────────
        # 순서: OB | CD | OPT | [MS5] | [ADAPT] | TIME
        parts = [ob_arr, cd_arr, opt_arr]
        if multiscale_5m:
            parts.append(ms5_arr)
        if adaptive_enabled:
            parts.append(adapt_arr)
        parts.append(time_arr)
        x = np.concatenate(parts, axis=1)
        X_list.append(x)
        y_list.append(int(label))

        if bool(tft):
            try:
                pk_arr = np.zeros((int(seq_len), FUTURE_KNOWN_DIM), dtype=np.float32)
                tail_pk = list(ob_buf)[-int(seq_len) :]
                start_pk = int(seq_len) - int(len(tail_pk))
                for i, r in enumerate(tail_pk):
                    try:
                        ts = float((r or {}).get("_ts_epoch") or 0.0)
                        dt = datetime.fromtimestamp(ts) if ts > 0 else cur_minute
                        pk_arr[start_pk + i] = np.array(build_time_features(dt), dtype=np.float32)
                    except Exception:
                        pass
                PK_list.append(pk_arr)
            except Exception:
                PK_list.append(np.zeros((int(seq_len), FUTURE_KNOWN_DIM), dtype=np.float32))

            try:
                fk_arr = np.zeros((int(tft_horizon_sec), FUTURE_KNOWN_DIM), dtype=np.float32)
                base_dt = cur_minute
                for step in range(int(tft_horizon_sec)):
                    try:
                        fdt = base_dt + timedelta(seconds=int(step))
                        fk_arr[int(step)] = np.array(build_time_features(fdt), dtype=np.float32)
                    except Exception:
                        pass
                FK_list.append(fk_arr)
            except Exception:
                FK_list.append(np.zeros((int(tft_horizon_sec), FUTURE_KNOWN_DIM), dtype=np.float32))

    if not X_list:
        if _skipped_rows > 0:
            logger.warning("[DataBuilder] %d개 레코드를 건너뜀 (예외 발생). 로그 확인 권장.", _skipped_rows)
        raise ValueError("no samples created; check FH0/FC0 presence in replay logs")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    logger.info("done: X=%s y=%s pos=%.1f%%", X.shape, y.shape, float(y.mean()) * 100.0)
    
    # ── KOSPI 지수 OHLCV 파일 저장 ───────────────────────────────────────────────
    if candle_df_kospi is not None:
        try:
            kospi_output_dir = Path("data/kospi_ohlcv")
            kospi_output_dir.mkdir(parents=True, exist_ok=True)
            kospi_output_file = kospi_output_dir / f"kospi_1m_{_now().strftime('%Y%m%d')}.csv"
            candle_df_kospi.to_csv(kospi_output_file)
            logger.info("[KOSPI] OHLCV 저장 완료: %s (%d rows)", kospi_output_file, len(candle_df_kospi))
        except Exception as e:
            logger.error("[KOSPI] OHLCV 저장 실패: %s", e)
    
    if bool(tft):
        PK = np.stack(PK_list).astype(np.float32) if PK_list else np.zeros((int(X.shape[0]), int(seq_len), FUTURE_KNOWN_DIM), dtype=np.float32)
        FK = np.stack(FK_list).astype(np.float32) if FK_list else np.zeros((int(X.shape[0]), int(tft_horizon_sec), FUTURE_KNOWN_DIM), dtype=np.float32)
        return X, y, PK, FK
    return X, y


def main(now_fn: Optional[Callable[[], datetime]] = None) -> None:
    """main.

    Args:
        now_fn: 시간 함수 (테스트/백테스트용 주입 가능)
    """
    _now = now_fn if now_fn is not None else datetime.now
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--out", default="dataset_5m.npz")
    parser.add_argument("--config", default="config.json", help="config.json path (controls whether adaptive features are included)")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--min-profit-ticks", type=float, default=1.5)
    parser.add_argument("--tft", action="store_true", help="TFT 학습용 past_known/future_known도 함께 저장")
    parser.add_argument("--tft-horizon-sec", type=int, default=HORIZON_SEC, help="future_known horizon(seconds)")
    parser.add_argument("--multiscale-5m", action="store_true",
                        help="5분봉 멀티스케일 피처(MS5_KEYS 8개) 추가. PAST_UNKNOWN_DIM +8.")
    args = parser.parse_args()

    cfg = None
    try:
        cfg = load_config(str(getattr(args, "config", "config.json") or "config.json"))
    except Exception:
        cfg = None

    adaptive_enabled = False
    option_feature_set = "v1"
    if cfg is not None:
        try:
            adaptive_enabled = bool(getattr(cfg, "adaptive_indicator", None) and cfg.adaptive_indicator.enabled)
        except Exception:
            adaptive_enabled = False
        try:
            option_feature_set = str(getattr(getattr(cfg, "prediction", None), "option_feature_set", "v1") or "v1")
        except Exception:
            option_feature_set = "v1"

    _ms5 = bool(getattr(args, "multiscale_5m", False))

    if bool(args.tft):
        X, y, PK, FK = build_dataset(
            args.files,
            seq_len=int(args.seq_len),
            horizon_min=int(args.horizon),
            tft=True,
            tft_horizon_sec=int(args.tft_horizon_sec),
            config_path=str(args.config or "config.json"),
            min_profit_ticks=float(getattr(args, "min_profit_ticks", 1.5) or 1.5),
            multiscale_5m=_ms5,
            now_fn=now_fn,
        )
        try:
            metadata = {
                "schema_version": f"ob{len(OB_KEYS)}_cd{len(CD_KEYS)}_opt{len(get_opt_keys(str(option_feature_set or 'v1')))}_ms5{len(MS5_KEYS) if _ms5 else 0}_adapt{len(ADAPT_KEYS) if bool(adaptive_enabled) else 0}_time{int(FUTURE_KNOWN_DIM)}",
                "feature_dim": int(X.shape[-1]),
                "seq_len": int(args.seq_len),
                "horizon_min": int(args.horizon),
                "min_profit_ticks": float(getattr(args, "min_profit_ticks", 1.5) or 1.5),
                "tft": True,
                "tft_horizon_sec": int(args.tft_horizon_sec),
                "option_feature_set": str(option_feature_set or "v1"),
                "adaptive_enabled": bool(adaptive_enabled),
                "multiscale_5m": _ms5,
                "ob_keys": list(OB_KEYS),
                "cd_keys": list(CD_KEYS),
                "opt_keys": list(get_opt_keys(str(option_feature_set or "v1"))),
                "ms5_keys": list(MS5_KEYS) if _ms5 else [],
                "adapt_keys": list(ADAPT_KEYS) if bool(adaptive_enabled) else [],
                "created_at": _now().isoformat(),
            }
            meta_s = json.dumps(metadata, ensure_ascii=False)
        except Exception:
            meta_s = ""
        np.savez(str(args.out), X=X, y=y, past_known=PK, future_known=FK, metadata=meta_s)
        print(f"saved: {args.out} X={X.shape} y={y.shape} past_known={PK.shape} future_known={FK.shape}")
    else:
        X, y = build_dataset(
            args.files,
            seq_len=int(args.seq_len),
            horizon_min=int(args.horizon),
            config_path=str(args.config or "config.json"),
            min_profit_ticks=float(getattr(args, "min_profit_ticks", 1.5) or 1.5),
            multiscale_5m=_ms5,
            now_fn=now_fn,
        )
        try:
            metadata = {
                "schema_version": f"ob{len(OB_KEYS)}_cd{len(CD_KEYS)}_opt{len(get_opt_keys(str(option_feature_set or 'v1')))}_ms5{len(MS5_KEYS) if _ms5 else 0}_adapt{len(ADAPT_KEYS) if bool(adaptive_enabled) else 0}_time{int(FUTURE_KNOWN_DIM)}",
                "feature_dim": int(X.shape[-1]),
                "seq_len": int(args.seq_len),
                "horizon_min": int(args.horizon),
                "min_profit_ticks": float(getattr(args, "min_profit_ticks", 1.5) or 1.5),
                "tft": False,
                "option_feature_set": str(option_feature_set or "v1"),
                "adaptive_enabled": bool(adaptive_enabled),
                "multiscale_5m": _ms5,
                "ob_keys": list(OB_KEYS),
                "cd_keys": list(CD_KEYS),
                "opt_keys": list(get_opt_keys(str(option_feature_set or "v1"))),
                "ms5_keys": list(MS5_KEYS) if _ms5 else [],
                "adapt_keys": list(ADAPT_KEYS) if bool(adaptive_enabled) else [],
                "created_at": _now().isoformat(),
            }
            meta_s = json.dumps(metadata, ensure_ascii=False)
        except Exception:
            meta_s = ""
        np.savez(str(args.out), X=X, y=y, metadata=meta_s)
        print(f"saved: {args.out} X={X.shape} y={y.shape}")


if __name__ == "__main__":
    main()
