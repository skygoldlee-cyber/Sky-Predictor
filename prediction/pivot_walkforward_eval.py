"""Walk-forward Real-mode Evaluator — 분봉 로드 + HybridAdaptivePivot 재검출

WalkForwardEvaluator(real 모드)에 주입하는 evaluator_fn 구현.

핵심
----
주어진 (date, params) 에 대해:
  1) 해당 일자의 1분봉 CSV 를 로드하고,
  2) params 로 구성한 HybridAdaptivePivot 을 그 분봉에 '재검출' 실행하여,
  3) 검출 기하 기반 5개 성능 지표를 산출한다.

반환 지표(키)는 PivotParameterDB._calc_composite_score 입력과 동일:
    pivot_confirmation_rate, avg_lag_bars, pivot_quality_score,
    alternation_rate, false_pivot_rate

이 지표들은 '거래 손익'이 아니라 '피봇 검출 품질'을 측정한다. 따라서
WalkForwardEvaluator 의 세 파라미터 소스(DB추천/폴백/고정)를 동일 기준으로
비교하므로 surrogate(_estimate_composite)의 자기충족 문제가 사라진다.

주의(척도)
----------
기존 DB 의 composite_score 가 '거래 백테스트' 기반으로 적재되었다면, 본
evaluator 의 real 점수(검출 기하 기반)와 절대값 척도는 다를 수 있다.
워크포워드의 improvement(score_db - score_fallback)는 세 소스를 같은
evaluator 로 평가하므로 내부 정합성은 보장된다. DB 의
actual_composite_score 는 '참고용'으로만 보라.

사용 예
-------
    from prediction.pivot_parameter_db import PivotParameterDB, WalkForwardEvaluator
    from prediction.pivot_walkforward_eval import HapRedetectionEvaluator

    db = PivotParameterDB("data/pivot_parameters.db")
    evaluator = HapRedetectionEvaluator(
        bars_dir="data/minute_bars",
        symbol="KP200 선물",          # → 파일 접두사 kp200
    )
    wf = WalkForwardEvaluator(db, symbol="KP200 선물", evaluator_fn=evaluator)
    result = wf.run(lookback_days=30, test_days=20)   # mode="real"
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:  # 패키지/평면 임포트 모두 지원
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

try:
    from indicators.hybrid_adaptive_pivot import (
        HybridAdaptivePivot, HybridAdaptivePivotConfig, PivotType,
    )
except ImportError:  # 평면 sys.path 환경 폴백
    from hybrid_adaptive_pivot import (  # type: ignore
        HybridAdaptivePivot, HybridAdaptivePivotConfig, PivotType,
    )

_logger = logging.getLogger(__name__)


# 심볼 → 분봉 파일 접두사 매핑 (필요 시 확장)
DEFAULT_SYMBOL_PREFIX = {
    "KP200 선물": "kp200",
    "KP200": "kp200",
    "KOSPI200": "kp200",
    "KOSPI": "kospi",
}


class HapRedetectionEvaluator:
    """분봉 재검출 기반 evaluator_fn (callable).

    Parameters
    ----------
    bars_dir:
        1분봉 CSV 디렉토리. 파일명 규칙: ``{prefix}_{YYYYMMDD}.csv``.
        CSV 컬럼: timestamp, Open, High, Low, Close, Volume.
    symbol:
        심볼명. symbol_prefix_map 으로 파일 접두사를 결정.
    symbol_prefix_map:
        심볼 → 파일 접두사 매핑. 기본 DEFAULT_SYMBOL_PREFIX.
    warmup_bars:
        HAP 워밍업 봉수. ATR/ER 안정화 전 신호 미출력.
    lookforward:
        방향 정확도(품질) 평가 시 전방 관찰 봉수.
    min_bars:
        최소 봉수. 미만이면 None 반환(측정 불가).
    min_confirmed:
        최소 확정 피봇 수. 미만이면 None 반환(지표 불안정).
    session_start / session_end:
        "HH:MM" 형식. 지정 시 해당 구간 봉만 사용(장 시작 전 노이즈 제거 등).
        None 이면 전체 사용.
    use_adaptive_engine:
        기본 False. 워크포워드는 '고정 파라미터 A/B/C' 비교가 목적이므로
        레짐 기반 atr_weight 자동조정을 끈다(켜면 비교가 흐려짐).
    """

    def __init__(
        self,
        bars_dir: str = "data/minute_bars",
        symbol: str = "KP200 선물",
        symbol_prefix_map: Optional[Dict[str, str]] = None,
        warmup_bars: int = 20,
        lookforward: int = 10,
        min_bars: int = 60,
        min_confirmed: int = 2,
        session_start: Optional[str] = None,
        session_end: Optional[str] = None,
        use_adaptive_engine: bool = False,
    ) -> None:
        if pd is None:
            raise ImportError("HapRedetectionEvaluator 는 pandas 가 필요합니다.")
        self.bars_dir = bars_dir
        self.symbol = symbol
        self._prefix_map = dict(symbol_prefix_map or DEFAULT_SYMBOL_PREFIX)
        self.warmup_bars = int(warmup_bars)
        self.lookforward = int(lookforward)
        self.min_bars = int(min_bars)
        self.min_confirmed = int(min_confirmed)
        self.session_start = session_start
        self.session_end = session_end
        self.use_adaptive_engine = bool(use_adaptive_engine)

    # ── evaluator_fn 진입점 ──────────────────────────────────────────────────

    def __call__(self, date: str, params: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """(date, params) → 검출 기하 기반 5개 지표 dict (측정 불가 시 None)."""
        df = self.load_bars(date)
        if df is None or len(df) < self.min_bars:
            return None

        highs = df["High"].to_numpy(dtype=float)
        lows = df["Low"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)
        times = df["__hhmm__"].tolist()

        cfg = self._params_to_config(params)
        hap = HybridAdaptivePivot(cfg)
        try:
            hap.set_symbol(self.symbol)
        except Exception:
            pass

        # 후보 회계 훅:
        #   _register_candidate → 확인창에 진입한 후보 수 (reg)
        #   _confirm_pivot      → 등록 후보가 '확정'된 수 (conf_reg)
        # 초기 방향 피봇(_init_direction)은 _add_pivot 직접 호출이라 두 카운터에
        # 모두 잡히지 않으므로 비율 분모를 오염시키지 않는다.
        ledger = {"reg": 0, "conf_reg": 0}
        _orig_register = hap._register_candidate
        _orig_confirm = hap._confirm_pivot

        def _counting_register(pt, price, idx, _o=_orig_register, _l=ledger):
            _l["reg"] += 1
            return _o(pt, price, idx)

        def _counting_confirm(pt, price, idx, atr, close, _o=_orig_confirm, _l=ledger):
            _l["conf_reg"] += 1
            return _o(pt, price, idx, atr, close)

        hap._register_candidate = _counting_register  # type: ignore[assignment]
        hap._confirm_pivot = _counting_confirm        # type: ignore[assignment]

        # 봉 단위 재검출
        confirmed: List[Dict[str, Any]] = []
        n = len(df)
        for i in range(n):
            state = hap.update(highs[i], lows[i], closes[i], bar_time=times[i])
            if state.new_pivot_signal in ("new_high", "new_low"):
                pivots = hap.confirmed_pivots
                if not pivots:
                    continue
                p = pivots[-1]
                confirmed.append({
                    "extreme_idx": int(p.index),
                    "confirm_bar": i,
                    "is_high": (p.pivot_type == PivotType.HIGH),
                    "price": float(p.price),
                })

        n_conf_all = len(confirmed)                       # 신호 기준 전체 확정(초기 피봇 포함)
        n_reg = ledger["reg"]                             # 등록(확인창 진입) 후보 수
        n_conf_reg = ledger["conf_reg"]                   # 등록 후보 중 확정 수
        pending_open = 1 if hap._pending_confirm is not None else 0
        n_cancel = max(0, n_reg - n_conf_reg - pending_open)

        if n_reg == 0 or n_conf_all < self.min_confirmed:
            return None  # 비교 불가 (지표 불안정)

        # ① 확정률 / 오탐률  (등록 후보 기준 → 항상 [0,1])
        confirmation_rate = n_conf_reg / n_reg
        false_pivot_rate = n_cancel / n_reg

        # ② 지연 (확정봉 - 극단봉)
        lags = [
            c["confirm_bar"] - c["extreme_idx"]
            for c in confirmed
            if c["confirm_bar"] >= c["extreme_idx"]
        ]
        avg_lag = float(sum(lags) / len(lags)) if lags else 0.0

        # ③ 교번율 (연속 확정 피봇이 H/L 로 교번하는 비율)
        alt = 0
        tot = 0
        for a, b in zip(confirmed, confirmed[1:]):
            tot += 1
            if a["is_high"] != b["is_high"]:
                alt += 1
        alternation_rate = (alt / tot) if tot > 0 else 0.0

        # ④ 품질 = 방향 정확도 (피봇 이후 반대방향 이동 여부)
        pivot_quality_score = self._direction_accuracy(confirmed, highs, lows, n)

        return {
            "pivot_confirmation_rate": float(np.clip(confirmation_rate, 0.0, 1.0)),
            "avg_lag_bars": float(avg_lag),
            "pivot_quality_score": float(np.clip(pivot_quality_score, 0.0, 1.0)),
            "alternation_rate": float(np.clip(alternation_rate, 0.0, 1.0)),
            "false_pivot_rate": float(np.clip(false_pivot_rate, 0.0, 1.0)),
            # 진단용 (composite 에는 미반영)
            "_registered": float(n_reg),
            "_confirmed_all": float(n_conf_all),
            "_confirmed_registered": float(n_conf_reg),
            "_cancelled": float(n_cancel),
            "_pending_open": float(pending_open),
            "_bars": float(n),
        }

    # ── 분봉 로드 ────────────────────────────────────────────────────────────

    def load_bars(self, date: str):
        """date(YYYY-MM-DD) 의 1분봉 CSV 로드 → 정규화 DataFrame (없으면 None)."""
        yyyymmdd = str(date).replace("-", "").strip()
        prefix = self._prefix_map.get(self.symbol)
        if prefix is None:
            # 부분 매칭 폴백
            for k, v in self._prefix_map.items():
                if k in self.symbol or self.symbol in k:
                    prefix = v
                    break
        if prefix is None:
            _logger.warning("[HapEval] 심볼 접두사 미정: %s", self.symbol)
            return None

        path = os.path.join(self.bars_dir, f"{prefix}_{yyyymmdd}.csv")
        if not os.path.exists(path):
            _logger.debug("[HapEval] 분봉 파일 없음: %s", path)
            return None

        try:
            df = pd.read_csv(path)
        except Exception as e:
            _logger.warning("[HapEval] CSV 로드 실패(%s): %s", path, e)
            return None

        # 컬럼 정규화 (대소문자 무시 매핑)
        lower = {c.lower(): c for c in df.columns}
        need = ["high", "low", "close"]
        if not all(k in lower for k in need):
            _logger.warning("[HapEval] 필수 컬럼 누락: %s (cols=%s)", path, list(df.columns))
            return None
        df = df.rename(columns={
            lower["high"]: "High", lower["low"]: "Low", lower["close"]: "Close",
        })

        # 시각 컬럼 → "HH:MM"
        ts_col = lower.get("timestamp") or lower.get("datetime") or lower.get("time")
        if ts_col is not None:
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            df["__hhmm__"] = ts.dt.strftime("%H:%M").fillna("")
        else:
            df["__hhmm__"] = ""

        # 세션 필터 (옵션)
        if self.session_start and self.session_end and ts_col is not None:
            hhmm = df["__hhmm__"]
            mask = (hhmm >= self.session_start) & (hhmm <= self.session_end)
            df = df[mask].reset_index(drop=True)

        # 유효성: High/Low/Close 결측 제거
        df = df.dropna(subset=["High", "Low", "Close"]).reset_index(drop=True)
        return df

    # ── 파라미터 → HAP 설정 ──────────────────────────────────────────────────

    def _params_to_config(self, params: Dict[str, Any]) -> HybridAdaptivePivotConfig:
        """파라미터 dict → HybridAdaptivePivotConfig.

        recommend()/REGIME_FALLBACK/FIXED_BASELINE 가 내보내는 키를 흡수한다.
        atr_multiplier 는 ATR 배수(base_multiplier)로 매핑.
        """
        def _f(key, default):
            v = params.get(key, default)
            try:
                return float(v) if v is not None else float(default)
            except (TypeError, ValueError):
                return float(default)

        base_mult = params.get("base_multiplier")
        if base_mult is None:
            base_mult = params.get("atr_multiplier", 2.0)
        try:
            base_mult = float(base_mult) if base_mult is not None else 2.0
        except (TypeError, ValueError):
            base_mult = 2.0

        try:
            conf_bars = int(params.get("confirmation_bars", 1) or 1)
        except (TypeError, ValueError):
            conf_bars = 1
        try:
            er_period = int(params.get("er_period", 10) or 10)
        except (TypeError, ValueError):
            er_period = 10

        return HybridAdaptivePivotConfig(
            base_pct=_f("base_pct", 0.3),
            base_multiplier=base_mult,
            atr_weight=_f("atr_weight", 0.5),
            er_period=er_period,
            confirmation_bars=conf_bars,
            min_wave_pct=_f("min_wave_pct", 0.15),
            warmup_bars=self.warmup_bars,
            # 고정 파라미터 비교가 목적 → 레짐 자동조정/프랙탈 비활성
            use_adaptive_engine=self.use_adaptive_engine,
            use_fractal_confirmation=False,
        )

    # ── 품질: 방향 정확도 ────────────────────────────────────────────────────

    def _direction_accuracy(
        self,
        confirmed: List[Dict[str, Any]],
        highs: np.ndarray,
        lows: np.ndarray,
        n: int,
    ) -> float:
        """피봇 극단봉 이후 lookforward 봉 내 '반대 방향' 이동 비율.

        HIGH 피봇이면 이후 저가가 피봇가 미만으로 내려갔는지,
        LOW 피봇이면 이후 고가가 피봇가 초과로 올라갔는지 확인한다.
        """
        K = self.lookforward
        correct = 0
        total = 0
        for c in confirmed:
            idx = c["extreme_idx"]
            start = idx + 1
            end = min(idx + K, n - 1)
            if start > end:
                continue  # 전방 봉 부족(주로 마지막 피봇)
            if c["is_high"]:
                seg = lows[start:end + 1]
                matched = bool(np.any(seg < c["price"])) if seg.size else False
            else:
                seg = highs[start:end + 1]
                matched = bool(np.any(seg > c["price"])) if seg.size else False
            total += 1
            if matched:
                correct += 1
        return (correct / total) if total > 0 else 0.0


def make_hap_evaluator(**kwargs) -> HapRedetectionEvaluator:
    """HapRedetectionEvaluator 생성 헬퍼."""
    return HapRedetectionEvaluator(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 데모 / 스모크 실행
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="HAP 재검출 evaluator 스모크")
    ap.add_argument("--bars-dir", default="data/minute_bars")
    ap.add_argument("--symbol", default="KP200 선물")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    ev = HapRedetectionEvaluator(bars_dir=args.bars_dir, symbol=args.symbol)
    # 두 가지 파라미터 소스를 같은 날에 평가해 차이를 확인
    p_loose = {"atr_multiplier": 1.2, "base_pct": 0.20, "atr_weight": 0.35, "confirmation_bars": 1}
    p_tight = {"atr_multiplier": 2.5, "base_pct": 0.50, "atr_weight": 0.90, "confirmation_bars": 3}
    print("loose:", json.dumps(ev(args.date, p_loose), ensure_ascii=False))
    print("tight:", json.dumps(ev(args.date, p_tight), ensure_ascii=False))
