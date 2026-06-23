# -*- coding: utf-8 -*-
"""
daily_screen.py
===============

피봇-숏 오버레이를 분봉 수집 *전에* 일봉만으로 싸게 거르는 스크리닝 도구.

장중 피봇반전 자체는 분봉이 있어야 검증되지만, 그 *전 단계*인 다음 세 가지는
일봉만으로 실제 약세장(2018·2020·2022·2024-25)에서 평가할 수 있다:

  A. 트리거별 발동 빈도        — 다운사이드 트리거가 약세장에서 충분히/깔끔히 켜지나
  B. 지속형 vs V자 분류        — 트리거가 '지속 하락'과 'V자 딥'을 가르나 (6월 실패 원인)
  C. 직진 숏 PnL/Sharpe        — 트리거 날 시가숏→종가청산이 플랫(0)을 이기나
                                 (못 이기면 장중 피봇으로 정교화해도 가망 없음 → 분봉수집 무의미)
  D. 롱-또는-플랫 지연손실      — 느린 MA20/60 게이트가 약세장에서 얼마나 늦게 꺼지고
                                 그 전에 롱으로 얼마를 까먹나

모든 트리거는 인과적(전일까지 정보, shift(1))이다.
B의 '지속/V자' 라벨만은 진단용으로 *전향(forward) 수익률*을 쓴다(매매신호 아님).

입력
----
일봉 OHLCV. CSV(여러 개 가능) 또는 DuckDB 테이블. 컬럼명은 대소문자/한글 자동 매핑:
  date|일자|timestamp, open|시가, high|고가, low|저가, close|종가, volume|거래량
지표(MA60/ADX) 워밍업 때문에 각 약세장 구간보다 최소 3개월 앞부분을 포함해 주세요.

실행
----
  python daily_screen.py --csv kospi200_daily.csv
  python daily_screen.py --csv 2018.csv 2020.csv 2022.csv 2024_25.csv
  python daily_screen.py --duckdb data/duckdb/market_data.duckdb --table futures_1min --resample
  python daily_screen.py --demo            # 합성 데이터로 출력 형식만 확인

주의: KOSPI200 선물 승수는 2017-03-27에 500,000 → 250,000 으로 바뀌었다.
      본 스크립트 기본값은 250,000 이므로 2017년 이후 구간에만 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# 비용 모델 (분봉 검증과 동일하게 유지)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Cost:
    multiplier: float = 250_000.0
    commission_pct_per_side: float = 0.00015
    slippage_ticks_per_side: float = 1.0
    tick_size: float = 0.05
    annualization: float = 252.0

    def round_trip_pts(self, entry_px: float, exit_px: float) -> float:
        comm = self.commission_pct_per_side * (entry_px + exit_px)
        slip = 2.0 * self.slippage_ticks_per_side * self.tick_size
        return comm + slip


# 기본 약세장 구간 (워밍업 위해 시작 전 여유 포함 권장)
DEFAULT_EPISODES: List[Tuple[str, str, str]] = [
    ("2018_무역분쟁", "2018-01-01", "2019-01-31"),
    ("2020_코로나",   "2020-01-01", "2020-06-30"),
    ("2022_금리인상", "2021-10-01", "2023-03-31"),
    ("2024_25_관세",  "2024-07-01", "2025-06-30"),
]


# ════════════════════════════════════════════════════════════════════════════
# 지표 (인과적)
# ════════════════════════════════════════════════════════════════════════════
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = df["CLOSE"].shift(1)
    tr = pd.concat([df["HIGH"] - df["LOW"],
                    (df["HIGH"] - pc).abs(),
                    (df["LOW"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["HIGH"], df["LOW"], df["CLOSE"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / a
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MA20"] = out["CLOSE"].rolling(20).mean()
    out["MA60"] = out["CLOSE"].rolling(60).mean()
    out["RET"] = out["CLOSE"].pct_change()
    out["ADX"] = adx(out, 14)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 트리거 (전부 인과적: shift(1) — 전일 조건으로 당일 매매 결정)
# ════════════════════════════════════════════════════════════════════════════
def build_triggers(d: pd.DataFrame) -> Dict[str, pd.Series]:
    raw = {
        "느린레짐(MA20<MA60&ADX>25)": (d["MA20"] < d["MA60"]) & (d["ADX"] > 25),
        "MA20_down(종가<MA20&하락)":   (d["CLOSE"] < d["MA20"]) & (d["RET"] < 0),
        "전일 ret<-1.5%":              d["RET"] < -0.015,
        "전일 ret<-2.5%":              d["RET"] < -0.025,
        "2일연속하락":                  (d["RET"] < 0) & (d["RET"].shift(1) < 0),
        "3일연속하락":                  (d["RET"] < 0) & (d["RET"].shift(1) < 0) & (d["RET"].shift(2) < 0),
    }
    return {k: v.shift(1).fillna(False).astype(bool) for k, v in raw.items()}


def long_or_flat_signal(d: pd.DataFrame, adx_thr: float = 25.0) -> pd.Series:
    """롱-또는-플랫 일봉 신호: 상승추세(MA20>MA60)+ADX>thr → 롱(1), 그 외 플랫(0). shift(1)."""
    reg = pd.Series(0, index=d.index, dtype=int)
    reg[(d["MA20"] > d["MA60"]) & (d["ADX"] > adx_thr)] = 1
    return reg.shift(1).fillna(0).astype(int)


# ════════════════════════════════════════════════════════════════════════════
# 메트릭
# ════════════════════════════════════════════════════════════════════════════
def sharpe(daily_krw: pd.Series, ann: float) -> float:
    s = daily_krw.dropna()
    return float(s.mean() / s.std(ddof=1) * math.sqrt(ann)) if len(s) >= 2 and s.std(ddof=1) > 0 else 0.0


def max_dd(daily_krw: pd.Series) -> float:
    eq = daily_krw.cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else 0.0


# ── A. 트리거 발동 빈도 ──────────────────────────────────────────────────────
def report_frequency(d: pd.DataFrame, trig: Dict[str, pd.Series]) -> pd.DataFrame:
    n = len(d)
    rows = [{"트리거": k, "발동일수": int(v.sum()), "비율%": round(100 * v.sum() / n, 1)}
            for k, v in trig.items()]
    return pd.DataFrame(rows)


# ── B. 지속형 vs V자 분류 (진단용 forward 라벨) ──────────────────────────────
def report_sustain_vs_v(d: pd.DataFrame, trig: Dict[str, pd.Series],
                        fwd: int = 5, thr: float = 0.01) -> pd.DataFrame:
    c = d["CLOSE"]
    fwd_ret = c.shift(-fwd) / c - 1.0           # 진단 전용(전향). 매매신호 아님.
    same_day_down = (d["CLOSE"] < d["OPEN"])    # 그날 시가>종가 (직진 숏에 유리)
    rows = []
    for k, v in trig.items():
        m = v & fwd_ret.notna()
        nfire = int(m.sum())
        if nfire == 0:
            rows.append({"트리거": k, "발동": 0, "지속%": None, "혼조%": None,
                         "V반등%": None, "당일하락%": None}); continue
        fr = fwd_ret[m]
        sust = int((fr < -thr).sum()); vrev = int((fr > thr).sum()); mix = nfire - sust - vrev
        rows.append({"트리거": k, "발동": nfire,
                     "지속%": round(100 * sust / nfire, 1),
                     "혼조%": round(100 * mix / nfire, 1),
                     "V반등%": round(100 * vrev / nfire, 1),
                     "당일하락%": round(100 * same_day_down[m].mean(), 1)})
    return pd.DataFrame(rows)


# ── C. 직진 숏 (트리거 날 시가숏→종가청산) ──────────────────────────────────
def report_naive_short(d: pd.DataFrame, trig: Dict[str, pd.Series], cost: Cost) -> pd.DataFrame:
    o, c = d["OPEN"], d["CLOSE"]
    rows = []
    for k, v in trig.items():
        days = d.index[v]
        if len(days) == 0:
            rows.append({"트리거": k, "거래": 0, "승률%": None, "PnL원": 0,
                         "기대값/거래": None, "Sharpe": 0.0, "MaxDD원": 0}); continue
        e, x = o.loc[days].to_numpy(), c.loc[days].to_numpy()
        gross = -1.0 * (x - e)                                   # 숏: 시가-종가
        net_pts = gross - np.array([cost.round_trip_pts(ep, xp) for ep, xp in zip(e, x)])
        net_krw = pd.Series(net_pts * cost.multiplier, index=days)
        rows.append({"트리거": k, "거래": len(days),
                     "승률%": round(100 * (net_pts > 0).mean(), 1),
                     "PnL원": round(float(net_krw.sum())),
                     "기대값/거래": round(float(net_krw.mean())),
                     "Sharpe": round(sharpe(net_krw, cost.annualization), 2),
                     "MaxDD원": round(max_dd(net_krw))})
    return pd.DataFrame(rows)


# ── D. 롱-또는-플랫 지연손실 ─────────────────────────────────────────────────
def report_lf_lag(d: pd.DataFrame, cost: Cost) -> Dict:
    sig = long_or_flat_signal(d)
    o, c = d["OPEN"], d["CLOSE"]
    lf_krw = pd.Series(0.0, index=d.index)
    longd = sig == 1
    if longd.any():
        e, x = o[longd].to_numpy(), c[longd].to_numpy()
        net = (x - e) - np.array([cost.round_trip_pts(ep, xp) for ep, xp in zip(e, x)])
        lf_krw.loc[d.index[longd]] = net * cost.multiplier

    peak_i = int(np.argmax(c.to_numpy()))
    trough_i = int(np.argmin(c.to_numpy()[peak_i:]) + peak_i) if peak_i < len(c) - 1 else peak_i
    peak_dt, trough_dt = d.index[peak_i], d.index[trough_i]
    dd_pct = (c.iloc[trough_i] / c.iloc[peak_i] - 1.0) * 100 if c.iloc[peak_i] else 0.0

    after = sig.iloc[peak_i:]
    flip_rel = next((j for j, val in enumerate(after.to_numpy()) if val == 0), None)
    if flip_rel is None:
        flip_dt, lag_days = None, None
        bleed = float(lf_krw.iloc[peak_i:].sum())
        bleed_to = d.index[-1]
    else:
        flip_i = peak_i + flip_rel
        flip_dt, lag_days = d.index[flip_i], int(flip_rel)
        bleed = float(lf_krw.iloc[peak_i:flip_i].sum())  # 게이트 꺼지기 전 롱으로 까먹은 금액
        bleed_to = flip_dt
    return {
        "고점": str(peak_dt.date()), "저점": str(trough_dt.date()), "낙폭%": round(dd_pct, 1),
        "게이트플립": (str(flip_dt.date()) if flip_dt is not None else "미플립(구간내)"),
        "플립지연(거래일)": lag_days,
        "지연중_롱손익원": round(bleed),
        "롱플랫_구간전체PnL원": round(float(lf_krw.sum())),
    }


# ════════════════════════════════════════════════════════════════════════════
# 입출력
# ════════════════════════════════════════════════════════════════════════════
_COLMAP = {
    "date": "DT", "일자": "DT", "timestamp": "DT", "time": "DT", "dt": "DT",
    "open": "OPEN", "시가": "OPEN", "high": "HIGH", "고가": "HIGH",
    "low": "LOW", "저가": "LOW", "close": "CLOSE", "종가": "CLOSE",
    "volume": "VOLUME", "거래량": "VOLUME", "vol": "VOLUME",
}


def _parse_dt(s: pd.Series) -> pd.Series:
    """여러 형식을 견고하게 파싱: 'YYYY-MM-DD', 'YYYYMMDD', 'YYYYMMDD HHMM' 등."""
    s = s.astype(str).str.strip()
    for fmt in ("%Y%m%d %H%M", "%Y%m%d %H%M%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        out = pd.to_datetime(s, format=fmt, errors="coerce")
        if out.notna().mean() > 0.8:
            return out
    return pd.to_datetime(s, errors="coerce")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    ren = {col: _COLMAP[col.strip().lower()] for col in df.columns
           if col.strip().lower() in _COLMAP}
    df = df.rename(columns=ren)
    if "DT" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "DT"})
    df["DT"] = _parse_dt(df["DT"])
    df = df.dropna(subset=["DT"]).set_index("DT").sort_index()
    need = {"OPEN", "HIGH", "LOW", "CLOSE"}
    if not need.issubset(df.columns):
        raise ValueError(f"필요 컬럼 누락: {need - set(df.columns)} | 가진 컬럼: {list(df.columns)}")
    if "VOLUME" not in df.columns:
        df["VOLUME"] = 0.0
    return df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]].astype(float)


def load_daily_from_csv(paths: List[str]) -> pd.DataFrame:
    parts = [_normalize(pd.read_csv(p)) for p in paths]
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated(keep="last")]


def load_daily_from_duckdb(db: str, table: str, ts_col: str = "timestamp",
                           resample: bool = False) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(db, read_only=True)
    raw = con.execute(f"SELECT * FROM {table} ORDER BY {ts_col}").df()
    con.close()
    df = _normalize(raw)
    if resample:  # 분봉 → 일봉
        df = df.resample("1D").agg({"OPEN": "first", "HIGH": "max", "LOW": "min",
                                    "CLOSE": "last", "VOLUME": "sum"}).dropna()
    return df


def make_demo() -> pd.DataFrame:
    """합성 일봉: 상승추세 + 지속형 하락 1개 + V자 딥 1개."""
    rng = pd.date_range("2024-01-01", periods=360, freq="B")
    np.random.seed(7)
    px = [300.0]
    for i in range(1, len(rng)):
        drift = 0.0008
        if 120 <= i < 150:   drift = -0.015      # 지속형 하락 30일
        if 240 <= i < 245:   drift = -0.030      # V자 급락 5일
        if 245 <= i < 252:   drift = +0.028      # 곧 반등
        px.append(px[-1] * (1 + drift + np.random.normal(0, 0.006)))
    c = np.array(px)
    o = c * (1 + np.random.normal(0, 0.002, len(c)))
    h = np.maximum(o, c) * (1 + np.abs(np.random.normal(0, 0.003, len(c))))
    lo = np.minimum(o, c) * (1 - np.abs(np.random.normal(0, 0.003, len(c))))
    return pd.DataFrame({"OPEN": o, "HIGH": h, "LOW": lo, "CLOSE": c, "VOLUME": 1.0}, index=rng)


# ════════════════════════════════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════════════════════════════════
def _pp(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def screen(full: pd.DataFrame, episodes: List[Tuple[str, str, str]], cost: Cost) -> None:
    d_all = add_indicators(full)
    print(f"\n로드된 일봉: {len(full)}일 | {full.index[0].date()} ~ {full.index[-1].date()}")
    print(f"비용모델: 승수 {cost.multiplier:,.0f} | 수수료 {cost.commission_pct_per_side*100:.3f}%/side "
          f"| 슬리피지 {cost.slippage_ticks_per_side}틱")

    bear_union = []
    for name, s, e in episodes:
        seg = d_all.loc[s:e]
        if len(seg) < 65:
            print(f"\n[{name}] {s}~{e}: 데이터 {len(seg)}일 (MA60/ADX 워밍업 부족) → 건너뜀")
            continue
        warm = seg["MA60"].notna().sum()
        bear_union.append(seg)
        trig = build_triggers(seg)
        print("\n" + "=" * 78)
        print(f"[{name}] {seg.index[0].date()} ~ {seg.index[-1].date()}  ({len(seg)}일, 지표유효 {warm}일)")
        print("-- A. 트리거 발동 빈도 --");          print(_pp(report_frequency(seg, trig)))
        print("-- B. 지속형 vs V자 (forward 5일, 진단용) --"); print(_pp(report_sustain_vs_v(seg, trig)))
        print("-- C. 직진 숏 (시가숏→종가청산) --");   print(_pp(report_naive_short(seg, trig, cost)))
        print("-- D. 롱-또는-플랫 지연손실 --")
        for k, v in report_lf_lag(seg, cost).items():
            print(f"     {k}: {v}")

    if len(bear_union) >= 2:
        allbear = pd.concat(bear_union).sort_index()
        allbear = allbear[~allbear.index.duplicated(keep="last")]
        trig = build_triggers(allbear)
        print("\n" + "#" * 78)
        print(f"[전체 약세장 합산] {len(allbear)}일")
        print("-- A. 트리거 발동 빈도 --");          print(_pp(report_frequency(allbear, trig)))
        print("-- B. 지속형 vs V자 --");             print(_pp(report_sustain_vs_v(allbear, trig)))
        print("-- C. 직진 숏 --");                   print(_pp(report_naive_short(allbear, trig, cost)))
        print("\n해석 가이드:")
        print("  · C에서 직진 숏 Sharpe<0 / PnL<0 이면 → 장중 피봇으로 정교화해도 가망 낮음(분봉수집 보류).")
        print("  · B에서 'V반등%'가 높은 트리거는 6월형 휩쏘 위험 → '3일연속하락' 등 지속성 강제 필요.")
        print("  · D '지연중_롱손익'이 크게 음수면 → 빠른 손절/숏오버레이의 가치가 실제로 존재.")


def main():
    ap = argparse.ArgumentParser(description="피봇-숏 오버레이 일봉 스크리닝")
    ap.add_argument("--csv", nargs="+", help="일봉 CSV 경로(복수 가능)")
    ap.add_argument("--duckdb", help="DuckDB 경로")
    ap.add_argument("--table", default="futures_1min", help="DuckDB 테이블명")
    ap.add_argument("--resample", action="store_true", help="DuckDB 분봉을 일봉으로 리샘플")
    ap.add_argument("--demo", action="store_true", help="합성 데이터로 형식 확인")
    ap.add_argument("--multiplier", type=float, default=250_000.0)
    args = ap.parse_args()

    cost = Cost(multiplier=args.multiplier)
    if args.demo:
        full = make_demo()
        eps = [("DEMO_지속하락", "2024-06-01", "2024-08-31"),
               ("DEMO_V자딥",    "2024-11-01", "2025-01-31")]
    elif args.csv:
        full = load_daily_from_csv(args.csv); eps = DEFAULT_EPISODES
    elif args.duckdb:
        full = load_daily_from_duckdb(args.duckdb, args.table, resample=args.resample)
        eps = DEFAULT_EPISODES
    else:
        ap.error("--csv / --duckdb / --demo 중 하나가 필요합니다.")
    screen(full, eps, cost)


if __name__ == "__main__":
    main()
