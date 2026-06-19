# -*- coding: utf-8 -*-
"""
regime_intraday_v2.py
=====================

레짐 기반 당일매매 전략의 **인과적(look-ahead 없는)** 재구현 + 공정 비교 하니스.

업로드된 pivot_optuna_v2.regime_based_intraday 의 두 결함을 교정한다:

  결함 A) 'MA 20/60' 이 실제로는 20/60 *분(分)* 이동평균이다.
          short_ma = Series(px_close).rolling(ma_short).mean()  ← px_close 는 1분봉 배열.
          문서는 20/60 '일' 레짐("MA 60일 워밍업")이라 설명하지만 코드는 분 단위 모멘텀이다.
  결함 B) look-ahead. 당일 첫 봉에서 regime 을 short_ma[first]/long_ma[first] 로 정하는데
          이 MA 는 그 첫 봉의 종가 px_close[first] 를 포함한다. 그런데 진입은 같은 봉의
          시가 px_open[first] 다 → 진입 시점에 알 수 없는 그 봉 종가를 사용(미래참조).

이 모듈은 (1) 일봉 OHLC 를 만들고 (2) 일봉 종가 MA 로 레짐을 정하되 shift(1) 로
'전일까지의 정보만' 사용해 D일 시가→종가 매매를 한다 → look-ahead 제거.

사용:
    import pivot_optuna_v2 as pv
    import regime_intraday_v2 as rg
    res = rg.regime_intraday_daily(df_daysession, bt, ma_short=20, ma_long=60)
    rg.compare_strategies(train_i, test_i, pcfg, fcfg, bt, daily_reset=True)
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

import pivot_optuna_v2 as pv


# ════════════════════════════════════════════════════════════════════════════
# 일봉 생성 + 일봉 ADX
# ════════════════════════════════════════════════════════════════════════════
def to_daily(df: pd.DataFrame, session_boundary_hour: int = 8) -> pd.DataFrame:
    """주간세션 intraday df → 거래일별 OHLC.

    open = 그 거래일 첫 봉 시가, close = 마지막 봉 종가, high/low = 당일 고저.
    index 는 '거래일 종료 timestamp'(마지막 봉 시각)로 둬서 날짜 정렬을 보존한다.
    """
    tday = pv.trading_day_key(df.index, session_boundary_hour)
    s = pd.Series(np.arange(len(df)), index=df.index)
    g_first = s.groupby(tday).first()
    g_last = s.groupby(tday).last()
    o = df["OPEN"].to_numpy() if "OPEN" in df.columns else df["CLOSE"].to_numpy()
    h = df["HIGH"].to_numpy(); l = df["LOW"].to_numpy(); c = df["CLOSE"].to_numpy()
    rows = []
    for k in g_first.index:
        fi, li = int(g_first[k]), int(g_last[k])
        rows.append({
            "end_time": df.index[li],
            "OPEN": o[fi], "HIGH": h[fi:li + 1].max(),
            "LOW": l[fi:li + 1].min(), "CLOSE": c[li],
        })
    daily = pd.DataFrame(rows).set_index("end_time").sort_index()
    return daily


def _daily_adx(daily: pd.DataFrame, period: int = 14) -> pd.Series:
    return pv._adx(daily, period)   # daily 프레임도 OPEN/HIGH/LOW/CLOSE 보유 → 재사용


# ════════════════════════════════════════════════════════════════════════════
# 인과적 레짐 당일매매
# ════════════════════════════════════════════════════════════════════════════
def daily_regime_signal(
    daily: pd.DataFrame,
    regime_method: str = "ma",
    ma_short: int = 20,
    ma_long: int = 60,
    adx_threshold: float = 25.0,
    allow_short: bool = True,
) -> pd.Series:
    """일봉 프레임에서 '거래 신호'(D일 진입 방향)를 계산한다.

    MA 는 일봉 종가에 대해 계산하고, shift(1) 로 D일 결정에 D-1 까지의 정보만 쓴다
    (look-ahead 제거). 반환은 +1(롱)/-1(숏)/0(스킵) Series, index=일봉 end_time.
    allow_short=False 면 bear(-1) 신호를 0(플랫)으로 바꿔 '롱-또는-플랫'(숏 금지)이 된다.

    ★ 이 함수를 '전체 기간 일봉'에 대해 한 번 계산한 뒤 train/test 로 슬라이스하면
      test 구간 앞단의 MA 워밍업 절단(=신호 0건) 문제가 사라진다.
    """
    close = daily["CLOSE"]
    ma_s = close.rolling(ma_short).mean()
    ma_l = close.rolling(ma_long).mean()
    regime = pd.Series(0, index=daily.index, dtype=int)
    regime[ma_s > ma_l] = 1
    regime[ma_s < ma_l] = -1
    if regime_method == "adx":
        adx = _daily_adx(daily, 14)
        regime[adx < adx_threshold] = 0
    if not allow_short:
        regime[regime < 0] = 0          # bear → 숏 대신 플랫(현금)
    return regime.shift(1).fillna(0).astype(int)


def _bt_from_daily_signal(daily: pd.DataFrame, signal: pd.Series, cfg) -> "pv.BacktestResult":
    """일봉 + 신호 → 당일 시가진입/종가청산 백테스트 결과."""
    sig = signal.reindex(daily.index).fillna(0).astype(int).to_numpy()
    o = daily["OPEN"].to_numpy(); c = daily["CLOSE"].to_numpy()
    rows = []
    for i in range(len(daily)):
        d = int(sig[i])
        if d == 0:
            continue
        e_px, x_px = o[i], c[i]
        net = d * (x_px - e_px) - cfg.round_trip_cost_pts(e_px, x_px)
        rows.append({"exit_time": daily.index[i], "direction": d,
                     "net_pts": net, "net_krw": net * cfg.multiplier})
    if not rows:
        return pv.BacktestResult(trades=pd.DataFrame())
    tdf = pd.DataFrame(rows)
    daily_pnl = tdf.groupby(pd.to_datetime(tdf["exit_time"]).dt.date)["net_krw"].sum()
    sharpe = (float(daily_pnl.mean() / daily_pnl.std(ddof=1) * math.sqrt(cfg.annualization))
              if len(daily_pnl) >= 2 and daily_pnl.std(ddof=1) > 0 else 0.0)
    equity = tdf["net_krw"].cumsum()
    # 승률 계산 (net_pts > 0인 비율)
    wins = (tdf["net_pts"] > 0).sum()
    total = len(tdf)
    win_rate = float((wins / total) * 100) if total > 0 else 0.0
    
    return pv.BacktestResult(
        n_trades=len(tdf),
        win_rate=win_rate,
        total_pnl_pts=float(tdf["net_pts"].sum()),
        total_pnl_krw=float(tdf["net_krw"].sum()),
        expectancy_pts=float(tdf["net_pts"].mean()),
        sharpe_daily=sharpe,
        max_drawdown_krw=float((equity - equity.cummax()).min()),
        trades=tdf,
    )


def annualized_sharpe_se(res, annualization: float = 252.0) -> float:
    """연율화 Sharpe 의 근사 표준오차. n(거래일수)이 작으면 Sharpe 신뢰구간이 넓다.
    SE ~ sqrt((1 + 0.5*SR_ann^2/ann) / n) * sqrt(ann)  (Lo, 2002 근사)."""
    if res.trades is None or len(res.trades) < 2:
        return float("nan")
    n = res.trades["exit_time"].pipe(lambda s: pd.to_datetime(s).dt.date).nunique()
    if n < 2:
        return float("nan")
    sr_ann = res.sharpe_daily
    return float(math.sqrt((1.0 + 0.5 * sr_ann**2 / annualization) / n) * math.sqrt(annualization))


def regime_intraday_daily(
    df: pd.DataFrame,
    cfg: "pv.BacktestConfig",
    regime_method: str = "ma",      # 'ma' | 'adx'
    ma_short: int = 20,
    ma_long: int = 60,
    adx_threshold: float = 25.0,
    signal: Optional[pd.Series] = None,
) -> "pv.BacktestResult":
    """일봉 MA 레짐으로 D일 시가→종가 매매. shift(1) 로 look-ahead 제거.

    signal 을 주면(전체 기간에서 미리 계산한 신호) 그것을 슬라이스해 쓴다 → 워밍업
    절단 없음. 안 주면 df 자체 일봉으로 계산(짧은 구간이면 MA 워밍업으로 신호 0 가능).
    bull(MA단기>장기) → 롱, bear → 숏, neutral → 스킵.
    """
    daily = to_daily(df, cfg.session_boundary_hour)
    if signal is None:
        if len(daily) < ma_long + 2:
            return pv.BacktestResult(trades=pd.DataFrame())
        signal = daily_regime_signal(daily, regime_method, ma_short, ma_long, adx_threshold)
    return _bt_from_daily_signal(daily, signal, cfg)


# ════════════════════════════════════════════════════════════════════════════
# 공정 비교 — train↔test (단일 full-period 수치 대신 out-of-sample 포함)
# ════════════════════════════════════════════════════════════════════════════
def _fmt(res) -> str:
    return (f"Sharpe {res.sharpe_daily:>7.3f} | PnL {res.total_pnl_krw:>14,.0f} | "
            f"MaxDD {res.max_drawdown_krw:>13,.0f} | 거래 {res.n_trades:>4}")


def compare_strategies(
    train_i: pd.DataFrame,
    test_i: pd.DataFrame,
    pcfg: "pv.HybridAdaptivePivotConfig",
    fcfg: "pv.FilterConfig",
    bt: "pv.BacktestConfig",
    daily_reset: bool = True,
    ma_configs=((10, 30), (20, 60), (50, 200)),
) -> None:
    """피봇(both) / 피봇 롱only / 장중상시롱(베타) / 인과적 레짐 을 train·test 동시 비교.

    핵심 기준: 어떤 후보든 *test 구간에서* '장중 상시 롱' 베타 베이스라인을
    유의미하게 이겨야 채택 가치가 있다. (단일 full-period 수치는 신뢰하지 않는다)
    """
    def _detect(d):
        return (pv.detect_pivots_daily(d, pcfg, fcfg, bt.session_boundary_hour)
                if daily_reset else pv.detect_pivots(d, pcfg, fcfg))

    print(f"{'전략':<26}{'TRAIN':<58}{'TEST'}")
    print("-" * 132)

    def line(name, tr, te):
        print(f"{name:<26}{_fmt(tr):<58}{_fmt(te)}")

    # 1) 피봇 반전 (both)
    line("피봇 반전(both)",
         pv.backtest(train_i, _detect(train_i), bt),
         pv.backtest(test_i, _detect(test_i), bt))

    # 2) 장중 상시 롱 (베타 베이스라인)
    line("장중 상시 롱(베타)",
         pv.intraday_long_baseline(train_i, bt),
         pv.intraday_long_baseline(test_i, bt))

    # 3) 인과적 레짐 (여러 MA)
    for s, l in ma_configs:
        line(f"레짐(causal) MA{s}/{l}",
             regime_intraday_daily(train_i, bt, ma_short=s, ma_long=l),
             regime_intraday_daily(test_i, bt, ma_short=s, ma_long=l))

    print("-" * 132)
    print("판정: test 열에서 '장중 상시 롱' 대비 Sharpe/PnL 우위 + MaxDD 개선이 있어야 진짜 대체가치.")
    print("      full-period 단일 수치(문서의 87M/1.456)는 out-of-sample 이 아니므로 신뢰 금지.")


def select_regime_oos(
    full_i: pd.DataFrame,
    test_start,
    cfg: "pv.BacktestConfig",
    ma_configs=((5, 15), (10, 30), (20, 60), (50, 200)),
    regime_method: str = "ma",
    adx_threshold: float = 25.0,
    allow_short: bool = True,
    verbose: bool = True,
) -> Dict:
    """올바른 OOS 절차로 MA 길이를 고른다.

    핵심 교정:
      (1) MA 워밍업 절단 제거 — 신호를 '전체 기간 일봉'에서 1회 계산 후 train/test 로 슬라이스.
          (기존 결과의 'MA20/60 test 0건'은 전략이 보수적이어서가 아니라 워밍업 절단 artifact)
      (2) test-set 선택 금지 — best 는 'TRAIN Sharpe' 로 고른다. test 는 최종 1회만 본다.
          (기존 'MA10/30 채택'은 test 를 보고 고른 것 → test 가 더 이상 OOS 아님)
      (3) Sharpe 표준오차 동반 — 짧은 test 에서 1.749 vs 1.250 같은 차이가 노이즈인지 본다.

    test_start: test 구간 첫 봉의 시각(예: test_i.index[0]).
    """
    daily_full = to_daily(full_i, cfg.session_boundary_hour)
    is_test = daily_full.index >= test_start
    d_tr, d_te = daily_full[~is_test], daily_full[is_test]

    rows = []
    for s, l in ma_configs:
        sig = daily_regime_signal(daily_full, regime_method, s, l, adx_threshold, allow_short)  # 전체 워밍업
        tr = _bt_from_daily_signal(d_tr, sig, cfg)
        te = _bt_from_daily_signal(d_te, sig, cfg)
        rows.append({"name": f"MA{s}/{l}", "ma_short": s, "ma_long": l,
                     "train": tr, "test": te})

    # baseline: 장중 상시 롱 (워밍업 불필요)
    base_tr = pv.intraday_long_baseline(full_i[full_i.index < test_start], cfg)
    base_te = pv.intraday_long_baseline(full_i[full_i.index >= test_start], cfg)

    valid = [r for r in rows if r["train"].n_trades > 0]
    best = max(valid, key=lambda r: r["train"].sharpe_daily) if valid else None

    if verbose:
        mode = "롱/숏" if allow_short else "롱/플랫"
        print(f"전략 모드: {mode}")
        print(f"{'전략':<16}{'TRAIN Sharpe':>14}{'TEST Sharpe':>14}{'TEST PnL':>16}"
              f"{'TEST MaxDD':>16}{'TEST거래':>9}")
        print("-" * 85)
        bse = annualized_sharpe_se(base_te, cfg.annualization)
        print(f"{'장중상시롱(베타)':<16}{base_tr.sharpe_daily:>14.3f}{base_te.sharpe_daily:>14.3f}"
              f"{base_te.total_pnl_krw:>16,.0f}{base_te.max_drawdown_krw:>16,.0f}{base_te.n_trades:>9}")
        print(f"{'  (테스트 Sharpe SE ~ +/-' + f'{bse:.2f})':<16}")
        for r in rows:
            te = r["test"]
            mark = " ★train선택" if (best and r["name"] == best["name"]) else ""
            print(f"{r['name']:<16}{r['train'].sharpe_daily:>14.3f}{te.sharpe_daily:>14.3f}"
                  f"{te.total_pnl_krw:>16,.0f}{te.max_drawdown_krw:>16,.0f}{te.n_trades:>9}{mark}")
        print("-" * 85)
        if best:
            te, tr = best["test"], best["train"]
            se = annualized_sharpe_se(te, cfg.annualization)
            beats = te.sharpe_daily - base_te.sharpe_daily
            print(f"TRAIN 기준 선택: {best['name']} | TEST Sharpe {te.sharpe_daily:.3f} "
                  f"(SE~+/-{se:.2f}) vs 베타 {base_te.sharpe_daily:.3f}")
            verdict = ("베타 초과 불확실(차이가 SE 이내 -> 노이즈)"
                       if abs(beats) <= (se if se == se else 1e9)
                       else ("베타 초과(차이가 SE 밖)" if beats > 0 else "베타 미달"))
            print(f"판정: {verdict}. |Delta|={abs(beats):.2f}, SE~{se:.2f}")
        else:
            print("TRAIN 구간에서 신호가 나는 MA 조합이 없음.")

    return {"rows": rows, "baseline_train": base_tr, "baseline_test": base_te, "best": best}


def regime_window_dispersion(
    full_i: pd.DataFrame,
    cfg: "pv.BacktestConfig",
    ma_short: int,
    ma_long: int,
    regime_method: str = "ma",
    adx_threshold: float = 25.0,
    allow_short: bool = True,
    n_windows: int = 6,
    verbose: bool = True,
) -> pd.DataFrame:
    """선택한 MA 조합을 전체 기간 n_windows 로 나눠 구간별 Sharpe/PnL 을 본다.

    단일 홀드아웃이 운인지(레짐 의존) 판단. 신호는 전체 일봉에서 1회 계산(워밍업 보존).
    """
    daily_full = to_daily(full_i, cfg.session_boundary_hour)
    sig = daily_regime_signal(daily_full, regime_method, ma_short, ma_long, adx_threshold, allow_short)
    n = len(daily_full); w = n // n_windows
    rows = []
    for i in range(n_windows):
        s = i * w; e = (i + 1) * w if i < n_windows - 1 else n
        seg = daily_full.iloc[s:e]
        res = _bt_from_daily_signal(seg, sig, cfg)
        rows.append({"window": i + 1, "start": seg.index[0].date(), "end": seg.index[-1].date(),
                     "n_trades": res.n_trades, "sharpe": round(res.sharpe_daily, 3),
                     "pnl_krw": round(res.total_pnl_krw)})
    out = pd.DataFrame(rows)
    if verbose:
        print(f"  [MA{ma_short}/{ma_long}] 구간별 (전체 일봉, 워밍업 보존)")
        for _, r in out.iterrows():
            print(f"   win{int(r['window'])} {r['start']}~{r['end']}  거래{int(r['n_trades']):>3}"
                  f"  Sharpe{r['sharpe']:>8.3f}  PnL{int(r['pnl_krw']):>13,}")
        print(f"   → 수익구간 {(out['pnl_krw']>0).sum()}/{len(out)}")
    return out


# ════════════════════════════════════════════════════════════════════════════
# 다운사이드 보호 정량화 (1) 베타 구간별 MaxDD (2) 롱-또는-플랫/손절 효과
# ════════════════════════════════════════════════════════════════════════════
def _result_from_daily_trades(tdf: pd.DataFrame, cfg) -> "pv.BacktestResult":
    """거래 DataFrame(net_pts/net_krw/exit_time/direction) → BacktestResult."""
    if tdf is None or len(tdf) == 0:
        return pv.BacktestResult(trades=pd.DataFrame())
    daily_pnl = tdf.groupby(pd.to_datetime(tdf["exit_time"]).dt.date)["net_krw"].sum()
    sharpe = (float(daily_pnl.mean() / daily_pnl.std(ddof=1) * math.sqrt(cfg.annualization))
              if len(daily_pnl) >= 2 and daily_pnl.std(ddof=1) > 0 else 0.0)
    equity = tdf["net_krw"].cumsum()
    # 승률 계산 (net_pts > 0인 비율)
    wins = (tdf["net_pts"] > 0).sum()
    total = len(tdf)
    win_rate = float((wins / total) * 100) if total > 0 else 0.0
    
    return pv.BacktestResult(
        n_trades=len(tdf),
        win_rate=win_rate,
        total_pnl_pts=float(tdf["net_pts"].sum()),
        total_pnl_krw=float(tdf["net_krw"].sum()),
        expectancy_pts=float(tdf["net_pts"].mean()),
        sharpe_daily=sharpe,
        max_drawdown_krw=float((equity - equity.cummax()).min()),
        trades=tdf,
    )


def baseline_window_dispersion(
    full_i: pd.DataFrame, cfg: "pv.BacktestConfig", n_windows: int = 6, verbose: bool = True
) -> pd.DataFrame:
    """'장중 상시 롱'(베타)을 전 구간 n_windows 로 나눠 구간별 Sharpe/PnL/MaxDD 출력.

    배포 대상(베타) 자체의 *최악 구간 MaxDD* 가 감내 가능한지 보는 것이 핵심.
    """
    n = len(full_i); w = n // n_windows
    rows = []
    for i in range(n_windows):
        s = i * w; e = (i + 1) * w if i < n_windows - 1 else n
        seg = full_i.iloc[s:e]
        r = pv.intraday_long_baseline(seg, cfg)
        rows.append({"window": i + 1, "start": seg.index[0].date(), "end": seg.index[-1].date(),
                     "n_trades": r.n_trades, "sharpe": round(r.sharpe_daily, 3),
                     "pnl_krw": round(r.total_pnl_krw), "maxdd_krw": round(r.max_drawdown_krw)})
    out = pd.DataFrame(rows)
    if verbose:
        print("  [장중 상시 롱(베타)] 구간별")
        for _, r in out.iterrows():
            print(f"   win{int(r['window'])} {r['start']}~{r['end']}  거래{int(r['n_trades']):>3}"
                  f"  Sharpe{r['sharpe']:>8.3f}  PnL{int(r['pnl_krw']):>13,}  MaxDD{int(r['maxdd_krw']):>13,}")
        print(f"   → 수익구간 {(out['pnl_krw']>0).sum()}/{len(out)} | "
              f"최악 구간 MaxDD {int(out['maxdd_krw'].min()):,}원")
    return out


def intraday_stop_backtest(
    df: pd.DataFrame,
    cfg: "pv.BacktestConfig",
    signal: pd.Series,
    stop_pct: Optional[float] = None,       # 진입가 대비 % 손절 (예: 0.5 = 0.5%)
    stop_atr_mult: Optional[float] = None,  # 전일 일봉 ATR 배수 손절
    atr_period: int = 14,
) -> "pv.BacktestResult":
    """장중 경로(intraday path)를 이용해 손절을 반영하는 백테스트.

    각 거래일에 신호 방향(+1/-1)이면 첫 봉 시가 진입. 손절선이 설정되면 장중 봉의
    저가(롱)/고가(숏)가 손절선을 건드리는 순간 손절가로 청산, 아니면 마지막 봉 종가 청산.
    신호 0이면 스킵(플랫). signal index=일봉 end_time 순서와 거래일 순서가 일치한다고 가정.
    """
    sbh = cfg.session_boundary_hour
    tday = pv.trading_day_key(df.index, sbh)
    o = df["OPEN"].to_numpy() if "OPEN" in df.columns else df["CLOSE"].to_numpy()
    h = df["HIGH"].to_numpy(); l = df["LOW"].to_numpy(); c = df["CLOSE"].to_numpy()
    pos = np.arange(len(df))

    # 전일 일봉 ATR (causal): ATR 손절용
    atr_prev = None
    if stop_atr_mult is not None:
        daily = to_daily(df, sbh)
        atr_prev = pv._atr(daily, atr_period).shift(1).to_numpy()

    sig = signal.to_numpy() if hasattr(signal, "to_numpy") else np.asarray(signal)
    uniq = pd.unique(tday)
    rows = []
    for j, day_val in enumerate(uniq):
        d = int(sig[j]) if j < len(sig) else 0
        if d == 0:
            continue
        mask = tday == day_val
        ii = pos[mask]
        first, last = int(ii[0]), int(ii[-1])
        e_px = o[first]
        # 손절선
        stop_px = None
        dist = None
        if stop_pct is not None:
            dist = e_px * (stop_pct / 100.0)
        elif stop_atr_mult is not None and atr_prev is not None and not np.isnan(atr_prev[j]):
            dist = stop_atr_mult * float(atr_prev[j])
        if dist is not None:
            stop_px = e_px - d * dist          # 롱이면 아래, 숏이면 위
        # 장중 경로 워크 → 손절 우선
        x_px, reason = c[last], "eod"
        if stop_px is not None:
            for k in range(first, last + 1):
                if d == 1 and l[k] <= stop_px:
                    x_px, reason = stop_px, "stop"; break
                if d == -1 and h[k] >= stop_px:
                    x_px, reason = stop_px, "stop"; break
        net = d * (x_px - e_px) - cfg.round_trip_cost_pts(e_px, x_px)
        rows.append({"exit_time": df.index[last], "direction": d, "exit_reason": reason,
                     "net_pts": net, "net_krw": net * cfg.multiplier})
    return _result_from_daily_trades(pd.DataFrame(rows), cfg)


def compare_downside_protection(
    full_i: pd.DataFrame,
    train_i: pd.DataFrame,
    test_i: pd.DataFrame,
    cfg: "pv.BacktestConfig",
    ma_short: int = 20,
    ma_long: int = 60,
    regime_method: str = "ma",
    adx_threshold: float = 25.0,
    allow_short: bool = False,
    stop_pct: Optional[float] = 0.5,
) -> None:
    """베타 vs (롱-또는-플랫) vs (+손절) 을 train/test 로 비교 — *MaxDD 개선*에 초점.

    목적은 PnL 극대화가 아니라 '하락 구간에서 덜 깨지는가'다.
    """
    daily_full = to_daily(full_i, cfg.session_boundary_hour)
    test_start = test_i.index[0]

    def split(res_fn):
        return res_fn(full_i[full_i.index < test_start]), res_fn(full_i[full_i.index >= test_start])

    # 신호(전체 워밍업 보존). 롱-또는-플랫: allow_short=False
    sig_lf = daily_regime_signal(daily_full, regime_method, ma_short, ma_long,
                                 adx_threshold, allow_short)

    # 후보들
    def beta(d):       return pv.intraday_long_baseline(d, cfg)
    def lf(d):         return regime_intraday_daily(d, cfg, regime_method, ma_short, ma_long,
                                                    adx_threshold, signal=sig_lf)
    def beta_stop(d):
        # 베타 + 손절: 매일 롱 신호
        tdk = pd.unique(pv.trading_day_key(d.index, cfg.session_boundary_hour))
        s = pd.Series(1, index=range(len(tdk)))
        return intraday_stop_backtest(d, cfg, s, stop_pct=stop_pct)
    def lf_stop(d):
        sub = sig_lf.reindex(to_daily(d, cfg.session_boundary_hour).index).fillna(0).astype(int)
        return intraday_stop_backtest(d, cfg, sub.reset_index(drop=True), stop_pct=stop_pct)

    cands = [
        ("장중상시롱(베타)", beta),
        (f"롱-또는-플랫 MA{ma_short}/{ma_long}", lf),
        (f"베타+손절 {stop_pct}%", beta_stop),
        (f"롱플랫+손절 {stop_pct}%", lf_stop),
    ]
    print(f"{'전략':<24}{'TRAIN: Sharpe/PnL/MaxDD':<46}{'TEST: Sharpe/PnL/MaxDD'}")
    print("-" * 120)
    for name, fn in cands:
        tr, te = split(fn)
        def f(r): return f"{r.sharpe_daily:>6.3f} / {r.total_pnl_krw:>12,.0f} / {r.max_drawdown_krw:>12,.0f}"
        print(f"{name:<24}{f(tr):<46}{f(te)}")
    print("-" * 120)
    print("초점: TEST MaxDD 가 베타 대비 줄면서 PnL 손실이 작으면 다운사이드 보호로서 채택 가치.")
    print("      (PnL 우위가 아니라 '덜 깨지는가'가 기준. 상승장 표본에선 PnL 은 베타가 더 클 수 있음)")
