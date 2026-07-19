# -*- coding: utf-8 -*-
"""
pivot_bull_gated.py — BULL 피봇 롱 '신호 게이팅' 백테스트 (엔진 수정판)

기존 방식(`_filter_df_by_regime`)의 3가지 구조적 문제를 수정한다.

  [문제 1] 데이터 절단(data-cut) 레짐 필터
      BULL이 아닌 날의 봉을 데이터프레임에서 제거한 뒤 백테스트하므로,
      오버나잇 보유 시 포지션이 '제거된 날들'을 건너뛰며 보유된다.
      그 구간의 가격 경로(급락 포함)가 손절 스캔에 보이지 않는다 → 낙관 편향.
      또한 피봇/ATR/ADX가 불연속 데이터 위에서 계산되어 실거래(연속 데이터에서
      피봇 탐지)와 신호 자체가 달라진다.
      → 수정: 전체 연속 데이터에서 피봇 탐지·백테스트를 수행하고,
        레짐은 '진입 게이트'로만 사용한다. 보유 중 레짐이 꺾이면
        명시적 청산 규칙(익일 첫 봉 시가)을 적용한다.

  [문제 2] 스톱 체결가 낙관 편향
      기존 backtest()는 갭 하락으로 시가가 스톱 아래에서 열려도
      스톱 가격 그대로 체결시킨다.
      → 수정: 롱 기준 exit_px = min(스톱, 해당 봉 시가). 오버나잇 갭 반영.

  [문제 3] 연도 단위 루프
      매년 데이터를 따로 로드해 피봇/지표/포지션이 1월마다 리셋된다.
      → 수정: 전체 기간 1회 로드, 지표 1회 계산, 신호 1회 계산 후 슬라이스.

추가로 exit_mode 를 노출한다.
  - 'next_entry'     : 기존 재현 — 다음 '롱 진입 이벤트'에서 청산
                       (long_only 에서 고점 피봇이 청산에 쓰이지 않던 기존 동작)
  - 'next_any_pivot' : 다음 확정 피봇(고점/저점 무관)의 익봉 시가 청산
                       (본래 의도된 '피봇 반전 청산')

사용:
    python pivot_bull_gated.py            # 비교 리포트 전체 실행
    python pivot_bull_gated.py --quick    # 게이팅 방식만 실행
"""
import os
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd

import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg
import pivot_regime_optimizer as pro

# ──────────────────────────────────────────────────────────────────────────────
# 경로 / 상수
# ──────────────────────────────────────────────────────────────────────────────
_CANDIDATE_DB = [
    os.environ.get("SKYP_DB", ""),
    str(Path(__file__).parent.parent / "Devcenter" / "duckdb" / "market_data.duckdb"),
    str(Path(__file__).parent.parent / "duckdb" / "market_data.duckdb"),
    "c:/Project/SkyPredictor v1/Devcenter/duckdb/market_data.duckdb",
]
DB_PATH = next((p for p in _CANDIDATE_DB if p and Path(p).exists()), _CANDIDATE_DB[-1])

TRAIN_YEARS = (2019, 2025)   # inclusive
TEST_YEARS = (2026, 2026)

# 2019-2025 train에서 도출된 BULL 최적 파라미터 (확정 지연 단축 연구 적용)
BULL_PARAMS = {
    "base_pct": 1.272989526401749,
    "base_multiplier": 1.3341908735602903,
    "atr_weight": 0.20831334967633547,
    "confirmation_bars": 2,  # 확정 지연 단축 연구: 1 → 2로 변경 (포획률 57.0% → 61.3% 개선)
    "min_wave_pct": 0.07699392762885474,
    "min_pivot_interval_bars": 28,
}

BT = pv.BacktestConfig(
    multiplier=250_000,
    commission_pct_per_side=0.00003,
    slippage_ticks_per_side=1.0,
    tick_size=0.05,
    entry_on="next_open",
    annualization=252.0,
    intraday_only=True,
    session_boundary_hour=8,
    direction_mode="long_only",
)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드 — 전체 기간 1회
# ──────────────────────────────────────────────────────────────────────────────
def load_full_data() -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("SELECT * FROM futures_5min ORDER BY timestamp").df()
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.columns = df.columns.str.upper()
    df = pv.filter_day_session(df, start="08:45", end="15:45")
    df = pv.compute_indicators(df)          # 지표 전체 1회 계산
    return df


def build_regime_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """봉 단위 레짐 배열 + 거래일 첫/마지막 봉 위치 배열.

    daily_regime_signal 내부에서 shift(1) 적용 → D일 레짐은 D-1 종가까지 정보만 사용.
    """
    daily = rg.to_daily(df, BT.session_boundary_hour)
    signal = rg.daily_regime_signal(daily, regime_method="ma", ma_short=20, ma_long=60)

    n = len(df)
    tday = pv.trading_day_key(df.index, BT.session_boundary_hour)
    pos = pd.Series(np.arange(n))
    grp = pos.groupby(tday)
    day_first = grp.transform("min").to_numpy()
    day_last = grp.transform("max").to_numpy()

    # 거래일별 마지막 봉 시각 → 일봉 end_time 과 매칭해 레짐 매핑
    last_idx = grp.max().to_numpy()
    end_times = df.index[last_idx]
    day_regime = signal.reindex(end_times).fillna(0).astype(int).to_numpy()

    regime_per_bar = np.zeros(n, dtype=int)
    first_idx = grp.min().to_numpy()
    for f, l, r in zip(first_idx, last_idx, day_regime):
        regime_per_bar[f:l + 1] = r
    return regime_per_bar, day_first, day_last


# ──────────────────────────────────────────────────────────────────────────────
# 게이팅 백테스트 엔진
# ──────────────────────────────────────────────────────────────────────────────
def gated_backtest(
    df: pd.DataFrame,
    pivots: pd.DataFrame,
    regime_per_bar: np.ndarray,
    day_first: np.ndarray,
    day_last: np.ndarray,
    cfg: pv.BacktestConfig,
    hold_mode: str = "intraday",          # 'intraday' | 'overnight'
    exit_mode: str = "next_entry",        # 'next_entry' | 'next_any_pivot'
    stop_atr_mult: float = 0.0,           # 진입봉 ATR × k 손절 (0=미사용)
    trail_atr_mult: float = 0.0,          # 고점 대비 ATR × k 트레일링 (0=미사용)
    gap_aware: bool = True,
) -> pv.BacktestResult:
    """전체 연속 데이터 위에서 BULL 게이트 롱 백테스트.

    청산 우선순위(시간순 최선착):
      ① 손절/트레일링(봉 내부 스캔, 갭 반영)
      ② exit_mode 에 따른 피봇 청산
      ③ hold_mode='intraday' → 당일 마지막 봉 종가 (eod)
         hold_mode='overnight' → 레짐이 BULL 이탈한 첫 거래일 첫 봉 시가 (regime)
    """
    empty = pv.BacktestResult(trades=pd.DataFrame())
    if pivots is None or len(pivots) == 0:
        return empty

    n = len(df)
    idx = df.index
    px_open = df["OPEN"].to_numpy() if "OPEN" in df.columns else df["CLOSE"].to_numpy()
    px_high = df["HIGH"].to_numpy()
    px_low = df["LOW"].to_numpy()
    px_close = df["CLOSE"].to_numpy()
    atr = df["ATR"].to_numpy() if "ATR" in df.columns else np.full(n, np.nan)

    piv = pivots.sort_values("confirm_pos").reset_index(drop=True)
    all_confirm = piv["confirm_pos"].to_numpy(dtype=int)
    is_high = piv["is_high"].to_numpy(dtype=bool)

    # 롱 진입 이벤트: 저점 피봇 확정 + 익봉 존재 + 진입봉 레짐 BULL
    long_entries: List[int] = []
    for cpos, hi in zip(all_confirm, is_high):
        if hi:
            continue
        epos = cpos + 1
        if epos >= n:
            continue
        if regime_per_bar[epos] != 1:
            continue
        if hold_mode == "intraday" and epos >= int(day_last[epos]):
            continue  # 당일 청산 불가
        long_entries.append(epos)

    if not long_entries:
        return empty

    # 레짐 이탈 거래일의 '첫 봉 위치' 목록 (overnight 청산용)
    nonbull_day_starts = np.unique(day_first[regime_per_bar != 1]) if hold_mode == "overnight" else np.array([])

    # exit_mode 별 예정 청산 후보 위치
    any_pivot_exit_pos = all_confirm + 1  # 확정 익봉 시가 청산

    rows = []
    prev_exit_pos = -1
    for k, e_pos in enumerate(long_entries):
        if e_pos <= prev_exit_pos:
            continue  # 포지션 중복 방지 (조기청산 후 재진입은 허용)

        e_px = px_open[e_pos]

        # ── 예정 청산 위치 결정 ────────────────────────────────────────
        cands: List[Tuple[int, float, str]] = []   # (pos, px, reason)

        if exit_mode == "next_entry":
            nxt = next((p for p in long_entries[k + 1:] if p > e_pos), None)
            if nxt is not None:
                cands.append((nxt, px_open[nxt], "pivot"))
        else:  # next_any_pivot
            later = any_pivot_exit_pos[(any_pivot_exit_pos > e_pos) & (any_pivot_exit_pos < n)]
            if len(later):
                xp = int(later[0])
                cands.append((xp, px_open[xp], "pivot"))

        if hold_mode == "intraday":
            eod = int(day_last[e_pos])
            cands.append((eod, px_close[eod], "eod"))
        else:
            j = np.searchsorted(nonbull_day_starts, e_pos, side="right")
            if j < len(nonbull_day_starts):
                rp = int(nonbull_day_starts[j])
                cands.append((rp, px_open[rp], "regime"))

        cands.append((n - 1, px_close[n - 1], "final"))
        x_pos, x_px, reason = min(cands, key=lambda t: t[0])
        if x_pos <= e_pos:
            prev_exit_pos = e_pos
            continue

        # ── 손절/트레일링 스캔 (전체 연속 봉, 갭 반영) ─────────────────
        exit_pos, exit_px, exit_reason = x_pos, x_px, reason
        a = float(atr[e_pos]) if not math.isnan(atr[e_pos]) else 0.0
        sl_price = e_px - stop_atr_mult * a if (stop_atr_mult > 0 and a > 0) else 0.0
        use_trail = trail_atr_mult > 0 and a > 0

        if sl_price > 0 or use_trail:
            hwm = e_px
            for i in range(e_pos + 1, x_pos + 1):
                bar_open = px_open[i]
                eff_sl = sl_price
                if use_trail:
                    trail = hwm - trail_atr_mult * a
                    eff_sl = max(eff_sl, trail)
                if eff_sl > 0:
                    if gap_aware and bar_open <= eff_sl:
                        exit_pos, exit_px = i, bar_open          # 갭 오픈 체결
                        exit_reason = "stop_gap"
                        break
                    if px_low[i] <= eff_sl:
                        exit_pos, exit_px = i, eff_sl
                        exit_reason = "stop" if (sl_price > 0 and abs(eff_sl - sl_price) < 1e-9) else "trail"
                        break
                if px_high[i] > hwm:
                    hwm = px_high[i]

        gross_pts = exit_px - e_px
        cost_pts = cfg.round_trip_cost_pts(e_px, exit_px)
        net_krw = (gross_pts - cost_pts) * cfg.multiplier
        rows.append({
            "entry_time": idx[e_pos], "exit_time": idx[exit_pos],
            "direction": 1, "entry_px": e_px, "exit_px": exit_px,
            "exit_reason": exit_reason,
            "entry_atr": a,
            "gross_pts": gross_pts, "cost_pts": cost_pts,
            "net_pts": gross_pts - cost_pts, "net_krw": net_krw,
            "entry_pos": e_pos, "exit_pos": exit_pos,
        })
        prev_exit_pos = exit_pos

    if not rows:
        return empty

    tdf = pd.DataFrame(rows)
    net = tdf["net_pts"]
    wins, losses = net[net > 0], net[net < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())

    tdf["exit_date"] = pd.to_datetime(tdf["exit_time"]).dt.date
    daily = tdf.groupby("exit_date")["net_krw"].sum()
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(cfg.annualization)) \
        if len(daily) >= 2 and daily.std(ddof=1) > 0 else 0.0
    equity = tdf["net_krw"].cumsum()
    max_dd = float((equity - equity.cummax()).min())

    return pv.BacktestResult(
        n_trades=len(tdf),
        win_rate=float((net > 0).mean() * 100),
        total_pnl_pts=float(net.sum()),
        total_pnl_krw=float(tdf["net_krw"].sum()),
        expectancy_pts=float(net.mean()),
        expectancy_krw=float(tdf["net_krw"].mean()),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        sharpe_daily=sharpe,
        max_drawdown_krw=max_dd,
        trades=tdf,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 실행 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def slice_years(res: pv.BacktestResult, y0: int, y1: int, cfg: pv.BacktestConfig) -> Dict:
    """거래 로그를 연도 범위로 슬라이스해 지표 재산출."""
    if res.trades is None or len(res.trades) == 0:
        return dict(n=0, wr=0.0, pnl=0.0, sharpe=0.0, mdd=0.0, pf=0.0)
    t = res.trades[(pd.to_datetime(res.trades["entry_time"]).dt.year >= y0)
                   & (pd.to_datetime(res.trades["entry_time"]).dt.year <= y1)]
    if len(t) == 0:
        return dict(n=0, wr=0.0, pnl=0.0, sharpe=0.0, mdd=0.0, pf=0.0)
    net = t["net_pts"]
    daily = t.groupby(pd.to_datetime(t["exit_time"]).dt.date)["net_krw"].sum()
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(cfg.annualization)) \
        if len(daily) >= 2 and daily.std(ddof=1) > 0 else 0.0
    eq = t["net_krw"].cumsum()
    gw, gl = float(net[net > 0].sum()), float(-net[net < 0].sum())
    return dict(
        n=len(t), wr=float((net > 0).mean() * 100), pnl=float(t["net_krw"].sum()),
        sharpe=sharpe, mdd=float((eq - eq.cummax()).min()),
        pf=float(gw / gl) if gl > 0 else float("inf"),
    )


def fmt_row(label: str, m: Dict) -> str:
    return (f"{label:<44} | 거래={m['n']:>4} | 승률={m['wr']:>6.2f}% | "
            f"PnL={m['pnl']:>13,.0f} | PF={m['pf']:>5.2f} | "
            f"Sharpe={m['sharpe']:>6.3f} | MaxDD={m['mdd']:>13,.0f}")


def yearly_table(res: pv.BacktestResult) -> str:
    if res.trades is None or len(res.trades) == 0:
        return "  (거래 없음)"
    t = res.trades.copy()
    t["year"] = pd.to_datetime(t["entry_time"]).dt.year
    lines = []
    for y, g in t.groupby("year"):
        net = g["net_pts"]
        lines.append(f"  {y}: 거래={len(g):>4} | 승률={(net > 0).mean() * 100:>6.2f}% | "
                     f"PnL={g['net_krw'].sum():>+13,.0f}")
    return "\n".join(lines)


def exit_reason_table(res: pv.BacktestResult) -> str:
    if res.trades is None or len(res.trades) == 0:
        return "  (거래 없음)"
    g = res.trades.groupby("exit_reason").agg(
        n=("net_krw", "size"), pnl=("net_krw", "sum"), avg_pts=("net_pts", "mean"))
    return "\n".join(f"  {r:<10}: n={int(v['n']):>4} | PnL={v['pnl']:>+13,.0f} | 평균={v['avg_pts']:>+7.3f}pt"
                     for r, v in g.iterrows())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="게이팅 방식만 실행")
    args = ap.parse_args()

    print(f"DB: {DB_PATH}")
    df = load_full_data()
    print(f"데이터: {len(df):,}행  {df.index[0]} ~ {df.index[-1]}")

    regime_per_bar, day_first, day_last = build_regime_arrays(df)
    print(f"BULL 봉 비율: {(regime_per_bar == 1).mean() * 100:.1f}%")

    pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=BULL_PARAMS["base_pct"],
        base_multiplier=BULL_PARAMS["base_multiplier"],
        atr_weight=BULL_PARAMS["atr_weight"],
        confirmation_bars=BULL_PARAMS["confirmation_bars"],
    )
    fcfg = pv.FilterConfig(
        enabled=True,
        min_wave_pct=BULL_PARAMS["min_wave_pct"],
        min_pivot_interval_bars=BULL_PARAMS["min_pivot_interval_bars"],
        st_distance_threshold=0.1,
        adx_hold_threshold=15.0,
    )

    # 전체 연속 데이터에서 피봇 1회 검출 (거래일별 검출기 리셋 = 기존과 동일 규약)
    pivots = pv.detect_pivots_daily(df, pcfg, fcfg, BT.session_boundary_hour)
    print(f"확정 피봇: {len(pivots)}건 (저점 {int((~pivots['is_high']).sum())} / 고점 {int(pivots['is_high'].sum())})")

    results: Dict[str, pv.BacktestResult] = {}

    # ── A. 기존 방식 재현 (데이터 절단) ─────────────────────────────────
    if not args.quick:
        df_bull = pro._filter_df_by_regime(df, _full_regime_signal(df), 1)
        piv_cut = pv.detect_pivots_daily(df_bull, pcfg, fcfg, BT.session_boundary_hour)

        bt_i = _clone_bt(intraday=True)
        results["A1. 기존(데이터절단) 인트라데이"] = pv.backtest(df_bull, piv_cut, bt_i)
        bt_o = _clone_bt(intraday=False)
        results["A2. 기존(데이터절단) 오버나잇"] = pv.backtest(df_bull, piv_cut, bt_o)

    # ── B. 게이팅 방식 (수정판) ─────────────────────────────────────────
    results["B1. 게이팅 인트라데이 (exit=next_entry)"] = gated_backtest(
        df, pivots, regime_per_bar, day_first, day_last, BT,
        hold_mode="intraday", exit_mode="next_entry")
    results["B2. 게이팅 인트라데이 (exit=next_any_pivot)"] = gated_backtest(
        df, pivots, regime_per_bar, day_first, day_last, BT,
        hold_mode="intraday", exit_mode="next_any_pivot")
    results["B3. 게이팅 오버나잇 (exit=next_entry+레짐청산)"] = gated_backtest(
        df, pivots, regime_per_bar, day_first, day_last, BT,
        hold_mode="overnight", exit_mode="next_entry")
    results["B4. 게이팅 오버나잇 (exit=next_any_pivot+레짐청산)"] = gated_backtest(
        df, pivots, regime_per_bar, day_first, day_last, BT,
        hold_mode="overnight", exit_mode="next_any_pivot")

    # ── 리포트 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print(f"{'전략':<44} | 구간별 성과 (train 2019-2025 / test 2026 / 전체)")
    print("=" * 120)
    for name, res in results.items():
        print(f"\n[{name}]")
        print(fmt_row("  train 2019-2025", slice_years(res, *TRAIN_YEARS, BT)))
        print(fmt_row("  test  2026", slice_years(res, *TEST_YEARS, BT)))
        print(fmt_row("  전체  2019-2026", slice_years(res, 2019, 2026, BT)))
        print("  연도별:")
        print(yearly_table(res))
        print("  청산 사유:")
        print(exit_reason_table(res))

    # 거래 로그 저장 (MFE/MAE 분석 입력)
    out_dir = Path(__file__).parent / "data" / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, fname in [
        ("B1. 게이팅 인트라데이 (exit=next_entry)", "gated_intraday_next_entry_trades.csv"),
        ("B2. 게이팅 인트라데이 (exit=next_any_pivot)", "gated_intraday_any_pivot_trades.csv"),
        ("B3. 게이팅 오버나잇 (exit=next_entry+레짐청산)", "gated_overnight_next_entry_trades.csv"),
        ("B4. 게이팅 오버나잇 (exit=next_any_pivot+레짐청산)", "gated_overnight_any_pivot_trades.csv"),
    ]:
        r = results.get(key)
        if r is not None and r.trades is not None and len(r.trades):
            r.trades.to_csv(out_dir / fname, index=False)
    print(f"\n거래 로그 저장: {out_dir}")


def _full_regime_signal(df: pd.DataFrame) -> pd.Series:
    daily = rg.to_daily(df, BT.session_boundary_hour)
    return rg.daily_regime_signal(daily, regime_method="ma", ma_short=20, ma_long=60)


def _clone_bt(intraday: bool) -> pv.BacktestConfig:
    b = pv.BacktestConfig(
        multiplier=250_000, commission_pct_per_side=0.00003,
        slippage_ticks_per_side=1.0, tick_size=0.05,
        entry_on="next_open", annualization=252.0,
        intraday_only=intraday, session_boundary_hour=8,
        direction_mode="long_only",
    )
    return b


if __name__ == "__main__":
    main()
