# -*- coding: utf-8 -*-
"""
pivot_mfe_mae_analysis.py — MFE/MAE 분석 및 손절/트레일링 설계 (train 전용)

목적
  손익비(payoff) 0.78 문제의 원인을 데이터로 진단하고, 손절/트레일링
  파라미터를 '추측'이 아니라 MFE/MAE 분포에서 설계한다.

원칙 (RECOMMENDED_STRATEGY.md 검증 게이트 준수)
  - 모든 분포 분석과 그리드 탐색은 train(2019-2025)에서만 수행한다.
  - test(2026)는 --confirm-test 로 '최종 후보 1개'를 단 1회만 평가한다.
    (grid 결과를 보고 test를 여러 번 돌리면 OOS 오염 — 절대 금지)

용어
  MFE (Maximum Favorable Excursion) : 보유 중 진입가 대비 최대 유리 이동(pt)
  MAE (Maximum Adverse Excursion)   : 보유 중 진입가 대비 최대 불리 이동(pt)
  - 손절이 '승리 거래의 MAE 분포' 안쪽에 있으면 승자를 죽인다.
  - 트레일링/익절이 '승리 거래의 MFE 분포' 대비 너무 이르면 이익을 깎는다.

사용
  python pivot_mfe_mae_analysis.py                       # 분석 + train 그리드
  python pivot_mfe_mae_analysis.py --mode overnight      # 오버나잇 기준
  python pivot_mfe_mae_analysis.py --confirm-test --stop 1.5 --trail 3.0
"""
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd

import pivot_optuna_v2 as pv
import pivot_bull_gated as gb

TRAIN = (2019, 2025)
TEST = (2026, 2026)

PCTS = [10, 25, 50, 75, 90, 95]


# ──────────────────────────────────────────────────────────────────────────────
def prepare(mode: str):
    """전체 데이터 + 피봇 + 레짐 배열 1회 준비."""
    df = gb.load_full_data()
    regime_per_bar, day_first, day_last = gb.build_regime_arrays(df)

    pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=gb.BULL_PARAMS["base_pct"],
        base_multiplier=gb.BULL_PARAMS["base_multiplier"],
        atr_weight=gb.BULL_PARAMS["atr_weight"],
        confirmation_bars=gb.BULL_PARAMS["confirmation_bars"],
    )
    fcfg = pv.FilterConfig(
        enabled=True,
        min_wave_pct=gb.BULL_PARAMS["min_wave_pct"],
        min_pivot_interval_bars=gb.BULL_PARAMS["min_pivot_interval_bars"],
        st_distance_threshold=0.1,
        adx_hold_threshold=15.0,
    )
    pivots = pv.detect_pivots_daily(df, pcfg, fcfg, gb.BT.session_boundary_hour)
    return df, pivots, regime_per_bar, day_first, day_last


def run(df, pivots, regime, dfirst, dlast, mode: str,
        stop: float = 0.0, trail: float = 0.0) -> pv.BacktestResult:
    return gb.gated_backtest(
        df, pivots, regime, dfirst, dlast, gb.BT,
        hold_mode=mode, exit_mode="next_entry",
        stop_atr_mult=stop, trail_atr_mult=trail, gap_aware=True,
    )


def add_mfe_mae(trades: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """각 거래의 MFE/MAE(pt 및 ATR 배수)를 계산해 컬럼 추가."""
    high = df["HIGH"].to_numpy()
    low = df["LOW"].to_numpy()
    t = trades.copy()
    mfe, mae = [], []
    for e, x, px in zip(t["entry_pos"].astype(int), t["exit_pos"].astype(int), t["entry_px"]):
        seg_h = high[e + 1: x + 1]
        seg_l = low[e + 1: x + 1]
        mfe.append(float(seg_h.max() - px) if len(seg_h) else 0.0)
        mae.append(float(px - seg_l.min()) if len(seg_l) else 0.0)
    t["mfe_pts"] = mfe
    t["mae_pts"] = mae
    a = t["entry_atr"].replace(0, np.nan)
    t["mfe_atr"] = t["mfe_pts"] / a
    t["mae_atr"] = t["mae_pts"] / a
    return t


def _pct_table(s: pd.Series) -> str:
    if len(s) == 0:
        return "  (없음)"
    q = np.percentile(s.dropna(), PCTS)
    return " | ".join(f"p{p}={v:>6.2f}" for p, v in zip(PCTS, q))


def in_years(t: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    yr = pd.to_datetime(t["entry_time"]).dt.year
    return t[(yr >= y0) & (yr <= y1)]


def print_mfe_mae_report(t: pd.DataFrame, label: str):
    win = t[t["net_pts"] > 0]
    loss = t[t["net_pts"] < 0]
    print(f"\n── MFE/MAE 분포 [{label}] (train {TRAIN[0]}-{TRAIN[1]}, n={len(t)}: 승 {len(win)} / 패 {len(loss)})")
    print(f"  승자 MAE (pt) : {_pct_table(win['mae_pts'])}")
    print(f"  승자 MAE (ATR): {_pct_table(win['mae_atr'])}")
    print(f"  패자 MAE (pt) : {_pct_table(loss['mae_pts'])}")
    print(f"  패자 MAE (ATR): {_pct_table(loss['mae_atr'])}")
    print(f"  승자 MFE (pt) : {_pct_table(win['mfe_pts'])}")
    print(f"  승자 MFE (ATR): {_pct_table(win['mfe_atr'])}")
    print(f"  패자 MFE (pt) : {_pct_table(loss['mfe_pts'])}")
    print(f"  패자 MFE (ATR): {_pct_table(loss['mfe_atr'])}")
    payoff = win["net_pts"].mean() / abs(loss["net_pts"].mean()) if len(win) and len(loss) else float("nan")
    print(f"  현재 payoff(평균이익/평균손실) = {payoff:.3f} | "
          f"평균이익 {win['net_pts'].mean():+.2f}pt / 평균손실 {loss['net_pts'].mean():+.2f}pt")
    # 설계 힌트
    if len(win):
        print(f"  [힌트] 승자를 90% 보존하려면 손절 ≳ 승자 MAE p90 "
              f"= {np.percentile(win['mae_atr'].dropna(), 90):.2f} ATR")
    if len(loss):
        print(f"  [힌트] 패자 MFE p50 = {np.percentile(loss['mfe_pts'].dropna(), 50):.2f}pt "
              f"→ 이 이상 부분익절/BE 스톱 검토 여지")


def metrics_of(t: pd.DataFrame) -> Dict:
    if len(t) == 0:
        return dict(n=0, wr=0, pnl=0, exp=0, pf=0, sharpe=0, mdd=0)
    net = t["net_pts"]
    gw, gl = float(net[net > 0].sum()), float(-net[net < 0].sum())
    daily = t.groupby(pd.to_datetime(t["exit_time"]).dt.date)["net_krw"].sum()
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(252)) \
        if len(daily) >= 2 and daily.std(ddof=1) > 0 else 0.0
    eq = t["net_krw"].cumsum()
    return dict(
        n=len(t), wr=float((net > 0).mean() * 100), pnl=float(t["net_krw"].sum()),
        exp=float(net.mean()), pf=float(gw / gl) if gl > 0 else float("inf"),
        sharpe=sharpe, mdd=float((eq - eq.cummax()).min()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["intraday", "overnight"], default="intraday")
    ap.add_argument("--confirm-test", action="store_true",
                    help="선택 확정된 stop/trail 로 test(2026) 1회 평가")
    ap.add_argument("--stop", type=float, default=0.0)
    ap.add_argument("--trail", type=float, default=0.0)
    args = ap.parse_args()

    df, pivots, regime, dfirst, dlast = prepare(args.mode)

    # ── test 1회 확인 모드 ──────────────────────────────────────────────
    if args.confirm_test:
        res = run(df, pivots, regime, dfirst, dlast, args.mode, args.stop, args.trail)
        t = add_mfe_mae(res.trades, df)
        for label, (y0, y1) in [("train", TRAIN), ("TEST(1회 확인)", TEST)]:
            m = metrics_of(in_years(t, y0, y1))
            print(f"[{label}] stop={args.stop} trail={args.trail} | 거래={m['n']} | "
                  f"승률={m['wr']:.2f}% | PnL={m['pnl']:+,.0f} | 기대값={m['exp']:+.3f}pt | "
                  f"PF={m['pf']:.2f} | Sharpe={m['sharpe']:.3f} | MaxDD={m['mdd']:+,.0f}")
        print("※ 이 결과가 마음에 안 든다고 다른 stop/trail 로 test 를 다시 돌리면 OOS 오염입니다.")
        return

    # ── 1) 베이스라인(스톱 없음) MFE/MAE 분포 — train만 ───────────────
    base = run(df, pivots, regime, dfirst, dlast, args.mode)
    t_all = add_mfe_mae(base.trades, df)
    t_train = in_years(t_all, *TRAIN)
    print(f"\n{'=' * 100}\nMFE/MAE 진단 — hold_mode={args.mode}, exit=next_entry (베이스라인, 스톱 없음)")
    print_mfe_mae_report(t_train, f"{args.mode} 베이스라인")

    # 청산사유별 (참고)
    print("\n  청산 사유별 (train):")
    for r, g in t_train.groupby("exit_reason"):
        print(f"    {r:<8}: n={len(g):>4} | 평균 {g['net_pts'].mean():+.3f}pt | "
              f"MFE중앙값 {g['mfe_pts'].median():.2f}pt | MAE중앙값 {g['mae_pts'].median():.2f}pt")

    # ── 2) 손절/트레일링 그리드 — train만 ─────────────────────────────
    stops = [0.0, 0.75, 1.0, 1.5, 2.0, 3.0]
    trails = [0.0, 1.0, 1.5, 2.0, 3.0]
    print(f"\n{'=' * 100}")
    print(f"손절/트레일링 그리드 (train {TRAIN[0]}-{TRAIN[1]} 전용, 단위: 진입봉 ATR 배수)")
    print(f"{'stop':>6} {'trail':>6} | {'거래':>4} {'승률%':>7} {'기대값pt':>9} {'PF':>6} "
          f"{'Sharpe':>7} {'PnL(원)':>14} {'MaxDD(원)':>14}")
    print("-" * 100)
    grid_rows = []
    for s in stops:
        for tr in trails:
            res = run(df, pivots, regime, dfirst, dlast, args.mode, s, tr)
            if res.trades is None or len(res.trades) == 0:
                continue
            m = metrics_of(in_years(res.trades, *TRAIN))
            grid_rows.append(dict(stop=s, trail=tr, **m))
            print(f"{s:>6.2f} {tr:>6.2f} | {m['n']:>4} {m['wr']:>7.2f} {m['exp']:>9.3f} "
                  f"{m['pf']:>6.2f} {m['sharpe']:>7.3f} {m['pnl']:>14,.0f} {m['mdd']:>14,.0f}")

    gdf = pd.DataFrame(grid_rows)
    out = Path(__file__).parent / "data" / "backtest_results" / f"mfe_mae_grid_{args.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_csv(out, index=False)
    t_all.to_csv(out.parent / f"mfe_mae_trades_{args.mode}.csv", index=False)

    # 요약: 베이스라인 대비 개선 후보 (기대값·PF·MaxDD 종합)
    base_m = metrics_of(t_train)
    print(f"\n베이스라인(train): 기대값={base_m['exp']:+.3f}pt | PF={base_m['pf']:.2f} | "
          f"Sharpe={base_m['sharpe']:.3f} | MaxDD={base_m['mdd']:+,.0f}")
    cand = gdf[(gdf["exp"] >= base_m["exp"]) & (gdf["mdd"] >= base_m["mdd"])]
    cand = cand.sort_values("sharpe", ascending=False).head(5)
    print("\n[train 기준 후보 상위 5 — 기대값 유지 이상 & MaxDD 개선]")
    print(cand.to_string(index=False) if len(cand) else "  (베이스라인을 이기는 조합 없음 → 스톱 미사용이 답일 수 있음)")
    print("\n※ 후보 중 '하나'를 고른 뒤 --confirm-test 로 test(2026)를 단 1회만 확인하세요.")
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
