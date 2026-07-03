# -*- coding: utf-8 -*-
"""
confirmation_delay_analysis.py — 확정 지연 단축 연구 (train 전용)

목적
  train에서 하루 상승분 2.15pt 중 0.18pt만 포획하는 문제를 해결하기 위해
  confirmation_bars 파라미터의 지연-정확도 트레이드오프를 탐색한다.

원칙 (RECOMMENDED_STRATEGY.md 검증 게이트 준수)
  - 모든 그리드 탐색은 train(2019-2025)에서만 수행한다.
  - test(2026)는 --confirm-test 로 '최종 후보 1개'를 단 1회만 평가한다.
    (grid 결과를 보고 test를 여러 번 돌리면 OOS 오염 — 절대 금지)

용어
  confirmation_bars: 피봇 확정에 필요한 연속 봉 수
  - 값이 작을수록 빠른 확정 (지연 감소, 정확도 감소)
  - 값이 클수록 느린 확정 (지연 증가, 정확도 증가)

사용
  python confirmation_delay_analysis.py                       # 분석 + train 그리드
  python confirmation_delay_analysis.py --confirm-test --bars 1
"""
import sys
import math
import argparse
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd

import pivot_optuna_v2 as pv
import pivot_bull_gated as gb

TRAIN = (2019, 2025)
TEST = (2026, 2026)


# ──────────────────────────────────────────────────────────────────────────────
def prepare():
    """전체 데이터 + 레짐 배열 1회 준비."""
    df = gb.load_full_data()
    regime_per_bar, day_first, day_last = gb.build_regime_arrays(df)
    return df, regime_per_bar, day_first, day_last


def run(df, regime, dfirst, dlast, confirmation_bars: int) -> pv.BacktestResult:
    """confirmation_bars를 변경하여 백테스트 실행."""
    pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=gb.BULL_PARAMS["base_pct"],
        base_multiplier=gb.BULL_PARAMS["base_multiplier"],
        atr_weight=gb.BULL_PARAMS["atr_weight"],
        confirmation_bars=confirmation_bars,
    )
    fcfg = pv.FilterConfig(
        enabled=True,
        min_wave_pct=gb.BULL_PARAMS["min_wave_pct"],
        min_pivot_interval_bars=gb.BULL_PARAMS["min_pivot_interval_bars"],
        st_distance_threshold=0.1,
        adx_hold_threshold=15.0,
    )
    pivots = pv.detect_pivots_daily(df, pcfg, fcfg, gb.BT.session_boundary_hour)
    return gb.gated_backtest(
        df, pivots, regime, dfirst, dlast, gb.BT,
        hold_mode="intraday", exit_mode="next_entry",
        stop_atr_mult=0.0, trail_atr_mult=0.0, gap_aware=True,
    )


def in_years(t: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    yr = pd.to_datetime(t["entry_time"]).dt.year
    return t[(yr >= y0) & (yr <= y1)]


def metrics_of(t: pd.DataFrame) -> Dict:
    if len(t) == 0:
        return dict(n=0, wr=0, pnl=0, exp=0, pf=0, sharpe=0, mdd=0, avg_capture=0)
    net = t["net_pts"]
    gw, gl = float(net[net > 0].sum()), float(-net[net < 0].sum())
    daily = t.groupby(pd.to_datetime(t["exit_time"]).dt.date)["net_krw"].sum()
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(252)) \
        if len(daily) >= 2 and daily.std(ddof=1) > 0 else 0.0
    eq = t["net_krw"].cumsum()
    
    # 승리 거래의 평균 포획률 계산 (MFE 대비 실제 수익)
    win = t[t["net_pts"] > 0]
    if len(win) > 0 and "mfe_pts" in win.columns:
        avg_capture = (win["net_pts"] / win["mfe_pts"]).replace([np.inf, -np.inf], np.nan).mean() * 100
    else:
        avg_capture = 0.0
    
    return dict(
        n=len(t), wr=float((net > 0).mean() * 100), pnl=float(t["net_krw"].sum()),
        exp=float(net.mean()), pf=float(gw / gl) if gl > 0 else float("inf"),
        sharpe=sharpe, mdd=float((eq - eq.cummax()).min()), avg_capture=avg_capture,
    )


def add_mfe(trades: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """각 거래의 MFE(pt)를 계산해 컬럼 추가."""
    high = df["HIGH"].to_numpy()
    t = trades.copy()
    mfe = []
    for e, x, px in zip(t["entry_pos"].astype(int), t["exit_pos"].astype(int), t["entry_px"]):
        seg_h = high[e + 1: x + 1]
        mfe.append(float(seg_h.max() - px) if len(seg_h) else 0.0)
    t["mfe_pts"] = mfe
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-test", action="store_true",
                    help="선택 확정된 confirmation_bars 로 test(2026) 1회 평가")
    ap.add_argument("--bars", type=int, default=1)
    args = ap.parse_args()

    df, regime, dfirst, dlast = prepare()

    # ── test 1회 확인 모드 ──────────────────────────────────────────────
    if args.confirm_test:
        res = run(df, regime, dfirst, dlast, args.bars)
        if res.trades is None or len(res.trades) == 0:
            print("거래 없음")
            return
        t = add_mfe(res.trades, df)
        for label, (y0, y1) in [("train", TRAIN), ("TEST(1회 확인)", TEST)]:
            m = metrics_of(in_years(t, y0, y1))
            print(f"[{label}] confirmation_bars={args.bars} | 거래={m['n']} | "
                  f"승률={m['wr']:.2f}% | PnL={m['pnl']:+,.0f} | 기대값={m['exp']:+.3f}pt | "
                  f"PF={m['pf']:.2f} | Sharpe={m['sharpe']:.3f} | MaxDD={m['mdd']:+,.0f} | "
                  f"평균 포획률={m['avg_capture']:.1f}%")
        print("※ 이 결과가 마음에 안 든다고 다른 bars 로 test 를 다시 돌리면 OOS 오염입니다.")
        return

    # -- 1) 베이스라인(confirmation_bars=1) 분석 - train만
    base = run(df, regime, dfirst, dlast, 1)
    t_all = add_mfe(base.trades, df)
    t_train = in_years(t_all, *TRAIN)
    print(f"\n{'=' * 100}")
    print(f"확정 지연 분석 - 베이스라인 (confirmation_bars=1)")
    print(f"train {TRAIN[0]}-{TRAIN[1]}, n={len(t_train)}")
    
    win = t_train[t_train["net_pts"] > 0]
    if len(win) > 0:
        print(f"승리 거래 통계:")
        print(f"  평균 MFE: {win['mfe_pts'].mean():.2f}pt")
        print(f"  평균 실제 수익: {win['net_pts'].mean():.2f}pt")
        print(f"  평균 포획률: {(win['net_pts'] / win['mfe_pts']).replace([np.inf, -np.inf], np.nan).mean() * 100:.1f}%")
        print(f"  MFE 중앙값: {win['mfe_pts'].median():.2f}pt")
        print(f"  실제 수익 중앙값: {win['net_pts'].median():.2f}pt")

    # -- 2) confirmation_bars 그리드 - train만
    bars_list = [0, 1, 2, 3, 4, 5]
    print(f"\n{'=' * 100}")
    print(f"confirmation_bars 그리드 (train {TRAIN[0]}-{TRAIN[1]} 전용)")
    print(f"{'bars':>6} | {'거래':>4} {'승률%':>7} {'기대값pt':>9} {'PF':>6} "
          f"{'Sharpe':>7} {'PnL(원)':>14} {'MaxDD(원)':>14} {'평균포획률%':>10}")
    print("-" * 100)
    grid_rows = []
    for bars in bars_list:
        res = run(df, regime, dfirst, dlast, bars)
        if res.trades is None or len(res.trades) == 0:
            continue
        t = add_mfe(res.trades, df)
        m = metrics_of(in_years(t, *TRAIN))
        grid_rows.append(dict(bars=bars, **m))
        print(f"{bars:>6} | {m['n']:>4} {m['wr']:>7.2f} {m['exp']:>9.3f} "
              f"{m['pf']:>6.2f} {m['sharpe']:>7.3f} {m['pnl']:>14,.0f} {m['mdd']:>14,.0f} {m['avg_capture']:>10.1f}")

    gdf = pd.DataFrame(grid_rows)
    out = Path(__file__).parent / "data" / "backtest_results" / "confirmation_delay_grid.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_csv(out, index=False)
    t_all.to_csv(out.parent / "confirmation_delay_trades.csv", index=False)

    # 요약: 베이스라인 대비 개선 후보
    base_m = metrics_of(t_train)
    print(f"\n베이스라인(train): 기대값={base_m['exp']:+.3f}pt | PF={base_m['pf']:.2f} | "
          f"Sharpe={base_m['sharpe']:.3f} | MaxDD={base_m['mdd']:+,.0f} | 평균 포획률={base_m['avg_capture']:.1f}%")
    
    # 포획률 개선 후보 (기대값·PF·MaxDD 종합)
    cand = gdf[(gdf["avg_capture"] > base_m["avg_capture"]) & (gdf["mdd"] >= base_m["mdd"])]
    cand = cand.sort_values("sharpe", ascending=False).head(5)
    print("\n[train 기준 후보 상위 5 - 포획률 개선 & MaxDD 유지 이상]")
    print(cand.to_string(index=False) if len(cand) else "  (베이스라인을 이기는 조합 없음 -> 기존 confirmation_bars=1 유지)")
    print("\n※ 후보 중 '하나'를 고른 뒤 --confirm-test 로 test(2026)를 단 1회만 확인하세요.")
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
