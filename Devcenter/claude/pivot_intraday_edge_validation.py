# -*- coding: utf-8 -*-
"""
pivot_intraday_edge_validation.py — 인트라데이 피봇 엣지 실재성 검증 배터리

배경
  게이팅 인트라데이 피봇은 베타(BULL일 시가→종가 롱, train Sharpe -0.33)를
  크게 이긴다(train +0.18pt/거래, Sharpe 1.34). 이것이 '진짜 엣지'인지
  '노이즈/과적합'인지 판정한다.

테스트 구성
  [A] 유의성: 거래 부트스트랩 기대값 CI + 일별 PnL Sharpe SE/t-stat
  [B] 순열검정 (핵심): 실제 거래와 '같은 날', 무작위 시점 진입 → EOD 청산의
      기대값 분포 vs 실제 피봇 진입 기대값. 피봇의 "시점 선택 스킬"만 분리.
      + 무작위 BULL일 변형(일 선택 + 시점 결합).
      ※ train은 파라미터가 그 위에서 최적화되어 유의성이 부풀 수 있음.
        params는 2019-2025에서 고정되었으므로 test(2026) 순열검정이 진짜 판정.
  [C] 파라미터 섭동: 핵심 5개 파라미터를 개별 ±20% → 엣지가 부드럽게
      유지되면 실재, 특정 값에서만 살면 과적합 신호.
  [D] 비용 스트레스: 슬리피지 1→2→3틱, 수수료 0.003%→0.015%.

비교 대상 전략: 게이팅 인트라데이, exit=next_entry (사실상 전 거래 EOD 청산
  → 순열 대조군 'EOD 청산'과 청산 규칙 동일해 공정 비교)

사용
  python pivot_intraday_edge_validation.py
"""
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd

import pivot_optuna_v2 as pv
import pivot_bull_gated as gb

RNG = np.random.default_rng(42)
N_BOOT = 10_000
N_PERM = 5_000
TRAIN = (2019, 2025)
TEST = (2026, 2026)


# ──────────────────────────────────────────────────────────────────────────────
def make_cfgs(params: Dict):
    pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=params["base_pct"],
        base_multiplier=params["base_multiplier"],
        atr_weight=params["atr_weight"],
        confirmation_bars=params["confirmation_bars"],
    )
    fcfg = pv.FilterConfig(
        enabled=True,
        min_wave_pct=params["min_wave_pct"],
        min_pivot_interval_bars=params["min_pivot_interval_bars"],
        st_distance_threshold=0.1,
        adx_hold_threshold=15.0,
    )
    return pcfg, fcfg


def run_intraday(df, regime, dfirst, dlast, params: Dict, cfg=None) -> pv.BacktestResult:
    pcfg, fcfg = make_cfgs(params)
    pivots = pv.detect_pivots_daily(df, pcfg, fcfg, gb.BT.session_boundary_hour)
    return gb.gated_backtest(
        df, pivots, regime, dfirst, dlast, cfg or gb.BT,
        hold_mode="intraday", exit_mode="next_entry",
    )


def yr_slice(t: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    yr = pd.to_datetime(t["entry_time"]).dt.year
    return t[(yr >= y0) & (yr <= y1)]


def summarize(t: pd.DataFrame) -> Dict:
    if len(t) == 0:
        return dict(n=0, exp=np.nan, wr=np.nan, sharpe=np.nan, pnl=0.0)
    net = t["net_pts"]
    daily = t.groupby(pd.to_datetime(t["exit_time"]).dt.date)["net_krw"].sum()
    sharpe = daily.mean() / daily.std(ddof=1) * math.sqrt(252) \
        if len(daily) >= 2 and daily.std(ddof=1) > 0 else np.nan
    return dict(n=len(t), exp=float(net.mean()), wr=float((net > 0).mean() * 100),
                sharpe=float(sharpe), pnl=float(t["net_krw"].sum()))


# ──────────────────────────────────────────────────────────────────────────────
def test_A_significance(t: pd.DataFrame, label: str):
    net = t["net_pts"].to_numpy()
    n = len(net)
    boots = np.array([RNG.choice(net, n, replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_le0 = float((boots <= 0).mean())

    daily = t.groupby(pd.to_datetime(t["exit_time"]).dt.date)["net_krw"].sum()
    mu, sd, nd = daily.mean(), daily.std(ddof=1), len(daily)
    sharpe = mu / sd * math.sqrt(252) if sd > 0 else np.nan
    se_sharpe = math.sqrt((1 + 0.5 * (sharpe / math.sqrt(252)) ** 2 * 252) / nd) \
        if sd > 0 else np.nan  # Lo(2002) 근사
    tstat = mu / (sd / math.sqrt(nd)) if sd > 0 else np.nan

    print(f"\n[A] 유의성 — {label} (거래 n={n}, 거래일 {nd})")
    print(f"    기대값 = {net.mean():+.3f}pt | 부트스트랩 95% CI [{lo:+.3f}, {hi:+.3f}] | P(기대값≤0) = {p_le0:.4f}")
    print(f"    Sharpe = {sharpe:.3f} ± {se_sharpe:.3f}(SE) | 일별 PnL t-stat = {tstat:.2f}")


def test_B_permutation(t: pd.DataFrame, df: pd.DataFrame, regime, dfirst, dlast,
                       cfg, label: str):
    """실제 진입 vs 무작위 진입 순열검정. 청산은 양쪽 모두 EOD 종가."""
    op = df["OPEN"].to_numpy()
    cl = df["CLOSE"].to_numpy()
    n_bars = len(df)

    # 실제 거래를 'EOD 청산' 기준으로 재계산 (대조군과 규칙 통일)
    e_pos = t["entry_pos"].astype(int).to_numpy()
    eods = dlast[e_pos].astype(int)
    actual = np.array([
        (cl[x] - op[e]) - cfg.round_trip_cost_pts(op[e], cl[x])
        for e, x in zip(e_pos, eods)
    ])
    n_tr = len(actual)
    act_exp = actual.mean()

    # 대조군 준비: 각 거래일의 (first, last) / 전체 BULL일 목록
    day_f = dfirst[e_pos].astype(int)
    day_l = eods
    bull_days = np.unique(dfirst[(regime == 1)]).astype(int)
    bull_last = dlast[bull_days].astype(int)
    ok = bull_last > bull_days           # 청산 가능일만
    bull_days, bull_last = bull_days[ok], bull_last[ok]

    def perm_same_day():
        ent = RNG.integers(day_f, day_l)          # [first, last) 균등
        return np.mean((cl[day_l] - op[ent])
                       - np.array([cfg.round_trip_cost_pts(op[e], cl[x])
                                   for e, x in zip(ent, day_l)]))

    def perm_rand_day():
        di = RNG.integers(0, len(bull_days), n_tr)
        f, l = bull_days[di], bull_last[di]
        ent = RNG.integers(f, l)
        return np.mean((cl[l] - op[ent])
                       - np.array([cfg.round_trip_cost_pts(op[e], cl[x])
                                   for e, x in zip(ent, l)]))

    null_same = np.array([perm_same_day() for _ in range(N_PERM)])
    null_rand = np.array([perm_rand_day() for _ in range(N_PERM)])
    p_same = float((null_same >= act_exp).mean())
    p_rand = float((null_rand >= act_exp).mean())

    print(f"\n[B] 순열검정 — {label} (n={n_tr}, EOD 청산 통일, {N_PERM}회)")
    print(f"    실제 피봇 진입 기대값          = {act_exp:+.3f}pt")
    print(f"    귀무1: 같은 날 무작위 시점 진입 = {null_same.mean():+.3f}pt "
          f"(p5~p95 [{np.percentile(null_same,5):+.3f}, {np.percentile(null_same,95):+.3f}]) "
          f"→ p-value = {p_same:.4f}")
    print(f"    귀무2: 무작위 BULL일+시점 진입  = {null_rand.mean():+.3f}pt "
          f"(p5~p95 [{np.percentile(null_rand,5):+.3f}, {np.percentile(null_rand,95):+.3f}]) "
          f"→ p-value = {p_rand:.4f}")
    print(f"    해석: 귀무1 p<0.05 → '그 날 안에서 언제 사느냐'의 스킬 실재 / "
          f"귀무2 p<0.05 → 일 선택+시점 결합 스킬 실재")


def test_C_perturbation(df, regime, dfirst, dlast, base_params: Dict):
    print(f"\n[C] 파라미터 섭동 (개별 ±20%, train 기대값/Sharpe — 부드러우면 실재, 스파이크면 과적합)")
    base = run_intraday(df, regime, dfirst, dlast, base_params)
    bm = summarize(yr_slice(base.trades, *TRAIN))
    print(f"    {'baseline':<38} | n={bm['n']:>4} | 기대값={bm['exp']:+.3f}pt | Sharpe={bm['sharpe']:6.3f}")

    perturb_keys = ["base_pct", "base_multiplier", "atr_weight",
                    "min_wave_pct", "min_pivot_interval_bars"]
    for key in perturb_keys:
        for f in (0.8, 1.2):
            p = dict(base_params)
            p[key] = int(round(p[key] * f)) if key == "min_pivot_interval_bars" else p[key] * f
            res = run_intraday(df, regime, dfirst, dlast, p)
            m = summarize(yr_slice(res.trades, *TRAIN)) if res.trades is not None and len(res.trades) else dict(n=0, exp=np.nan, sharpe=np.nan)
            print(f"    {key:>24} ×{f:<4} | n={m['n']:>4} | 기대값={m['exp']:+.3f}pt | Sharpe={m['sharpe']:6.3f}")


def test_D_cost_stress(df, regime, dfirst, dlast, params: Dict):
    print(f"\n[D] 비용 스트레스 (train / test 기대값 pt)")
    combos = [(0.00003, 1.0), (0.00003, 2.0), (0.00003, 3.0),
              (0.00015, 1.0), (0.00015, 2.0)]
    pcfg, fcfg = make_cfgs(params)
    pivots = pv.detect_pivots_daily(df, pcfg, fcfg, gb.BT.session_boundary_hour)
    for comm, slip in combos:
        cfg = pv.BacktestConfig(
            multiplier=250_000, commission_pct_per_side=comm,
            slippage_ticks_per_side=slip, tick_size=0.05,
            entry_on="next_open", annualization=252.0,
            intraday_only=True, session_boundary_hour=8,
            direction_mode="long_only",
        )
        res = gb.gated_backtest(df, pivots, regime, dfirst, dlast, cfg,
                                hold_mode="intraday", exit_mode="next_entry")
        tr = summarize(yr_slice(res.trades, *TRAIN))
        te = summarize(yr_slice(res.trades, *TEST))
        print(f"    수수료 {comm*100:.3f}% + 슬리피지 {slip:.0f}틱 | "
              f"train {tr['exp']:+.3f}pt (Sharpe {tr['sharpe']:5.2f}) | "
              f"test {te['exp']:+.3f}pt (Sharpe {te['sharpe']:5.2f})")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-perturb", action="store_true", help="[C] 생략(시간 절약)")
    args = ap.parse_args()

    df = gb.load_full_data()
    regime, dfirst, dlast = gb.build_regime_arrays(df)

    base = run_intraday(df, regime, dfirst, dlast, gb.BULL_PARAMS)
    t_all = base.trades
    t_train, t_test = yr_slice(t_all, *TRAIN), yr_slice(t_all, *TEST)

    print("=" * 100)
    print("인트라데이 피봇 엣지 검증 배터리 — 게이팅, exit=next_entry")
    for lbl, t in [("train 2019-2025", t_train), ("test 2026", t_test)]:
        m = summarize(t)
        print(f"  {lbl}: n={m['n']} | 기대값={m['exp']:+.3f}pt | 승률={m['wr']:.2f}% | "
              f"Sharpe={m['sharpe']:.3f} | PnL={m['pnl']:+,.0f}")

    # [A] 유의성
    test_A_significance(t_train, "train (주의: 파라미터가 이 구간에서 최적화됨 → 유의성 상방 편향)")
    test_A_significance(t_test, "test 2026 (파라미터 사전 고정 → 편향 없음, 단 n 작음)")

    # [B] 순열검정
    test_B_permutation(t_train, df, regime, dfirst, dlast, gb.BT,
                       "train (상방 편향 주의)")
    test_B_permutation(t_test, df, regime, dfirst, dlast, gb.BT,
                       "test 2026 ★ 진짜 판정")

    # [C] 섭동
    if not args.skip_perturb:
        test_C_perturbation(df, regime, dfirst, dlast, gb.BULL_PARAMS)

    # [D] 비용
    test_D_cost_stress(df, regime, dfirst, dlast, gb.BULL_PARAMS)

    print("\n판정 가이드:")
    print("  엣지 '실재' 인정 조건: [B] test 귀무1 p<0.05 AND [C] 섭동 전 구간 기대값>0 AND [D] 2틱에서 생존")
    print("  하나라도 실패 → 해당 축을 보수적으로 재설계 후 재검증 (test 반복 조회는 금지)")


if __name__ == "__main__":
    main()
