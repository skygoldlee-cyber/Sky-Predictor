# -*- coding: utf-8 -*-
"""
숏-전용 피봇 + 하락장 필터 Walk-forward 검증.

- 6개월 train / 1개월 test 롤링 윈도우
- 복합 필터 "MA20_down_or_ADX_bear"를 고정하고 피봇 파라미터만 최적화
- 선택된 파라미터를 다음 1개월 test에 적용하여 OOS 누적 평가
- 무조건 숏-전용 및 롱-또는-플랫과 비교
"""
import sys
import math
from pathlib import Path
from datetime import date
from itertools import product

sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg
from pivot_short_bear_optimize import (
    load_data, compute_filter_dates, run_short_pivot, run_long_or_flat,
    filter_trades_by_dates, compute_metrics, fmt_result, BT, PARAM_GRID,
)

# Walk-forward 모드 설정
FIXED_FILTER = "MA20_down_or_ADX_bear"  # 복합 필터를 고정하고 파라미터만 최적화
OUTPUT_MD = (
    Path(__file__).parent / f"pivot_short_bear_walkforward_{FIXED_FILTER}.md"
    if FIXED_FILTER
    else Path(__file__).parent / "pivot_short_bear_walkforward.md"
)

FILTERS = [FIXED_FILTER] if FIXED_FILTER else [
    "MA20_down",
    "ADX_bear",
    "consecutive_2down",
    "MA20_down_or_ADX_bear",
    "any2_of_3",
    "strong_or_MA20_not_vol",
]

MIN_TRAIN_TRADES = 5
TRAIN_MONTHS = 6  # 짧은 윈도우보다 안정적인 6개월 train
TEST_MONTHS = 1


def generate_windows(daily):
    """월 단위 롤링 윈도우를 생성."""
    first_month = daily.index[0].normalize().replace(day=1)
    last_month_plus = (daily.index[-1].normalize() + pd.offsets.MonthBegin(1)).replace(day=1)
    month_starts = pd.date_range(first_month, last_month_plus, freq="MS")
    n = len(month_starts)
    windows = []
    for i in range(n - TRAIN_MONTHS - TEST_MONTHS):
        train_start = month_starts[i]
        test_start = month_starts[i + TRAIN_MONTHS]
        test_end = month_starts[i + TRAIN_MONTHS + TEST_MONTHS] - pd.Timedelta(days=1)
        train_end = test_start - pd.Timedelta(days=1)
        windows.append((
            train_start.date(), train_end.date(),
            test_start.date(), test_end.date(),
        ))
    return windows


def precompute_short_results(df_5min):
    """모든 파라미터 조합에 대해 한 번씩만 숏-전용 피봇 백테스트를 수행."""
    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
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
        print(f"  [{i}/{len(combos)}] params={params}")
        res = run_short_pivot(df_5min, pcfg, fcfg)
        results.append({
            "params": params,
            "trades": res.trades if res is not None else None,
        })
    return results


def select_best_for_window(combo_results, filter_dates, train_dates, min_trades=MIN_TRAIN_TRADES):
    """주어진 train 윈도우에서 Sharpe 최고의 (필터, 파라미터)를 선택."""
    best_score = -float("inf")
    best = None
    for combo in combo_results:
        trades = combo["trades"]
        if trades is None or trades.empty:
            continue
        for fname in FILTERS:
            train_res = compute_metrics(
                filter_trades_by_dates(trades, filter_dates[fname] & train_dates),
                BT,
            )
            if train_res.n_trades < min_trades:
                continue
            score = train_res.sharpe_daily
            if score > best_score:
                best_score = score
                best = (fname, combo["params"], trades)
    return best


def main():
    df_5min = load_data()
    daily = rg.to_daily(df_5min, BT.session_boundary_hour)
    all_dates = set(daily.index.date)
    filter_dates, _, descriptions = compute_filter_dates(daily)
    windows = generate_windows(daily)

    print(f"전체 5분봉: {len(df_5min)}봉, {df_5min.index[0]} ~ {df_5min.index[-1]}")
    print(f"전체 거래일: {len(all_dates)}일")
    print(f"Walk-forward 윈도우 개수: {len(windows)}")
    for i, (ts, te, ps, pe) in enumerate(windows, 1):
        print(f"  window {i}: train {ts}~{te}, test {ps}~{pe}")

    # 1) 모든 파라미터 조합에 대해 한 번씩 백테스트
    print("\n모든 파라미터 조합 백테스트 중...")
    combo_results = precompute_short_results(df_5min)

    # 2) 베이스라인 전략 미리 계산
    print("\n베이스라인 전략 계산 중...")
    default_pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=0.5, base_multiplier=2.0, atr_weight=0.3, confirmation_bars=2
    )
    default_fcfg = pv.FilterConfig(
        enabled=True, min_wave_pct=0.3, min_pivot_interval_bars=20,
        st_distance_threshold=0.1, adx_hold_threshold=15.0,
    )
    default_short_res = run_short_pivot(df_5min, default_pcfg, default_fcfg)
    default_short_trades = default_short_res.trades if default_short_res is not None else None

    lf_res = run_long_or_flat(df_5min)
    lf_trades = lf_res.trades if lf_res is not None else None

    # 3) 각 윈도우별 최적 선택 및 OOS 평가
    print("\nWalk-forward 평가 중...")
    wf_results = []
    for idx, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
        train_dates = {d for d in all_dates if train_start <= d <= train_end}
        test_dates = {d for d in all_dates if test_start <= d <= test_end}
        if not train_dates or not test_dates:
            continue

        best = select_best_for_window(combo_results, filter_dates, train_dates)
        if best is None:
            print(f"  window {idx}: test={test_start}~{test_end} - train 후보 없음, 스킵")
            continue

        fname, params, trades = best
        test_res = compute_metrics(
            filter_trades_by_dates(trades, filter_dates[fname] & test_dates),
            BT,
        )
        wf_results.append({
            "window": idx,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "filter": fname,
            "params": params,
            "test_res": test_res,
        })
        print(f"  window {idx}: filter={fname}, params={params}, test={fmt_result(test_res)}")

    # 4) 베이스라인 윈도우별 평가
    baseline_rows = []
    for idx, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
        test_dates = {d for d in all_dates if test_start <= d <= test_end}
        if not test_dates:
            continue
        ds_res = compute_metrics(
            filter_trades_by_dates(default_short_trades, test_dates), BT
        )
        lf_window_res = compute_metrics(
            filter_trades_by_dates(lf_trades, test_dates), BT
        )
        baseline_rows.append({
            "window": idx,
            "test_start": test_start,
            "test_end": test_end,
            "default_short": ds_res,
            "long_or_flat": lf_window_res,
        })

    # 5) 누적 OOS 평가
    def combined_metrics(wf_list, key="test_res"):
        trades = [r[key].trades for r in wf_list if r[key].trades is not None and not r[key].trades.empty]
        if not trades:
            return pv.BacktestResult(trades=pd.DataFrame())
        combined = pd.concat(trades, ignore_index=True)
        return compute_metrics(combined, BT)

    wf_combined = combined_metrics(wf_results)
    ds_combined = combined_metrics(baseline_rows, key="default_short")
    lf_combined = combined_metrics(baseline_rows, key="long_or_flat")

    # 6) 보고서 작성
    lines = [
        "# 숏-전용 피봇 + 하락장 필터 Walk-forward 검증",
        "",
        "## 개요",
        f"- **Train 윈도우**: {TRAIN_MONTHS}개월",
        f"- **Test 윈도우**: {TEST_MONTHS}개월",
        f"- **필터 후보**: {', '.join(FILTERS)}",
        f"- **최소 train 거래 수**: {MIN_TRAIN_TRADES}",
        f"- **전체 거래일**: {len(all_dates)}일",
        f"- **Walk-forward 윈도우 개수**: {len(windows)}",
        "",
        "## 윈도우별 최적 선택 및 OOS 결과",
        "",
        "| 윈도우 | train 기간 | test 기간 | 선택 필터 | 거래 | 승률 | PnL | Sharpe | MaxDD |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in wf_results:
        res = r["test_res"]
        lines.append(
            f"| {r['window']} | {r['train_start']}~{r['train_end']} | "
            f"{r['test_start']}~{r['test_end']} | {r['filter']} | "
            f"{res.n_trades} | {res.win_rate:.2f}% | {res.total_pnl_krw:,.0f} | "
            f"{res.sharpe_daily:.3f} | {res.max_drawdown_krw:,.0f} |"
        )
    lines.append("")

    lines.append("## 누적 OOS 결과")
    lines.append("")
    lines.append("| 전략 | 거래 | 승률 | PnL | Sharpe | MaxDD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for label, res in [
        ("WF 최적 필터+파라미터", wf_combined),
        ("기준 숏-전용(무조건)", ds_combined),
        ("롱-또는-플랫", lf_combined),
    ]:
        lines.append(
            f"| {label} | {res.n_trades} | {res.win_rate:.2f}% | "
            f"{res.total_pnl_krw:,.0f} | {res.sharpe_daily:.3f} | {res.max_drawdown_krw:,.0f} |"
        )
    lines.append("")

    lines.append("## 해석")
    lines.append("- 각 윈도우는 **오직 과거 train 데이터만**으로 필터와 파라미터를 선택했습니다.")
    lines.append("- 누적 OOS는 선택된 조합들을 실제 test 기간에 적용한 결과입니다.")
    lines.append("- train Sharpe 기준 선택 + 최소 거래 수 제한으로 과적합을 일부 억제합니다.")
    lines.append("- 거래 횟수가 적은 윈도우는 통계적 검증력이 낮으므로 추가 분석이 필요합니다.")

    md_text = "\n".join(lines)
    OUTPUT_MD.write_text(md_text, encoding="utf-8")
    print(f"\n결과 저장 완료: {OUTPUT_MD}")
    print(md_text)


if __name__ == "__main__":
    main()
