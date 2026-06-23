# -*- coding: utf-8 -*-
"""
BULL 레짐 전환 타이밍 최적화

BULL 레짐(MA20>MA60) 진입/청산 시점을 다양한 딜레이/확인 기간으로 테스트.

- 기본 모드: 전체 기간(2019-2026) 타이밍 파라미터 그리드 탐색
- WFO 모드(--wfo): 훈련(2019-2025)에서 최적 타이밍 찾고, 검증(2026)에 적용

타이밍 파라미터:
    - confirm_days: MA20>MA60 조건이 연속 confirm_days일 동안 유지되어야 BULL 시작
    - entry_delay: BULL 시작 후 entry_delay일 후부터 실제 거래
    - exit_delay: MA20<MA60 발생 후 exit_delay일 동안 BULL 거래 유지
"""
import sys
import gc
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from itertools import product

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import regime_intraday_v2 as rg
import pivot_regime_optimizer as pro
import pivot_bull_strategy as pbs
import pivot_viability_analysis as pva
import pivot_optuna_v2 as pv

OUTPUT_LOG = Path(__file__).parent / "pivot_bull_timing_optimizer.log"
ALL_YEARS = list(range(2019, 2027))

# WFO 모드 기본 설정
TRAIN_YEARS = list(range(2019, 2026))
TEST_YEARS = [2026]

BULL_LONG_PARAMS = {
    "base_pct": 1.272989526401749,
    "base_multiplier": 1.3341908735602903,
    "atr_weight": 0.20831334967633547,
    "confirmation_bars": 1,
    "min_wave_pct": 0.07699392762885474,
    "min_pivot_interval_bars": 28,
    "direction_mode": "long_only",
}


def log_and_print(msg: str, log_file):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


def fmt_metrics(metrics: Dict[str, Any], label: str = "") -> str:
    return (f"{label:<50} | 거래={metrics['n_trades']:>4} | 승률={metrics['win_rate']:>6.2f}% | "
            f"PnL={metrics['total_pnl_krw']:>13,.0f} | Sharpe={metrics['sharpe_daily']:>7.3f} | "
            f"MaxDD={metrics['max_drawdown_krw']:>13,.0f}")


def generate_regime_signal_with_timing(years: List[int],
                                       confirm_days: int = 1,
                                       entry_delay: int = 0,
                                       exit_delay: int = 0) -> pd.Series:
    """타이밍 파라미터를 적용한 BULL 레짐 신호 생성."""
    raw_signals = []
    for year in years:
        df = pva.load_data_by_year(year)
        if len(df) == 0:
            continue
        daily = rg.to_daily(df, pva.BT.session_boundary_hour)
        signal = rg.daily_regime_signal(daily, regime_method="ma", ma_short=20, ma_long=60)
        raw_signals.append(signal)
        del df, daily
        gc.collect()
    
    raw = pd.concat(raw_signals).sort_index()
    bull = (raw == 1).astype(int)
    
    # confirm_days: 연속 confirm_days일 동안 bull이어야 bull 시작
    if confirm_days > 1:
        confirmed = bull.rolling(window=confirm_days).min().shift(-(confirm_days - 1)).fillna(0).astype(int)
    else:
        confirmed = bull
    
    # bull 상태 확장: bull 시작 후 entry_delay일 후부터, bull 종료 후 exit_delay일까지
    state = pd.Series(0, index=raw.index, dtype=int)
    in_bull = False
    bull_start_idx = None
    
    for i, (idx, val) in enumerate(confirmed.items()):
        if val == 1 and not in_bull:
            in_bull = True
            bull_start_idx = i
        elif val == 0 and in_bull:
            in_bull = False
            # 시작: entry_delay 적용
            start_idx = bull_start_idx + entry_delay
            # 종료: 현재 인덱스 + exit_delay - 1
            end_idx = i + exit_delay - 1
            if start_idx <= end_idx and start_idx < len(state):
                state.iloc[start_idx:min(end_idx + 1, len(state))] = 1
            bull_start_idx = None
    
    # 끝까지 bull이 지속되는 경우
    if in_bull and bull_start_idx is not None:
        start_idx = bull_start_idx + entry_delay
        if start_idx < len(state):
            state.iloc[start_idx:] = 1
    
    return state


def evaluate_bull_timing(years: List[int], confirm_days: int, entry_delay: int, exit_delay: int) -> Dict[str, Any]:
    """특정 타이밍 파라미터로 BULL 피봇 롱 전략 평가."""
    pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=BULL_LONG_PARAMS["base_pct"],
        base_multiplier=BULL_LONG_PARAMS["base_multiplier"],
        atr_weight=BULL_LONG_PARAMS["atr_weight"],
        confirmation_bars=BULL_LONG_PARAMS["confirmation_bars"],
    )
    fcfg = pv.FilterConfig(
        enabled=True,
        min_wave_pct=BULL_LONG_PARAMS["min_wave_pct"],
        min_pivot_interval_bars=BULL_LONG_PARAMS["min_pivot_interval_bars"],
        st_distance_threshold=0.1,
        adx_hold_threshold=15.0,
    )
    
    regime_signal = generate_regime_signal_with_timing(years, confirm_days, entry_delay, exit_delay)
    
    total_pnl_krw = 0.0
    total_pnl_pts = 0.0
    total_trades = 0
    weighted_win_rate = 0.0
    gross_win_pts = 0.0
    gross_loss_pts = 0.0
    daily_pnl_list: List[pd.Series] = []
    equity_list: List[pd.Series] = []
    
    for year in years:
        df = pva.load_data_by_year(year)
        if len(df) == 0:
            continue
        df_bull = pro._filter_df_by_regime(df, regime_signal, 1)
        if len(df_bull) == 0:
            del df, df_bull
            gc.collect()
            continue
        
        res = pva.run_pivot(df_bull, pcfg, fcfg, "long_only")
        
        total_pnl_krw += res.total_pnl_krw
        total_pnl_pts += res.total_pnl_pts
        total_trades += res.n_trades
        weighted_win_rate += res.win_rate * res.n_trades
        
        if res.trades is not None and len(res.trades) > 0:
            net_pts = res.trades["net_pts"]
            gross_win_pts += net_pts[net_pts > 0].sum()
            gross_loss_pts += -net_pts[net_pts < 0].sum()
            
            res.trades["exit_date"] = pd.to_datetime(res.trades["exit_time"]).dt.date
            daily = res.trades.groupby("exit_date")["net_krw"].sum()
            daily_pnl_list.append(daily)
            equity_list.append(res.trades["net_krw"].cumsum())
        
        del df, df_bull, res
        gc.collect()
    
    if total_trades == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl_krw": 0.0,
            "total_pnl_pts": 0.0, "expectancy_krw": 0.0, "expectancy_pts": 0.0,
            "profit_factor": 0.0, "sharpe_daily": -1.0, "max_drawdown_krw": 0.0,
            "confirm_days": confirm_days, "entry_delay": entry_delay, "exit_delay": exit_delay,
        }
    
    win_rate = weighted_win_rate / total_trades
    expectancy_krw = total_pnl_krw / total_trades
    expectancy_pts = total_pnl_pts / total_trades
    profit_factor = float(gross_win_pts / gross_loss_pts) if gross_loss_pts > 0 else float("inf") if gross_win_pts > 0 else 0.0
    
    if daily_pnl_list:
        combined_daily = pd.concat(daily_pnl_list)
        if len(combined_daily) >= 2 and combined_daily.std(ddof=1) > 0:
            sharpe = float(combined_daily.mean() / combined_daily.std(ddof=1) * math.sqrt(pva.BT.annualization))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
    
    if equity_list:
        combined_equity = pd.concat(equity_list)
        max_dd = float((combined_equity - combined_equity.cummax()).min())
    else:
        max_dd = 0.0
    
    return {
        "n_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl_krw": total_pnl_krw,
        "total_pnl_pts": total_pnl_pts,
        "expectancy_krw": expectancy_krw,
        "expectancy_pts": expectancy_pts,
        "profit_factor": profit_factor,
        "sharpe_daily": sharpe,
        "max_drawdown_krw": max_dd,
        "confirm_days": confirm_days,
        "entry_delay": entry_delay,
        "exit_delay": exit_delay,
    }


def _run_default_mode(log):
    """기본 모드: 전체 기간 타이밍 그리드 탐색."""
    log_and_print("=" * 100, log)
    log_and_print("BULL 레짐 전환 타이밍 최적화", log)
    log_and_print(f"대상 기간: {ALL_YEARS}", log)
    log_and_print("=" * 100, log)
    
    # 1) 기본 타이밍 (confirm=1, delay=0, exit=0) 평가
    log_and_print(f"\n[1] 기본 타이밍 (confirm=1, entry_delay=0, exit_delay=0)", log)
    baseline = evaluate_bull_timing(ALL_YEARS, 1, 0, 0)
    log_and_print(fmt_metrics(baseline, "기본 BULL 피봇 롱"), log)
    
    # 2) 그리드 탐색
    log_and_print(f"\n[2] 타이밍 파라미터 그리드 탐색", log)
    param_grid = {
        "confirm_days": [1, 2, 3],
        "entry_delay": [0, 1, 2, 3],
        "exit_delay": [0, 1, 2, 3],
    }
    combos = list(product(*param_grid.values()))
    log_and_print(f"  총 조합: {len(combos)}", log)
    
    results = []
    for i, combo in enumerate(combos, 1):
        confirm_days, entry_delay, exit_delay = combo
        metrics = evaluate_bull_timing(ALL_YEARS, confirm_days, entry_delay, exit_delay)
        results.append(metrics)
        log_and_print(f"  [{i}/{len(combos)}] confirm={confirm_days}, entry_delay={entry_delay}, exit_delay={exit_delay} | "
                      f"거래={metrics['n_trades']}, PnL={metrics['total_pnl_krw']:,.0f}, "
                      f"Sharpe={metrics['sharpe_daily']:.3f}, MaxDD={metrics['max_drawdown_krw']:,.0f}", log)
    
    # 3) 최고 Sharpe / PnL / Calmar
    best_sharpe = max(results, key=lambda x: x["sharpe_daily"])
    best_pnl = max(results, key=lambda x: x["total_pnl_krw"])
    best_calmar = max(results, key=lambda x: x["total_pnl_krw"] / abs(x["max_drawdown_krw"]) if x["max_drawdown_krw"] else 0)
    top_sharpe = sorted(results, key=lambda x: x["sharpe_daily"], reverse=True)[:10]
    top_pnl = sorted(results, key=lambda x: x["total_pnl_krw"], reverse=True)[:10]
    
    log_and_print(f"\n[최고 Sharpe]", log)
    log_and_print(f"  confirm={best_sharpe['confirm_days']}, entry_delay={best_sharpe['entry_delay']}, exit_delay={best_sharpe['exit_delay']}", log)
    log_and_print(f"  거래={best_sharpe['n_trades']}, 승률={best_sharpe['win_rate']:.2f}%, "
                  f"PnL={best_sharpe['total_pnl_krw']:,.0f}, Sharpe={best_sharpe['sharpe_daily']:.3f}, "
                  f"MaxDD={best_sharpe['max_drawdown_krw']:,.0f}", log)
    
    log_and_print(f"\n[최고 PnL]", log)
    log_and_print(f"  confirm={best_pnl['confirm_days']}, entry_delay={best_pnl['entry_delay']}, exit_delay={best_pnl['exit_delay']}", log)
    log_and_print(f"  거래={best_pnl['n_trades']}, 승률={best_pnl['win_rate']:.2f}%, "
                  f"PnL={best_pnl['total_pnl_krw']:,.0f}, Sharpe={best_pnl['sharpe_daily']:.3f}, "
                  f"MaxDD={best_pnl['max_drawdown_krw']:,.0f}", log)
    
    log_and_print(f"\n[최고 Calmar]", log)
    log_and_print(f"  confirm={best_calmar['confirm_days']}, entry_delay={best_calmar['entry_delay']}, exit_delay={best_calmar['exit_delay']}", log)
    log_and_print(f"  거래={best_calmar['n_trades']}, 승률={best_calmar['win_rate']:.2f}%, "
                  f"PnL={best_calmar['total_pnl_krw']:,.0f}, Sharpe={best_calmar['sharpe_daily']:.3f}, "
                  f"MaxDD={best_calmar['max_drawdown_krw']:,.0f}", log)
    
    log_and_print(f"\n[상위 10 Sharpe]", log)
    for r in top_sharpe:
        log_and_print(f"  c={r['confirm_days']}, e={r['entry_delay']}, x={r['exit_delay']} | "
                      f"거래={r['n_trades']:>4} | PnL={r['total_pnl_krw']:>13,.0f} | "
                      f"Sharpe={r['sharpe_daily']:>7.3f} | MaxDD={r['max_drawdown_krw']:>13,.0f}", log)
    
    log_and_print(f"\n[상위 10 PnL]", log)
    for r in top_pnl:
        log_and_print(f"  c={r['confirm_days']}, e={r['entry_delay']}, x={r['exit_delay']} | "
                      f"거래={r['n_trades']:>4} | PnL={r['total_pnl_krw']:>13,.0f} | "
                      f"Sharpe={r['sharpe_daily']:>7.3f} | MaxDD={r['max_drawdown_krw']:>13,.0f}", log)
    
    # 4) 요약
    log_and_print("\n" + "=" * 100, log)
    log_and_print("요약", log)
    log_and_print("=" * 100, log)
    log_and_print(f"기본 타이밍:      PnL={baseline['total_pnl_krw']:,.0f}, Sharpe={baseline['sharpe_daily']:.3f}, MaxDD={baseline['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"최고 Sharpe:     PnL={best_sharpe['total_pnl_krw']:,.0f}, Sharpe={best_sharpe['sharpe_daily']:.3f}, MaxDD={best_sharpe['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"  (confirm={best_sharpe['confirm_days']}, entry_delay={best_sharpe['entry_delay']}, exit_delay={best_sharpe['exit_delay']})", log)
    log_and_print(f"최고 PnL:        PnL={best_pnl['total_pnl_krw']:,.0f}, Sharpe={best_pnl['sharpe_daily']:.3f}, MaxDD={best_pnl['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"  (confirm={best_pnl['confirm_days']}, entry_delay={best_pnl['entry_delay']}, exit_delay={best_pnl['exit_delay']})", log)
    log_and_print("=" * 100, log)


def _run_wfo_mode(log):
    """WFO 모드: 훈련 기간 타이밍 최적화 후 검증 기간 테스트."""
    log_and_print("=" * 100, log)
    log_and_print("BULL 레짐 타이밍 파라미터 OOS 검증", log)
    log_and_print(f"훈련: {TRAIN_YEARS} | 검증: {TEST_YEARS}", log)
    log_and_print("=" * 100, log)
    
    # 1) 훈련 기간 그리드 탐색
    log_and_print(f"\n[1] 훈련 기간 타이밍 파라미터 그리드 탐색", log)
    param_grid = {
        "confirm_days": [1, 2, 3],
        "entry_delay": [0, 1, 2, 3],
        "exit_delay": [0, 1, 2, 3],
    }
    combos = list(product(*param_grid.values()))
    log_and_print(f"  총 조합: {len(combos)}", log)
    
    train_results = []
    for i, combo in enumerate(combos, 1):
        confirm_days, entry_delay, exit_delay = combo
        metrics = evaluate_bull_timing(TRAIN_YEARS, confirm_days, entry_delay, exit_delay)
        train_results.append(metrics)
        log_and_print(f"  [{i}/{len(combos)}] c={confirm_days}, e={entry_delay}, x={exit_delay} | "
                      f"거래={metrics['n_trades']}, PnL={metrics['total_pnl_krw']:,.0f}, "
                      f"Sharpe={metrics['sharpe_daily']:.3f}, MaxDD={metrics['max_drawdown_krw']:,.0f}", log)
    
    # 2) 훈련 최고 조합 선정
    best_sharpe = max(train_results, key=lambda x: x["sharpe_daily"])
    best_pnl = max(train_results, key=lambda x: x["total_pnl_krw"])
    best_calmar = max(train_results, key=lambda x: x["total_pnl_krw"] / abs(x["max_drawdown_krw"]) if x["max_drawdown_krw"] else 0)
    
    log_and_print(f"\n[훈련 최고 Sharpe]", log)
    log_and_print(f"  c={best_sharpe['confirm_days']}, e={best_sharpe['entry_delay']}, x={best_sharpe['exit_delay']}", log)
    log_and_print(f"  거래={best_sharpe['n_trades']}, PnL={best_sharpe['total_pnl_krw']:,.0f}, "
                  f"Sharpe={best_sharpe['sharpe_daily']:.3f}, MaxDD={best_sharpe['max_drawdown_krw']:,.0f}", log)
    
    log_and_print(f"\n[훈련 최고 PnL]", log)
    log_and_print(f"  c={best_pnl['confirm_days']}, e={best_pnl['entry_delay']}, x={best_pnl['exit_delay']}", log)
    log_and_print(f"  거래={best_pnl['n_trades']}, PnL={best_pnl['total_pnl_krw']:,.0f}, "
                  f"Sharpe={best_pnl['sharpe_daily']:.3f}, MaxDD={best_pnl['max_drawdown_krw']:,.0f}", log)
    
    # 3) 검증 기간 테스트
    log_and_print(f"\n[2] 검증 기간 테스트", log)
    
    baseline_test = evaluate_bull_timing(TEST_YEARS, 1, 0, 0)
    log_and_print(fmt_metrics(baseline_test, "기본 타이밍 (c=1,e=0,x=0) 검증"), log)
    
    sharpe_test = evaluate_bull_timing(TEST_YEARS, best_sharpe["confirm_days"], best_sharpe["entry_delay"], best_sharpe["exit_delay"])
    log_and_print(fmt_metrics(sharpe_test, f"최고 Sharpe 타이밍 검증 c={best_sharpe['confirm_days']},e={best_sharpe['entry_delay']},x={best_sharpe['exit_delay']}"), log)
    
    pnl_test = evaluate_bull_timing(TEST_YEARS, best_pnl["confirm_days"], best_pnl["entry_delay"], best_pnl["exit_delay"])
    log_and_print(fmt_metrics(pnl_test, f"최고 PnL 타이밍 검증 c={best_pnl['confirm_days']},e={best_pnl['entry_delay']},x={best_pnl['exit_delay']}"), log)
    
    calmar_test = evaluate_bull_timing(TEST_YEARS, best_calmar["confirm_days"], best_calmar["entry_delay"], best_calmar["exit_delay"])
    log_and_print(fmt_metrics(calmar_test, f"최고 Calmar 타이밍 검증 c={best_calmar['confirm_days']},e={best_calmar['entry_delay']},x={best_calmar['exit_delay']}"), log)
    
    # 4) 요약
    log_and_print("\n" + "=" * 100, log)
    log_and_print("OOS 검증 요약", log)
    log_and_print("=" * 100, log)
    log_and_print(f"훈련 최고 Sharpe: c={best_sharpe['confirm_days']}, e={best_sharpe['entry_delay']}, x={best_sharpe['exit_delay']}", log)
    log_and_print(f"  훈련: PnL={best_sharpe['total_pnl_krw']:,.0f}, Sharpe={best_sharpe['sharpe_daily']:.3f}", log)
    log_and_print(f"  검증: PnL={sharpe_test['total_pnl_krw']:,.0f}, Sharpe={sharpe_test['sharpe_daily']:.3f}", log)
    
    log_and_print(f"\n훈련 최고 PnL: c={best_pnl['confirm_days']}, e={best_pnl['entry_delay']}, x={best_pnl['exit_delay']}", log)
    log_and_print(f"  훈련: PnL={best_pnl['total_pnl_krw']:,.0f}, Sharpe={best_pnl['sharpe_daily']:.3f}", log)
    log_and_print(f"  검증: PnL={pnl_test['total_pnl_krw']:,.0f}, Sharpe={pnl_test['sharpe_daily']:.3f}", log)
    
    log_and_print(f"\n기본 타이밍", log)
    log_and_print(f"  훈련: PnL={evaluate_bull_timing(TRAIN_YEARS, 1, 0, 0)['total_pnl_krw']:,.0f}, "
                  f"Sharpe={evaluate_bull_timing(TRAIN_YEARS, 1, 0, 0)['sharpe_daily']:.3f}", log)
    log_and_print(f"  검증: PnL={baseline_test['total_pnl_krw']:,.0f}, Sharpe={baseline_test['sharpe_daily']:.3f}", log)
    log_and_print("=" * 100, log)


def main():
    parser = argparse.ArgumentParser(description="BULL 레짐 전환 타이밍 최적화")
    parser.add_argument("--wfo", action="store_true", help="WFO 모드 (훈련/검증 분할)")
    args = parser.parse_args()
    
    output_log = Path(__file__).parent / ("pivot_bull_timing_wfo.log" if args.wfo else "pivot_bull_timing_optimizer.log")
    
    with open(output_log, "w", encoding="utf-8") as log:
        if args.wfo:
            _run_wfo_mode(log)
        else:
            _run_default_mode(log)


if __name__ == "__main__":
    main()
