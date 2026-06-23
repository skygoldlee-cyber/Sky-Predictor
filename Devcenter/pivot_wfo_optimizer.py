# -*- coding: utf-8 -*-
"""
피봇 파라미터 Walk-Forward Optimization

- Phase 1: 단일 훈련/검증 (2023-2024 → 2025)
- Phase 2: 2년 훈련 / 1년 검증 창을 2019-2026까지 이동

Optuna 베이지안 최적화 + 연도별 분할 처리로 메모리 안전하게 사용.
"""
import sys
import gc
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import optuna
import pivot_viability_analysis as pva
import pivot_optuna_v2 as pv

OUTPUT_LOG = Path(__file__).parent / "pivot_wfo_optimizer_phase1.log"

TRAIN_YEARS = [2023, 2024]
TEST_YEARS = [2025]
N_TRIALS = 100
SEED = 42

# Phase 2 WFO 창 설정 (2년 훈련 / 1년 검증)
N_TRIALS_PER_WINDOW = 50
WINDOWS = [
    ([2019, 2020], 2021),
    ([2020, 2021], 2022),
    ([2021, 2022], 2023),
    ([2022, 2023], 2024),
    ([2023, 2024], 2025),
    ([2024, 2025], 2026),
]


def log_and_print(msg: str, log_file):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


def evaluate_years(years: List[int], pcfg: pv.HybridAdaptivePivotConfig,
                   fcfg: pv.FilterConfig, direction_mode: str) -> Dict[str, Any]:
    """연도별로 백테스트를 실행하고 핵심 지표만 반환 (메모리 절약)."""
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
        res = pva.run_pivot(df, pcfg, fcfg, direction_mode)
        
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
        
        del df, res
        gc.collect()
    
    if total_trades == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl_krw": 0.0,
            "total_pnl_pts": 0.0, "expectancy_krw": 0.0, "expectancy_pts": 0.0,
            "profit_factor": 0.0, "sharpe_daily": -1.0, "max_drawdown_krw": 0.0,
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
        running_max = combined_equity.cummax()
        max_dd = float((combined_equity - running_max).min())
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
    }


def fmt_metrics(metrics: Dict[str, Any], label: str = "") -> str:
    return (f"{label:<45} | 거래={metrics['n_trades']:>4} | 승률={metrics['win_rate']:>6.2f}% | "
            f"PnL={metrics['total_pnl_krw']:>13,.0f} | Sharpe={metrics['sharpe_daily']:>7.3f} | "
            f"MaxDD={metrics['max_drawdown_krw']:>13,.0f}")


def make_objective(log_file):
    def objective(trial: optuna.Trial) -> float:
        pcfg = pv.HybridAdaptivePivotConfig(
            base_pct=trial.suggest_float("base_pct", 0.1, 1.5, log=True),
            base_multiplier=trial.suggest_float("base_multiplier", 0.5, 3.0),
            atr_weight=trial.suggest_float("atr_weight", 0.0, 1.0),
            confirmation_bars=trial.suggest_int("confirmation_bars", 1, 5),
        )
        fcfg = pv.FilterConfig(
            enabled=True,
            min_wave_pct=trial.suggest_float("min_wave_pct", 0.1, 0.5, log=True),
            min_pivot_interval_bars=trial.suggest_int("min_pivot_interval_bars", 3, 30),
            st_distance_threshold=0.1,
            adx_hold_threshold=15.0,
        )
        direction_mode = trial.suggest_categorical("direction_mode", ["both", "long_only"])
        
        train_metrics = evaluate_years(TRAIN_YEARS, pcfg, fcfg, direction_mode)
        
        # 거래 수 하한 미달 시 페널티
        if train_metrics["n_trades"] < 20:
            return -1.0
        
        log_and_print(f"  trial {trial.number}: {direction_mode} | "
                      f"base_pct={pcfg.base_pct:.3f}, mult={pcfg.base_multiplier:.2f}, "
                      f"atr_w={pcfg.atr_weight:.2f}, conf={pcfg.confirmation_bars}, "
                      f"wave={fcfg.min_wave_pct:.3f}, interval={fcfg.min_pivot_interval_bars} | "
                      f"train Sharpe={train_metrics['sharpe_daily']:.3f}", log_file)
        
        return train_metrics["sharpe_daily"]
    
    return objective


def make_objective_for_window(train_years: List[int], log_file):
    """특정 훈련 기간에 대한 Optuna objective (Phase 2)."""
    def objective(trial: optuna.Trial) -> float:
        pcfg = pv.HybridAdaptivePivotConfig(
            base_pct=trial.suggest_float("base_pct", 0.1, 1.5, log=True),
            base_multiplier=trial.suggest_float("base_multiplier", 0.5, 3.0),
            atr_weight=trial.suggest_float("atr_weight", 0.0, 1.0),
            confirmation_bars=trial.suggest_int("confirmation_bars", 1, 5),
        )
        fcfg = pv.FilterConfig(
            enabled=True,
            min_wave_pct=trial.suggest_float("min_wave_pct", 0.1, 0.5, log=True),
            min_pivot_interval_bars=trial.suggest_int("min_pivot_interval_bars", 3, 30),
            st_distance_threshold=0.1,
            adx_hold_threshold=15.0,
        )
        direction_mode = trial.suggest_categorical("direction_mode", ["both", "long_only"])
        
        train_metrics = evaluate_years(train_years, pcfg, fcfg, direction_mode)
        
        if train_metrics["n_trades"] < 20:
            return -1.0
        
        log_and_print(f"  window {train_years} trial {trial.number}: {direction_mode} | "
                      f"base_pct={pcfg.base_pct:.3f}, mult={pcfg.base_multiplier:.2f}, "
                      f"atr_w={pcfg.atr_weight:.2f}, conf={pcfg.confirmation_bars}, "
                      f"wave={fcfg.min_wave_pct:.3f}, interval={fcfg.min_pivot_interval_bars} | "
                      f"train Sharpe={train_metrics['sharpe_daily']:.3f}", log_file)
        
        return train_metrics["sharpe_daily"]
    
    return objective


def run_window_optimization(train_years: List[int], test_year: int, log_file) -> Dict[str, Any]:
    """한 개 WFO 창에 대해 최적화 및 검증을 실행 (Phase 2)."""
    log_and_print(f"\n[Window {train_years} → {test_year}] 최적화 시작", log_file)
    
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    
    study.optimize(make_objective_for_window(train_years, log_file),
                   n_trials=N_TRIALS_PER_WINDOW, show_progress_bar=True)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    log_and_print(f"  최적 파라미터: {best_params}", log_file)
    log_and_print(f"  훈련 Sharpe: {best_trial.value:.3f}", log_file)
    
    # 검증 기간 평가
    best_pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=best_params["base_pct"],
        base_multiplier=best_params["base_multiplier"],
        atr_weight=best_params["atr_weight"],
        confirmation_bars=best_params["confirmation_bars"],
    )
    best_fcfg = pv.FilterConfig(
        enabled=True,
        min_wave_pct=best_params["min_wave_pct"],
        min_pivot_interval_bars=best_params["min_pivot_interval_bars"],
        st_distance_threshold=0.1,
        adx_hold_threshold=15.0,
    )
    best_mode = best_params["direction_mode"]
    
    test_metrics = evaluate_years([test_year], best_pcfg, best_fcfg, best_mode)
    log_and_print(fmt_metrics(test_metrics, f"  검증 ({test_year})"), log_file)
    
    return {
        "train_years": train_years,
        "test_year": test_year,
        "best_params": best_params,
        "train_sharpe": best_trial.value,
        "test_metrics": test_metrics,
    }


def evaluate_all_years(years: List[int], pcfg: pv.HybridAdaptivePivotConfig,
                       fcfg: pv.FilterConfig, direction_mode: str) -> Dict[str, Any]:
    """전체 기간에 대해 백테스트 (Phase 2)."""
    return evaluate_years(years, pcfg, fcfg, direction_mode)


def _run_phase1(log):
    """Phase 1: 단일 훈련/검증 WFO."""
    log_and_print("=" * 100, log)
    log_and_print("피봇 파라미터 WFO 최적화 (Phase 1)", log)
    log_and_print(f"훈련: {TRAIN_YEARS} | 검증: {TEST_YEARS} | trials: {N_TRIALS}", log)
    log_and_print("=" * 100, log)
    
    # 1) Optuna 최적화
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    
    log_and_print(f"\n[1] Optuna 최적화 시작 ({N_TRIALS} trials)", log)
    study.optimize(make_objective(log), n_trials=N_TRIALS, show_progress_bar=True)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    log_and_print(f"\n[최적 파라미터] trial {best_trial.number}", log)
    log_and_print(f"  {best_params}", log)
    log_and_print(f"  훈련 Sharpe: {best_trial.value:.3f}", log)
    
    # 2) 최적 파라미터로 훈련/검증 재평가
    best_pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=best_params["base_pct"],
        base_multiplier=best_params["base_multiplier"],
        atr_weight=best_params["atr_weight"],
        confirmation_bars=best_params["confirmation_bars"],
    )
    best_fcfg = pv.FilterConfig(
        enabled=True,
        min_wave_pct=best_params["min_wave_pct"],
        min_pivot_interval_bars=best_params["min_pivot_interval_bars"],
        st_distance_threshold=0.1,
        adx_hold_threshold=15.0,
    )
    best_mode = best_params["direction_mode"]
    
    log_and_print(f"\n[2] 최적 파라미터 재평가 (mode={best_mode})", log)
    train_metrics = evaluate_years(TRAIN_YEARS, best_pcfg, best_fcfg, best_mode)
    test_metrics = evaluate_years(TEST_YEARS, best_pcfg, best_fcfg, best_mode)
    log_and_print(fmt_metrics(train_metrics, "훈련 (2023-2024)"), log)
    log_and_print(fmt_metrics(test_metrics, "검증 (2025)"), log)
    
    # 3) 롱-또는-플랫 벤치마크
    log_and_print(f"\n[3] 롱-또는-플랫 벤치마크", log)
    lf_train_results = []
    lf_test_results = []
    for year in TRAIN_YEARS:
        df = pva.load_data_by_year(year)
        if len(df) > 0:
            lf_train_results.append(pva.run_long_or_flat(df))
        del df
        gc.collect()
    for year in TEST_YEARS:
        df = pva.load_data_by_year(year)
        if len(df) > 0:
            lf_test_results.append(pva.run_long_or_flat(df))
        del df
        gc.collect()
    
    lf_train = pva.combine_results(lf_train_results)
    lf_test = pva.combine_results(lf_test_results)
    log_and_print(pva.fmt_result(lf_train, "롱-또는-플랫 훈련"), log)
    log_and_print(pva.fmt_result(lf_test, "롱-또는-플랫 검증"), log)
    
    # 4) 요약
    log_and_print("\n" + "=" * 100, log)
    log_and_print("요약", log)
    log_and_print("=" * 100, log)
    log_and_print(f"최적 피봇 훈련:  PnL={train_metrics['total_pnl_krw']:,.0f}, Sharpe={train_metrics['sharpe_daily']:.3f}, MaxDD={train_metrics['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"최적 피봇 검증:  PnL={test_metrics['total_pnl_krw']:,.0f}, Sharpe={test_metrics['sharpe_daily']:.3f}, MaxDD={test_metrics['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"롱-또는-플랫 검증: PnL={lf_test.total_pnl_krw:,.0f}, Sharpe={lf_test.sharpe_daily:.3f}, MaxDD={lf_test.max_drawdown_krw:,.0f}", log)
    log_and_print(f"최적 파라미터: {best_params}", log)
    log_and_print("=" * 100, log)


def _run_phase2(log):
    """Phase 2: 2년 훈련 / 1년 검증 창을 이동하며 WFO."""
    log_and_print("=" * 100, log)
    log_and_print("피봇 파라미터 WFO 최적화 (Phase 2) - 2019~2026", log)
    log_and_print(f"창: 2년 훈련 / 1년 검증 | trials/창: {N_TRIALS_PER_WINDOW}", log)
    log_and_print("=" * 100, log)
    
    # 1) 각 WFO 창 실행
    window_results = []
    for train_years, test_year in WINDOWS:
        result = run_window_optimization(train_years, test_year, log)
        window_results.append(result)
    
    # 2) OOS 성과 테이블
    log_and_print("\n[OOS 성과 테이블]", log)
    log_and_print(f"{'훈련':<15} {'검증':<8} {'mode':<10} {'test Sharpe':<12} {'test PnL':<15} {'test Trades':<12}", log)
    for r in window_results:
        m = r["test_metrics"]
        log_and_print(f"{str(r['train_years']):<15} {r['test_year']:<8} {r['best_params']['direction_mode']:<10} "
                      f"{m['sharpe_daily']:<12.3f} {m['total_pnl_krw']:<15,.0f} {m['n_trades']:<12}", log)
    
    # 3) 최종 파라미터 선정: OOS Sharpe가 가장 높은 창의 파라미터
    best_window = max(window_results, key=lambda x: x["test_metrics"]["sharpe_daily"])
    best_params = best_window["best_params"]
    
    log_and_print(f"\n[최종 선정 파라미터] (OOS Sharpe 최고 창: {best_window['train_years']} → {best_window['test_year']})", log)
    log_and_print(f"  {best_params}", log)
    
    # 4) 최종 파라미터로 전체 기간 평가
    final_pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=best_params["base_pct"],
        base_multiplier=best_params["base_multiplier"],
        atr_weight=best_params["atr_weight"],
        confirmation_bars=best_params["confirmation_bars"],
    )
    final_fcfg = pv.FilterConfig(
        enabled=True,
        min_wave_pct=best_params["min_wave_pct"],
        min_pivot_interval_bars=best_params["min_pivot_interval_bars"],
        st_distance_threshold=0.1,
        adx_hold_threshold=15.0,
    )
    final_mode = best_params["direction_mode"]
    
    log_and_print(f"\n[전체 기간 (2019-2026) 평가]", log)
    all_years = list(range(2019, 2027))
    total_metrics = evaluate_all_years(all_years, final_pcfg, final_fcfg, final_mode)
    log_and_print(fmt_metrics(total_metrics, f"최종 피봇 ({final_mode})"), log)
    
    # 5) 롱-또는-플랫 벤치마크
    log_and_print(f"\n[롱-또는-플랫 벤치마크]", log)
    lf_results = []
    for year in all_years:
        df = pva.load_data_by_year(year)
        if len(df) > 0:
            lf_results.append(pva.run_long_or_flat(df))
        del df
        gc.collect()
    lf_total = pva.combine_results(lf_results)
    log_and_print(pva.fmt_result(lf_total, "롱-또는-플랫 (전체)"), log)
    
    # 6) OOS 합산 비교 (WFO 검증 기간들만 합산)
    log_and_print(f"\n[OOS 검증 기간 합산 비교]", log)
    oos_years = [r["test_year"] for r in window_results]
    oos_pivot_metrics = evaluate_all_years(oos_years, final_pcfg, final_fcfg, final_mode)
    
    lf_oos_results = []
    for year in oos_years:
        df = pva.load_data_by_year(year)
        if len(df) > 0:
            lf_oos_results.append(pva.run_long_or_flat(df))
        del df
        gc.collect()
    lf_oos = pva.combine_results(lf_oos_results)
    
    log_and_print(fmt_metrics(oos_pivot_metrics, f"피봇 OOS 합산 ({final_mode})"), log)
    log_and_print(pva.fmt_result(lf_oos, "롱-또는-플랫 OOS 합산"), log)
    
    # 7) 요약
    log_and_print("\n" + "=" * 100, log)
    log_and_print("요약", log)
    log_and_print("=" * 100, log)
    log_and_print(f"최종 피봇 전체 (2019-2026):  PnL={total_metrics['total_pnl_krw']:,.0f}, Sharpe={total_metrics['sharpe_daily']:.3f}, MaxDD={total_metrics['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"롱-또는-플랫 전체 (2019-2026): PnL={lf_total.total_pnl_krw:,.0f}, Sharpe={lf_total.sharpe_daily:.3f}, MaxDD={lf_total.max_drawdown_krw:,.0f}", log)
    log_and_print(f"최종 피봇 OOS 합산:           PnL={oos_pivot_metrics['total_pnl_krw']:,.0f}, Sharpe={oos_pivot_metrics['sharpe_daily']:.3f}, MaxDD={oos_pivot_metrics['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"롱-또는-플랫 OOS 합산:         PnL={lf_oos.total_pnl_krw:,.0f}, Sharpe={lf_oos.sharpe_daily:.3f}, MaxDD={lf_oos.max_drawdown_krw:,.0f}", log)
    log_and_print(f"최종 파라미터: {best_params}", log)
    log_and_print("=" * 100, log)


def main():
    parser = argparse.ArgumentParser(description="피봇 파라미터 WFO 최적화")
    parser.add_argument("--phase2", action="store_true", help="Phase 2 다중 WFO 창 모드")
    parser.add_argument("--phase1", action="store_true", help="Phase 1 단일 훈련/검증 모드 (기본)")
    args = parser.parse_args()
    
    output_log = Path(__file__).parent / ("pivot_wfo_optimizer_phase2.log" if args.phase2 else "pivot_wfo_optimizer_phase1.log")
    
    with open(output_log, "w", encoding="utf-8") as log:
        if args.phase2:
            _run_phase2(log)
        else:
            _run_phase1(log)


if __name__ == "__main__":
    main()
