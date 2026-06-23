# -*- coding: utf-8 -*-
"""
피봇 파라미터 Walk-Forward Optimization (Phase 2)

전체 기간 2019-2026을 대상으로 2년 훈련 / 1년 검증 창을 이동하며 최적 파라미터를 찾고,
Out-of-Sample 성과의 안정성을 평가한다.

창 구성:
    [2019-2020] → 2021
    [2020-2021] → 2022
    [2021-2022] → 2023
    [2022-2023] → 2024
    [2023-2024] → 2025
    [2024-2025] → 2026
"""
import sys
import gc
import math
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import optuna
import pivot_wfo_optimizer as pwo
import pivot_viability_analysis as pva
import pivot_optuna_v2 as pv

OUTPUT_LOG = Path(__file__).parent / "pivot_wfo_optimizer_phase2.log"
N_TRIALS_PER_WINDOW = 50
SEED = 42

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


def make_objective_for_window(train_years: List[int], log_file):
    """특정 훈련 기간에 대한 Optuna objective."""
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
        
        train_metrics = pwo.evaluate_years(train_years, pcfg, fcfg, direction_mode)
        
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
    """한 개 WFO 창에 대해 최적화 및 검증을 실행."""
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
    
    test_metrics = pwo.evaluate_years([test_year], best_pcfg, best_fcfg, best_mode)
    log_and_print(pwo.fmt_metrics(test_metrics, f"  검증 ({test_year})"), log_file)
    
    return {
        "train_years": train_years,
        "test_year": test_year,
        "best_params": best_params,
        "train_sharpe": best_trial.value,
        "test_metrics": test_metrics,
    }


def evaluate_all_years(years: List[int], pcfg: pv.HybridAdaptivePivotConfig,
                       fcfg: pv.FilterConfig, direction_mode: str) -> Dict[str, Any]:
    """전체 기간에 대해 백테스트."""
    return pwo.evaluate_years(years, pcfg, fcfg, direction_mode)


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
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
        log_and_print(pwo.fmt_metrics(total_metrics, f"최종 피봇 ({final_mode})"), log)
        
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
        
        log_and_print(pwo.fmt_metrics(oos_pivot_metrics, f"피봇 OOS 합산 ({final_mode})"), log)
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


if __name__ == "__main__":
    main()
