# -*- coding: utf-8 -*-
"""
시장 레짐별 피봇 파라미터 WFO (Out-of-Sample 검증)

훈련: 2019-2023년, 검증: 2024-2026년
각 기간의 데이터를 bull/bear/neutral로 분류한 뒤,
훈련 레짐 데이터로 최적 파라미터를 찾고 검증 레짐 데이터로 평가한다.
"""
import sys
import gc
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import optuna
import regime_intraday_v2 as rg
import pivot_regime_optimizer as pro
import pivot_viability_analysis as pva
import pivot_optuna_v2 as pv

OUTPUT_LOG = Path(__file__).parent / "pivot_regime_wfo_optimizer.log"
N_TRIALS = 50
SEED = 42
TRAIN_YEARS = list(range(2019, 2024))
TEST_YEARS = list(range(2024, 2027))

REGIME_LABELS = {1: "bull", -1: "bear", 0: "neutral"}


def log_and_print(msg: str, log_file):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


def build_regime_cache(years: List[int]) -> Dict[int, pd.DataFrame]:
    """pivot_regime_optimizer의 캐시 구축 함수 재사용."""
    return pro.build_regime_cache(years)


def evaluate_cached_regime(df_regime: pd.DataFrame,
                           pcfg: pv.HybridAdaptivePivotConfig, fcfg: pv.FilterConfig,
                           direction_mode: str) -> Dict[str, Any]:
    """캐시된 레짐 데이터로 백테스트."""
    if len(df_regime) == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl_krw": 0.0,
            "total_pnl_pts": 0.0, "expectancy_krw": 0.0, "expectancy_pts": 0.0,
            "profit_factor": 0.0, "sharpe_daily": -1.0, "max_drawdown_krw": 0.0,
        }
    
    res = pva.run_pivot(df_regime, pcfg, fcfg, direction_mode)
    
    if res.n_trades == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl_krw": 0.0,
            "total_pnl_pts": 0.0, "expectancy_krw": 0.0, "expectancy_pts": 0.0,
            "profit_factor": 0.0, "sharpe_daily": -1.0, "max_drawdown_krw": 0.0,
        }
    
    net_pts = res.trades["net_pts"] if res.trades is not None else pd.Series(dtype=float)
    gross_win_pts = net_pts[net_pts > 0].sum() if len(net_pts) > 0 else 0.0
    gross_loss_pts = -net_pts[net_pts < 0].sum() if len(net_pts) > 0 else 0.0
    profit_factor = float(gross_win_pts / gross_loss_pts) if gross_loss_pts > 0 else float("inf") if gross_win_pts > 0 else 0.0
    
    if res.trades is not None and len(res.trades) > 0:
        res.trades["exit_date"] = pd.to_datetime(res.trades["exit_time"]).dt.date
        daily = res.trades.groupby("exit_date")["net_krw"].sum()
        if len(daily) >= 2 and daily.std(ddof=1) > 0:
            sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(pva.BT.annualization))
        else:
            sharpe = 0.0
        equity = res.trades["net_krw"].cumsum()
        max_dd = float((equity - equity.cummax()).min())
    else:
        sharpe = 0.0
        max_dd = 0.0
    
    return {
        "n_trades": res.n_trades,
        "win_rate": res.win_rate,
        "total_pnl_krw": res.total_pnl_krw,
        "total_pnl_pts": res.total_pnl_pts,
        "expectancy_krw": res.expectancy_krw,
        "expectancy_pts": res.expectancy_pts,
        "profit_factor": profit_factor,
        "sharpe_daily": sharpe,
        "max_drawdown_krw": max_dd,
    }


def fmt_metrics(metrics: Dict[str, Any], label: str = "") -> str:
    return (f"{label:<45} | 거래={metrics['n_trades']:>4} | 승률={metrics['win_rate']:>6.2f}% | "
            f"PnL={metrics['total_pnl_krw']:>13,.0f} | Sharpe={metrics['sharpe_daily']:>7.3f} | "
            f"MaxDD={metrics['max_drawdown_krw']:>13,.0f}")


def make_objective(train_cache: Dict[int, pd.DataFrame], regime_value: int, log_file):
    """훈련 레짐 데이터에 대한 Optuna objective."""
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
        
        df_train = train_cache.get(regime_value, pd.DataFrame())
        metrics = evaluate_cached_regime(df_train, pcfg, fcfg, direction_mode)
        
        if metrics["n_trades"] < 20:
            return -1.0
        
        log_and_print(f"  {REGIME_LABELS[regime_value]} trial {trial.number}: {direction_mode} | "
                      f"base_pct={pcfg.base_pct:.3f}, mult={pcfg.base_multiplier:.2f}, "
                      f"atr_w={pcfg.atr_weight:.2f}, conf={pcfg.confirmation_bars}, "
                      f"wave={fcfg.min_wave_pct:.3f}, interval={fcfg.min_pivot_interval_bars} | "
                      f"train Sharpe={metrics['sharpe_daily']:.3f}, Trades={metrics['n_trades']}", log_file)
        
        return metrics["sharpe_daily"]
    
    return objective


def optimize_regime(train_cache: Dict[int, pd.DataFrame], test_cache: Dict[int, pd.DataFrame],
                    regime_value: int, log_file) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """특정 레짐을 훈련 데이터로 최적화하고 검증 데이터로 평가."""
    regime_name = REGIME_LABELS[regime_value]
    log_and_print(f"\n[{regime_name.upper()} 레짐 WFO]", log_file)
    
    df_train = train_cache.get(regime_value, pd.DataFrame())
    df_test = test_cache.get(regime_value, pd.DataFrame())
    log_and_print(f"  훈련 데이터: {len(df_train)} 봉 | 검증 데이터: {len(df_test)} 봉", log_file)
    
    if len(df_train) == 0:
        log_and_print(f"  훈련 데이터가 없어 스킵", log_file)
        return {}, {"n_trades": 0, "sharpe_daily": -1.0, "total_pnl_krw": 0.0}, {"n_trades": 0, "sharpe_daily": -1.0, "total_pnl_krw": 0.0}
    
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    
    study.optimize(make_objective(train_cache, regime_value, log_file),
                   n_trials=N_TRIALS, show_progress_bar=True)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    log_and_print(f"  최적 파라미터: {best_params}", log_file)
    log_and_print(f"  훈련 Sharpe: {best_trial.value:.3f}", log_file)
    
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
    
    train_metrics = evaluate_cached_regime(df_train, best_pcfg, best_fcfg, best_mode)
    test_metrics = evaluate_cached_regime(df_test, best_pcfg, best_fcfg, best_mode)
    
    log_and_print(fmt_metrics(train_metrics, f"  {regime_name} 훈련"), log_file)
    log_and_print(fmt_metrics(test_metrics, f"  {regime_name} 검증"), log_file)
    
    return best_params, train_metrics, test_metrics


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        log_and_print("=" * 100, log)
        log_and_print("시장 레짐별 피봇 파라미터 WFO (OOS 검증)", log)
        log_and_print(f"훈련: {TRAIN_YEARS} | 검증: {TEST_YEARS}", log)
        log_and_print("=" * 100, log)
        
        # 훈련/검증 레짐 데이터 캐시 구축
        log_and_print(f"\n[훈련 데이터 캐시 구축...]", log)
        train_cache = build_regime_cache(TRAIN_YEARS)
        for regime_value, df in train_cache.items():
            log_and_print(f"  {REGIME_LABELS[regime_value]}: {len(df)} 봉", log)
        
        log_and_print(f"\n[검증 데이터 캐시 구축...]", log)
        test_cache = build_regime_cache(TEST_YEARS)
        for regime_value, df in test_cache.items():
            log_and_print(f"  {REGIME_LABELS[regime_value]}: {len(df)} 봉", log)
        
        # 레짐별 WFO
        regime_results = {}
        for regime_value in [1, -1, 0]:
            best_params, train_metrics, test_metrics = optimize_regime(train_cache, test_cache, regime_value, log)
            regime_results[regime_value] = {
                "name": REGIME_LABELS[regime_value],
                "best_params": best_params,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }
        
        # 검증 기간 레짐별 파라미터 적용 합산
        log_and_print(f"\n[검증 기간 레짐별 파라미터 적용 합산]", log)
        total_pnl_krw = 0.0
        total_trades = 0
        daily_pnl_list: List[pd.Series] = []
        equity_list: List[pd.Series] = []
        
        for regime_value, r in regime_results.items():
            df_test = test_cache.get(regime_value, pd.DataFrame())
            if len(df_test) == 0 or not r["best_params"]:
                continue
            
            params = r["best_params"]
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
            mode = params["direction_mode"]
            
            res = pva.run_pivot(df_test, pcfg, fcfg, mode)
            total_pnl_krw += res.total_pnl_krw
            total_trades += res.n_trades
            
            if res.trades is not None and len(res.trades) > 0:
                res.trades["exit_date"] = pd.to_datetime(res.trades["exit_time"]).dt.date
                daily = res.trades.groupby("exit_date")["net_krw"].sum()
                daily_pnl_list.append(daily)
                equity_list.append(res.trades["net_krw"].cumsum())
            
            del res
            gc.collect()
        
        if daily_pnl_list:
            combined_daily = pd.concat(daily_pnl_list)
            if len(combined_daily) >= 2 and combined_daily.std(ddof=1) > 0:
                oos_sharpe = float(combined_daily.mean() / combined_daily.std(ddof=1) * math.sqrt(pva.BT.annualization))
            else:
                oos_sharpe = 0.0
        else:
            oos_sharpe = 0.0
        
        if equity_list:
            combined_equity = pd.concat(equity_list)
            oos_max_dd = float((combined_equity - combined_equity.cummax()).min())
        else:
            oos_max_dd = 0.0
        
        log_and_print(f"레짐별 WFO 검증 합산 | 거래={total_trades} | PnL={total_pnl_krw:,.0f} | Sharpe={oos_sharpe:.3f} | MaxDD={oos_max_dd:,.0f}", log)
        
        # 롱-또는-플랫 검증 기간 비교
        log_and_print(f"\n[롱-또는-플랫 검증 기간 벤치마크]", log)
        lf_test_results = []
        for year in TEST_YEARS:
            df = pva.load_data_by_year(year)
            if len(df) > 0:
                lf_test_results.append(pva.run_long_or_flat(df))
            del df
            gc.collect()
        lf_test = pva.combine_results(lf_test_results)
        log_and_print(pva.fmt_result(lf_test, "롱-또는-플랫 검증 (2024-2026)"), log)
        
        # 요약
        log_and_print("\n" + "=" * 100, log)
        log_and_print("요약", log)
        log_and_print("=" * 100, log)
        for regime_value, r in regime_results.items():
            tm = r["train_metrics"]
            vm = r["test_metrics"]
            log_and_print(f"{r['name'].upper():<10} | 훈련: 거래={tm['n_trades']:>4}, PnL={tm['total_pnl_krw']:>12,.0f}, Sharpe={tm['sharpe_daily']:>6.3f}", log)
            log_and_print(f"           | 검증: 거래={vm['n_trades']:>4}, PnL={vm['total_pnl_krw']:>12,.0f}, Sharpe={vm['sharpe_daily']:>6.3f}", log)
            log_and_print(f"  params: {r['best_params']}", log)
        log_and_print(f"\n레짐별 WFO 검증 합산 (2024-2026): PnL={total_pnl_krw:,.0f}, Sharpe={oos_sharpe:.3f}, MaxDD={oos_max_dd:,.0f}", log)
        log_and_print(f"롱-또는-플랫 검증 (2024-2026):     PnL={lf_test.total_pnl_krw:,.0f}, Sharpe={lf_test.sharpe_daily:.3f}, MaxDD={lf_test.max_drawdown_krw:,.0f}", log)
        log_and_print("=" * 100, log)


if __name__ == "__main__":
    main()
