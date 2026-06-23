# -*- coding: utf-8 -*-
"""
시장 레짐별 피봇 파라미터 최적화

- 기본 모드: 전체 기간(2019-2026)을 일봉 MA20/60 레짐(bull/bear/neutral)으로 분류한 뒤,
  각 레짐별 데이터만으로 피봇 파라미터를 최적화한다.
- WFO 모드(--wfo): 훈련 기간(2019-2023)으로 레짐별 최적화하고,
  검증 기간(2024-2026)으로 Out-of-Sample 성과를 평가한다.
"""
import sys
import gc
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import optuna
import regime_intraday_v2 as rg
import pivot_wfo_optimizer as pwo
import pivot_viability_analysis as pva
import pivot_optuna_v2 as pv

OUTPUT_LOG = Path(__file__).parent / "pivot_regime_optimizer.log"
N_TRIALS = 50
SEED = 42
ALL_YEARS = list(range(2019, 2027))

# WFO 모드 기본 설정
TRAIN_YEARS = list(range(2019, 2024))
TEST_YEARS = list(range(2024, 2027))

REGIME_LABELS = {1: "bull", -1: "bear", 0: "neutral"}


def log_and_print(msg: str, log_file):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


def _filter_df_by_regime(df: pd.DataFrame, regime_signal: pd.Series, regime_value: int) -> pd.DataFrame:
    """데이터프레임에서 특정 레짐의 거래일만 남긴다."""
    tday = pv.trading_day_key(df.index, pva.BT.session_boundary_hour)
    s = pd.Series(np.arange(len(df)), index=df.index)
    g_last = s.groupby(tday).last()
    g_first = s.groupby(tday).first()
    end_times = df.index[g_last.values]
    
    daily_regime = regime_signal.reindex(end_times).fillna(0).astype(int).values
    
    tday_to_regime = dict(zip(end_times, daily_regime))
    bar_regime = np.array([tday_to_regime[et] for et in end_times])
    
    regime_per_bar = np.zeros(len(df), dtype=int)
    for i, (day, end_idx) in enumerate(g_last.items()):
        start_idx = int(g_first[day])
        regime_per_bar[start_idx:end_idx + 1] = bar_regime[i]
    
    return df.iloc[regime_per_bar == regime_value].copy()


def build_regime_cache(years: List[int]) -> Dict[int, pd.DataFrame]:
    """연도별로 데이터를 한 번만 로드하여 레짐별 캐시를 구축."""
    regime_signals = []
    for year in years:
        df = pva.load_data_by_year(year)
        if len(df) == 0:
            continue
        daily = rg.to_daily(df, pva.BT.session_boundary_hour)
        signal = rg.daily_regime_signal(daily, regime_method="ma", ma_short=20, ma_long=60)
        regime_signals.append(signal)
        del daily
        gc.collect()
    full_regime_signal = pd.concat(regime_signals).sort_index()
    
    cache = {1: [], -1: [], 0: []}
    for year in years:
        df = pva.load_data_by_year(year)
        if len(df) == 0:
            continue
        for regime_value in [1, -1, 0]:
            df_regime = _filter_df_by_regime(df, full_regime_signal, regime_value)
            if len(df_regime) > 0:
                cache[regime_value].append(df_regime)
        del df
        gc.collect()
    
    return {k: pd.concat(v).sort_index() if v else pd.DataFrame() for k, v in cache.items()}


REGIME_CACHE: Dict[int, pd.DataFrame] = {}


def evaluate_regime_df(df_regime: pd.DataFrame,
                       pcfg: pv.HybridAdaptivePivotConfig, fcfg: pv.FilterConfig,
                       direction_mode: str) -> Dict[str, Any]:
    """주어진 레짐 데이터로 백테스트."""
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


def evaluate_regime(regime_value: int,
                    pcfg: pv.HybridAdaptivePivotConfig, fcfg: pv.FilterConfig,
                    direction_mode: str) -> Dict[str, Any]:
    """캐시된 레짐 데이터로 백테스트."""
    df_regime = REGIME_CACHE.get(regime_value, pd.DataFrame())
    return evaluate_regime_df(df_regime, pcfg, fcfg, direction_mode)


def fmt_metrics(metrics: Dict[str, Any], label: str = "") -> str:
    return (f"{label:<45} | 거래={metrics['n_trades']:>4} | 승률={metrics['win_rate']:>6.2f}% | "
            f"PnL={metrics['total_pnl_krw']:>13,.0f} | Sharpe={metrics['sharpe_daily']:>7.3f} | "
            f"MaxDD={metrics['max_drawdown_krw']:>13,.0f}")


def make_objective(regime_value: int, log_file):
    """특정 레짐에 대한 Optuna objective."""
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
        
        metrics = evaluate_regime(regime_value, pcfg, fcfg, direction_mode)
        
        if metrics["n_trades"] < 20:
            return -1.0
        
        log_and_print(f"  {REGIME_LABELS[regime_value]} trial {trial.number}: {direction_mode} | "
                      f"base_pct={pcfg.base_pct:.3f}, mult={pcfg.base_multiplier:.2f}, "
                      f"atr_w={pcfg.atr_weight:.2f}, conf={pcfg.confirmation_bars}, "
                      f"wave={fcfg.min_wave_pct:.3f}, interval={fcfg.min_pivot_interval_bars} | "
                      f"Sharpe={metrics['sharpe_daily']:.3f}, Trades={metrics['n_trades']}", log_file)
        
        return metrics["sharpe_daily"]
    
    return objective


def optimize_regime(regime_value: int, log_file) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """특정 레짐을 최적화하고 결과 반환."""
    regime_name = REGIME_LABELS[regime_value]
    log_and_print(f"\n[{regime_name.upper()} 레짐 최적화 시작]", log_file)
    
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    
    study.optimize(make_objective(regime_value, log_file),
                   n_trials=N_TRIALS, show_progress_bar=True)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    log_and_print(f"  최적 파라미터: {best_params}", log_file)
    log_and_print(f"  최적 Sharpe: {best_trial.value:.3f}", log_file)
    
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
    
    metrics = evaluate_regime(regime_value, best_pcfg, best_fcfg, best_mode)
    log_and_print(fmt_metrics(metrics, f"  {regime_name} 최종 평가"), log_file)
    
    return best_params, metrics


def make_objective_df(df_regime: pd.DataFrame, regime_name: str, log_file):
    """주어진 레짐 데이터에 대한 Optuna objective."""
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
        
        metrics = evaluate_regime_df(df_regime, pcfg, fcfg, direction_mode)
        
        if metrics["n_trades"] < 20:
            return -1.0
        
        log_and_print(f"  {regime_name} trial {trial.number}: {direction_mode} | "
                      f"base_pct={pcfg.base_pct:.3f}, mult={pcfg.base_multiplier:.2f}, "
                      f"atr_w={pcfg.atr_weight:.2f}, conf={pcfg.confirmation_bars}, "
                      f"wave={fcfg.min_wave_pct:.3f}, interval={fcfg.min_pivot_interval_bars} | "
                      f"Sharpe={metrics['sharpe_daily']:.3f}, Trades={metrics['n_trades']}", log_file)
        
        return metrics["sharpe_daily"]
    
    return objective


def optimize_regime_wfo(train_cache: Dict[int, pd.DataFrame], test_cache: Dict[int, pd.DataFrame],
                        regime_value: int, log_file) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """특정 레짐을 훈련 데이터로 최적화하고 검증 데이터로 평가 (WFO 모드)."""
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
    
    study.optimize(make_objective_df(df_train, regime_name, log_file),
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
    
    train_metrics = evaluate_regime_df(df_train, best_pcfg, best_fcfg, best_mode)
    test_metrics = evaluate_regime_df(df_test, best_pcfg, best_fcfg, best_mode)
    
    log_and_print(fmt_metrics(train_metrics, f"  {regime_name} 훈련"), log_file)
    log_and_print(fmt_metrics(test_metrics, f"  {regime_name} 검증"), log_file)
    
    return best_params, train_metrics, test_metrics


def apply_regime_params(regime_results: Dict[int, Dict[str, Any]]) -> Tuple[float, int, float, float]:
    """레짐별 최적 파라미터를 캐시된 데이터에 적용하여 전체 백테스트."""
    total_pnl_krw = 0.0
    total_trades = 0
    daily_pnl_list: List[pd.Series] = []
    equity_list: List[pd.Series] = []
    
    for regime_value, r in regime_results.items():
        df_regime = REGIME_CACHE.get(regime_value, pd.DataFrame())
        if len(df_regime) == 0:
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
        
        res = pva.run_pivot(df_regime, pcfg, fcfg, mode)
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
    
    return total_pnl_krw, total_trades, sharpe, max_dd


def _run_default_mode(log):
    """기본 모드: 전체 기간 레짐별 최적화."""
    global REGIME_CACHE
    
    log_and_print("=" * 100, log)
    log_and_print("시장 레짐별 피봇 파라미터 최적화", log)
    log_and_print(f"대상 기간: {ALL_YEARS} | trials/레짐: {N_TRIALS}", log)
    log_and_print("=" * 100, log)
    
    # 레짐별 데이터 캐시 구축 (한 번만 로드)
    log_and_print(f"\n[데이터 캐시 구축 중...]", log)
    REGIME_CACHE = build_regime_cache(ALL_YEARS)
    for regime_value, df in REGIME_CACHE.items():
        log_and_print(f"  {REGIME_LABELS[regime_value]}: {len(df)} 봉", log)
    
    # 레짐별 최적화
    regime_results = {}
    for regime_value in [1, -1, 0]:
        best_params, metrics = optimize_regime(regime_value, log)
        regime_results[regime_value] = {
            "name": REGIME_LABELS[regime_value],
            "best_params": best_params,
            "metrics": metrics,
        }
    
    # 레짐별 최적 파라미터로 전체 기간 백테스트
    log_and_print(f"\n[레짐별 파라미터 적용 - 전체 기간 백테스트]", log)
    total_pnl_krw, total_trades, sharpe, max_dd = apply_regime_params(regime_results)
    log_and_print(f"레짐별 최적 파라미터 적용 | 거래={total_trades} | PnL={total_pnl_krw:,.0f} | Sharpe={sharpe:.3f} | MaxDD={max_dd:,.0f}", log)
    
    # 롱-또는-플랫 비교
    log_and_print(f"\n[롱-또는-플랫 벤치마크]", log)
    lf_results = []
    for year in ALL_YEARS:
        df = pva.load_data_by_year(year)
        if len(df) > 0:
            lf_results.append(pva.run_long_or_flat(df))
        del df
        gc.collect()
    lf_total = pva.combine_results(lf_results)
    log_and_print(pva.fmt_result(lf_total, "롱-또는-플랫 (전체)"), log)
    
    # 요약
    log_and_print("\n" + "=" * 100, log)
    log_and_print("요약", log)
    log_and_print("=" * 100, log)
    for regime_value, r in regime_results.items():
        m = r["metrics"]
        log_and_print(f"{r['name'].upper():<10} | 거래={m['n_trades']:>4} | PnL={m['total_pnl_krw']:>13,.0f} | Sharpe={m['sharpe_daily']:>7.3f} | MaxDD={m['max_drawdown_krw']:>13,.0f}", log)
        log_and_print(f"  params: {r['best_params']}", log)
    log_and_print(f"\n레짐별 파라미터 적용 전체: PnL={total_pnl_krw:,.0f}, Sharpe={sharpe:.3f}, MaxDD={max_dd:,.0f}", log)
    log_and_print(f"롱-또는-플랫 전체:           PnL={lf_total.total_pnl_krw:,.0f}, Sharpe={lf_total.sharpe_daily:.3f}, MaxDD={lf_total.max_drawdown_krw:,.0f}", log)
    log_and_print("=" * 100, log)


def _run_wfo_mode(log, train_years: List[int], test_years: List[int]):
    """WFO 모드: 훈련/검증 기간 분할 레짐별 최적화."""
    log_and_print("=" * 100, log)
    log_and_print("시장 레짐별 피봇 파라미터 WFO (OOS 검증)", log)
    log_and_print(f"훈련: {train_years} | 검증: {test_years} | trials/레짐: {N_TRIALS}", log)
    log_and_print("=" * 100, log)
    
    # 훈련/검증 레짐 데이터 캐시 구축
    log_and_print(f"\n[훈련 데이터 캐시 구축...]", log)
    train_cache = build_regime_cache(train_years)
    for regime_value, df in train_cache.items():
        log_and_print(f"  {REGIME_LABELS[regime_value]}: {len(df)} 봉", log)
    
    log_and_print(f"\n[검증 데이터 캐시 구축...]", log)
    test_cache = build_regime_cache(test_years)
    for regime_value, df in test_cache.items():
        log_and_print(f"  {REGIME_LABELS[regime_value]}: {len(df)} 봉", log)
    
    # 레짐별 WFO
    regime_results = {}
    for regime_value in [1, -1, 0]:
        best_params, train_metrics, test_metrics = optimize_regime_wfo(train_cache, test_cache, regime_value, log)
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
    for year in test_years:
        df = pva.load_data_by_year(year)
        if len(df) > 0:
            lf_test_results.append(pva.run_long_or_flat(df))
        del df
        gc.collect()
    lf_test = pva.combine_results(lf_test_results)
    log_and_print(pva.fmt_result(lf_test, f"롱-또는-플랫 검증 ({min(test_years)}-{max(test_years)})"), log)
    
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
    log_and_print(f"\n레짐별 WFO 검증 합산 ({min(test_years)}-{max(test_years)}): PnL={total_pnl_krw:,.0f}, Sharpe={oos_sharpe:.3f}, MaxDD={oos_max_dd:,.0f}", log)
    log_and_print(f"롱-또는-플랫 검증 ({min(test_years)}-{max(test_years)}):     PnL={lf_test.total_pnl_krw:,.0f}, Sharpe={lf_test.sharpe_daily:.3f}, MaxDD={lf_test.max_drawdown_krw:,.0f}", log)
    log_and_print("=" * 100, log)


def main():
    parser = argparse.ArgumentParser(description="시장 레짐별 피봇 파라미터 최적화")
    parser.add_argument("--wfo", action="store_true", help="WFO 모드 (훈련/검증 분할)")
    parser.add_argument("--train-years", type=str, help="쉼표로 구분된 훈련 연도 (예: 2019,2020,2021,2022,2023)")
    parser.add_argument("--test-years", type=str, help="쉼표로 구분된 검증 연도 (예: 2024,2025,2026)")
    args = parser.parse_args()
    
    output_log = Path(__file__).parent / ("pivot_regime_wfo_optimizer.log" if args.wfo else "pivot_regime_optimizer.log")
    
    train_years = [int(y.strip()) for y in args.train_years.split(",")] if args.train_years else TRAIN_YEARS
    test_years = [int(y.strip()) for y in args.test_years.split(",")] if args.test_years else TEST_YEARS
    
    with open(output_log, "w", encoding="utf-8") as log:
        if args.wfo:
            _run_wfo_mode(log, train_years, test_years)
        else:
            _run_default_mode(log)


if __name__ == "__main__":
    main()
