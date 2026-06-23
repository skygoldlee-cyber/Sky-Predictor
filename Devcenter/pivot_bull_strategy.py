# -*- coding: utf-8 -*-
"""
BULL 레짐 전용 피봇 롱 전략

조건: 일봉 MA20 > MA60 (BULL 레짐)일 때만 피봇 반전 롱 진입
파라미터: 레짐별 WFO에서 도출된 BULL 최적 파라미터
"""
import sys
import gc
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import optuna
import regime_intraday_v2 as rg
import pivot_regime_optimizer as pro
import pivot_viability_analysis as pva
import pivot_optuna_v2 as pv

OUTPUT_LOG = Path(__file__).parent / "pivot_bull_strategy.log"
ALL_YEARS = list(range(2019, 2026))

# 레짐별 WFO에서 도출된 BULL 최적 파라미터 (초기값)
DEFAULT_BULL_PARAMS = {
    "base_pct": 1.4260371592457055,
    "base_multiplier": 1.5101170928896888,
    "atr_weight": 0.2345611522015273,
    "confirmation_bars": 4,
    "min_wave_pct": 0.15646519751630317,
    "min_pivot_interval_bars": 23,
    "direction_mode": "long_only",
}

# 하이브리드 전략용 BULL 레짐 피봇 롱 파라미터 (2019-2025 최적)
HYBRID_BULL_PARAMS = {
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
    return (f"{label:<45} | 거래={metrics['n_trades']:>4} | 승률={metrics['win_rate']:>6.2f}% | "
            f"PnL={metrics['total_pnl_krw']:>13,.0f} | Sharpe={metrics['sharpe_daily']:>7.3f} | "
            f"MaxDD={metrics['max_drawdown_krw']:>13,.0f}")


def get_daily_regime_for_years(years: List[int]) -> pd.Series:
    """연도 리스트에 대해 일봉 MA20/60 레짐 신호 생성."""
    regime_signals = []
    for year in years:
        df = pva.load_data_by_year(year)
        if len(df) == 0:
            continue
        daily = rg.to_daily(df, pva.BT.session_boundary_hour)
        signal = rg.daily_regime_signal(daily, regime_method="ma", ma_short=20, ma_long=60)
        regime_signals.append(signal)
        del df, daily
        gc.collect()
    return pd.concat(regime_signals).sort_index()


def evaluate_bull_strategy(years: List[int], params: Dict[str, Any]) -> Dict[str, Any]:
    """BULL 레짐에서만 피봇 롱 전략 실행."""
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
    
    total_pnl_krw = 0.0
    total_pnl_pts = 0.0
    total_trades = 0
    weighted_win_rate = 0.0
    gross_win_pts = 0.0
    gross_loss_pts = 0.0
    daily_pnl_list: List[pd.Series] = []
    equity_list: List[pd.Series] = []
    
    regime_signal = get_daily_regime_for_years(years)
    
    for year in years:
        df = pva.load_data_by_year(year)
        if len(df) == 0:
            continue
        df_bull = pro._filter_df_by_regime(df, regime_signal, 1)
        if len(df_bull) == 0:
            del df, df_bull
            gc.collect()
            continue
        
        res = pva.run_pivot(df_bull, pcfg, fcfg, mode)
        
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
    }


def combine_trade_results(results_list: List[pv.BacktestResult]) -> Dict[str, Any]:
    """여러 BacktestResult의 거래 내역을 합쳐서 지표 계산."""
    valid_results = [r for r in results_list if r.trades is not None and len(r.trades) > 0]
    if not valid_results:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl_krw": 0.0,
            "total_pnl_pts": 0.0, "expectancy_krw": 0.0, "expectancy_pts": 0.0,
            "profit_factor": 0.0, "sharpe_daily": 0.0, "max_drawdown_krw": 0.0,
        }
    
    all_trades = pd.concat([r.trades for r in valid_results], ignore_index=True)
    all_trades["exit_date"] = pd.to_datetime(all_trades["exit_time"]).dt.date
    
    total_pnl_krw = float(all_trades["net_krw"].sum())
    total_pnl_pts = float(all_trades["net_pts"].sum())
    n_trades = len(all_trades)
    win_rate = float((all_trades["net_pts"] > 0).mean() * 100)
    expectancy_krw = total_pnl_krw / n_trades
    expectancy_pts = total_pnl_pts / n_trades
    
    wins = all_trades[all_trades["net_pts"] > 0]["net_pts"].sum()
    losses = -all_trades[all_trades["net_pts"] < 0]["net_pts"].sum()
    profit_factor = float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0
    
    daily = all_trades.groupby("exit_date")["net_krw"].sum()
    if len(daily) >= 2 and daily.std(ddof=1) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(pva.BT.annualization))
    else:
        sharpe = 0.0
    
    equity = all_trades["net_krw"].cumsum()
    max_dd = float((equity - equity.cummax()).min())
    
    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "total_pnl_krw": total_pnl_krw,
        "total_pnl_pts": total_pnl_pts,
        "expectancy_krw": expectancy_krw,
        "expectancy_pts": expectancy_pts,
        "profit_factor": profit_factor,
        "sharpe_daily": sharpe,
        "max_drawdown_krw": max_dd,
    }


def evaluate_hybrid_strategy(years: List[int]) -> Dict[str, Any]:
    """BULL 피봇 롱 + 롱-또는-플랫 하이브리드 전략 평가."""
    regime_signal = get_daily_regime_for_years(years)
    
    bull_results = []
    lf_results = []
    
    for year in years:
        df = pva.load_data_by_year(year)
        if len(df) == 0:
            continue
        
        # BULL 레짐 데이터에서 피봇 롱
        df_bull = pro._filter_df_by_regime(df, regime_signal, 1)
        if len(df_bull) > 0:
            pcfg = pv.HybridAdaptivePivotConfig(
                base_pct=HYBRID_BULL_PARAMS["base_pct"],
                base_multiplier=HYBRID_BULL_PARAMS["base_multiplier"],
                atr_weight=HYBRID_BULL_PARAMS["atr_weight"],
                confirmation_bars=HYBRID_BULL_PARAMS["confirmation_bars"],
            )
            fcfg = pv.FilterConfig(
                enabled=True,
                min_wave_pct=HYBRID_BULL_PARAMS["min_wave_pct"],
                min_pivot_interval_bars=HYBRID_BULL_PARAMS["min_pivot_interval_bars"],
                st_distance_threshold=0.1,
                adx_hold_threshold=15.0,
            )
            bull_res = pva.run_pivot(df_bull, pcfg, fcfg, HYBRID_BULL_PARAMS["direction_mode"])
            bull_results.append(bull_res)
        
        # BEAR/NEUTRAL 레짐 데이터에서 롱-또는-플랫
        df_non_bull = pro._filter_df_by_regime(df, regime_signal, -1)
        df_neutral = pro._filter_df_by_regime(df, regime_signal, 0)
        df_lf = pd.concat([df_non_bull, df_neutral]).sort_index()
        
        if len(df_lf) > 0:
            lf_res = pva.run_long_or_flat(df_lf)
            lf_results.append(lf_res)
        
        del df, df_bull, df_non_bull, df_neutral, df_lf
        gc.collect()
    
    bull_metrics = combine_trade_results(bull_results)
    lf_metrics = combine_trade_results(lf_results)
    
    # 하이브리드 합산
    all_results = bull_results + lf_results
    hybrid_metrics = combine_trade_results(all_results)
    
    return {
        "bull": bull_metrics,
        "lf": lf_metrics,
        "hybrid": hybrid_metrics,
    }


def make_objective(train_years: List[int], log_file):
    """BULL 레짐 롱 전략 파라미터 최적화 objective."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "base_pct": trial.suggest_float("base_pct", 0.5, 2.0, log=True),
            "base_multiplier": trial.suggest_float("base_multiplier", 1.0, 3.0),
            "atr_weight": trial.suggest_float("atr_weight", 0.0, 0.5),
            "confirmation_bars": trial.suggest_int("confirmation_bars", 1, 6),
            "min_wave_pct": trial.suggest_float("min_wave_pct", 0.05, 0.3, log=True),
            "min_pivot_interval_bars": trial.suggest_int("min_pivot_interval_bars", 10, 30),
            "direction_mode": "long_only",
        }
        
        metrics = evaluate_bull_strategy(train_years, params)
        
        if metrics["n_trades"] < 20:
            return -1.0
        
        log_and_print(f"  trial {trial.number}: base_pct={params['base_pct']:.3f}, "
                      f"mult={params['base_multiplier']:.2f}, atr_w={params['atr_weight']:.2f}, "
                      f"conf={params['confirmation_bars']}, wave={params['min_wave_pct']:.3f}, "
                      f"interval={params['min_pivot_interval_bars']} | "
                      f"Sharpe={metrics['sharpe_daily']:.3f}, Trades={metrics['n_trades']}", log_file)
        
        return metrics["sharpe_daily"]
    
    return objective


def _run_bull_mode(log):
    """BULL 레짐 롱 전략 단일 모드."""
    log_and_print("=" * 100, log)
    log_and_print("BULL 레짐 전용 피봇 롱 전략", log)
    log_and_print(f"대상 기간: {ALL_YEARS}", log)
    log_and_print("=" * 100, log)
    
    # 1) 기본 파라미터로 평가
    log_and_print(f"\n[1] WFO BULL 최적 파라미터로 평가", log)
    default_metrics = evaluate_bull_strategy(ALL_YEARS, DEFAULT_BULL_PARAMS)
    log_and_print(fmt_metrics(default_metrics, "BULL 피봇 롱 (기본 파라미터)"), log)
    
    # 2) 추가 Optuna 최적화 (전체 기간 대상)
    log_and_print(f"\n[2] BULL 레짐 롱 전략 추가 최적화", log)
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    
    study.optimize(make_objective(ALL_YEARS, log), n_trials=50, show_progress_bar=True)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    best_params["direction_mode"] = "long_only"
    
    log_and_print(f"\n[최적 파라미터] trial {best_trial.number}", log)
    log_and_print(f"  {best_params}", log)
    log_and_print(f"  훈련 Sharpe: {best_trial.value:.3f}", log)
    
    # 3) 최적 파라미터로 평가
    log_and_print(f"\n[3] 최적 파라미터로 전체 기간 평가", log)
    best_metrics = evaluate_bull_strategy(ALL_YEARS, best_params)
    log_and_print(fmt_metrics(best_metrics, "BULL 피봇 롱 (최적 파라미터)"), log)
    
    # 4) 롱-또는-플랫과 비교
    log_and_print(f"\n[4] 롱-또는-플랫 벤치마크", log)
    lf_results = []
    for year in ALL_YEARS:
        df = pva.load_data_by_year(year)
        if len(df) > 0:
            lf_results.append(pva.run_long_or_flat(df))
        del df
        gc.collect()
    lf_total = pva.combine_results(lf_results)
    log_and_print(pva.fmt_result(lf_total, "롱-또는-플랫 (전체)"), log)
    
    # 5) 연도별 BULL 전략 성과
    log_and_print(f"\n[5] 연도별 BULL 피봇 롱 성과", log)
    for year in ALL_YEARS:
        metrics = evaluate_bull_strategy([year], best_params)
        if metrics["n_trades"] > 0:
            log_and_print(fmt_metrics(metrics, f"  {year}년 BULL"), log)
    
    # 6) 요약
    log_and_print("\n" + "=" * 100, log)
    log_and_print("요약", log)
    log_and_print("=" * 100, log)
    log_and_print(f"BULL 피봇 롱 (기본):  PnL={default_metrics['total_pnl_krw']:,.0f}, Sharpe={default_metrics['sharpe_daily']:.3f}, MaxDD={default_metrics['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"BULL 피봇 롱 (최적):  PnL={best_metrics['total_pnl_krw']:,.0f}, Sharpe={best_metrics['sharpe_daily']:.3f}, MaxDD={best_metrics['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"롱-또는-플랫 전체:    PnL={lf_total.total_pnl_krw:,.0f}, Sharpe={lf_total.sharpe_daily:.3f}, MaxDD={lf_total.max_drawdown_krw:,.0f}", log)
    log_and_print(f"최적 파라미터: {best_params}", log)
    log_and_print("=" * 100, log)


def _run_hybrid_mode(log):
    """BULL 피봇 롱 + 롱-또는-플랫 하이브리드 모드."""
    log_and_print("=" * 100, log)
    log_and_print("BULL 피봇 롱 + 롱-또는-플랫 하이브리드 전략", log)
    log_and_print(f"대상 기간: {ALL_YEARS}", log)
    log_and_print("=" * 100, log)
    
    # 1) 하이브리드 전략 평가
    log_and_print(f"\n[1] 하이브리드 전략 평가", log)
    results = evaluate_hybrid_strategy(ALL_YEARS)
    log_and_print(fmt_metrics(results["bull"], "BULL 레짐 피봇 롱"), log)
    log_and_print(fmt_metrics(results["lf"], "BEAR/NEUTRAL 롱-또는-플랫"), log)
    log_and_print(fmt_metrics(results["hybrid"], "하이브리드 전략 (합산)"), log)
    
    # 2) 벤치마크: BULL 피봇 롱만
    log_and_print(f"\n[2] BULL 피봇 롱 단일 전략", log)
    bull_only_metrics = evaluate_bull_strategy(ALL_YEARS, HYBRID_BULL_PARAMS)
    log_and_print(fmt_metrics(bull_only_metrics, "BULL 피봇 롱 (전체 기간)"), log)
    
    # 3) 벤치마크: 롱-또는-플랫만
    log_and_print(f"\n[3] 롱-또는-플랫 단일 전략", log)
    lf_total_results = []
    for year in ALL_YEARS:
        df = pva.load_data_by_year(year)
        if len(df) > 0:
            lf_total_results.append(pva.run_long_or_flat(df))
        del df
        gc.collect()
    lf_total = pva.combine_results(lf_total_results)
    log_and_print(pva.fmt_result(lf_total, "롱-또는-플랫 (전체 기간)"), log)
    
    # 4) 연도별 하이브리드 성과
    log_and_print(f"\n[4] 연도별 하이브리드 성과", log)
    for year in ALL_YEARS:
        year_results = evaluate_hybrid_strategy([year])
        if year_results["hybrid"]["n_trades"] > 0:
            log_and_print(fmt_metrics(year_results["hybrid"], f"  {year}년 하이브리드"), log)
    
    # 5) 요약
    log_and_print("\n" + "=" * 100, log)
    log_and_print("요약", log)
    log_and_print("=" * 100, log)
    log_and_print(f"BULL 피봇 롱 단일:     PnL={bull_only_metrics['total_pnl_krw']:,.0f}, Sharpe={bull_only_metrics['sharpe_daily']:.3f}, MaxDD={bull_only_metrics['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"롱-또는-플랫 단일:     PnL={lf_total.total_pnl_krw:,.0f}, Sharpe={lf_total.sharpe_daily:.3f}, MaxDD={lf_total.max_drawdown_krw:,.0f}", log)
    log_and_print(f"하이브리드 전략:       PnL={results['hybrid']['total_pnl_krw']:,.0f}, Sharpe={results['hybrid']['sharpe_daily']:.3f}, MaxDD={results['hybrid']['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"  └ BULL 기여:        PnL={results['bull']['total_pnl_krw']:,.0f}, Sharpe={results['bull']['sharpe_daily']:.3f}", log)
    log_and_print(f"  └ BEAR/NEUTRAL 기여: PnL={results['lf']['total_pnl_krw']:,.0f}, Sharpe={results['lf']['sharpe_daily']:.3f}", log)
    log_and_print("=" * 100, log)


def main():
    parser = argparse.ArgumentParser(description="BULL 피봇 롱 전략")
    parser.add_argument("--hybrid", action="store_true", help="BULL 피봇 롱 + 롱-또는-플랫 하이브리드 모드")
    parser.add_argument("--years", type=str, help="쉼표로 구분된 연도 (예: 2019,2020,2021)")
    args = parser.parse_args()
    
    global ALL_YEARS
    if args.years:
        ALL_YEARS = [int(y.strip()) for y in args.years.split(",")]
    
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        if args.hybrid:
            _run_hybrid_mode(log)
        else:
            _run_bull_mode(log)


if __name__ == "__main__":
    main()
