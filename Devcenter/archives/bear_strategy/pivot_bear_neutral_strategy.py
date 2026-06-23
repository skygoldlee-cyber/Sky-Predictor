# -*- coding: utf-8 -*-
"""
레짐별 피봇 전략 (BULL 롱 / BEAR 숏 / NEUTRAL 현금)

- BULL 레짐(MA20>MA60): 피봇 반전 롱
- BEAR 레짐(MA20<MA60): 피봇 반전 숏
- NEUTRAL 레짐: 현금 보유 (no trade)
"""
import sys
import gc
import math
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
import pivot_bull_strategy as pbs
import pivot_viability_analysis as pva
import pivot_optuna_v2 as pv

OUTPUT_LOG = Path(__file__).parent / "pivot_bear_neutral_strategy.log"
ALL_YEARS = list(range(2019, 2027))
TRAIN_YEARS = list(range(2019, 2026))  # 2026 제외
TEST_YEARS = [2026]

# BULL 레짐용 피봇 롱 파라미터 (2019-2025 최적)
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
    return (f"{label:<45} | 거래={metrics['n_trades']:>4} | 승률={metrics['win_rate']:>6.2f}% | "
            f"PnL={metrics['total_pnl_krw']:>13,.0f} | Sharpe={metrics['sharpe_daily']:>7.3f} | "
            f"MaxDD={metrics['max_drawdown_krw']:>13,.0f}")


def evaluate_pivot_in_regime(years: List[int], regime_value: int, direction_mode: str,
                              pcfg: pv.HybridAdaptivePivotConfig, fcfg: pv.FilterConfig) -> Dict[str, Any]:
    """특정 레짐에서 피봇 전략 실행."""
    total_pnl_krw = 0.0
    total_pnl_pts = 0.0
    total_trades = 0
    weighted_win_rate = 0.0
    gross_win_pts = 0.0
    gross_loss_pts = 0.0
    daily_pnl_list: List[pd.Series] = []
    equity_list: List[pd.Series] = []
    
    regime_signal = pbs.get_daily_regime_for_years(years)
    
    for year in years:
        df = pva.load_data_by_year(year)
        if len(df) == 0:
            continue
        df_regime = pro._filter_df_by_regime(df, regime_signal, regime_value)
        if len(df_regime) == 0:
            del df, df_regime
            gc.collect()
            continue
        
        res = pva.run_pivot(df_regime, pcfg, fcfg, direction_mode)
        
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
        
        del df, df_regime, res
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


def make_bear_objective(log_file):
    """BEAR 레짐에서 피봇 숏 최적화."""
    def objective(trial: optuna.Trial) -> float:
        pcfg = pv.HybridAdaptivePivotConfig(
            base_pct=trial.suggest_float("base_pct", 0.5, 2.0, log=True),
            base_multiplier=trial.suggest_float("base_multiplier", 1.0, 3.0),
            atr_weight=trial.suggest_float("atr_weight", 0.0, 0.5),
            confirmation_bars=trial.suggest_int("confirmation_bars", 1, 6),
        )
        fcfg = pv.FilterConfig(
            enabled=True,
            min_wave_pct=trial.suggest_float("min_wave_pct", 0.05, 0.3, log=True),
            min_pivot_interval_bars=trial.suggest_int("min_pivot_interval_bars", 3, 30),
            st_distance_threshold=0.1,
            adx_hold_threshold=15.0,
        )
        
        metrics = evaluate_pivot_in_regime(TRAIN_YEARS, -1, "short_only", pcfg, fcfg)
        
        if metrics["n_trades"] < 10:
            return -1.0
        
        log_and_print(f"  BEAR short trial {trial.number}: base_pct={pcfg.base_pct:.3f}, "
                      f"mult={pcfg.base_multiplier:.2f}, atr_w={pcfg.atr_weight:.2f}, "
                      f"conf={pcfg.confirmation_bars}, wave={fcfg.min_wave_pct:.3f}, "
                      f"interval={fcfg.min_pivot_interval_bars} | "
                      f"Sharpe={metrics['sharpe_daily']:.3f}, Trades={metrics['n_trades']}", log_file)
        
        return metrics["sharpe_daily"]
    
    return objective


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        log_and_print("=" * 100, log)
        log_and_print("레짐별 피봇 전략 (BULL 롱 / BEAR 숏 / NEUTRAL 현금)", log)
        log_and_print(f"훈련: {TRAIN_YEARS} | 검증: {TEST_YEARS}", log)
        log_and_print("=" * 100, log)
        
        # 1) BULL 롱 (기존 파라미터)
        log_and_print(f"\n[1] BULL 레짐 피봇 롱", log)
        bull_pcfg = pv.HybridAdaptivePivotConfig(
            base_pct=BULL_LONG_PARAMS["base_pct"],
            base_multiplier=BULL_LONG_PARAMS["base_multiplier"],
            atr_weight=BULL_LONG_PARAMS["atr_weight"],
            confirmation_bars=BULL_LONG_PARAMS["confirmation_bars"],
        )
        bull_fcfg = pv.FilterConfig(
            enabled=True,
            min_wave_pct=BULL_LONG_PARAMS["min_wave_pct"],
            min_pivot_interval_bars=BULL_LONG_PARAMS["min_pivot_interval_bars"],
            st_distance_threshold=0.1,
            adx_hold_threshold=15.0,
        )
        bull_train = evaluate_pivot_in_regime(TRAIN_YEARS, 1, "long_only", bull_pcfg, bull_fcfg)
        bull_test = evaluate_pivot_in_regime(TEST_YEARS, 1, "long_only", bull_pcfg, bull_fcfg)
        log_and_print(fmt_metrics(bull_train, "BULL 롱 훈련"), log)
        log_and_print(fmt_metrics(bull_test, "BULL 롱 검증"), log)
        
        # 2) BEAR 숏 최적화
        log_and_print(f"\n[2] BEAR 레짐 피봇 숏 최적화", log)
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        
        study.optimize(make_bear_objective(log), n_trials=50, show_progress_bar=True)
        
        best_trial = study.best_trial
        bear_params = best_trial.params
        log_and_print(f"  BEAR 최적 파라미터: {bear_params}", log)
        log_and_print(f"  훈련 Sharpe: {best_trial.value:.3f}", log)
        
        bear_pcfg = pv.HybridAdaptivePivotConfig(
            base_pct=bear_params["base_pct"],
            base_multiplier=bear_params["base_multiplier"],
            atr_weight=bear_params["atr_weight"],
            confirmation_bars=bear_params["confirmation_bars"],
        )
        bear_fcfg = pv.FilterConfig(
            enabled=True,
            min_wave_pct=bear_params["min_wave_pct"],
            min_pivot_interval_bars=bear_params["min_pivot_interval_bars"],
            st_distance_threshold=0.1,
            adx_hold_threshold=15.0,
        )
        bear_train = evaluate_pivot_in_regime(TRAIN_YEARS, -1, "short_only", bear_pcfg, bear_fcfg)
        bear_test = evaluate_pivot_in_regime(TEST_YEARS, -1, "short_only", bear_pcfg, bear_fcfg)
        log_and_print(fmt_metrics(bear_train, "BEAR 숏 훈련"), log)
        log_and_print(fmt_metrics(bear_test, "BEAR 숏 검증"), log)
        
        # 3) NEUTRAL 현금 보유 (no trade) → 0
        log_and_print(f"\n[3] NEUTRAL 레짐: 현금 보유 (거래 없음)", log)
        
        # 4) 통합 하이브리드 성과 (BULL + BEAR + NEUTRAL)
        log_and_print(f"\n[4] 통합 레짐별 전략 성과", log)
        
        # 거래 내역 합산
        all_trades_list = []
        for year in ALL_YEARS:
            df = pva.load_data_by_year(year)
            if len(df) == 0:
                continue
            regime_signal = pbs.get_daily_regime_for_years([year])
            
            # BULL: 롱
            df_bull = pro._filter_df_by_regime(df, regime_signal, 1)
            if len(df_bull) > 0:
                res = pva.run_pivot(df_bull, bull_pcfg, bull_fcfg, "long_only")
                if res.trades is not None and len(res.trades) > 0:
                    all_trades_list.append(res.trades)
                del res
            
            # BEAR: 숏
            df_bear = pro._filter_df_by_regime(df, regime_signal, -1)
            if len(df_bear) > 0:
                res = pva.run_pivot(df_bear, bear_pcfg, bear_fcfg, "short_only")
                if res.trades is not None and len(res.trades) > 0:
                    all_trades_list.append(res.trades)
                del res
            
            # NEUTRAL: 거래 없음
            
            del df, df_bull, df_bear, regime_signal
            gc.collect()
        
        if all_trades_list:
            all_trades = pd.concat(all_trades_list, ignore_index=True)
            all_trades["exit_date"] = pd.to_datetime(all_trades["exit_time"]).dt.date
            daily = all_trades.groupby("exit_date")["net_krw"].sum()
            if len(daily) >= 2 and daily.std(ddof=1) > 0:
                hybrid_sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(pva.BT.annualization))
            else:
                hybrid_sharpe = 0.0
            equity = all_trades["net_krw"].cumsum()
            hybrid_max_dd = float((equity - equity.cummax()).min())
            hybrid_pnl = float(all_trades["net_krw"].sum())
            hybrid_trades = len(all_trades)
            hybrid_win_rate = float((all_trades["net_pts"] > 0).mean() * 100)
        else:
            hybrid_sharpe = 0.0
            hybrid_max_dd = 0.0
            hybrid_pnl = 0.0
            hybrid_trades = 0
            hybrid_win_rate = 0.0
        
        log_and_print(f"통합 전략 (BULL 롱 + BEAR 숏) | 거래={hybrid_trades} | 승률={hybrid_win_rate:.2f}% | "
                      f"PnL={hybrid_pnl:,.0f} | Sharpe={hybrid_sharpe:.3f} | MaxDD={hybrid_max_dd:,.0f}", log)
        
        # 5) 벤치마크: BULL 피봇 롱만
        log_and_print(f"\n[5] 벤치마크: BULL 피봇 롱 전체", log)
        bull_total = evaluate_pivot_in_regime(ALL_YEARS, 1, "long_only", bull_pcfg, bull_fcfg)
        log_and_print(fmt_metrics(bull_total, "BULL 롱 (전체)"), log)
        
        # 6) 벤치마크: 롱-또는-플랫
        log_and_print(f"\n[6] 벤치마크: 롱-또는-플랫", log)
        lf_results = []
        for year in ALL_YEARS:
            df = pva.load_data_by_year(year)
            if len(df) > 0:
                lf_results.append(pva.run_long_or_flat(df))
            del df
            gc.collect()
        lf_total = pva.combine_results(lf_results)
        log_and_print(pva.fmt_result(lf_total, "롱-또는-플랫 (전체)"), log)
        
        # 7) 요약
        log_and_print("\n" + "=" * 100, log)
        log_and_print("요약", log)
        log_and_print("=" * 100, log)
        log_and_print(f"BULL 롱 (훈련):      PnL={bull_train['total_pnl_krw']:,.0f}, Sharpe={bull_train['sharpe_daily']:.3f}, MaxDD={bull_train['max_drawdown_krw']:,.0f}", log)
        log_and_print(f"BULL 롱 (검증):      PnL={bull_test['total_pnl_krw']:,.0f}, Sharpe={bull_test['sharpe_daily']:.3f}, MaxDD={bull_test['max_drawdown_krw']:,.0f}", log)
        log_and_print(f"BEAR 숏 (훈련):      PnL={bear_train['total_pnl_krw']:,.0f}, Sharpe={bear_train['sharpe_daily']:.3f}, MaxDD={bear_train['max_drawdown_krw']:,.0f}", log)
        log_and_print(f"BEAR 숏 (검증):      PnL={bear_test['total_pnl_krw']:,.0f}, Sharpe={bear_test['sharpe_daily']:.3f}, MaxDD={bear_test['max_drawdown_krw']:,.0f}", log)
        log_and_print(f"통합 전략 (2019-2026): PnL={hybrid_pnl:,.0f}, Sharpe={hybrid_sharpe:.3f}, MaxDD={hybrid_max_dd:,.0f}", log)
        log_and_print(f"BULL 롱 단일:        PnL={bull_total['total_pnl_krw']:,.0f}, Sharpe={bull_total['sharpe_daily']:.3f}, MaxDD={bull_total['max_drawdown_krw']:,.0f}", log)
        log_and_print(f"롱-또는-플랫:        PnL={lf_total.total_pnl_krw:,.0f}, Sharpe={lf_total.sharpe_daily:.3f}, MaxDD={lf_total.max_drawdown_krw:,.0f}", log)
        log_and_print(f"BEAR 최적 파라미터: {bear_params}", log)
        log_and_print("=" * 100, log)


if __name__ == "__main__":
    main()
