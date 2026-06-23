# -*- coding: utf-8 -*-
"""
BULL 피봇 롱 전략 포지션 사이즈 및 거래 비용 민감도 최적화

다음 사이징 방식을 테스트:
    1. fixed: 기본 고정 사이즈
    2. atr: ATR 기반 변동성 타겟팅 (atr_sizing_target_krw)
    3. kelly: 거래 내역 기반 켈리 비율 적용
    4. cost: 수수료/슬리피지 변화에 따른 민감도 분석

실행 모드:
    - atr: ATR 기반 사이징 그리드 탐색 (기본)
    - kelly: Kelly 비율 기반 고정 비율 투자 WFO
    - cost: 거래 비용/슬리피지 민감도 테스트
"""
import sys
import gc
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
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

OUTPUT_LOG = Path(__file__).parent / "pivot_bull_sizing_optimizer.log"
ALL_YEARS = list(range(2019, 2027))
TRAIN_YEARS = list(range(2019, 2026))
TEST_YEARS = [2026]

# 비용 민감도 모드 기본 설정 (Half Kelly)
KELLY_FACTOR = 0.126

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


def evaluate_bull_sizing(years: List[int], sizing_mode: str, sizing_param: float) -> Dict[str, Any]:
    """특정 사이징 방식으로 BULL 피봇 롱 전략 평가."""
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
    
    bt_cfg = pva.BT
    if sizing_mode == "atr":
        bt_cfg = pv.BacktestConfig(
            multiplier=bt_cfg.multiplier,
            commission_pct_per_side=bt_cfg.commission_pct_per_side,
            slippage_ticks_per_side=bt_cfg.slippage_ticks_per_side,
            tick_size=bt_cfg.tick_size,
            entry_on=bt_cfg.entry_on,
            annualization=bt_cfg.annualization,
            position_size_mode="atr",
            atr_sizing_period=14,
            atr_sizing_target_krw=sizing_param,
            max_position_size_factor=3.0,
        )
    elif sizing_mode == "fixed":
        bt_cfg = pv.BacktestConfig(
            multiplier=bt_cfg.multiplier,
            commission_pct_per_side=bt_cfg.commission_pct_per_side,
            slippage_ticks_per_side=bt_cfg.slippage_ticks_per_side,
            tick_size=bt_cfg.tick_size,
            entry_on=bt_cfg.entry_on,
            annualization=bt_cfg.annualization,
            position_size_mode="fixed",
        )
    
    regime_signal = pbs.get_daily_regime_for_years(years)
    
    total_pnl_krw = 0.0
    total_pnl_pts = 0.0
    total_trades = 0
    weighted_win_rate = 0.0
    gross_win_pts = 0.0
    gross_loss_pts = 0.0
    daily_pnl_list: List[pd.Series] = []
    equity_list: List[pd.Series] = []
    all_trades_list: List[pd.DataFrame] = []
    
    for year in years:
        df = pva.load_data_by_year(year)
        if len(df) == 0:
            continue
        df_bull = pro._filter_df_by_regime(df, regime_signal, 1)
        if len(df_bull) == 0:
            del df, df_bull
            gc.collect()
            continue
        
        res = pva.run_pivot(df_bull, pcfg, fcfg, "long_only", bt_cfg)
        
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
            all_trades_list.append(res.trades)
        
        del df, df_bull, res
        gc.collect()
    
    if total_trades == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl_krw": 0.0,
            "total_pnl_pts": 0.0, "expectancy_krw": 0.0, "expectancy_pts": 0.0,
            "profit_factor": 0.0, "sharpe_daily": -1.0, "max_drawdown_krw": 0.0,
            "sizing_mode": sizing_mode, "sizing_param": sizing_param,
        }
    
    # Kelly 계산을 위한 거래 내역
    if all_trades_list:
        all_trades = pd.concat(all_trades_list, ignore_index=True)
        net_pts = all_trades["net_pts"]
        wins = net_pts[net_pts > 0]
        losses = net_pts[net_pts < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        win_rate_pct = len(wins) / total_trades
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly_f = (b * win_rate_pct - (1 - win_rate_pct)) / b
        else:
            kelly_f = 0.0
    else:
        kelly_f = 0.0
    
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
        "sizing_mode": sizing_mode,
        "sizing_param": sizing_param,
        "kelly_f": kelly_f,
    }


def evaluate_with_multiplier(years: List[int], multiplier_factor: float) -> Dict[str, Any]:
    """특정 multiplier 비율로 BULL 피봇 롱 전략 평가 (Kelly WFO용)."""
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
    
    base_bt = pva.BT
    bt_cfg = pv.BacktestConfig(
        multiplier=base_bt.multiplier * multiplier_factor,
        commission_pct_per_side=base_bt.commission_pct_per_side,
        slippage_ticks_per_side=base_bt.slippage_ticks_per_side,
        tick_size=base_bt.tick_size,
        entry_on=base_bt.entry_on,
        annualization=base_bt.annualization,
        position_size_mode="fixed",
    )
    
    regime_signal = pbs.get_daily_regime_for_years(years)
    
    total_pnl_krw = 0.0
    total_pnl_pts = 0.0
    total_trades = 0
    weighted_win_rate = 0.0
    gross_win_pts = 0.0
    gross_loss_pts = 0.0
    daily_pnl_list: List[pd.Series] = []
    equity_list: List[pd.Series] = []
    all_trades_list: List[pd.DataFrame] = []
    
    for year in years:
        df = pva.load_data_by_year(year)
        if len(df) == 0:
            continue
        df_bull = pro._filter_df_by_regime(df, regime_signal, 1)
        if len(df_bull) == 0:
            del df, df_bull
            gc.collect()
            continue
        
        res = pva.run_pivot(df_bull, pcfg, fcfg, "long_only", bt_cfg)
        
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
            all_trades_list.append(res.trades)
        
        del df, df_bull, res
        gc.collect()
    
    if total_trades == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl_krw": 0.0,
            "total_pnl_pts": 0.0, "expectancy_krw": 0.0, "expectancy_pts": 0.0,
            "profit_factor": 0.0, "sharpe_daily": -1.0, "max_drawdown_krw": 0.0,
            "multiplier_factor": multiplier_factor,
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
    
    # Kelly f 계산
    if all_trades_list:
        all_trades = pd.concat(all_trades_list, ignore_index=True)
        net_pts = all_trades["net_pts"]
        wins = net_pts[net_pts > 0]
        losses = net_pts[net_pts < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        win_rate_pct = len(wins) / total_trades
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly_f = (b * win_rate_pct - (1 - win_rate_pct)) / b
        else:
            kelly_f = 0.0
    else:
        kelly_f = 0.0
    
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
        "multiplier_factor": multiplier_factor,
        "kelly_f": kelly_f,
    }


def evaluate_cost_scenario(years: List[int], commission: float, slippage: float) -> Dict[str, Any]:
    """특정 수수료/슬리피지로 BULL 피봇 롱 전략 평가 (Half Kelly)."""
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
    
    base_bt = pva.BT
    bt_cfg = pv.BacktestConfig(
        multiplier=base_bt.multiplier * KELLY_FACTOR,
        commission_pct_per_side=commission,
        slippage_ticks_per_side=slippage,
        tick_size=base_bt.tick_size,
        entry_on=base_bt.entry_on,
        annualization=base_bt.annualization,
        position_size_mode="fixed",
    )
    
    regime_signal = pbs.get_daily_regime_for_years(years)
    
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
        
        res = pva.run_pivot(df_bull, pcfg, fcfg, "long_only", bt_cfg)
        
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
            "commission": commission, "slippage": slippage,
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
        "commission": commission,
        "slippage": slippage,
    }


def _run_atr_mode(log):
    """ATR 기반 사이징 최적화 모드."""
    log_and_print("=" * 100, log)
    log_and_print("BULL 피봇 롱 포지션 사이즈 최적화 (ATR 모드)", log)
    log_and_print(f"훈련: {TRAIN_YEARS} | 검증: {TEST_YEARS}", log)
    log_and_print("=" * 100, log)
    
    # 1) 고정 사이즈 기준
    log_and_print(f"\n[1] 고정 사이즈 (fixed) 기준", log)
    fixed_baseline = evaluate_bull_sizing(TRAIN_YEARS, "fixed", 0.0)
    log_and_print(fmt_metrics(fixed_baseline, "고정 사이즈"), log)
    log_and_print(f"  Kelly f: {fixed_baseline['kelly_f']:.3f}", log)
    
    # 2) ATR 사이즈 그리드 탐색
    log_and_print(f"\n[2] ATR 기반 사이즈 그리드 탐색", log)
    atr_targets = [1_000_000, 2_000_000, 3_000_000, 5_000_000, 7_000_000, 10_000_000, 15_000_000, 20_000_000]
    atr_results = []
    for target in atr_targets:
        metrics = evaluate_bull_sizing(TRAIN_YEARS, "atr", target)
        atr_results.append(metrics)
        log_and_print(fmt_metrics(metrics, f"ATR target={target:,.0f}"), log)
    
    best_atr_sharpe = max(atr_results, key=lambda x: x["sharpe_daily"])
    best_atr_pnl = max(atr_results, key=lambda x: x["total_pnl_krw"])
    best_atr_calmar = max(atr_results, key=lambda x: x["total_pnl_krw"] / abs(x["max_drawdown_krw"]) if x["max_drawdown_krw"] else 0)
    
    log_and_print(f"\n[ATR 최고 Sharpe] target={best_atr_sharpe['sizing_param']:,.0f}", log)
    log_and_print(f"  PnL={best_atr_sharpe['total_pnl_krw']:,.0f}, Sharpe={best_atr_sharpe['sharpe_daily']:.3f}, MaxDD={best_atr_sharpe['max_drawdown_krw']:,.0f}", log)
    
    # 3) 검증 기간 테스트
    log_and_print(f"\n[3] 검증 기간 테스트", log)
    fixed_test = evaluate_bull_sizing(TEST_YEARS, "fixed", 0.0)
    atr_sharpe_test = evaluate_bull_sizing(TEST_YEARS, "atr", best_atr_sharpe["sizing_param"])
    atr_pnl_test = evaluate_bull_sizing(TEST_YEARS, "atr", best_atr_pnl["sizing_param"])
    
    log_and_print(fmt_metrics(fixed_test, "고정 사이즈 검증"), log)
    log_and_print(fmt_metrics(atr_sharpe_test, f"ATR 최고 Sharpe 검증 target={best_atr_sharpe['sizing_param']:,.0f}"), log)
    log_and_print(fmt_metrics(atr_pnl_test, f"ATR 최고 PnL 검증 target={best_atr_pnl['sizing_param']:,.0f}"), log)
    
    # 4) Kelly 적용 결과
    log_and_print(f"\n[4] Kelly 비율 기반 사이즈", log)
    kelly_half = max(0.0, min(fixed_baseline['kelly_f'] / 2, 1.0))
    log_and_print(f"  Kelly f={fixed_baseline['kelly_f']:.3f}, Half Kelly={kelly_half:.3f}", log)
    log_and_print(f"  (고정 사이즈의 {kelly_half:.1%}만큼 투자 시 PnL/PnL_kelly, MaxDD/MaxDD_kelly)", log)
    
    # 5) 요약
    log_and_print("\n" + "=" * 100, log)
    log_and_print("OOS 사이즈 최적화 요약", log)
    log_and_print("=" * 100, log)
    log_and_print(f"고정 사이즈 훈련:  PnL={fixed_baseline['total_pnl_krw']:,.0f}, Sharpe={fixed_baseline['sharpe_daily']:.3f}, MaxDD={fixed_baseline['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"고정 사이즈 검증:  PnL={fixed_test['total_pnl_krw']:,.0f}, Sharpe={fixed_test['sharpe_daily']:.3f}, MaxDD={fixed_test['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"ATR 최고 Sharpe 훈련: target={best_atr_sharpe['sizing_param']:,.0f}, PnL={best_atr_sharpe['total_pnl_krw']:,.0f}, Sharpe={best_atr_sharpe['sharpe_daily']:.3f}, MaxDD={best_atr_sharpe['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"ATR 최고 Sharpe 검증: target={best_atr_sharpe['sizing_param']:,.0f}, PnL={atr_sharpe_test['total_pnl_krw']:,.0f}, Sharpe={atr_sharpe_test['sharpe_daily']:.3f}, MaxDD={atr_sharpe_test['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"ATR 최고 PnL 훈련: target={best_atr_pnl['sizing_param']:,.0f}, PnL={best_atr_pnl['total_pnl_krw']:,.0f}, Sharpe={best_atr_pnl['sharpe_daily']:.3f}, MaxDD={best_atr_pnl['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"ATR 최고 PnL 검증: target={best_atr_pnl['sizing_param']:,.0f}, PnL={atr_pnl_test['total_pnl_krw']:,.0f}, Sharpe={atr_pnl_test['sharpe_daily']:.3f}, MaxDD={atr_pnl_test['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"Kelly f (고정): {fixed_baseline['kelly_f']:.3f}, Half Kelly: {kelly_half:.3f}", log)
    log_and_print("=" * 100, log)


def _run_kelly_mode(log):
    """Kelly 기반 고정 비율 투자 WFO 모드."""
    log_and_print("=" * 100, log)
    log_and_print("Kelly 기반 고정 비율 투자 OOS 검증", log)
    log_and_print(f"훈련: {TRAIN_YEARS} | 검증: {TEST_YEARS}", log)
    log_and_print("=" * 100, log)
    
    # 1) 훈련에서 Kelly f 계산 (고정 사이즈)
    log_and_print(f"\n[1] 훈련 기간 고정 사이즈로 Kelly f 계산", log)
    train_baseline = evaluate_with_multiplier(TRAIN_YEARS, 1.0)
    log_and_print(fmt_metrics(train_baseline, "고정 사이즈 (1.0x)"), log)
    kelly_f = train_baseline["kelly_f"]
    log_and_print(f"  Kelly f = {kelly_f:.3f}", log)
    
    # 2) 다양한 Kelly 비율 테스트
    ratios = [
        ("1/8 Kelly", max(0.0, kelly_f / 8)),
        ("1/4 Kelly", max(0.0, kelly_f / 4)),
        ("Half Kelly", max(0.0, kelly_f / 2)),
        ("Full Kelly", max(0.0, kelly_f)),
        ("1.5x Kelly", max(0.0, kelly_f * 1.5)),
        ("2x Kelly", max(0.0, kelly_f * 2.0)),
    ]
    
    log_and_print(f"\n[2] 훈련 기간 Kelly 비율 테스트", log)
    train_baseline["name"] = "고정 사이즈 (1.0x)"
    train_results = [train_baseline]
    for name, ratio in ratios:
        metrics = evaluate_with_multiplier(TRAIN_YEARS, ratio)
        metrics["name"] = name
        train_results.append(metrics)
        log_and_print(fmt_metrics(metrics, f"{name} (ratio={ratio:.3f})"), log)
    
    # 3) 검증 기간 테스트
    log_and_print(f"\n[3] 검증 기간 Kelly 비율 테스트", log)
    test_results = []
    for name, ratio in [("고정 사이즈", 1.0)] + ratios:
        metrics = evaluate_with_multiplier(TEST_YEARS, ratio)
        metrics["name"] = name
        test_results.append(metrics)
        log_and_print(fmt_metrics(metrics, f"{name} (ratio={ratio:.3f})"), log)
    
    # 4) 최고 성과 조합
    best_train_sharpe = max(train_results, key=lambda x: x["sharpe_daily"])
    best_train_pnl = max(train_results, key=lambda x: x["total_pnl_krw"])
    best_train_calmar = max(train_results, key=lambda x: x["total_pnl_krw"] / abs(x["max_drawdown_krw"]) if x["max_drawdown_krw"] else 0)
    
    best_test_sharpe = max(test_results, key=lambda x: x["sharpe_daily"])
    best_test_pnl = max(test_results, key=lambda x: x["total_pnl_krw"])
    
    log_and_print(f"\n[훈련 최고 Sharpe] {best_train_sharpe['name']}", log)
    log_and_print(f"  PnL={best_train_sharpe['total_pnl_krw']:,.0f}, Sharpe={best_train_sharpe['sharpe_daily']:.3f}, MaxDD={best_train_sharpe['max_drawdown_krw']:,.0f}", log)
    
    log_and_print(f"\n[훈련 최고 PnL] {best_train_pnl['name']}", log)
    log_and_print(f"  PnL={best_train_pnl['total_pnl_krw']:,.0f}, Sharpe={best_train_pnl['sharpe_daily']:.3f}, MaxDD={best_train_pnl['max_drawdown_krw']:,.0f}", log)
    
    log_and_print(f"\n[검증 최고 Sharpe] {best_test_sharpe['name']}", log)
    log_and_print(f"  PnL={best_test_sharpe['total_pnl_krw']:,.0f}, Sharpe={best_test_sharpe['sharpe_daily']:.3f}, MaxDD={best_test_sharpe['max_drawdown_krw']:,.0f}", log)
    
    log_and_print(f"\n[검증 최고 PnL] {best_test_pnl['name']}", log)
    log_and_print(f"  PnL={best_test_pnl['total_pnl_krw']:,.0f}, Sharpe={best_test_pnl['sharpe_daily']:.3f}, MaxDD={best_test_pnl['max_drawdown_krw']:,.0f}", log)
    
    # 5) 요약
    log_and_print("\n" + "=" * 100, log)
    log_and_print("Kelly 기반 고정 비율 투자 OOS 요약", log)
    log_and_print("=" * 100, log)
    log_and_print(f"Kelly f (훈련): {kelly_f:.3f}", log)
    log_and_print(f"\n{'비율':<15} {'훈련 PnL':>15} {'훈련 Sharpe':>12} {'훈련 MaxDD':>15} {'검증 PnL':>15} {'검증 Sharpe':>12} {'검증 MaxDD':>15}", log)
    for r in train_results:
        name = r.get("name", "고정 사이즈")
        test_r = next((x for x in test_results if x.get("name") == name), None)
        if test_r:
            log_and_print(f"{name:<15} {r['total_pnl_krw']:>15,.0f} {r['sharpe_daily']:>12.3f} {r['max_drawdown_krw']:>15,.0f} "
                          f"{test_r['total_pnl_krw']:>15,.0f} {test_r['sharpe_daily']:>12.3f} {test_r['max_drawdown_krw']:>15,.0f}", log)
    log_and_print("=" * 100, log)


def _run_cost_mode(log):
    """거래 비용/슬리피지 민감도 분석 모드."""
    log_and_print("=" * 100, log)
    log_and_print("BULL 피봇 롱 거래 비용/슬리피지 민감도 테스트", log)
    log_and_print(f"훈련: {TRAIN_YEARS} | 검증: {TEST_YEARS} | Kelly factor: {KELLY_FACTOR}", log)
    log_and_print("=" * 100, log)
    
    # 1) 기준 비용 (commission=0.003%, slippage=1tick) 평가
    log_and_print(f"\n[1] 기준 비용 (commission=0.003%, slippage=1tick)", log)
    baseline_train = evaluate_cost_scenario(TRAIN_YEARS, 0.00003, 1.0)
    baseline_test = evaluate_cost_scenario(TEST_YEARS, 0.00003, 1.0)
    log_and_print(fmt_metrics(baseline_train, "기준 비용 훈련"), log)
    log_and_print(fmt_metrics(baseline_test, "기준 비용 검증"), log)
    
    # 2) 그리드 탐색
    log_and_print(f"\n[2] 비용/슬리피지 그리드 탐색", log)
    commissions = [0.00003, 0.00005, 0.0001, 0.0002, 0.0003]
    slippages = [1.0, 2.0, 3.0, 5.0]
    combos = list(product(commissions, slippages))
    log_and_print(f"  총 조합: {len(combos)}", log)
    
    train_results = []
    for i, (commission, slippage) in enumerate(combos, 1):
        metrics = evaluate_cost_scenario(TRAIN_YEARS, commission, slippage)
        train_results.append(metrics)
        log_and_print(f"  [{i}/{len(combos)}] commission={commission:.5f}, slippage={slippage:.1f} | "
                      f"거래={metrics['n_trades']}, PnL={metrics['total_pnl_krw']:,.0f}, "
                      f"Sharpe={metrics['sharpe_daily']:.3f}, MaxDD={metrics['max_drawdown_krw']:,.0f}", log)
    
    # 3) 최고 조합 선정
    best_train_sharpe = max(train_results, key=lambda x: x["sharpe_daily"])
    best_train_pnl = max(train_results, key=lambda x: x["total_pnl_krw"])
    best_train_calmar = max(train_results, key=lambda x: x["total_pnl_krw"] / abs(x["max_drawdown_krw"]) if x["max_drawdown_krw"] else 0)
    
    log_and_print(f"\n[훈련 최고 Sharpe]", log)
    log_and_print(f"  commission={best_train_sharpe['commission']:.5f}, slippage={best_train_sharpe['slippage']:.1f}", log)
    log_and_print(f"  PnL={best_train_sharpe['total_pnl_krw']:,.0f}, Sharpe={best_train_sharpe['sharpe_daily']:.3f}, MaxDD={best_train_sharpe['max_drawdown_krw']:,.0f}", log)
    
    log_and_print(f"\n[훈련 최고 PnL]", log)
    log_and_print(f"  commission={best_train_pnl['commission']:.5f}, slippage={best_train_pnl['slippage']:.1f}", log)
    log_and_print(f"  PnL={best_train_pnl['total_pnl_krw']:,.0f}, Sharpe={best_train_pnl['sharpe_daily']:.3f}, MaxDD={best_train_pnl['max_drawdown_krw']:,.0f}", log)
    
    # 4) 검증 기간 테스트 (기준 + 최고 조합)
    log_and_print(f"\n[3] 검증 기간 테스트", log)
    test_baseline = evaluate_cost_scenario(TEST_YEARS, 0.00003, 1.0)
    test_best_sharpe = evaluate_cost_scenario(TEST_YEARS, best_train_sharpe["commission"], best_train_sharpe["slippage"])
    test_best_pnl = evaluate_cost_scenario(TEST_YEARS, best_train_pnl["commission"], best_train_pnl["slippage"])
    
    log_and_print(fmt_metrics(test_baseline, "기준 비용 검증"), log)
    log_and_print(fmt_metrics(test_best_sharpe, f"최고 Sharpe 비용 검증 c={best_train_sharpe['commission']:.5f},s={best_train_sharpe['slippage']:.1f}"), log)
    log_and_print(fmt_metrics(test_best_pnl, f"최고 PnL 비용 검증 c={best_train_pnl['commission']:.5f},s={best_train_pnl['slippage']:.1f}"), log)
    
    # 5) 비용 내구 한계 찾기 (Sharpe>0 or PnL>0)
    log_and_print(f"\n[4] 비용 내구 한계", log)
    train_positive = [r for r in train_results if r["sharpe_daily"] > 0 and r["total_pnl_krw"] > 0]
    if train_positive:
        worst_train = min(train_positive, key=lambda x: x["total_pnl_krw"])
        log_and_print(f"  훈련에서 최악의 양수 조합: commission={worst_train['commission']:.5f}, slippage={worst_train['slippage']:.1f}", log)
        log_and_print(f"  PnL={worst_train['total_pnl_krw']:,.0f}, Sharpe={worst_train['sharpe_daily']:.3f}", log)
    
    test_results = [evaluate_cost_scenario(TEST_YEARS, c, s) for c, s in combos]
    test_positive = [r for r in test_results if r["sharpe_daily"] > 0 and r["total_pnl_krw"] > 0]
    if test_positive:
        worst_test = min(test_positive, key=lambda x: x["total_pnl_krw"])
        log_and_print(f"  검증에서 최악의 양수 조합: commission={worst_test['commission']:.5f}, slippage={worst_test['slippage']:.1f}", log)
        log_and_print(f"  PnL={worst_test['total_pnl_krw']:,.0f}, Sharpe={worst_test['sharpe_daily']:.3f}", log)
    
    # 6) 요약
    log_and_print("\n" + "=" * 100, log)
    log_and_print("비용/슬리피지 민감도 요약", log)
    log_and_print("=" * 100, log)
    log_and_print(f"기준 비용 (0.003%, 1tick) 훈련: PnL={baseline_train['total_pnl_krw']:,.0f}, Sharpe={baseline_train['sharpe_daily']:.3f}, MaxDD={baseline_train['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"기준 비용 (0.003%, 1tick) 검증: PnL={baseline_test['total_pnl_krw']:,.0f}, Sharpe={baseline_test['sharpe_daily']:.3f}, MaxDD={baseline_test['max_drawdown_krw']:,.0f}", log)
    log_and_print(f"\n훈련 최고 Sharpe: c={best_train_sharpe['commission']:.5f}, s={best_train_sharpe['slippage']:.1f}", log)
    log_and_print(f"  훈련: PnL={best_train_sharpe['total_pnl_krw']:,.0f}, Sharpe={best_train_sharpe['sharpe_daily']:.3f}", log)
    log_and_print(f"  검증: PnL={test_best_sharpe['total_pnl_krw']:,.0f}, Sharpe={test_best_sharpe['sharpe_daily']:.3f}", log)
    log_and_print(f"\n비용 상승 시 생존 여부:", log)
    if train_positive and test_positive:
        log_and_print(f"  훈련: 최소 양수 Sharpe까지 commission={worst_train['commission']:.5f}, slippage={worst_train['slippage']:.1f} 가능", log)
        log_and_print(f"  검증: 최소 양수 Sharpe까지 commission={worst_test['commission']:.5f}, slippage={worst_test['slippage']:.1f} 가능", log)
    log_and_print("=" * 100, log)


def main():
    parser = argparse.ArgumentParser(description="BULL 피봇 롱 포지션 사이즈 및 거래 비용 민감도 최적화")
    parser.add_argument("--mode", type=str, choices=["atr", "kelly", "cost"], default="atr",
                        help="사이징/비용 모드 (atr: ATR 기반, kelly: Kelly WFO, cost: 비용 민감도)")
    args = parser.parse_args()
    
    if args.mode == "kelly":
        output_log = Path(__file__).parent / "pivot_bull_kelly_wfo.log"
    elif args.mode == "cost":
        output_log = Path(__file__).parent / "pivot_bull_cost_sensitivity.log"
    else:
        output_log = Path(__file__).parent / "pivot_bull_sizing_optimizer.log"
    
    with open(output_log, "w", encoding="utf-8") as log:
        if args.mode == "kelly":
            _run_kelly_mode(log)
        elif args.mode == "cost":
            _run_cost_mode(log)
        else:
            _run_atr_mode(log)


if __name__ == "__main__":
    main()
