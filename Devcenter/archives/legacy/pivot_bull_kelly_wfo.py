# -*- coding: utf-8 -*-
"""
Kelly 기반 고정 비율 투자 OOS 검증

훈련 기간에서 고정 사이즈로 Kelly f를 계산한 뒤,
Kelly f의 비율(전체, 절반, 1/4, 1/8, 2배)을 multiplier에 적용하여
훈련/검증 성과를 비교한다.
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
import regime_intraday_v2 as rg
import pivot_regime_optimizer as pro
import pivot_bull_strategy as pbs
import pivot_viability_analysis as pva
import pivot_optuna_v2 as pv

OUTPUT_LOG = Path(__file__).parent / "pivot_bull_kelly_wfo.log"
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


def evaluate_with_multiplier(years: List[int], multiplier_factor: float) -> Dict[str, Any]:
    """특정 multiplier 비율로 BULL 피봇 롱 전략 평가."""
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
    
    # Kelly f 계산 (고정 사이즈 기준, multiplier_factor=1.0일 때와 동일)
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


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
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
        log_and_print(f"\n[3] 검증 기간 (2026) Kelly 비율 테스트", log)
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


if __name__ == "__main__":
    main()
