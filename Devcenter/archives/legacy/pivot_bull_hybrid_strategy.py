# -*- coding: utf-8 -*-
"""
BULL 피봇 롱 + 롱-또는-플랫 하이브리드 전략

- BULL 레짐(MA20>MA60): BULL 피봇 롱 최적 파라미터 사용
- BEAR/NEUTRAL 레짐: 롱-또는-플랫(MA20/60 + ADX25) 사용
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

OUTPUT_LOG = Path(__file__).parent / "pivot_bull_hybrid_strategy.log"
ALL_YEARS = list(range(2019, 2027))

# BULL 레짐용 피봇 롱 파라미터 (2019-2025 최적)
BULL_PARAMS = {
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
    regime_signal = pbs.get_daily_regime_for_years(years)
    
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
                base_pct=BULL_PARAMS["base_pct"],
                base_multiplier=BULL_PARAMS["base_multiplier"],
                atr_weight=BULL_PARAMS["atr_weight"],
                confirmation_bars=BULL_PARAMS["confirmation_bars"],
            )
            fcfg = pv.FilterConfig(
                enabled=True,
                min_wave_pct=BULL_PARAMS["min_wave_pct"],
                min_pivot_interval_bars=BULL_PARAMS["min_pivot_interval_bars"],
                st_distance_threshold=0.1,
                adx_hold_threshold=15.0,
            )
            bull_res = pva.run_pivot(df_bull, pcfg, fcfg, BULL_PARAMS["direction_mode"])
            bull_results.append(bull_res)
        
        # BEAR/NEUTRAL 레짐 데이터에서 롱-또는-플랫
        df_non_bull = pro._filter_df_by_regime(df, regime_signal, -1)
        df_neutral = pro._filter_df_by_regime(df, regime_signal, 0)
        df_lf = pd.concat([df_non_bull, df_neutral]).sort_index()
        
        if len(df_lf) > 0:
            # 롱-또는-플랫은 일봉 기준 시가→종가 매매이므로, 필터링된 5분봉 데이터로 실행
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


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
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
        bull_only_metrics = pbs.evaluate_bull_strategy(ALL_YEARS, BULL_PARAMS)
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


if __name__ == "__main__":
    main()
