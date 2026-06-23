# -*- coding: utf-8 -*-
"""
BULL 레짐 타이밍 파라미터 OOS 검증

훈련: 2019-2025년 그리드 탐색으로 최적 타이밍 찾기
검증: 2026년에 해당 타이밍 적용
"""
import sys
import gc
import math
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
import pivot_bull_timing_optimizer as pbto
import pivot_viability_analysis as pva
import pivot_optuna_v2 as pv

OUTPUT_LOG = Path(__file__).parent / "pivot_bull_timing_wfo.log"
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


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
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
            metrics = pbto.evaluate_bull_timing(TRAIN_YEARS, confirm_days, entry_delay, exit_delay)
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
        log_and_print(f"\n[2] 검증 기간 (2026) 테스트", log)
        
        # 기본 타이밍
        baseline_test = pbto.evaluate_bull_timing(TEST_YEARS, 1, 0, 0)
        log_and_print(fmt_metrics(baseline_test, "기본 타이밍 (c=1,e=0,x=0) 검증"), log)
        
        # 최고 Sharpe 타이밍
        sharpe_test = pbto.evaluate_bull_timing(TEST_YEARS, best_sharpe["confirm_days"], best_sharpe["entry_delay"], best_sharpe["exit_delay"])
        log_and_print(fmt_metrics(sharpe_test, f"최고 Sharpe 타이밍 검증 c={best_sharpe['confirm_days']},e={best_sharpe['entry_delay']},x={best_sharpe['exit_delay']}"), log)
        
        # 최고 PnL 타이밍
        pnl_test = pbto.evaluate_bull_timing(TEST_YEARS, best_pnl["confirm_days"], best_pnl["entry_delay"], best_pnl["exit_delay"])
        log_and_print(fmt_metrics(pnl_test, f"최고 PnL 타이밍 검증 c={best_pnl['confirm_days']},e={best_pnl['entry_delay']},x={best_pnl['exit_delay']}"), log)
        
        # 최고 Calmar 타이밍
        calmar_test = pbto.evaluate_bull_timing(TEST_YEARS, best_calmar["confirm_days"], best_calmar["entry_delay"], best_calmar["exit_delay"])
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
        log_and_print(f"  훈련: PnL={pbto.evaluate_bull_timing(TRAIN_YEARS, 1, 0, 0)['total_pnl_krw']:,.0f}, "
                      f"Sharpe={pbto.evaluate_bull_timing(TRAIN_YEARS, 1, 0, 0)['sharpe_daily']:.3f}", log)
        log_and_print(f"  검증: PnL={baseline_test['total_pnl_krw']:,.0f}, Sharpe={baseline_test['sharpe_daily']:.3f}", log)
        log_and_print("=" * 100, log)


if __name__ == "__main__":
    main()
