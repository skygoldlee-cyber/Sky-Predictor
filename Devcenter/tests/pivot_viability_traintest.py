# -*- coding: utf-8 -*-
"""
피봇반전 vs 롱-또는-플랫 — train/test 분리 검증

전체 데이터를 2025-12-31 기준으로 train/test로 나누고,
train에서 최적 피봇 파라미터를 선택한 뒤 test에서 평가한다.
"""
import sys
import os
import contextlib
from pathlib import Path
from itertools import product
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

import pandas as pd
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg

DB_PATH = "c:/Project/SkyPredictor/Devcenter/data/duckdb/market_data.duckdb"
OUTPUT_LOG = Path(__file__).parent / "pivot_viability_traintest.log"

BT = pv.BacktestConfig(
    multiplier=250_000,
    commission_pct_per_side=0.00015,
    slippage_ticks_per_side=1.0,
    tick_size=0.05,
    entry_on="next_open",
    annualization=252.0,
    intraday_only=True,
    session_boundary_hour=8,
    direction_mode="both",
)


def log_and_print(msg: str, log_file):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    log_file.write(line + "\n")
    log_file.flush()
    print(line)


def load_data():
    df_1min = pv.load_data_by_date(
        DB_PATH, "futures_1min", start="2025-06-24", end="2026-06-18"
    )
    df_1min = pv.filter_day_session(df_1min, start="08:45", end="15:45")
    df_5min = df_1min.resample("5min").agg({
        "OPEN": "first", "HIGH": "max", "LOW": "min",
        "CLOSE": "last", "VOLUME": "sum",
    }).dropna()
    df_5min = pv.compute_indicators(df_5min)
    return df_5min


def run_long_or_flat(df: pd.DataFrame):
    return rg.regime_intraday_daily(
        df, BT, regime_method="adx", ma_short=20, ma_long=60, adx_threshold=25.0
    )


def run_pivot(df: pd.DataFrame, pcfg: pv.HybridAdaptivePivotConfig,
              fcfg: pv.FilterConfig, direction_mode: str = "both"):
    bt = pv.BacktestConfig(**BT.__dict__)
    bt.direction_mode = direction_mode
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        pivots = pv.detect_pivots_daily(df, pcfg, fcfg, bt.session_boundary_hour)
        return pv.backtest(df, pivots, bt)


def fmt_result(res: pv.BacktestResult, params: str = ""):
    return (f"{params:<45} | 거래={res.n_trades:>4} | 승률={res.win_rate:>6.2f}% | "
            f"PnL={res.total_pnl_krw:>13,.0f} | Sharpe={res.sharpe_daily:>7.3f} | "
            f"MaxDD={res.max_drawdown_krw:>13,.0f}")


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        log_and_print("=" * 100, log)
        log_and_print("피봇반전 vs 롱-또는-플랫 - train/test 분리 검증", log)
        log_and_print("=" * 100, log)

        df = load_data()
        split_date = "2026-01-01"
        df_train = df.loc[df.index < split_date]
        df_test = df.loc[df.index >= split_date]
        log_and_print(f"전체: {len(df)} 봉, {df.index[0]} ~ {df.index[-1]}", log)
        log_and_print(f"Train: {len(df_train)} 봉, {df_train.index[0]} ~ {df_train.index[-1]}", log)
        log_and_print(f"Test:  {len(df_test)} 봉, {df_test.index[0]} ~ {df_test.index[-1]}", log)

        # 1) 롱-또는-플랫
        log_and_print("\n[1] 롱-또는-플랫 (고정 파라미터)", log)
        lf_train = run_long_or_flat(df_train)
        lf_test = run_long_or_flat(df_test)
        log_and_print("Train " + fmt_result(lf_train, "롱-또는-플랫"), log)
        log_and_print("Test  " + fmt_result(lf_test, "롱-또는-플랫"), log)

        # 2) 피봇 그리드 탐색 — train에서 최적 선택
        log_and_print("\n[2] 피봇 파라미터 train 내 탐색 (출력 억제)", log)
        param_grid = {
            "base_pct": [0.5],
            "base_multiplier": [1.5, 2.0, 2.5],
            "atr_weight": [0.0, 0.3],
            "confirmation_bars": [1, 2],
            "min_wave_pct": [0.2, 0.3],
            "min_pivot_interval_bars": [10, 20, 30],
        }
        direction_modes = ["both", "long_only"]
        keys = list(param_grid.keys())
        combos = list(product(*[param_grid[k] for k in keys]))
        log_and_print(f"총 조합: {len(combos) * len(direction_modes)}", log)

        results = []
        for i, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
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
            for mode in direction_modes:
                try:
                    res = run_pivot(df_train, pcfg, fcfg, mode)
                    results.append({
                        "params": params,
                        "mode": mode,
                        "sharpe": res.sharpe_daily,
                        "pnl": res.total_pnl_krw,
                        "maxdd": res.max_drawdown_krw,
                        "trades": res.n_trades,
                        "win_rate": res.win_rate,
                    })
                except Exception as e:
                    log_and_print(f"  [{i}] {mode} 오류: {e}", log)

        best_sharpe_train = max(results, key=lambda x: x["sharpe"])
        best_pnl_train = max(results, key=lambda x: x["pnl"])
        log_and_print("\n[Train 최고 Sharpe]", log)
        log_and_print(f"  mode={best_sharpe_train['mode']}, params={best_sharpe_train['params']}", log)
        log_and_print(f"  거래={best_sharpe_train['trades']}, 승률={best_sharpe_train['win_rate']:.2f}%, "
                      f"PnL={best_sharpe_train['pnl']:,.0f}, Sharpe={best_sharpe_train['sharpe']:.3f}, "
                      f"MaxDD={best_sharpe_train['maxdd']:,.0f}", log)
        log_and_print("\n[Train 최고 PnL]", log)
        log_and_print(f"  mode={best_pnl_train['mode']}, params={best_pnl_train['params']}", log)
        log_and_print(f"  거래={best_pnl_train['trades']}, 승률={best_pnl_train['win_rate']:.2f}%, "
                      f"PnL={best_pnl_train['pnl']:,.0f}, Sharpe={best_pnl_train['sharpe']:.3f}, "
                      f"MaxDD={best_pnl_train['maxdd']:,.0f}", log)

        # 3) Train 최고 Sharpe / PnL 파라미터를 test에서 평가
        log_and_print("\n[3] Train 최고 파라미터의 test 성과", log)
        for label, best in (("Train 최고 Sharpe", best_sharpe_train),
                            ("Train 최고 PnL", best_pnl_train)):
            pcfg = pv.HybridAdaptivePivotConfig(
                base_pct=best["params"]["base_pct"],
                base_multiplier=best["params"]["base_multiplier"],
                atr_weight=best["params"]["atr_weight"],
                confirmation_bars=best["params"]["confirmation_bars"],
            )
            fcfg = pv.FilterConfig(
                enabled=True,
                min_wave_pct=best["params"]["min_wave_pct"],
                min_pivot_interval_bars=best["params"]["min_pivot_interval_bars"],
                st_distance_threshold=0.1,
                adx_hold_threshold=15.0,
            )
            test_res = run_pivot(df_test, pcfg, fcfg, best["mode"])
            log_and_print("Test  " + fmt_result(test_res, f"{label} ({best['mode']})"), log)

        # 4) in-sample 최고 파라미터(test 2026 상반기)도 test에 그대로 적용
        log_and_print("\n[4] In-sample 최고 파라미터의 test 성과 (2026-01-01 ~)", log)
        oos_pcfg = pv.HybridAdaptivePivotConfig(
            base_pct=0.5, base_multiplier=2.0, atr_weight=0.3, confirmation_bars=2
        )
        oos_fcfg = pv.FilterConfig(
            enabled=True, min_wave_pct=0.3, min_pivot_interval_bars=20,
            st_distance_threshold=0.1, adx_hold_threshold=15.0
        )
        oos_res = run_pivot(df_test, oos_pcfg, oos_fcfg, "both")
        log_and_print("Test  " + fmt_result(oos_res, "In-sample 최고 파라미터 (both)"), log)

        log_and_print("\n" + "=" * 100, log)
        log_and_print("요약", log)
        log_and_print("=" * 100, log)
        log_and_print(f"롱-또는-플랫 Train: {fmt_result(lf_train, '')}", log)
        log_and_print(f"롱-또는-플랫 Test:  {fmt_result(lf_test, '')}", log)
        log_and_print(f"Train 최고 Sharpe (test): mode={best_sharpe_train['mode']}, "
                      f"params={best_sharpe_train['params']}", log)
        log_and_print(f"Train 최고 PnL  (test): mode={best_pnl_train['mode']}, "
                      f"params={best_pnl_train['params']}", log)
        log_and_print("=" * 100, log)


if __name__ == "__main__":
    main()
