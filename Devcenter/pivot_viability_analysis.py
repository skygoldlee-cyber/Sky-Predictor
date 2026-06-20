# -*- coding: utf-8 -*-
"""
피봇반전 vs 롱-또는-플랫 비교 분석

동일 데이터(5분봉, 동일 비용 모델)로 기본/탐색 파라미터의 피봇반전을 백테스트하고
롱-또는-플랫(MA20/60+ADX)과 비교한다.
"""
import sys
from pathlib import Path
from itertools import product
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

import pandas as pd
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg

DB_PATH = "c:/Project/SkyPredictor/Devcenter/data/duckdb/market_data.duckdb"
OUTPUT_LOG = Path(__file__).parent / "pivot_viability_analysis.log"

# 동일 비용/승수 모델 (롱-또는-플랫 config 와 통일)
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
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


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
    pivots = pv.detect_pivots_daily(df, pcfg, fcfg, bt.session_boundary_hour)
    return pv.backtest(df, pivots, bt)


def fmt_result(res: pv.BacktestResult, params: str = ""):
    return (f"{params:<45} | 거래={res.n_trades:>4} | 승률={res.win_rate:>6.2f}% | "
            f"PnL={res.total_pnl_krw:>13,.0f} | Sharpe={res.sharpe_daily:>7.3f} | "
            f"MaxDD={res.max_drawdown_krw:>13,.0f}")


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        log_and_print("=" * 100, log)
        log_and_print("피봇반전 vs 롱-또는-플랫 비교 분석", log)
        log_and_print("=" * 100, log)

        df = load_data()
        log_and_print(f"5분봉 데이터: {len(df)} 봉, {df.index[0]} ~ {df.index[-1]}", log)

        # 1) 롱-또는-플랫 벤치마크
        log_and_print("\n[1] 롱-또는-플랫 벤치마크 (MA20/60 + ADX25, 숏 금지)", log)
        lf_res = run_long_or_flat(df)
        log_and_print(fmt_result(lf_res, "롱-또는-플랫"), log)

        # 2) 피봇반전 기본 파라미터
        log_and_print("\n[2] 피봇반전 기본 파라미터", log)
        pcfg_default = pv.HybridAdaptivePivotConfig(
            base_pct=0.5, base_multiplier=1.5, atr_weight=0.3, confirmation_bars=3
        )
        fcfg_default = pv.FilterConfig(
            enabled=True, min_wave_pct=0.3, min_pivot_interval_bars=10,
            st_distance_threshold=0.1, adx_hold_threshold=15.0
        )
        for mode in ("both", "long_only"):
            res = run_pivot(df, pcfg_default, fcfg_default, mode)
            log_and_print(fmt_result(res, f"기본 피봇 ({mode})"), log)

        # 3) 집중 그리드 탐색 (비교적 적은 조합)
        log_and_print("\n[3] 피봇 파라미터 집중 그리드 탐색", log)
        param_grid = {
            "base_pct": [0.2, 0.5],
            "base_multiplier": [1.0, 2.0],
            "atr_weight": [0.0, 0.3],
            "confirmation_bars": [1, 2],
            "min_wave_pct": [0.2, 0.3],
            "min_pivot_interval_bars": [5, 10, 20],
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
                    res = run_pivot(df, pcfg, fcfg, mode)
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

        # 정렬
        best_sharpe = max(results, key=lambda x: x["sharpe"])
        best_pnl = max(results, key=lambda x: x["pnl"])
        best_calmar = max(results, key=lambda x: x["pnl"] / abs(x["maxdd"]) if x["maxdd"] else 0)
        top_sharpe = sorted(results, key=lambda x: x["sharpe"], reverse=True)[:10]
        top_pnl = sorted(results, key=lambda x: x["pnl"], reverse=True)[:10]

        log_and_print("\n[최고 Sharpe]", log)
        log_and_print(f"  mode={best_sharpe['mode']}, params={best_sharpe['params']}", log)
        log_and_print(f"  거래={best_sharpe['trades']}, 승률={best_sharpe['win_rate']:.2f}%, "
                      f"PnL={best_sharpe['pnl']:,.0f}, Sharpe={best_sharpe['sharpe']:.3f}, "
                      f"MaxDD={best_sharpe['maxdd']:,.0f}", log)

        log_and_print("\n[최고 PnL]", log)
        log_and_print(f"  mode={best_pnl['mode']}, params={best_pnl['params']}", log)
        log_and_print(f"  거래={best_pnl['trades']}, 승률={best_pnl['win_rate']:.2f}%, "
                      f"PnL={best_pnl['pnl']:,.0f}, Sharpe={best_pnl['sharpe']:.3f}, "
                      f"MaxDD={best_pnl['maxdd']:,.0f}", log)

        log_and_print("\n[최고 Calmar (PnL/|MaxDD|)]", log)
        log_and_print(f"  mode={best_calmar['mode']}, params={best_calmar['params']}", log)
        log_and_print(f"  거래={best_calmar['trades']}, 승률={best_calmar['win_rate']:.2f}%, "
                      f"PnL={best_calmar['pnl']:,.0f}, Sharpe={best_calmar['sharpe']:.3f}, "
                      f"MaxDD={best_calmar['maxdd']:,.0f}", log)

        log_and_print("\n[상위 10 Sharpe]", log)
        for r in top_sharpe:
            log_and_print(f"  {r['mode']:<10} {str(r['params']):<60} | "
                          f"거래={r['trades']:>4} | 승률={r['win_rate']:>6.2f}% | "
                          f"PnL={r['pnl']:>13,.0f} | Sharpe={r['sharpe']:>7.3f} | "
                          f"MaxDD={r['maxdd']:>13,.0f}", log)

        log_and_print("\n[상위 10 PnL]", log)
        for r in top_pnl:
            log_and_print(f"  {r['mode']:<10} {str(r['params']):<60} | "
                          f"거래={r['trades']:>4} | 승률={r['win_rate']:>6.2f}% | "
                          f"PnL={r['pnl']:>13,.0f} | Sharpe={r['sharpe']:>7.3f} | "
                          f"MaxDD={r['maxdd']:>13,.0f}", log)

        # 4) 요약
        log_and_print("\n" + "=" * 100, log)
        log_and_print("요약", log)
        log_and_print("=" * 100, log)
        log_and_print(f"롱-또는-플랫:  거래={lf_res.n_trades}, 승률={lf_res.win_rate:.2f}%, "
                      f"PnL={lf_res.total_pnl_krw:,.0f}, Sharpe={lf_res.sharpe_daily:.3f}, "
                      f"MaxDD={lf_res.max_drawdown_krw:,.0f}", log)
        log_and_print(f"최고 피봇(Sharpe):  mode={best_sharpe['mode']}, PnL={best_sharpe['pnl']:,.0f}, "
                      f"Sharpe={best_sharpe['sharpe']:.3f}, MaxDD={best_sharpe['maxdd']:,.0f}", log)
        log_and_print(f"최고 피봇(PnL):     mode={best_pnl['mode']}, PnL={best_pnl['pnl']:,.0f}, "
                      f"Sharpe={best_pnl['sharpe']:.3f}, MaxDD={best_pnl['maxdd']:,.0f}", log)
        log_and_print("=" * 100, log)


if __name__ == "__main__":
    main()
