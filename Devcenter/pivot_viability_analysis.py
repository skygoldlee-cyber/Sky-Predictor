# -*- coding: utf-8 -*-
"""
피봇반전 vs 롱-또는-플랫 비교 분석

동일 데이터(5분봉, 동일 비용 모델)로 기본/탐색 파라미터의 피봇반전을 백테스트하고
롱-또는-플랫(MA20/60+ADX)과 비교한다.
"""
import sys
import math
from pathlib import Path
from itertools import product
from datetime import datetime

# 프로젝트 루트 경로 추가 (indicators 모듈 import용)
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg

DB_PATH = "c:/Project/SkyPredictor v1/Devcenter/duckdb/market_data.duckdb"
OUTPUT_LOG = Path(__file__).parent / "pivot_viability_analysis_2019plus.log"

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


def load_data_by_year(year: int):
    """특정 연도의 5분봉 데이터 로드"""
    import duckdb
    start_date = f"{year}-01-01 00:00:00"
    end_date = f"{year}-12-31 23:59:59"
    
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(
        f"SELECT * FROM futures_5min "
        f"WHERE timestamp >= '{start_date}' "
        f"AND timestamp <= '{end_date}' "
        f"ORDER BY timestamp"
    ).df()
    con.close()
    
    # timestamp 컬럼 처리 (VARCHAR -> datetime)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.columns = df.columns.str.upper()
    
    # 주간세션 필터
    df = pv.filter_day_session(df, start="08:45", end="15:45")
    
    # 지표 계산
    df = pv.compute_indicators(df)
    return df


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


def combine_results(results_list):
    """여러 BacktestResult를 합산하여 하나의 결과로 반환"""
    if not results_list:
        return pv.BacktestResult()
    
    total_trades = sum(r.n_trades for r in results_list)
    total_pnl_pts = sum(r.total_pnl_pts for r in results_list)
    total_pnl_krw = sum(r.total_pnl_krw for r in results_list)
    
    if total_trades > 0:
        win_rate = sum(r.win_rate * r.n_trades for r in results_list) / total_trades
        expectancy_pts = total_pnl_pts / total_trades
        expectancy_krw = total_pnl_krw / total_trades
        
        # Profit Factor 계산
        gross_win = sum(r.trades[r.trades["net_pts"] > 0]["net_pts"].sum() if r.trades is not None else 0 for r in results_list)
        gross_loss = sum(-r.trades[r.trades["net_pts"] < 0]["net_pts"].sum() if r.trades is not None else 0 for r in results_list)
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
        
        # 일별 PnL 합산 후 Sharpe 계산
        all_daily = []
        for r in results_list:
            if r.trades is not None and len(r.trades) > 0:
                r.trades["exit_date"] = pd.to_datetime(r.trades["exit_time"]).dt.date
                daily = r.trades.groupby("exit_date")["net_krw"].sum()
                all_daily.append(daily)
        
        if all_daily:
            combined_daily = pd.concat(all_daily)
            if len(combined_daily) >= 2 and combined_daily.std(ddof=1) > 0:
                sharpe = float(combined_daily.mean() / combined_daily.std(ddof=1) * math.sqrt(BT.annualization))
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        # Max Drawdown 계산
        all_equity = []
        for r in results_list:
            if r.trades is not None and len(r.trades) > 0:
                equity = r.trades["net_krw"].cumsum()
                all_equity.append(equity)
        
        if all_equity:
            combined_equity = pd.concat(all_equity)
            running_max = combined_equity.cummax()
            max_dd = float((combined_equity - running_max).min())
        else:
            max_dd = 0.0
        
        # 모든 거래 합치기
        all_trades = pd.concat([r.trades for r in results_list if r.trades is not None], ignore_index=True)
    else:
        win_rate = 0.0
        expectancy_pts = 0.0
        expectancy_krw = 0.0
        profit_factor = 0.0
        sharpe = 0.0
        max_dd = 0.0
        all_trades = None
    
    return pv.BacktestResult(
        n_trades=total_trades,
        win_rate=win_rate,
        total_pnl_pts=total_pnl_pts,
        total_pnl_krw=total_pnl_krw,
        expectancy_pts=expectancy_pts,
        expectancy_krw=expectancy_krw,
        profit_factor=profit_factor,
        sharpe_daily=sharpe,
        max_drawdown_krw=max_dd,
        trades=all_trades,
    )


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        log_and_print("=" * 100, log)
        log_and_print("피봇반전 vs 롱-또는-플랫 비교 분석 (연도별 분할 백테스트)", log)
        log_and_print("=" * 100, log)

        # 연도별로 백테스트 실행 (최근 2년: 2024-2025)
        years = list(range(2024, 2026))
        
        # 1) 롱-또는-플랫 벤치마크 (연도별 실행 후 합산)
        log_and_print("\n[1] 롱-또는-플랫 벤치마크 (MA20/60 + ADX25, 숏 금지)", log)
        lf_results_by_year = []
        for year in years:
            log_and_print(f"  [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_long_or_flat(df_year)
                lf_results_by_year.append(res)
                log_and_print(f"    거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}", log)
        
        lf_res = combine_results(lf_results_by_year)
        log_and_print(fmt_result(lf_res, "롱-또는-플랫 (전체 합산)"), log)

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
            log_and_print(f"  기본 피봇 ({mode})", log)
            pivot_results_by_year = []
            for year in years:
                log_and_print(f"    [{year}년 백테스트 중...]", log)
                df_year = load_data_by_year(year)
                if len(df_year) > 0:
                    res = run_pivot(df_year, pcfg_default, fcfg_default, mode)
                    pivot_results_by_year.append(res)
                    log_and_print(f"      거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}", log)
            res = combine_results(pivot_results_by_year)
            log_and_print(fmt_result(res, f"기본 피봇 ({mode}) - 전체 합산"), log)

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
                    log_and_print(f"  [{i}/{len(combos)}] {mode} {params}", log)
                    pivot_results_by_year = []
                    for year in years:
                        df_year = load_data_by_year(year)
                        if len(df_year) > 0:
                            res = run_pivot(df_year, pcfg, fcfg, mode)
                            pivot_results_by_year.append(res)
                    combined_res = combine_results(pivot_results_by_year)
                    results.append({
                        "params": params,
                        "mode": mode,
                        "sharpe": combined_res.sharpe_daily,
                        "pnl": combined_res.total_pnl_krw,
                        "maxdd": combined_res.max_drawdown_krw,
                        "trades": combined_res.n_trades,
                        "win_rate": combined_res.win_rate,
                    })
                    log_and_print(f"    거래={combined_res.n_trades}, PnL={combined_res.total_pnl_krw:,.0f}, Sharpe={combined_res.sharpe_daily:.3f}", log)
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
