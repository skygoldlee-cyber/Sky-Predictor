# -*- coding: utf-8 -*-
"""
숏-전용 피봇 반전 + 하락장 필터 파라미터 재최적화.

- 하락장 필터를 5가지 정의로 확장 (MA20, ADX, 연속하락, 변동성 확대, 복합)
- 각 필터별로 숏-전용 피봇 파라미터를 train 기간에서 grid search
- train Sharpe/PnL 최고 조합을 test 기간(2026-01-01 ~ 2026-06-19)에서 OOS 평가
- 추가로 요청 OOS 기간(2025-10-01 ~ 2026-01-31)도 평가
"""
import sys
import os
import math
import contextlib
from pathlib import Path
from datetime import datetime, date
from itertools import product

sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg

DB_PATH = "c:/Project/SkyPredictor/Devcenter/data/duckdb/market_data.duckdb"
OUTPUT_MD = Path(__file__).parent / "pivot_short_bear_optimize.md"

BT = pv.BacktestConfig(
    multiplier=250_000,
    commission_pct_per_side=0.00015,
    slippage_ticks_per_side=1.0,
    tick_size=0.05,
    entry_on="next_open",
    annualization=252.0,
    intraday_only=True,
    session_boundary_hour=8,
    direction_mode="short_only",
)

# 최적화 파라미터 그리드
PARAM_GRID = {
    "base_pct": [0.5],
    "base_multiplier": [1.5, 2.0, 2.5],
    "atr_weight": [0.0, 0.3],
    "confirmation_bars": [1, 2],
    "min_wave_pct": [0.3],
    "min_pivot_interval_bars": [10, 20, 30],
}


def load_data():
    df_1min = pv.load_data_by_date(
        DB_PATH, "futures_1min", start="2025-06-24", end="2026-06-19"
    )
    df_1min = pv.filter_day_session(df_1min, start="08:45", end="15:45")
    df_5min = df_1min.resample("5min").agg({
        "OPEN": "first", "HIGH": "max", "LOW": "min",
        "CLOSE": "last", "VOLUME": "sum",
    }).dropna()
    df_5min = pv.compute_indicators(df_5min)
    return df_5min


def compute_filter_dates(daily: pd.DataFrame):
    """다양한 하락장 필터 정의에 따른 매매 대상일을 반환."""
    daily = daily.copy()
    daily["MA20"] = daily["CLOSE"].rolling(20).mean()
    daily["RETURN"] = daily["CLOSE"].pct_change()
    daily["ATR14"] = pv._atr(daily, 14)
    daily["ATR20_MA"] = daily["ATR14"].rolling(20).mean()
    daily["ADX14"] = pv._adx(daily, 14)

    base_conditions = {
        "MA20_down": (daily["CLOSE"] < daily["MA20"]) & (daily["RETURN"] < 0),
        "ADX_bear": (daily["ADX14"] > 25) & (daily["CLOSE"] < daily["MA20"]),
        "consecutive_2down": (daily["RETURN"] < 0) & (daily["RETURN"].shift(1) < 0),
        "volatility_exp": (daily["ATR14"] > 1.2 * daily["ATR20_MA"]) & (daily["RETURN"] < 0),
        "MA20_or_strong": (daily["CLOSE"] < daily["MA20"]) | (daily["RETURN"] < -0.01),
    }

    # 복합 필터 (당일 조건의 조합)
    composite_conditions = {
        "MA20_down_and_ADX_bear": base_conditions["MA20_down"] & base_conditions["ADX_bear"],
        "MA20_down_and_2down": base_conditions["MA20_down"] & base_conditions["consecutive_2down"],
        "MA20_down_or_ADX_bear": base_conditions["MA20_down"] | base_conditions["ADX_bear"],
        "any2_of_3": (
            base_conditions["MA20_down"].astype(int)
            + base_conditions["ADX_bear"].astype(int)
            + base_conditions["consecutive_2down"].astype(int)
        ) >= 2,
        "strong_or_MA20_not_vol": (
            base_conditions["MA20_or_strong"] | base_conditions["MA20_down"]
        ) & ~base_conditions["volatility_exp"],
    }

    conditions = {**base_conditions, **composite_conditions}

    descriptions = {
        "MA20_down": "CLOSE < MA20 and return < 0",
        "ADX_bear": "ADX > 25 and CLOSE < MA20",
        "consecutive_2down": "today and yesterday returns < 0",
        "volatility_exp": "ATR14 > 1.2 * ATR20_MA and return < 0",
        "MA20_or_strong": "CLOSE < MA20 or return < -1%",
        "MA20_down_and_ADX_bear": "MA20_down AND ADX_bear",
        "MA20_down_and_2down": "MA20_down AND consecutive_2down",
        "MA20_down_or_ADX_bear": "MA20_down OR ADX_bear",
        "any2_of_3": "at least 2 of MA20_down/ADX_bear/consecutive_2down",
        "strong_or_MA20_not_vol": "(MA20_down or strong_down) AND NOT volatility_exp",
    }

    trade_dates = {}
    for name, cond in conditions.items():
        # 인과적: 전일 조건 충족 → 오늘 매매
        trade_dates[name] = set(cond.shift(1).fillna(False).loc[lambda s: s].index.date)
    return trade_dates, conditions, descriptions


def run_short_pivot(df: pd.DataFrame, pcfg: pv.HybridAdaptivePivotConfig,
                    fcfg: pv.FilterConfig):
    bt = pv.BacktestConfig(**BT.__dict__)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        pivots = pv.detect_pivots_daily(df, pcfg, fcfg, bt.session_boundary_hour)
    return pv.backtest(df, pivots, bt)


def run_long_or_flat(df: pd.DataFrame):
    return rg.regime_intraday_daily(
        df, BT, regime_method="adx", ma_short=20, ma_long=60, adx_threshold=25.0
    )


def get_entry_dates(trades: pd.DataFrame):
    if trades is None or trades.empty:
        return pd.Series([], dtype=object)
    if "entry_time" in trades.columns:
        return pd.to_datetime(trades["entry_time"]).dt.date
    return pd.to_datetime(trades["exit_time"]).dt.date


def filter_trades_by_dates(trades: pd.DataFrame, dates: set):
    if trades is None or trades.empty:
        return trades
    entry_dates = get_entry_dates(trades)
    return trades[entry_dates.isin(dates)].copy()


def compute_metrics(tdf: pd.DataFrame, cfg: pv.BacktestConfig):
    if tdf is None or tdf.empty:
        return pv.BacktestResult(trades=pd.DataFrame())
    net = tdf["net_pts"]
    wins = net[net > 0]
    losses = net[net < 0]
    total_pts = float(net.sum())
    n_trades = int(len(tdf))
    win_rate = float((net > 0).mean() * 100)
    expectancy = float(net.mean())
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = (
        float(gross_win / gross_loss) if gross_loss > 0
        else (float("inf") if gross_win > 0 else 0.0)
    )
    tdf["exit_date"] = pd.to_datetime(tdf["exit_time"]).dt.date
    daily = tdf.groupby("exit_date")["net_krw"].sum()
    if len(daily) >= 2 and daily.std(ddof=1) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(cfg.annualization))
    else:
        sharpe = 0.0
    equity = tdf["net_krw"].cumsum()
    max_dd = float((equity - equity.cummax()).min())
    return pv.BacktestResult(
        n_trades=n_trades,
        win_rate=win_rate,
        total_pnl_pts=total_pts,
        total_pnl_krw=total_pts * cfg.multiplier,
        expectancy_pts=expectancy,
        profit_factor=profit_factor,
        sharpe_daily=sharpe,
        max_drawdown_krw=max_dd,
        trades=tdf,
    )


def fmt_result(res: pv.BacktestResult):
    return (f"거래={res.n_trades:>4} | 승률={res.win_rate:>6.2f}% | "
            f"PnL={res.total_pnl_krw:>13,.0f} | Sharpe={res.sharpe_daily:>7.3f} | "
            f"MaxDD={res.max_drawdown_krw:>13,.0f}")


def main():
    df_5min = load_data()
    daily = rg.to_daily(df_5min, BT.session_boundary_hour)
    all_dates = set(daily.index.date)
    filter_dates, conditions, descriptions = compute_filter_dates(daily)

    train_dates = {d for d in all_dates if d <= date(2025, 12, 31)}
    test_dates = {d for d in all_dates if d >= date(2026, 1, 1)}
    request_oos_dates = {d for d in all_dates if date(2025, 10, 1) <= d <= date(2026, 1, 31)}

    print(f"전체 5분봉: {len(df_5min)}봉, {df_5min.index[0]} ~ {df_5min.index[-1]}")
    print(f"전체 거래일: {len(all_dates)}일, train={len(train_dates)}일, test={len(test_dates)}일")
    for name, dates in filter_dates.items():
        desc = descriptions.get(name, name)
        print(f"  필터 '{name}' ({desc}): 전체 {len(dates)}일, train {len(dates & train_dates)}일, test {len(dates & test_dates)}일")

    # 기준 숏-전용 피봇 (in-sample 최고 파라미터)
    default_pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=0.5, base_multiplier=2.0, atr_weight=0.3, confirmation_bars=2
    )
    default_fcfg = pv.FilterConfig(
        enabled=True, min_wave_pct=0.3, min_pivot_interval_bars=20,
        st_distance_threshold=0.1, adx_hold_threshold=15.0
    )
    print("\n기준 숏-전용 피봇 계산 중...")
    default_res = run_short_pivot(df_5min, default_pcfg, default_fcfg)

    # 그리드 서치
    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    print(f"\n총 파라미터 조합: {len(combos)}")

    results = []  # (filter_name, combo, train_metrics, test_metrics, request_metrics)
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
        print(f"  [{i}/{len(combos)}] params={params}")
        res = run_short_pivot(df_5min, pcfg, fcfg)
        if res.trades is None or res.trades.empty:
            continue
        for fname, fdates in filter_dates.items():
            train_res = compute_metrics(filter_trades_by_dates(res.trades, fdates & train_dates), BT)
            test_res = compute_metrics(filter_trades_by_dates(res.trades, fdates & test_dates), BT)
            request_res = compute_metrics(filter_trades_by_dates(res.trades, fdates & request_oos_dates), BT)
            results.append({
                "filter": fname,
                "params": params,
                "train_sharpe": train_res.sharpe_daily,
                "train_pnl": train_res.total_pnl_krw,
                "test_sharpe": test_res.sharpe_daily,
                "test_pnl": test_res.total_pnl_krw,
                "request_sharpe": request_res.sharpe_daily,
                "request_pnl": request_res.total_pnl_krw,
                "train_res": train_res,
                "test_res": test_res,
                "request_res": request_res,
            })

    # 롱-또는-플랫 베이스라인
    print("\n롱-또는-플랫 베이스라인 계산 중...")
    lf_res = run_long_or_flat(df_5min)

    lines = [
        "# 숏-전용 피봇 + 하락장 필터 파라미터 재최적화",
        "",
        "## 개요",
        "- **목표**: 숏-전용 피봇 반전에 맞춘 하락장 필터와 피봇 파라미터를 동시에 최적화",
        "- **Train 기간**: 2025-06-25 ~ 2025-12-31",
        "- **Test 기간 (OOS)**: 2026-01-01 ~ 2026-06-19",
        "- **추가 OOS 기간**: 2025-10-01 ~ 2026-01-31 (train과 일부 겹침)",
        "",
        f"- **전체 거래일**: {len(all_dates)}일",
        f"- **Train 거래일**: {len(train_dates)}일",
        f"- **Test 거래일**: {len(test_dates)}일",
        "",
        "## 하락장 필터 정의",
        "",
        "| 필터 | 조건 | 전체 일수 | train | test |",
        "|---|---|---:|---:|---:|",
    ]

    for name, dates in filter_dates.items():
        desc = descriptions.get(name, name)
        lines.append(
            f"| {name} | {desc} | {len(dates)} | {len(dates & train_dates)} | {len(dates & test_dates)} |"
        )
    lines.append("")

    # 각 필터별 최고 Sharpe / PnL
    lines.append("## 필터별 최적 파라미터 (train 기준)")
    lines.append("")
    for fname in filter_dates.keys():
        sub = [r for r in results if r["filter"] == fname]
        if not sub:
            continue
        best_sharpe = max(sub, key=lambda x: x["train_sharpe"])
        best_pnl = max(sub, key=lambda x: x["train_pnl"])

        lines.append(f"### {fname}")
        lines.append("")
        lines.append("**Train 최고 Sharpe**")
        lines.append(f"- params={best_sharpe['params']}")
        lines.append(f"- train: " + fmt_result(best_sharpe["train_res"]))
        lines.append(f"- test:  " + fmt_result(best_sharpe["test_res"]))
        lines.append(f"- request OOS:  " + fmt_result(best_sharpe["request_res"]))
        lines.append("")
        lines.append("**Train 최고 PnL**")
        lines.append(f"- params={best_pnl['params']}")
        lines.append(f"- train: " + fmt_result(best_pnl["train_res"]))
        lines.append(f"- test:  " + fmt_result(best_pnl["test_res"]))
        lines.append(f"- request OOS:  " + fmt_result(best_pnl["request_res"]))
        lines.append("")

    # 종합 비교
    lines.append("## 종합 비교 (test 기준)")
    lines.append("")
    lines.append("| 필터 | 최적기준 | 거래 | 승률 | PnL | Sharpe | MaxDD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for fname in filter_dates.keys():
        sub = [r for r in results if r["filter"] == fname]
        if not sub:
            continue
        best_sharpe = max(sub, key=lambda x: x["train_sharpe"])
        lines.append(
            f"| {fname} | train Sharpe | {best_sharpe['test_res'].n_trades} | "
            f"{best_sharpe['test_res'].win_rate:.2f}% | {best_sharpe['test_res'].total_pnl_krw:,.0f} | "
            f"{best_sharpe['test_res'].sharpe_daily:.3f} | {best_sharpe['test_res'].max_drawdown_krw:,.0f} |"
        )
    # 베이스라인
    default_test = compute_metrics(filter_trades_by_dates(default_res.trades, test_dates), BT)
    default_request = compute_metrics(filter_trades_by_dates(default_res.trades, request_oos_dates), BT)
    lines.append(
        f"| 기준 숏-전용(무조건) | default | {default_test.n_trades} | "
        f"{default_test.win_rate:.2f}% | {default_test.total_pnl_krw:,.0f} | "
        f"{default_test.sharpe_daily:.3f} | {default_test.max_drawdown_krw:,.0f} |"
    )
    lf_train = compute_metrics(filter_trades_by_dates(lf_res.trades, train_dates), BT)
    lf_test = compute_metrics(filter_trades_by_dates(lf_res.trades, test_dates), BT)
    lf_request = compute_metrics(filter_trades_by_dates(lf_res.trades, request_oos_dates), BT)
    lines.append(
        f"| 롱-또는-플랫 | default | {lf_test.n_trades} | "
        f"{lf_test.win_rate:.2f}% | {lf_test.total_pnl_krw:,.0f} | "
        f"{lf_test.sharpe_daily:.3f} | {lf_test.max_drawdown_krw:,.0f} |"
    )
    lines.append("")

    lines.append("## 해석")
    lines.append("- **train Sharpe 기준**으로 고른 파라미터가 test에서도 우수한지 확인.")
    lines.append("- **test 기준**으로 보는 것이 가장 신뢰할 만한 OOS 평가.")
    lines.append("- 필터별 trade 빈도가 적으면(예: ADX_bear 13일) 통계적 검증력이 낮음.")
    lines.append("- **요청 OOS 기간(2025-10~2026-01)**은 train과 겹치므로 참고용으로만 사용.")

    md_text = "\n".join(lines)
    OUTPUT_MD.write_text(md_text, encoding="utf-8")
    print(f"\n결과 저장 완료: {OUTPUT_MD}")
    print(md_text)


if __name__ == "__main__":
    main()
