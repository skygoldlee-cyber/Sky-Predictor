# -*- coding: utf-8 -*-
"""
롱-또는-플랫 개선 백테스트

기존 롱-또는-플랫에 다음을 추가해 비교한다.
- 손절/익절 (ATR 배수)
- ATR 기반 포지션 사이징
- 일일 손실 한도 (하드 리스크 캡)
"""
import sys
import math
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg

DB_PATH = "c:/Project/SkyPredictor/Devcenter/data/duckdb/market_data.duckdb"
OUTPUT_LOG = Path(__file__).parent / "long_or_flat_improved_backtest.log"

BT = pv.BacktestConfig(
    multiplier=250_000,
    commission_pct_per_side=0.00015,
    slippage_ticks_per_side=1.0,
    tick_size=0.05,
    entry_on="next_open",
    annualization=252.0,
    intraday_only=True,
    session_boundary_hour=8,
    direction_mode="long_only",
)


def log(msg: str, fh):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    fh.write(line + "\n")
    fh.flush()
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


def fmt_result(res: pv.BacktestResult, label: str = "", avg_contracts: float = 0.0):
    return (f"{label:<40} | 거래={res.n_trades:>4} | 승률={res.win_rate:>6.2f}% | "
            f"PnL={res.total_pnl_krw:>13,.0f} | Sharpe={res.sharpe_daily:>7.3f} | "
            f"MaxDD={res.max_drawdown_krw:>13,.0f} | 평균계약={avg_contracts:>5.2f}")


def baseline_long_or_flat(df: pd.DataFrame):
    """기존 롱-또는-플랫: 시가 진입, 종가 청산 (숏 금지)"""
    daily = rg.to_daily(df, BT.session_boundary_hour)
    signal = rg.daily_regime_signal(
        daily, regime_method="adx", ma_short=20, ma_long=60,
        adx_threshold=25.0, allow_short=False
    )
    return rg._bt_from_daily_signal(daily, signal, BT)


def improved_long_or_flat(
    df: pd.DataFrame,
    stop_atr_mult: float = 1.0,
    take_atr_mult: float = 2.0,
    risk_per_trade: float = 0.01,
    max_daily_loss: float = 5_000_000,
    initial_capital: float = 50_000_000,
):
    """개선 롱-또는-플랫: 일봉 ATR 기반 손절/익절 + 사이징 + 일일 손실 한도"""
    # 일봉 신호 + 일봉 ATR (전일까지 데이터만 사용)
    daily = rg.to_daily(df, BT.session_boundary_hour)
    signal = rg.daily_regime_signal(
        daily, regime_method="adx", ma_short=20, ma_long=60,
        adx_threshold=25.0, allow_short=False
    )
    daily_atr = pv._atr(daily, 14).shift(1)

    idx = df.index
    tday = pv.trading_day_key(idx, BT.session_boundary_hour)
    pos = np.arange(len(df))

    rows = []
    capital = initial_capital

    for day_val in pd.unique(tday):
        mask = tday == day_val
        day_idx = pos[mask]
        first = int(day_idx[0])
        last = int(day_idx[-1])
        if last <= first:
            continue

        day_end = idx[last]
        if day_end not in signal.index or signal.loc[day_end] != 1:
            continue

        # 전일 일봉 ATR 기준 손절/익절, 사이징
        if day_end not in daily_atr.index:
            continue
        prev_atr = float(daily_atr.loc[day_end])
        if pd.isna(prev_atr) or prev_atr <= 0:
            continue

        entry_px = float(df["OPEN"].iloc[first])
        stop_px = entry_px - stop_atr_mult * prev_atr
        take_px = entry_px + take_atr_mult * prev_atr
        stop_dist = entry_px - stop_px

        # 포지션 사이징: risk_per_trade% 자본을 stop_dist 만큼의 손실에 배당
        risk_amount = capital * risk_per_trade
        contracts = max(1, int(risk_amount / (stop_dist * BT.multiplier)))

        # 일일 손실 한도 초과 시 축소
        max_contracts_by_daily_limit = int(max_daily_loss / (stop_dist * BT.multiplier))
        if contracts > max_contracts_by_daily_limit:
            contracts = max_contracts_by_daily_limit
        if contracts <= 0:
            continue

        # 장중 손절/익절 체크 (진입 봉 이후부터)
        exit_px = float(df["CLOSE"].iloc[last])
        exit_time = idx[last]
        stopped = False
        taken = False
        for i in range(first + 1, last + 1):
            if df["LOW"].iloc[i] <= stop_px:
                exit_px = stop_px
                exit_time = idx[i]
                stopped = True
                break
            if df["HIGH"].iloc[i] >= take_px:
                exit_px = take_px
                exit_time = idx[i]
                taken = True
                break

        net_pts = (exit_px - entry_px) - BT.round_trip_cost_pts(entry_px, exit_px)
        net_krw = net_pts * BT.multiplier * contracts
        capital += net_krw
        if capital < 0:
            capital = 0

        rows.append({
            "exit_time": exit_time,
            "direction": 1,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "stop_px": stop_px,
            "take_px": take_px,
            "contracts": contracts,
            "net_pts": net_pts,
            "net_krw": net_krw,
            "stopped": stopped,
            "taken": taken,
        })

    if not rows:
        return pv.BacktestResult(trades=pd.DataFrame())

    tdf = pd.DataFrame(rows)
    daily_pnl = tdf.groupby(pd.to_datetime(tdf["exit_time"]).dt.date)["net_krw"].sum()
    sharpe = (float(daily_pnl.mean() / daily_pnl.std(ddof=1) * math.sqrt(BT.annualization))
              if len(daily_pnl) >= 2 and daily_pnl.std(ddof=1) > 0 else 0.0)
    equity = tdf["net_krw"].cumsum()
    return pv.BacktestResult(
        n_trades=len(tdf),
        win_rate=float((tdf["net_pts"] > 0).mean() * 100),
        total_pnl_pts=float(tdf["net_pts"].sum()),
        total_pnl_krw=float(tdf["net_krw"].sum()),
        expectancy_pts=float(tdf["net_pts"].mean()),
        sharpe_daily=sharpe,
        max_drawdown_krw=float((equity - equity.cummax()).min()),
        trades=tdf,
    )


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log_fh:
        log("=" * 100, log_fh)
        log("롱-또는-플랫 개선 백테스트", log_fh)
        log("=" * 100, log_fh)

        df = load_data()
        log(f"5분봉 데이터: {len(df)} 봉, {df.index[0]} ~ {df.index[-1]}", log_fh)

        # 1) 기존 롱-또는-플랫
        base = baseline_long_or_flat(df)
        log(fmt_result(base, "기존 롱-또는-플랫"), log_fh)

        # 2) 개선 버전 파라미터 탐색
        log("\n[개선 버전 파라미터 비교]", log_fh)
        candidates = [
            (0.5, 1.0, 0.01, 5_000_000),
            (0.5, 2.0, 0.01, 5_000_000),
            (1.0, 2.0, 0.01, 5_000_000),
            (1.0, 2.0, 0.01, 10_000_000),
            (1.0, 2.0, 0.02, 10_000_000),
            (1.5, 3.0, 0.01, 5_000_000),
            (1.5, 3.0, 0.01, 10_000_000),
            (1.5, 3.0, 0.01, 20_000_000),
            (2.0, 4.0, 0.01, 10_000_000),
            (2.0, 4.0, 0.01, 20_000_000),
            (2.0, 4.0, 0.02, 20_000_000),
        ]
        results = []
        for stop_atr, take_atr, risk, max_loss in candidates:
            res = improved_long_or_flat(df, stop_atr, take_atr, risk, max_loss)
            avg_contracts = (res.trades["contracts"].mean() if res.trades is not None and len(res.trades) else 0.0)
            label = f"SL{stop_atr}ATR TP{take_atr}ATR R{risk*100:.0f}% L{max_loss/1e6:.0f}M"
            log(fmt_result(res, label, avg_contracts), log_fh)
            results.append((label, res, stop_atr, take_atr, risk, max_loss, avg_contracts))

        # 최고 Sharpe (최소 30거래 이상)
        eligible = [r for r in results if r[1].n_trades >= 30]
        best = max(eligible, key=lambda x: x[1].sharpe_daily) if eligible else max(results, key=lambda x: x[1].sharpe_daily)
        log("\n[Sharpe 기준 최고 개선 조합 (거래수 30 이상)]", log_fh)
        log(f"{best[0]} | Sharpe={best[1].sharpe_daily:.3f} | PnL={best[1].total_pnl_krw:,.0f} | MaxDD={best[1].max_drawdown_krw:,.0f}", log_fh)

        # 최고 PnL
        best_pnl = max(results, key=lambda x: x[1].total_pnl_krw)
        log("\n[PnL 기준 최고 개선 조합]", log_fh)
        log(f"{best_pnl[0]} | PnL={best_pnl[1].total_pnl_krw:,.0f} | Sharpe={best_pnl[1].sharpe_daily:.3f} | MaxDD={best_pnl[1].max_drawdown_krw:,.0f}", log_fh)

        log("\n" + "=" * 100, log_fh)


if __name__ == "__main__":
    main()
