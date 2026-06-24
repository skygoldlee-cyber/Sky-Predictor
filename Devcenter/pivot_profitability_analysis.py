# -*- coding: utf-8 -*-
"""
피봇반전 vs 롱-또는-플랫 수익성 분석

pivot_viability_analysis.py를 기반으로 전체 기간(2019-2025)에 대해
피봇 반전 전략과 롱-또는-플랫 전략의 수익성을 비교 분석한다.
"""
import sys
import math
from pathlib import Path
from datetime import datetime
from typing import List

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg
import pivot_regime_optimizer as pro

DB_PATH = "c:/Project/SkyPredictor v1/Devcenter/duckdb/market_data.duckdb"
OUTPUT_LOG = Path(__file__).parent / "pivot_profitability_analysis.log"

# 동일 비용/승수 모델 (기존 보고서와 동일한 조건)
BT_FULL = pv.BacktestConfig(
    multiplier=250_000,
    commission_pct_per_side=0.00003,  # 0.003% per side
    slippage_ticks_per_side=1.0,
    tick_size=0.05,
    entry_on="next_open",
    annualization=252.0,
    intraday_only=True,
    session_boundary_hour=8,
    direction_mode="both",
)

# Half Kelly 사이징 (Kelly f = 0.252, Half Kelly = 0.126x)
BT_HALF_KELLY = pv.BacktestConfig(
    multiplier=250_000 * 0.126,  # 31,500
    commission_pct_per_side=0.00003,
    slippage_ticks_per_side=1.0,
    tick_size=0.05,
    entry_on="next_open",
    annualization=252.0,
    intraday_only=True,
    session_boundary_hour=8,
    direction_mode="both",
)

# Half Kelly + 오버나잇 보유 모델
BT_HALF_KELLY_OVERNIGHT = pv.BacktestConfig(
    multiplier=250_000 * 0.126,  # 31,500
    commission_pct_per_side=0.00003,
    slippage_ticks_per_side=1.0,
    tick_size=0.05,
    entry_on="next_open",
    annualization=252.0,
    intraday_only=False,  # 오버나잇 허용
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
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.columns = df.columns.str.upper()
    
    df = pv.filter_day_session(df, start="08:45", end="15:45")
    df = pv.compute_indicators(df)
    return df


def run_long_or_flat(df: pd.DataFrame, bt_cfg: pv.BacktestConfig = None):
    bt = bt_cfg if bt_cfg is not None else BT_FULL
    return rg.regime_intraday_daily(
        df, bt, regime_method="adx", ma_short=20, ma_long=60, adx_threshold=25.0
    )


def run_pivot(df: pd.DataFrame, pcfg: pv.HybridAdaptivePivotConfig,
              fcfg: pv.FilterConfig, direction_mode: str = "both",
              bt_cfg: pv.BacktestConfig = None):
    bt = bt_cfg if bt_cfg is not None else BT_FULL
    bt.direction_mode = direction_mode
    pivots = pv.detect_pivots_daily(df, pcfg, fcfg, bt.session_boundary_hour)
    return pv.backtest(df, pivots, bt)


def get_daily_regime_for_years(years: List[int]) -> pd.Series:
    """연도 리스트에 대해 일봉 MA20/60 레짐 신호 생성."""
    regime_signals = []
    for year in years:
        df = load_data_by_year(year)
        if len(df) == 0:
            continue
        daily = rg.to_daily(df, BT_FULL.session_boundary_hour)
        signal = rg.daily_regime_signal(daily, regime_method="ma", ma_short=20, ma_long=60)
        regime_signals.append(signal)
        del df, daily
    return pd.concat(regime_signals).sort_index()


def run_pivot_bull(df: pd.DataFrame, regime_signal: pd.Series, pcfg: pv.HybridAdaptivePivotConfig,
                   fcfg: pv.FilterConfig, direction_mode: str = "long_only",
                   bt_cfg: pv.BacktestConfig = None):
    """BULL 레짐에서만 피봇 롱 전략 실행."""
    df_bull = pro._filter_df_by_regime(df, regime_signal, 1)
    if len(df_bull) == 0:
        return pv.BacktestResult()
    return run_pivot(df_bull, pcfg, fcfg, direction_mode, bt_cfg)


def run_pivot_bull_neutral(df: pd.DataFrame, regime_signal: pd.Series, pcfg: pv.HybridAdaptivePivotConfig,
                           fcfg: pv.FilterConfig, direction_mode: str = "long_only",
                           bt_cfg: pv.BacktestConfig = None):
    """BULL + NEUTRAL 레짐에서 피봇 롱 전략 실행."""
    # 레짐 신호를 데이터프레임 인덱스에 맞춰 리샘플링
    regime_per_bar = regime_signal.reindex(df.index, method='ffill')
    # BULL(1) + NEUTRAL(0) 필터링
    df_filtered = df[regime_per_bar.isin([0, 1])].copy()
    if len(df_filtered) == 0:
        return pv.BacktestResult()
    return run_pivot(df_filtered, pcfg, fcfg, direction_mode, bt_cfg)


def run_ma_crossover(df: pd.DataFrame, bt_cfg: pv.BacktestConfig = None):
    """이동평균선 크로스오버 전략 (MA20/60 Golden Cross/Death Cross)"""
    bt = bt_cfg if bt_cfg is not None else BT_FULL
    
    # 이동평균선 계산 (컬럼명은 대문자)
    df['MA20'] = df['CLOSE'].rolling(window=20).mean()
    df['MA60'] = df['CLOSE'].rolling(window=60).mean()
    
    # 크로스오버 신호 계산
    df['MA_cross'] = (df['MA20'] > df['MA60']).astype(int)
    df['MA_cross_signal'] = df['MA_cross'].diff()
    
    # Golden Cross (MA20이 MA60을 상향 돌파): 롱 진입
    # Death Cross (MA20이 MA60을 하향 돌파): 청산
    entries = []
    exits = []
    
    for i in range(1, len(df)):
        if df['MA_cross_signal'].iloc[i] == 1:  # Golden Cross
            entries.append(df.index[i])
        elif df['MA_cross_signal'].iloc[i] == -1:  # Death Cross
            exits.append(df.index[i])
    
    # 백테스트 실행을 위한 피봇 형식으로 변환
    if not entries:
        return pv.BacktestResult()
    
    # 간단한 백테스트: 진입 시점부터 청산 시점까지 수익 계산
    trades = []
    
    for entry_time in entries:
        # 해당 진입 이후의 청산 시점 찾기
        future_exits = [e for e in exits if e > entry_time]
        if not future_exits:
            continue
        
        exit_time = future_exits[0]
        entry_price = df.loc[entry_time, 'OPEN']
        exit_price = df.loc[exit_time, 'OPEN']
        
        net_pts = exit_price - entry_price
        net_krw = net_pts * bt.multiplier
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'net_pts': net_pts,
            'net_krw': net_krw
        })
    
    if not trades:
        return pv.BacktestResult()
    
    # BacktestResult 생성
    trades_df = pd.DataFrame(trades)
    total_pnl_pts = trades_df['net_pts'].sum()
    total_pnl_krw = trades_df['net_krw'].sum()
    n_trades = len(trades_df)
    n_wins = (trades_df['net_pts'] > 0).sum()
    win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0
    
    # 일별 수익 계산
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time']).dt.date
    daily_pnl = trades_df.groupby('exit_date')['net_krw'].sum()
    
    # Sharpe 계산
    if len(daily_pnl) >= 2 and daily_pnl.std(ddof=1) > 0:
        sharpe = float(daily_pnl.mean() / daily_pnl.std(ddof=1) * math.sqrt(252.0))
    else:
        sharpe = 0.0
    
    # MaxDD 계산
    cumulative = daily_pnl.cumsum()
    max_dd = (cumulative - cumulative.cummax()).min()
    
    return pv.BacktestResult(
        n_trades=n_trades,
        win_rate=win_rate,
        total_pnl_pts=total_pnl_pts,
        total_pnl_krw=total_pnl_krw,
        sharpe_daily=sharpe,
        max_drawdown_krw=max_dd,
        trades=trades_df
    )


def run_bollinger_bands(df: pd.DataFrame, bt_cfg: pv.BacktestConfig = None):
    """Bollinger Bands 전략 (하단 밴드 터치 시 롱 진입, 중심선 도달 시 청산)"""
    bt = bt_cfg if bt_cfg is not None else BT_FULL
    
    # Bollinger Bands 계산 (중심선: MA20, 상/하단 밴드: ±2 표준편차)
    df['BB_middle'] = df['CLOSE'].rolling(window=20).mean()
    df['BB_std'] = df['CLOSE'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # 하단 밴드 터치 감지 (CLOSE가 하단 밴드 이하로 하락)
    df['BB_touch_lower'] = (df['CLOSE'] <= df['BB_lower']).astype(int)
    df['BB_touch_signal'] = df['BB_touch_lower'].diff()
    
    # 중심선 도달 감지 (CLOSE가 중심선 이상으로 상승)
    df['BB_touch_middle'] = (df['CLOSE'] >= df['BB_middle']).astype(int)
    df['BB_middle_signal'] = df['BB_touch_middle'].diff()
    
    # 진입/청산 시점 수집
    entries = []
    exits = []
    
    for i in range(1, len(df)):
        if df['BB_touch_signal'].iloc[i] == 1:  # 하단 밴드 터치 (진입)
            entries.append(df.index[i])
        elif df['BB_middle_signal'].iloc[i] == 1:  # 중심선 도달 (청산)
            exits.append(df.index[i])
    
    # 백테스트 실행
    if not entries:
        return pv.BacktestResult()
    
    trades = []
    
    for entry_time in entries:
        # 해당 진입 이후의 청산 시점 찾기
        future_exits = [e for e in exits if e > entry_time]
        if not future_exits:
            continue
        
        exit_time = future_exits[0]
        entry_price = df.loc[entry_time, 'OPEN']
        exit_price = df.loc[exit_time, 'OPEN']
        
        net_pts = exit_price - entry_price
        net_krw = net_pts * bt.multiplier
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'net_pts': net_pts,
            'net_krw': net_krw
        })
    
    if not trades:
        return pv.BacktestResult()
    
    # BacktestResult 생성
    trades_df = pd.DataFrame(trades)
    total_pnl_pts = trades_df['net_pts'].sum()
    total_pnl_krw = trades_df['net_krw'].sum()
    n_trades = len(trades_df)
    n_wins = (trades_df['net_pts'] > 0).sum()
    win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0
    
    # 일별 수익 계산
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time']).dt.date
    daily_pnl = trades_df.groupby('exit_date')['net_krw'].sum()
    
    # Sharpe 계산
    if len(daily_pnl) >= 2 and daily_pnl.std(ddof=1) > 0:
        sharpe = float(daily_pnl.mean() / daily_pnl.std(ddof=1) * math.sqrt(252.0))
    else:
        sharpe = 0.0
    
    # MaxDD 계산
    cumulative = daily_pnl.cumsum()
    max_dd = (cumulative - cumulative.cummax()).min()
    
    return pv.BacktestResult(
        n_trades=n_trades,
        win_rate=win_rate,
        total_pnl_pts=total_pnl_pts,
        total_pnl_krw=total_pnl_krw,
        sharpe_daily=sharpe,
        max_drawdown_krw=max_dd,
        trades=trades_df
    )


def run_supertrend(df: pd.DataFrame, bt_cfg: pv.BacktestConfig = None, period: int = 10, multiplier: float = 1.5):
    """SuperTrend 독립 전략 (가격이 SuperTrend 상향 돌파 시 롱 진입, 하향 돌파 시 청산)"""
    bt = bt_cfg if bt_cfg is not None else BT_FULL
    
    # ATR 계산
    high = df['HIGH'].to_numpy()
    low = df['LOW'].to_numpy()
    close = df['CLOSE'].to_numpy()
    
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    atr = pd.Series(tr).rolling(window=period).mean().to_numpy()
    
    # SuperTrend 계산
    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    n = len(df)
    st = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1: uptrend, -1: downtrend
    
    for i in range(1, n):
        if np.isnan(atr[i]):
            continue
        
        if direction[i-1] == 1:  # Uptrend
            if close[i] <= lower[i]:
                st[i] = lower[i]
                direction[i] = -1
            else:
                st[i] = max(upper[i], st[i-1] if not np.isnan(st[i-1]) else upper[i])
                direction[i] = 1
        else:  # Downtrend
            if close[i] >= upper[i]:
                st[i] = upper[i]
                direction[i] = 1
            else:
                st[i] = min(lower[i], st[i-1] if not np.isnan(st[i-1]) else lower[i])
                direction[i] = -1
    
    # 첫 번째 값 설정
    st[0] = lower[0]
    
    # SuperTrend 방향 변화 감지
    df['ST'] = st
    df['ST_DIR'] = direction
    df['ST_DIR_SIGNAL'] = df['ST_DIR'].diff()
    
    # 진입/청산 시점 수집
    entries = []
    exits = []
    
    for i in range(1, len(df)):
        if df['ST_DIR_SIGNAL'].iloc[i] == 2:  # -1에서 1로 변화 (하향에서 상향): 롱 진입
            entries.append(df.index[i])
        elif df['ST_DIR_SIGNAL'].iloc[i] == -2:  # 1에서 -1로 변화 (상향에서 하향): 청산
            exits.append(df.index[i])
    
    # 백테스트 실행
    if not entries:
        return pv.BacktestResult()
    
    trades = []
    
    for entry_time in entries:
        # 해당 진입 이후의 청산 시점 찾기
        future_exits = [e for e in exits if e > entry_time]
        if not future_exits:
            continue
        
        exit_time = future_exits[0]
        entry_price = df.loc[entry_time, 'OPEN']
        exit_price = df.loc[exit_time, 'OPEN']
        
        net_pts = exit_price - entry_price
        net_krw = net_pts * bt.multiplier
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'net_pts': net_pts,
            'net_krw': net_krw
        })
    
    if not trades:
        return pv.BacktestResult()
    
    # BacktestResult 생성
    trades_df = pd.DataFrame(trades)
    total_pnl_pts = trades_df['net_pts'].sum()
    total_pnl_krw = trades_df['net_krw'].sum()
    n_trades = len(trades_df)
    n_wins = (trades_df['net_pts'] > 0).sum()
    win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0
    
    # 일별 수익 계산
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time']).dt.date
    daily_pnl = trades_df.groupby('exit_date')['net_krw'].sum()
    
    # Sharpe 계산
    if len(daily_pnl) >= 2 and daily_pnl.std(ddof=1) > 0:
        sharpe = float(daily_pnl.mean() / daily_pnl.std(ddof=1) * math.sqrt(252.0))
    else:
        sharpe = 0.0
    
    # MaxDD 계산
    cumulative = daily_pnl.cumsum()
    max_dd = (cumulative - cumulative.cummax()).min()
    
    return pv.BacktestResult(
        n_trades=n_trades,
        win_rate=win_rate,
        total_pnl_pts=total_pnl_pts,
        total_pnl_krw=total_pnl_krw,
        sharpe_daily=sharpe,
        max_drawdown_krw=max_dd,
        trades=trades_df
    )


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
        
        gross_win = sum(r.trades[r.trades["net_pts"] > 0]["net_pts"].sum() if r.trades is not None else 0 for r in results_list)
        gross_loss = sum(-r.trades[r.trades["net_pts"] < 0]["net_pts"].sum() if r.trades is not None else 0 for r in results_list)
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
        
        all_daily = []
        all_dates = []
        for r in results_list:
            if r.trades is not None and len(r.trades) > 0:
                r.trades["exit_date"] = pd.to_datetime(r.trades["exit_time"]).dt.date
                daily = r.trades.groupby("exit_date")["net_krw"].sum()
                all_daily.append(daily)
                all_dates.extend(r.trades["exit_date"].unique())
        
        if all_daily:
            combined_daily = pd.concat(all_daily)
            if len(combined_daily) >= 2 and combined_daily.std(ddof=1) > 0:
                sharpe = float(combined_daily.mean() / combined_daily.std(ddof=1) * math.sqrt(252.0))
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
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
        
        all_trades = pd.concat([r.trades for r in results_list if r.trades is not None], ignore_index=True)
        
        # 거래일 수 계산
        n_trading_days = len(set(all_dates)) if all_dates else 0
    else:
        win_rate = 0.0
        expectancy_pts = 0.0
        expectancy_krw = 0.0
        profit_factor = 0.0
        sharpe = 0.0
        max_dd = 0.0
        all_trades = None
        n_trading_days = 0
    
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
    ), n_trading_days


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        log_and_print("=" * 100, log)
        log_and_print("피봇반전 vs 롱-또는-플랫 수익성 분석 (2019-2026)", log)
        log_and_print("=" * 100, log)

        years = list(range(2019, 2027))
        
        # 1) 롱-또는-플랫 벤치마크
        log_and_print("\n[1] 롱-또는-플랫 벤치마크 (MA20/60 + ADX25, 숏 금지)", log)
        lf_results_by_year = []
        for year in years:
            log_and_print(f"  [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_long_or_flat(df_year)
                lf_results_by_year.append(res)
                log_and_print(f"    거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}", log)
        
        lf_res, lf_days = combine_results(lf_results_by_year)
        log_and_print(fmt_result(lf_res, "롱-또는-플랫 (전체 합산)"), log)
        log_and_print(f"  총 거래일: {lf_days}일", log)

        # 2) 피봇반전 기본 파라미터
        log_and_print("\n[2] 피봇반전 기본 파라미터", log)
        pcfg_default = pv.HybridAdaptivePivotConfig(
            base_pct=0.5, base_multiplier=1.5, atr_weight=0.3, confirmation_bars=3
        )
        fcfg_default = pv.FilterConfig(
            enabled=True, min_wave_pct=0.3, min_pivot_interval_bars=10,
            st_distance_threshold=0.1, adx_hold_threshold=15.0
        )
        
        # 기본 피봇(both) 결과 저장
        log_and_print(f"  기본 피봇 (both)", log)
        pivot_both_results_by_year = []
        for year in years:
            log_and_print(f"    [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_pivot(df_year, pcfg_default, fcfg_default, "both")
                pivot_both_results_by_year.append(res)
                log_and_print(f"      거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}", log)
        pivot_both_res, pivot_both_days = combine_results(pivot_both_results_by_year)
        log_and_print(fmt_result(pivot_both_res, "기본 피봇 (both) - 전체 합산"), log)
        log_and_print(f"  총 거래일: {pivot_both_days}일", log)
        
        # 기본 피봇(롱) 결과 저장
        log_and_print(f"  기본 피봇 (long_only)", log)
        pivot_long_results_by_year = []
        for year in years:
            log_and_print(f"    [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_pivot(df_year, pcfg_default, fcfg_default, "long_only")
                pivot_long_results_by_year.append(res)
                log_and_print(f"      거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}", log)
        pivot_long_res, pivot_long_days = combine_results(pivot_long_results_by_year)
        log_and_print(fmt_result(pivot_long_res, "기본 피봇 (long_only) - 전체 합산"), log)
        log_and_print(f"  총 거래일: {pivot_long_days}일", log)

        # 3) 최적 파라미터 (pivot_bull_strategy.py의 HYBRID_BULL_PARAMS 사용)
        log_and_print("\n[3] BULL 레짐 최적 파라미터 (롱 전용)", log)
        pcfg_bull = pv.HybridAdaptivePivotConfig(
            base_pct=1.272989526401749,
            base_multiplier=1.3341908735602903,
            atr_weight=0.20831334967633547,
            confirmation_bars=1,
        )
        fcfg_bull = pv.FilterConfig(
            enabled=True,
            min_wave_pct=0.07699392762885474,
            min_pivot_interval_bars=28,
            st_distance_threshold=0.1,
            adx_hold_threshold=15.0,
        )
        
        # 레짐 신호 생성
        regime_signal = get_daily_regime_for_years(years)
        
        # 3-1) 고정 사이즈 (Full)
        log_and_print("\n[3-1] BULL 최적 피봇 (고정 사이즈)", log)
        pivot_bull_full_results_by_year = []
        for year in years:
            log_and_print(f"    [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_pivot_bull(df_year, regime_signal, pcfg_bull, fcfg_bull, "long_only", BT_FULL)
                pivot_bull_full_results_by_year.append(res)
                log_and_print(f"      거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}", log)
        bull_full_res, bull_full_days = combine_results(pivot_bull_full_results_by_year)
        log_and_print(fmt_result(bull_full_res, "BULL 최적 피봇 (고정 사이즈) - 전체 합산"), log)
        log_and_print(f"  총 거래일: {bull_full_days}일", log)
        
        # 3-2) Half Kelly 사이징
        log_and_print("\n[3-2] BULL 최적 피봇 (Half Kelly 사이징)", log)
        pivot_bull_half_kelly_results_by_year = []
        pivot_bull_half_kelly_days_by_year = {}
        for year in years:
            log_and_print(f"    [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_pivot_bull(df_year, regime_signal, pcfg_bull, fcfg_bull, "long_only", BT_HALF_KELLY)
                pivot_bull_half_kelly_results_by_year.append(res)
                # 연도별 거래일 계산
                if res.trades is not None and len(res.trades) > 0:
                    res.trades["exit_date"] = pd.to_datetime(res.trades["exit_time"]).dt.date
                    n_days = len(res.trades["exit_date"].unique())
                    pivot_bull_half_kelly_days_by_year[year] = n_days
                else:
                    pivot_bull_half_kelly_days_by_year[year] = 0
                log_and_print(f"      거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}, 거래일={pivot_bull_half_kelly_days_by_year[year]}일", log)
        bull_half_kelly_res, bull_half_kelly_days = combine_results(pivot_bull_half_kelly_results_by_year)
        log_and_print(fmt_result(bull_half_kelly_res, "BULL 최적 피봇 (Half Kelly) - 전체 합산"), log)
        log_and_print(f"  총 거래일: {bull_half_kelly_days}일", log)
        log_and_print(f"  2026년 거래일: {pivot_bull_half_kelly_days_by_year.get(2026, 0)}일", log)
        
        # 3-3) Half Kelly + 오버나잇 보유 모델
        log_and_print("\n[3-3] BULL 최적 피봇 (Half Kelly + 오버나잇 보유)", log)
        pivot_bull_overnight_results_by_year = []
        pivot_bull_overnight_days_by_year = {}
        for year in years:
            log_and_print(f"    [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_pivot_bull(df_year, regime_signal, pcfg_bull, fcfg_bull, "long_only", BT_HALF_KELLY_OVERNIGHT)
                pivot_bull_overnight_results_by_year.append(res)
                # 연도별 거래일 계산
                if res.trades is not None and len(res.trades) > 0:
                    res.trades["exit_date"] = pd.to_datetime(res.trades["exit_time"]).dt.date
                    n_days = len(res.trades["exit_date"].unique())
                    pivot_bull_overnight_days_by_year[year] = n_days
                else:
                    pivot_bull_overnight_days_by_year[year] = 0
                log_and_print(f"      거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}, 거래일={pivot_bull_overnight_days_by_year[year]}일", log)
        bull_overnight_res, bull_overnight_days = combine_results(pivot_bull_overnight_results_by_year)
        log_and_print(fmt_result(bull_overnight_res, "BULL 최적 피봇 (Half Kelly + 오버나잇) - 전체 합산"), log)
        log_and_print(f"  총 거래일: {bull_overnight_days}일", log)
        log_and_print(f"  2026년 거래일: {pivot_bull_overnight_days_by_year.get(2026, 0)}일", log)
        
        # 3-4) Half Kelly + 오버나잇 + 공격적 파라미터
        log_and_print("\n[3-4] BULL 피봇 (Half Kelly + 오버나잇 + 공격적 파라미터)", log)
        pcfg_aggressive = pv.HybridAdaptivePivotConfig(
            base_pct=0.5,  # 감소 (1.273 → 0.5)
            base_multiplier=1.0,  # 감소 (1.334 → 1.0)
            atr_weight=0.3,  # 증가 (0.208 → 0.3)
            confirmation_bars=1,
        )
        fcfg_aggressive = pv.FilterConfig(
            enabled=True,
            min_wave_pct=0.03,  # 감소 (0.077 → 0.03)
            min_pivot_interval_bars=15,  # 감소 (28 → 15)
            st_distance_threshold=0.05,  # 감소 (0.1 → 0.05)
            adx_hold_threshold=10.0,  # 감소 (15 → 10)
        )
        pivot_bull_aggressive_results_by_year = []
        pivot_bull_aggressive_days_by_year = {}
        for year in years:
            log_and_print(f"    [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_pivot_bull(df_year, regime_signal, pcfg_aggressive, fcfg_aggressive, "long_only", BT_HALF_KELLY_OVERNIGHT)
                pivot_bull_aggressive_results_by_year.append(res)
                # 연도별 거래일 계산
                if res.trades is not None and len(res.trades) > 0:
                    res.trades["exit_date"] = pd.to_datetime(res.trades["exit_time"]).dt.date
                    n_days = len(res.trades["exit_date"].unique())
                    pivot_bull_aggressive_days_by_year[year] = n_days
                else:
                    pivot_bull_aggressive_days_by_year[year] = 0
                log_and_print(f"      거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}, 거래일={pivot_bull_aggressive_days_by_year[year]}일", log)
        bull_aggressive_res, bull_aggressive_days = combine_results(pivot_bull_aggressive_results_by_year)
        log_and_print(fmt_result(bull_aggressive_res, "BULL 피봇 (Half Kelly + 오버나잇 + 공격적) - 전체 합산"), log)
        log_and_print(f"  총 거래일: {bull_aggressive_days}일", log)
        log_and_print(f"  2026년 거래일: {pivot_bull_aggressive_days_by_year.get(2026, 0)}일", log)
        
        # 3-5) Half Kelly + 오버나잇 + 레짐 필터링 완화 (BULL + NEUTRAL)
        log_and_print("\n[3-5] BULL 피봇 (Half Kelly + 오버나잇 + 레짐 완화: BULL+NEUTRAL)", log)
        pivot_bull_neutral_results_by_year = []
        pivot_bull_neutral_days_by_year = {}
        for year in years:
            log_and_print(f"    [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_pivot_bull_neutral(df_year, regime_signal, pcfg_bull, fcfg_bull, "long_only", BT_HALF_KELLY_OVERNIGHT)
                pivot_bull_neutral_results_by_year.append(res)
                # 연도별 거래일 계산
                if res.trades is not None and len(res.trades) > 0:
                    res.trades["exit_date"] = pd.to_datetime(res.trades["exit_time"]).dt.date
                    n_days = len(res.trades["exit_date"].unique())
                    pivot_bull_neutral_days_by_year[year] = n_days
                else:
                    pivot_bull_neutral_days_by_year[year] = 0
                log_and_print(f"      거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}, 거래일={pivot_bull_neutral_days_by_year[year]}일", log)
        bull_neutral_res, bull_neutral_days = combine_results(pivot_bull_neutral_results_by_year)
        log_and_print(fmt_result(bull_neutral_res, "BULL 피봇 (Half Kelly + 오버나잇 + 레짐 완화) - 전체 합산"), log)
        log_and_print(f"  총 거래일: {bull_neutral_days}일", log)
        log_and_print(f"  2026년 거래일: {pivot_bull_neutral_days_by_year.get(2026, 0)}일", log)
        
        # 3-6) Half Kelly + 인트라데이 + 레짐 필터링 완화 (BULL + NEUTRAL)
        log_and_print("\n[3-6] BULL 피봇 (Half Kelly + 인트라데이 + 레짐 완화: BULL+NEUTRAL)", log)
        pivot_bull_neutral_intraday_results_by_year = []
        pivot_bull_neutral_intraday_days_by_year = {}
        for year in years:
            log_and_print(f"    [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_pivot_bull_neutral(df_year, regime_signal, pcfg_bull, fcfg_bull, "long_only", BT_HALF_KELLY)
                pivot_bull_neutral_intraday_results_by_year.append(res)
                # 연도별 거래일 계산
                if res.trades is not None and len(res.trades) > 0:
                    res.trades["exit_date"] = pd.to_datetime(res.trades["exit_time"]).dt.date
                    n_days = len(res.trades["exit_date"].unique())
                    pivot_bull_neutral_intraday_days_by_year[year] = n_days
                else:
                    pivot_bull_neutral_intraday_days_by_year[year] = 0
                log_and_print(f"      거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}, 거래일={pivot_bull_neutral_intraday_days_by_year[year]}일", log)
        bull_neutral_intraday_res, bull_neutral_intraday_days = combine_results(pivot_bull_neutral_intraday_results_by_year)
        log_and_print(fmt_result(bull_neutral_intraday_res, "BULL 피봇 (Half Kelly + 인트라데이 + 레짐 완화) - 전체 합산"), log)
        log_and_print(f"  총 거래일: {bull_neutral_intraday_days}일", log)
        log_and_print(f"  2026년 거래일: {pivot_bull_neutral_intraday_days_by_year.get(2026, 0)}일", log)
        
        # 4) 이동평균선 크로스오버 전략
        log_and_print("\n[4] 이동평균선 크로스오버 전략 (MA20/60)", log)
        ma_crossover_results_by_year = []
        ma_crossover_days_by_year = {}
        for year in years:
            log_and_print(f"  [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_ma_crossover(df_year, BT_HALF_KELLY_OVERNIGHT)
                ma_crossover_results_by_year.append(res)
                # 연도별 거래일 계산
                if res.trades is not None and len(res.trades) > 0:
                    res.trades["exit_date"] = pd.to_datetime(res.trades["exit_time"]).dt.date
                    n_days = len(res.trades["exit_date"].unique())
                    ma_crossover_days_by_year[year] = n_days
                else:
                    ma_crossover_days_by_year[year] = 0
                log_and_print(f"    거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}, 거래일={ma_crossover_days_by_year[year]}일", log)
        ma_crossover_res, ma_crossover_days = combine_results(ma_crossover_results_by_year)
        log_and_print(fmt_result(ma_crossover_res, "이동평균선 크로스오버 (Half Kelly + 오버나잇) - 전체 합산"), log)
        log_and_print(f"  총 거래일: {ma_crossover_days}일", log)
        log_and_print(f"  2026년 거래일: {ma_crossover_days_by_year.get(2026, 0)}일", log)
        
        # 5) Bollinger Bands 전략
        log_and_print("\n[5] Bollinger Bands 전략 (하단 밴드 터치 시 롱 진입)", log)
        bb_results_by_year = []
        bb_days_by_year = {}
        for year in years:
            log_and_print(f"  [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_bollinger_bands(df_year, BT_HALF_KELLY_OVERNIGHT)
                bb_results_by_year.append(res)
                # 연도별 거래일 계산
                if res.trades is not None and len(res.trades) > 0:
                    res.trades["exit_date"] = pd.to_datetime(res.trades["exit_time"]).dt.date
                    n_days = len(res.trades["exit_date"].unique())
                    bb_days_by_year[year] = n_days
                else:
                    bb_days_by_year[year] = 0
                log_and_print(f"    거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}, 거래일={bb_days_by_year[year]}일", log)
        bb_res, bb_days = combine_results(bb_results_by_year)
        log_and_print(fmt_result(bb_res, "Bollinger Bands (Half Kelly + 오버나잇) - 전체 합산"), log)
        log_and_print(f"  총 거래일: {bb_days}일", log)
        log_and_print(f"  2026년 거래일: {bb_days_by_year.get(2026, 0)}일", log)
        
        # 6) SuperTrend 독립 전략
        log_and_print("\n[6] SuperTrend 독립 전략 (가격이 SuperTrend 상향 돌파 시 롱 진입)", log)
        st_results_by_year = []
        st_days_by_year = {}
        for year in years:
            log_and_print(f"  [{year}년 백테스트 중...]", log)
            df_year = load_data_by_year(year)
            if len(df_year) > 0:
                res = run_supertrend(df_year, BT_HALF_KELLY_OVERNIGHT, period=10, multiplier=1.5)
                st_results_by_year.append(res)
                # 연도별 거래일 계산
                if res.trades is not None and len(res.trades) > 0:
                    res.trades["exit_date"] = pd.to_datetime(res.trades["exit_time"]).dt.date
                    n_days = len(res.trades["exit_date"].unique())
                    st_days_by_year[year] = n_days
                else:
                    st_days_by_year[year] = 0
                log_and_print(f"    거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}, 거래일={st_days_by_year[year]}일", log)
        st_res, st_days = combine_results(st_results_by_year)
        log_and_print(fmt_result(st_res, "SuperTrend (Half Kelly + 오버나잇) - 전체 합산"), log)
        log_and_print(f"  총 거래일: {st_days}일", log)
        log_and_print(f"  2026년 거래일: {st_days_by_year.get(2026, 0)}일", log)

        # 7) 요약 비교
        log_and_print("\n" + "=" * 100, log)
        log_and_print("수익성 비교 요약", log)
        log_and_print("=" * 100, log)
        log_and_print(f"롱-또는-플랫:            거래={lf_res.n_trades:>4}, 승률={lf_res.win_rate:>6.2f}%, "
                      f"PnL={lf_res.total_pnl_krw:>13,.0f}, Sharpe={lf_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={lf_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print(f"기본 피봇(both):         거래={pivot_both_res.n_trades:>4}, 승률={pivot_both_res.win_rate:>6.2f}%, "
                      f"PnL={pivot_both_res.total_pnl_krw:>13,.0f}, Sharpe={pivot_both_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={pivot_both_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print(f"기본 피봇(롱):           거래={pivot_long_res.n_trades:>4}, 승률={pivot_long_res.win_rate:>6.2f}%, "
                      f"PnL={pivot_long_res.total_pnl_krw:>13,.0f}, Sharpe={pivot_long_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={pivot_long_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print(f"BULL 최적 피봇 (고정):    거래={bull_full_res.n_trades:>4}, 승률={bull_full_res.win_rate:>6.2f}%, "
                      f"PnL={bull_full_res.total_pnl_krw:>13,.0f}, Sharpe={bull_full_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={bull_full_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print(f"BULL 최적 피봇 (Half Kelly): 거래={bull_half_kelly_res.n_trades:>4}, 승률={bull_half_kelly_res.win_rate:>6.2f}%, "
                      f"PnL={bull_half_kelly_res.total_pnl_krw:>13,.0f}, Sharpe={bull_half_kelly_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={bull_half_kelly_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print(f"BULL 최적 피봇 (Half Kelly + 오버나잇): 거래={bull_overnight_res.n_trades:>4}, 승률={bull_overnight_res.win_rate:>6.2f}%, "
                      f"PnL={bull_overnight_res.total_pnl_krw:>13,.0f}, Sharpe={bull_overnight_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={bull_overnight_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print(f"BULL 피봇 (Half Kelly + 오버나잇 + 공격적): 거래={bull_aggressive_res.n_trades:>4}, 승률={bull_aggressive_res.win_rate:>6.2f}%, "
                      f"PnL={bull_aggressive_res.total_pnl_krw:>13,.0f}, Sharpe={bull_aggressive_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={bull_aggressive_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print(f"BULL 피봇 (Half Kelly + 오버나잇 + 레짐 완화): 거래={bull_neutral_res.n_trades:>4}, 승률={bull_neutral_res.win_rate:>6.2f}%, "
                      f"PnL={bull_neutral_res.total_pnl_krw:>13,.0f}, Sharpe={bull_neutral_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={bull_neutral_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print(f"BULL 피봇 (Half Kelly + 인트라데이 + 레짐 완화): 거래={bull_neutral_intraday_res.n_trades:>4}, 승률={bull_neutral_intraday_res.win_rate:>6.2f}%, "
                      f"PnL={bull_neutral_intraday_res.total_pnl_krw:>13,.0f}, Sharpe={bull_neutral_intraday_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={bull_neutral_intraday_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print(f"이동평균선 크로스오버 (Half Kelly + 오버나잇): 거래={ma_crossover_res.n_trades:>4}, 승률={ma_crossover_res.win_rate:>6.2f}%, "
                      f"PnL={ma_crossover_res.total_pnl_krw:>13,.0f}, Sharpe={ma_crossover_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={ma_crossover_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print(f"Bollinger Bands (Half Kelly + 오버나잇): 거래={bb_res.n_trades:>4}, 승률={bb_res.win_rate:>6.2f}%, "
                      f"PnL={bb_res.total_pnl_krw:>13,.0f}, Sharpe={bb_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={bb_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print(f"SuperTrend (Half Kelly + 오버나잇): 거래={st_res.n_trades:>4}, 승률={st_res.win_rate:>6.2f}%, "
                      f"PnL={st_res.total_pnl_krw:>13,.0f}, Sharpe={st_res.sharpe_daily:>7.3f}, "
                      f"MaxDD={st_res.max_drawdown_krw:>13,.0f}", log)
        log_and_print("=" * 100, log)


if __name__ == "__main__":
    main()
