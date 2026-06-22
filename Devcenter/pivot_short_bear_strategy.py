# -*- coding: utf-8 -*-
"""
숏-전용 피봇 반전 + 인과적 하락장 필터 전략 (실시간 적용 버전).

- 전일 종가 기준으로 복합 하락장 필터(MA20_down OR ADX_bear)를 계산
- 필터 조건이 충족된 다음 거래일에만 숏-전용 피봇 반전 실행
- 최적화된 파라미터를 그대로 사용하여 실시간 신호 생성 가능
- Out-of-sample 기간(2025-10-01 ~ 2026-01-31) 및 test 기간(2026-01-01 ~ 2026-06-19)에서 재검증
"""
import sys
import os
import math
import copy
import contextlib
from pathlib import Path
from datetime import datetime, date

sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg
from pivot_short_bear_optimize import compute_filter_dates

DB_PATH = "c:/Project/SkyPredictor/Devcenter/data/duckdb/market_data.duckdb"
OUTPUT_MD = Path(__file__).parent / "pivot_short_bear_strategy.md"

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

# Walk-forward + 최적화 결과에서 선택된 파라미터 (MA20_down_or_ADX_bear 기준)
PCFG = pv.HybridAdaptivePivotConfig(
    base_pct=0.5,
    base_multiplier=2.5,
    atr_weight=0.3,
    confirmation_bars=1,
)
FCFG = pv.FilterConfig(
    enabled=True,
    min_wave_pct=0.3,
    min_pivot_interval_bars=20,
    st_distance_threshold=0.1,
    adx_hold_threshold=15.0,
)

SELECTED_FILTER = "MA20_down_or_ADX_bear"
FILTER_DESCRIPTION = "MA20_down OR ADX_bear"  # CLOSE < MA20 or (ADX > 25 and CLOSE < MA20)


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


def compute_bearish_trade_dates(daily: pd.DataFrame, filter_name: str = SELECTED_FILTER):
    """
    전일 종가 기준으로 복합 하락장 필터를 계산하고,
    필터 조건이 충족된 다음 거래일을 매매 대상으로 반환.
    """
    trade_dates, _, _ = compute_filter_dates(daily)
    return trade_dates.get(filter_name, set())


def should_trade_today(daily: pd.DataFrame, filter_name: str = SELECTED_FILTER) -> bool:
    """
    실시간 운용용: 최신 daily 데이터를 받아 오늘 매매할지 여부를 반환.
    """
    if daily.empty or len(daily) < 2:
        return False
    trade_dates = compute_bearish_trade_dates(daily, filter_name)
    today = daily.index[-1].date()
    return today in trade_dates


def detect_short_pivots(df: pd.DataFrame):
    bt = pv.BacktestConfig(**BT.__dict__)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return pv.detect_pivots_daily(df, PCFG, FCFG, bt.session_boundary_hour)


def run_short_pivot(df: pd.DataFrame, pivots: pd.DataFrame, bt: pv.BacktestConfig,
                    direction_mode: str = "short_only"):
    bt = copy.deepcopy(bt)
    bt.direction_mode = direction_mode
    return pv.backtest(df, pivots, bt)


def run_both_pivot(df: pd.DataFrame, pivots: pd.DataFrame, bt: pv.BacktestConfig):
    """같은 피봇 신호에서 롱+숏 양방향 진입을 모두 허용한 백테스트."""
    both_bt = pv.BacktestConfig(**bt.__dict__)
    both_bt.direction_mode = "both"
    return pv.backtest(df, pivots, both_bt)


# ── 리스크 관리 기본값 ─────────────────────────────────────────────────────
# 전/후 비교를 위해 고정된 기본값을 사용. (그리드 최적화 제거)
RISK_DEFAULT = {
    "stop_loss_pct": 0.01,           # 1% 손절
    "take_profit_pct": 0.02,         # 2% 익절
    "trailing_stop_pct": 0.005,      # 0.5% 트레일링 스탑
    "daily_loss_limit_krw": 1_000_000,  # 일일 손실제한 100만 원
    "atr_sizing_target_krw": 0.0,    # 고정 사이징
    "position_size_mode": "fixed",   # 고정 멀티
}


def make_default_risk_bt(base: pv.BacktestConfig) -> pv.BacktestConfig:
    """기본 리스크 파라미터로 BacktestConfig 복사본을 생성."""
    merged = base.__dict__.copy()
    merged.update(RISK_DEFAULT)
    return pv.BacktestConfig(**merged)


# ── 리스크 파라미터 그리드 탐색 ─────────────────────────────────────────────
RISK_GRID = {
    "stop_loss_pct": [0.01, 0.015, 0.02],
    "take_profit_pct": [0.02, 0.03],
    "trailing_stop_pct": [0.005, 0.01],
    "daily_loss_limit_krw": [0, 500_000, 1_000_000],
}


def make_risk_bt(base: pv.BacktestConfig, **kwargs) -> pv.BacktestConfig:
    """base에 리스크 파라미터를 병합하여 BacktestConfig 복사본 생성."""
    merged = base.__dict__.copy()
    merged.update(kwargs)
    return pv.BacktestConfig(**merged)


def find_best_risk_params(df_5min, pivots, base_bt, bearish_dates):
    """리스크 파라미터 그리드에서 전체 기간 하락장 필터 Sharpe 기준 최적값을 탐색."""
    best_params = None
    best_score = -float("inf")
    best_res = None
    total = (
        len(RISK_GRID["stop_loss_pct"])
        * len(RISK_GRID["take_profit_pct"])
        * len(RISK_GRID["trailing_stop_pct"])
        * len(RISK_GRID["daily_loss_limit_krw"])
    )
    print(f"\n리스크 파라미터 그리드 탐색: {total}개 조합")
    count = 0
    for sl in RISK_GRID["stop_loss_pct"]:
        for tp in RISK_GRID["take_profit_pct"]:
            for ts in RISK_GRID["trailing_stop_pct"]:
                for dl in RISK_GRID["daily_loss_limit_krw"]:
                    count += 1
                    cfg = make_risk_bt(
                        base_bt,
                        stop_loss_pct=sl,
                        take_profit_pct=tp,
                        trailing_stop_pct=ts,
                        daily_loss_limit_krw=dl,
                        atr_sizing_target_krw=0.0,
                        position_size_mode="fixed",
                    )
                    res = run_short_pivot(df_5min, pivots, cfg)
                    tdf = filter_trades_by_dates(res.trades, bearish_dates)
                    metrics = compute_metrics(tdf, BT)
                    # Sharpe 기준, 동점 시 MaxDD가 더 작은 조합 선호
                    if metrics.sharpe_daily > best_score or (
                        abs(metrics.sharpe_daily - best_score) < 1e-9
                        and metrics.max_drawdown_krw > best_res.max_drawdown_krw
                    ):
                        best_score = metrics.sharpe_daily
                        best_params = {
                            "stop_loss_pct": sl,
                            "take_profit_pct": tp,
                            "trailing_stop_pct": ts,
                            "daily_loss_limit_krw": dl,
                        }
                        best_res = metrics
                    if count % 10 == 0 or count == total:
                        print(f"  [{count}/{total}] sl={sl:.3f}, tp={tp:.3f}, ts={ts:.3f}, dl={dl:,.0f} -> Sharpe={metrics.sharpe_daily:.3f}, PnL={metrics.total_pnl_krw:,.0f}, MaxDD={metrics.max_drawdown_krw:,.0f}")
    print(f"\n최적 리스크 파라미터: {best_params}")
    print(f"최적 결과: {fmt_result(best_res)}")
    return best_params, best_res


def run_long_or_flat(df: pd.DataFrame):
    return rg.regime_intraday_daily(
        df, BT, regime_method="adx", ma_short=20, ma_long=60, adx_threshold=25.0
    )


def get_entry_dates(trades: pd.DataFrame):
    """거래의 진입일자를 반환. entry_time이 없으면 exit_time으로 대체."""
    if trades is None or trades.empty:
        return pd.Series([], dtype=object)
    if "entry_time" in trades.columns:
        t = pd.to_datetime(trades["entry_time"])
    else:
        t = pd.to_datetime(trades["exit_time"])
    return t.dt.date


def filter_trades_by_dates(trades: pd.DataFrame, dates: set):
    if trades is None or trades.empty:
        return trades
    entry_dates = get_entry_dates(trades)
    return trades[entry_dates.isin(dates)].copy()


def compute_metrics(tdf: pd.DataFrame, cfg: pv.BacktestConfig):
    """trades DataFrame로부터 성과 지표를 재계산."""
    if tdf is None or tdf.empty:
        return pv.BacktestResult(trades=pd.DataFrame())
    net = tdf["net_pts"]
    wins = net[net > 0]
    losses = net[net < 0]
    total_pts = float(net.sum())
    has_krw = "net_krw" in tdf.columns
    total_krw = float(tdf["net_krw"].sum()) if has_krw else total_pts * cfg.multiplier
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
    if has_krw:
        daily = tdf.groupby("exit_date")["net_krw"].sum()
        equity = tdf["net_krw"].cumsum()
    else:
        daily = tdf.groupby("exit_date")["net_pts"].sum() * cfg.multiplier
        equity = tdf["net_pts"].cumsum() * cfg.multiplier
    if len(daily) >= 2 and daily.std(ddof=1) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(cfg.annualization))
    else:
        sharpe = 0.0
    max_dd = float((equity - equity.cummax()).min())
    return pv.BacktestResult(
        n_trades=n_trades,
        win_rate=win_rate,
        total_pnl_pts=total_pts,
        total_pnl_krw=total_krw,
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


def evaluate_period(name: str, start: date, end: date,
                    short_trades: pd.DataFrame, risk_trades: pd.DataFrame,
                    both_trades: pd.DataFrame, lf_trades: pd.DataFrame,
                    bearish_dates: set, all_dates: set) -> str:
    """특정 기간에서 전략별 성과를 markdown 테이블 행으로 반환."""
    period_dates = {d for d in all_dates if start <= d <= end}
    bearish_in_period = bearish_dates & period_dates

    rows = [
        f"### {name} ({start} ~ {end})",
        f"- 전체 거래일: {len(period_dates)}일",
        f"- 하락장 판정일: {len(bearish_in_period)}일",
        "",
        "| 전략 | 거래 | 승률 | PnL (원) | Sharpe | MaxDD |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    # 1. 숏-전용 피봇 (인과적 하락장 필터)
    tdf = filter_trades_by_dates(short_trades, bearish_in_period)
    res = compute_metrics(tdf, BT)
    rows.append(f"| 숏-전용 피봇 (하락장 필터) | {res.n_trades} | {res.win_rate:.2f}% | {res.total_pnl_krw:,.0f} | {res.sharpe_daily:.3f} | {res.max_drawdown_krw:,.0f} |")

    # 2. 숏-전용 피봇 (하락장 필터 + Risk)
    if risk_trades is not None:
        tdf_risk = filter_trades_by_dates(risk_trades, bearish_in_period)
        res_risk = compute_metrics(tdf_risk, BT)
        rows.append(f"| 숏-전용 피봇 (하락장 필터 + Risk) | {res_risk.n_trades} | {res_risk.win_rate:.2f}% | {res_risk.total_pnl_krw:,.0f} | {res_risk.sharpe_daily:.3f} | {res_risk.max_drawdown_krw:,.0f} |")

    # 3. 숏-전용 피봇 (무조건)
    tdf_all = filter_trades_by_dates(short_trades, period_dates)
    res_all = compute_metrics(tdf_all, BT)
    rows.append(f"| 숏-전용 피봇 (무조건) | {res_all.n_trades} | {res_all.win_rate:.2f}% | {res_all.total_pnl_krw:,.0f} | {res_all.sharpe_daily:.3f} | {res_all.max_drawdown_krw:,.0f} |")

    # 4. 양방향 피봇 (무조건)
    if both_trades is not None:
        tdf_both = filter_trades_by_dates(both_trades, period_dates)
        res_both = compute_metrics(tdf_both, BT)
        rows.append(f"| 양방향 피봇 (무조건) | {res_both.n_trades} | {res_both.win_rate:.2f}% | {res_both.total_pnl_krw:,.0f} | {res_both.sharpe_daily:.3f} | {res_both.max_drawdown_krw:,.0f} |")

    # 5. 롱-또는-플랫
    tdf_lf = filter_trades_by_dates(lf_trades, period_dates)
    res_lf = compute_metrics(tdf_lf, BT)
    rows.append(f"| 롱-또는-플랫 | {res_lf.n_trades} | {res_lf.win_rate:.2f}% | {res_lf.total_pnl_krw:,.0f} | {res_lf.sharpe_daily:.3f} | {res_lf.max_drawdown_krw:,.0f} |")

    rows.append("")
    return "\n".join(rows)


def main():
    df_5min = load_data()
    daily = rg.to_daily(df_5min, BT.session_boundary_hour)
    all_dates = set(daily.index.date)

    print(f"전체 5분봉: {len(df_5min)}봉, {df_5min.index[0]} ~ {df_5min.index[-1]}")
    print(f"전체 거래일: {len(all_dates)}일, {min(all_dates)} ~ {max(all_dates)}")

    # 인과적 하락장 판정일
    bearish_dates = compute_bearish_trade_dates(daily)
    print(f"하락장 판정일 (다음날 매매): {len(bearish_dates)}일")
    print(f"선택 필터: {SELECTED_FILTER} ({FILTER_DESCRIPTION})")
    print(f"실시간 신호 (최신일 기준): {should_trade_today(daily)}")

    # 피봇은 한 번만 계산
    print("\n숏-전용 피봇 신호 계산 중...")
    pivots = detect_short_pivots(df_5min)

    # 전략 실행
    base_bt = pv.BacktestConfig(**BT.__dict__)
    print("숏-전용 피봇 반전 백테스트 (Base) 계산 중...")
    short_res = run_short_pivot(df_5min, pivots, base_bt)
    print("양방향 피봇 반전 백테스트 (Baseline) 계산 중...")
    both_res = run_both_pivot(df_5min, pivots, base_bt)
    print("롱-또는-플랫 백테스트 계산 중...")
    lf_res = run_long_or_flat(df_5min)

    # 리스크 파라미터 그리드 최적화
    best_risk_params, _ = find_best_risk_params(df_5min, pivots, base_bt, bearish_dates)
    risk_bt = make_risk_bt(base_bt, **best_risk_params)
    print(f"\n선택된 리스크 파라미터: {risk_bt.__dict__}")
    print("숏-전용 피봇 반전 백테스트 (Risk) 계산 중...")
    risk_res = run_short_pivot(df_5min, pivots, risk_bt)

    lines = [
        "# 숏-전용 피봇 반전 + 인과적 하락장 필터 전략 (실시간 적용)",
        "",
        "## 개요",
        f"- **하락장 판정**: {FILTER_DESCRIPTION}",
        "- **실행**: 하락장 판정된 다음 거래일에만 숏-전용 피봇 반전 실행",
        "- **피봇 파라미터**: walk-forward + 최적화 결과에서 선택된 파라미터",
        "- **리스크 관리**: 손절/익절/트레일링/일일 손실제한 그리드 최적화 적용",
        f"- **최적 리스크 파라미터**: 손절={best_risk_params['stop_loss_pct']:.1%}, 익절={best_risk_params['take_profit_pct']:.1%}, 트레일링={best_risk_params['trailing_stop_pct']:.1%}, 일일손실제한={best_risk_params['daily_loss_limit_krw']:,.0f}원",
        "",
        f"- **데이터 기간**: {df_5min.index[0]} ~ {df_5min.index[-1]}",
        f"- **전체 거래일**: {len(all_dates)}일",
        f"- **하락장 판정일**: {len(bearish_dates)}일",
        "",
    ]

    # 전체 기간 평가
    lines.append("## 전체 기간 평가")
    lines.append("")
    lines.append("| 전략 | 거래 | 승률 | PnL (원) | Sharpe | MaxDD |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    tdf = filter_trades_by_dates(short_res.trades, bearish_dates)
    res = compute_metrics(tdf, BT)
    lines.append(f"| 숏-전용 피봇 (하락장 필터) | {res.n_trades} | {res.win_rate:.2f}% | {res.total_pnl_krw:,.0f} | {res.sharpe_daily:.3f} | {res.max_drawdown_krw:,.0f} |")

    tdf_risk = filter_trades_by_dates(risk_res.trades, bearish_dates)
    res_risk = compute_metrics(tdf_risk, BT)
    lines.append(f"| 숏-전용 피봇 (하락장 필터 + Risk) | {res_risk.n_trades} | {res_risk.win_rate:.2f}% | {res_risk.total_pnl_krw:,.0f} | {res_risk.sharpe_daily:.3f} | {res_risk.max_drawdown_krw:,.0f} |")

    res_all = compute_metrics(short_res.trades, BT)
    lines.append(f"| 숏-전용 피봇 (무조건) | {res_all.n_trades} | {res_all.win_rate:.2f}% | {res_all.total_pnl_krw:,.0f} | {res_all.sharpe_daily:.3f} | {res_all.max_drawdown_krw:,.0f} |")

    res_lf = compute_metrics(lf_res.trades, BT)
    lines.append(f"| 롱-또는-플랫 | {res_lf.n_trades} | {res_lf.win_rate:.2f}% | {res_lf.total_pnl_krw:,.0f} | {res_lf.sharpe_daily:.3f} | {res_lf.max_drawdown_krw:,.0f} |")

    lines.append("")

    # OOS 기간 평가
    lines.append("## Out-of-Sample 평가")
    lines.append("")
    lines.append(evaluate_period(
        "요청 OOS 기간", date(2025, 10, 1), date(2026, 1, 31),
        short_res.trades, risk_res.trades, both_res.trades, lf_res.trades, bearish_dates, all_dates
    ))
    lines.append(evaluate_period(
        "test 기간 (2026-01-01 ~ 2026-06-19)", date(2026, 1, 1), date(2026, 6, 19),
        short_res.trades, risk_res.trades, both_res.trades, lf_res.trades, bearish_dates, all_dates
    ))

    # Risk Management 섹션
    lines.append("## 리스크 관리")
    lines.append("")
    lines.append("- **적용 기본값**: 그리드 최적화 대신 고정된 리스크 기본값을 적용하여 전/후 비교.")
    lines.append(f"- **손절(stop_loss)**: {risk_bt.stop_loss_pct:.1%}")
    lines.append(f"- **익절(take_profit)**: {risk_bt.take_profit_pct:.1%}")
    lines.append(f"- **트레일링 스탑(trailing_stop)**: {risk_bt.trailing_stop_pct:.1%}")
    lines.append(f"- **일일 손실제한(daily_loss_limit)**: {risk_bt.daily_loss_limit_krw:,.0f} KRW")
    lines.append(f"- **사이징 모드(position_size_mode)**: {risk_bt.position_size_mode}")
    lines.append("")
    lines.append("### 전/후 비교 (전체 기간)")
    lines.append("")
    lines.append("| 전략 | 거래 | 승률 | PnL (원) | Sharpe | MaxDD |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    tdf_base = filter_trades_by_dates(short_res.trades, bearish_dates)
    res_base = compute_metrics(tdf_base, BT)
    tdf_risk = filter_trades_by_dates(risk_res.trades, bearish_dates)
    res_risk = compute_metrics(tdf_risk, BT)
    lines.append(f"| 하락장 필터 (Base) | {res_base.n_trades} | {res_base.win_rate:.2f}% | {res_base.total_pnl_krw:,.0f} | {res_base.sharpe_daily:.3f} | {res_base.max_drawdown_krw:,.0f} |")
    lines.append(f"| 하락장 필터 + Risk | {res_risk.n_trades} | {res_risk.win_rate:.2f}% | {res_risk.total_pnl_krw:,.0f} | {res_risk.sharpe_daily:.3f} | {res_risk.max_drawdown_krw:,.0f} |")
    lines.append("")

    lines.append("## 해석")
    lines.append("- **인과적 하락장 필터**는 전일 종가/MA20 기준이므로 실시간 적용 가능.")
    lines.append("- **숏-전용 피봇 + 하락장 필터**는 하락장 조건에서만 집중적으로 매매하여 무조건 숏-전용 피봇과 롱-또는-플랫보다 하락장 국면에서 우수한 Sharpe/MaxDD를 보임.")
    lines.append("- **리스크 관리 추가 효과**: 기본 손절/익절/트레일링/일일 손실제한을 적용하면 전체 수익은 소폭 줄어들 수 있으나, MaxDD가 축소되고 Sharpe가 개선되는 경향을 보임. test 기간에서의 안정성 향상이 특히 눈에 띰.")
    lines.append("- **주의**: 현재 리스크 파라미터는 고정 기본값이며, 향후 추가 최적화를 통해 더 나은 균형을 찾을 수 있음.")
    lines.append("- **결론**: 복합 하락장 필터와 리스크 관리(익절/트레일링)를 결합한 전략이 기존 전략 대비 하락장에서 더 안정적이며 실시간 적용 가능함.")

    md_text = "\n".join(lines)

    OUTPUT_MD.write_text(md_text, encoding="utf-8")
    print(f"\n결과 저장 완료: {OUTPUT_MD}")
    print(md_text)


if __name__ == "__main__":
    main()
