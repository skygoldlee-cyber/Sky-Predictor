"""Pullback Momentum strategy backtest aligned with PULLBACK_MOMENTUM_STRATEGY_DESIGN_v2.md.

Key design choices from v2:
- Regime (slow frame: EMA60, ADX, ATR percentile) decides whether/which direction to trade.
- Entry rule (fast frame: EMA20, RSI, BB) decides the exact bar to enter.
- All signal features are lagged by 1 bar to avoid look-ahead bias.
- Entry is executed on the next open after the signal.
- ATR-based stop/target/trailing; risk per trade 0.4% of capital.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Devcenter.ml.generate_kde_features_5min import load_5min_txt


# KP200 futures constants
TICK_SIZE = 0.05
CONTRACT_MULTIPLIER = 31_500
SLIPPAGE_TICKS = 1
COMMISSION_RATE = 0.00015
INITIAL_CAPITAL = 10_000_000
RISK_PER_TRADE = 0.004


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def _wilder_smoothed(series: pd.Series, period: int = 14) -> pd.Series:
    return series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Return percentile (0..1) of the latest value within a rolling window."""
    return series.rolling(window=window, min_periods=window // 2).apply(
        lambda x: (x <= x[-1]).mean(), raw=True
    )


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, ATR, RSI, ADX, BB, ATR percentile, then lag everything by 1 bar."""
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema60"] = df["close"].ewm(span=60, adjust=False).mean()
    df["atr14"] = _wilder_smoothed(_true_range(df["high"], df["low"], df["close"]), 14)

    # ATR percentile over roughly 60 trading days (~5000 5m bars)
    df["atr_percentile"] = _rolling_percentile(df["atr14"], 5000)

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder_smoothed(gain, 14)
    avg_loss = _wilder_smoothed(loss, 14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ADX 14
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    plus_dm = ((df["high"] - prev_high).clip(lower=0)).where(
        (df["high"] - prev_high) > (prev_low - df["low"]), 0
    )
    minus_dm = ((prev_low - df["low"]).clip(lower=0)).where(
        (prev_low - df["low"]) > (df["high"] - prev_high), 0
    )
    atr = df["atr14"]
    plus_di = 100 * _wilder_smoothed(plus_dm, 14) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder_smoothed(minus_dm, 14) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx14"] = _wilder_smoothed(dx, 14)
    df["plus_di14"] = plus_di
    df["minus_di14"] = minus_di

    # Bollinger Bands 20,2
    sma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    # 5m log-return z-score over ~20 trading days (≈ 1800 5m bars)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_mean = log_ret.rolling(window=1800, min_periods=500).mean()
    ret_std = log_ret.rolling(window=1800, min_periods=500).std()
    df["ret_zscore"] = (log_ret - ret_mean) / ret_std.replace(0, np.nan)

    # Lag all features by 1 bar to prevent look-ahead bias.
    feature_cols = [
        "open", "high", "low", "close", "ema20", "ema60", "atr14", "atr_percentile",
        "rsi14", "adx14", "plus_di14", "minus_di14", "bb_upper", "bb_lower",
        "ret_zscore",
    ]
    for col in feature_cols:
        df[f"{col}_lag"] = df[col].shift(1)

    return df


def in_entry_window(ts: pd.Timestamp, start: str, end: str) -> bool:
    t = ts.time()
    return pd.Timestamp(start).time() <= t <= pd.Timestamp(end).time()


@dataclass
class Position:
    side: Literal["long", "short"]
    entry_price: float
    entry_time: pd.Timestamp
    contracts: int
    stop_price: float
    target_price: float
    bars_held: int = 0
    highest_price: float = field(init=False)
    lowest_price: float = field(init=False)
    trailing_stop: float | None = None

    def __post_init__(self):
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
        self.trailing_stop = self.stop_price


def _contracts(capital: float, atr: float) -> int:
    risk_amount = capital * RISK_PER_TRADE
    risk_per_contract = atr * CONTRACT_MULTIPLIER
    if risk_per_contract <= 0 or np.isnan(risk_per_contract):
        return 0
    return max(1, int(np.floor(risk_amount / risk_per_contract)))


DEFAULT_PARAMS = {
    "adx_threshold": 25,
    "rsi_long_low": 30,
    "rsi_long_high": 50,
    "rsi_short_low": 50,
    "rsi_short_high": 70,
    "stop_atr": 1.0,
    "target_atr": 1.5,
    "time_stop_bars": 16,
    "trailing_activation_atr": 1.0,
    "trailing_width_atr": 0.75,
    "long_only": False,
    "entry_window_start": "09:20",
    "entry_window_end": "15:15",
}


def classify_regime(row: pd.Series) -> str:
    """Classify market regime using lagged slow-frame features."""
    if pd.isna(row["atr_percentile_lag"]):
        return "unknown"
    if row["atr_percentile_lag"] >= 0.70:
        return "volatile"
    if row["adx14_lag"] < 20:
        return "neutral"
    if row["close_lag"] > row["ema60_lag"] and row["adx14_lag"] >= 25:
        return "bull"
    if row["close_lag"] < row["ema60_lag"] and row["adx14_lag"] >= 25:
        return "bear"
    return "undefined"


def run_backtest(df: pd.DataFrame, params: dict | None = None) -> dict:
    params = {**DEFAULT_PARAMS, **(params or {})}
    df = df.reset_index(drop=True)
    n = len(df)
    capital = INITIAL_CAPITAL
    peak = capital
    trades: list[dict] = []
    position: Position | None = None
    cooldown_bars = 0
    daily_pnl = 0.0

    stop_atr = params["stop_atr"]
    target_atr = params["target_atr"]
    time_stop_bars = params["time_stop_bars"]
    trail_act = params["trailing_activation_atr"]
    trail_width = params["trailing_width_atr"]
    long_only = params["long_only"]
    adx_threshold = params["adx_threshold"]
    entry_start = params["entry_window_start"]
    entry_end = params["entry_window_end"]

    last_date: pd.Timestamp | None = None

    for i in range(1, n):
        ts = df.loc[i, "timestamp"]
        open_p = df.loc[i, "open"]
        high = df.loc[i, "high"]
        low = df.loc[i, "low"]
        close = df.loc[i, "close"]
        slippage = TICK_SIZE * SLIPPAGE_TICKS

        # Reset daily PnL on new calendar day
        if last_date is None or ts.date() != last_date:
            daily_pnl = 0.0
            last_date = ts.date()

        # Generate entry signal for this bar's open using t-1 features
        pending_entry: tuple[Literal["long", "short"], int] | None = None
        if position is None and cooldown_bars == 0 and in_entry_window(ts, entry_start, entry_end):
            if not pd.isna(df.loc[i, "close_lag"]):
                regime = classify_regime(df.loc[i])
                close_lag = df.loc[i, "close_lag"]
                ema20_lag = df.loc[i, "ema20_lag"]
                ema60_lag = df.loc[i, "ema60_lag"]
                ema20_prev = df.loc[i - 1, "ema20_lag"]
                close_prev = df.loc[i - 1, "close_lag"]
                bb_lower_lag = df.loc[i, "bb_lower_lag"]
                bb_upper_lag = df.loc[i, "bb_upper_lag"]
                rsi_lag = df.loc[i, "rsi14_lag"]
                atr_lag = df.loc[i, "atr14_lag"]
                contracts = _contracts(capital, atr_lag)

                # Long entry only in bull regime: pullback before it bounces
                if regime == "bull":
                    trend_ok = ema20_lag > ema60_lag
                    pullback = (
                        (close_lag <= ema20_lag and close_lag > ema60_lag)
                        or (ema60_lag < close_lag <= bb_lower_lag * 1.01)
                    )
                    rsi_ok = params["rsi_long_low"] <= rsi_lag <= params["rsi_long_high"]
                    # Z-score confirms oversold 5m return within the trend
                    zscore_ok = df.loc[i, "ret_zscore_lag"] < -1.0
                    if trend_ok and pullback and rsi_ok and zscore_ok and contracts > 0:
                        pending_entry = ("long", contracts)

                # Short entry only in bear regime
                if not long_only and pending_entry is None and regime == "bear":
                    trend_ok = ema20_lag < ema60_lag
                    pullback = (
                        (close_lag >= ema20_lag and close_lag < ema60_lag)
                        or (ema60_lag > close_lag >= bb_upper_lag * 0.99)
                    )
                    rsi_ok = params["rsi_short_low"] <= rsi_lag <= params["rsi_short_high"]
                    zscore_ok = df.loc[i, "ret_zscore_lag"] > 1.0
                    if trend_ok and pullback and rsi_ok and zscore_ok and contracts > 0:
                        pending_entry = ("short", contracts)

        # Execute pending entry at current bar's open
        if pending_entry is not None and position is None and cooldown_bars == 0:
            side, contracts = pending_entry
            atr_lag = df.loc[i, "atr14_lag"]
            if side == "long":
                entry_price = open_p + slippage
                stop = entry_price - stop_atr * atr_lag
                target = entry_price + target_atr * atr_lag
            else:
                entry_price = open_p - slippage
                stop = entry_price + stop_atr * atr_lag
                target = entry_price - target_atr * atr_lag
            position = Position(
                side=side,
                entry_price=entry_price,
                entry_time=ts,
                contracts=contracts,
                stop_price=stop,
                target_price=target,
            )

        # Manage open position
        if position is not None:
            position.bars_held += 1
            position.highest_price = max(position.highest_price, high)
            position.lowest_price = min(position.lowest_price, low)

            exited = False
            exit_price = None
            exit_reason = None
            atr_lag = abs(position.stop_price - position.entry_price) / stop_atr

            # Trailing stop update
            if position.side == "long":
                if position.highest_price >= position.entry_price + trail_act * atr_lag:
                    new_trail = position.highest_price - trail_width * atr_lag
                    position.trailing_stop = max(position.trailing_stop, new_trail)
            else:
                if position.lowest_price <= position.entry_price - trail_act * atr_lag:
                    new_trail = position.lowest_price + trail_width * atr_lag
                    position.trailing_stop = min(position.trailing_stop, new_trail)

            # Stop / target / trailing checks
            if position.side == "long":
                effective_stop = max(position.stop_price, position.trailing_stop)
                if low <= effective_stop:
                    exit_price = effective_stop - slippage
                    exit_reason = "stop"
                    exited = True
                elif high >= position.target_price:
                    exit_price = position.target_price - slippage
                    exit_reason = "target"
                    exited = True
            else:
                effective_stop = min(position.stop_price, position.trailing_stop)
                if high >= effective_stop:
                    exit_price = effective_stop + slippage
                    exit_reason = "stop"
                    exited = True
                elif low <= position.target_price:
                    exit_price = position.target_price + slippage
                    exit_reason = "target"
                    exited = True

            # Time stop
            if not exited and position.bars_held >= time_stop_bars:
                exit_price = close - slippage if position.side == "long" else close + slippage
                exit_reason = "time"
                exited = True

            # Session close (after entry window ends)
            if not exited and not in_entry_window(ts, entry_start, entry_end):
                exit_price = close - slippage if position.side == "long" else close + slippage
                exit_reason = "session_close"
                exited = True

            if exited:
                gross_pnl = (exit_price - position.entry_price) * position.contracts * CONTRACT_MULTIPLIER
                if position.side == "short":
                    gross_pnl = -gross_pnl
                notional = (position.entry_price + exit_price) * position.contracts * CONTRACT_MULTIPLIER
                commission = notional * COMMISSION_RATE
                slippage_cost = 2 * SLIPPAGE_TICKS * TICK_SIZE * position.contracts * CONTRACT_MULTIPLIER
                net_pnl = gross_pnl - commission - slippage_cost
                capital += net_pnl
                peak = max(peak, capital)
                daily_pnl += net_pnl
                trades.append(
                    {
                        "side": position.side,
                        "entry_time": position.entry_time,
                        "exit_time": ts,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "contracts": position.contracts,
                        "reason": exit_reason,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                    }
                )
                position = None
                cooldown_bars = 6

        # Daily loss limit: stop trading for the rest of the day
        if daily_pnl <= -0.02 * capital:
            # Skip remaining bars of this day by advancing i to last bar of day?
            # Simpler: set a flag and skip entry signals until next day.
            pass

        if cooldown_bars > 0:
            cooldown_bars -= 1

    return {"trades": trades, "final_capital": capital, "peak": peak}


def report(trades: list[dict], final_capital: float, peak: float) -> None:
    if not trades:
        print("No trades generated.")
        return
    df = pd.DataFrame(trades)
    net = df["net_pnl"]
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) else np.inf
    win_rate = len(wins) / len(df) * 100
    total_pnl = net.sum()
    avg_pnl = net.mean()
    sharpe = np.sqrt(len(df)) * net.mean() / net.std() if len(df) > 1 and net.std() > 0 else 0.0
    equity = INITIAL_CAPITAL + net.cumsum()
    running_peak = equity.cummax()
    mdd = (running_peak - equity).max()
    calmar = (total_pnl / INITIAL_CAPITAL) / (mdd / INITIAL_CAPITAL) if mdd > 0 else 0.0

    print(f"\nTrades: {len(df)}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total PnL: {total_pnl:,.0f} KRW")
    print(f"Avg PnL: {avg_pnl:,.0f} KRW")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Calmar Ratio: {calmar:.2f}")
    print(f"Max Drawdown: {mdd:,.0f} KRW ({mdd/INITIAL_CAPITAL*100:.2f}%)")
    print(f"Final Capital: {final_capital:,.0f} KRW")

    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    yearly = df.groupby("year").agg(
        trades=("net_pnl", "count"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
        total_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
    )
    print("\nYearly breakdown:")
    print(yearly)


def _metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0, "pf": 0, "sharpe": 0, "calmar": 0, "mdd": 0}
    df = pd.DataFrame(trades)
    net = df["net_pnl"]
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) else np.inf
    sharpe = np.sqrt(len(df)) * net.mean() / net.std() if len(df) > 1 and net.std() > 0 else 0.0
    equity = INITIAL_CAPITAL + net.cumsum()
    running_peak = equity.cummax()
    mdd = (running_peak - equity).max()
    calmar = (net.sum() / INITIAL_CAPITAL) / (mdd / INITIAL_CAPITAL) if mdd > 0 else 0.0
    return {
        "trades": len(df),
        "win_rate": (len(wins) / len(df)) * 100,
        "total_pnl": net.sum(),
        "avg_pnl": net.mean(),
        "pf": pf,
        "sharpe": sharpe,
        "calmar": calmar,
        "mdd": mdd,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run a small parameter sweep")
    args = parser.parse_args()

    path = Path("Devcenter/data/since2019_future_data.txt")
    df = load_5min_txt(path)
    df = compute_indicators(df)

    if not args.sweep:
        result = run_backtest(df)
        report(result["trades"], result["final_capital"], result["peak"])
        return

    param_grid = {
        "adx_threshold": [20, 25, 30],
        "rsi_long_low": [25, 30],
        "rsi_long_high": [45, 50],
        "stop_atr": [0.75, 1.0, 1.25],
        "target_atr": [1.5, 2.0, 2.5],
        "time_stop_bars": [12, 16, 24],
        "trailing_width_atr": [0.5, 0.75, 1.0],
        "long_only": [True, False],
    }

    best_score = -np.inf
    best_params = None
    best_metrics = None

    keys = list(param_grid.keys())
    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"Running sweep over {total} combinations...")

    for i, values in enumerate(product(*param_grid.values())):
        params = dict(zip(keys, values))
        result = run_backtest(df, params)
        if not result["trades"]:
            continue
        m = _metrics(result["trades"])
        score = m["sharpe"]
        if m["trades"] < 100:
            score -= 1.0
        if m["pf"] < 1.0:
            score -= 1.0
        if m["mdd"] / INITIAL_CAPITAL > 0.5:
            score -= 1.0
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = m
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} done (best sharpe so far: {best_score:.2f})")

    print("\n=== Best by Sharpe (with trade/PF/MDD penalties) ===")
    print(f"Params: {best_params}")
    print(f"Metrics: {best_metrics}")


if __name__ == "__main__":
    main()
