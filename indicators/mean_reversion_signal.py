"""Mean-reversion signal generator based on return distribution tails.

Combines KDE return tails, ZigZag structure, VWAP deviation, ATR and ADX
to produce filtered BUY/SELL/HOLD signals. Designed to complement existing
AdaptiveSuperTrend / AdaptiveZigZag infrastructure rather than replace it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


try:
    from .return_distribution import DistributionMetrics
except ImportError:  # pragma: no cover
    from return_distribution import DistributionMetrics


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeanReversionSignal:
    """Output of mean-reversion signal generation."""

    action: str = "HOLD"
    strength: str = "NONE"
    reason: str = ""
    metrics: Optional[DistributionMetrics] = None
    suppressions: tuple = ()


class MeanReversionSignalGenerator:
    """Generate mean-reversion signals from return-tail metrics.

    The generator is intentionally decoupled from specific state classes:
    it accepts dict-like inputs so it can be wired to AdaptiveZigZag,
    AdaptiveSuperTrend, or simple dictionaries.

    Args:
        lower_tail_threshold: CDF tail below which a LONG candidate is
            considered (default 0.02 = bottom 2%).
        upper_tail_threshold: CDF tail above which a SHORT candidate is
            considered (default 0.98 = top 2%).
        adx_max_for_mean_reversion: ADX below this value allows mean-reversion
            trades. Above it, the signal is suppressed.
        vwap_dev_threshold: Minimum absolute VWAP deviation as a fraction of
            price (e.g. 0.0005 = 0.05%).
        atr_increase_lookback: Bars to look back for ATR increase check.
        cooldown_bars: Minimum bars between same-direction signals.
        require_zigzag_pivot: If True, require a recent ZigZag swing low/high.
        require_vwap_cross: If True, price must be on the mean-reverting side
            of VWAP.
        strong_tail_prob: Tail probability threshold for STRONG signals.
        normal_tail_prob: Tail probability threshold for NORMAL signals.
    """

    def __init__(
        self,
        lower_tail_threshold: float = 0.02,
        upper_tail_threshold: float = 0.98,
        adx_max_for_mean_reversion: float = 25.0,
        vwap_dev_threshold: float = 0.0005,
        atr_increase_lookback: int = 5,
        cooldown_bars: int = 3,
        require_zigzag_pivot: bool = True,
        require_vwap_cross: bool = True,
        strong_tail_prob: float = 0.01,
        normal_tail_prob: float = 0.02,
    ) -> None:
        if not (0.0 < lower_tail_threshold < upper_tail_threshold < 1.0):
            raise ValueError(
                "Thresholds must satisfy 0 < lower < upper < 1"
            )

        self.lower_tail_threshold = float(lower_tail_threshold)
        self.upper_tail_threshold = float(upper_tail_threshold)
        self.adx_max_for_mean_reversion = float(adx_max_for_mean_reversion)
        self.vwap_dev_threshold = float(vwap_dev_threshold)
        self.atr_increase_lookback = max(1, int(atr_increase_lookback))
        self.cooldown_bars = max(0, int(cooldown_bars))
        self.require_zigzag_pivot = bool(require_zigzag_pivot)
        self.require_vwap_cross = bool(require_vwap_cross)
        self.strong_tail_prob = float(strong_tail_prob)
        self.normal_tail_prob = float(normal_tail_prob)

        self._last_signal_bar: Dict[str, int] = {"BUY": -self.cooldown_bars, "SELL": -self.cooldown_bars}
        self._current_bar: int = 0

    # ─────────────────────────────── public API ───────────────────────────────

    def generate(
        self,
        metrics: Optional[DistributionMetrics],
        current_price: float,
        vwap: float,
        atr_current: float,
        atr_history: Optional[Any] = None,
        adx: float = 0.0,
        supertrend_direction: Optional[int] = None,
        zigzag_state: Optional[Dict[str, Any]] = None,
        bar_index: Optional[int] = None,
    ) -> MeanReversionSignal:
        """Generate a mean-reversion signal.

        Args:
            metrics: DistributionMetrics from ReturnDistributionEstimator.
            current_price: Latest price.
            vwap: Latest VWAP.
            atr_current: Latest ATR value.
            atr_history: Optional sequence of prior ATR values.
            adx: Latest ADX value.
            supertrend_direction: 1 for uptrend, -1 for downtrend.
            zigzag_state: Optional dict with keys such as ``current_direction``,
                ``last_swing_high``, ``last_swing_low``.
            bar_index: Optional monotonic bar index for cooldown tracking.

        Returns:
            MeanReversionSignal with action, strength, reason, suppressions.
        """
        if bar_index is not None:
            self._current_bar = int(bar_index)

        if metrics is None or not metrics.is_ready:
            return MeanReversionSignal(
                action="HOLD",
                strength="NONE",
                reason="kde_not_ready",
                metrics=metrics,
            )

        suppressions: list[str] = []

        # 1. Tail detection
        # left_tail_prob  = P(R <= current) — small values mean extreme low tail
        # right_tail_prob = P(R >= current) — small values mean extreme high tail
        tail_action: Optional[str] = None
        tail_prob = 0.0
        if metrics.left_tail_prob < self.lower_tail_threshold:
            tail_action = "BUY"
            tail_prob = metrics.left_tail_prob
        elif metrics.right_tail_prob < self.lower_tail_threshold:
            tail_action = "SELL"
            tail_prob = metrics.right_tail_prob
        else:
            return MeanReversionSignal(
                action="HOLD",
                strength="NONE",
                reason="tail_not_extreme",
                metrics=metrics,
            )

        # 2. ADX trend filter
        if adx > self.adx_max_for_mean_reversion:
            suppressions.append("adx_too_high")

        # 3. SuperTrend direction filter (do not fight the trend)
        if supertrend_direction is not None:
            if tail_action == "BUY" and supertrend_direction == -1:
                pass  # price below ST band; still allowed
            if tail_action == "SELL" and supertrend_direction == 1:
                pass  # price above ST band; still allowed
            # More aggressive filter: if ST direction aligns with the tail,
            # it is likely a trend continuation, not mean reversion.
            if tail_action == "BUY" and supertrend_direction == 1:
                suppressions.append("st_uptrend_no_long_mean_rev")
            if tail_action == "SELL" and supertrend_direction == -1:
                suppressions.append("st_downtrend_no_short_mean_rev")

        # 4. VWAP side filter
        vwap_dev = (current_price - vwap) / current_price if current_price > 0 and vwap > 0 else 0.0
        if self.require_vwap_cross and current_price > 0 and vwap > 0:
            if tail_action == "BUY" and vwap_dev > -self.vwap_dev_threshold:
                suppressions.append("not_below_vwap")
            if tail_action == "SELL" and vwap_dev < self.vwap_dev_threshold:
                suppressions.append("not_above_vwap")

        # 5. ZigZag pivot filter
        if self.require_zigzag_pivot and zigzag_state is not None:
            zz_dir = zigzag_state.get("current_direction")
            if tail_action == "BUY" and zz_dir != -1:
                suppressions.append("no_zigzag_low")
            if tail_action == "SELL" and zz_dir != 1:
                suppressions.append("no_zigzag_high")

        # 6. ATR increase / availability
        if atr_history is not None and len(atr_history) >= self.atr_increase_lookback:
            hist = np.asarray(atr_history[-self.atr_increase_lookback :], dtype=float)
            if len(hist) > 1 and not np.isnan(hist).all():
                if atr_current < float(np.nanmean(hist)) * 0.9:
                    suppressions.append("atr_contracting")

        # 7. Cooldown
        if self.cooldown_bars > 0 and tail_action in self._last_signal_bar:
            if (self._current_bar - self._last_signal_bar[tail_action]) < self.cooldown_bars:
                suppressions.append("cooldown")

        if suppressions:
            return MeanReversionSignal(
                action="HOLD",
                strength="NONE",
                reason=f"suppressed:{'|'.join(suppressions)}",
                metrics=metrics,
                suppressions=tuple(suppressions),
            )

        # 8. Strength assignment
        if tail_prob < self.strong_tail_prob or tail_prob > (1.0 - self.strong_tail_prob):
            strength = "STRONG"
        elif tail_prob < self.normal_tail_prob or tail_prob > (1.0 - self.normal_tail_prob):
            strength = "NORMAL"
        else:
            strength = "WEAK"

        reason = (
            f"mean_reversion_{tail_action}:tail={tail_prob:.4f},"
            f"adx={adx:.1f},vwap_dev={vwap_dev:.4f},st_dir={supertrend_direction}"
        )

        self._last_signal_bar[tail_action] = self._current_bar
        return MeanReversionSignal(
            action=tail_action,
            strength=strength,
            reason=reason,
            metrics=metrics,
        )

    def reset(self) -> None:
        """Clear cooldown state."""
        self._last_signal_bar = {"BUY": -self.cooldown_bars, "SELL": -self.cooldown_bars}
        self._current_bar = 0
