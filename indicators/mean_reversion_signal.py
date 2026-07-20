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
        lower_tail_threshold: BUY tail-probability below which the signal is
            STRONG or NORMAL (default 0.02 = bottom 2%).
        upper_tail_threshold: CDF threshold for the SELL side.  The derived
            tail-probability boundary is ``1 - upper_tail_threshold``
            (default 0.98 -> 0.02 = top 2%).
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
        normal_tail_prob: WEAK-entry tail probability threshold.  Signals with
            tail probability between ``lower_tail_threshold`` (BUY) or
            ``1 - upper_tail_threshold`` (SELL) and this value are classified
            as WEAK (default 0.05).
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
        normal_tail_prob: float = 0.05,
    ) -> None:
        # Validate thresholds so signal strengths are well-ordered:
        # STRONG < NORMAL < WEAK.  For SELL, the NORMAL boundary is derived
        # from upper_tail_threshold as 1 - upper_tail_threshold.
        right_normal_threshold = 1.0 - float(upper_tail_threshold)
        if not (0.0 < strong_tail_prob < lower_tail_threshold < normal_tail_prob < 1.0):
            raise ValueError(
                "BUY thresholds must satisfy 0 < strong < normal-entry < weak-entry < 1"
            )
        if not (0.0 < strong_tail_prob < right_normal_threshold < normal_tail_prob < 1.0):
            raise ValueError(
                "SELL thresholds must satisfy 0 < strong < (1-upper) < weak-entry < 1"
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
        self._right_normal_threshold = right_normal_threshold

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
        # Cooldown requires a monotonic bar counter. If the caller does not
        # provide one, auto-increment so the first signal does not permanently
        # lock out all later signals.
        if bar_index is None:
            self._current_bar += 1
        else:
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
        #
        # normal_tail_prob is the WEAK-entry threshold.  lower_tail_threshold
        # (BUY) and 1 - upper_tail_threshold (SELL) are the NORMAL-entry
        # thresholds.  strong_tail_prob is the STRONG threshold.
        tail_action: Optional[str] = None
        tail_prob = 0.0
        if metrics.left_tail_prob < self.normal_tail_prob:
            tail_action = "BUY"
            tail_prob = metrics.left_tail_prob
        if metrics.right_tail_prob < self.normal_tail_prob:
            # Prefer the more extreme tail if both sides trigger.
            if tail_action is None or metrics.right_tail_prob < tail_prob:
                tail_action = "SELL"
                tail_prob = metrics.right_tail_prob
        if tail_action is None:
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
        # BUY-the-dip is only safe in an uptrend; SELL-the-rally only in a
        # downtrend.  The previous logic was inverted.
        if supertrend_direction is not None:
            if tail_action == "BUY" and supertrend_direction == -1:
                suppressions.append("st_downtrend_no_long_mean_rev")
            if tail_action == "SELL" and supertrend_direction == 1:
                suppressions.append("st_uptrend_no_short_mean_rev")

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
        # Tail probability is always the *small* tail prob (< normal_tail_prob),
        # so the upper branches (tail_prob > 1 - strong/normal) were dead code.
        if tail_prob < self.strong_tail_prob:
            strength = "STRONG"
        elif (tail_action == "BUY" and tail_prob < self.lower_tail_threshold) or (
            tail_action == "SELL" and tail_prob < self._right_normal_threshold
        ):
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
