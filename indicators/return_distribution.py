"""Return distribution estimation using Kernel Density Estimation (KDE).

Provides rolling-window PDF/CDF for log-returns to detect statistically
extreme price movements. Designed for mean-reversion signal generation in
futures markets (KP200, NQ, ES, etc.).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np


try:
    from scipy.integrate import quad
    from scipy.interpolate import interp1d
    from scipy.stats import gaussian_kde
except ImportError as _ie:  # pragma: no cover
    gaussian_kde = None  # type: ignore
    quad = None  # type: ignore
    interp1d = None  # type: ignore


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributionMetrics:
    """Container for KDE-based return distribution metrics."""

    pdf: float = 0.0
    cdf: float = 0.5
    z_score: float = 0.0
    left_tail_prob: float = 0.0
    right_tail_prob: float = 0.0
    median: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    is_ready: bool = False
    samples: int = 0


class ReturnDistributionEstimator:
    """Rolling-window KDE estimator for returns.

    Maintains a fixed-size buffer of recent log-returns and fits a Gaussian
    KDE. Evaluates the probability density, CDF, and tail probabilities of
    the latest return. For real-time performance a grid-based CDF lookup table
    is rebuilt whenever the distribution is updated.

    Args:
        window: Maximum number of recent returns to keep.
        min_samples: Minimum samples required before the estimator reports
            ``is_ready=True``.
        bandwidth: KDE bandwidth selection. ``"scott"`` or ``"silverman"``,
            or a positive float.
        decay: Optional exponential decay factor. ``None`` means uniform
            weighting. A float ``alpha`` (0 < alpha < 1) weights recent
            samples with ``(1-alpha) ** age``.
        grid_points: Number of grid points used for fast CDF lookup.
        outlier_clip: Number of standard deviations used to clip the support
            of the grid around the current sample mean.
    """

    def __init__(
        self,
        window: int = 3000,
        min_samples: int = 500,
        bandwidth: str | float = "scott",
        decay: Optional[float] = None,
        grid_points: int = 2048,
        outlier_clip: float = 5.0,
    ) -> None:
        if window < min_samples:
            raise ValueError(f"window({window}) must be >= min_samples({min_samples})")

        self.window = int(window)
        self.min_samples = int(min_samples)
        self.bandwidth = bandwidth
        self.decay = float(decay) if decay is not None else None
        if self.decay is not None and not (0.0 < self.decay < 1.0):
            raise ValueError("decay must be in (0, 1)")

        self.grid_points = max(512, int(grid_points))
        self.outlier_clip = float(outlier_clip)

        self._returns: Deque[float] = deque(maxlen=self.window)
        self._kde: Optional["gaussian_kde"] = None
        self._cdf_lookup: Optional["interp1d"] = None
        self._grid: Optional[np.ndarray] = None
        self._mean: float = 0.0
        self._std: float = 0.0
        self._median: float = 0.0
        self._last_metrics: Optional[DistributionMetrics] = None

    # ─────────────────────────────── public API ───────────────────────────────

    def update(self, return_value: float) -> DistributionMetrics:
        """Append a new return and refresh distribution / lookup tables.

        Args:
            return_value: The latest return (log-return preferred).

        Returns:
            DistributionMetrics for the supplied return.
        """
        self._returns.append(float(return_value))
        self._refresh()
        return self.evaluate(return_value)

    def evaluate(self, return_value: float) -> DistributionMetrics:
        """Evaluate metrics for an arbitrary return using the current KDE.

        If the estimator is not ready yet, returns neutral defaults with
        ``is_ready=False``.
        """
        if not self.is_ready or self._kde is None:
            return DistributionMetrics(
                pdf=0.0,
                cdf=0.5,
                z_score=0.0,
                left_tail_prob=0.0,
                right_tail_prob=0.0,
                median=self._median,
                mean=self._mean,
                std=self._std,
                is_ready=False,
                samples=len(self._returns),
            )

        rv = float(return_value)

        # Fast grid-based CDF
        if self._cdf_lookup is not None and self._grid is not None:
            try:
                cdf = float(self._cdf_lookup(rv))
                cdf = float(np.clip(cdf, 0.0, 1.0))
            except ValueError:
                # Extrapolate using the nearest edge value
                if rv <= self._grid[0]:
                    cdf = 0.0
                else:
                    cdf = 1.0
        else:
            cdf = 0.5

        pdf = float(self._kde(rv)[0])

        z_score = (rv - self._mean) / max(self._std, 1e-12)
        left_tail = cdf
        right_tail = 1.0 - cdf

        return DistributionMetrics(
            pdf=pdf,
            cdf=cdf,
            z_score=z_score,
            left_tail_prob=left_tail,
            right_tail_prob=right_tail,
            median=self._median,
            mean=self._mean,
            std=self._std,
            is_ready=True,
            samples=len(self._returns),
        )

    @property
    def is_ready(self) -> bool:
        return len(self._returns) >= self.min_samples

    def reset(self) -> None:
        """Clear all internal state."""
        self._returns.clear()
        self._kde = None
        self._cdf_lookup = None
        self._grid = None
        self._mean = 0.0
        self._std = 0.0
        self._median = 0.0
        self._last_metrics = None

    # ─────────────────────────────── internals ──────────────────────────────────

    def _refresh(self) -> None:
        n = len(self._returns)
        if n < 2:
            return

        arr = np.asarray(self._returns, dtype=float)
        self._mean = float(np.mean(arr))
        self._std = float(np.std(arr))

        if not self.is_ready:
            self._median = float(np.median(arr))
            return

        # Build weighted data if decay is requested.
        weights = self._build_weights(n)

        try:
            # scipy 1.17: gaussian_kde supports weights.
            if weights is not None:
                self._kde = gaussian_kde(arr, bw_method=self.bandwidth, weights=weights)
            else:
                self._kde = gaussian_kde(arr, bw_method=self.bandwidth)
        except Exception as exc:  # pragma: no cover
            _logger.warning("KDE fit failed: %s", exc)
            self._kde = None
            self._cdf_lookup = None
            self._grid = None
            return

        # Determine support for grid-based CDF.
        std = max(self._std, 1e-9)
        grid_min = self._mean - self.outlier_clip * std
        grid_max = self._mean + self.outlier_clip * std

        # Ensure the latest sample is inside the grid.
        if self._returns:
            latest = float(self._returns[-1])
            grid_min = min(grid_min, latest - std)
            grid_max = max(grid_max, latest + std)

        self._grid = np.linspace(grid_min, grid_max, self.grid_points)
        pdf_grid = self._kde(self._grid)
        cdf_grid = np.cumsum(pdf_grid) * (self._grid[1] - self._grid[0])
        cdf_grid = np.clip(cdf_grid / max(cdf_grid[-1], 1e-12), 0.0, 1.0)

        self._cdf_lookup = interp1d(
            self._grid,
            cdf_grid,
            kind="linear",
            bounds_error=False,
            fill_value=(0.0, 1.0),
        )

        # Median via binary search on CDF lookup.
        try:
            self._median = float(
                self._find_root_monotonic(self._grid, cdf_grid - 0.5)
            )
        except Exception:  # pragma: no cover
            self._median = float(np.median(arr))

    def _build_weights(self, n: int) -> Optional[np.ndarray]:
        if self.decay is None:
            return None
        ages = np.arange(n - 1, -1, -1, dtype=float)
        weights = (1.0 - self.decay) ** ages
        weights /= weights.sum()
        return weights

    @staticmethod
    def _find_root_monotonic(grid: np.ndarray, values: np.ndarray) -> float:
        """Linear interpolation root finding for monotonic ``values`` crossing 0."""
        idx = int(np.searchsorted(values, 0.0))
        if idx <= 0:
            return float(grid[0])
        if idx >= len(grid):
            return float(grid[-1])
        x0, x1 = grid[idx - 1], grid[idx]
        y0, y1 = values[idx - 1], values[idx]
        if abs(y1 - y0) < 1e-15:
            return float(x0)
        return float(x0 - y0 * (x1 - x0) / (y1 - y0))

    # ─────────────────────────────── legacy helpers ───────────────────────────

    def cdf_quad(self, x: float) -> float:
        """Accurate CDF using numerical integration (slower, for verification)."""
        if self._kde is None or not self.is_ready:
            return 0.5
        if quad is None:  # pragma: no cover
            return float(self.evaluate(x).cdf)

        dataset_min = float(np.min(self._kde.dataset))
        dataset_max = float(np.max(self._kde.dataset))
        factor = float(self._kde.factor)
        lower = dataset_min - 5.0 * factor
        try:
            result, _ = quad(lambda y: float(self._kde(y)[0]), lower, x, limit=100)
            return float(np.clip(result, 0.0, 1.0))
        except Exception as exc:  # pragma: no cover
            _logger.debug("quad CDF failed: %s", exc)
            return float(self.evaluate(x).cdf)


# ─────────────────────────────── dataframe helpers ─────────────────────────────


def add_return_features(
    df: "pd.DataFrame",
    *,
    close_col: str = "close",
    timeframes: Tuple[int, ...] = (1, 5),
    prefix: str = "ret",
) -> "pd.DataFrame":
    """Add log-return columns for multiple timeframes to a DataFrame.

    Args:
        df: Price DataFrame with a close column.
        close_col: Name of the close price column (case-insensitive fallback).
        timeframes: Tuple of bar lengths, e.g. (1, 5) for 1-min and 5-min.
        prefix: Column prefix.

    Returns:
        DataFrame with added ``{prefix}_log_{tf}m`` columns.
    """
    import pandas as pd

    cc = _resolve_col(df, close_col)
    closes = pd.to_numeric(df[cc], errors="coerce")

    out = df.copy()
    for tf in timeframes:
        col_name = f"{prefix}_log_{tf}m"
        out[col_name] = np.log(closes / closes.shift(tf))
    return out


def build_kde_features(
    df: "pd.DataFrame",
    return_col: str,
    *,
    window: int = 3000,
    min_samples: int = 500,
    bandwidth: str | float = "scott",
    decay: Optional[float] = None,
    grid_points: int = 2048,
    outlier_clip: float = 5.0,
    prefix: Optional[str] = None,
    refit_every: int = 1,
) -> "pd.DataFrame":
    """Build rolling KDE metrics for a single return column.

    This is intended for offline backtesting. Each row uses the prior
    ``window`` returns (not including the current row) to fit the KDE and then
    evaluates the current row's return.

    Args:
        df: DataFrame containing ``return_col``.
        return_col: Return column to model.
        window/min_samples/bandwidth/decay/grid_points/outlier_clip:
            Passed to ``ReturnDistributionEstimator``.
        prefix: Output column prefix. Defaults to ``{return_col}_kde``.
        refit_every: Refit the KDE every N rows. ``1`` means exact refit at
            every row (slow), larger values trade some staleness for speed.

    Returns:
        DataFrame with added KDE metric columns.
    """
    import pandas as pd

    prefix = prefix or f"{return_col}_kde"
    est = ReturnDistributionEstimator(
        window=window,
        min_samples=min_samples,
        bandwidth=bandwidth,
        decay=decay,
        grid_points=grid_points,
        outlier_clip=outlier_clip,
    )

    returns = pd.to_numeric(df[return_col], errors="coerce").to_numpy(dtype=float)
    rows: List[dict] = []
    refit_every = max(1, int(refit_every))

    for idx, ret in enumerate(returns):
        if math.isnan(ret):
            rows.append({
                f"{prefix}_pdf": np.nan,
                f"{prefix}_cdf": np.nan,
                f"{prefix}_zscore": np.nan,
                f"{prefix}_left_tail": np.nan,
                f"{prefix}_right_tail": np.nan,
                f"{prefix}_median": np.nan,
                f"{prefix}_mean": np.nan,
                f"{prefix}_std": np.nan,
                f"{prefix}_ready": False,
            })
            continue

        # Refit the distribution either every row or every refit_every rows.
        if refit_every == 1 or (idx % refit_every) == 0:
            metrics = est.update(float(ret))
        else:
            est._returns.append(float(ret))
            metrics = est.evaluate(float(ret))

        rows.append({
            f"{prefix}_pdf": metrics.pdf,
            f"{prefix}_cdf": metrics.cdf,
            f"{prefix}_zscore": metrics.z_score,
            f"{prefix}_left_tail": metrics.left_tail_prob,
            f"{prefix}_right_tail": metrics.right_tail_prob,
            f"{prefix}_median": metrics.median,
            f"{prefix}_mean": metrics.mean,
            f"{prefix}_std": metrics.std,
            f"{prefix}_ready": metrics.is_ready,
        })

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _resolve_col(df: "pd.DataFrame", base: str) -> str:
    """Resolve column name case-insensitively."""
    lower_map = {c.lower(): c for c in df.columns}
    base_lower = base.lower()
    if base_lower in lower_map:
        return lower_map[base_lower]
    raise KeyError(f"Column '{base}' not found in DataFrame")
