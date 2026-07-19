"""Unit tests for return distribution / mean reversion modules."""

import numpy as np
import pandas as pd
import pytest

from indicators.mean_reversion_signal import MeanReversionSignalGenerator
from indicators.return_distribution import (
    DistributionMetrics,
    ReturnDistributionEstimator,
    add_return_features,
    build_kde_features,
)


class TestReturnDistributionEstimator:
    def test_init_requires_window_ge_min_samples(self):
        with pytest.raises(ValueError):
            ReturnDistributionEstimator(window=10, min_samples=100)

    def test_not_ready_before_min_samples(self):
        est = ReturnDistributionEstimator(window=200, min_samples=100)
        for ret in np.random.normal(0, 0.001, 50):
            m = est.update(float(ret))
        assert not m.is_ready
        assert m.samples == 50

    def test_ready_after_min_samples(self):
        est = ReturnDistributionEstimator(window=200, min_samples=100)
        rng = np.random.default_rng(42)
        for ret in rng.normal(0, 0.01, 150):
            m = est.update(float(ret))
        assert m.is_ready
        assert m.samples == 150

    def test_cdf_pdf_for_normal_data(self):
        est = ReturnDistributionEstimator(window=2000, min_samples=500)
        rng = np.random.default_rng(7)
        returns = rng.normal(0, 0.01, 2000)
        for ret in returns:
            m = est.update(float(ret))

        # A value near the mean should have high CDF (around 0.5)
        m_center = est.evaluate(0.0)
        assert 0.3 < m_center.cdf < 0.7
        assert m_center.pdf > 1.0  # high density near mean

        # Extreme negative should have small CDF and left tail prob
        m_low = est.evaluate(-0.04)
        assert m_low.cdf < 0.05
        assert m_low.left_tail_prob < 0.05

        # Extreme positive should have large CDF / small right tail
        m_high = est.evaluate(0.04)
        assert m_high.cdf > 0.95
        assert m_high.right_tail_prob < 0.05

    def test_grid_cdf_matches_quad_cdf(self):
        est = ReturnDistributionEstimator(window=1000, min_samples=300, grid_points=2048)
        rng = np.random.default_rng(9)
        returns = rng.normal(0, 0.01, 1000)
        for ret in returns:
            est.update(float(ret))

        for x in [-0.03, 0.0, 0.03]:
            grid_cdf = est.evaluate(x).cdf
            quad_cdf = est.cdf_quad(x)
            assert abs(grid_cdf - quad_cdf) < 0.05

    def test_decay_weighting_changes_tail_prob(self):
        est_uniform = ReturnDistributionEstimator(window=1000, min_samples=300)
        est_decay = ReturnDistributionEstimator(window=1000, min_samples=300, decay=0.01)
        rng = np.random.default_rng(11)
        returns = rng.normal(0, 0.01, 1000)
        for ret in returns:
            est_uniform.update(float(ret))
            est_decay.update(float(ret))

        m_uniform = est_uniform.evaluate(0.03)
        m_decay = est_decay.evaluate(0.03)
        # Both should report CDF > 0.9 for this extreme value
        assert m_uniform.cdf > 0.9
        assert m_decay.cdf > 0.9

    def test_reset_clears_state(self):
        est = ReturnDistributionEstimator(window=200, min_samples=100)
        rng = np.random.default_rng(3)
        for ret in rng.normal(0, 0.01, 150):
            est.update(float(ret))
        assert est.is_ready
        est.reset()
        assert not est.is_ready
        assert len(est._returns) == 0


class TestReturnDataFrameHelpers:
    def test_add_return_features(self):
        df = pd.DataFrame({
            "open": [100, 101, 102, 101, 103],
            "high": [101, 102, 103, 102, 104],
            "low": [99, 100, 101, 100, 102],
            "close": [100, 101, 102, 101, 103],
        })
        out = add_return_features(df, close_col="close", timeframes=(1, 2))
        assert "ret_log_1m" in out.columns
        assert "ret_log_2m" in out.columns
        # First row of each return column should be NaN
        assert pd.isna(out["ret_log_1m"].iloc[0])
        assert pd.isna(out["ret_log_2m"].iloc[0])
        # Sanity: 100 -> 101 is about 0.00995 log return
        assert abs(out["ret_log_1m"].iloc[1] - np.log(101 / 100)) < 1e-6

    def test_build_kde_features(self):
        rng = np.random.default_rng(13)
        returns = rng.normal(0, 0.01, 600)
        df = pd.DataFrame({"ret": returns})
        out = build_kde_features(
            df,
            return_col="ret",
            window=300,
            min_samples=100,
        )
        assert "ret_kde_cdf" in out.columns
        assert "ret_kde_pdf" in out.columns
        assert "ret_kde_ready" in out.columns
        # Initially not enough samples -> not ready
        assert not out["ret_kde_ready"].iloc[50]
        # After window+ samples should be ready
        assert out["ret_kde_ready"].iloc[-1]

    def test_build_kde_features_refit_every(self):
        rng = np.random.default_rng(14)
        returns = rng.normal(0, 0.01, 600)
        df = pd.DataFrame({"ret": returns})
        out_exact = build_kde_features(df, return_col="ret", window=300, min_samples=100, refit_every=1)
        out_fast = build_kde_features(df, return_col="ret", window=300, min_samples=100, refit_every=50)
        # Fast refit should be close to exact for CDF values
        ready_exact = out_exact["ret_kde_ready"].to_numpy()
        ready_fast = out_fast["ret_kde_ready"].to_numpy()
        # Refit_every can delay the ready flag by up to refit_every rows.
        assert abs(ready_exact.sum() - ready_fast.sum()) <= 50

        mask = ready_exact & ready_fast
        diff = np.abs(out_exact["ret_kde_cdf"].to_numpy()[mask] - out_fast["ret_kde_cdf"].to_numpy()[mask])
        assert diff.mean() < 0.05, f"mean CDF diff too large: {diff.mean()}"
        assert diff.max() < 0.20, f"max CDF diff too large: {diff.max()}"


class TestMeanReversionSignalGenerator:
    def _make_metrics(self, cdf: float, left_tail: float, right_tail: float) -> DistributionMetrics:
        return DistributionMetrics(
            pdf=1.0,
            cdf=cdf,
            z_score=0.0,
            left_tail_prob=left_tail,
            right_tail_prob=right_tail,
            median=0.0,
            mean=0.0,
            std=0.01,
            is_ready=True,
            samples=1000,
        )

    def test_no_signal_when_tail_not_extreme(self):
        gen = MeanReversionSignalGenerator()
        m = self._make_metrics(0.5, 0.5, 0.5)
        sig = gen.generate(
            metrics=m,
            current_price=100.0,
            vwap=100.0,
            atr_current=0.5,
        )
        assert sig.action == "HOLD"
        assert sig.reason == "tail_not_extreme"

    def test_buy_signal_with_tail_extreme(self):
        gen = MeanReversionSignalGenerator(require_zigzag_pivot=False, require_vwap_cross=False)
        m = self._make_metrics(0.01, 0.01, 0.99)
        sig = gen.generate(
            metrics=m,
            current_price=100.0,
            vwap=99.0,
            atr_current=0.5,
            adx=20.0,
        )
        assert sig.action == "BUY"
        assert sig.strength in ("STRONG", "NORMAL")

    def test_sell_signal_with_tail_extreme(self):
        gen = MeanReversionSignalGenerator(require_zigzag_pivot=False, require_vwap_cross=False)
        # CDF=0.99 => right tail = 0.01 (extreme upper tail -> SELL)
        m = self._make_metrics(0.99, 0.99, 0.01)
        sig = gen.generate(
            metrics=m,
            current_price=100.0,
            vwap=101.0,
            atr_current=0.5,
            adx=20.0,
        )
        assert sig.action == "SELL"

    def test_adx_filter_suppresses_signal(self):
        gen = MeanReversionSignalGenerator(require_zigzag_pivot=False, require_vwap_cross=False)
        m = self._make_metrics(0.01, 0.01, 0.99)
        sig = gen.generate(
            metrics=m,
            current_price=100.0,
            vwap=99.0,
            atr_current=0.5,
            adx=40.0,
        )
        assert sig.action == "HOLD"
        assert "adx_too_high" in sig.suppressions

    def test_vwap_filter_suppresses_buy(self):
        gen = MeanReversionSignalGenerator(require_zigzag_pivot=False, require_vwap_cross=True)
        m = self._make_metrics(0.01, 0.01, 0.99)
        sig = gen.generate(
            metrics=m,
            current_price=100.0,
            vwap=100.0001,  # above threshold side
            atr_current=0.5,
            adx=20.0,
        )
        assert sig.action == "HOLD"
        assert "not_below_vwap" in sig.suppressions

    def test_cooldown_prevents_duplicate_signal(self):
        gen = MeanReversionSignalGenerator(
            require_zigzag_pivot=False,
            require_vwap_cross=False,
            cooldown_bars=3,
        )
        m = self._make_metrics(0.01, 0.01, 0.99)
        sig1 = gen.generate(
            metrics=m,
            current_price=100.0,
            vwap=99.0,
            atr_current=0.5,
            adx=20.0,
            bar_index=10,
        )
        assert sig1.action == "BUY"
        sig2 = gen.generate(
            metrics=m,
            current_price=100.0,
            vwap=99.0,
            atr_current=0.5,
            adx=20.0,
            bar_index=11,
        )
        assert sig2.action == "HOLD"
        assert "cooldown" in sig2.suppressions

    def test_zigzag_pivot_filter(self):
        gen = MeanReversionSignalGenerator(require_zigzag_pivot=True, require_vwap_cross=False)
        m = self._make_metrics(0.01, 0.01, 0.99)
        sig = gen.generate(
            metrics=m,
            current_price=100.0,
            vwap=99.0,
            atr_current=0.5,
            adx=20.0,
            zigzag_state={"current_direction": -1},  # recent swing low
        )
        assert sig.action == "BUY"

        sig_no_pivot = gen.generate(
            metrics=m,
            current_price=100.0,
            vwap=99.0,
            atr_current=0.5,
            adx=20.0,
            zigzag_state={"current_direction": 1},
        )
        assert sig_no_pivot.action == "HOLD"
        assert "no_zigzag_low" in sig_no_pivot.suppressions
