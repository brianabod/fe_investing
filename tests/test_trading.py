import unittest

import numpy as np
import pandas as pd

from investing import PairsBuilder, compute_positions, performance_summary, walk_forward
from investing.threshold import parametric_threshold, smooth_frequency
from synthetic_data import common_trend_pair


class TestThreshold(unittest.TestCase):
    def test_gaussian_optimum(self):
        # argmax threshold * (1 - Phi(threshold)) = 0.75 (Example 10.5)
        spread = np.random.default_rng(6).normal(size=200_000)
        self.assertAlmostEqual(parametric_threshold(spread).best_threshold, 0.75, delta=0.06)

    def test_no_smoothing_is_identity(self):
        frequency = np.array([0.5, 0.3, 0.35, 0.1])
        np.testing.assert_allclose(smooth_frequency(frequency, 0.0), frequency)

    def test_positions(self):
        spread = np.array([0, 1.5, 2.1, 1.0, -0.1, -2.5, -1.0, 0.2, 5.0, 1.0])
        positions = compute_positions(spread, entry_threshold=2.0, stop_loss=4.0)
        np.testing.assert_array_equal(positions, [0, 0, -1, -1, 0, 1, 1, 0, 0, 0])


class TestBacktest(unittest.TestCase):
    def test_no_lookahead(self):
        prices = np.exp(common_trend_pair(count=900) + 4)
        builder = PairsBuilder(max_spreads=1)
        base = walk_forward(prices, builder, fitting_days=250, trading_days=100).returns
        shocked = prices.copy()
        shocked.iloc[700:] *= 1.5  # change the future only
        after = walk_forward(shocked, builder, fitting_days=250, trading_days=100).returns
        pd.testing.assert_series_equal(base.iloc[:449], after.iloc[:449])

    def test_profitable_on_cointegrated_pair(self):
        prices = np.exp(common_trend_pair(count=1500) + 4)
        result = walk_forward(prices, PairsBuilder(max_spreads=1), cost_basis_points=0)
        self.assertGreater(performance_summary(result.returns)["sharpe_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
