import unittest

import numpy as np

from investing.mean_reversion import KalmanSpreadFilter, fit_ornstein_uhlenbeck
from synthetic_data import autoregressive


class TestMeanReversion(unittest.TestCase):
    def test_half_life(self):
        self.assertAlmostEqual(fit_ornstein_uhlenbeck(autoregressive(5000, 0.5, seed=4)).half_life, 1.0, delta=0.1)

    def test_kalman_filter_removes_noise(self):
        hidden = autoregressive(3000, 0.9, scale=0.01, seed=5)
        observed = hidden + 0.02 * np.random.default_rng(6).normal(size=len(hidden))
        filtered = KalmanSpreadFilter.fit(observed).filter(observed)
        self.assertLess(np.mean((filtered - hidden) ** 2), np.mean((observed - hidden) ** 2))


if __name__ == "__main__":
    unittest.main()
