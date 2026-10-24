import unittest

import numpy as np
import pandas as pd

from investing.cointegration import augmented_dickey_fuller, engle_granger, johansen
from synthetic_data import autoregressive, common_trend_pair


class TestCointegration(unittest.TestCase):
    def test_dickey_fuller_random_walk_versus_autoregressive(self):
        random_walk = np.cumsum(np.random.default_rng(1).normal(size=1000))
        self.assertFalse(augmented_dickey_fuller(random_walk).is_stationary())
        self.assertTrue(augmented_dickey_fuller(autoregressive(1000, 0.5, seed=1)).is_stationary())

    def test_engle_granger_finds_hedge_ratio(self):
        result = engle_granger(common_trend_pair(hedge_ratio=0.8), "A")
        self.assertTrue(result.is_cointegrated())
        self.assertAlmostEqual(result.hedge_ratios["B"], 0.8, delta=0.05)

    def test_engle_granger_independent_random_walks(self):
        walks = np.cumsum(np.random.default_rng(2).normal(size=(1000, 2)), axis=0)
        self.assertGreater(engle_granger(pd.DataFrame(walks, columns=["A", "B"])).p_value, 0.05)

    def test_johansen_rank(self):
        log_prices = common_trend_pair()
        log_prices["C"] = log_prices["B"] * 0.5 + np.random.default_rng(3).normal(0, 0.005, len(log_prices))
        self.assertEqual(johansen(log_prices).rank(), 2)  # three series, one common trend


if __name__ == "__main__":
    unittest.main()
