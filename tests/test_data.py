import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from investing import Portfolio, Stock
from investing.time_series import function_of_time, mean, window_time_vector
from synthetic_data import write_price_csv


class TestStockAndPortfolio(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        dates = pd.bdate_range("2024-01-01", periods=10)
        # Written newest first to check that loading sorts oldest first.
        write_price_csv(os.path.join(self.directory.name, "AAA.csv"), dates[::-1], np.arange(10, 0, -1.0) + 100)
        # BBB is missing two dates that AAA has.
        write_price_csv(os.path.join(self.directory.name, "BBB.csv"), dates.delete([3, 7]), np.arange(8) + 50.0)

    def tearDown(self):
        self.directory.cleanup()

    def test_returns_are_close_to_close(self):
        stock = Stock("AAA", os.path.join(self.directory.name, "AAA.csv"))
        prices = stock.prices
        self.assertTrue(prices.index.is_monotonic_increasing)
        expected = prices.iloc[-1] / prices.iloc[-2] - 1
        self.assertAlmostEqual(stock.time_series.df[Stock.RETURN].iloc[-1], expected)

    def test_rolling_statistics_labeled_with_newest_date(self):
        stock = Stock("AAA", os.path.join(self.directory.name, "AAA.csv"))
        rolling = stock.mean_over_time(Stock.RETURN, 3).df
        self.assertEqual(rolling[Stock.DATE].iloc[-1], stock.prices.index[-1])
        self.assertAlmostEqual(rolling[Stock.RETURN].iloc[-1], stock.time_series.df[Stock.RETURN].iloc[-3:].mean())

    def test_portfolio_aligns_by_date(self):
        portfolio = Portfolio.from_directory(self.directory.name, ["AAA", "BBB"])
        prices = portfolio.prices()
        self.assertEqual(len(prices), 8)
        self.assertFalse(prices.isna().any().any())
        returns = portfolio.metrics[Stock.RETURN].df
        self.assertFalse(returns[Stock.DATE].isna().any())


    def test_discovers_files_and_skips_sparse_ones(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        write_price_csv(os.path.join(self.directory.name, "CCC.csv"), dates[:3], [1.0, 2.0, 3.0])
        with self.assertWarns(UserWarning):
            portfolio = Portfolio.from_directory(self.directory.name, minimum_days=5)
        self.assertEqual(portfolio.symbols, ["AAA", "BBB"])


class TestTimeFunctions(unittest.TestCase):
    def test_function_of_time(self):
        outputs, times = function_of_time([1, 2, 3, 4], mean, 2, ["a", "b", "c", "d"])
        self.assertEqual(outputs, [1.5, 2.5, 3.5])
        self.assertEqual(times, ["b", "c", "d"])
        self.assertEqual(window_time_vector([1, 2, 3], 3), [3])


if __name__ == "__main__":
    unittest.main()
