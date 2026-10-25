"""Exploratory plots: prices, return distributions, and rolling mean and
volatility for the stocks in the data folder. They show why returns are
non-Gaussian and non-stationary, which motivates modeling spreads instead of
single stocks.

Usage:  python examples/explore_returns.py [--data-directory DIR] [--symbols A B C]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make the investing package importable

import matplotlib.pyplot as plt

from command_line import load_portfolio, parse_arguments
from investing import Portfolio
from investing.plotting import apply_style

WINDOW_SIZE = 252  # one trading year


def explore(portfolio: Portfolio, window_size: int = WINDOW_SIZE):
    return [
        portfolio.plot_time_series(),
        portfolio.plot_distributions(),
        portfolio.plot_statistics_over_time(window_size=window_size),
        portfolio.compare_stocks(window_size=window_size),
    ]


if __name__ == "__main__":
    arguments = parse_arguments(__doc__)
    apply_style()
    explore(load_portfolio(arguments))
    plt.show()
