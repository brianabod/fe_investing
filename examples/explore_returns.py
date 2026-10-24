"""Exploratory plots of the defense stocks: prices, return distributions, and
rolling mean/volatility. These show why returns are non-Gaussian and
non-stationary, which motivates modeling spreads instead of single stocks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make the investing package importable

import matplotlib.pyplot as plt

from investing import Portfolio

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "defense"
WINDOW_SIZE = 252  # one trading year


def explore(portfolio: Portfolio, window_size: int = WINDOW_SIZE):
    return [
        portfolio.plot_time_series(),
        portfolio.plot_distributions(),
        portfolio.plot_statistics_over_time(window_size=window_size),
        portfolio.compare_stocks(window_size=window_size),
    ]


if __name__ == "__main__":
    portfolio = Portfolio.from_directory(
        ["LMT", "NOC", "GD"],  # also available: "BA", "LDOS"
        DATA_DIRECTORY,
        companies={"LMT": "Lockheed Martin", "NOC": "Northrop Grumman", "GD": "General Dynamics",
                   "BA": "Boeing", "LDOS": "Leidos"},
        colors={"LMT": "forestgreen", "NOC": "slategrey", "GD": "darkblue",
                "BA": "tomato", "LDOS": "sandybrown"},
    )
    explore(portfolio)
    plt.show()
