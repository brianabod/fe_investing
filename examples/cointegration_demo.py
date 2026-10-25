"""Synthetic demo of cointegration (Feng & Palomar, Sec. 10.1).

Two log-price series share one random-walk trend. Each one is non-stationary
(integrated of order 1), but their first differences are stationary, and so
is their spread: they are cointegrated. The stationarity and pair
cointegration tests confirm this.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make the investing package importable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from investing.cointegration import stationarity_test, pair_cointegration_test
from investing.market_simulation import generate_cointegrated_time_series
from investing.plotting import (
    INK_PRIMARY,
    INK_SECONDARY,
    add_figure_legend,
    apply_style,
    draw_zero_axis,
    series_color,
    set_figure_title,
)
from investing.time_series import function_of_time, mean, volatility, window_time_vector

if __name__ == "__main__":
    apply_style()

    # A pair of time series with an integration order of 1
    count = 1000
    time = list(range(count))
    first, second = generate_cointegrated_time_series(count, hedge_ratio=1.0)

    # Each series is a random walk, but the spread between them is stationary.
    pair = pd.DataFrame({"first": first, "second": second})
    for name in pair:
        test = stationarity_test(pair[name])
        print(f"ADF {name}: statistic={test.statistic:.2f}, p-value={test.p_value:.3f} "
              f"-> {'stationary' if test.is_stationary() else 'random walk'}")
    result = pair_cointegration_test(pair)
    print(f"Pair cointegration test: hedge ratio={result.hedge_ratios.iloc[0]:.3f}, "
          f"statistic={result.statistic:.2f}, p-value={result.p_value:.4f} "
          f"-> {'cointegrated' if result.is_cointegrated() else 'not cointegrated'}")

    # The first difference of each series is stationary: its rolling mean and
    # volatility stay flat over time.
    window = 100
    time_difference = time[1:]
    window_times = window_time_vector(time_difference, window)

    fig, (series_ax, difference_ax, spread_ax) = plt.subplots(
        3, 1, sharex=True, figsize=(10, 8.5), height_ratios=[1.2, 1, 1])
    for index, (name, values) in enumerate(pair.items()):
        color = series_color(index)
        differences = np.diff(values)
        means = np.array(function_of_time(differences, mean, window))
        volatilities = np.array(function_of_time(differences, volatility, window))

        series_ax.plot(time, values, color=color, linewidth=1.1, label=f"{name.capitalize()} series")
        difference_ax.plot(time_difference, differences, color=color, linewidth=0.5, alpha=0.35)
        difference_ax.fill_between(window_times, means - volatilities, means + volatilities,
                                   color=color, alpha=0.15, linewidth=0)
        difference_ax.plot(window_times, means, color=color, linewidth=1.4)

    series_ax.set_title("Two random walks that share a common trend (integrated of order 1)")
    difference_ax.set_title(f"First differences, with rolling mean ± 1 standard deviation ({window}-step window)")
    draw_zero_axis(difference_ax)

    spread_deviation = result.residuals.std()
    spread_ax.plot(time, result.residuals, color=INK_PRIMARY, linewidth=0.8, label="Spread")
    for level in (spread_deviation, -spread_deviation):
        spread_ax.axhline(level, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))
    spread_ax.plot([], [], color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)),
                   label="±1 standard deviation thresholds")
    draw_zero_axis(spread_ax)
    spread_ax.set_title(f"Spread: first − {result.hedge_ratios.iloc[0]:.2f} × second "
                        "(stationary and mean-reverting)")
    spread_ax.set_xlabel("Time step")

    add_figure_legend(fig)
    set_figure_title(fig, f"Cointegration: pair test p-value {result.p_value:.4f}")
    plt.show()
