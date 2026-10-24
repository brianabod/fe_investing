"""Synthetic demo of cointegration (Feng & Palomar, Sec. 10.1).

Two log-price series share one random-walk trend. Each one is non-stationary
(integrated of order 1), but their first differences are stationary, and so
is their spread: they are cointegrated. The Engle-Granger and ADF tests
confirm this.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make the investing package importable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from investing.cointegration import augmented_dickey_fuller, engle_granger
from investing.market_simulation import generate_cointegrated_time_series
from investing.time_series import function_of_time, mean, volatility, window_time_vector

if __name__ == "__main__":
    # A pair of time series with an integration order of 1
    count = 1000
    time = list(range(count))
    first, second = generate_cointegrated_time_series(count, hedge_ratio=1.0)

    # Each series is a random walk, but the spread between them is stationary.
    pair = pd.DataFrame({"first": first, "second": second})
    for name in pair:
        test = augmented_dickey_fuller(pair[name])
        print(f"ADF {name}: statistic={test.statistic:.2f}, p-value={test.p_value:.3f} "
              f"-> {'stationary' if test.is_stationary() else 'random walk'}")
    result = engle_granger(pair)
    print(f"Engle-Granger: hedge ratio={result.hedge_ratios.iloc[0]:.3f}, "
          f"statistic={result.statistic:.2f}, p-value={result.p_value:.4f} "
          f"-> {'cointegrated' if result.is_cointegrated() else 'not cointegrated'}")

    # Show that the first difference of each time series is stationary
    time_difference = time[:-1]
    first_difference = np.diff(first)
    second_difference = np.diff(second)

    window = 100
    window_times = window_time_vector(time_difference, window)
    first_means = function_of_time(first_difference, mean, window)
    first_volatilities = function_of_time(first_difference, volatility, window)
    second_means = function_of_time(second_difference, mean, window)
    second_volatilities = function_of_time(second_difference, volatility, window)

    fig = plt.figure()
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(time, first, color="blue", alpha=0.5)
    ax1.plot(time, second, color="red", alpha=0.5)
    ax1.grid(True)
    ax1.set_title("I(1) Time Series Pair")

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(time_difference, first_difference, color="blue", alpha=0.2)
    ax2.plot(time_difference, second_difference, color="red", alpha=0.2)
    ax2.plot(window_times, first_means, color="blue", alpha=0.7)
    ax2.plot(window_times, first_volatilities, color="blue", alpha=0.7)
    ax2.plot(window_times, second_means, color="red", alpha=0.7)
    ax2.plot(window_times, second_volatilities, color="red", alpha=0.7)
    ax2.grid(True)
    ax2.set_title(f"First Difference Mean and Volatility (Window Size: {window})")

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(time_difference, first_difference, color="blue", alpha=0.2)
    ax3.plot(time_difference, second_difference, color="red", alpha=0.2)
    for means, volatilities, color in [(first_means, first_volatilities, "blue"),
                                       (second_means, second_volatilities, "red")]:
        upper = [value + spread for value, spread in zip(means, volatilities)]
        lower = [value - spread for value, spread in zip(means, volatilities)]
        ax3.plot(window_times, upper, color=color, alpha=0.7)
        ax3.plot(window_times, means, color=color, alpha=0.7)
        ax3.plot(window_times, lower, color=color, alpha=0.7)
    ax3.grid(True)
    ax3.set_title(f"First Differences, Mean +/- StDev (Window Size: {window})")

    ax4 = fig.add_subplot(4, 1, 4)
    spread_deviation = result.residuals.std()
    ax4.plot(time, result.residuals, color="black", alpha=0.7)
    ax4.axhline(spread_deviation, color="green")   # sell the spread above
    ax4.axhline(-spread_deviation, color="green")  # buy the spread below
    ax4.axhline(0, color="red")                    # unwind at the mean
    ax4.grid(True)
    ax4.set_title("Engle-Granger spread (stationary and mean-reverting), +/- 1 standard deviation thresholds")

    plt.show()
