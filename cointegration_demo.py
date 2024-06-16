import numpy as np
import matplotlib.pyplot as plt

from market import generate_cointegrated_time_series
from nonstationary import function_of_time, mean, volatility, window_time_vector
from statistical_arbitrage import compute_spread, set_optimal_trading_thresholds


def cointegration_demo():
    num_samples = 1000
    gamma = 1
    y1, y2 = generate_cointegrated_time_series(num_samples, gamma)

    # Compute the spread
    estimated_gamma = gamma # assume we estimated gamma accurately
    spread = compute_spread(y1, y2, estimated_gamma)

    buy1_threshold, buy2_threshold, sell_value = set_optimal_trading_thresholds(spread)

    # Simulate trade on the spread
    balance = 1000.00
    owned = None
    print('LEDGER')
    print('Balance\tSpread\tAction')
    print(f'${balance}')
    for i, (value, y1_price, y2_price) in enumerate(zip(spread, y1, y2)):
        if owned == 'y1':
            if value >= sell_value:
                price_sold = y1_price
                return_value = price_sold - price_bought
                balance += price_sold # sell y1
                print(f'${balance:.2f}\t\t{value:.1f}\tSold {owned} at ${price_sold:.2f} for ${return_value:.2f}')
                owned = None
        if owned == 'y2':
            if value <= sell_value:
                price_sold = y2_price
                return_value = price_sold - price_bought
                balance += price_sold # sell y2
                print(f'${balance:.2f}\t\t{value:.1f}\tSold {owned} at ${price_sold:.2f} for ${return_value:.2f}')
                owned = None
        if owned == None:
            if value < buy1_threshold:
                price_bought = y1_price
                balance -= price_bought # buy y1
                owned = 'y1'
                print(f'${balance:.2f}\t\t{value:.1f}\tBought {owned} at ${price_bought:.2f} when the spread was {value:.1f}')
            if value > buy2_threshold:
                price_bought = y2_price
                balance -= price_bought # buy y2
                owned = 'y2'
                print(f'${balance:.2f}\t\t{value:.1f}\tBought {owned} for ${price_bought:.2f}')

    # Show that the first difference of each time series is stationary
    t = list(range(len(y1)))
    t_diff = t[:-1]
    y1_d1 = np.diff(y1)
    y2_d1 = np.diff(y2)

    window = 100
    t_window = window_time_vector(t_diff, window)
    y1_means = function_of_time(y1_d1, mean, window)
    y1_volatilities = function_of_time(y1_d1, volatility, window)
    y2_means = function_of_time(y2_d1, mean, window)
    y2_volatilities = function_of_time(y2_d1, volatility, window)

    y1_upper = [m + v for m, v in zip(y1_means, y1_volatilities)]
    y1_lower = [m - v for m, v in zip(y1_means, y1_volatilities)]
    y2_upper = [m + v for m, v in zip(y2_means, y2_volatilities)]
    y2_lower = [m - v for m, v in zip(y2_means, y2_volatilities)]

    # Plot the first differences
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t, y1, color="blue", alpha=0.5)
    ax1.plot(t, y2, color="red", alpha=0.5)
    ax1.grid(True)
    ax1.set_title("I(1) Time Series Pair")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(t_diff, y1_d1, color="blue", alpha=0.2)
    ax2.plot(t_diff, y2_d1, color="red", alpha=0.2)
    ax2.plot(t_window, y1_means, color="blue", alpha=0.7)
    ax2.plot(t_window, y1_volatilities, color="blue", alpha=0.7)
    ax2.plot(t_window, y2_means, color="red", alpha=0.7)
    ax2.plot(t_window, y2_volatilities, color="red", alpha=0.7)
    ax2.grid(True)
    ax2.set_title(f"First Difference Mean and Volatility (Window Size: {window})")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(t_diff, y1_d1, color="blue", alpha=0.2)
    ax3.plot(t_diff, y2_d1, color="red", alpha=0.2)
    ax3.plot(t_window, y1_upper, color="blue", alpha=0.7)
    ax3.plot(t_window, y1_means, color="blue", alpha=0.7)
    ax3.plot(t_window, y1_lower, color="blue", alpha=0.7)
    ax3.plot(t_window, y2_upper, color="red", alpha=0.7)
    ax3.plot(t_window, y2_means, color="red", alpha=0.7)
    ax3.plot(t_window, y2_lower, color="red", alpha=0.7)
    ax3.grid(True)
    ax3.set_title(f"First Differences, Mean +/- StDev (Window Size: {window})")

    # Plot the spread and the trading strategy
    fig2 = plt.figure()
    ax21 = fig2.add_subplot(1, 1, 1)
    ax21.plot(t, spread, color="black", alpha=0.9)
    ax21.hlines([buy2_threshold, buy1_threshold], min(t), max(t), color="green")
    ax21.hlines([sell_value], min(t), max(t), color="red")
    ax21.grid(True)
    ax21.set_title(f"Spread (Stationary and Mean-Reverting)\n(Window Size: {window})")


if __name__ == "__main__":
    cointegration_demo()
    plt.show()

