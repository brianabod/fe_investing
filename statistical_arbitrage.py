import os
import numpy as np
import matplotlib.pyplot as plt

from copy import copy

from Stock import Stock
from Portfolio import Portfolio
from market import trade_triggers, trade

"""
Statistical Arbitrage is a strategy for trading on the cointegrated
relationship between stocks. Statistical Arbitrage involving only two stocks
is referred to as pairs trading.
"""


def statistical_arbitrage(portfolio: Portfolio, metric=Stock.LOG_RETURN):

    y1_stock: Stock = portfolio.stocks[0]
    y1 = y1_stock.time_series.df[Stock.CLOSE][::-1]

    y2_stock: Stock = portfolio.stocks[1]
    y2 = y2_stock.time_series.df[Stock.CLOSE][::-1]

    estimated_gamma = estimate_gamma(y1, y2)

    # Set thresholds to trade on the spread
    spread_relative_to_y1 = compute_spread(y1, y2, estimated_gamma)
    y1_buy_threshold = -1 * np.std(spread_relative_to_y1)
    spread_relative_to_y2 = compute_spread(y2, y1, estimated_gamma)
    y2_buy_threshold = -1 * np.std(spread_relative_to_y2)
    sell_threshold = 0

    # Define trade triggers
    y1_triggers = trade_triggers(spread_relative_to_y1, y1_buy_threshold, sell_threshold)
    y2_triggers = trade_triggers(spread_relative_to_y2, y2_buy_threshold, sell_threshold)

    # Simulate trading using these triggers
    balance_y1, returns_y1 = trade(y1, y1_triggers)
    balance_y2, returns_y2 = trade(y2, y2_triggers)

    # Combine balance and returns
    balance = [sum([b1, b2]) for b1, b2 in zip(balance_y1, balance_y2)]
    returns = []
    for r1, r2 in zip(returns_y1, returns_y2):
        if r1 == np.nan:
            returns.append(r2)
            continue
        if r2 == np.nan:
            returns.append(r1)
            continue
        returns.append(r1 + r2)

    # Add an intial timestep so that the balances start at zero
    t = list(range(len(y1)))
    tb = copy(t)
    tb.insert(0, -1)
    balance_y1 = list(balance_y1)
    balance_y1.insert(0, 0)
    balance_y2 = list(balance_y2)
    balance_y2.insert(0, 0)
    balance.insert(0, 0)

    print(f'ROI from y1: ${balance_y1[-1] - balance_y1[0]}')
    print(f'ROI from y2: ${balance_y2[-1] - balance_y2[0]}')
    print(f'Total ROI:   ${balance[-1] - balance[0]}')

    # Plot
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t, y1, color="blue", alpha=0.5, label='y1 price')
    ax1.plot(tb, balance_y1, color="blue", alpha=0.7, label='balance')
    ax1.plot(t, returns_y1, color="blue", linestyle='None', marker='x', alpha=0.6, label='return')
    ax1.grid(True)
    ax1.set_title("Trading y1")
    ax1.legend()

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(t, y2, color="red", alpha=0.5, label='y2 price')
    ax2.plot(tb, balance_y2, color="red", alpha=0.7, label='balance')
    ax2.plot(t, returns_y2, color="red", linestyle='None', marker='x', alpha=0.6, label='return')
    ax2.grid(True)
    ax2.set_title("Trading y2")
    ax2.legend()

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(t, y1, color="blue", alpha=0.5, label='y1 price')
    ax3.plot(t, y2, color="red", alpha=0.5, label='y2 price')
    ax3.plot(tb, balance, color="purple", alpha=0.7, label='balance')
    ax3.plot(t, returns, color="purple", linestyle='None', marker='x', alpha=0.6, label='return')
    ax3.grid(True)
    ax3.set_title("Pairs Trading")
    ax3.legend()

    return fig

def estimate_gamma(series_1, series_2):
    """Gamma is the differential amplitude of the the trend between the two
    series."""
    gamma = 1
    return gamma

def compute_spread(series_1, series_2, gamma=None):
    """Returns the point-wise spread between the two series. If the series are
    cointegrated, the spread will be stationary and mean-reverting."""
    if gamma == None:
        gamma = estimate_gamma(series_1, series_2)
    spread = series_1 - (series_2 * gamma)
    return spread

def set_optimal_trading_thresholds(spread):
    """Set thesholds to trade on."""
    buy1_threshold = np.std(spread) # buy series 1 when spread is high
    buy2_threshold = -1 * np.std(spread) # buy series 2 when spread is low
    sell_value = 0 # sell when the value returns to zero
    return buy1_threshold, buy2_threshold, sell_value


if __name__ == "__main__":
    portfolio = Portfolio(
        [
            Stock("Lockheed Martin", "LMT", os.path.join("defense", "LMT.csv"), "forestgreen"),
            Stock("Northrop Grumman", "NOC", os.path.join("defense", "NOC.csv"), "slategrey"),
            # Stock("General Dynamics", "GD", os.path.join("defense", "GD.csv"), "darkblue"),
            # Stock("Boeing", "BA", os.path.join("defense", "BA.csv"), 'tomato'),
            # Stock("Leidos", "LDOS", os.path.join("defense", "LDOS.csv"), 'sandybrown'),
            # Stock("Raytheon", "RTX", os.path.join("defense", "^RTX.csv"), 'indianred'),
        ], 
        max_values=1000,
    )

    fig1 = portfolio.plot_time_series()
    fig2 = portfolio.plot_distributions()
    fig3 = portfolio.plot_statistics_over_time(window_size=365) # Demonsrate volatility clustering, where the standard deviation tends to be "sticky" over time.
    fig4 = portfolio.compare_stocks()

    statistical_arbitrage(portfolio)

    plt.show()
