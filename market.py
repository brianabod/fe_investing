import warnings
import numpy as np

"""This module implements a market simulation."""


def generate_cointegrated_time_series(num_samples, gamma=1):
    """Returns two time series that are cointegrated with an integration order
    of 1 and trend differential amplitude gamma."""
    common_trend = np.cumsum(np.random.normal(0, 1, num_samples))
    series_1 = common_trend + np.random.normal(0, 1, num_samples) # add stationary noise
    series_2 = common_trend * gamma + np.random.normal(0, 1, num_samples) # add stationary noise
    return series_1, series_2


def trade_triggers(series, buy_threshold, sell_threshold):
    """Returns a vector of triggers the same size as series that contains a 1
    to indicate a buy and a -1 to indicate a sell, and is 0 otherwise."""
    triggers = []
    owned = False
    for value in series:
        if not owned and value <= buy_threshold:
            triggers.append(1)
            owned = True
            continue
        if owned and value >= sell_threshold:
            triggers.append(-1)
            owned = False
            continue
        triggers.append(0)
    return np.array(triggers)


def trade(price_series, trade_triggers):
    """Returns a series of the cumulative account balance incurred under the
    trading strategy defined by trade_triggers."""
    owned = False # for error checking
    balance_deltas = []
    returns = []
    for price, trigger in zip(price_series, trade_triggers):
        if trigger < 0:
            if owned == False:
                warnings.warn('Attempted to sell an asset that was not owned.')
                balance_deltas.append(0)
                returns.append(np.nan)
                continue
            # Sell
            owned = False
            balance_deltas.append(price)
            returns.append(price - buy_price)
            continue
        if trigger > 0:
            if owned == True:
                warnings.warn('Attempted to buy an asset that was already owned.')
                balance_deltas.append(0)
                returns.append(0)
                continue
            # Buy
            owned = True
            buy_price = price
            balance_deltas.append(-1 * price)
            returns.append(np.nan)
            continue
        balance_deltas.append(0)
        returns.append(np.nan)
    balances = np.cumsum(balance_deltas)
    return balances, np.array(returns)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from cointegration_demo import compute_spread

    # Create a pair of time series with an integration order of 1
    num_days = 4*365
    gamma = 1
    y1, y2 = generate_cointegrated_time_series(4*365, gamma)

    # Set thresholds to trade on the spread
    estimated_gamma = gamma # assume we estimated gamma accurately
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
    returns = [sum([r1, r2]) for r1, r2 in zip(returns_y1, returns_y2)]

    # Plot
    t = list(range(num_days))

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t, y1, color="blue", alpha=0.5, label='y1 price')
    ax1.plot(t, balance_y1, color="blue", alpha=0.7, label='balance')
    ax1.grid(True)
    ax1.set_title("Trading y1")
    ax1.legend()

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(t, y2, color="red", alpha=0.5, label='y2 price')
    ax2.plot(t, balance_y2, color="red", alpha=0.7, label='balance')
    ax2.grid(True)
    ax2.set_title("Trading y2")
    ax2.legend()

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(t, y1, color="blue", alpha=0.5, label='y1 price')
    ax3.plot(t, y2, color="red", alpha=0.5, label='y2 price')
    ax3.plot(t, balance, color="purple", alpha=0.7, label='balance')
    ax3.grid(True)
    ax3.set_title("Pairs Trading")
    ax3.legend()

    plt.show()
