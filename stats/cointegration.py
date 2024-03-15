import numpy as np

from nonstationary import function_of_time, mean, volatility, window_time_vector

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create an pair of time series with an integration order of 1
    N = 1000
    t = list(range(N))
    trend = np.cumsum(np.random.normal(0, 1, len(t)))
    x1 = trend + np.random.normal(0, 1, len(trend))
    x2 = trend + np.random.normal(0, 1, len(trend))

    # Show that the first difference of each time series is stationary
    t_diff = t[:-1]
    x1_d1 = np.diff(x1)
    x2_d1 = np.diff(x2)

    window = 100
    t_window = window_time_vector(t_diff, window)
    x1_means = function_of_time(x1_d1, mean, window)
    x1_volatilities = function_of_time(x1_d1, volatility, window)
    x2_means = function_of_time(x2_d1, mean, window)
    x2_volatilities = function_of_time(x2_d1, volatility, window)

    x1_upper = [m + v for m, v in zip(x1_means, x1_volatilities)]
    x1_lower = [m - v for m, v in zip(x1_means, x1_volatilities)]
    x2_upper = [m + v for m, v in zip(x2_means, x2_volatilities)]
    x2_lower = [m - v for m, v in zip(x2_means, x2_volatilities)]

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t, x1, color="blue", alpha=0.5)
    ax1.plot(t, x2, color="red", alpha=0.5)
    ax1.grid(True)
    ax1.set_title("I(1) Time Series Pair")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(t_diff, x1_d1, color="blue", alpha=0.2)
    ax2.plot(t_diff, x2_d1, color="red", alpha=0.2)
    ax2.plot(t_window, x1_means, color="blue", alpha=0.7)
    ax2.plot(t_window, x1_volatilities, color="blue", alpha=0.7)
    ax2.plot(t_window, x2_means, color="red", alpha=0.7)
    ax2.plot(t_window, x2_volatilities, color="red", alpha=0.7)
    ax2.grid(True)
    ax2.set_title(f"First Difference Mean and Volatility (Window Size: {window})")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(t_diff, x1_d1, color="blue", alpha=0.2)
    ax3.plot(t_diff, x2_d1, color="red", alpha=0.2)
    ax3.plot(t_window, x1_upper, color="blue", alpha=0.7)
    ax3.plot(t_window, x1_means, color="blue", alpha=0.7)
    ax3.plot(t_window, x1_lower, color="blue", alpha=0.7)
    ax3.plot(t_window, x2_upper, color="red", alpha=0.7)
    ax3.plot(t_window, x2_means, color="red", alpha=0.7)
    ax3.plot(t_window, x2_lower, color="red", alpha=0.7)
    ax3.grid(True)
    ax3.set_title(f"First Differences, Mean +/- StDev (Window Size: {window})")

    plt.show()
