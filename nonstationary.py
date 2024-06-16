import numpy as np

"""
A univariate time series is said to be "integrated" if it can be brought to
stationary through differencing. The number of differences required to achieve
stationarity is called the order of integration.

An n-dimensional time series is "cointegrated" if some linear combination of
the component variables is stationary. The combination is called a
cointegrated relation, and the coefficients form a cointegrated vector.

Cointegration is distinguished from traditional economic equilibrium, in which
a balance of forces produces stable long-term levels in the variables.
Cointegrated variables are generally unstable in their levels, but exhibit
mean-reverting "spreads" (generalized by the cointegrating relation) that
force the variables to move around common stochastic trends. Cointegration is
also distinguished from the short-term synchronies of positive covariance,
which only measures the tendency to move together at each time step.
Modifications of the VAR model to include cointegrated variables balances the
short-term dynamics of the system with long-term tendencies.
"""

def mean(time_series):
    return np.mean(time_series)


def volatility(time_series):
    return np.std(time_series)


def window_time_vector(time_vector: list, window_size: int) -> list:
    N = len(time_vector)
    window_times = []
    for i in range(N - window_size + 1):
        window = time_vector[i : i + window_size]
        window_times.append(window[-1])
    return window_times


def function_of_time(
    time_series, func: callable, window_size: int = 1, time_vector: list = None
):
    N = len(time_series)
    time_varying_output = []
    for i in range(N - window_size + 1):
        window = time_series[i : i + window_size]
        time_varying_output.append(func(window))
    if time_vector is None:
        return time_varying_output
    window_times = window_time_vector(time_vector, window_size)
    return time_varying_output, window_times
