import numpy as np


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
