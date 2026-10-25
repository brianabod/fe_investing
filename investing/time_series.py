"""Time series with a date column, and statistics that drift over time.

Financial time series are non-stationary: their mean and volatility change
over time. The rolling ("time-varying") statistics here make that visible.
All data is kept oldest first, so each rolling value uses only past data and
is labeled with the newest date in its window.
"""

import os

import numpy as np
import pandas as pd


def mean(values) -> float:
    return float(np.nanmean(np.asarray(values, dtype=float)))


def volatility(values) -> float:
    """Sample standard deviation (ddof=1), ignoring NaNs."""
    return float(np.nanstd(np.asarray(values, dtype=float), ddof=1))


def window_time_vector(time_vector, window_size: int) -> list:
    """The time of the last (newest) point in each full window.
    Assumes time_vector is sorted oldest first."""
    return list(time_vector)[window_size - 1:]


def function_of_time(values, function: callable, window_size: int = 1, time_vector=None):
    """Apply ``function`` to every full sliding window of ``window_size`` points.

    Returns the outputs, plus the matching window-end times if a time vector is
    given. Data must be ordered oldest first so each output only uses the past.
    """
    series = pd.Series(np.asarray(values, dtype=float))
    outputs = series.rolling(window_size).apply(function, raw=True).iloc[window_size - 1:].tolist()
    if time_vector is None:
        return outputs
    return outputs, window_time_vector(time_vector, window_size)


class TimeSeries:
    """A DataFrame with a time column, sorted oldest first."""

    def __init__(self, source, time_column: str, time_format: str | None = None):
        if isinstance(source, pd.DataFrame):
            dataframe = source.sort_values(by=time_column, ignore_index=True)
        elif isinstance(source, str | os.PathLike):
            dataframe = TimeSeries._from_file(source, time_column, time_format)
        else:
            raise TypeError(
                "Unknown source type. source must be a path or pandas.DataFrame, "
                f"not {type(source)}"
            )
        self.df: pd.DataFrame = dataframe
        self.time_column = time_column

    def columns(self):
        return self.df.columns

    def mean(self, column: str) -> float:
        return mean(self.df[column])

    def volatility(self, column: str) -> float:
        return volatility(self.df[column])

    def time_varying_mean(self, column: str, window_size: int):
        """Rolling mean and the matching window-end times."""
        return self._rolling(column, window_size, "mean")

    def time_varying_volatility(self, column: str, window_size: int):
        """Rolling volatility and the matching window-end times."""
        return self._rolling(column, window_size, "std")

    def _rolling(self, column: str, window_size: int, statistic: str):
        rolled = getattr(self.df[column].rolling(window_size), statistic)().iloc[window_size - 1:]
        times = self.df[self.time_column].iloc[window_size - 1:]
        return rolled.tolist(), times.tolist()

    def plot(self, column: str, ax=None, **line_kwargs):
        """Draw one column against time as a line; returns the axes."""
        import matplotlib.pyplot as plt

        ax = ax if ax is not None else plt.gca()
        ax.plot(self.df[self.time_column], self.df[column], **line_kwargs)
        return ax

    @staticmethod
    def _from_file(file_path, time_column: str, time_format: str | None) -> pd.DataFrame:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        dataframe = pd.read_csv(file_path, na_values=["null"])
        dataframe[time_column] = pd.to_datetime(dataframe[time_column], format=time_format)
        dataframe = dataframe.drop_duplicates(subset=time_column, keep="last")
        return dataframe.sort_values(by=time_column, ignore_index=True)
