import os

import pandas as pd

from nonstationary import function_of_time, mean, volatility


class TimeSeries:

    def __init__(self, source, time_col: str, time_fmt: str = None):
        if isinstance(source, pd.DataFrame):
            df = source
        elif isinstance(source, str):
            df = TimeSeries._from_file(source, time_col, time_fmt)
        else:
            raise Exception(
                f'Unkown source type. source must be a str or pandas.DataFrame, '
                'not {type(source)}'
            )
        self.df: pd.DataFrame = df
        self.tcol = time_col

    def columns(self):
        return self.df.columns

    def mean(self, col: str):
        return mean(list(self.df[col]))

    def volatility(self, col: str):
        return volatility(self.df[col].tolist())

    def tv_mean(self, col: str, win_sz: int):
        """Return a vector of the time-varying mean and the assocated time vector."""
        values = self.df[col].tolist()
        times = self.df[self.tcol].tolist()
        means, times = function_of_time(values, mean, win_sz, times)
        return means, times

    def tv_volatility(self, col: str, win_size: int):
        """Return a vector of the time-varying volatility and the assocated time vector."""
        values = self.df[col].tolist()
        times = self.df[self.tcol].tolist()
        volatilities, times = function_of_time(values, volatility, win_size, times)
        return volatilities, times

    def plot(self, col: str, ax=None, plot_kwargs={}):
        if ax:
            plot_kwargs.update({"ax": ax})
        self.df.plot(x=self.tcol, y=col, **plot_kwargs)
        return ax

    @staticmethod
    def _from_file(filepath: str, time_col: str, time_fmt: str) -> pd.DataFrame:
        if not os.path.isfile(filepath):
            raise Exception(f"File not found: {filepath}")
        df = pd.read_csv(filepath)
        df[time_col] = pd.to_datetime(df[time_col], format=time_fmt)
        df.sort_values(by=time_col, ascending=False, ignore_index=True, inplace=True)
        return df
