"""A single stock loaded from a Yahoo Finance-style CSV."""

import numpy as np
import pandas as pd
from scipy.stats import cauchy

from .time_series import TimeSeries


class Stock:

    DATE = "Date"
    OPEN = "Open"
    CLOSE = "Close"
    ADJUSTED_CLOSE = "Adj Close"
    RETURN = "Return"
    LOG_RETURN = "Log-Return"
    UNITS = "USD"

    def __init__(
        self,
        symbol: str,
        file_path,
        company: str | None = None,
        color=None,
        time_column: str = "Date",
        time_format: str = "%Y-%m-%d",
    ):
        self.symbol = symbol
        self.company = company or symbol
        self.color = color
        self.time_series = TimeSeries(file_path, time_column, time_format)
        dataframe = self.time_series.df

        # Adjusted closes remove the jumps caused by splits and dividends.
        # Fall back to the raw close if the file has no adjusted column.
        self.price_column = Stock.ADJUSTED_CLOSE if Stock.ADJUSTED_CLOSE in dataframe else Stock.CLOSE
        price = dataframe[self.price_column]

        # The simple return over an interval is the change in price over the
        # initial price (close to close). Use the simple return for portfolio
        # optimization because it has an additive property over assets.
        dataframe[Stock.RETURN] = price.pct_change()

        # The log-return is the continuously compounded return. Use the
        # log-return for modeling financial time series because it has a nice
        # additive property over periods that the simple return does not have.
        dataframe[Stock.LOG_RETURN] = np.log(price).diff()

    @property
    def prices(self) -> pd.Series:
        """Date-indexed prices, named by symbol."""
        dataframe = self.time_series.df
        prices = dataframe.set_index(self.time_series.time_column)[self.price_column]
        return prices[prices > 0].rename(self.symbol)

    def column(self, metric: str) -> pd.Series:
        """Date-indexed values of any column, named by symbol."""
        return self.time_series.df.set_index(self.time_series.time_column)[metric].rename(self.symbol)

    def mean(self, metric: str) -> float:
        return self.time_series.mean(metric)

    def volatility(self, metric: str) -> float:
        return self.time_series.volatility(metric)

    def mean_over_time(self, metric: str, window_size: int) -> TimeSeries:
        means, times = self.time_series.time_varying_mean(metric, window_size)
        return TimeSeries(pd.DataFrame({Stock.DATE: times, metric: means}), Stock.DATE)

    def volatility_over_time(self, metric: str, window_size: int) -> TimeSeries:
        volatilities, times = self.time_series.time_varying_volatility(metric, window_size)
        return TimeSeries(pd.DataFrame({Stock.DATE: times, metric: volatilities}), Stock.DATE)

    def fit_cauchy(self, metric: str):
        """Fit a Cauchy distribution (heavy-tailed) to a return column; returns (location, scale)."""
        values = self.time_series.df[metric]
        return cauchy.fit(values[np.isfinite(values)])

    def plot_time_series(self, ax, metric, **line_kwargs):
        """Line of one column over time, labeled with the company and drawn in the stock's color."""
        line_kwargs = {"label": self.company, "color": self.color, **line_kwargs}
        return self.time_series.plot(metric, ax, **line_kwargs)

    def plot_distribution(self, ax, column=RETURN, value_range=(-0.1, 0.1), bins=160, **hist_kwargs):
        """Density histogram of a return column within ``value_range``."""
        values = self.time_series.df[column]
        values = values[np.isfinite(values)]
        hist_kwargs = {"color": self.color, "alpha": 0.55, "histtype": "stepfilled",
                       "linewidth": 0, "label": self.company, **hist_kwargs}
        return ax.hist(values, bins=bins, range=value_range, density=True, **hist_kwargs)
