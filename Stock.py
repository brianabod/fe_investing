import numpy as np
import pandas as pd

from TimeSeries import TimeSeries


class Stock:

    DATE = "Date"
    OPEN = "Open"
    CLOSE = "Close"
    RETURN = "Return"
    LOG_RETURN = "Log-Return"
    UNITS = "USD"

    def __init__(
        self,
        company,
        symbol,
        filepath,
        color,
        time_column="Date",
        time_format="%Y-%m-%d",
    ):
        self.company: str = company
        self.symbol: str = symbol
        self.color = color
        self.time_series = TimeSeries(filepath, time_column, time_format)

        # The simple return over an interval is the change in price over the
        # initial price, assuming the asset pays no dividends. Use the simple
        # return for portfolio optimization because it has an additive
        # property over assets.
        self.time_series.df[Stock.RETURN] = (self.time_series.df[Stock.CLOSE] / self.time_series.df[Stock.OPEN]) - 1

        # The log-return is the continuously compounded return. Use the
        # log-return for modeling financial time series because it has a nice
        # additive property over periods that the simple return does not have.
        self.time_series.df[Stock.LOG_RETURN] = np.log(1 + self.time_series.df[Stock.RETURN])

    def mean(self, metric: str):
        return self.time_series.mean(metric)

    def volatility(self, metric: str):
        return self.time_series.volatility(metric)

    def mean_over_time(self, metric: str, window_size: int):
        means, times = self.time_series.tv_mean(metric, window_size)
        return TimeSeries(pd.DataFrame({Stock.DATE: times, metric: means}), Stock.DATE)

    def volatility_over_time(self, metric: str, window_size: int):
        volatilities, times = self.time_series.tv_volatility(metric, window_size)
        return TimeSeries(
            pd.DataFrame({Stock.DATE: times, metric: volatilities}),
            Stock.DATE
        )

    def plot_time_series(self, ax, metric, label=None, legend=False, alpha=0.7, kwargs_overrides={}):
        plot_kwargs = {
            "label": self.company,
            "xlabel": Stock.DATE,
            "ylabel": f"{metric if label is None else label} ({Stock.UNITS})",
            "alpha": alpha,
            "legend": legend,
            "color": self.color,
            "grid": True,
        }
        plot_kwargs.update(kwargs_overrides)
        return self.time_series.plot(metric, ax, plot_kwargs)

    def plot_distribution(
        self,
        ax,
        column="Return",
        label=None,
        units="USD",
        kwargs_overrides={},
    ):
        if label is None:
            label = column
        df = self.time_series.df[np.isfinite(self.time_series.df[column])]
        mean = round(np.mean(df[column]), 3)
        volatility = round(np.std(df[column]), 3)
        legend_label = f"{self.company} ($\mu$={mean}, $\sigma$={volatility})"
        kwargs = {}
        kwargs.update(kwargs_overrides)
        return df[column].plot.hist(
            ax=ax,
            bins=500,
            density=True,
            xlabel=f"{label} ({units})",
            ylabel="Density",
            label=legend_label,
            alpha=0.7,
            grid=True,
            color=self.color,
            **kwargs,
        )
