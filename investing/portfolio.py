"""A group of stocks lined up by date.

``Portfolio`` is the single entry point for multi-stock data. The exploratory
plots use its metric tables, and the statistical arbitrage code uses
``prices()`` / ``log_prices()``. Every table keeps only the dates all stocks
traded on.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import draw_zero_axis, plot_cauchy, plot_gaussian
from .stock import Stock
from .time_series import TimeSeries


def align_by_date(series: list[pd.Series]) -> pd.DataFrame:
    """Join date-indexed series into one table, keeping only shared dates."""
    return pd.concat(series, axis=1, join="inner").sort_index().dropna()


class Portfolio:

    def __init__(self, stocks: list[Stock], metrics=(Stock.RETURN, Stock.LOG_RETURN)):
        self.stocks = stocks
        self.symbols = [stock.symbol for stock in stocks]
        self.reference_stock: Stock = stocks[0]
        # One table per metric with a column per stock, matched by date.
        self.metrics = {
            metric: TimeSeries(self.aligned(metric).reset_index(), Stock.DATE)
            for metric in metrics
        }

    @classmethod
    def from_directory(cls, symbols: list[str], directory, companies: dict | None = None,
                       colors: dict | None = None) -> "Portfolio":
        """Load ``<directory>/<SYMBOL>.csv`` for each symbol."""
        companies, colors = companies or {}, colors or {}
        stocks = [
            Stock(symbol, os.path.join(directory, f"{symbol}.csv"),
                  company=companies.get(symbol), color=colors.get(symbol))
            for symbol in symbols
        ]
        return cls(stocks)

    def aligned(self, metric: str) -> pd.DataFrame:
        """Date-indexed table of one column for every stock, on shared dates."""
        table = align_by_date([stock.column(metric) for stock in self.stocks])
        table.index.name = Stock.DATE
        return table

    def prices(self) -> pd.DataFrame:
        """Date-indexed adjusted prices, one column per symbol, on shared dates."""
        table = align_by_date([stock.prices for stock in self.stocks])
        table.index.name = Stock.DATE
        return table

    def log_prices(self) -> pd.DataFrame:
        return np.log(self.prices())

    # --- Exploratory plots -------------------------------------------------

    def plot_time_series(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        for stock in self.stocks:
            stock.plot_time_series(ax1, stock.price_column, label="Price", legend=True)
        for index, metric in enumerate([Stock.RETURN, Stock.LOG_RETURN], 2):
            ax = fig.add_subplot(3, 1, index)
            draw_zero_axis(ax)
            for stock in self.stocks:
                stock.plot_time_series(ax, metric)
        fig.suptitle("Time Series")
        return fig

    def plot_distributions(self):
        """Histogram of each stock's returns with that stock's Gaussian and
        Cauchy fits. Fat tails show up as mass the Gaussian misses."""
        fig = plt.figure()
        for index, metric in enumerate([Stock.RETURN, Stock.LOG_RETURN], 1):
            ax = fig.add_subplot(2, 1, index)
            for stock in self.stocks:
                stock.plot_distribution(ax, metric)
                color = {"color": stock.color}
                plot_gaussian(ax, stock.mean(metric), stock.volatility(metric),
                              plot_kwargs={**color, "linestyle": "--"})
                plot_cauchy(ax, *stock.fit_cauchy(metric), plot_kwargs={**color, "linestyle": ":"})
            ax.set_xlim([-0.25, 0.25])
            ax.set_ylim([0, 100])
            if index == 1:
                ax.legend(fontsize="small")
        fig.suptitle("Distributions\n(Non-Gaussian)")
        return fig

    def plot_statistics_over_time(self, window_size: int, metric=Stock.LOG_RETURN):
        """Plot mean and volatility over a rolling window (window_size in trading days)."""
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        for stock in self.stocks:
            stock.plot_time_series(ax1, metric, alpha=0.3, kwargs_overrides={"linewidth": 0.5})
            common_kwargs = {
                "drawstyle": "steps",
                "xlabel": Stock.DATE,
                "ylabel": f"{metric} (log-{Stock.UNITS})",
                "alpha": 0.7,
                "legend": True,
                "color": stock.color,
                "grid": True,
            }
            stock.mean_over_time(metric, window_size).plot(
                metric, ax1, {**common_kwargs, "label": f"{stock.company} Mean", "linewidth": 2}
            )
            stock.volatility_over_time(metric, window_size).plot(
                metric, ax1, {**common_kwargs, "label": f"{stock.company} Volatility", "linewidth": 1}
            )
        ax1.legend()
        fig.suptitle(
            "Non-Stationary Mean & Volatility of Log-Returns\n"
            f"({window_size}-day rolling statistics)"
        )
        return fig

    def compare_stocks(self, metric=Stock.LOG_RETURN, window_size=252):
        """Plot each stock minus the reference stock, on shared dates: the
        daily difference plus its rolling mean and rolling volatility gap."""
        if len(self.stocks) < 2:
            raise ValueError("Portfolio must contain at least 2 stocks to compare.")
        table = self.metrics[metric].df.set_index(Stock.DATE)
        reference = self.reference_stock.symbol
        reference_volatility = table[reference].rolling(window_size).std()

        fig, (difference_ax, mean_ax, volatility_ax) = plt.subplots(3, 1, sharex=True)
        panels = [
            (difference_ax, f"Δ{metric}", [-0.1, 0.1], 0.3),
            (mean_ax, f"Mean\nΔ{metric}", [-0.01, 0.01], 0.7),
            (volatility_ax, f"Volatility\nΔ{metric}", [-0.01, 0.01], 0.7),
        ]
        for stock in self.stocks[1:]:
            difference = table[stock.symbol] - table[reference]
            series = [
                difference,
                difference.rolling(window_size).mean(),
                table[stock.symbol].rolling(window_size).std() - reference_volatility,
            ]
            for (ax, _, _, alpha), values in zip(panels, series):
                values.plot(ax=ax, label=f"{stock.symbol}-{reference}", color=stock.color, alpha=alpha)
        for ax, ylabel, ylim, _ in panels:
            draw_zero_axis(ax)
            ax.set_ylabel(ylabel)
            ax.set_ylim(ylim)
            ax.grid(True)
            ax.legend()
        volatility_ax.set_xlabel(Stock.DATE)
        fig.suptitle(f"Difference from {self.reference_stock.company}\n"
                     f"({window_size}-day rolling statistics)")
        return fig
