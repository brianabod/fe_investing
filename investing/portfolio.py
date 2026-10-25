"""A group of stocks lined up by date.

``Portfolio`` is the single entry point for multi-stock data. The exploratory
plots use its metric tables, and the statistical arbitrage code uses
``prices()`` / ``log_prices()``. Every table keeps only the dates all stocks
traded on.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .plotting import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES_COLORS,
    add_figure_legend,
    draw_zero_axis,
    format_percent,
    plot_cauchy,
    plot_gaussian,
    series_color,
    set_figure_title,
)
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
        # Stocks without a color get the next palette slot, in portfolio order.
        # The palette has 8 colors; any further stocks are drawn in neutral gray.
        for index, stock in enumerate(stocks):
            if stock.color is None:
                stock.color = series_color(index) if index < len(SERIES_COLORS) else INK_MUTED
        # One table per metric with a column per stock, matched by date.
        self.metrics = {
            metric: TimeSeries(self.aligned(metric).reset_index(), Stock.DATE)
            for metric in metrics
        }

    @classmethod
    def from_directory(cls, directory, symbols: list[str] | None = None, companies: dict | None = None,
                       colors: dict | None = None, minimum_days: int = 252) -> "Portfolio":
        """Load ``<directory>/<SYMBOL>.csv`` for each symbol.

        With no ``symbols``, every CSV in the directory is loaded, sorted by
        name. Discovered files with fewer than ``minimum_days`` valid prices
        are skipped with a warning, since one sparse file would shrink every
        date-aligned table to almost nothing.
        """
        companies, colors = companies or {}, colors or {}
        discover = symbols is None
        if discover:
            symbols = sorted(os.path.splitext(name)[0] for name in os.listdir(directory)
                             if name.lower().endswith(".csv"))
        stocks = []
        for symbol in symbols:
            stock = Stock(symbol, os.path.join(directory, f"{symbol}.csv"),
                          company=companies.get(symbol), color=colors.get(symbol))
            if discover and len(stock.prices.dropna()) < minimum_days:
                warnings.warn(f"Skipping {symbol}: fewer than {minimum_days} valid prices.")
                continue
            stocks.append(stock)
        if not stocks:
            raise ValueError(f"No usable price files found in {directory}.")
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

    def _warn_if_too_many_lines(self, count: int):
        if count > len(SERIES_COLORS):
            warnings.warn(f"{count} stocks on one chart is hard to read; stocks past the first "
                          f"{len(SERIES_COLORS)} are gray. Plot a subset (e.g. --symbols) for clearer charts.")

    def plot_time_series(self):
        """Adjusted prices (log scale) and daily returns for every stock."""
        self._warn_if_too_many_lines(len(self.stocks))
        fig, (price_ax, return_ax, log_return_ax) = plt.subplots(
            3, 1, sharex=True, figsize=(10, 8), height_ratios=[1.4, 1, 1])
        for stock in self.stocks:
            stock.plot_time_series(price_ax, stock.price_column, linewidth=1.2)
            stock.plot_time_series(return_ax, Stock.RETURN, linewidth=0.4, alpha=0.7)
            stock.plot_time_series(log_return_ax, Stock.LOG_RETURN, linewidth=0.4, alpha=0.7)
        price_ax.set_yscale("log")
        price_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}" if value >= 1 else f"${value:g}"))
        price_ax.set_title("Adjusted price (log scale)")
        return_ax.set_title("Daily return")
        log_return_ax.set_title("Daily log-return")
        for ax in (return_ax, log_return_ax):
            draw_zero_axis(ax)
            format_percent(ax)
        add_figure_legend(fig, [price_ax])
        set_figure_title(fig, "Prices and daily returns")
        return fig

    def plot_distributions(self, value_range=(-0.1, 0.1)):
        """Histogram of each stock's returns with its own Gaussian and Cauchy
        fits. Fat tails show up as mass the Gaussian misses."""
        metrics = [Stock.RETURN, Stock.LOG_RETURN]
        fig, axes = plt.subplots(len(metrics), len(self.stocks), sharex=True, sharey=True,
                                 figsize=(3.4 * len(self.stocks) + 1, 6.5), squeeze=False)
        x = np.linspace(*value_range, 801)
        for row, metric in enumerate(metrics):
            for column, stock in enumerate(self.stocks):
                ax = axes[row, column]
                stock.plot_distribution(ax, metric, value_range=value_range)
                plot_gaussian(ax, stock.mean(metric), stock.volatility(metric), x=x, label="Gaussian fit",
                              plot_kwargs={"color": INK_PRIMARY, "linewidth": 1.1})
                plot_cauchy(ax, *stock.fit_cauchy(metric), x=x, label="Cauchy fit",
                            plot_kwargs={"color": INK_SECONDARY, "linewidth": 1.1, "linestyle": (0, (4, 2))})
                ax.text(0.03, 0.95, f"σ = {stock.volatility(metric):.1%}", transform=ax.transAxes,
                        va="top", fontsize=8.5, color=INK_SECONDARY)
                ax.set_title(stock.company if row == 0 else "")
                ax.xaxis.set_major_locator(MultipleLocator(0.05))
                format_percent(ax, "x", decimals=0)
                if column == 0:
                    ax.set_ylabel(f"{metric.capitalize()} density")
                if row == len(metrics) - 1:
                    ax.set_xlabel(f"Daily {metric.lower()}")
        histogram = Patch(color=INK_MUTED, alpha=0.55, linewidth=0)
        handles = [histogram] + axes[0, 0].get_lines()[:2]
        add_figure_legend(fig, handles=handles, labels=["Observed (stock color)", "Gaussian fit", "Cauchy fit"])
        set_figure_title(fig, "Daily returns have fatter tails than a Gaussian")
        return fig

    def plot_statistics_over_time(self, window_size: int, metric=Stock.LOG_RETURN):
        """Rolling mean and volatility (window_size in trading days)."""
        self._warn_if_too_many_lines(len(self.stocks))
        fig, (mean_ax, volatility_ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 6.5))
        for stock in self.stocks:
            stock.mean_over_time(metric, window_size).plot(
                metric, mean_ax, label=stock.company, color=stock.color, linewidth=1.2)
            stock.volatility_over_time(metric, window_size).plot(
                metric, volatility_ax, label=stock.company, color=stock.color, linewidth=1.2)
        draw_zero_axis(mean_ax)
        format_percent(mean_ax, decimals=2)
        format_percent(volatility_ax, decimals=1)
        mean_ax.set_title(f"Rolling mean of daily {metric.lower()}")
        volatility_ax.set_title(f"Rolling volatility of daily {metric.lower()}")
        volatility_ax.set_ylim(bottom=0)
        add_figure_legend(fig, [mean_ax])
        set_figure_title(fig, f"Mean and volatility drift over time ({window_size}-day rolling window)")
        return fig

    def compare_stocks(self, metric=Stock.LOG_RETURN, window_size=252):
        """Each stock minus the reference stock, on shared dates: the daily
        difference, its rolling mean, and the gap in rolling volatility."""
        if len(self.stocks) < 2:
            raise ValueError("Portfolio must contain at least 2 stocks to compare.")
        self._warn_if_too_many_lines(len(self.stocks) - 1)
        table = self.metrics[metric].df.set_index(Stock.DATE)
        reference = self.reference_stock.symbol
        reference_volatility = table[reference].rolling(window_size).std()

        fig, (difference_ax, mean_ax, volatility_ax) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        for stock in self.stocks[1:]:
            difference = table[stock.symbol] - table[reference]
            label = f"{stock.symbol} − {reference}"
            difference_ax.plot(difference.index, difference, color=stock.color, linewidth=0.4, alpha=0.7)
            mean_ax.plot(difference.index, difference.rolling(window_size).mean(),
                         color=stock.color, linewidth=1.2, label=label)
            volatility_ax.plot(difference.index,
                               table[stock.symbol].rolling(window_size).std() - reference_volatility,
                               color=stock.color, linewidth=1.2)
        titles = [f"Daily {metric.lower()} difference",
                  f"Rolling mean of the difference ({window_size} days)",
                  f"Rolling volatility minus {reference}'s ({window_size} days)"]
        for ax, title in zip((difference_ax, mean_ax, volatility_ax), titles):
            draw_zero_axis(ax)
            format_percent(ax, decimals=1)
            ax.set_title(title)
        add_figure_legend(fig, [mean_ax])
        set_figure_title(fig, f"Each stock compared with {self.reference_stock.company}")
        return fig
