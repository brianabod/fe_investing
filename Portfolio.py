from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd

from Stock import Stock
from TimeSeries import TimeSeries
from utility import draw_zero_axis, plot_cauchy, plot_gaussian


class Portfolio:

    def __init__(self, stocks: list, metrics=[Stock.RETURN, Stock.LOG_RETURN]):
        self.stocks = stocks
        self.symbols = [stock.symbol for stock in stocks]
        self.reference_stock: Stock = stocks[0]
        time_vector = self.reference_stock.time_series.df[Stock.DATE]
        self.metrics = {}
        for metric in metrics:
            self.metrics[metric] = {Stock.DATE: time_vector}
            for stock in stocks:
                stock: Stock
                self.metrics[metric][stock.symbol] = stock.time_series.df[metric]
            self.metrics[metric] = TimeSeries(
                pd.DataFrame(self.metrics[metric]), Stock.DATE
            )

    def plot_time_series(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        draw_zero_axis(ax1)
        for stock in self.stocks:
            stock: Stock
            stock.plot_time_series(
                ax1, Stock.CLOSE, label=f"Price ({Stock.UNITS})", legend=True
            )
        ax2 = fig.add_subplot(3, 1, 2)
        draw_zero_axis(ax2)
        for stock in self.stocks:
            stock.plot_time_series(ax2, Stock.RETURN)
        ax3 = fig.add_subplot(3, 1, 3)
        draw_zero_axis(ax3)
        for stock in self.stocks:
            stock.plot_time_series(ax3, Stock.LOG_RETURN)
        fig.suptitle("Time Series")
        return fig

    def plot_distributions(self):
        fig = plt.figure()
        legend = True
        for figure_idx, metric in enumerate([Stock.RETURN, Stock.LOG_RETURN], 1):
            ax = fig.add_subplot(2, 1, figure_idx)
            if figure_idx != 1:
                legend = False
            for stock in self.stocks:
                stock: Stock
                stock.plot_distribution(ax, metric)
            plot_gaussian(ax, stock.mean(metric), stock.volatility(metric))
            plot_cauchy(ax, stock.mean(metric), 0.25 * stock.volatility(metric))
            ax.set_xlim([-0.25, 0.25])
            ax.set_ylim([0, 100])
            if legend:
                ax.legend()
        fig.suptitle("Distributions\n(Non-Gaussian)")
        return fig

    def plot_statistics_over_time(self, window_size: int, metric=Stock.LOG_RETURN):
        """Plot mean and volatility over a rolling window (window_size in days)."""
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        for stock in self.stocks:
            stock: Stock
            stock.plot_time_series(
                ax1, metric, alpha=0.3, kwargs_overrides={"linewidth": 0.5}
            )
            common_kwargs = {
                "drawstyle": "steps",
                "xlabel": stock.DATE,
                "ylabel": f"{metric} (log-{stock.UNITS})",
                "alpha": 0.7,
                "legend": True,
                "color": stock.color,
                "grid": True,
            }
            mean_ts = stock.mean_over_time(metric, window_size)
            mean_kwargs = deepcopy(common_kwargs)
            mean_kwargs.update(
                {
                    "label": f"{stock.company} Mean",
                    "linewidth": 2,
                }
            )
            mean_ts.plot(metric, ax1, mean_kwargs)
            volatility_ts = stock.volatility_over_time(metric, window_size)
            volatility_kwargs = deepcopy(common_kwargs)
            volatility_kwargs.update(
                {
                    "label": f"{stock.company} Volatility",
                    "linewidth": 1,
                }
            )
            volatility_ts.plot(metric, ax1, volatility_kwargs)
        ax1.legend()
        fig.suptitle(
            "Non-Stationary Mean & Volatility of Log-Returns\n"
            f"({window_size}-day rolling statistics)"
        )
        return fig

    def compare_stocks(self, metric=Stock.LOG_RETURN, window_size=365):
        if len(self.stocks) < 2:
            raise Exception("Portfolio must contain at least 2 stocks to compare.")
        metric_ts: TimeSeries = self.metrics[metric]
        reference_metric = metric_ts.df[self.reference_stock.symbol]
        reference_mean_ts = self.reference_stock.mean_over_time(metric, window_size)
        reference_volatility_ts = self.reference_stock.volatility_over_time(
            metric, window_size
        )
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        draw_zero_axis(ax)
        for stock in self.stocks[1:]:
            stock: Stock
            label = f"{stock.symbol}-{self.reference_stock.symbol}"
            difference_ts = TimeSeries(
                pd.DataFrame(
                    {
                        Stock.DATE: metric_ts.df[Stock.DATE],
                        label: metric_ts.df[stock.symbol] - reference_metric,
                    }
                ),
                Stock.DATE,
            )
            mean_ts = stock.mean_over_time(metric, window_size)
            mean_difference_ts = TimeSeries(
                pd.DataFrame(
                    {
                        Stock.DATE: mean_ts.df[Stock.DATE],
                        label: mean_ts.df[metric] - reference_mean_ts.df[metric],
                    }
                ),
                Stock.DATE,
            )
            volatility_ts = stock.volatility_over_time(metric, window_size)
            volatility_difference_ts = TimeSeries(
                pd.DataFrame(
                    {
                        Stock.DATE: volatility_ts.df[Stock.DATE],
                        label: volatility_ts.df[metric]
                        - reference_volatility_ts.df[metric],
                    }
                ),
                Stock.DATE,
            )

            common_kwargs = {
                "label": label,
                "color": stock.color,
                "xlabel": Stock.DATE,
                "grid": True,
            }
            difference_kwargs = deepcopy(common_kwargs)
            difference_kwargs.update(
                {
                    "ylabel": f"{metric} Difference",
                    "alpha": 0.3,
                }
            )
            difference_ts.plot(label, ax, difference_kwargs)
            mean_difference_kwargs = deepcopy(common_kwargs)
            mean_difference_kwargs.update(
                {
                    "ylabel": f"{metric} Mean Difference",
                    "alpha": 0.7,
                }
            )
            mean_difference_ts.plot(label, ax, mean_difference_kwargs)
            volatility_difference_kwargs = deepcopy(common_kwargs)
            volatility_difference_kwargs.update(
                {
                    "ylabel": f"{metric} Volatility Difference",
                    "alpha": 0.7,
                }
            )
            volatility_difference_ts.plot(label, ax, volatility_difference_kwargs)
        ax.legend()
        return fig
