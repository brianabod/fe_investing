from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd

from Stock import Stock
from TimeSeries import TimeSeries
from utility import draw_zero_axis, plot_cauchy, plot_gaussian


class Portfolio:

    def __init__(self, stocks: list, metrics=[Stock.RETURN, Stock.LOG_RETURN], max_values:int=10000):
        for stock in stocks:
            stock.time_series.df = stock.time_series.df.head(max_values)
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
            means = []
            volatilities = []
            for stock in self.stocks:
                stock: Stock
                stock.plot_distribution(ax, metric)
                means.append(stock.mean(metric))
                volatilities.append(stock.volatility(metric))
            avg_mean = sum(means) / len(means)
            avg_volatility = sum(volatilities) / len(volatilities)
            plot_gaussian(ax, avg_mean, avg_volatility)
            plot_cauchy(ax, avg_mean, 0.25 * avg_volatility)
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
        ax1 = fig.add_subplot(3, 1, 1)
        draw_zero_axis(ax1)
        difference_kwargs = {
            "ylabel": f"{u'Δ'}{metric} (USD)",
            "ylim": [-0.1, 0.1],
            "alpha": 0.3,
        }
        ax2 = fig.add_subplot(3, 1, 2)
        draw_zero_axis(ax2)
        mean_difference_kwargs = {
            "ylabel": f"Mean\n{u'Δ'}{metric} (USD)",
            "ylim": [-0.01, 0.01],
            "alpha": 0.7,
        }
        ax3 = fig.add_subplot(3, 1, 3)
        draw_zero_axis(ax3)
        volatility_difference_kwargs = {
            "ylabel": f"Volatility\n{u'Δ'}{metric} (USD)",
            "ylim": [-0.01, 0.01],
            "alpha": 0.7,
        }
        for stock in self.stocks[1:]:
            stock: Stock
            label = f"{stock.symbol}-{self.reference_stock.symbol}"
            mean_ts = stock.mean_over_time(metric, window_size)
            volatility_ts = stock.volatility_over_time(metric, window_size)
            difference_ts = TimeSeries(
                pd.DataFrame(
                    {
                        Stock.DATE: metric_ts.df[Stock.DATE],
                        label: metric_ts.df[stock.symbol] - reference_metric,
                    }
                ),
                Stock.DATE,
            )
            mean_difference_ts = TimeSeries(
                pd.DataFrame(
                    {
                        Stock.DATE: mean_ts.df[Stock.DATE],
                        label: mean_ts.df[metric] - reference_mean_ts.df[metric],
                    }
                ),
                Stock.DATE,
            )
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
            difference_kwargs.update(common_kwargs)
            mean_difference_kwargs.update(common_kwargs)
            volatility_difference_kwargs.update(common_kwargs)
            difference_ts.plot(label, ax1, difference_kwargs)
            mean_difference_ts.plot(label, ax2, mean_difference_kwargs)
            volatility_difference_ts.plot(label, ax3, volatility_difference_kwargs)
        ax1.legend()
        ax2.legend()
        ax3.legend()
        return fig
