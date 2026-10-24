"""Shared plot helpers: distribution curves and statistical arbitrage charts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import Axes
from scipy.stats import cauchy, norm


def draw_zero_axis(ax: Axes):
    """Horizontal line at y=0 that doesn't change the x-axis limits."""
    ax.axhline(0, color="black", linewidth=0.8)


def plot_gaussian(ax: Axes, mean: float = 0, standard_deviation: float = 1, x=None, plot_kwargs=None):
    x = np.linspace(-0.5, 0.5, 501) if x is None else x
    label = rf"Gaussian ($\mu$={mean:.3f}, $\sigma$={standard_deviation:.3f})"
    ax.plot(x, norm.pdf(x, mean, standard_deviation), label=label, **(plot_kwargs or {}))


def plot_cauchy(ax: Axes, location: float = 0, scale: float = 1, x=None, plot_kwargs=None):
    x = np.linspace(-0.5, 0.5, 501) if x is None else x
    label = rf"Cauchy ($x_0$={location:.3f}, $\gamma$={scale:.3f})"
    ax.plot(x, cauchy.pdf(x, location, scale), label=label, **(plot_kwargs or {}))


def plot_spread_trading(log_prices: pd.DataFrame, standardized_spread: pd.Series,
                        entry_threshold: float, positions, filtered_spread=None, title: str = ""):
    """Log-prices, the standardized spread with +/- entry thresholds, the
    positions taken, and the cumulative profit (Feng & Palomar Figs. 10.6, 10.10)."""
    positions = np.asarray(positions, dtype=float)
    profit = np.cumsum(np.r_[0.0, positions[:-1] * np.diff(standardized_spread.to_numpy())])

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
    log_prices.plot(ax=axes[0], grid=True, ylabel="Log-price")
    axes[1].plot(standardized_spread.index, standardized_spread, color="black", lw=1,
                 label="spread (standard deviations)")
    if filtered_spread is not None:
        axes[1].plot(standardized_spread.index, filtered_spread, color="tab:orange", lw=1,
                     label="Kalman-filtered")
    for level in (entry_threshold, -entry_threshold):
        axes[1].axhline(level, color="magenta", ls="--")
    axes[1].axhline(0, color="tab:blue", ls=":")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].step(standardized_spread.index, positions, where="post")
    axes[2].set_ylabel("Position")
    axes[2].grid(True)
    axes[3].plot(standardized_spread.index, profit)
    axes[3].set_ylabel("Cumulative profit\n(standard deviations)")
    axes[3].grid(True)
    fig.suptitle(title)
    return fig


def plot_threshold_curves(results: dict):
    """Trade frequency and total profit against the entry threshold for each
    method in ``results`` (name -> ThresholdResult), as in Figs. 10.11-10.12."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, result in results.items():
        axes[0].plot(result.thresholds, result.frequency, label=name)
        best_profit = max(result.profit.max(), 1e-12)
        axes[1].plot(result.thresholds, result.profit / best_profit,
                     label=f"{name} (best {result.best_threshold:.2f})")
        if result.empirical_frequency is not None:
            axes[0].plot(result.thresholds, result.empirical_frequency, ".", ms=3,
                         label=f"{name}: raw exceedance")
    axes[0].set_xlabel("Entry threshold (standard deviations)")
    axes[0].set_ylabel("Trade frequency")
    axes[1].set_xlabel("Entry threshold (standard deviations)")
    axes[1].set_ylabel("Total profit (normalized)")
    for ax in axes:
        ax.legend()
        ax.grid(True)
    fig.suptitle("Optimal entry threshold (Sec. 10.4.1)")
    return fig


def plot_equity_curves(results: dict, title: str = "Out-of-sample statistical arbitrage"):
    """Equity curves for each backtest in ``results`` (name -> BacktestResult)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, result in results.items():
        result.equity.plot(ax=ax, label=name)
    ax.set_ylabel("Equity (start = 1)")
    ax.grid(True)
    ax.legend()
    fig.suptitle(title)
    return fig
