"""Shared plot style and charts.

Call ``apply_style()`` once before plotting. It sets one consistent look for
every figure:
- a categorical color order that is safe for color-blind readers;
- hairline gridlines;
- compact date labels;
- automatic spacing, so labels don't overlap.

Colors follow the entity, never its rank: stock i in a Portfolio always gets
``series_color(i)``. A threshold result always gets the color of its method.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.lines import Line2D
from matplotlib.pyplot import Axes
from matplotlib.ticker import PercentFormatter
from scipy.stats import cauchy, norm

# --- Palette -------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed categorical order (blue, orange, aqua, yellow, magenta, green, violet, red).
# The order is what keeps neighboring series distinguishable; don't reorder.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948"]

STYLE = {
    "figure.figsize": (10, 6),
    "figure.facecolor": SURFACE,
    "figure.constrained_layout.use": True,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=SERIES_COLORS),
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK_PRIMARY,
    "axes.titlepad": 8,
    "axes.labelsize": 9.5,
    "axes.labelcolor": INK_SECONDARY,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": BASELINE,
    "ytick.color": BASELINE,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.handlelength": 1.6,
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "text.color": INK_PRIMARY,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "date.converter": "concise",
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 150,
}


def apply_style():
    """Use the shared style for every figure created afterwards."""
    plt.rcParams.update(STYLE)


def series_color(index: int) -> str:
    """Color for the index-th entity. There are only 8; fold extra series into
    "Other" or separate charts rather than reusing a color."""
    if not 0 <= index < len(SERIES_COLORS):
        raise ValueError(f"Only {len(SERIES_COLORS)} series colors; use fewer series or separate charts.")
    return SERIES_COLORS[index]


# --- Building blocks ------------------------------------------------------------

def draw_zero_axis(ax: Axes, value: float = 0.0):
    """Horizontal reference line that doesn't change the x-axis limits."""
    ax.axhline(value, color=BASELINE, linewidth=0.9, zorder=1)


def format_percent(ax: Axes, axis: str = "y", decimals: int | None = None):
    """Show fractional values (0.01) as percentages (1%)."""
    formatter = PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def add_figure_legend(fig, axes=None, handles=None, labels=None, columns: int | None = None):
    """One legend for the whole figure, below the plots, so it never covers data
    or collides with the figure title. Collects labeled lines from ``axes``
    (default: all axes) when no handles are given."""
    if handles is None:
        handles, labels, seen = [], [], set()
        for ax in (axes if axes is not None else fig.axes):
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in seen:
                    seen.add(label)
                    handles.append(handle)
                    labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncols=columns or min(len(handles), 6))


def set_figure_title(fig, title: str):
    fig.suptitle(title, x=0.01, ha="left")


def plot_gaussian(ax: Axes, mean: float = 0, standard_deviation: float = 1, x=None,
                  label: str | None = None, plot_kwargs=None):
    x = np.linspace(-0.5, 0.5, 1001) if x is None else x
    if label is None:
        label = rf"Gaussian ($\mu$={mean:.3f}, $\sigma$={standard_deviation:.3f})"
    ax.plot(x, norm.pdf(x, mean, standard_deviation), label=label, **(plot_kwargs or {}))


def plot_cauchy(ax: Axes, location: float = 0, scale: float = 1, x=None,
                label: str | None = None, plot_kwargs=None):
    x = np.linspace(-0.5, 0.5, 1001) if x is None else x
    if label is None:
        label = rf"Cauchy ($x_0$={location:.3f}, $\gamma$={scale:.3f})"
    ax.plot(x, cauchy.pdf(x, location, scale), label=label, **(plot_kwargs or {}))


# --- Statistical arbitrage charts -----------------------------------------------

def plot_spread_trading(log_prices: pd.DataFrame, standardized_spread: pd.Series,
                        entry_threshold: float, positions, filtered_spread=None, title: str = ""):
    """Log-prices, the standardized spread with entry thresholds, the positions
    taken, and cumulative profit (Feng & Palomar Figs. 10.6, 10.10)."""
    positions = np.asarray(positions, dtype=float)
    profit = np.cumsum(np.r_[0.0, positions[:-1] * np.diff(standardized_spread.to_numpy())])
    dates = standardized_spread.index

    fig, (price_ax, spread_ax, position_ax, profit_ax) = plt.subplots(
        4, 1, sharex=True, figsize=(10, 9), height_ratios=[1.2, 1.4, 0.6, 1.0])

    for index, symbol in enumerate(log_prices.columns):
        price_ax.plot(log_prices.index, log_prices[symbol], color=series_color(index), label=symbol)
    price_ax.set_title("Log-price")

    if filtered_spread is not None:
        # Wide and light underneath, so the raw spread stays visible where they coincide.
        spread_ax.plot(dates, filtered_spread, color=series_color(2), linewidth=3.0, alpha=0.45,
                       label="Kalman-filtered spread")
    spread_ax.plot(dates, standardized_spread, color=INK_PRIMARY, linewidth=1.0, label="Spread")
    for level in (entry_threshold, -entry_threshold):
        spread_ax.axhline(level, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))
    spread_ax.plot([], [], color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)),
                   label=f"Entry thresholds (±{entry_threshold:.2f})")
    draw_zero_axis(spread_ax)
    spread_ax.set_title("Spread (standard deviations from the mean)")

    position_ax.fill_between(dates, positions, step="post", color=series_color(0), alpha=0.25, linewidth=0)
    position_ax.step(dates, positions, where="post", color=series_color(0), linewidth=1.0)
    position_ax.set_yticks([-1, 0, 1], ["Short", "Flat", "Long"])
    position_ax.set_ylim(-1.3, 1.3)
    position_ax.set_title("Position in the spread")

    profit_ax.plot(dates, profit, color=series_color(0))
    draw_zero_axis(profit_ax)
    profit_ax.set_title("Cumulative profit (standard deviations)")

    add_figure_legend(fig)
    set_figure_title(fig, title)
    return fig


THRESHOLD_METHOD_COLORS = {"parametric": series_color(0), "nonparametric": series_color(1),
                           "empirical": series_color(2)}


def plot_threshold_curves(results: dict):
    """Trade frequency and total profit against the entry threshold for each
    method in ``results`` (name -> ThresholdResult), as in Figs. 10.11-10.12."""
    fig, (frequency_ax, profit_ax) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, result in results.items():
        color = THRESHOLD_METHOD_COLORS.get(name, INK_SECONDARY)
        frequency_ax.plot(result.thresholds, result.frequency, color=color,
                          label=f"{name.capitalize()} (best {result.best_threshold:.2f})")
        if result.empirical_frequency is not None:
            frequency_ax.plot(result.thresholds, result.empirical_frequency, linestyle="none",
                              marker="o", markersize=2.5, color=color, alpha=0.6,
                              label=f"{name.capitalize()} (before smoothing)")
        normalized = result.profit / max(result.profit.max(), 1e-12)
        profit_ax.plot(result.thresholds, normalized, color=color)
        best = int(np.argmin(np.abs(result.thresholds - result.best_threshold)))
        profit_ax.plot(result.thresholds[best], normalized[best], marker="o", markersize=8, color=color,
                       markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    frequency_ax.set_title("Estimated trades per day")
    profit_ax.set_title("Total profit (1 = best); dots mark each best threshold")
    profit_ax.set_ylim(bottom=0)
    for ax in (frequency_ax, profit_ax):
        ax.set_xlabel("Entry threshold (standard deviations)")
    add_figure_legend(fig)
    set_figure_title(fig, "Optimal entry threshold (Sec. 10.4.1)")
    return fig


def plot_equity_curves(results: dict, title: str = "Out-of-sample statistical arbitrage"):
    """Cumulative return of each backtest in ``results`` (name -> BacktestResult)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for index, (name, result) in enumerate(results.items()):
        cumulative = result.equity - 1
        ax.plot(cumulative.index, cumulative, color=series_color(index), label=name)
        ax.annotate(f"{cumulative.iloc[-1]:+.1%}", (cumulative.index[-1], cumulative.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5,
                    color=INK_SECONDARY)
    draw_zero_axis(ax)
    format_percent(ax)
    ax.set_title("Cumulative return after costs")
    add_figure_legend(fig)
    set_figure_title(fig, title)
    return fig
