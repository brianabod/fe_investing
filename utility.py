from datetime import datetime

import numpy as np
from matplotlib.pyplot import Axes
from scipy.stats import cauchy, norm


def draw_zero_axis(ax: Axes, xmin=datetime(1960, 1, 1), xmax=datetime(2030, 1, 1)):
    ax.hlines([0], xmin, xmax, color="black")


def plot_gaussian(
    ax: Axes,
    mean: float = 0,
    std: float = 1,
    x=np.linspace(-0.5, 0.5, 501),
    plot_kwargs={},
):
    label = f"Gaussian ($\mu$={round(mean, 3)}, $\sigma$={round(std, 3)})"
    ax.plot(x, norm.pdf(x, mean, std), label=label, **plot_kwargs)


def plot_cauchy(
    ax: Axes,
    loc: float = 0,
    scale: float = 1,
    x=np.linspace(-0.5, 0.5, 501),
    plot_kwargs={},
):
    label = f"Cauchy ($x_0$={round(loc, 3)}, $\gamma$={round(scale, 3)})"
    ax.plot(x, cauchy.pdf(x, loc, scale), label=label, **plot_kwargs)
