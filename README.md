# Iron Investing

Iron (Fe) Investing is a financial engineering tool that implements
**statistical arbitrage** (mean-reversion) methods. The approach follows
Part III of Feng & Palomar, *A Signal Processing Perspective on Financial
Engineering* (2016).

Author: Brian Abod

## Overview

Pairs trading trades on the relationship between two stocks rather than on
either stock alone. **Statistical arbitrage** extends this to groups of more
than two stocks.

Statistical arbitrage relies on **cointegration**. Two cointegrated series
share a common trend. This differs from covariance, which only measures
whether two series move together at each time step. Think of a dog and its
owner on a walk. We can't predict where either one goes next, but because
the dog is on a leash, they are always near each other. They are
cointegrated.

If the common trend between two stocks is consistent enough, the spread
between them is mean-reverting. That means we can set thresholds: buy the
spread when it is low, sell it when it is high, and expect it to return to
its mean. The strategy trades on the noise around the shared trend.

The trend may not have the same size in both stocks. The scaling factor
**gamma** (the hedge ratio) is estimated by regression: least squares in the
Engle-Granger test. A Kalman filter can additionally clean up the noisy
spread before trading.

The strategy isn't foolproof. The main risk is that the cointegration doesn't
last. If the spread's statistics change, a model that worked in one period
can fail in the next. That is why everything here is tested out-of-sample:
fit on one window, trade the next, and repeat.

The code builds up in four steps:

1. **Exploration.** Returns are fat-tailed and non-stationary: their mean and
   volatility drift over time.
2. **Cointegration.** Single stocks wander like random walks, but some
   weighted combinations of stocks (spreads) keep returning to a long-run
   level.
3. **Pairs trading.** Find two stocks whose spread is stationary. Sell it
   when it's high and buy it when it's low.
4. **Statistical arbitrage.** Generalize to baskets of stocks (Johansen) and
   to stock-minus-market-factor spreads. Then test everything out-of-sample
   with trading costs.

## Setup

Requires Python 3.11+ (developed on 3.13).

```bash
git clone https://github.com/brianabod/fe_investing.git
cd fe_investing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To use the package from another project:
`pip install git+https://github.com/brianabod/fe_investing.git`, then
`import investing`.

Price data is not stored in the repository; see [Data](#data). The unit tests
use synthetic data and need no downloads.

## Quick start

```bash
.venv/bin/python examples/explore_returns.py        # step 1: return distributions, rolling statistics
.venv/bin/python examples/cointegration_demo.py     # step 2: synthetic cointegrated pair + tests
.venv/bin/python examples/statistical_arbitrage.py  # steps 3-4: selection, tests, thresholds, backtest
.venv/bin/python -m unittest discover tests         # unit tests
```

By default the examples load every CSV in `data/`. Use `--symbols` to pick a
subset, or `--data-directory` to point at another folder:

```bash
.venv/bin/python examples/statistical_arbitrage.py --symbols AAA BBB CCC
```

The examples can be run from any directory. Each one adds the repo folder to
Python's search path so `import investing` works without installing anything.

## Layout

```
investing/            the package
examples/             runnable scripts, one per step above, plus shared command-line options
tests/                unit tests (synthetic data only)
data/                 daily price CSVs, one per stock (not in git)
requirements.txt
pyproject.toml        makes the repo pip-installable
```

### `investing/` modules

| Module | Contents | Book section |
|---|---|---|
| `time_series.py` | `TimeSeries` (a DataFrame with a date column, oldest first), `mean`, `volatility`, `function_of_time` (sliding-window statistics) | — |
| `stock.py` | `Stock`: one CSV, adjusted prices, close-to-close `Return` and `Log-Return`, distribution plots | — |
| `portfolio.py` | `Portfolio`: stocks matched by date. `prices()` and `log_prices()` feed the statistical arbitrage code; also exploratory plots | — |
| `plotting.py` | Shared chart style (`apply_style`, color-blind-safe `series_color`), Gaussian/Cauchy curves, spread trading chart, threshold curves, equity curves | Figs. 10.6, 10.10-10.12 |
| `market_simulation.py` | `generate_cointegrated_time_series`: synthetic pairs with a known hedge ratio | 10.1 |
| `regression.py` | `least_squares` and trend terms shared by the tests and models | — |
| `critical_values.py` | MacKinnon critical-value and p-value tables | — |
| `cointegration.py` | `augmented_dickey_fuller`, `engle_granger`, `engle_granger_strongest`, `johansen`, `mean_crossing_rate` | 10.3, 2.6 |
| `pair_selection.py` | `normalized_price_distance`, principal-component factor model, `rank_pairs`, `closest_basket` | 10.2 |
| `spread.py` | `Spread` (weighted log-prices), `standardize` | 10.3 |
| `mean_reversion.py` | `OrnsteinUhlenbeck` (half-life, reversion speed), `KalmanSpreadFilter` (fitted with EM) | 10.4.2 |
| `threshold.py` | Optimal entry threshold: parametric, smoothed nonparametric, or by replaying the rule | 10.4.1 |
| `spread_builders.py` | `PairsBuilder`, `JohansenBuilder`, `FactorBuilder`: find spreads in a window of data | 10.5 |
| `backtest.py` | `TradingRule`, `compute_positions`, `walk_forward`, `performance_summary` | — |

The statistical tests are written with numpy and scipy only. Their
statistics and p-values match statsmodels to within rounding error.

## Example

```python
from investing import PairsBuilder, Portfolio, TradingRule, performance_summary, walk_forward

portfolio = Portfolio.from_directory("data")  # every CSV in data/, or pass symbols=[...]
result = walk_forward(
    portfolio.prices(),
    PairsBuilder(max_spreads=3),
    fitting_days=252,
    trading_days=126,
    rule=TradingRule(threshold_method="empirical"),
    cost_basis_points=5,
)
print(performance_summary(result.returns))
print(result.summary())
```

### Trading rule

A spread is a fixed combination of log-prices,
`spread = weights · log_prices − intercept`. For a pair, the weights are
`[1, −hedge_ratio]`. The spread is measured in standard deviations from its
mean, using the mean and standard deviation from the fitting window.

- **Enter:** buy the spread at `−entry_threshold` and sell it at
  `+entry_threshold`.
- **Exit:** close when the spread crosses its mean.
- **Stop-loss:** close when the spread moves `stop_loss` (default 4) standard
  deviations away.
- **End of window:** close everything when the trading window ends.

### Backtest

`walk_forward` fits on a window (default 252 days) and then trades the
following window (default 126 days) using only past data. It then moves
forward and repeats.

- **Timing:** positions are set at the close and earn the next day's return.
- **Costs:** charged per dollar of gross exposure traded.
- **Capital:** split equally across the builder's `max_spreads` slots. An
  unused slot sits in cash.

## What to expect

On a small universe of a handful of related stocks, the strategy roughly
breaks even out-of-sample after costs. Spreads fitted and traded on the same
year look highly profitable, which shows how much in-sample results overstate
performance. A few stocks give only a few possible pairs, so the strategy
rarely has a strong spread to trade. A larger universe is the obvious next
step.

## Data

Price data lives in the local `data/` folder, which git ignores. Save one
daily CSV per stock, named `<SYMBOL>.csv`, in Yahoo Finance's format:

```
Date,Open,High,Low,Close,Adj Close,Volume
```

`Portfolio.from_directory("data")` loads every CSV it finds, sorted by name.
It skips files with less than a year of valid prices (with a warning) and
keeps only the dates all stocks share. Pass `symbols=[...]` to load specific
files.

## References

D. Palomar, Y. Feng. *A Signal Processing Perspective on Financial
Engineering.* Foundations and Trends in Signal Processing, 2016.
https://palomar.home.ece.ust.hk/papers/2016/Feng&Palomar-FnT2016.pdf

## Conventions

- **Names:** use full words for files, functions and variables
  (`normalized_price_distance`, not `npd`). Exceptions are kept to standard
  idioms:
  - library aliases: `np`, `pd`, `plt`;
  - matplotlib objects: `fig`, `ax`, `kwargs`;
  - loop indices: `t` for time;
  - the book's symbols, in comments and docstrings only.
- **Files:** lowercase with underscores. Classes are CamelCase.
- **Dates:** data is ordered oldest first. A rolling statistic is labeled
  with the newest date in its window, so it uses only past data.
- **Prices and returns:** prices are adjusted closes, and returns are
  close-to-close. Multi-stock tables keep only dates every stock traded on.
- **Window lengths:** counted in trading days, with 252 per year.
- **Charts:** call `investing.plotting.apply_style()` before plotting.
  - Stocks take palette colors in portfolio order and keep them on every chart.
  - Each figure has one shared legend below the plots, and every panel has a
    left-aligned title.
  - Returns are shown as percentages.
