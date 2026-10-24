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

Price data is not stored in the repository. Download daily CSVs from Yahoo
Finance (columns `Date,Open,High,Low,Close,Adj Close,Volume`) into
`data/defense/`, e.g. `data/defense/LMT.csv`. The examples use LMT, NOC, GD,
BA and LDOS. The unit tests use synthetic data and need no downloads.

## Quick start

```bash
.venv/bin/python examples/explore_returns.py        # step 1: return distributions, rolling statistics
.venv/bin/python examples/cointegration_demo.py     # step 2: synthetic cointegrated pair + tests
.venv/bin/python examples/statistical_arbitrage.py  # steps 3-4: selection, tests, thresholds, backtest
.venv/bin/python -m unittest discover tests         # unit tests
```

The examples can be run from any directory. Each one adds the repo folder to
Python's search path so `import investing` works without installing anything.

## Layout

```
investing/            the package
examples/             runnable scripts, one per step above
tests/                unit tests (synthetic data only)
data/defense/         daily price CSVs (not in git)
requirements.txt
pyproject.toml        makes the repo pip-installable
```

### `investing/` modules

| Module | Contents | Book section |
|---|---|---|
| `time_series.py` | `TimeSeries` (a DataFrame with a date column, oldest first), `mean`, `volatility`, `function_of_time` (sliding-window statistics) | — |
| `stock.py` | `Stock`: one CSV, adjusted prices, close-to-close `Return` and `Log-Return`, distribution plots | — |
| `portfolio.py` | `Portfolio`: stocks matched by date. `prices()` and `log_prices()` feed the statistical arbitrage code; also exploratory plots | — |
| `plotting.py` | Shared plots: Gaussian/Cauchy curves, spread trading chart, threshold curves, equity curves | Figs. 10.6, 10.10-10.12 |
| `market_simulation.py` | `generate_cointegrated_time_series`: synthetic pairs with a known hedge ratio | 10.1 |
| `regression.py` | `least_squares` and trend terms shared by the tests and models | — |
| `critical_values.py` | MacKinnon critical-value and p-value tables | — |
| `cointegration.py` | `augmented_dickey_fuller`, `engle_granger`, `engle_granger_strongest`, `johansen`, `mean_crossing_rate` | 10.3, 2.6 |
| `pair_selection.py` | `normalized_price_distance`, principal-component factor model, `rank_pairs` | 10.2 |
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

portfolio = Portfolio.from_directory(["LMT", "NOC", "GD", "BA", "LDOS"], "data/defense")
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

## Results on the defense stocks (2006–2024, 5 basis point costs)

| Strategy | Annual return | Sharpe ratio | Max drawdown |
|---|---|---|---|
| Pairs (Engle-Granger) | +0.2% | 0.07 | −19% |
| Johansen basket (LMT/NOC/GD) | −0.5% | −0.10 | −24% |
| Factor residuals (1 principal component) | −0.1% | −0.13 | −3% |

The strategy roughly breaks even out-of-sample. Spreads fitted and traded on
the same year look highly profitable, which shows how much in-sample results
overstate performance. Five stocks give only 10 possible pairs, so the
strategy rarely has a strong spread to trade. A larger universe is the
obvious next step.

## Data

The results above use Yahoo Finance daily data for BA, GD, LDOS, LMT and NOC
through 2024-02-02. `data/` is ignored by git. To add stocks, save CSVs in
the same format and load them with
`Portfolio.from_directory(symbols, directory)`.

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
