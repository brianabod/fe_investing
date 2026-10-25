"""Stock analysis building up to statistical arbitrage.

Data and exploration:
  time_series.py      TimeSeries, rolling (time-varying) statistics
  stock.py            Stock: one CSV, prices and returns
  portfolio.py        Portfolio: stocks aligned by date, exploratory plots
  plotting.py         shared plot helpers
  price_download.py   download daily prices from Yahoo Finance

Statistical arbitrage (Feng & Palomar, "A Signal Processing Perspective on
Financial Engineering", Part III):
  pair_selection.py   rank candidate pairs                  (Sec. 10.2)
  cointegration.py    stationarity, pair and basket tests   (Sec. 10.3, 2.6)
  spread.py           Spread: weighted log-prices
  mean_reversion.py   reversion speed, Kalman filter        (Sec. 10.4.2)
  threshold.py        optimal entry threshold               (Sec. 10.4.1)
  spread_builders.py  pair, basket and factor spreads       (Sec. 10.5)
  backtest.py         walk-forward out-of-sample backtest
Helpers: regression.py (least squares), critical_values.py (test tables).
"""

from .backtest import BacktestResult, TradingRule, compute_positions, performance_summary, walk_forward
from .cointegration import (
    stationarity_test,
    pair_cointegration_test,
    pair_cointegration_test,
    basket_cointegration_test,
    mean_crossing_rate,
)
from .mean_reversion import KalmanSpreadFilter, MeanReversionModel, fit_mean_reversion
from .pair_selection import principal_component_factor_model, rank_pairs
from .portfolio import Portfolio
from .spread import Spread, standardize
from .spread_builders import FactorBuilder, BasketBuilder, PairsBuilder, SpreadCandidate
from .stock import Stock
from .threshold import optimal_threshold
from .time_series import TimeSeries
