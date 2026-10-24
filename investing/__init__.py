"""Stock analysis building up to statistical arbitrage.

Data and exploration:
  time_series.py      TimeSeries, rolling (time-varying) statistics
  stock.py            Stock: one CSV, prices and returns
  portfolio.py        Portfolio: stocks aligned by date, exploratory plots
  plotting.py         shared plot helpers

Statistical arbitrage (Feng & Palomar, "A Signal Processing Perspective on
Financial Engineering", Part III):
  pair_selection.py   rank candidate pairs                  (Sec. 10.2)
  cointegration.py    ADF, Engle-Granger, Johansen tests    (Sec. 10.3, 2.6)
  spread.py           Spread: weighted log-prices
  mean_reversion.py   Ornstein-Uhlenbeck, Kalman filter     (Sec. 10.4.2)
  threshold.py        optimal entry threshold               (Sec. 10.4.1)
  spread_builders.py  pairs, Johansen, factor spreads       (Sec. 10.5)
  backtest.py         walk-forward out-of-sample backtest
Helpers: regression.py (least squares), critical_values.py (MacKinnon tables).
"""

from .backtest import BacktestResult, TradingRule, compute_positions, performance_summary, walk_forward
from .cointegration import (
    augmented_dickey_fuller,
    engle_granger,
    engle_granger_strongest,
    johansen,
    mean_crossing_rate,
)
from .mean_reversion import KalmanSpreadFilter, OrnsteinUhlenbeck, fit_ornstein_uhlenbeck
from .pair_selection import principal_component_factor_model, rank_pairs
from .portfolio import Portfolio
from .spread import Spread, standardize
from .spread_builders import FactorBuilder, JohansenBuilder, PairsBuilder, SpreadCandidate
from .stock import Stock
from .threshold import optimal_threshold
from .time_series import TimeSeries
