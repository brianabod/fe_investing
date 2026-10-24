"""Walk-forward backtest of the mean-reversion rule (Feng & Palomar, Sec. 10.4).

The book's Fig. 10.6 example is in-sample and ignores costs, and the authors
note that real results need out-of-sample testing with costs. So the
backtest repeats this loop:

  1. Fitting window (e.g. 252 days): find spreads with a builder, estimate
     their mean and standard deviation, optionally fit a Kalman filter, and
     choose the entry threshold.
  2. Trading window (e.g. 126 days): trade those fixed spreads using only
     information up to each day, then close everything at the end.
  3. Move forward by one trading window and repeat.

Positions are decided at the close of day t and earn the return of day t+1.
Costs are charged on every change in position.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .mean_reversion import KalmanSpreadFilter
from .spread import Spread, standardize
from .spread_builders import SpreadCandidate
from .threshold import optimal_threshold

TRADING_DAYS_PER_YEAR = 252


@dataclass
class TradingRule:
    entry_threshold: float | None = None    # standard deviations; None = optimize on the fitting window
    exit_threshold: float = 0.0             # unwind when the spread crosses mean +/- this
    stop_loss: float | None = 4.0           # close if the spread moves this far from the mean
    max_holding_days: int | None = None     # force an exit after this many days
    threshold_method: str = "empirical"     # see threshold.py
    use_kalman_filter: bool = False         # trade the Kalman-filtered spread (10.34)


def compute_positions(standardized_spread, entry_threshold: float, exit_threshold: float = 0.0,
                      stop_loss: float | None = None, max_holding_days: int | None = None) -> np.ndarray:
    """Position in the spread after each observation: +1 long, -1 short, 0 flat.

    After a stop-loss, wait until the spread comes back inside the entry band
    before trading again, so the same divergence isn't re-entered.
    """
    values = np.asarray(standardized_spread, dtype=float)
    positions = np.zeros(len(values))
    position, days_held, locked = 0, 0, False
    for t, value in enumerate(values):
        if locked and abs(value) < entry_threshold:
            locked = False
        if position != 0:
            days_held += 1
            stopped = stop_loss is not None and abs(value) >= stop_loss
            timed_out = max_holding_days is not None and days_held >= max_holding_days
            reverted = (position == 1 and value >= -exit_threshold) or (position == -1 and value <= exit_threshold)
            if stopped or timed_out or reverted:
                position, days_held = 0, 0
                locked = stopped
        if position == 0 and not locked and np.isfinite(value):
            inside_stop = stop_loss is None or abs(value) < stop_loss
            if value >= entry_threshold and inside_stop:
                position, days_held = -1, 0
            elif value <= -entry_threshold and inside_stop:
                position, days_held = 1, 0
        positions[t] = position
    return positions


@dataclass
class SpreadTrade:
    """One spread traded during one trading window."""
    window_start: pd.Timestamp
    spread: Spread
    candidate: SpreadCandidate
    entry_threshold: float
    mean: float
    standard_deviation: float
    standardized_spread: pd.Series   # signal during the trading window
    positions: pd.Series
    returns: pd.Series               # daily return on the capital in this spread, after costs
    trade_count: int


@dataclass
class BacktestResult:
    returns: pd.Series               # daily strategy return on total capital
    trades: list[SpreadTrade] = field(default_factory=list)

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    def summary(self) -> pd.DataFrame:
        """One row per spread traded in each window."""
        return pd.DataFrame([{
            "window": trade.window_start.date(),
            "spread": trade.spread.label(),
            "p_value": trade.candidate.p_value,
            "half_life": trade.candidate.half_life,
            "entry_threshold": trade.entry_threshold,
            "trades": trade.trade_count,
            "return": float((1 + trade.returns).prod() - 1),
        } for trade in self.trades])


def performance_summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict[str, float]:
    returns = returns.fillna(0.0)
    equity = (1 + returns).cumprod()
    years = len(returns) / periods_per_year
    volatility = returns.std()
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
        "annual_volatility": float(volatility * np.sqrt(periods_per_year)),
        "sharpe_ratio": float(returns.mean() / volatility * np.sqrt(periods_per_year)) if volatility > 0 else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "days_in_market": float((returns != 0).mean()),
    }


def _trade_spread(candidate: SpreadCandidate, fitting_log_prices: pd.DataFrame,
                  trading_log_prices: pd.DataFrame, trading_returns: pd.DataFrame,
                  rule: TradingRule, cost_per_dollar: float) -> SpreadTrade:
    spread = candidate.spread.gross_normalized()
    fitting_values = spread.values(fitting_log_prices)
    trading_values = spread.values(trading_log_prices)

    if rule.use_kalman_filter:
        kalman_filter = KalmanSpreadFilter.fit(fitting_values.to_numpy())
        filtered = kalman_filter.filter(pd.concat([fitting_values, trading_values]))
        fitting_values = filtered.iloc[: len(fitting_values)]
        trading_values = filtered.iloc[len(fitting_values):]

    mean, standard_deviation = float(fitting_values.mean()), float(fitting_values.std())
    fitting_signal = standardize(fitting_values, mean, standard_deviation)
    trading_signal = standardize(trading_values, mean, standard_deviation)
    entry_threshold = rule.entry_threshold
    if entry_threshold is None:
        entry_threshold = optimal_threshold(fitting_signal, rule.threshold_method).best_threshold

    positions = compute_positions(trading_signal, entry_threshold, rule.exit_threshold,
                                  rule.stop_loss, rule.max_holding_days)
    positions[-1] = 0.0  # close at the end of the trading window
    positions = pd.Series(positions, index=trading_log_prices.index, name="position")

    held = positions.shift(1, fill_value=0.0)
    turnover = positions.diff().fillna(positions).abs()
    daily_returns = held * spread.returns(trading_returns) - cost_per_dollar * turnover
    trade_count = int(((positions != 0) & (held == 0)).sum())
    return SpreadTrade(trading_log_prices.index[0], spread, candidate, float(entry_threshold),
                       mean, standard_deviation, trading_signal, positions, daily_returns, trade_count)


def walk_forward(prices: pd.DataFrame, builder, fitting_days: int = TRADING_DAYS_PER_YEAR,
                 trading_days: int = TRADING_DAYS_PER_YEAR // 2, rule: TradingRule | None = None,
                 cost_basis_points: float = 5.0) -> BacktestResult:
    """Run the fit/trade loop over the whole price history.

    ``prices`` is a date-indexed table, e.g. ``Portfolio.prices()``.
    ``cost_basis_points`` is charged per $1 of gross exposure traded, so a
    round trip on a $1 spread costs about twice that.

    Capital is split into ``builder.max_spreads`` equal slots. A slot with no
    spread to trade sits in cash, so leverage doesn't jump around.
    """
    rule = rule or TradingRule()
    cost_per_dollar = cost_basis_points / 1e4
    log_prices = np.log(prices)
    simple_returns = prices.pct_change()
    slot_count = max(int(builder.max_spreads), 1)

    strategy_returns = pd.Series(0.0, index=prices.index[fitting_days:])
    trades: list[SpreadTrade] = []
    for start in range(0, len(prices) - fitting_days, trading_days):
        fitting = slice(start, start + fitting_days)
        trading = slice(start + fitting_days, min(start + fitting_days + trading_days, len(prices)))
        fitting_log_prices, trading_log_prices = log_prices.iloc[fitting], log_prices.iloc[trading]
        if len(trading_log_prices) < 2:
            break
        for candidate in builder(fitting_log_prices):
            trade = _trade_spread(candidate, fitting_log_prices, trading_log_prices,
                                  simple_returns.iloc[trading], rule, cost_per_dollar)
            trades.append(trade)
            strategy_returns.loc[trade.returns.index] += trade.returns / slot_count
    return BacktestResult(strategy_returns, trades)
