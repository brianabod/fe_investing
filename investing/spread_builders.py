"""Spread builders: find mean-reverting spreads in a window of log-prices.

Each builder is one way to go from pairs trading to statistical arbitrage:

- ``PairsBuilder``: preselect pairs (Sec. 10.2), then run Engle-Granger (10.3).
- ``JohansenBuilder``: Johansen/VECM on a basket of stocks (Sec. 10.5.1).
- ``FactorBuilder``: each stock minus its common-factor exposure (Sec. 10.5.2).

A builder is called with fitting-window data only, so nothing it computes can
see the trading window that follows. ``max_spreads`` is how many spreads it
returns at most; the backtest splits capital into that many equal slots.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .cointegration import augmented_dickey_fuller, engle_granger_strongest, johansen
from .mean_reversion import fit_ornstein_uhlenbeck
from .pair_selection import normalized_price_distance, principal_component_factor_model
from .spread import Spread


@dataclass
class SpreadCandidate:
    spread: Spread
    p_value: float          # smaller = stronger evidence of mean reversion
    half_life: float        # days
    details: dict = field(default_factory=dict)


def _half_life_in_range(half_life: float, minimum: float, maximum: float) -> bool:
    return np.isfinite(half_life) and minimum <= half_life <= maximum


@dataclass
class PairsBuilder:
    max_spreads: int = 3
    preselect_count: int | None = 10    # test only the pairs with the smallest price distance
    max_p_value: float = 0.05
    min_half_life: float = 1.0          # days
    max_half_life: float = 60.0         # days; slower spreads rarely revert in time
    min_hedge_ratio: float = 0.2        # a hedge ratio near 0 is a one-stock bet, not a pair
    symbols: list[str] | None = None

    def __call__(self, log_prices: pd.DataFrame) -> list[SpreadCandidate]:
        symbols = self.symbols or list(log_prices.columns)
        pairs = list(combinations(symbols, 2))
        if self.preselect_count and len(pairs) > self.preselect_count:
            prices = np.exp(log_prices[symbols])
            pairs = sorted(pairs, key=lambda pair: normalized_price_distance(prices, *pair))
            pairs = pairs[: self.preselect_count]

        candidates = []
        for pair in pairs:
            result = engle_granger_strongest(log_prices[list(pair)])
            hedge_ratio = float(result.hedge_ratios.iloc[0])
            if result.p_value > self.max_p_value or abs(hedge_ratio) < self.min_hedge_ratio:
                continue
            half_life = fit_ornstein_uhlenbeck(result.residuals).half_life
            if not _half_life_in_range(half_life, self.min_half_life, self.max_half_life):
                continue
            other = result.hedge_ratios.index[0]
            spread = Spread(result.weights, result.intercept, name=f"{result.dependent}/{other}")
            candidates.append(SpreadCandidate(spread, result.p_value, half_life,
                                              {"hedge_ratio": hedge_ratio, "statistic": result.statistic}))
        return sorted(candidates, key=lambda candidate: candidate.p_value)[: self.max_spreads]


@dataclass
class JohansenBuilder:
    max_spreads: int = 1
    lag_differences: int = 1
    confidence_level: str = "95%"
    max_half_life: float = 60.0
    symbols: list[str] | None = None

    def __call__(self, log_prices: pd.DataFrame) -> list[SpreadCandidate]:
        prices = log_prices[self.symbols] if self.symbols else log_prices
        result = johansen(prices, lag_differences=self.lag_differences)
        rank = result.rank(self.confidence_level)
        candidates = []
        for index in range(rank):
            weights = result.vectors.iloc[:, index]
            intercept = float((prices.to_numpy() @ weights.to_numpy()).mean())
            spread = Spread(weights, intercept, name=f"Johansen beta{index + 1}")
            half_life = fit_ornstein_uhlenbeck(spread.values(prices)).half_life
            if not _half_life_in_range(half_life, 1.0, self.max_half_life):
                continue
            # No p-value for Johansen: report critical value / statistic (below 1 = significant).
            margin = result.critical_value(index, self.confidence_level) / result.trace_statistics[index]
            candidates.append(SpreadCandidate(spread, float(margin), half_life,
                                              {"eigenvalue": float(result.eigenvalues[index]), "rank": rank}))
        # Strongest mean reversion first (Sec. 10.5.1).
        return sorted(candidates, key=lambda candidate: candidate.half_life)[: self.max_spreads]


@dataclass
class FactorBuilder:
    max_spreads: int = 3
    factor_count: int = 1
    max_p_value: float = 0.05
    max_half_life: float = 60.0
    symbols: list[str] | None = None

    def __call__(self, log_prices: pd.DataFrame) -> list[SpreadCandidate]:
        prices = log_prices[self.symbols] if self.symbols else log_prices
        factor_model = principal_component_factor_model(prices.diff().dropna(), self.factor_count)
        eigenvectors = factor_model.eigenvectors.to_numpy()   # factors f_t = V' r_t
        candidates = []
        for index, symbol in enumerate(prices.columns):
            # eps_i = r_i - pi_i' f = (e_i - V pi_i)' r  ->  weights on log-prices
            weights = -eigenvectors @ factor_model.loadings.loc[symbol].to_numpy()
            weights[index] += 1.0
            weights = pd.Series(weights, index=prices.columns)
            intercept = float((prices.to_numpy() @ weights.to_numpy()).mean())
            spread = Spread(weights, intercept, name=f"{symbol} vs factors")
            values = spread.values(prices)
            # Count the factors as extra series, like an Engle-Granger regression.
            test = augmented_dickey_fuller(values, series_count=min(1 + self.factor_count, 6))
            half_life = fit_ornstein_uhlenbeck(values).half_life
            if test.p_value > self.max_p_value or not _half_life_in_range(half_life, 1.0, self.max_half_life):
                continue
            candidates.append(SpreadCandidate(spread, test.p_value, half_life, {"statistic": test.statistic}))
        return sorted(candidates, key=lambda candidate: candidate.p_value)[: self.max_spreads]
