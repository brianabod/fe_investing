"""A spread: a fixed weighted combination of log-prices.

    spread_t = weights . log_prices_t - intercept

Examples: weights [1, -gamma] for a pair (10.14), a Johansen vector beta
(10.36), or a stock minus its factor exposure (10.37). Holding one unit of the
spread means holding ``weights[i]`` dollars of stock i, rebalanced daily
(Remark 10.3).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Spread:
    weights: pd.Series      # weight per symbol on log-prices
    intercept: float = 0.0
    name: str = ""

    @property
    def symbols(self) -> list[str]:
        return list(self.weights.index)

    def values(self, log_prices: pd.DataFrame) -> pd.Series:
        """The spread over time for the given date-indexed log-prices."""
        combined = log_prices[self.symbols].to_numpy() @ self.weights.to_numpy() - self.intercept
        return pd.Series(combined, index=log_prices.index, name=self.name or "spread")

    def returns(self, simple_returns: pd.DataFrame) -> pd.Series:
        """Daily return of holding one unit of the spread."""
        combined = simple_returns[self.symbols].to_numpy() @ self.weights.to_numpy()
        return pd.Series(combined, index=simple_returns.index, name=self.name or "spread")

    def gross_normalized(self) -> "Spread":
        """Rescale so one unit of spread = $1 of gross exposure (sum |weights| = 1).
        This makes profit read as a return on the capital at work."""
        scale = float(np.abs(self.weights).sum())
        return Spread(self.weights / scale, self.intercept / scale, self.name)

    def label(self) -> str:
        if self.name:
            return self.name
        return " ".join(f"{weight:+.2f}*{symbol}" for symbol, weight in self.weights.items())


def standardize(values: pd.Series, mean: float | None = None,
                standard_deviation: float | None = None) -> pd.Series:
    """Express a spread in standard deviations from its mean. Pass the mean
    and standard deviation from a fitting window to avoid using future data."""
    mean = values.mean() if mean is None else mean
    standard_deviation = values.std() if standard_deviation is None else standard_deviation
    return (values - mean) / standard_deviation
