"""Optimal entry threshold (Feng & Palomar, Sec. 10.4.1).

Rule: on the standardized spread, buy the spread when it falls to
-threshold, sell it when it rises to +threshold, and unwind when it crosses
the mean. Each round trip earns about ``threshold``, so

    total profit(threshold) = threshold * T * trade_frequency(threshold).

A higher threshold earns more per trade but trades less often. Three ways to
estimate the trade frequency:

- "parametric": spread modeled as i.i.d. Gaussian, frequency = 1 - Phi(threshold).
- "nonparametric": the share of days beyond the threshold (10.23), smoothed
  with a difference penalty (10.24)-(10.27). The smoothing strength (lambda)
  is chosen by generalized cross-validation.
- "empirical": replay the rule on the sample path and add up each round
  trip's profit. Real spreads stay beyond the threshold for many days in a
  row, so counting days overstates the number of trades. Replaying the rule
  counts real round trips. The backtest uses this method by default.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

DEFAULT_THRESHOLDS = np.round(np.arange(0.1, 3.01, 0.05), 4)


@dataclass
class ThresholdResult:
    thresholds: np.ndarray            # candidate entry thresholds, in standard deviations
    frequency: np.ndarray             # estimated trades per day at each threshold
    profit: np.ndarray                # estimated total profit, in standard deviations
    best_threshold: float
    method: str
    empirical_frequency: np.ndarray | None = None   # before smoothing (nonparametric only)
    smoothing_strength: float | None = None         # lambda (nonparametric only)


def parametric_threshold(standardized_spread, thresholds=DEFAULT_THRESHOLDS) -> ThresholdResult:
    """White-Gaussian-noise model (Example 10.5)."""
    values = np.asarray(standardized_spread, dtype=float)
    frequency = 1 - norm.cdf((thresholds - values.mean()) / values.std())
    profit = thresholds * len(values) * frequency
    return ThresholdResult(thresholds, frequency, profit,
                           float(thresholds[np.argmax(profit)]), "parametric")


def smooth_frequency(frequency: np.ndarray, smoothing_strength: float) -> np.ndarray:
    """Solve  min ||f_raw - f||^2 + lambda ||D f||^2  ->  f = (I + lambda D'D)^-1 f_raw  (10.27)."""
    size = len(frequency)
    difference_matrix = np.diff(np.eye(size), axis=0)  # first-difference matrix D (10.26)
    return np.linalg.solve(np.eye(size) + smoothing_strength * difference_matrix.T @ difference_matrix,
                           frequency)


def _choose_smoothing_strength(frequency: np.ndarray, candidates=np.logspace(-2, 3, 60)) -> float:
    """Pick lambda by generalized cross-validation."""
    size = len(frequency)
    difference_matrix = np.diff(np.eye(size), axis=0)
    best, best_score = candidates[0], np.inf
    for strength in candidates:
        smoother = np.linalg.inv(np.eye(size) + strength * difference_matrix.T @ difference_matrix)
        residual = frequency - smoother @ frequency
        score = size * (residual @ residual) / (size - np.trace(smoother)) ** 2
        if score < best_score:
            best, best_score = strength, score
    return float(best)


def nonparametric_threshold(standardized_spread, thresholds=DEFAULT_THRESHOLDS,
                            smoothing_strength: float | None = None) -> ThresholdResult:
    """Empirical trading frequency (10.23), regularized (10.27).

    Both tails are used, which is the symmetric version of (10.23):
    f = 0.5 * share of days with |spread| beyond the threshold.
    """
    values = np.asarray(standardized_spread, dtype=float)
    raw = 0.5 * np.array([np.mean(values > level) + np.mean(values < -level) for level in thresholds])
    if smoothing_strength is None:
        smoothing_strength = _choose_smoothing_strength(raw)
    frequency = np.clip(smooth_frequency(raw, smoothing_strength), 0, None)
    profit = thresholds * len(values) * frequency
    return ThresholdResult(thresholds, frequency, profit, float(thresholds[np.argmax(profit)]),
                           "nonparametric", raw, smoothing_strength)


def simulate_round_trips(standardized_spread, entry_threshold: float, exit_threshold: float = 0.0) -> list[float]:
    """Replay the threshold rule on a path and return each completed round
    trip's profit, in standard deviations."""
    position, entry_value, profits = 0, 0.0, []
    for value in np.asarray(standardized_spread, dtype=float):
        if position == 0:
            if value >= entry_threshold:
                position, entry_value = -1, value
            elif value <= -entry_threshold:
                position, entry_value = 1, value
        elif position == 1 and value >= -exit_threshold:
            profits.append(value - entry_value)
            position = 0
        elif position == -1 and value <= exit_threshold:
            profits.append(entry_value - value)
            position = 0
    return profits


def empirical_threshold(standardized_spread, thresholds=DEFAULT_THRESHOLDS,
                        minimum_trades: int = 3) -> ThresholdResult:
    """Pick the threshold by replaying the rule on the fitting window.

    ``minimum_trades`` stops the optimizer from choosing a huge threshold that
    happened to fire once.
    """
    values = np.asarray(standardized_spread, dtype=float)
    round_trips = [simulate_round_trips(values, level) for level in thresholds]
    trade_counts = np.array([len(trips) for trips in round_trips], dtype=float)
    profit = np.array([sum(trips) for trips in round_trips], dtype=float)
    enough_trades = trade_counts >= minimum_trades
    if not enough_trades.any():
        enough_trades[:] = True
    best = float(thresholds[np.argmax(np.where(enough_trades, profit, -np.inf))])
    return ThresholdResult(thresholds, trade_counts / len(values), profit, best, "empirical")


THRESHOLD_METHODS = {
    "parametric": parametric_threshold,
    "nonparametric": nonparametric_threshold,
    "empirical": empirical_threshold,
}


def optimal_threshold(standardized_spread, method: str = "empirical",
                      thresholds=DEFAULT_THRESHOLDS, **kwargs) -> ThresholdResult:
    return THRESHOLD_METHODS[method](standardized_spread, thresholds=thresholds, **kwargs)
