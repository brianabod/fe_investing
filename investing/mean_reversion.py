"""Models of how a spread reverts to its mean (Feng & Palomar, Sec. 10.4.2).

- ``OrnsteinUhlenbeck``: the discretized Ornstein-Uhlenbeck / AR(1) model
      x_{t+1} = A + B x_t + C eps_{t+1}            (10.30)-(10.33)
  Gives the speed of mean reversion and the half-life.
- ``KalmanSpreadFilter``: the observed spread is a noisy reading of a hidden
  Ornstein-Uhlenbeck spread,
      z_t = x_t + D w_t                            (10.34)
  Its parameters are fitted with Expectation-Maximization, and ``filter``
  returns the real-time estimate of the hidden spread.

Field names map to the book's symbols: intercept = A, persistence = B,
state_noise = C, observation_noise = D.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regression import least_squares


@dataclass
class OrnsteinUhlenbeck:
    intercept: float        # A
    persistence: float      # B, between 0 and 1 for a mean-reverting spread
    state_noise: float      # C
    time_step: float = 1.0  # tau; 1.0 means "per trading day"

    @property
    def mean(self) -> float:
        """Long-run mean mu = A / (1 - B)."""
        return self.intercept / (1 - self.persistence)

    @property
    def reversion_speed(self) -> float:
        """Strength of mean reversion kappa = (1 - B) / tau."""
        return (1 - self.persistence) / self.time_step

    @property
    def volatility(self) -> float:
        """Conditional volatility sigma = C / sqrt(tau)."""
        return self.state_noise / np.sqrt(self.time_step)

    @property
    def long_run_standard_deviation(self) -> float:
        """Stationary standard deviation of the AR(1): C / sqrt(1 - B^2)."""
        return self.state_noise / np.sqrt(max(1 - self.persistence ** 2, 1e-12))

    @property
    def half_life(self) -> float:
        """Steps for a deviation to halve. Infinite if the spread doesn't revert."""
        if not 0 < self.persistence < 1:
            return np.inf
        return -np.log(2) / np.log(self.persistence)

    def reversion_time(self, deviations: float) -> float:
        """Most likely time to return to the mean from ``deviations`` long-run
        standard deviations away, eq. (10.29), in units of the time step."""
        if self.reversion_speed <= 0:
            return np.inf
        squared = deviations ** 2
        return np.log(1 + 0.5 * (np.sqrt((squared - 3) ** 2 + 4 * squared) + squared - 3)) / self.reversion_speed


def fit_ornstein_uhlenbeck(spread, time_step: float = 1.0) -> OrnsteinUhlenbeck:
    """Least-squares AR(1) fit of the spread, eq. (10.33)."""
    values = np.asarray(spread, dtype=float)
    regressors = np.column_stack([np.ones(len(values) - 1), values[:-1]])
    fit = least_squares(values[1:], regressors)
    intercept, persistence = fit.coefficients
    state_noise = float(np.std(fit.residuals, ddof=2))
    return OrnsteinUhlenbeck(float(intercept), float(persistence), state_noise, time_step)


@dataclass
class KalmanSpreadFilter:
    """Hidden spread x_t (AR(1)) observed with noise:
        x_{t+1} = A + B x_t + C eps_{t+1}   (10.33)
        z_t     = x_t + D w_t               (10.34)
    """
    intercept: float            # A
    persistence: float          # B
    state_noise: float          # C
    observation_noise: float    # D
    initial_state: float = 0.0
    initial_variance: float = 1.0
    log_likelihood: float = np.nan

    @property
    def hidden_spread_model(self) -> OrnsteinUhlenbeck:
        return OrnsteinUhlenbeck(self.intercept, self.persistence, self.state_noise)

    def _run_filter(self, observations: np.ndarray):
        count = len(observations)
        state_variance_noise = self.state_noise ** 2
        observation_variance = self.observation_noise ** 2
        predicted_state, predicted_variance = np.empty(count), np.empty(count)
        filtered_state, filtered_variance = np.empty(count), np.empty(count)
        state, variance, log_likelihood = self.initial_state, self.initial_variance, 0.0
        for t in range(count):
            if t > 0:
                state = self.intercept + self.persistence * state
                variance = self.persistence ** 2 * variance + state_variance_noise
            predicted_state[t], predicted_variance[t] = state, variance
            innovation_variance = variance + observation_variance
            gain = variance / innovation_variance
            innovation = observations[t] - state
            log_likelihood += -0.5 * (np.log(2 * np.pi * innovation_variance)
                                      + innovation ** 2 / innovation_variance)
            state = state + gain * innovation
            variance = (1 - gain) * variance
            filtered_state[t], filtered_variance[t] = state, variance
        return predicted_state, predicted_variance, filtered_state, filtered_variance, log_likelihood

    def filter(self, spread):
        """Real-time estimate E[x_t | z_1..z_t]. Uses only past data, so it's safe to trade on."""
        filtered = self._run_filter(np.asarray(spread, dtype=float))[2]
        if isinstance(spread, pd.Series):
            return pd.Series(filtered, index=spread.index, name="filtered")
        return filtered

    def smooth(self, spread):
        """Rauch-Tung-Striebel smoother E[x_t | z_1..z_T]. Uses future data,
        so use it for fitting and analysis only, never for trading signals."""
        observations = np.asarray(spread, dtype=float)
        predicted_state, predicted_variance, filtered_state, filtered_variance, log_likelihood = \
            self._run_filter(observations)
        count = len(observations)
        smoothed_state, smoothed_variance = filtered_state.copy(), filtered_variance.copy()
        smoother_gain = np.zeros(count)
        for t in range(count - 2, -1, -1):
            smoother_gain[t] = filtered_variance[t] * self.persistence / predicted_variance[t + 1]
            smoothed_state[t] = filtered_state[t] + smoother_gain[t] * (smoothed_state[t + 1] - predicted_state[t + 1])
            smoothed_variance[t] = filtered_variance[t] + smoother_gain[t] ** 2 * (
                smoothed_variance[t + 1] - predicted_variance[t + 1])
        return smoothed_state, smoothed_variance, smoother_gain, log_likelihood

    @classmethod
    def fit(cls, spread, max_iterations: int = 200, tolerance: float = 1e-7) -> "KalmanSpreadFilter":
        """Expectation-Maximization estimate of A, B, C, D."""
        observations = np.asarray(spread, dtype=float)
        start = fit_ornstein_uhlenbeck(observations)
        # Start by splitting the AR(1) residual variance between state and observation noise.
        model = cls(intercept=start.intercept,
                    persistence=float(np.clip(start.persistence, 0.0, 0.999)),
                    state_noise=start.state_noise / np.sqrt(2),
                    observation_noise=start.state_noise / np.sqrt(2),
                    initial_state=observations[0], initial_variance=np.var(observations))
        previous_log_likelihood = -np.inf
        transitions = len(observations) - 1
        for _ in range(max_iterations):
            state, variance, smoother_gain, log_likelihood = model.smooth(observations)
            if log_likelihood - previous_log_likelihood < tolerance * max(1.0, abs(log_likelihood)):
                break
            previous_log_likelihood = log_likelihood
            # Expected sufficient statistics (Shumway & Stoffer).
            second_moment = variance + state ** 2
            cross_moment = smoother_gain[:-1] * variance[1:] + state[1:] * state[:-1]  # E[x_{t+1} x_t]
            sum_previous, sum_next = state[:-1].sum(), state[1:].sum()
            sum_squares_previous, sum_squares_next = second_moment[:-1].sum(), second_moment[1:].sum()
            sum_cross = cross_moment.sum()
            intercept, persistence = np.linalg.solve(
                [[transitions, sum_previous], [sum_previous, sum_squares_previous]],
                [sum_next, sum_cross],
            )
            persistence = float(np.clip(persistence, -0.9999, 0.9999))
            state_variance_noise = (
                sum_squares_next - 2 * intercept * sum_next - 2 * persistence * sum_cross
                + transitions * intercept ** 2 + 2 * intercept * persistence * sum_previous
                + persistence ** 2 * sum_squares_previous
            ) / transitions
            observation_variance = np.mean((observations - state) ** 2 + variance)
            model = cls(intercept=float(intercept), persistence=persistence,
                        state_noise=float(np.sqrt(max(state_variance_noise, 1e-16))),
                        observation_noise=float(np.sqrt(max(observation_variance, 1e-16))),
                        initial_state=float(state[0]), initial_variance=float(variance[0]))
        model.log_likelihood = model._run_filter(observations)[4]
        return model
