"""Stationarity and cointegration tests (Feng & Palomar, Sec. 10.3 and 2.6).

A univariate time series is "integrated" if it can be made stationary by
differencing. The number of differences needed is its order of integration.
Log-prices are typically integrated of order 1: they wander like random walks,
but their differences (log-returns) are stationary.

An n-dimensional time series is "cointegrated" if some linear combination of
its components is stationary. The combination is called a cointegrating
relation, and its coefficients form a cointegrating vector. Cointegrated
variables are unstable in their levels but have mean-reverting "spreads" that
keep them moving around common stochastic trends. This differs from
covariance, which only measures the tendency to move together at each time
step. The cointegrating spread is what a statistical arbitrage strategy trades.

- ``augmented_dickey_fuller``: tests whether one series is a random walk, eq. (10.22).
- ``engle_granger``: two-step test. Regress one log-price on the others with
  least squares (10.16), then ADF-test the residuals (10.19).
- ``johansen``: rank test on the VECM matrix Pi = alpha beta^T (2.49)-(2.55).
  It works for more than two stocks and returns every cointegrating vector.
- ``mean_crossing_rate``: the quick mean-reversion check from Sec. 10.3.2.

Trend options: "none", "constant", or "linear" (constant + time trend).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from .critical_values import (
    JOHANSEN_LEVELS,
    JOHANSEN_MAXIMUM_EIGENVALUE,
    JOHANSEN_TRACE,
    MAXIMUM_SERIES,
    dickey_fuller_critical_values,
    dickey_fuller_p_value,
)
from .regression import deterministic_terms, least_squares

_TREND_TERM_COUNT = {"none": 0, "constant": 1, "linear": 2}


# --- Augmented Dickey-Fuller -------------------------------------------------

@dataclass
class DickeyFullerResult:
    statistic: float
    p_value: float
    lag_count: int
    observation_count: int
    critical_values: dict[str, float]
    level_coefficient: float  # phi1 on x_{t-1}; more negative = faster mean reversion

    def is_stationary(self, level: str = "5%") -> bool:
        return self.statistic < self.critical_values[level]


def _dickey_fuller_design(values: np.ndarray, lag_count: int):
    """Target dx_t and regressors [x_{t-1}, dx_{t-1}, ..., dx_{t-lag_count}]."""
    differences = np.diff(values)
    target = differences[lag_count:]
    columns = [values[lag_count:-1]]
    columns += [differences[lag_count - lag: len(differences) - lag] for lag in range(1, lag_count + 1)]
    return target, np.column_stack(columns)


def augmented_dickey_fuller(
    values,
    trend: str = "constant",
    max_lags: int | None = None,
    choose_lags: bool = True,
    series_count: int = 1,
    critical_value_trend: str | None = None,
) -> DickeyFullerResult:
    """Augmented Dickey-Fuller test of H0: the series has a unit root (is a random walk).

    Fits  dx_t = phi0 + c0*t + phi1*x_{t-1} + sum_i phi_{i+1} dx_{t-i} + e_t  (10.22)
    and returns the t-statistic of phi1 (10.21). With ``choose_lags`` the
    number of lagged differences is picked by AIC on a common sample, the
    same way statsmodels does it.

    ``series_count`` and ``critical_value_trend`` are only for residual tests
    such as Engle-Granger, which need different critical values.
    """
    values = np.asarray(values, dtype=float)
    count = len(values)
    trend_terms = _TREND_TERM_COUNT[trend]
    if max_lags is None:
        max_lags = int(np.ceil(12.0 * (count / 100.0) ** 0.25))
        max_lags = max(0, min(count // 2 - trend_terms - 1, max_lags))

    lag_count = max_lags
    if choose_lags and max_lags > 0:
        target, regressors = _dickey_fuller_design(values, max_lags)
        full = np.column_stack([deterministic_terms(len(target), trend), regressors])
        best_criterion = np.inf
        for lags in range(max_lags + 1):
            fit = least_squares(target, full[:, : trend_terms + 1 + lags])
            if fit.akaike_information_criterion < best_criterion:
                best_criterion, lag_count = fit.akaike_information_criterion, lags

    target, regressors = _dickey_fuller_design(values, lag_count)
    fit = least_squares(target, np.column_stack([regressors, deterministic_terms(len(target), trend)]))
    statistic = float(fit.t_values[0])

    table_trend = critical_value_trend or trend
    if table_trend == "none":
        table_trend = "constant"  # MacKinnon's no-constant surface isn't tabulated here
    table_count = len(target) if series_count == 1 else count - 1
    return DickeyFullerResult(
        statistic=statistic,
        p_value=dickey_fuller_p_value(statistic, series_count, table_trend),
        lag_count=lag_count,
        observation_count=len(target),
        critical_values=dickey_fuller_critical_values(series_count, table_trend, table_count),
        level_coefficient=float(fit.coefficients[0]),
    )


# --- Engle-Granger -------------------------------------------------------------

@dataclass
class EngleGrangerResult:
    """Result of regressing ``dependent`` on the other stocks.

    ``weights`` are the cointegrating weights on log-prices, normalized so the
    dependent stock has weight 1: spread_t = weights . y_t - intercept
    (= y1 - gamma*y2 - mu for a pair, eq. 10.19).
    """
    dependent: str
    weights: pd.Series
    intercept: float
    trend_coefficient: float
    dickey_fuller: DickeyFullerResult
    residuals: pd.Series

    @property
    def statistic(self) -> float:
        return self.dickey_fuller.statistic

    @property
    def p_value(self) -> float:
        return self.dickey_fuller.p_value

    @property
    def hedge_ratios(self) -> pd.Series:
        """Units of each other stock per unit of the dependent stock (gamma)."""
        return -self.weights.drop(self.dependent)

    def is_cointegrated(self, level: str = "5%") -> bool:
        return self.dickey_fuller.is_stationary(level)


def engle_granger(
    log_prices: pd.DataFrame,
    dependent: str | None = None,
    trend: str = "constant",
    max_lags: int | None = None,
) -> EngleGrangerResult:
    """Engle-Granger two-step cointegration test.

    Step 1: least squares  y1 = mu (+ c0*t) + gamma . y_others + eps   (10.16)
    Step 2: ADF test on the residuals with Engle-Granger critical values.
            The regression already removed the mean, so the residual test
            has no constant; the critical values account for that.

    ``trend="linear"`` allows cointegration with a deterministic trend (Example 10.2).
    """
    symbols = list(log_prices.columns)
    dependent = dependent or symbols[0]
    others = [symbol for symbol in symbols if symbol != dependent]
    target = log_prices[dependent].to_numpy(float)
    regressors = np.column_stack([deterministic_terms(len(target), trend),
                                  log_prices[others].to_numpy(float)])
    fit = least_squares(target, regressors)
    trend_terms = _TREND_TERM_COUNT[trend]
    hedge_ratios = fit.coefficients[trend_terms:]

    residual_test = augmented_dickey_fuller(
        fit.residuals, trend="none", max_lags=max_lags,
        series_count=len(symbols), critical_value_trend=trend,
    )
    return EngleGrangerResult(
        dependent=dependent,
        weights=pd.Series(np.r_[1.0, -hedge_ratios], index=[dependent] + others),
        intercept=float(fit.coefficients[0]),
        trend_coefficient=float(fit.coefficients[1]) if trend == "linear" else 0.0,
        dickey_fuller=residual_test,
        residuals=pd.Series(fit.residuals, index=log_prices.index, name="residual"),
    )


def engle_granger_strongest(log_prices: pd.DataFrame, **kwargs) -> EngleGrangerResult:
    """Run Engle-Granger with each stock as the dependent one and keep the
    strongest result. The test depends on regression order (Remark 10.4), so
    trying every order makes it more robust."""
    results = [engle_granger(log_prices, dependent=symbol, **kwargs) for symbol in log_prices.columns]
    return min(results, key=lambda result: result.statistic)


# --- Johansen ------------------------------------------------------------------

@dataclass
class JohansenResult:
    eigenvalues: np.ndarray
    vectors: pd.DataFrame                   # columns = cointegrating vectors beta_i, strongest first
    trace_statistics: np.ndarray
    trace_critical_values: np.ndarray       # rows: rank 0..N-1; columns: 90%, 95%, 99%
    maximum_eigenvalue_statistics: np.ndarray
    maximum_eigenvalue_critical_values: np.ndarray
    lag_differences: int
    symbols: list[str] = field(default_factory=list)

    def critical_value(self, rank: int, level: str = "95%") -> float:
        return float(self.trace_critical_values[rank, JOHANSEN_LEVELS.index(level)])

    def rank(self, level: str = "95%") -> int:
        """Number of cointegrating relations by the sequential trace test."""
        for rank, statistic in enumerate(self.trace_statistics):
            if statistic < self.critical_value(rank, level):
                return rank
        return len(self.trace_statistics)


def johansen(log_prices: pd.DataFrame, lag_differences: int = 1) -> JohansenResult:
    """Johansen cointegration test with an unrestricted constant.

    Fits the VECM  r_t = phi0 + Pi y_{t-1} + sum_j Phi_j r_{t-j} + w_t   (2.49)
    by reduced-rank regression. The eigenvectors give beta in Pi = alpha beta^T
    (2.54). Each beta_i^T y_t is a candidate mean-reverting spread (Sec. 10.5.1),
    and a larger eigenvalue means stronger mean reversion.
    """
    levels = log_prices.to_numpy(float)
    series_count = levels.shape[1]
    if series_count > MAXIMUM_SERIES:
        raise ValueError(f"Critical values are tabulated for up to {MAXIMUM_SERIES} series.")
    levels = levels - levels.mean(axis=0)
    differences = np.diff(levels, axis=0)

    count = len(differences) - lag_differences
    lagged_differences = (
        np.column_stack([differences[lag_differences - lag: len(differences) - lag]
                         for lag in range(1, lag_differences + 1)])
        if lag_differences > 0 else np.empty((count, 0))
    )
    current_differences = differences[lag_differences:]
    lagged_levels = levels[lag_differences: lag_differences + count]  # y_{t-1}

    def demean(array):
        return array - array.mean(axis=0)

    def remove_lagged_differences(array):
        if lagged_differences.size == 0:
            return array
        nuisance = demean(lagged_differences)
        return array - nuisance @ np.linalg.lstsq(nuisance, array, rcond=None)[0]

    difference_residuals = remove_lagged_differences(demean(current_differences))
    level_residuals = remove_lagged_differences(demean(lagged_levels))
    difference_covariance = difference_residuals.T @ difference_residuals / count
    level_covariance = level_residuals.T @ level_residuals / count
    cross_covariance = difference_residuals.T @ level_residuals / count

    # Solve  S10 S00^-1 S01 v = lambda S11 v ; eigh normalizes v' S11 v = I.
    eigenvalues, eigenvectors = eigh(
        cross_covariance.T @ np.linalg.solve(difference_covariance, cross_covariance),
        level_covariance,
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, 1 - 1e-12)
    eigenvectors = eigenvectors[:, order]
    eigenvectors *= np.sign(eigenvectors[0, :] + (eigenvectors[0, :] == 0))

    trace = np.array([-count * np.sum(np.log(1 - eigenvalues[rank:])) for rank in range(series_count)])
    maximum_eigenvalue = -count * np.log(1 - eigenvalues)
    return JohansenResult(
        eigenvalues=eigenvalues,
        vectors=pd.DataFrame(eigenvectors, index=log_prices.columns,
                             columns=[f"beta{index + 1}" for index in range(series_count)]),
        trace_statistics=trace,
        trace_critical_values=np.array([JOHANSEN_TRACE[series_count - rank - 1]
                                        for rank in range(series_count)]),
        maximum_eigenvalue_statistics=maximum_eigenvalue,
        maximum_eigenvalue_critical_values=np.array([JOHANSEN_MAXIMUM_EIGENVALUE[series_count - rank - 1]
                                                     for rank in range(series_count)]),
        lag_differences=lag_differences,
        symbols=list(log_prices.columns),
    )


def mean_crossing_rate(spread) -> float:
    """Share of steps where the spread crosses its sample mean (Sec. 10.3.2).
    Higher means stronger mean reversion. It's only a heuristic, with no
    significance level attached."""
    values = np.asarray(spread, dtype=float)
    side = np.sign(values - values.mean())
    return float(np.mean(side[1:] * side[:-1] < 0))
