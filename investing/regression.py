"""Ordinary least squares, shared by the cointegration tests and spread models."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LeastSquaresResult:
    coefficients: np.ndarray
    residuals: np.ndarray
    t_values: np.ndarray
    sum_squared_residuals: float
    observation_count: int

    @property
    def akaike_information_criterion(self) -> float:
        """AIC up to a constant, used to choose the number of lags."""
        count, parameters = self.observation_count, len(self.coefficients)
        return count * np.log(self.sum_squared_residuals / count) + 2 * parameters


def least_squares(target: np.ndarray, regressors: np.ndarray) -> LeastSquaresResult:
    """Regress ``target`` on the columns of ``regressors``."""
    coefficients, *_ = np.linalg.lstsq(regressors, target, rcond=None)
    residuals = target - regressors @ coefficients
    count, parameters = regressors.shape
    sum_squared_residuals = float(residuals @ residuals)
    residual_variance = sum_squared_residuals / max(count - parameters, 1)
    covariance = residual_variance * np.linalg.pinv(regressors.T @ regressors)
    t_values = coefficients / np.sqrt(np.diag(covariance))
    return LeastSquaresResult(coefficients, residuals, t_values, sum_squared_residuals, count)


def deterministic_terms(count: int, trend: str) -> np.ndarray:
    """Regressor columns for a trend: "none", "constant", or "linear" (constant + time trend)."""
    columns = []
    if trend in ("constant", "linear"):
        columns.append(np.ones(count))
    if trend == "linear":
        columns.append(np.arange(1, count + 1, dtype=float))
    return np.column_stack(columns) if columns else np.empty((count, 0))
