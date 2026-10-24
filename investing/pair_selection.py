"""Pair selection (Feng & Palomar, Sec. 10.2).

Testing every pair for cointegration is slow and produces false positives, so
first rank pairs by a cheap score and only test the best ones.

- Normalized price distance (10.7)-(10.8). It assumes a hedge ratio of 1.
- Normalized factor-loading difference (10.12).
- Common-trend correlation |rho| = |cos theta| (10.13). It uses only the part
  of each return explained by common factors. Remark 10.1 explains why the raw
  return correlation is the wrong measure.

Factors are the top principal components ("eigen-portfolios") of the
log-return covariance.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


def normalized_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """p~_t = prod_{i<=t} (1 + R_i): each price path rescaled to start at 1 (10.8)."""
    return prices / prices.iloc[0]


def normalized_price_distance(prices: pd.DataFrame, first: str, second: str, norm: str = "l2") -> float:
    """Sum of squared gaps between normalized prices (10.7).
    ``norm="l1"`` uses absolute gaps, which is less sensitive to outliers."""
    normalized = normalized_prices(prices[[first, second]])
    gap = normalized[first] - normalized[second]
    return float((gap ** 2).sum() if norm == "l2" else gap.abs().sum())


@dataclass
class FactorModel:
    """r_it = pi_i' f_t + eps_it  (10.11)."""
    loadings: pd.DataFrame          # one row per stock: pi_i
    factor_covariance: np.ndarray   # Sigma_f
    factors: pd.DataFrame           # one row per date: f_t
    eigenvectors: pd.DataFrame      # eigen-portfolio weights, one column per factor

    def residuals(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """Stock-specific returns eps_it = r_it - pi_i' f_t."""
        return log_returns - self.factors.to_numpy() @ self.loadings.to_numpy().T


def principal_component_factor_model(log_returns: pd.DataFrame, factor_count: int = 1) -> FactorModel:
    """Factor model whose factors are the top principal components of the
    return covariance."""
    returns = log_returns.to_numpy(float)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(returns, rowvar=False))
    top = np.argsort(eigenvalues)[::-1][:factor_count]
    eigenvectors = eigenvectors[:, top]
    eigenvectors *= np.sign(eigenvectors.sum(axis=0))  # make the first eigen-portfolio mostly long
    factors = returns @ eigenvectors
    # Loadings by least squares (equal to the eigenvectors for exact PCA; kept general).
    loadings, *_ = np.linalg.lstsq(factors, returns, rcond=None)
    names = [f"factor{index + 1}" for index in range(factor_count)]
    return FactorModel(
        loadings=pd.DataFrame(loadings.T, index=log_returns.columns, columns=names),
        factor_covariance=np.atleast_2d(np.cov(factors, rowvar=False)),
        factors=pd.DataFrame(factors, index=log_returns.index, columns=names),
        eigenvectors=pd.DataFrame(eigenvectors, index=log_returns.columns, columns=names),
    )


def normalized_factor_loading_difference(factor_model: FactorModel, first: str, second: str) -> float:
    """|| pi_1/||pi_1|| - pi_2/||pi_2|| ||  (10.12). Small means similar factor exposure."""
    first_loading = factor_model.loadings.loc[first].to_numpy()
    second_loading = factor_model.loadings.loc[second].to_numpy()
    return float(np.linalg.norm(first_loading / np.linalg.norm(first_loading)
                                - second_loading / np.linalg.norm(second_loading)))


def common_trend_correlation(factor_model: FactorModel, first: str, second: str) -> float:
    """|pi_1' S pi_2| / sqrt((pi_1' S pi_1)(pi_2' S pi_2))  (10.13). Near 1 is good.
    With a single factor this is always 1, so use at least 2 factors."""
    first_loading = factor_model.loadings.loc[first].to_numpy()
    second_loading = factor_model.loadings.loc[second].to_numpy()
    covariance = factor_model.factor_covariance
    return float(abs(first_loading @ covariance @ second_loading)
                 / np.sqrt((first_loading @ covariance @ first_loading)
                           * (second_loading @ covariance @ second_loading)))


def rank_pairs(prices: pd.DataFrame, factor_count: int = 2) -> pd.DataFrame:
    """Score every pair with all the measures, smallest price distance first."""
    log_returns = np.log(prices).diff().dropna()
    factor_model = principal_component_factor_model(log_returns, min(factor_count, prices.shape[1]))
    rows = []
    for first, second in combinations(prices.columns, 2):
        rows.append({
            "pair": (first, second),
            "price_distance": normalized_price_distance(prices, first, second),
            "factor_loading_difference": normalized_factor_loading_difference(factor_model, first, second),
            "trend_correlation": common_trend_correlation(factor_model, first, second),
            "return_correlation": float(log_returns[first].corr(log_returns[second])),
        })
    return pd.DataFrame(rows).sort_values("price_distance", ignore_index=True)
