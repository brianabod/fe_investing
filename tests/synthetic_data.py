"""Synthetic price data shared by the tests."""

import numpy as np
import pandas as pd


def common_trend_pair(count=1000, hedge_ratio=1.0, seed=0, persistence=0.8):
    """Stochastic common-trend model (Feng & Palomar 10.1-10.3) with AR(1)
    stock-specific noise, so the spread has a half-life of a few days."""
    generator = np.random.default_rng(seed)
    trend = np.cumsum(generator.normal(0, 0.01, count))
    noise = np.zeros(count)
    for t in range(1, count):
        noise[t] = persistence * noise[t - 1] + generator.normal(0, 0.005)
    second = trend + generator.normal(0, 0.002, count)
    first = hedge_ratio * trend + noise
    dates = pd.bdate_range("2020-01-01", periods=count)
    return pd.DataFrame({"A": first, "B": second}, index=dates)


def autoregressive(count, persistence, scale=1.0, seed=0):
    generator = np.random.default_rng(seed)
    values = np.zeros(count)
    for t in range(1, count):
        values[t] = persistence * values[t - 1] + scale * generator.normal()
    return values


def write_price_csv(path, dates, prices):
    """Write a Yahoo Finance-style CSV."""
    pd.DataFrame({"Date": pd.to_datetime(dates).strftime("%Y-%m-%d"), "Open": prices, "High": prices,
                  "Low": prices, "Close": prices, "Adj Close": prices, "Volume": 1000}).to_csv(path, index=False)
