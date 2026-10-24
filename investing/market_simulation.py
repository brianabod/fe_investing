"""Simulated markets for testing statistical arbitrage on known answers."""

import numpy as np


def generate_cointegrated_time_series(sample_count: int, hedge_ratio: float = 1.0,
                                      noise_scale: float = 1.0, seed: int | None = None):
    """Two series that are integrated of order 1 and cointegrated.

    Both follow one random-walk common trend; the second one's trend is scaled
    by ``hedge_ratio`` (gamma). Each adds its own stationary noise, so
    ``second - hedge_ratio * first`` is stationary even though neither series is.
    """
    generator = np.random.default_rng(seed)
    common_trend = np.cumsum(generator.normal(0, 1, sample_count))
    first = common_trend + generator.normal(0, noise_scale, sample_count)
    second = hedge_ratio * common_trend + generator.normal(0, noise_scale, sample_count)
    return first, second
