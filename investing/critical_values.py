"""Critical values and p-values for unit-root and cointegration tests.

These are MacKinnon's published response surfaces, the same tables statsmodels
uses:
- Dickey-Fuller / Engle-Granger critical values: MacKinnon (2010).
- Dickey-Fuller / Engle-Granger p-values: MacKinnon (1994).
- Johansen trace and maximum-eigenvalue tests: MacKinnon, Haug & Michelis (1999),
  for the case of a constant term.

``series_count`` is the number of I(1) series in the test: 1 for a plain ADF
test, and the number of stocks for an Engle-Granger test.
"""

import numpy as np
from scipy.stats import norm

CONFIDENCE_LEVELS = ("1%", "5%", "10%")

# [series_count - 1][level] -> coefficients of 1, 1/T, 1/T^2, 1/T^3
_DICKEY_FULLER_SURFACES = {
    "constant": np.array([
        [[-3.43035, -6.5393, -16.786, -79.433], [-2.86154, -2.8903, -4.234, -40.040], [-2.56677, -1.5384, -2.809, 0]],
        [[-3.89644, -10.9519, -33.527, 0], [-3.33613, -6.1101, -6.823, 0], [-3.04445, -4.2412, -2.720, 0]],
        [[-4.29374, -14.4354, -33.195, 47.433], [-3.74066, -8.5632, -10.852, 27.982], [-3.45218, -6.2143, -3.718, 0]],
        [[-4.64332, -18.1031, -37.972, 0], [-4.09600, -11.2349, -11.175, 0], [-3.81020, -8.3931, -4.137, 0]],
        [[-4.95756, -21.8883, -45.142, 0], [-4.41519, -14.0405, -12.575, 0], [-4.13157, -10.7417, -3.784, 0]],
        [[-5.24568, -25.6688, -57.737, 88.639], [-4.70693, -16.9178, -17.492, 60.007], [-4.42501, -13.1875, -5.104, 27.877]],
    ]),
    "linear": np.array([
        [[-3.95877, -9.0531, -28.428, -134.155], [-3.41049, -4.3904, -9.036, -45.374], [-3.12705, -2.5856, -3.925, -22.38]],
        [[-4.32762, -15.4387, -35.679, 0], [-3.78057, -9.5106, -12.074, 0], [-3.49631, -7.0815, -7.538, 21.892]],
        [[-4.66305, -18.7688, -49.793, 104.244], [-4.1189, -11.8922, -19.031, 77.332], [-3.83511, -9.0723, -8.504, 35.403]],
        [[-4.9694, -22.4694, -52.599, 51.314], [-4.42871, -14.5876, -18.228, 39.647], [-4.14633, -11.25, -9.873, 54.109]],
        [[-5.25276, -26.2183, -59.631, 50.646], [-4.71537, -17.3569, -22.66, 91.359], [-4.43422, -13.6078, -10.238, 76.781]],
        [[-5.51727, -29.976, -75.222, 202.253], [-4.98228, -20.305, -25.224, 132.03], [-4.70233, -16.1253, -9.836, 94.272]],
    ]),
}

_P_VALUE_TABLES = {
    "constant": dict(
        maximum=[2.74, 0.92, 0.55, 0.61, 0.79, 1.0],
        minimum=[-18.83, -18.86, -23.48, -28.07, -25.96, -23.27],
        switch_point=[-1.61, -2.62, -3.13, -3.47, -3.78, -3.93],
        small_statistic=[[2.1659, 1.4412, 0.038269], [2.92, 1.5012, 0.039796], [3.4699, 1.4856, 0.03164],
                         [3.9673, 1.4777, 0.026315], [4.5509, 1.5338, 0.029545], [5.1399, 1.6036, 0.034445]],
        large_statistic=[[1.7339, 0.93202, -0.12745, -0.010368], [2.1945, 0.64695, -0.29198, -0.042377],
                         [2.5893, 0.45168, -0.36529, -0.050074], [3.0387, 0.45452, -0.33666, -0.041921],
                         [3.5049, 0.52098, -0.29158, -0.033468], [3.9489, 0.58933, -0.25359, -0.02721]],
    ),
    "linear": dict(
        maximum=[0.7, 0.63, 0.71, 0.93, 1.19, 1.42],
        minimum=[-16.18, -21.15, -25.37, -26.63, -26.53, -26.18],
        switch_point=[-2.89, -3.19, -3.5, -3.65, -3.8, -4.36],
        small_statistic=[[3.2512, 1.6047, 0.049588], [3.6646, 1.5419, 0.036448], [4.0983, 1.5173, 0.029898],
                         [4.5844, 1.5338, 0.028796], [5.0722, 1.5634, 0.029472], [5.53, 1.5914, 0.030392]],
        large_statistic=[[2.5261, 0.61654, -0.37956, -0.060285], [2.85, 0.5272, -0.36622, -0.051695],
                         [3.221, 0.5255, -0.32685, -0.041501], [3.652, 0.59758, -0.27483, -0.032081],
                         [4.0712, 0.66428, -0.23464, -0.02546], [4.4735, 0.71757, -0.20681, -0.021196]],
    ),
}

# Johansen, constant term. [series_count - rank - 1] -> (90%, 95%, 99%)
JOHANSEN_LEVELS = ("90%", "95%", "99%")
JOHANSEN_TRACE = np.array([
    [2.7055, 3.8415, 6.6349], [13.4294, 15.4943, 19.9349], [27.0669, 29.7961, 35.4628],
    [44.4929, 47.8545, 54.6815], [65.8202, 69.8189, 77.8202], [91.109, 95.7542, 104.9637],
])
JOHANSEN_MAXIMUM_EIGENVALUE = np.array([
    [2.7055, 3.8415, 6.6349], [12.2971, 14.2639, 18.52], [18.8928, 21.1314, 25.865],
    [25.1236, 27.5858, 32.7172], [31.2379, 33.8777, 39.3693], [37.2786, 40.0763, 45.8662],
])
MAXIMUM_SERIES = 6


def dickey_fuller_critical_values(series_count: int = 1, trend: str = "constant",
                                  observation_count: float = np.inf) -> dict[str, float]:
    """1% / 5% / 10% critical values for an ADF or Engle-Granger statistic."""
    table = _DICKEY_FULLER_SURFACES[trend]
    if not 1 <= series_count <= MAXIMUM_SERIES:
        raise ValueError(f"Critical values are tabulated for 1-{MAXIMUM_SERIES} series.")
    inverse_count = 0.0 if np.isinf(observation_count) else 1.0 / observation_count
    values = table[series_count - 1] @ (inverse_count ** np.arange(4))
    return dict(zip(CONFIDENCE_LEVELS, values.tolist()))


def dickey_fuller_p_value(statistic: float, series_count: int = 1, trend: str = "constant") -> float:
    """Approximate p-value for an ADF or Engle-Granger statistic."""
    table = _P_VALUE_TABLES[trend]
    index = series_count - 1
    if statistic > table["maximum"][index]:
        return 1.0
    if statistic < table["minimum"][index]:
        return 0.0
    small = statistic <= table["switch_point"][index]
    coefficients = table["small_statistic" if small else "large_statistic"][index]
    return float(norm.cdf(np.polyval(coefficients[::-1], statistic)))
