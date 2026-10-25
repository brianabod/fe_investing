"""Statistical arbitrage on the stocks in the data folder (Feng & Palomar, Ch. 10).

1. Rank pairs (Sec. 10.2) and test them for cointegration (10.3) on the most
   recent year.
2. Pick an entry threshold for the strongest pair (10.4.1) and show its
   spread, positions and in-sample profit (Figs. 10.6 / 10.10 / 10.12).
3. Walk-forward, out-of-sample backtest with costs for pairs, baskets
   and factor-residual spreads (10.5).

Usage:  python examples/statistical_arbitrage.py [--data-directory DIR] [--symbols A B C]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make the investing package importable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from command_line import load_portfolio, parse_arguments
from investing import (
    FactorBuilder,
    BasketBuilder,
    KalmanSpreadFilter,
    PairsBuilder,
    Spread,
    TradingRule,
    compute_positions,
    pair_cointegration_test,
    fit_mean_reversion,
    basket_cointegration_test,
    mean_crossing_rate,
    optimal_threshold,
    performance_summary,
    rank_pairs,
    standardize,
    walk_forward,
)
from investing.plotting import apply_style, plot_equity_curves, plot_spread_trading, plot_threshold_curves
from investing.pair_selection import closest_basket
from investing.threshold import THRESHOLD_METHODS, simulate_round_trips

BASKET_SIZE = 3         # stocks in each basket
FITTING_DAYS = 252      # days used to fit each model
TRADING_DAYS = 126      # days each fitted model is traded out-of-sample
COST_BASIS_POINTS = 5.0 # per $1 of gross exposure traded


def analyze_recent(prices: pd.DataFrame):
    recent = prices.iloc[-FITTING_DAYS:]
    log_prices = np.log(recent)

    print("\n== Pair selection (last year) ==")
    ranking = rank_pairs(recent)
    print(ranking.to_string(index=False, float_format="%.3f"))

    print("\n== Pair cointegration tests ==")
    rows = []
    for pair in ranking.pair:
        result = pair_cointegration_test(log_prices[list(pair)])
        rows.append({
            "pair": "-".join(pair),
            "dependent": result.dependent,
            "hedge_ratio": float(result.hedge_ratios.iloc[0]),
            "statistic": result.statistic,
            "p_value": result.p_value,
            "half_life": fit_mean_reversion(result.residuals).half_life,
            "crossing_rate": mean_crossing_rate(result.residuals),
            "result": result,
        })
    tests = pd.DataFrame(rows).sort_values("p_value", ignore_index=True)
    print(tests.drop(columns="result").to_string(index=False, float_format="%.3f"))

    symbols = closest_basket(recent, BASKET_SIZE)
    print(f"\n== Basket cointegration test ({', '.join(symbols)}: the {len(symbols)} most closely related stocks) ==")
    basket = basket_cointegration_test(log_prices[symbols])
    for rank, statistic in enumerate(basket.trace_statistics):
        print(f"  H0: rank <= {rank}: trace={statistic:.2f}  5% critical value={basket.critical_value(rank):.2f}")
    print(f"  cointegration rank at 95%: {basket.rank()}")

    best = tests.iloc[0]
    if best.p_value > 0.05:
        print(f"\nNote: no pair is cointegrated at 5% this year; plotting the closest ({best.pair}).")
    return recent, best["result"]


def plot_best_pair(recent: pd.DataFrame, result):
    spread = Spread(result.weights, result.intercept)
    values = spread.values(np.log(recent))
    standardized = standardize(values)
    filtered = KalmanSpreadFilter.fit(standardized.to_numpy()).filter(standardized)

    threshold_results = {method: optimal_threshold(standardized, method) for method in THRESHOLD_METHODS}
    entry_threshold = threshold_results["empirical"].best_threshold
    positions = compute_positions(standardized, entry_threshold)
    round_trips = len(simulate_round_trips(standardized, entry_threshold))
    print(f"\nBest pair in-sample: entry threshold={entry_threshold:.2f}, {round_trips} round trips")

    other = result.hedge_ratios.index[0]
    title = (f"In-sample pair trade: {result.dependent} − {result.hedge_ratios.iloc[0]:.2f} × {other}  "
             f"(entry threshold {entry_threshold:.2f}, half-life {fit_mean_reversion(values).half_life:.1f} days)")
    plot_spread_trading(np.log(recent[[result.dependent, other]]), standardized, entry_threshold,
                        positions, filtered, title)
    plot_threshold_curves(threshold_results)


def run_backtests(prices: pd.DataFrame):
    strategies = {
        "Pairs": PairsBuilder(max_spreads=3),
        f"Baskets ({BASKET_SIZE} most related stocks)": BasketBuilder(basket_size=BASKET_SIZE),
        "Factor residuals (1 principal component)": FactorBuilder(factor_count=1),
    }
    rule = TradingRule(threshold_method="empirical", stop_loss=4.0)
    print(f"\n== Walk-forward backtest ({FITTING_DAYS}-day fitting / {TRADING_DAYS}-day trading, "
          f"{COST_BASIS_POINTS:.0f} basis point costs) ==")
    results, rows = {}, {}
    for name, builder in strategies.items():
        result = walk_forward(prices, builder, FITTING_DAYS, TRADING_DAYS, rule, COST_BASIS_POINTS)
        performance = performance_summary(result.returns)
        performance["spreads_traded"] = len(result.trades)
        performance["round_trips"] = int(result.summary().trades.sum()) if result.trades else 0
        results[name], rows[name] = result, performance
    print(pd.DataFrame(rows).T.to_string(float_format="%.3f"))
    plot_equity_curves(results)


if __name__ == "__main__":
    arguments = parse_arguments(__doc__)
    apply_style()
    prices = load_portfolio(arguments).prices()
    print(f"{len(prices)} common trading days, {prices.index[0].date()} to {prices.index[-1].date()}")
    recent, best = analyze_recent(prices)
    plot_best_pair(recent, best)
    run_backtests(prices)
    plt.show()
