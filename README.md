# fe_investing

Fe Investing, or "Iron Investing", is an investment analysis tool implementing statistical arbitrage for portfolio optimization.

Author: Brian Abod

## Environment Setup

The following commands should be run in a terminal to setup a recommended python environment. Some commands may vary slightly based on your platform.

```shell
# Setup a virtual environment
python3 -m venv .venv

# Activate the virtual environnment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Overview

Pairs trading is a technique that seeks to trade on the relationship between two stocks, rather than on one stock in isolation. **Statistical Arbitrage** is the term for trading on the relationship between groups of more than two stocks.

Statistical Arbitrage hinges on the concept of **cointegration**. If two series are cointegrated, then they share a common trend. This might seem similar to correlation, however for two series to be correlated, each value must move in tandem at each time step. A good image of cointegration is to think of a dog and his owner going for a walk. Although we might not be able to predict the movement of the dog or the owner individually, since the dog is on a leash, we know that at any point in time the dog and his owner will be near each other. So we can say they are cointegrated.

If the common trend between two stocks is adequately consistent, then it can be assumed that the spread (difference in price) between the two stocks is mean-reverting. With a mean-reverting signal, we can set thresholds to buy and sell with the confidence that, when the spread goes low, it will increase back to its mean value over time. This property can be exploited to effectively trade on the noise.

It is important to note that this strategy is not fool-proof. The main risk with this method lies in the consistency of the cointegration trend. If the statistics of the mean-reverting spread are non-stationary, then a model that worked for one period might fail to perform well in a different period where the underlying distribution has changed. So it is important to track the statical parameters of the metrics that you set your trading thresholds on to ensure good results.

To start, we need to identify pairs or groups of stocks that are cointegrated. This can be done either by brute force or by more sophisticated methods. The common trend between two cointegrated stocks might not be scaled identically. We use a factor, **gamma**, to represent the scaling difference of the common trend between two stocks. Once a group of potentially cointegrated stocks have been identified, we need to estimate the gamma factor. For this we can use a regression model. Least squares is a simple approach, but one that might break easily with changing statistical parameters in the data. A Kalman filter is a more robust tool for tracking gamma. A Kalman filter is essentially a adaptive way to track an unknown value given some indirect measurements and some statistical model of the relationship between those measurements and the unknown value.

## Testing

To test the statistical arbitrage methods, market trading is simulated using either generated data or real data (downloaded from Yahoo Finance).
