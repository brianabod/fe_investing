import os

import matplotlib.pyplot as plt

from Portfolio import Portfolio


def pairs_trading(portfolio: Portfolio):
    fig1 = portfolio.plot_time_series()
    fig2 = portfolio.plot_distributions()
    fig3 = portfolio.plot_statistics_over_time(window_size=365)
    fig4 = portfolio.compare_stocks()


if __name__ == "__main__":
    from Stock import Stock

    portfolio = Portfolio(
        [
            Stock("Lockheed Martin", "LMT", os.path.join("defense", "LMT.csv"), "forestgreen"),
            Stock("Northrop Grumman", "NOC", os.path.join("defense", "NOC.csv"), "slategrey"),
            Stock("General Dynamics", "GD", os.path.join("defense", "GD.csv"), "darkblue"),
            # Stock("Boeing", "BA", os.path.join("defense", "BA.csv"), 'tomato'),
            # Stock("Leidos", "LDOS", os.path.join("defense", "LDOS.csv"), 'sandybrown'),
            # Stock("Raytheon", "RTX", os.path.join("defense", "^RTX.csv"), 'indianred'),
        ]
    )
    pairs_trading(portfolio)
    plt.show()
