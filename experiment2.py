"""
Conduct an experiment with your StrategyLearner that shows how changing the value of
impact should affect in-sample trading behavior (use at least two metrics).
Trade JPM on the in-sample period with a commission of $0.00.

The code that implements this experiment and generates the relevant charts
and data should be submitted as experiment2.py.

See the ‘Report’ section on Experiment 2 for more details.
"""
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt

from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals, compute_optimized_portfolio_stats


def author():
    return 'narora62'


def experiment2(save_fig=False ,fig_name='Experiment2.png'):
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    symbols = 'JPM'
    commission = 9.95
    hr = "-" * 80

    learner_1 = StrategyLearner(verbose=False, impact=0.0090, commission=9.95)
    learner_2 = StrategyLearner(verbose=False, impact=0.0050, commission=9.95)
    learner_3 = StrategyLearner(verbose=False, impact=0.0010, commission=9.95)
    learner_4 = StrategyLearner(verbose=False, impact=0.0005, commission=9.95)
    learner_5 = StrategyLearner(verbose=False, impact=0.0000, commission=9.95)

    learner_1.add_evidence(symbol=symbols, sd=start_date, ed=end_date, sv=100000)  # training phase
    learner_2.add_evidence(symbol=symbols, sd=start_date, ed=end_date, sv=100000)  # training phase
    learner_3.add_evidence(symbol=symbols, sd=start_date, ed=end_date, sv=100000)  # training phase
    learner_4.add_evidence(symbol=symbols, sd=start_date, ed=end_date, sv=100000)  # training phase
    learner_5.add_evidence(symbol=symbols, sd=start_date, ed=end_date, sv=100000)  # training phase

    strategy_1 = learner_1.testPolicy(symbols, start_date, end_date, 100000)
    strategy_2 = learner_2.testPolicy(symbols, start_date, end_date, 100000)
    strategy_3 = learner_3.testPolicy(symbols, start_date, end_date, 100000)
    strategy_4 = learner_4.testPolicy(symbols, start_date, end_date, 100000)
    strategy_5 = learner_5.testPolicy(symbols, start_date, end_date, 100000)

    port_val_1 = compute_portvals(orders=strategy_1, start_val=100000, commission=commission, impact=0.0090)
    port_val_2 = compute_portvals(orders=strategy_2, start_val=100000, commission=commission, impact=0.0050)
    port_val_3 = compute_portvals(orders=strategy_3, start_val=100000, commission=commission, impact=0.0010)
    port_val_4 = compute_portvals(orders=strategy_4, start_val=100000, commission=commission, impact=0.0005)
    port_val_5 = compute_portvals(orders=strategy_5, start_val=100000, commission=commission, impact=0.0000)

    # Compute metrics for Manual Strategy, Strategy Learner and Benchmark
    cum_ret_sl_1, avg_daily_ret_sl_1, std_daily_ret_sl_1, sharpe_ratio_sl_1 = compute_optimized_portfolio_stats(port_val_1['portval'])
    cum_ret_sl_2, avg_daily_ret_sl_2, std_daily_ret_sl_2, sharpe_ratio_sl_2 = compute_optimized_portfolio_stats(port_val_2['portval'])
    cum_ret_sl_3, avg_daily_ret_sl_3, std_daily_ret_sl_3, sharpe_ratio_sl_3 = compute_optimized_portfolio_stats(port_val_3['portval'])
    cum_ret_sl_4, avg_daily_ret_sl_4, std_daily_ret_sl_4, sharpe_ratio_sl_4 = compute_optimized_portfolio_stats(port_val_4['portval'])
    cum_ret_sl_5, avg_daily_ret_sl_5, std_daily_ret_sl_5, sharpe_ratio_sl_5 = compute_optimized_portfolio_stats(port_val_5['portval'])

    print(hr)
    print("Date Range: {} to {}".format(start_date, end_date))
    print(hr)

    # Cumulative Returns
    print("Cumulative Return of Strategy Learner(impact: 0.0090): {}".format(cum_ret_sl_1))
    print("Cumulative Return of Strategy Learner(impact: 0.0050): {}".format(cum_ret_sl_2))
    print("Cumulative Return of Strategy Learner(impact: 0.0010): {}".format(cum_ret_sl_3))
    print("Cumulative Return of Strategy Learner(impact: 0.0005): {}".format(cum_ret_sl_4))
    print("Cumulative Return of Strategy Learner(impact: 0.0000): {}".format(cum_ret_sl_5))
    print(hr)

    # Standard Deviation
    print("Standard Deviation of Strategy Learner(impact: 0.0090): {}".format(std_daily_ret_sl_1))
    print("Standard Deviation of Strategy Learner(impact: 0.0050): {}".format(std_daily_ret_sl_2))
    print("Standard Deviation of Strategy Learner(impact: 0.0010): {}".format(std_daily_ret_sl_3))
    print("Standard Deviation of Strategy Learner(impact: 0.0005): {}".format(std_daily_ret_sl_4))
    print("Standard Deviation of Strategy Learner(impact: 0.0000): {}".format(std_daily_ret_sl_5))
    print(hr)

    # Average Daily Return
    print("Average Daily Return of Strategy Learner(impact: 0.0090): {}".format(avg_daily_ret_sl_1))
    print("Average Daily Return of Strategy Learner(impact: 0.0050): {}".format(avg_daily_ret_sl_2))
    print("Average Daily Return of Strategy Learner(impact: 0.0010): {}".format(avg_daily_ret_sl_3))
    print("Average Daily Return of Strategy Learner(impact: 0.0005): {}".format(avg_daily_ret_sl_4))
    print("Average Daily Return of Strategy Learner(impact: 0.0000): {}".format(avg_daily_ret_sl_5))
    print(hr)

    # Sharpe Ratio
    print("Sharpe Ratio of Strategy Learner(impact: 0.0090): {}".format(sharpe_ratio_sl_1))
    print("Sharpe Ratio of Strategy Learner(impact: 0.0050): {}".format(sharpe_ratio_sl_2))
    print("Sharpe Ratio of Strategy Learner(impact: 0.0010): {}".format(sharpe_ratio_sl_3))
    print("Sharpe Ratio of Strategy Learner(impact: 0.0005): {}".format(sharpe_ratio_sl_4))
    print("Sharpe Ratio of Strategy Learner(impact: 0.0000): {}".format(sharpe_ratio_sl_5))
    print(hr)

    # Final Portfolio value
    print("Final Portfolio Strategy Learner(impact: 0.0090): {}".format(port_val_1['portval'][-1]))
    print("Final Portfolio Strategy Learner(impact: 0.0050): {}".format(port_val_2['portval'][-1]))
    print("Final Portfolio Strategy Learner(impact: 0.0010): {}".format(port_val_3['portval'][-1]))
    print("Final Portfolio Strategy Learner(impact: 0.0005): {}".format(port_val_4['portval'][-1]))
    print("Final Portfolio Strategy Learner(impact: 0.0000): {}".format(port_val_5['portval'][-1]))
    print(hr)

    # Normalize all DataFrames
    port_val_1 /= port_val_1.iloc[0]
    port_val_2 /= port_val_2.iloc[0]
    port_val_3 /= port_val_3.iloc[0]
    port_val_4 /= port_val_4.iloc[0]
    port_val_5 /= port_val_5.iloc[0]

    # Rename columns to set legend and colors
    port_val_1.rename(columns={'portval': 'Strategy Learner 1 (impact: 0.0090)'}, inplace=True)
    port_val_2.rename(columns={'portval': 'Strategy Learner 2 (impact: 0.0050)'}, inplace=True)
    port_val_3.rename(columns={'portval': 'Strategy Learner 3 (impact: 0.0010)'}, inplace=True)
    port_val_4.rename(columns={'portval': 'Strategy Learner 4 (impact: 0.0005)'}, inplace=True)
    port_val_5.rename(columns={'portval': 'Strategy Learner 5 (impact: 0.000)'}, inplace=True)

    # Setup plot info and save fig
    ax = pd.concat([port_val_1, port_val_2, port_val_3, port_val_4, port_val_5], axis=1).plot(title="Strategy Learners with different Impacts", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio")
    plt.grid(True, 'both')
    if save_fig:
        plt.savefig(fig_name, dpi=550)
    else:
        plt.show()


if __name__ == '__main__':
    experiment2()
