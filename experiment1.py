"""
Compare your Manual Strategy with your Strategy Learner in-sample trading JPM. Create a chart that shows:

Value of the ManualStrategy portfolio (normalized to 1.0 at the start)
Value of the StrategyLearner portfolio (normalized to 1.0 at the start)
Value of the Benchmark portfolio (normalized to 1.0 at the start)
The code that implements this experiment and generates the relevant charts and data should be submitted as experiment1.py.

See DATA DETAILS, DATES & RULES section above for commission and impact information.
"""
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt

from ManualStrategy import testPolicy as testPolicy_ms
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals, compute_optimized_portfolio_stats


def author():
    return "narora62"


def experiment1(save_fig=False, fig_name='Experiment1.png'):
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    symbols = 'JPM'
    start_date_o = dt.datetime(2010, 1, 1)
    end_date_o = dt.datetime(2011, 12, 31)
    impact = 0.005
    commission = 9.95

    test_out_sample = False  # Change to True to test the out-sample results

    if test_out_sample:
        portval_ms = testPolicy_ms(symbols, start_date_o, end_date_o, 100000)
    else:
        portval_ms = testPolicy_ms(symbols, start_date, end_date, 100000)
    port_val_ms = compute_portvals(orders=portval_ms, start_val=100000, commission=commission, impact=impact)

    sl = StrategyLearner(verbose=False, impact=impact, commission=9.95)
    sl.add_evidence(symbol=symbols, sd=start_date, ed=end_date, sv=100000)  # training phase

    if test_out_sample:
        portval_sl = sl.testPolicy(symbols, start_date_o, end_date_o, 100000)
    else:
        portval_sl = sl.testPolicy(symbols, start_date, end_date, 100000)

    port_val_sl = compute_portvals(orders=portval_sl, start_val=100000, commission=commission, impact=impact)

    if test_out_sample:
        portval_bench = sl.bench_mark(symbols, start_date_o, end_date_o, 100000)
    else:
        portval_bench = sl.bench_mark(symbols, start_date, end_date, 100000)
    port_val_bench = compute_portvals(orders=portval_bench, start_val=100000, commission=commission, impact=impact)

    # Compute metrics for Manual Strategy, Strategy Learner and Benchmark
    cum_ret_ms, avg_daily_ret_ms, std_daily_ret_ms, sharpe_ratio_ms = compute_optimized_portfolio_stats(port_val_ms['portval'])
    cum_ret_sl, avg_daily_ret_sl, std_daily_ret_sl, sharpe_ratio_sl = compute_optimized_portfolio_stats(port_val_sl['portval'])
    cum_ret_bchm, avg_daily_ret_bchm, std_daily_ret_bchm, sharpe_ratio_bchm = compute_optimized_portfolio_stats(port_val_bench['portval'])

    print("Date Range: {} to {}".format(start_date, end_date))

    # Cumulative Returns
    print("Cumulative Return of Manual Strategy: {}".format(cum_ret_ms))
    print("Cumulative Return of Strategy Learner: {}".format(cum_ret_sl))
    print("Cumulative Return of Benchmark: {}".format(cum_ret_bchm))

    # Standard Deviation
    print("Standard Deviation of Manual Strategy: {}".format(std_daily_ret_ms))
    print("Standard Deviation of Strategy Learner: {}".format(std_daily_ret_sl))
    print("Standard Deviation of Benchmark: {}".format(std_daily_ret_bchm))

    # Average Daily Return
    print("Average Daily Return of Manual Strategy: {}".format(avg_daily_ret_sl))
    print("Average Daily Return of Strategy Learner: {}".format(avg_daily_ret_sl))
    print("Average Daily Return of Benchmark: {}".format(avg_daily_ret_bchm))

    # Sharpe Ratio
    print("Sharpe Ratio of Manual Strategy: {}".format(sharpe_ratio_sl))
    print("Sharpe Ratio of Strategy Learner: {}".format(sharpe_ratio_sl))
    print("Sharpe Ratio of Benchmark: {}".format(sharpe_ratio_bchm))

    # Final Portfolio value
    print("Final Portfolio Manual Strategy: {}".format(port_val_sl['portval'][-1]))
    print("Final Portfolio Strategy Learner: {}".format(port_val_sl['portval'][-1]))
    print("Final Benchmark Value: {}".format(port_val_bench['portval'][-1]))

    # Normalize all DataFrames
    port_val_ms /= port_val_ms.iloc[0]
    port_val_sl /= port_val_sl.iloc[0]
    port_val_bench /= port_val_bench.iloc[0]

    # Rename columns to set legend and colors
    port_val_ms.rename(columns={'portval': 'Manual Strategy'}, inplace=True)
    port_val_sl.rename(columns={'portval': 'Strategy Learner'}, inplace=True)
    port_val_bench.rename(columns={'portval': 'Benchmark'}, inplace=True)

    # Set colors
    colors = {'Benchmark': 'green', 'Manual Strategy': 'red', 'Optimized Learner': 'yellow', 'Strategy Learner': '#1f77b4'}

    # Setup plot info and save fig
    ax = pd.concat([port_val_ms, port_val_sl, port_val_bench], axis=1).plot(title="Normalized Benchmark with Manual Strategy and Strategy Learner", fontsize=12, color=colors)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio")
    plt.grid(True, 'both')

    if save_fig:
        plt.savefig(fig_name, dpi=550)
    else:
        plt.show()


if __name__ == '__main__':
    experiment1()
