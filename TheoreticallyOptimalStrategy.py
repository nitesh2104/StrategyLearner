from marketsimcode import compute_portvals, compute_optimized_portfolio_stats
from util import get_data

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


def author():
    return "nitarora"


def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    if not isinstance(symbol, list):
        symbol = [symbol]

    port_val = get_data(symbol, dates=pd.date_range(sd, ed))
    port_val.fillna(method='ffill', inplace=True)
    port_val.fillna(method='bfill', inplace=True)

    net_shares = 0
    for date, _ in port_val.iterrows():
        current_price = port_val.loc[date, symbol][0]
        future_price = port_val.shift(-1).loc[date, symbol][0]

        if current_price < future_price:
            if net_shares == -1000:
                port_val.loc[date, 'shares'] = 2000
                net_shares += 2000

            elif net_shares == 0:
                port_val.loc[date, 'shares'] = 1000
                net_shares += 1000

        elif current_price > future_price:
            if net_shares == 0:
                port_val.loc[date, 'shares'] = -1000
                net_shares -= 1000

            elif net_shares == 1000:
                port_val.loc[date, 'shares'] = -2000
                net_shares -= 2000

    if net_shares != 0:
        if net_shares < 0:
            port_val.loc[ed, 'shares'] = -net_shares
        else:
            port_val.loc[ed, 'shares'] = net_shares

    return port_val[['shares']]


def bench_mark(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    port_val = get_data([symbol], pd.date_range(sd, ed))
    return pd.DataFrame(data=[1000, -1000], index=[port_val.index[0], port_val.index[port_val.shape[0] - 1]], columns=['shares'])


def plt_data(port_vals_bchm_norm, port_vals_norm, plot=True):
    if plot:
        new_pd = pd.concat([port_vals_bchm_norm, port_vals_norm], axis=1)
        ax =  new_pd.plot(title="Normalized Benchmark with Theoretically Optimum Strategy", fontsize=12, color=["green", "red"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Portfolio")
        plt.grid(True, 'both')
        plt.savefig("TOS.png", dpi=500)
        plt.close()


if __name__ == '__main__':
    start_date = '2008-1-1'
    end_date = '2009-12-31'
    symbols = 'JPM'

    portval_tos = testPolicy_tos(symbols, start_date, end_date, 100000)
    port_val_tos = compute_portvals(orders=portval_tos, start_val=100000, commission=0.00, impact=0.00)

    portval_bench = bench_mark(symbols, start_date, end_date, 100000)
    port_val_bench = compute_portvals(orders=portval_bench, start_val=100000, commission=0.00, impact=0.00)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_optimized_portfolio_stats(port_val_tos['portval'])
    cum_ret_bchm, avg_daily_ret_bchm, std_daily_ret_bchm, sharpe_ratio_bchm = compute_optimized_portfolio_stats(port_val_bench['portval'])

    print("Date Range: {} to {}".format(start_date, end_date))
    print("Cumulative Return of Portfolio: {}".format(cum_ret))
    print("Cumulative Return of Benchmark: {}".format(cum_ret_bchm))
    print("Standard Deviation of Portfolio: {}".format(std_daily_ret))
    print("Standard Deviation of Benchmark: {}".format(std_daily_ret_bchm))
    print("Average Daily Return of Portfolio: {}".format(avg_daily_ret))
    print("Average Daily Return of Benchmark: {}".format(avg_daily_ret_bchm))
    print("Sharpe Ratio of Portfolio: {}".format(sharpe_ratio))
    print("Sharpe Ratio of Benchmark: {}".format(sharpe_ratio_bchm))
    print("Final Portfolio Value: {}".format(port_val_tos['portval'][-1]))
    print("Final Benchmark Value: {}".format(port_val_bench['portval'][-1]))

    port_vals_bchm_norm = port_val_bench / port_val_bench.iloc[0]
    port_vals_norm = port_val_tos / port_val_tos.iloc[0]
    port_vals_bchm_norm.rename(columns={'portval': 'Portval Benchmark'}, inplace=True)
    port_vals_norm.rename(columns={'portval': 'Portval TOS'}, inplace=True)

    colors = {'Portval Benchmark': 'green', 'Portval TOS': 'red'}
    ax = pd.concat([port_vals_bchm_norm, port_vals_norm], axis=1).plot(title="Normalized Benchmark with Theoretically Optimum Strategy", fontsize=12, color=colors)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio")
    plt.grid(True, 'both')
    plt.savefig('TOS.jpg', dpi=550)
