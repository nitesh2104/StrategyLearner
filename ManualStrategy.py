import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt

from indicators import calculate_williamsR, calculate_RSI_EMV, calculate_momentum
from marketsimcode import compute_portvals, compute_optimized_portfolio_stats
from util import get_data


def testPolicy(symbol="AAPL", sd='2010-01-01', ed='2011-12-31', sv=100000):
    start_date = sd
    end_date = ed

    port_val = get_data([symbol], dates=pd.date_range(start_date, end_date))
    port_val.drop(columns=["SPY"], inplace=True)
    port_val.fillna(method='ffill', inplace=True)
    port_val.fillna(method='bfill', inplace=True)

    net_shares = 0

    # Get all indicators port_vals
    williamsR = calculate_williamsR(port_val.copy(), window=4, plot=False, ret_val=True, symbol=symbol)[['williams']]
    rsi = calculate_RSI_EMV(port_val.copy(), window=4, plot=False, ret_val=True, symbol=symbol)[['RSI_EMV']]
    momentum = calculate_momentum(port_val.copy(), window=8, plot=False, ret_val=True, symbol=symbol)[['momentum']]

    port_val['williamsR'] = williamsR
    port_val['rsi'] = rsi
    port_val['momentum'] = momentum

    port_val.fillna(method='ffill', inplace=True)
    port_val.fillna(method='bfill', inplace=True)

    for date, _ in port_val.iterrows():
        momentum = port_val.loc[date, 'momentum']
        williamsR = port_val.loc[date, 'williamsR']

        rsi_index_today = port_val.loc[date, 'rsi']
        rsi_index_yesterday = port_val[['rsi']].shift(1).fillna(0).loc[date][0]

        rsi_upper_band = 0.7 * port_val['rsi'].max()
        rsi_lower_band = 0.3 * port_val['rsi'].max()

        # rsi today is less than 30 %
        if rsi_index_today < rsi_lower_band:
            if rsi_index_yesterday <= rsi_lower_band:
                port_val.loc[date, 'shares'] = None
            elif rsi_index_yesterday > rsi_lower_band or williamsR < -60 or momentum < -0.25:  # BUY
                if net_shares == 0:
                    port_val.loc[date, 'shares'] = 1000
                    net_shares += 1000
                elif net_shares == -1000:
                    port_val.loc[date, 'shares'] = 2000
                    net_shares += 2000

        # rsi today is between 30% and 70%
        elif rsi_lower_band < rsi_index_today < rsi_upper_band:
            if rsi_index_yesterday < rsi_lower_band or williamsR < -60 or momentum < -0.25:
                if net_shares == 0:
                    port_val.loc[date, 'shares'] = 1000
                    net_shares += 1000
                elif net_shares == -1000:
                    port_val.loc[date, 'shares'] = 2000
                    net_shares += 2000
            elif rsi_lower_band < rsi_index_yesterday < rsi_upper_band:
                port_val.loc[date, 'shares'] = None
            elif rsi_index_yesterday > 70 or williamsR > -50:
                if net_shares == 0:
                    port_val.loc[date, 'shares'] = -1000
                    net_shares -= 1000
                elif net_shares == 1000:
                    port_val.loc[date, 'shares'] = -2000
                    net_shares -= 2000

        # rsi today is greater than 70%
        elif rsi_index_today > rsi_upper_band:
            if rsi_index_yesterday >= rsi_upper_band:
                port_val.loc[date, 'shares'] = None
            elif rsi_index_yesterday < rsi_upper_band or williamsR > -50:  # SELL
                if net_shares == 0:
                    port_val.loc[date, 'shares'] = -1000
                    net_shares -= 1000
                elif net_shares == 1000:
                    port_val.loc[date, 'shares'] = -2000
                    net_shares -= 2000

        # Something else is going on here
        else:
            print(f"Wrong rsi_index_today: {rsi_index_today}")

    if net_shares != 0:
        if net_shares < 0:
            port_val.loc[ed, 'shares'] = -net_shares
        else:
            port_val.loc[ed, 'shares'] = net_shares
    port_val.dropna(inplace=True)
    return port_val[['shares']]


def bench_mark(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    port_val = get_data([symbol], pd.date_range(sd, ed))
    return pd.DataFrame(data=[1000, -1000], index=[port_val.index[0], port_val.index[port_val.shape[0] - 1]], columns=['shares'])


def author():
    return "narora62"


def plt_data(port_vals_bchm_norm, port_vals_norm, plot=False):
    if plot:
        new_pd = pd.concat([port_vals_bchm_norm, port_vals_norm], axis=1)
        ax = new_pd.plot(title="Normalized Benchmark with Theoretically Optimum Strategy", fontsize=12, color=["green", "red"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Portfolio")
        plt.grid(True, 'both')
        plt.savefig("TOS.png", dpi=500)
        plt.close()


def run_manual_strategy(df_trades, symbols, start_date, end_date, save_fig=False, fig_name='Manual-Strategy.png'):
    port_val_ms = compute_portvals(orders=df_trades, start_val=100000, commission=9.95, impact=0.00)

    portval_bench = bench_mark(symbols, start_date, end_date, 100000)
    port_val_bench = compute_portvals(orders=portval_bench, start_val=100000, commission=0.00, impact=0.00)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_optimized_portfolio_stats(port_val_ms['portval'])
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
    print("Final Portfolio Value: {}".format(port_val_ms['portval'][-1]))
    print("Final Benchmark Value: {}".format(port_val_bench['portval'][-1]))

    port_val_bench /= port_val_bench.iloc[0]
    port_val_ms /= port_val_ms.iloc[0]

    port_val_bench.rename(columns={'portval': 'Benchmark'}, inplace=True)
    port_val_ms.rename(columns={'portval': 'Manual Strategy'}, inplace=True)

    colors = {'Benchmark': 'green', 'Manual Strategy': 'red'}
    ax = pd.concat([port_val_bench, port_val_ms], axis=1).plot(title="Normalized Benchmark with Manual Strategy", fontsize=12, color=['green', 'red'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio")
    plt.grid(True, 'both')
    for date, _ in df_trades.iterrows():
        if df_trades.loc[date, 'shares'] > 0:
            plt.axvline(x=date, color='#1f77b4', lw=1)
        else:
            plt.axvline(x=date, color='black', lw=1)
    if save_fig:
        plt.savefig(fig_name, dpi=550)
    else:
        plt.show()

if __name__ == '__main__':
    start_date = '2008-1-1'
    end_date = '2009-12-31'
    symbols = 'JPM'

    portval_ms = testPolicy(symbols, start_date, end_date, 100000)
    port_val_ms = compute_portvals(orders=portval_ms, start_val=100000, commission=9.95, impact=0.00)

    portval_bench = bench_mark(symbols, start_date, end_date, 100000)
    port_val_bench = compute_portvals(orders=portval_bench, start_val=100000, commission=0.00, impact=0.00)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_optimized_portfolio_stats(port_val_ms['portval'])
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
    print("Final Portfolio Value: {}".format(port_val_ms['portval'][-1]))
    print("Final Benchmark Value: {}".format(port_val_bench['portval'][-1]))

    port_val_bench /=  port_val_bench.iloc[0]
    port_val_ms /= port_val_ms.iloc[0]

    port_val_bench.rename(columns={'portval': 'Benchmark'}, inplace=True)
    port_val_ms.rename(columns={'portval': 'Manual Strategy'}, inplace=True)

    colors = {'Benchmark': 'green', 'Manual Strategy': 'red'}
    ax = pd.concat([port_val_bench, port_val_ms], axis=1).plot(title="Normalized Benchmark with Manual Strategy (Out-Sample)", fontsize=12, color=colors)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio")
    plt.grid(True, 'both')
    for date, _ in portval_ms.iterrows():
        if portval_ms.loc[date, 'shares'] > 0:
            plt.axvline(x=date, color='#1f77b4', lw=1)
        else:
            plt.axvline(x=date, color='black', lw=1)
    plt.savefig('Manual-Strategy.jpg', dpi=550)
