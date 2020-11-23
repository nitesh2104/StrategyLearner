import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt

import util as ut
from BagLearner import BagLearner
from RTLearner import RTLearner
from indicators import calculate_momentum, calculate_RSI_EMV, calculate_williamsR
from marketsimcode import compute_portvals, compute_optimized_portfolio_stats


class StrategyLearner(object):

    def __init__(self, verbose=False, impact=0.0, commission=0.0, learner=BagLearner, leaf_size=5):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.leaf_size = leaf_size
        self.learner = learner(learner=RTLearner, kwargs={'leaf_size': self.leaf_size}, bags=15, boost=False, verbose=self.verbose)
        self.train_x = None
        self.train_y = None

    def discretize(self, df, symbol, window=5):
        prices = df[symbol]
        df['ratio'] = pd.Series(0, index=df.index)
        df['train_y'] = pd.Series(0, index=df.index)
        for i in range(prices.shape[0] - window):
            ratio = (prices[i + window] / prices[i]) - 1
            df['ratio'].iloc[i] = ratio
            if ratio > self.impact * 10:
                df['train_y'].iloc[i] = 1
            elif ratio < -1 * (self.impact * 20):
                df['train_y'].iloc[i] = -1
            else:
                df['train_y'].iloc[i] = 0

        if self.verbose:
            plt.rcParams['axes.grid'] = True
            plt.subplot(3, 1, 1)
            ax = df[symbol].plot(title=f'Stock Price ({symbol})', fontsize=8)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")

            plt.subplot(3, 1, 2)
            ax = df['ratio'].plot(title=f'Ratio, Window={window} days', fontsize=8)
            ax.set_xlabel("Date")

            plt.subplot(3, 1, 3)
            ax = df['train_y'].plot(title=f'train_y, Window={window} days', fontsize=8)
            ax.set_xlabel("Date")
            plt.show()

        return df['train_y'].to_numpy()

    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=100000):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates, addSPY=False)  # automatically adds SPY
        prices_all.dropna(inplace=True)

        prices_all = self.compute_indicators(prices_all)

        prices_all_no_0 = prices_all.dropna(0)
        self.train_x = prices_all_no_0[['rsi', 'momentum', 'williamsR']].to_numpy()
        self.train_y = self.discretize(df=prices_all_no_0, symbol=symbol, window=15)

        self.learner.add_evidence(self.train_x, self.train_y)

        return prices_all

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=100000):

        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates, addSPY=False)  # automatically adds SPY
        prices_all.dropna(inplace=True)

        prices_all = self.compute_indicators(prices_all)
        prices_all.dropna(0, inplace=True)

        prices_all['Y_out'] = self.learner.query(prices_all[['rsi', 'momentum', 'williamsR']].to_numpy())

        if self.verbose:
            plt.rcParams['axes.grid'] = True
            plt.subplot(2, 1, 1)
            ax = prices_all[symbol].plot(title=f'Stock Price ({symbol})', fontsize=8)
            ax.set_xlabel("Date")
            plt.legend()
            ax.set_ylabel("Normalized Price")

            plt.subplot(2, 1, 2)
            ax = prices_all['Y_out'].plot(title=f'Momentum, Window= days', fontsize=8, color='green')
            ax.set_xlabel("Date")
            plt.legend()
            plt.subplots_adjust(hspace=0.9)

            plt.plot()
            plt.show()
            plt.close()

        net_shares = 0
        for date, _ in prices_all.iterrows():
            position = prices_all[["Y_out"]].loc[date][0]

            # BUY
            if position > 0.5:
                if net_shares == 0:
                    prices_all.loc[date, 'shares'] = 1000
                    net_shares += 1000
                elif net_shares == -1000:
                    prices_all.loc[date, 'shares'] = 2000
                    net_shares += 2000
            # SELL
            elif position < 0:
                if net_shares == 0:
                    prices_all.loc[date, 'shares'] = -1000
                    net_shares -= 1000
                elif net_shares == 1000:
                    prices_all.loc[date, 'shares'] = -2000
                    net_shares -= 2000
            # HOLD
            else:
                prices_all.loc[date, 'shares'] = None

        if net_shares != 0:
            if net_shares < 0:
                prices_all.loc[ed, 'shares'] = -net_shares
            else:
                prices_all.loc[ed, 'shares'] = net_shares

        if self.verbose:
            print(type(prices_all))  # it better be a DataFrame!
        if self.verbose:
            print(prices_all)
        if self.verbose:
            print(prices_all)
        prices_all.dropna(inplace=True)
        return prices_all[['shares']]

    def compute_indicators(self, prices_all):
        rsi = calculate_RSI_EMV(prices_all.copy(), window=4, plot=False, ret_val=True)
        williamsR = calculate_williamsR(prices_all.copy(), window=14, plot=False, ret_val=True)
        momentum = calculate_momentum(prices_all.copy(), window=14, plot=False, ret_val=True)
        prices_all['rsi'] = rsi[['RSI_EMV']]
        prices_all['momentum'] = momentum[['momentum']]
        prices_all['williamsR'] = williamsR[['williams']]
        return prices_all

    def bench_mark(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        port_val = ut.get_data([symbol], pd.date_range(sd, ed))
        return pd.DataFrame(data=[1000, -1000], index=[port_val.index[0], port_val.index[port_val.shape[0] - 1]], columns=['shares'])

    def author(self):
        return "narora62"

    def run_strategy_learner(self, df_trades, symbols, start_date, end_date, save_fig, fig_name='Strategy-Learner.png'):
        port_val_sl = compute_portvals(orders=df_trades, start_val=100000, commission=9.95, impact=0.005)

        portval_bench = self.bench_mark(symbols, start_date, end_date, 100000)
        port_val_bench = compute_portvals(orders=portval_bench, start_val=100000, commission=9.95, impact=0.005)

        cum_ret_sl, avg_daily_ret_sl, std_daily_ret_sl, sharpe_ratio_sl = compute_optimized_portfolio_stats(port_val_sl['portval'])
        cum_ret_bchm, avg_daily_ret_bchm, std_daily_ret_bchm, sharpe_ratio_bchm = compute_optimized_portfolio_stats(port_val_bench['portval'])

        print("Date Range: {} to {}".format(start_date, end_date))

        # Cumulative Returns
        print("Cumulative Return of Strategy Learner: {}".format(cum_ret_sl))
        print("Cumulative Return of Benchmark: {}".format(cum_ret_bchm))

        # Standard Deviation
        print("Standard Deviation of Strategy Learner: {}".format(std_daily_ret_sl))
        print("Standard Deviation of Benchmark: {}".format(std_daily_ret_bchm))

        # Average Daily Return
        print("Average Daily Return of Strategy Learner: {}".format(avg_daily_ret_sl))
        print("Average Daily Return of Benchmark: {}".format(avg_daily_ret_bchm))

        # Sharpe Ratio
        print("Sharpe Ratio of Strategy Learner: {}".format(sharpe_ratio_sl))
        print("Sharpe Ratio of Benchmark: {}".format(sharpe_ratio_bchm))

        # Final Portfolio value
        print("Final Portfolio Strategy Learner: {}".format(port_val_sl['portval'][-1]))
        print("Final Benchmark Value: {}".format(port_val_bench['portval'][-1]))

        # Normalize all DataFrames
        port_val_sl_norm = port_val_sl / port_val_sl.iloc[0]
        port_vals_bchm_norm = port_val_bench / port_val_bench.iloc[0]

        # Rename columns to set legend and colors
        port_val_sl_norm.rename(columns={'portval': 'Strategy Learner'}, inplace=True)
        port_vals_bchm_norm.rename(columns={'portval': 'Benchmark'}, inplace=True)

        # Set colors
        colors = {'Benchmark': 'green', 'Strategy Learner': '#1f77b4'}

        # Setup plot info and save fig
        ax = pd.concat([port_vals_bchm_norm, port_val_sl_norm], axis=1).plot(title="Normalized Benchmark with Strategy Learner", fontsize=12, color=['green', 'red'])
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Portfolio")
        plt.grid(True, 'both')

        if save_fig:
            plt.savefig(fig_name, dpi=550)
        else:
            plt.show()

# if __name__ == '__main__':
#     start_date = dt.datetime(2008, 1, 1)
#     end_date = dt.datetime(2009, 12, 31)
#     symbols = 'JPM'
#     start_date_o = dt.datetime(2008, 1, 1)
#     end_date_o = dt.datetime(2011, 12, 31)
#
#     portval_ms = testPolicy_ms(symbols, start_date_o, end_date_o, 100000)
#     port_val_ms = compute_portvals(orders=portval_ms, start_val=100000, commission=9.95, impact=0.005)
#
#     sl = StrategyLearner(verbose=False, impact=0.005, commission=9.95)
#     sl.add_evidence(symbol=symbols, sd=start_date, ed=end_date, sv=100000)  # training phase
#
#     portval_sl = sl.testPolicy(symbols, start_date_o, end_date_o, 100000)
#     port_val_sl = compute_portvals(orders=portval_sl, start_val=100000, commission=9.95, impact=0.005)
#
#     portval_bench = sl.bench_mark(symbols, start_date_o, end_date_o, 100000)
#     port_val_bench = compute_portvals(orders=portval_bench, start_val=100000, commission=9.95, impact=0.005)
#
#     cum_ret_ms, avg_daily_ret_ms, std_daily_ret_ms, sharpe_ratio_ms = compute_optimized_portfolio_stats(port_val_ms['portval'])
#     cum_ret_sl, avg_daily_ret_sl, std_daily_ret_sl, sharpe_ratio_sl = compute_optimized_portfolio_stats(port_val_sl['portval'])
#     cum_ret_bchm, avg_daily_ret_bchm, std_daily_ret_bchm, sharpe_ratio_bchm = compute_optimized_portfolio_stats(port_val_bench['portval'])
#
#     print("Date Range: {} to {}".format(start_date, end_date))
#
#     # Cumulative Returns
#     print("Cumulative Return of Manual Learner: {}".format(cum_ret_ms))
#     print("Cumulative Return of Strategy Learner: {}".format(cum_ret_sl))
#     print("Cumulative Return of Benchmark: {}".format(cum_ret_bchm))
#
#     # Standard Deviation
#     print("Standard Deviation of Manual Learner: {}".format(std_daily_ret_ms))
#     print("Standard Deviation of Strategy Learner: {}".format(std_daily_ret_sl))
#     print("Standard Deviation of Benchmark: {}".format(std_daily_ret_bchm))
#
#     # Average Daily Return
#     print("Average Daily Return of Manual Learner: {}".format(avg_daily_ret_sl))
#     print("Average Daily Return of Strategy Learner: {}".format(avg_daily_ret_sl))
#     print("Average Daily Return of Benchmark: {}".format(avg_daily_ret_bchm))
#
#     # Sharpe Ratio
#     print("Sharpe Ratio of Manual Learner: {}".format(sharpe_ratio_sl))
#     print("Sharpe Ratio of Strategy Learner: {}".format(sharpe_ratio_sl))
#     print("Sharpe Ratio of Benchmark: {}".format(sharpe_ratio_bchm))
#
#     # Final Portfolio value
#     print("Final Portfolio Manual Learner: {}".format(port_val_sl['portval'][-1]))
#     print("Final Portfolio Strategy Learner: {}".format(port_val_sl['portval'][-1]))
#     print("Final Benchmark Value: {}".format(port_val_bench['portval'][-1]))
#
#     # Normalize all DataFrames
#     port_vals_norm = port_val_ms / port_val_ms.iloc[0]
#     port_val_sl_norm = port_val_sl / port_val_sl.iloc[0]
#     port_vals_bchm_norm = port_val_bench / port_val_bench.iloc[0]
#
#     # Rename columns to set legend and colors
#     port_vals_norm.rename(columns={'portval': 'Manual Strategy'}, inplace=True)
#     port_val_sl_norm.rename(columns={'portval': 'Strategy Learner'}, inplace=True)
#     port_vals_bchm_norm.rename(columns={'portval': 'Benchmark'}, inplace=True)
#
#     # Set colors
#     colors = {'Benchmark': 'green', 'Manual Strategy': 'red', 'Optimized Learner': 'yellow', 'Strategy Learner': '#1f77b4'}
#
#     # Setup plot info and save fig
#     ax = pd.concat([port_vals_bchm_norm, port_vals_norm, port_val_sl_norm], axis=1).plot(title="Normalized Benchmark with Manual Strategy and Strategy Learner", fontsize=12, color=colors)
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Normalized Portfolio")
#     plt.grid(True, 'both')
#     plt.savefig('Exp-1.jpg', dpi=550)
