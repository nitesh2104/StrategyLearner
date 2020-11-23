import matplotlib.pyplot as plt
import pandas as pd

from util import get_data


def author():
    return "nitarora"


def calculate_SMA(port_val, window, plot=False, ret_val=False, symbol='JPM'):
    port_val = port_val / port_val.iloc[0]
    port_val["SMA"] = port_val[symbol].rolling(window=window).mean()
    port_val['SMA'] = port_val[symbol] / port_val['SMA']
    if plot:
        ax = port_val.plot(title='SMA (Simple Moving Average, Window=20 days)', fontsize=8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price")
        plt.grid(True, 'both')
        plt.legend()
        plt.savefig("SMA.png", dpi=500)
        plt.close()

    if ret_val:
        return port_val


def calculate_momentum(port_val, window, plot=False, ret_val=False, symbol='JPM'):
    port_val['momentum'] = (port_val / port_val.shift(-window)) - 1
    if plot:
        plt.rcParams['axes.grid'] = True
        plt.subplot(2, 1, 1)
        ax = port_val[symbol].plot(title=f'Stock Price ({symbol})', fontsize=8)
        ax.set_xlabel("Date")
        plt.legend()
        ax.set_ylabel("Normalized Price")

        plt.subplot(2, 1, 2)
        ax = port_val['momentum'].plot(title=f'Momentum, Window={window} days', fontsize=8, color='green')
        ax.set_xlabel("Date")
        plt.legend()
        plt.subplots_adjust(hspace=0.9)
        plt.savefig('momemtum.png', dpi=500)
        plt.close()

    if ret_val:
        return port_val


def calculate_RSI_EMV(port_val, window, plot=False, ret_val=False, symbol='JPM'):
    diff = port_val.dropna().diff()
    diff = diff[1:]

    up_diff, down_diff = diff.copy(), diff.copy()

    up_diff[up_diff < 0] = 0
    down_diff[down_diff > 0] = 0

    total_average_gain = up_diff.ewm(span=window).mean()
    total_average_loss = down_diff.abs().ewm(span=window).mean()

    rs_emv = abs(total_average_gain / total_average_loss)

    port_val['RSI_EMV'] = 100.0 - 100.0 / (1.0 + rs_emv)

    if plot:
        plt.rcParams['axes.grid'] = True
        plt.subplot(2, 1, 1)
        ax = port_val[symbol].plot(title=f'Stock Price ({symbol})', fontsize=8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        plt.legend()

        plt.subplot(2, 1, 2)
        ax = port_val['RSI_EMV'].plot(title='RSI (Relative Strength Index), Window: 14', fontsize=8)
        ax.set_xlabel("Date")
        plt.legend()

        ax.axhline(70, color='green', lw=1)
        ax.axhline(30, color='red', lw=1)
        ax.fill_between(port_val.index, 0, 0, where=port_val['RSI_EMV'] < 70, facecolor='green')
        ax.fill_between(port_val.index, 0, 0, where=port_val['RSI_EMV'] > 30, facecolor='red')
        plt.subplots_adjust(hspace=0.9)
        plt.savefig("RSI.png", dpi=500)
        plt.close()

    if ret_val:
        return port_val


def calculate_TRIX(port_val, window, plot=False, ret_val=False, symbol='JPM'):
    port_val = port_val / port_val.iloc[0]
    port_val['ex1'] = port_val[symbol].ewm(span=window, min_periods=1).mean()
    port_val['ex2'] = port_val['ex1'].ewm(span=window, min_periods=1).mean()
    port_val['ex3'] = port_val['ex2'].ewm(span=window, min_periods=1).mean()

    port_val['trix'] = 10000 * (port_val['ex3'].diff() / port_val['ex3'])
    if plot:
        plt.rcParams['axes.grid'] = True
        ax = port_val[['ex1', 'ex2', 'ex3', symbol]].plot(title=f'Stock Price {symbol}', fontsize=8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price")
        ax.legend()

        plt.savefig("Trix.png", dpi=500)
        plt.close()
    if ret_val:
        return port_val


def calculate_williamsR(port_val, window, plot=False, ret_val=False, symbol='AAPL'):
    port_val = port_val / port_val.iloc[0]
    max = port_val.rolling(window=window).max()
    min = port_val.rolling(window=window).min()
    port_val['williams'] = 100.0 * (port_val - max) / (max - min)
    if plot:
        plt.rcParams['axes.grid'] = True
        plt.subplot(2, 1, 1)
        ax = port_val[symbol].plot(title=f'Stock Price {symbol}', fontsize=8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price")
        plt.legend()

        plt.subplot(2, 1, 2)
        ax = port_val['williams'].plot(title='Williams%R Window: 14', fontsize=8, color='green')
        ax.set_xlabel("Date")
        plt.legend()
        plt.ylim(-110, 10)

        plt.subplots_adjust(hspace=0.9)
        plt.savefig("Williams.png", dpi=500)
        plt.close()

    if ret_val:
        return port_val


def run_indicators(symbols="JPM", sd="2008-01-01", ed="2009-12-31", save_fig=True):
    if not isinstance(symbols, list):
        symbols = [symbols]
    start_date = sd
    end_date = ed
    author()
    port_val = get_data(symbols, pd.date_range(start_date, end_date), addSPY=False)
    port_val.fillna(method='ffill', inplace=True)
    port_val.fillna(method='bfill', inplace=True)
    # calculate_TRIX(port_val.copy(), 18, save_fig)
    calculate_momentum(port_val.copy(), 20, save_fig)
    # calculate_williamsR(port_val.copy(), 14, save_fig)
    # calculate_SMA(port_val.copy(), 20, save_fig)
    # calculate_RSI_EMV(port_val.copy(), 14, save_fig)


if __name__ == '__main__':
    run_indicators()
