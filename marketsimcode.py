import numpy as np
import pandas as pd

from util import get_data


def author():
    return "narora62"


def compute_portvals(orders="./orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005):
    # Keep legacy functionality
    if not isinstance(orders, pd.DataFrame):
        # Read the orders file and create a dataframe
        orders_df = pd.read_csv(orders, index_col="Date", parse_dates=True, na_values=["nan"])
    else:
        # Instead create orders
        orders_df = create_orders(orders)

    # Start and end date is continuous datetime range
    start_date = min(orders_df.index)
    end_date = max(orders_df.index)

    # Set of ticker symbols traded within the order df
    symbols = set(orders_df['Symbol'])

    # Create a df of all the above symbols containing their price chart
    portvals = get_data(symbols, pd.date_range(start_date, end_date))

    # Create cash flow column in above price chart df
    # This cash val will be used to show the overall portfolio val
    # i.e. For a specific date: Cash + (# of shares * share val) = Portfolio val
    # If we add commission and impact, the portfolio value goes further down when BUY
    # And when SELL happens, commission and impact lowers the portfolio value again
    portvals['Cash'] = pd.Series(start_val, index=portvals.index)

    # Keep track of number of shares
    for symbol in symbols:
        portvals[f"{symbol} Shares"] = pd.Series(0, index=portvals.index)

    for date, transaction in orders_df.iterrows():
        # For that date and symbol, get the closing share val
        share_price = portvals.loc[date, transaction['Symbol']]

        # Get the number of shares for that specific transaction
        no_of_shares = int(transaction['Shares'])

        if transaction['Order'] == "BUY":
            # Add number of shares bought in the respective <Symbol> + Share column
            portvals.loc[date:, f"{transaction['Symbol']} Shares"] += no_of_shares

            # Immediately reduce the cash val including commission and impact
            portvals.loc[date:, "Cash"] -= ((share_price * no_of_shares * (1 + impact)) + commission)

        elif transaction['Order'] == "SELL":
            # Reduce the number of shares for that symbol, can take negative vals
            portvals.loc[date:, f"{transaction['Symbol']} Shares"] -= no_of_shares

            # For that days share price, add the cash val to existing cash at hand
            portvals.loc[date:, "Cash"] += ((share_price * no_of_shares * (1 - impact)) - commission)

    # To compute portfolio value for each valid trading day
    # Here, the portfolio value is described as:
    # Cash at hand + (# of shares * share price)
    # Cash at hand has already paid the commission and impact, so that is included already
    for date, transaction in portvals.iterrows():
        symbol_value = 0
        # Multiple ticker symbols can be traded in 1 day, so we need to account for that
        for symbol in symbols:
            # Get the ticker value for that date
            symbol_value += portvals.loc[date, f"{symbol} Shares"] * transaction[symbol]
            # Add it to cash at hand
            portvals.loc[date, 'portval'] = portvals.loc[date, 'Cash'] + symbol_value

    # Return the portval column
    return portvals.loc[:, ['portval']]


def create_orders(orders_df):
    orders_df.dropna(subset=['shares'], inplace=True)
    orders_df['Symbol'] = pd.Series('JPM', index=orders_df.index)
    orders_df['Order'] = pd.Series(0, index=orders_df.index)
    orders_df['Shares'] = pd.Series(0, index=orders_df.index)

    for date, _ in orders_df.iterrows():
        if orders_df.loc[date, 'shares'] > 0:
            orders_df.loc[date:, 'Order'] = "BUY"
            orders_df.loc[date:, 'Shares'] = orders_df.loc[date, 'shares']

        elif orders_df.loc[date, 'shares'] < 0:
            orders_df.loc[date:, 'Order'] = "SELL"
            orders_df.loc[date:, 'Shares'] = -orders_df.loc[date, 'shares']

    return orders_df


def compute_optimized_portfolio_stats(port_val, rfr=0.0, sf=252.0):
    cr = (port_val[-1] / port_val[0]) - 1
    daily_return = (port_val / port_val.shift(1)) - 1
    daily_return = daily_return
    adr = daily_return.mean()
    sddr = daily_return.std()
    sr = np.sqrt(sf) * (daily_return - rfr).mean() / sddr
    return cr, adr, sddr, sr


def compute_portfolio(allocs, prices, sv=1):
    prices_normalized = prices / prices.iloc[0]
    pos_vals = prices_normalized * allocs * sv
    return pos_vals.sum(axis=1)


def test_code():
    orders_file = "./orders/orders-02.csv"
    portvals = compute_portvals(orders=orders_file, start_val=1000000)

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]
    else:
        print("warning, code did not return a DataFrame")

    orders_df = pd.read_csv(orders_file, index_col="Date", parse_dates=True, na_values=["nan"])
    start_date = min(orders_df.index)
    end_date = max(orders_df.index)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_optimized_portfolio_stats(portvals)

    SPY_df = get_data(["SPY"], pd.date_range(start_date, end_date))
    portval_SPY = compute_portfolio([1], SPY_df.loc[:, ['SPY']])
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_optimized_portfolio_stats(portval_SPY)

    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
