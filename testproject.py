import ManualStrategy as ms
import marketsimcode

save_fig = True

# Compute TOS / Benchmark
start_date = '2008-1-1'
end_date = '2009-12-31'
symbols = 'JPM'

portval_tos = ms.testPolicy(symbols, start_date, end_date, 100000)
port_val_tos = marketsimcode.compute_portvals(orders=portval_tos, start_val=100000, commission=0.00, impact=0.00)

portval_bench = ms.bench_mark(symbols, start_date, end_date, 100000)
port_val_bench = marketsimcode.compute_portvals(orders=portval_bench, start_val=100000, commission=0.00, impact=0.00)

cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = marketsimcode.compute_optimized_portfolio_stats(port_val_tos['portval'])
cum_ret_bchm, avg_daily_ret_bchm, std_daily_ret_bchm, sharpe_ratio_bchm = marketsimcode.compute_optimized_portfolio_stats(port_val_bench['portval'])

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

ms.plt_data(port_vals_bchm_norm, port_vals_norm, plot=save_fig)

if __name__ == '__main__':
    pass
