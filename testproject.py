import datetime as dt

from ManualStrategy import run_manual_strategy
from ManualStrategy import testPolicy as testPolicy_ms
from StrategyLearner import StrategyLearner
from experiment1 import experiment1
from experiment2 import experiment2

save_fig = True


def author():
    return 'narora62'


def main():
    symbol = "JPM"
    test_out_sample = True
    hr = "-" * 80
    print(hr)
    print("Running Manual Strategy...\n")
    df_trades_ms = testPolicy_ms(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    print("\tRunning In-Sample Strategy...\n")
    run_manual_strategy(df_trades_ms, symbols='JPM', start_date=dt.datetime(2008, 1, 1), end_date=dt.datetime(2009, 12, 31), save_fig=save_fig, fig_name='MS-IN.png')

    print("\tRunning Out-Sample Strategy...\n")
    df_trades_ms = testPolicy_ms(symbol=symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    run_manual_strategy(df_trades_ms, symbols='JPM', start_date=dt.datetime(2010, 1, 1), end_date=dt.datetime(2011, 12, 31), save_fig=save_fig, fig_name='MS-OUT.png')

    print(hr)
    print("Running Strategy Learner...\n")
    learner = StrategyLearner(verbose=False, impact=0.005, commission=9.95)  # constructor
    learner.add_evidence(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # training phase
    print("\tRunning In-Sample Strategy Learner...\n")
    df_trades_sl = learner.testPolicy(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # testing phase
    learner.run_strategy_learner(df_trades=df_trades_sl, symbols=symbol, start_date=dt.datetime(2008, 1, 1), end_date=dt.datetime(2009, 12, 31), save_fig=save_fig, fig_name='SL-IN.png')

    if test_out_sample:
        print("\tRunning Out-Sample Strategy Learner...\n")
        df_trades_sl = learner.testPolicy(symbol=symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)  # testing phase
        learner.run_strategy_learner(df_trades=df_trades_sl, symbols=symbol, start_date=dt.datetime(2010, 1, 1), end_date=dt.datetime(2011, 12, 31), save_fig=save_fig, fig_name='SL-OUT.png')

    print(hr)
    print("Running experiment 1...\n")
    experiment1(save_fig=save_fig, fig_name='Experiment1.png')

    print(hr)
    print("Running experiment 2...\n")
    experiment2(save_fig=save_fig, fig_name='Experiment2.png')
    print(hr)


if __name__ == '__main__':
    main()
