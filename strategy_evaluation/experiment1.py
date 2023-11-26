import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import StrategyLearner as sl
from ManualStrategy import testPolicy
from marketsimcode import compute_portvals


def author():
    return 'dlamotto3'


def run_experiment(symbol, sd, ed, sv, verbose):
    commission = 9.95
    impact = 0.005
    symbol = symbol
    # print(symbol)
  
    # Initialize StrategyLearner
    strategy_learner = sl.StrategyLearner(False, impact, commission)

    # Train and test StrategyLearner
    strategy_learner.add_evidence(symbol, sd, ed, sv)
    learner_trades = strategy_learner.testPolicy(symbol, sd, ed, sv)
    learner_portvals = compute_portvals(learner_trades, start_val=sv, commission=9.95, impact=0.005)

    # Get manual strategy trades and compute portfolio values
    manual_trades = testPolicy([symbol], sd, ed, sv)
    manual_portvals = compute_portvals(manual_trades, start_val=sv)

    # Benchmark: Invest in the stock and hold
    benchmark_trades = manual_trades.copy()
    benchmark_trades.iloc[:] = 0
    benchmark_trades.iloc[0] = 1000  # Assuming 1000 shares
    benchmark_portvals = compute_portvals(benchmark_trades, start_val=sv)

    # Normalize the portfolio values
    manual_portvals_normalized = manual_portvals / manual_portvals.iloc[0]
    learner_portvals_normalized = learner_portvals / learner_portvals.iloc[0]
    benchmark_portvals_normalized = benchmark_portvals / benchmark_portvals.iloc[0]

    return manual_portvals_normalized, learner_portvals_normalized, benchmark_portvals_normalized


def graph(manual_portvals, learner_portvals, benchmark_portvals, title):
    register_matplotlib_converters()
    plt.figure(figsize=(10, 6))
    plt.plot(manual_portvals, label="Manual Strategy")
    plt.plot(learner_portvals, label="Strategy Learner")
    plt.plot(benchmark_portvals, label="Benchmark")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.savefig("images/exp1_{}.png".format(title))
    # plt.show()


def report_1():
    symbol = "JPM"
    sv = 100000

    # In-sample period
    sd_insample = dt.datetime(2008, 1, 1)
    ed_insample = dt.datetime(2009, 12, 31)
    manual_in, learner_in, benchmark_in = run_experiment(symbol, sd_insample, ed_insample, sv, False)
    graph(manual_in, learner_in, benchmark_in, "In-Sample Comparison")

    # Out-of-sample period
    sd_outsample = dt.datetime(2010, 1, 1)
    ed_outsample = dt.datetime(2011, 12, 31)
    manual_out, learner_out, benchmark_out = run_experiment(symbol, sd_outsample, ed_outsample, sv, False)
    graph(manual_out, learner_out, benchmark_out, "Out-of-Sample Comparison")


if __name__ == "__main__":
    report_1()
