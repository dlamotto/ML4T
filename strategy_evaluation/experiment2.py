import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
from util import get_data

from ManualStrategy import testPolicy
import StrategyLearner as sl
from marketsimcode import compute_portvals


def author():
    return 'dlamotto3'


def run_experiment(symbol, sd, ed, sv):
    dates = pd.date_range(sd, ed)
    # prices_all = get_data(symbol, dates)

    # Initialize StrategyLearner with impact = 0.0
    impact1 = 0.0
    strategy_learner_1 = sl.StrategyLearner(False, impact=impact1, commission=0.0)
    strategy_learner_1.add_evidence(symbol, sd, ed, sv)
    learner_trades_1 = strategy_learner_1.testPolicy(symbol, sd, ed, sv)
    learner_portvals_1 = compute_portvals(learner_trades_1, start_val=sv)


    # Initialize StrategyLearner with impact =0.005
    impact2 = 0.005
    strategy_learner_2 = sl.StrategyLearner(False, impact=impact2, commission=0.0)
    strategy_learner_2.add_evidence(symbol, sd, ed, sv)
    learner_trades_2 = strategy_learner_2.testPolicy(symbol, sd, ed, sv)
    learner_portvals_2 = compute_portvals(learner_trades_2, start_val=sv)

    # Initialize StrategyLearner with impact = 0.0005
    impact3 = 0.0005
    strategy_learner_3= sl.StrategyLearner(False, impact=impact3, commission=0.0)
    strategy_learner_3.add_evidence(symbol, sd, ed, sv)
    learner_trades_3 = strategy_learner_3.testPolicy(symbol, sd, ed, sv)
    learner_portvals_3 = compute_portvals(learner_trades_3, start_val=sv)

    return learner_portvals_1, impact1, learner_portvals_2, impact2, learner_portvals_3, impact3


def plot_results(learner_portvals_1, learner_portvals_2, learner_portvals_3, title):
    register_matplotlib_converters()
    plt.figure(figsize=(10, 6))
    plt.plot(learner_portvals_1, label="Learner w/ Impact = 0.0")
    plt.plot(learner_portvals_2, label="Learner w/ Impact = 0.005")
    plt.plot(learner_portvals_3, label="Learner w/ Impact = 0.0005")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Normalized)")
    plt.legend()
    plt.savefig("images/exp2_{}.png".format(title))
    plt.show()


def stats(portval, impact):
    # daily return percentage
    drp = (portval / portval.shift(1)) - 1
    drp = drp[1:]

    # cumulative  return
    cr = (portval.iloc[-1] / portval.iloc[0]) - 1

    # std of daily returns
    sddr = drp.std()

    # mean of daily returns
    adr = drp.mean()

    # sharpe ratio
    alpha = np.sqrt(252.0)
    sr = (alpha * adr) / sddr
    print("Strategy Learner with impact {}".format(impact))
    print("Cum return: " + str(cr))
    print("Stdev of daily returns: " + str(sddr))
    print("Mean of daily returns: " + str(adr))
    print("Sharpe Ratio: " + str(sr))
    print("\n")


def report_2():
    symbol = "JPM"
    sv = 100000

    # In-sample period
    sd_insample = dt.datetime(2008, 1, 1)
    ed_insample = dt.datetime(2009, 12, 31)
    learner_portvals_1, impact1, learner_portvals_2, impact2, learner_portvals_3, impact3 = run_experiment(symbol, sd_insample, ed_insample, sv)

    # get the metrics
    stats(learner_portvals_1, impact1)
    stats(learner_portvals_2, impact2)
    stats(learner_portvals_3, impact3)

    # normalize the learners
    learner_portvals_1_norm = learner_portvals_1 / learner_portvals_1.iloc[0]
    learner_portvals_2_norm = learner_portvals_2 / learner_portvals_2.iloc[0]
    learner_portvals_3_norm = learner_portvals_3 / learner_portvals_3.iloc[0]

    plot_results(learner_portvals_1_norm, learner_portvals_2_norm, learner_portvals_3_norm, "In-Sample Comparison")


if __name__ == "__main__":
    report_2()