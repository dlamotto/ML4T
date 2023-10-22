"""
Student Name: Danielle LaMotto
GT User ID: dlamotto3
GT ID: 903951588
"""


from marketsimcode import *
import datetime as dt
from util import get_data
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def graphs(portvals, portvals_benchmark):
    portvals_benchmark['value'] = portvals_benchmark['value']/portvals_benchmark['value'][0]
    portvals['value'] = portvals['value']/ portvals['value'][0]

    plt.plot(portvals, label="theoretically optimal portfolio", color="red")
    plt.plot(portvals_benchmark, label="benchmark portfolio", color="purple")
    plt.xlabel("Dates: 1/1/08 - 12/31/09")
    plt.ylabel("Normalized Prices")
    plt.legend()
    plt.xticks(rotation=30)
    plt.savefig("images/TOS.png")
    plt.close()


def stats(portvals, portvals_benchmark):
    portvals, portvals_benchmark = portvals['value'], portvals_benchmark['value']

    tos_cum_ret = portvals.iloc[-1]/portvals.iloc[0] - 1
    bm_cum_ret = portvals_benchmark.iloc[-1]/portvals_benchmark.iloc[0] - 1
    tos_daily_ret = (portvals / portvals.shift(1) - 1).iloc[1:]
    bm_daily_ret = (portvals_benchmark / portvals_benchmark.shift(1) - 1).iloc[1:]
    tos_std = tos_daily_ret.std()
    bm_std = bm_daily_ret.std()
    tos_mean = tos_daily_ret.mean()
    bm_mean = bm_daily_ret.mean()
    line1 = "Theoretically Optimal Strategy"
    line2 = f"Cumulative Return: {tos_cum_ret}; STD: {tos_std}; Mean: {tos_mean}"
    line3 = "Benchmark"
    line4 = f"Cumulative Return: {bm_cum_ret}; STD: {bm_std}; Mean: {bm_mean}"
    data_lines = [line1, line2, line3, line4]

    with open('p6_results.txt', 'w') as file:
        for line in data_lines:
            file.write(line + '\n')


def testPolicy(symbol, sd, ed, sv):
    """
    Returns data that performs significantly better with DTLearner than LinRegLearner.
    The data set should include from 2 to 10 columns in X, and one column in Y.
    The data should contain from 10 (minimum) to 1000 (maximum) rows.

    :param symbol: the stock symbol to act on
    :type symbol: String
    :param sd: A DateTime object that represents the start date
    :type sd: DateTime object
    :param ed: A DateTime object that represents the end date
    :type ed: DateTime object
    :param sv:  Start value of the portfolio
    :type sv: int
    :return: Returns a single column data frame, indexed by date, whose values represent trades for each trading day
    :rtype: pandas.dataframe
    """
    # getting the stock data and cleaning it
    df = get_data([symbol], pd.date_range(sd, ed))
    prices = df[symbol].ffill().bfill()

    # copying the dataframe but with no values
    trade = pd.DataFrame(index=df.index)
    trade[symbol] = 0
    trading_days = trade.index

    # num of shares we already have
    position = 0

    # time to trade based on the future
    for day in range(len(trading_days) - 1):
        current_price = prices.loc[trading_days[day]]
        next_price = prices.loc[trading_days[day+1]]

        if next_price > current_price:
            trade_val = 1000 - position
        else:
            trade_val = -1000 - position

        trade.loc[trading_days[day]].loc[symbol] = trade_val
        position += trade_val

    # compute our portfolio values
    portvals = compute_portvals(trade, sv, commission=0.0, impact=0.0)

    # getting the benchmark portfolio values
    df_bm = get_data(['SPY'], pd.date_range(sd, ed))
    df_bm = df_bm.rename(columns={'SPY': symbol})
    df_bm[:] = 0
    df_bm.loc[df_bm.index[0]] = 1000 # investing in 1000 shares of JPM and holding
    portvals_bm = compute_portvals(df_bm, sv, commission=0.00, impact=0.00)

    # compute the stats
    stats(portvals, portvals_bm)
    # plot TOS vs Benchmark
    graphs(portvals, portvals_bm)
    return portvals




def author():
    return 'dlamotto3'