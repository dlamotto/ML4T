import pandas as pd
import datetime as dt
from util import get_data
from marketsimcode import compute_portvals
from indicators import *
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
def author():
    return 'dlamotto3'

def testPolicy(symbol, sd, ed, sv):
    # initializing dataframe w/ symbol
    # print(symbol)
    df = get_data(symbol, pd.date_range(sd, ed))
    df_price = df[symbol].ffill().bfill()
    # normalized dataframe
    normalized_df = df_price / df_price.iloc[0]

    df_trades = df[['SPY']]
    symbol_str = symbol[0]
    df_trades = df_trades.rename(columns={'SPY': symbol_str}).astype({symbol_str: 'int32'})
    df_trades[:] = 0
    # date range
    dates = df_trades.index

    # getting indicators
    sma = compute_sma(sd, ed, symbol, 20)[symbol]
    bbp = compute_bbp(sd, ed, symbol, 20)[symbol]
    momentum = compute_momentum(sd, ed, symbol, 20)[symbol]

    # track moves within the market
    position = 0
    last_action = 0

    # making trades
    for i in range(len(dates)):
        current_date = dates[i]
        last_action += 1

        # votes based where price lies based on indicator
        sma_flag = 1 if normalized_df.loc[current_date, symbol_str] > sma.loc[current_date, symbol_str] else -1 if\
            normalized_df.loc[current_date, symbol_str] < sma.loc[current_date, symbol_str] else 0
        bbp_flag = 1 if bbp.loc[current_date, symbol_str] < 0 else -1 if bbp.loc[current_date, symbol_str] > 1 else 0
        momentum_flag = 1 if momentum.loc[current_date, symbol_str] > 0 else -1 if momentum.loc[current_date, symbol_str] < 0 else 0

        # determine the trade action
        flag = sma_flag + bbp_flag + momentum_flag
        action = (1000 if flag >= 2 else -1000 if flag <= -2 else 0) - position
        # only execute trading action if last action was 3 or more days ago
        if last_action >= 3:
            df_trades.at[dates[i], symbol] = action
            position += action
            last_action = 0

    return df_trades


def benchmark(sd, ed, sv):
    df_bm = get_data(['SPY'], pd.date_range(sd, ed))
    df_bm = df_bm.rename(columns={'SPY': 'JPM'}).astype({'JPM': 'int32'})
    df_bm[:] = 0
    df_bm.loc[df_bm.index[0]] = 1000
    portvals_bm = compute_portvals(df_bm, sv, 9.95, 0.005)
    return portvals_bm


def stats(bm, msp):
    bm, msp = bm['value'], msp['value']

    # cumm return
    cr_bm = bm[-1] / bm[0] - 1
    cr_msp = msp[-1] / msp[0] - 1

    # dailys return in percentage
    dr_bm = (bm / bm.shift(1) - 1).iloc[1:]
    dr_msp = (msp / msp.shift(1) - 1).iloc[1:]

    # std of daily returns
    sddr_bm = dr_bm.std()
    sddr_msp = dr_msp.std()

    # mean of daily returns
    adr_bm = dr_bm.mean()
    adr_msp = dr_msp.mean()

    print("Manual Strategy")
    print("Cum return: " + str(cr_msp))
    print("Stdev of daily returns: " + str(sddr_msp))
    print("Mean of daily returns: " + str(adr_msp))
    print("\n")
    print("\n")
    print("Benchmark")
    print("Cum return: " + str(cr_bm))
    print("Stdev of daily returns: " + str(sddr_bm))
    print("Mean of daily returns: " + str(adr_bm))


def graphs(benchmark_portvals, theoretical_portvals, short, long, label):
    register_matplotlib_converters()
    # normalize
    benchmark_portvals['value'] = benchmark_portvals['value'] / benchmark_portvals['value'][0]
    theoretical_portvals['value'] = theoretical_portvals['value'] / theoretical_portvals['value'][0]

    plt.figure()
    plt.title("Manual Strategy " + label)
    plt.xticks(rotation=25)
    plt.plot(benchmark_portvals, label="benchmark", color="purple")
    plt.plot(theoretical_portvals, label="manual", color="red")

    for date in set(short + long):
        color = "black" if date in short else "blue"
        plt.axvline(date, color=color)

    plt.legend()
    plt.savefig("images/manual_{}.png".format(label))


def plot_long_short(symbol, df_trades):
    long = []
    short = []
    current = 0
    last_action = 'OUT'
    for date in df_trades.index:
        current += df_trades.loc[date, symbol]
        if current < 0:
            if last_action == 'OUT' or last_action == 'LONG':
                last_action = 'SHORT'
                short.append(date)
        elif current > 0:
            if last_action == 'OUT' or last_action == 'SHORT':
                last_action = 'LONG'
                long.append(date)
        else:
            last_action = 'OUT'

    return short, long


def report_ms():
    # in-sample data
    sv_is = 100000
    sd_is = dt.datetime(2008, 1, 1)
    ed_is = dt.datetime(2009, 12, 31)
    symbol_is = ['JPM']
    # get theoretical in-sample portfolio
    df_trades_is = testPolicy(symbol_is, sd=sd_is, ed=ed_is, sv=sv_is)
    manual_portvals_is = compute_portvals(df_trades_is, sv_is, commission=9.95, impact=0.005)
    # get benchmark performance; in-sample
    benchmark_portvals_is = benchmark(sd_is, ed_is, sv_is)

    # plot in-sample data period
    short, long = plot_long_short(symbol_is[0], df_trades_is)
    graphs(benchmark_portvals_is, manual_portvals_is, short, long, 'in_sample')

    # out-of-sample data
    sv_oos = 100000
    sd_oos = dt.datetime(2010, 1, 1)
    ed_oos = dt.datetime(2011, 12, 31)
    symbol_oos = ['JPM']
    # get theoretical out-of-sample portfolio
    df_trades_oos = testPolicy(symbol_oos, sd=sd_oos, ed=ed_oos, sv=sv_oos)
    manual_portvals_oos = compute_portvals(df_trades_oos, sv_oos, commission=9.95, impact=0.005)
    # get benchmark performance; out-of-sample
    benchmark_portvals_oos = benchmark(sd_oos, ed_oos, sv_oos)

    # plot out-of-sample period
    short, long = plot_long_short(symbol_oos[0], df_trades_oos)
    graphs(benchmark_portvals_oos, manual_portvals_oos, short, long, 'out_sample')

    # print Cumulative return, STDEV of daily returns, and Mean of daily returns
    # of the benchmark and Manual Strategy portfolio
    stats(benchmark_portvals_is, manual_portvals_is)


if __name__ == "__main__":
   report_ms()