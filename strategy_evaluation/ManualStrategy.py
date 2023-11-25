import datetime as dt
from util import get_data
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import matplotlib.pyplot as plt
from indicators import *
from marketsimcode import compute_portvals

def testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    # sym = [symbol]
    all_prices = get_data(symbol, pd.date_range(sd, ed)).fillna(method='ffill').fillna(method='bfill')
    prices = all_prices[symbol]

    df_trades = prices.copy()
    df_trades.iloc[:] = 0
    lookback = 10

    # Calculate indicators
    sma = compute_sma(prices, lookback)
    sma_price_ratio = prices / sma
    bbp = compute_bb(prices, lookback)
    momentum = compute_momentum(prices, lookback)

    holding = 0
    for i in range(lookback - 1, len(prices) - 1):
        # Simplify condition checks using direct scalar comparison
        is_oversold = bbp.iloc[i] < 0 and sma_price_ratio.iloc[i] < 0.95
        is_overbought = bbp.iloc[i] > 1 and sma_price_ratio.iloc[i] > 1.05
        is_momentum_positive = momentum.iloc[i] > 0
        is_momentum_negative = momentum.iloc[i] < 0

        # Check for buy or sell conditions
        if (is_oversold or (is_momentum_positive and sma_price_ratio.iloc[i] < 0.95) or (
                is_momentum_positive and bbp.iloc[i] < 0)) and holding <= 0:
            trade_amount = 2000 if holding == -1000 else 1000
            holding += trade_amount
            df_trades.iloc[i] = trade_amount
        elif (is_overbought or (is_momentum_negative and sma_price_ratio.iloc[i] > 1.05) or (
                is_overbought and is_momentum_negative)) and holding >= 0:
            trade_amount = -2000 if holding == 1000 else -1000
            holding += trade_amount
            df_trades.iloc[i] = trade_amount
    return df_trades

def plot_ManualStrategy(symbol, sd, ed, sv, verbose):
    register_matplotlib_converters()
    #manual strategy in-sample
    df_trades = testPolicy(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    optimal_portvals = compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)
    #statistics
    portfolio_cum_return = optimal_portvals[-1] / optimal_portvals[0] - 1
    daily_returns = optimal_portvals / optimal_portvals.shift(1) - 1
    portfolio_mean = daily_returns.mean()
    portfolio_std = daily_returns.std()


    # manual strategy out-sample
    outSample_df = testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    outsample_portvals = compute_portvals(outSample_df, start_val=100000, commission=9.95, impact=0.005)

    #statistics
    outsample_cum_return = outsample_portvals[-1] / outsample_portvals[0] - 1
    outsample_dr = outsample_portvals / outsample_portvals.shift(1) - 1
    outsample_mean = outsample_dr.mean()
    outsample_std = outsample_dr.std()

    # benchmark in-sample
    benchmark_df = df_trades.copy()
    benchmark_df.iloc[:, 0] = 0
    benchmark_df.iloc[0, 0] = 1000
    portvals_benchmark = compute_portvals(benchmark_df, start_val=100000, commission=9.95, impact=0.005)
    #statistics
    benchmark_cum_return = portvals_benchmark[-1] / portvals_benchmark[0] - 1
    benchmark_daily_return = portvals_benchmark / portvals_benchmark.shift(1) - 1
    benchmark_mean = benchmark_daily_return.mean()
    benchmark_std = benchmark_daily_return.std()

    #benchmark out-sample
    benchmark_outsample_df = outSample_df.copy()
    benchmark_outsample_df.iloc[:, 0] = 0
    benchmark_outsample_df.iloc[0, 0] = 1000
    benchmark_oustample_portvals = compute_portvals(benchmark_outsample_df, start_val=100000, commission=9.95,
                                                       impact=0.005)
    #statistics
    bechmark_outsample_cum_return = benchmark_oustample_portvals[-1] / benchmark_oustample_portvals[0] - 1
    benchmark_oustample_dr = benchmark_oustample_portvals / benchmark_oustample_portvals.shift(1) - 1
    benchmark_oustample_dr_avg = benchmark_oustample_dr.mean()
    benchmark_oustample_dr_std = benchmark_oustample_dr.std()

    #normalized prices
    normalized_optim_pvals = optimal_portvals / optimal_portvals[0]
    normalized_optim_benchmark_pvals = portvals_benchmark / portvals_benchmark[0]
    normalized_outsample_portvals = outsample_portvals / outsample_portvals[0]
    normalized_banchmark_outsample = benchmark_oustample_portvals / benchmark_oustample_portvals[0]

    if verbose:
        print('cumulative return of benchmark in-sample', round(benchmark_cum_return, 4))
        print('cumulative return of Manual strategy in-sample', round(portfolio_cum_return, 4))
        print('standard deviation of benchmark daily returns in-sample', round(benchmark_std, 4))
        print('standard deviation of Manual Strategy daily returns in-sample', round(portfolio_std, 4))
        print('mean of benchmark daily returns in-sample', round(benchmark_mean, 4))
        print('mean of manual strategy daily returns in-sample', round(portfolio_mean, 4))

        print('cumulative return of benchmark out-sample', round(bechmark_outsample_cum_return, 4))
        print('cumulative return of manual strategy out-sample', round(outsample_cum_return, 4))
        print('standard deviation of benchmark daily returns out-sample', round(benchmark_oustample_dr_std, 4))
        print('standard deviation of manual strategy daily returns out-sample', round(outsample_std, 4))
        print('mean of benchmark daily returns out-sample', round(benchmark_oustample_dr_avg, 4))
        print('mean daily returns for manual strategy out-sample', round(outsample_mean, 4))

    fig1, ax = plt.subplots()
    ax.plot(normalized_optim_benchmark_pvals, color="g", label="benchmark")
    ax.plot(normalized_optim_pvals, color="r", label="Manual strategy in-sample")
    for i in range(len(df_trades)):
        if df_trades.iloc[i, 0] < 0:  # SHORT
            ax.axvline(df_trades.index[i], color='black')
        elif df_trades.iloc[i, 0] > 0:
            ax.axvline(df_trades.index[i], color='blue')
    ax.legend()
    plt.xlabel("Date")
    plt.ylabel("Normalized portfolio value")
    plt.title("Normalized portfolio value, manual strategy vs benchmark, in-sample")
    plt.xticks(fontsize=7)
    # plt.show()
    fig1.savefig('manual1.png')

    fig2, ax = plt.subplots()
    ax.plot(normalized_banchmark_outsample, color="g", label="benchmark")
    ax.plot(normalized_outsample_portvals, color="r", label="Manual strategy out-sample")
    for i in range(len(outSample_df)):
        if outSample_df.iloc[i, 0] < 0:  # SHORT
            ax.axvline(outSample_df.index[i], color='black')
        elif outSample_df.iloc[i, 0] > 0:
            ax.axvline(outSample_df.index[i], color='blue')
    ax.legend()
    plt.xlabel("Date")
    plt.ylabel("Normalized portfolio value")
    plt.title("Normalized portfolio value, manual strategy vs benchmark, out-sample")
    plt.xticks(fontsize=7)
    fig2.savefig('manual2.png')

    fig3, ax = plt.subplots()
    ax.plot(normalized_banchmark_outsample, color="g", label="benchmark out-sample")
    ax.plot(normalized_optim_benchmark_pvals, color="g", label="benchmark in-sample")
    ax.plot(normalized_optim_pvals, color="r", label="Manual strategy in-sample")
    ax.plot(normalized_outsample_portvals, color="r", label="Manual strategy out-sample")
    ax.legend()
    plt.xlabel("Date")
    plt.ylabel("Normalized portfolio value")
    plt.title("Normalized portfolio value, manual strategy vs benchmark, out-sample")
    plt.xticks(fontsize=7)
    fig3.savefig('manual3.png')
def author():
    return 'dlamotto3'


if __name__ == "__main__":
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    symbol = ['JPM']
    dates = pd.date_range(sd, ed)
    #prices = get_data([symbol], dates)
    #prices = prices['JPM']
    # trade portfolio
    df_trades = testPolicy(symbol=symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000)
    portvals = compute_portvals(df_trades, 100000, 9.95, 0.005)
    print(df_trades)
    print(portvals)
    plot_ManualStrategy(symbol, sd, ed, 100000, True)


