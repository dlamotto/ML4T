import pandas as pd
import datetime as dt
from util import get_data

def author():
    return 'dlamotto3'

def compute_sma(sd, ed, symbol, window_size=20):
    # Adjust start date for lookback window
    extended_sd = sd - dt.timedelta(window_size * 2)

    df_price = get_data(symbol, pd.date_range(extended_sd, ed))
    df_price = df_price[symbol].ffill().bfill()

    sma = df_price.rolling(window=window_size, min_periods=window_size).mean()
    sma = sma.truncate(before=sd)

    return sma

def compute_bbp(sd, ed, symbol, window_size=20):
    extended_sd = sd - dt.timedelta(window_size * 2)

    df_price = get_data(symbol, pd.date_range(extended_sd, ed))
    df_price = df_price[symbol].ffill().bfill()

    rolling_mean = df_price.rolling(window=window_size, min_periods=window_size).mean()
    rolling_std = df_price.rolling(window=window_size, min_periods=window_size).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

    bbp = (df_price - lower_band) / (upper_band - lower_band)
    bbp = bbp.truncate(before=sd)

    return bbp

def compute_momentum(sd, ed, symbol, window_size=20):
    extended_sd = sd - dt.timedelta(window_size * 2)

    df_price = get_data(symbol, pd.date_range(extended_sd, ed))
    df_price = df_price[symbol].ffill().bfill()

    momentum = df_price / df_price.shift(window_size) - 1
    momentum = momentum.truncate(before=sd)

    return momentum

# Example usage
if __name__ == "__main__":
    sd = dt.datetime(2020, 1, 1)
    ed = dt.datetime(2021, 1, 1)
    symbol = "JPM"

    sma = compute_sma(sd, ed, symbol)
    bbp = compute_bbp(sd, ed, symbol)
    momentum = compute_momentum(sd, ed, symbol)