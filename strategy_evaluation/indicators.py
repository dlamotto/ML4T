import pandas as pd

def compute_sma(normal_prices, lookback):
    sma = normal_prices.rolling(window=lookback, min_periods=lookback).mean()
    #sma.bfill()
    #sma_ratio = normal_prices / sma
    return sma

def compute_bb(normal_prices,lookback):
    sma = normal_prices.rolling(window=lookback, min_periods=lookback).mean()
    top_band = sma + 2 * normal_prices.rolling(window=lookback, min_periods=lookback).std()
    bottom_band = sma - 2 * normal_prices.rolling(window=lookback, min_periods=lookback).std()
    bbp = (normal_prices - bottom_band) / (top_band - bottom_band)
    return bbp

def compute_momentum(normal_prices, lookback):
    momentum = normal_prices / normal_prices.shift(lookback-1) - 1
    return momentum

