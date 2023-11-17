"""
Student Name: Danielle LaMotto
GT User ID: dlamotto3
GT ID: 903951588
"""
import datetime as dt
from util import get_data
import pandas as pd
import matplotlib.pyplot as plt


class Indicators:

    def __init__(self, prices, date_range):
        self.prices = prices
        self.normal_prices = self.prices / self.prices.iloc[0]
        self.date_range = date_range

    def plot_macd(self):
        plt.plot(self.macd, label="MACD")
        plt.plot(self.signal, label="Signal Line")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.legend(loc='upper left')
        plt.xticks(rotation=30)
        plt.legend()
        plt.savefig("images/MACD.png")
        plt.close()

    def plot_cci(self):
        plt.plot(self.rolling_mean, label="Rolling Mean")
        plt.plot(self.cci, label="CCI")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.legend(loc='upper left')
        plt.xticks(rotation=30)
        plt.legend()
        plt.savefig("images/CCI.png")
        plt.close()

    def plot_bb(self):
        plt.plot(self.normal_prices, label="Normalized Price")
        plt.plot(self.upper_bb, label="Upper band")
        plt.plot(self.lower_bb, label="Lower band")
        plt.xlabel("Date")
        plt.ylabel("Normalized Prices")
        plt.legend(loc='best')
        plt.xticks(rotation=30)
        plt.title("Indicator Bollinger Bands")
        plt.savefig("images/BollingerBands.png")
        plt.close()

        plt.plot(self.b_index, label="BB Value")
        plt.xlabel("Date")
        plt.ylabel("Normalized Prices")
        plt.legend(loc='best')
        plt.xticks(rotation=30)
        plt.title("Indicator Bollinger Value")
        plt.savefig("images/BB_Value.png")
        plt.close()

    def plot_momentum(self):
        plt.plot(self.normal_prices, label="Normalized Price")
        plt.plot(self.momentum, label="20-day Momentum")
        plt.title("Price and 20-day Momentum")
        plt.xlabel("Date")
        plt.ylabel("Normalized Prices")
        plt.legend(loc='best')
        plt.xticks(rotation=30)
        plt.savefig("images/momentum.png")
        plt.close()

    def plot_sma(self):
        plt.plot(self.normal_prices, label="Normalized Price")
        plt.plot(self.sma, label='20-day SMA')
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.title("Price and 20-day SMA")
        plt.legend(loc='best')
        plt.xticks(rotation=30)
        plt.savefig("images/sma.png")
        plt.close()


    def compute_sma(self, plot=True):
        self.sma = self.normal_prices.rolling(window=20, center=False).mean()
        if plot:
            self.plot_sma()
        return self.sma

    def compute_bb(self, plot=True):
        self.rolling_std = self.normal_prices.rolling(window=20, center=False).std()
        self.rolling_mean = self.normal_prices.rolling(window=20, center=False).mean()
        self.upper_bb = self.rolling_mean + (2 * self.rolling_std)
        self.lower_bb = self.rolling_mean - (2 * self.rolling_std)
        self.b_index = (self.normal_prices - self.rolling_mean)/(2*self.rolling_std)
        if plot:
            self.plot_bb()
        return self.lower_bb, self.upper_bb, self.b_index

    def compute_cci(self, plot=True):
        std = self.normal_prices.std()
        self.rolling_mean = self.normal_prices.rolling(window=20, center=False).mean()
        temp = self.normal_prices - self.rolling_mean
        self.cci = temp/(0.015 * std)
        if plot:
            self.plot_cci()
        return self.cci, self.rolling_mean

    def compute_momentum(self, plot=True):

        self.momentum = self.normal_prices - self.normal_prices.shift(20)
        if plot:
            self.plot_momentum()
        return self.momentum

    def compute_macd(self, plot=True):

        self.short_ema = self.normal_prices.ewm(ignore_na=False, span=12, min_periods=0, adjust=True).mean
        self.long_ema = self.normal_prices.ewm(ignore_na=False, span=26, min_periods=0, adjust=True).mean

        self.macd = self.short_ema()-self.long_ema()
        self.signal = self.macd.ewm(ignore_na=False, span=9, min_periods=0, adjust=True).mean()
        if plot:
            self.plot_macd()
        return self.macd, self.signal

    def author(self):
        return 'dlamotto3'

def main():
    symbolList = ["JPM"]
    startDate = dt.datetime(2008, 1, 1)
    endDate = dt.datetime(2009, 12, 31)
    date_range = pd.date_range(startDate, endDate)
    prices = get_data(symbolList, date_range)[symbolList]
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    indicators = Indicators(prices, date_range)
    indicators.compute_bb()
    indicators.compute_macd()
    indicators.compute_sma()
    indicators.compute_cci()
    indicators.compute_momentum()

if __name__ == "__main__":
    main()
