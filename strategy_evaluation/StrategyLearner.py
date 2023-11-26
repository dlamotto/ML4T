""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		  
import random  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
from util import get_data
import numpy as np
import RTLearner as rt
import BagLearner as bl
from indicators import *
  		  	   		  		 		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    # constructor  		  	   		  		 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=9.95, commission=0.005):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		  		 		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		  		 		  		  		    	 		 		   		 		  
        self.commission = commission
        self.learner = bl.BagLearner(rt.RTLearner, kwargs={"leaf_size": 5}, bags=20, boost=False, verbose=False)
  		  	   		  		 		  		  		    	 		 		   		 		  

    def add_evidence(self, symbol="IBM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000):
        # define the symbol and date range
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        # get stock data for the given symbol and date range
        df = get_data(syms, dates)
        prices = df[syms]

        # make a dataframe to store trades
        df_trades = df[['SPY']]
        df_trades = df_trades.rename(columns={'SPY': symbol}).astype({symbol: 'int32'})
        df_trades[:] = 0

        # get indicator data; sma, bbp and momentum
        sma = compute_sma(sd, ed, [symbol], window_size=20)
        bbp = compute_bbp(sd, ed, [symbol], window_size=20)
        momentum = compute_momentum(sd, ed, [symbol], window_size=20)

        # rename columns and concatenate indicators into a single dataframe
        sma_df = sma.rename(columns={symbol: 'SMA'})
        bbp_df = bbp.rename(columns={symbol: 'BBP'})
        mom_df = momentum.rename(columns={symbol: 'MOM'})
        indicators = pd.concat((sma_df, bbp_df, mom_df), axis=1)
        indicators.fillna(0, inplace=True)
        indicators = indicators[:-5]  # Removing the last 5 rows
        x_train = indicators.values

        # Construct trainY based on the future price movement
        y_train = []
        for i in range(prices.shape[0] - 5):
            # calculate the price ratio for a 5-day future window
            ratio = (prices.ix[i + 5, symbol] - prices.ix[i, symbol]) / prices.ix[i, symbol]
            # determine the signal based on the price movement and impact
            if ratio > (0.02 + self.impact):
                y_train.append(1)  # Positive signal
            elif ratio < (-0.02 - self.impact):
                y_train.append(-1)  # Negative signal
            else:
                y_train.append(0)  # Neutral signal
        y_train = np.array(y_train)
        self.learner.add_evidence(x_train, y_train)


    def testPolicy(self, symbol="IBM", \
                   sd=dt.datetime(2009, 1, 1), \
                   ed=dt.datetime(2010, 1, 1), \
                   sv=10000):
        # define the symbol and date range
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        # get stock data for the given symbol and date range
        prices_all = get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        # get indicator data; sma, bbp and momentum
        lookback = 20
        sma = compute_sma(sd, ed, [symbol], lookback)
        bbp = compute_bbp(sd, ed, [symbol], lookback)
        momentum = compute_momentum(sd, ed, [symbol], lookback)

        # construct the indicators for the learning algorithm
        sma_df = sma.rename(columns={symbol: 'SMA'})
        bbp_df = bbp.rename(columns={symbol: 'BBP'})
        mom_df = momentum.rename(columns={symbol: 'MOM'})
        indicators = pd.concat((sma_df, bbp_df, mom_df), axis=1)
        indicators.fillna(0, inplace=True)
        x_test = indicators.values

        # query the learner for y_test
        y_test = self.learner.query(x_test)

        # make trades dataframe
        trades = prices_all[syms].copy()
        trades.loc[:] = 0  # initialize all trades to zero
        flag = 0  # A flag to track the current position state

        # go through prices and determine trades based on learner's predictions
        for i in range(0, prices.shape[0] - 1):
            if flag == 0:  # Currently not in the market
                if y_test[i] > 0:  # buy
                    trades.values[i, :] = 1000
                    flag = 1
                elif y_test[i] < 0:  # sell
                    trades.values[i, :] = -1000
                    flag = -1

            elif flag == 1:  # Currently holding a long position
                if y_test[i] < 0:  # sell
                    trades.values[i, :] = -2000  # go short
                    flag = -1
                elif y_test[i] == 0:  # no change
                    trades.values[i, :] = -1000  # sell and move to cash
                    flag = 0

            else:  # Currently holding a short position
                if y_test[i] > 0:  # buy
                    trades.values[i, :] = 2000  # buy to cover short and go long
                    flag = 1
                elif y_test[i] == 0:  # no change
                    trades.values[i, :] = 1000  # buy to cover short and move to cash
                    flag = 0

        # Adjust for the last day
        if flag == -1:  # short position
            trades.iloc[-1] = 1000  # buy to cover
        elif flag == 1:  # long position
            trades.iloc[-1] = -1000  # sell what we have

        return trades


def author():
    return 'dlamotto3'


def report_sl():
    learner = StrategyLearner(verbose=False, impact=0.0, commission=0.0)  # constructor
    learner.add_evidence(symbol="AAPL", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                         sv=100000)  # training phase
    df_trades = learner.testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                   sv=100000)  # testing phase


if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")
    report_sl()
