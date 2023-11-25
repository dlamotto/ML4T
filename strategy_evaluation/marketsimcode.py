""""""
"""MC2-P1: Market simulator.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
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
  		  	   		  		 		  		  		    
Student Name: Danielle LaMotto
GT User ID: dlamotto3
GT ID: 903951588
"""

import datetime as dt
import pandas as pd
from util import get_data


def compute_portvals(dataframe=None,start_val=1000000,commission=9.95,impact=0.005):
    """
    Computes the portfolio values.

    :param dataframe: Thdataframe
    :type pd.dataframe
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """

    # Get the start and end dates for all the orders
    start_date = dataframe.index.min()
    end_date = dataframe.index.max()
    symbol = dataframe.columns[0]

    # Get prices dataframe with [date, symbols, cash]
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[[symbol]].fillna(method='ffill').fillna(method='bfill')
    prices['Cash'] = 1.0

    # Initialize trades DataFrame
    trades = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    # Populate trades DataFrame
    for i, row in dataframe.iterrows():
        shares = row[symbol]
        trade_val = shares * prices.loc[i, symbol]
        if shares > 0:  # buy
            trades.loc[i, symbol] += abs(shares)
            trades.loc[i, 'Cash'] -= (trade_val * (1 + impact) + commission)
        elif shares < 0:  # sell
            trades.loc[i, symbol] -= abs(shares)
            trades.loc[i, 'Cash'] += (trade_val * (1 - impact) - commission)

    # Calculate holdings
    holdings = trades.cumsum()
    holdings['Cash'] += start_val  # Add starting cash value

    # Calculate portfolio values
    portvals = (holdings * prices).sum(axis=1)

    return portvals


def author():
    return 'dlamotto3'
