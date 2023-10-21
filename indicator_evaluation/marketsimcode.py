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

    # start date = first day in orders df, end date = last day in orders df
    sd = dataframe.index[0]
    ed = dataframe.index[-1]
    # a date range for all days in that range
    all_dates = pd.date_range(start=sd, end=ed)

    # get the prices for our date range and symbols and clean the data, the dates are the index
    portvals = get_data(['SPY'], all_dates, addSPY=True, colname = 'Adj Close')
    portvals = portvals.rename(columns={'SPY': 'value'})

    # gets the list of symbols that are in the order sheet
    symbol = dataframe.columns[0]

    balance = start_val
    portfolio = {}  # symbol (str) -> number (int)
    symbols = {}  # symbol (str) -> prices (pd.df)


    # Iterate through the portvals dataframe
    for index in portvals.index:
        trade = dataframe.loc[index].loc[symbol]
        if trade != 0:
            if trade < 0:
                order = 'SELL'
                shares = abs(trade)
            else:
                order = 'BUY'
                shares = trade

            symbols = get_sym_data(symbol, symbols, index, ed)
            balance, portfolio = calculate(symbol, order, shares, balance, portfolio, symbols, index, commission, impact)
        val = 0
        for symbol in portfolio:
            val += symbols[symbol].loc[index].loc[symbol] * portfolio[symbol]
        portvals.loc[index].loc['value'] = balance + val

    return portvals


def get_sym_data(symbol, symbols, index, ed):
    if symbol not in symbols:
        symbols_df  = get_data([symbol], pd.date_range(index, ed), addSPY=True, colname='Adj Close')
        symbols_df.ffill().bfill()
        symbols[symbol] = symbols_df
    return symbols


def calculate(symbol, order, shares, balance, portfolio, symbols, index, impact, commission):
    if order == 'BUY':
        portfolio_delta = shares
        balance_delta = -symbols[symbol].loc[index].loc[symbol] * (1 + impact) * shares
    elif order == 'SELL':
        portfolio_delta = -shares
        balance_delta = symbols[symbol].loc[index].loc[symbol] * (1 - impact) * shares

    portfolio[symbol] = portfolio.get(symbol, 0) + portfolio_delta
    balance += balance_delta - commission

    return balance, portfolio


def author():
    return 'dlamotto3'
