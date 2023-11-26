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


def author():
    return 'dlamotto3'


def compute_portvals(orders_df, start_val=1000000, commission=9.95, impact=0.005):
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
    sd = orders_df.index[0]
    ed = orders_df.index[-1]
    portvals = get_data(['SPY'], pd.date_range(sd, ed), addSPY=True, colname='Adj Close')
    portvals = portvals.rename(columns={'SPY': 'value'})
    dates = portvals.index
    symbol = orders_df.columns[0]

    ##### my account
    balance = start_val
    portfolio = {}
    symbols = {}

    for date in dates:
        trade = orders_df.loc[date, symbol]

        if trade != 0:
            order = 'SELL' if trade < 0 else 'BUY'
            shares = abs(trade)

            balance, portfolio, symbols = calculate(
                symbol, order, shares, balance, portfolio,
                symbols, date, ed, commission, impact
            )

        ### calculating current protfolio value
        portvals.at[date, 'value'] = compute_portval(date, balance, portfolio, symbols)

    return portvals


# update current_cash and shares_owned from an order
def calculate(symbol, order, shares, balance, portfolio, symbols, current_date, end_date, commission,
                      impact):
    # get symbol data into symbol_table if not already there
    symbols.setdefault(symbol, get_data([symbol], pd.date_range(current_date, end_date), addSPY=True,
                                             colname='Adj Close').ffill().bfill())
    portfolio_delta = 0
    balance_delta = 0
    # update the share and cash information
    if order == 'BUY':
        portfolio_delta = shares
        balance_delta = -symbols[symbol].loc[current_date].loc[symbol] * (1 + impact) * shares
    elif order == 'SELL':
        portfolio_delta = -shares
        balance_delta = symbols[symbol].loc[current_date].loc[symbol] * (1 - impact) * shares

    portfolio[symbol] = portfolio.get(symbol, 0) + portfolio_delta
    balance += balance_delta - commission

    return balance, portfolio, symbols


# compute the portfolio value for a day
def compute_portval(curr_date, current_cash, shares_owned, symbol_table):
    shares_worth = 0
    for symbol in shares_owned:
        shares_worth += symbol_table[symbol].loc[curr_date].loc[symbol] * shares_owned[symbol]
    return current_cash + shares_worth

