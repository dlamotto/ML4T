""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""MC1-P2: Optimize a portfolio.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
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
GT User ID: dlamotto3 	   		  		 		  		  		    	 		 		   		 		  
GT ID: 903951588		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  


import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		  
import math
import numpy as np
import matplotlib.pyplot as plt  		  	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd
from scipy.optimize import minimize
from util import get_data, plot_data  		  	   		  		 		  		  		    	 		 		   		 		  


# This is the function that will be tested by the autograder  		  	   		  		 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		  		 		  		  		    	 		 		   		 		  
def optimize_portfolio(sd, ed, syms, gen_plot=False):
    """
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and
    statistics.

    :param sd: A datetime object that represents the start date, defaults to 1/1/2008
    :type sd: datetime
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009
    :type ed: datetime
    :param syms: A list of symbols that make up the portfolio (note that your code should support any
        symbol in the data directory)
    :type syms: list
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your
        code with gen_plot = False.
    :type gen_plot: bool
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,
        standard deviation of daily returns, and Sharpe ratio
    :rtype: tuple
    """

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)
    prices = prices_all[syms]  # only portfolio symbols
    prices.fillna(method="ffill", inplace=True)  # fill missing data forward
    prices.fillna(method="bfill", inplace=True)  # fill missing data backward

    #print(prices)

    # computing the daily returns for the stock prices
    daily_returns = (prices / prices.shift(1)) - 1
    daily_returns.iloc[0, :] = 0  # replace the first row's NaNs with zeroes
    daily_returns = daily_returns[1:]  # excluding the first row

    # SPY
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    # average daily return
    avg_dr = daily_returns.mean()   # expected returns

    def negative_share_ratio(weights):
        portfolio_return = avg_dr.dot(weights)
        portfolio_std = np.sqrt(weights.T.dot(daily_returns.cov()).dot(weights))
        negative_sharpe = -portfolio_return / portfolio_std
        return negative_sharpe



    # find the allocations for the optimal portfolio.
    num_stocks = len(prices.columns)
    allocs = [1./num_stocks] * num_stocks

    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for stock in range(num_stocks))

    # Optimization
    solution = minimize(negative_share_ratio, allocs, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': True})

    # Optimal weights
    optimal_weights = solution.x  # best weights for our allocations
    allocated_daily_returns = (daily_returns * optimal_weights).sum(axis=1)

    cr, adr, sddr, sr = [
        (allocated_daily_returns[-1] / allocated_daily_returns[0])-1,
        allocated_daily_returns.mean(),
        allocated_daily_returns.std(),
        math.sqrt(252) * ((allocated_daily_returns.mean())/allocated_daily_returns.std()), # a risk-free rate of 0
    ]  # add code here to compute stats

    # Get daily portfolio value
    normalized_df = prices / prices.iloc[0]  # normalize the stock prices
    allocated = normalized_df * optimal_weights   # maybe replace allocs with optimal_weights
    port_val = allocated.sum(axis=1)  # This gives the daily value of the portfolio


    normalized_prices_SPY = prices_SPY / prices_SPY.iloc[0]    # normalized SPY
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat(
            # [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1
            [port_val, normalized_prices_SPY], keys=["Portfolio", "SPY"], axis=1
        )

        df_temp.plot(figsize=(10, 6))
        plt.title("Optimized Portfolio vs. SPY")
        plt.ylabel("Prices")
        plt.xlabel("Date")
        plt.savefig('images/figure1.png')
        plt.close()

    return optimal_weights, cr, adr, sddr, sr


def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		  		 		  		  		    	 		 		   		 		  
    """

    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 1, 1)
    symbols = ["GOOG", "AAPL", "GLD", "XOM"]
    # symbols = ["WFR", "ANR", "MWW", "FSLR"]
    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )


    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
