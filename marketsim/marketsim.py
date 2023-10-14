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
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: dlamotto3 (replace with your User ID)  		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 903951588 (replace with your GT ID)  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
from util import get_data, plot_data


def author():
    return 'dlamotto3'


def compute_portvals(  		  	   		  		 		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		  		 		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		  		 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		  		 		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		  		 		  		  		    	 		 		   		 		  
):
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		  		 		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		  		 		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  		 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    # convert the orders to a dataframe
    # Might need to do an if else statement for a string type???
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])

    orders_df.sort_index(inplace=True)
    # start date = first day in orders df, end date = last day in orders df
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]
    # a date range for all days in that range
    all_dates = pd.date_range(start=start_date, end=end_date)
    # gets the list of symbols that are in the order sheet
    symbols = orders_df['Symbol'].unique().tolist()
    # initialize a dictionary of shares with each symbol starting with 0 shares
    shares = {symbol: 0 for symbol in symbols}
    # initialize our balance with our starting value (start_val)
    balance = start_val
    # get the prices for our date range and symbols and clean the data, the dates are the index
    adjusted_closing_prices = get_data(symbols, all_dates)
    adjusted_closing_prices.ffill(inplace=True)  # fill missing data forward
    # adjusted_closing_prices.bfill(inplace=True)  # fill missing data backward

    # initialize the empty dataframe to be returned
    portvals = pd.DataFrame(index=adjusted_closing_prices.index, columns=['Balance'])

    # iterates through the orders dataframe
    i = 0
    for index in portvals.index:
        # Check if the current date matches any in orders_df and that we haven't processed all orders
        while i < len(orders_df) and index == orders_df.index[i]:
            symbol = orders_df['Symbol'].iloc[i]
            order_type = orders_df['Order'].iloc[i]
            num_shares = orders_df['Shares'].iloc[i]
            current_price = adjusted_closing_prices.loc[index, symbol]

            if order_type == 'BUY':
                shares[symbol] += num_shares  # add the number of shares we are buying to our portfolio
                balance -= (1 + impact) * num_shares * current_price  # account for the market impact
                balance -= commission  # cash amount minus commission
            else:
                shares[symbol] -= num_shares  # subtract the number of shares we are selling from our portfolio
                balance -= (-1 + impact) * num_shares * current_price  # account for the market impact
                balance -= commission  # cash amount minus commission

            # move to the next row in orders
            i += 1

        # update the portfolio value based on our trades
        portvals.loc[index] = balance
        # update the portfolio value based on the prices in the market for each day there is no trade
        total_value = sum(shares[key] * adjusted_closing_prices[key].loc[index] for key in shares)
        portvals.loc[index] += total_value

    #print(portvals)
    return portvals

  		  	   		  		 		  		  		    	 		 		   		 		  
def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		  		 		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    of = "./orders/orders-09.csv"
    sv = 1000000
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Process orders  		  	   		  		 		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv)
    print(portvals)
    # if isinstance(portvals, pd.DataFrame):
    #     portvals = portvals[portvals.columns[0]]  # just get the first column
    # else:
    #     "warning, code did not return a DataFrame"
    #
    # # Get portfolio stats
    # # Here we just fake the data. you should use your code from previous assignments.
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
    #     (allocated_daily_returns[-1] / allocated_daily_returns[0])-1,
    #     allocated_daily_returns.mean(),
    #     allocated_daily_returns.std(),
    #     math.sqrt(252) * ((allocated_daily_returns.mean())/allocated_daily_returns.std()), # a risk-free rate of 0
    # ]
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
    #     0.2,
    #     0.01,
    #     0.02,
    #     1.5,
    # ]
  	#
    # # Compare portfolio against $SPX
    # print(f"Date Range: {start_date} to {end_date}")
    # print()
    # print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    # print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    # print()
    # print(f"Cumulative Return of Fund: {cum_ret}")
    # print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    # print()
    # print(f"Standard Deviation of Fund: {std_daily_ret}")
    # print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    # print()
    # print(f"Average Daily Return of Fund: {avg_daily_ret}")
    # print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    # print()
    # print(f"Final Portfolio Value: {portvals[-1]}")
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
