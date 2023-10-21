"""
Student Name: Danielle LaMotto
GT User ID: dlamotto3
GT ID: 903951588
"""
import datetime as dt
import TheoreticallyOptimalStrategy as tos





def author():
    return 'dlamotto3'

if __name__ == "__main__":

    df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
