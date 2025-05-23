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
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import datetime as dt  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import numpy as np  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import matplotlib.pyplot as plt  		  	   		 	 	 			  		 			 	 	 		 		 	
import pandas as pd  		  	   		 	 	 			  		 			 	 	 		 		 	
from util import get_data, plot_data  	

from scipy.optimize import minimize
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
# This is the function that will be tested by the autograder  		  	   		 	 	 			  		 			 	 	 		 		 	
# The student must update this code to properly implement the functionality  		  	   		 	 	 			  		 			 	 	 		 		 	
def optimize_portfolio(  		  	   		 	 	 			  		 			 	 	 		 		 	
    sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	 	 			  		 			 	 	 		 		 	
    gen_plot=False,  		  	   		 	 	 			  		 			 	 	 		 		 	
):  		  	   		 	 	 			  		 			 	 	 		 		 	
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
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # getting the dates data
    dates = pd.date_range(sd, ed)  	

    # gets all the prices for the given tickers AND SPY 	  	   		 	 	 			  		 			 	 	 		 		 	
    prices_all = get_data(syms, dates) 	

    # splits into the ticker symbols and SPY	  	   		 	 	 			  		 			 	 	 		 		 	
    prices = prices_all[syms]  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices_SPY = prices_all["SPY"]  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 			  	   		 	 	 			  		 			 	 	 		 		 		   		 	 	 			  		 			 	 	 		 		 	
    # get ready for the minimize function
    number_of_assets = len(syms)
    initial_guess = [1.0 / number_of_assets] * number_of_assets
    bounds = [(0.0, 1.0)] * number_of_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    def negative_sharpe_ratio(allocs):

        # normalize stock prices so they start at 1 on the first day
        normalized_prices = prices / prices.iloc[0]

        # given our current percentage allocation, what are the prices we have?
        allocated_prices = normalized_prices * allocs

        # what is the value of the portfolio then on each day?
        portfolio_value = allocated_prices.sum(axis=1)

        # the percentage returns of the portfolio on a day to day basis.
        daily_returns = portfolio_value.pct_change().dropna()

        # what is the average of that percentage return?
        average_daily_return = daily_returns.mean()

        # what is the standard deviation of that percentage return?
        standard_deviation_of_daily_returns = daily_returns.std()

        # math magic
        sharpe_ratio = (average_daily_return / standard_deviation_of_daily_returns) * np.sqrt(252)
        return -sharpe_ratio
    
    result = minimize(negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_allocations = result.x
  		  	   		 	 	 			  		 			 	 	 		 		 	
    if gen_plot:  		  	   		 	 	 	

        # the first item on the graph will be the date/normalized value of the optimally allocated portfolio
        normalized_prices = prices / prices.iloc[0]
        portfolio_value = (normalized_prices * optimal_allocations).sum(axis=1)
        normalized_portfolio_value = portfolio_value / portfolio_value.iloc[0]

        # the second item will be SPY to see if we outperformed SPY
        normalized_spy = prices_SPY / prices_SPY.iloc[0]

        # then we just throw those mfs up there
        df_plot = pd.concat([normalized_portfolio_value, normalized_spy], axis=1)
        df_plot.columns = ['Portfolio', 'SPY']

        df_plot.plot(title="DAily Portfolio Value vs. SPY")
        plt.xlabel("Date")
        plt.ylabel("Normalized Value")
        plt.savefig("Figure2.png")
        plt.close  	   

    # and here we wrap up some status to judge the function
    portfolio_value = (prices / prices.iloc[0] * optimal_allocations).sum(axis=1)
    cumulative_return = (portfolio_value[-1] / portfolio_value[0]) - 1
    daily_returns = portfolio_value.pct_change().dropna()
    average_daily_return = daily_returns.mean()
    standard_deviation_of_daily_returns = daily_returns.std()	
    sharpe_ratio = (average_daily_return / standard_deviation_of_daily_returns) * np.sqrt(252)	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    return optimal_allocations, cumulative_return, average_daily_return, standard_deviation_of_daily_returns, sharpe_ratio  	  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def test_code():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    start_date = dt.datetime(2009, 1, 1)  		  	   		 	 	 			  		 			 	 	 		 		 	
    end_date = dt.datetime(2010, 1, 1)  		  	   		 	 	 			  		 			 	 	 		 		 	
    symbols = ["GOOG", "AAPL", "GLD", "XOM"]  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
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
