"""MC2-P1: Market simulator.
    A market simulator - reads a csv file of orders and returns statistics
    about the portfolio that made the orders.
    Keeps track of the following, using dataframes:
    1) The prices for symbols traded between the dates in the order csv (get_prices_df).
    2) The trades made each day during that period (get_trades_df).
    3) The holding positions - outstanding positions during each day (get_holdings_df).
    4) The money value of each holding position for each day (get_value_df).
    5) The total value of the portfolio for each day (get_portval).
    Finally, there is assess_portval function for analysing the portfolio's performance
    given the csv of orders.
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
import sys
from math import sqrt
from util import get_data, plot_data, read_orders

sys.path.insert(0, '../mc1_p1')
from analysis import compute_daily_ret, cumulative_return, assess_portfolio, normalise_data


BUY = 'BUY' # BUY order
SELL = 'SELL' # SELL order
CASH = 'CASH' # A symbol for cash. Used in dataframes.
K_DAILY = sqrt(252) # K adjustment factor (daily)


def get_prices_df(symbols, dates):
    """ Construct the dataframe of (adjusted close) prices for each ticker
        from the start date to the end date (the whole trading period). """
    date_range = pd.date_range(dates[0], dates[-1]) # range from start to end date
    df = get_data(symbols, date_range).dropna() # get data and drop NaN
    return df

def get_trades_df(df_orders, df_prices, symbols):
    """ Construct the dataframe of trades made. 
    If no trade for a stock is made on a particular date, it's marked as 0.
    Otherwise: mark it as +(number of stocks) for BUY order.
                mark it as -(number of stocks) for SELL order.
    """
    # For dates, use the prices dataframe index to avoid adding non-trading days.
    all_dates = df_prices.index.get_values() 

    trades_df = pd.DataFrame(0, index=all_dates, columns=symbols) # Create columns for each symbol.
    df_cash = pd.DataFrame(0, index=all_dates, columns=['CASH']) # Create a column for CASH.
    trades_df = trades_df.join(df_cash) # Add the CASH column.

    for row in df_orders.itertuples():
        date, symbol, order, shares = row # extract from row
        price_stock = df_prices.loc[date, symbol] # lookup the stock price

        if order == BUY:
            trades_df.loc[date, symbol] += shares # add the order's quantity
            trades_df.loc[date, CASH] += -shares * price_stock # subtract from cash position
        else:
            trades_df.loc[date, symbol] += -shares # add negative order's quantity (SALE)
            trades_df.loc[date, CASH] += shares * price_stock # add to cash position

    return trades_df

def get_holdings_df(df_trades, start_val):
    """ Construct the dataframe of holded stocks. It indicates the 
        positions currently taken by the client. """
    holdings_df = df_trades.cumsum(axis=0) # Sum the the trades made to get the holdings.
    holdings_df.loc[:, CASH] += start_val # Add the start cash value to CASH column.

    return holdings_df

def get_value_df(df_holdings, df_prices):
    """ Dataframe that represents the value (in cash)
        for each position taken by the client.
    """
    # Multiply each holding position by its price.
    value_df = df_holdings * df_prices
    return value_df

def get_portval(df_value):
    """ Compute portfolio's total value. """
    df_portval = df_value.sum(axis=1)
    return df_portval


def compute_portvals(orders_file = "orders/orders_IBM.csv", start_val = 1000):
    """ Compute a portfolio's value over time according to orders made. """
    orders_df, symbols, dates = read_orders(orders_file) # Read the csv containing the orders
    prices_df = get_prices_df(symbols, dates) # prices for each symbol for this date range.

    trades_df = get_trades_df(orders_df, prices_df, symbols) # dataframe with trades made each day
    holdings_df = get_holdings_df(trades_df, start_val) # holding positions summary for each day
    value_df = get_value_df(holdings_df, prices_df) # money value for each position 
    portval = get_portval(value_df) # total portfolio value for each day
    
    df = portval.to_frame() # Transform to dataframe.
    return df

def assess_portval(df_portval, rfr=0.0, sf=252.0):
    daily_rets = compute_daily_ret(df_portval)
    cr = cumulative_return(df_portval)
    adr = daily_rets.mean().values[0] # average daily return
    sddr = daily_rets.std().values[0] # daily return standard deviation

    # Compute Sharpe ratio.
    rfr = ((1.0 + rfr) ** (1/sf)) - 1 # Daily risk free return.
    daily_rets = daily_rets.subtract(rfr) # Subtract the risk free return from daily returns before computing SR.
    sr = K_DAILY * (daily_rets.mean() / sddr).values[0] # Sharpe ratio.

    return cr, adr, sddr, sr    

def test_code():

    of = "orders/orders_IBM.csv"
    sv = 10000

    # Process orders
    try:
        portvals = compute_portvals(orders_file = of, start_val = sv)
    except IOError as err:
        print('Cannot open:', err)
        sys.exit(0)
    # Check if portvals returned a dataframe.    
    if not isinstance(portvals, pd.DataFrame):
        raise TypeError('Expected a DataFrame. Found {0} instead.'.format(type(portvals)))
            
    # Get dates.
    dates = portvals.index.get_values()
    start_date = pd.to_datetime(dates[0])
    end_date = pd.to_datetime(dates[-1])
        
    # Get portfolio stats.
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = assess_portval(portvals)
    end_val = portvals.ix[-1].values[0]

    # Get SPY stats.
    spy = get_data(['SPY'], pd.date_range(start_date, end_date), False, False)
    spy = normalise_data(spy)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = assess_portval(spy)


    # Compare portfolio against $SPX
    print("Date Range: {} to {}\n".format(start_date, end_date))
    print("Sharpe Ratio of Fund: {}".format(sharpe_ratio))
    print("Sharpe Ratio of SPY : {}\n".format(sharpe_ratio_SPY))
    print("Cumulative Return of Fund: {}".format(cum_ret))
    print("Cumulative Return of SPY : {}\n".format(cum_ret_SPY))
    print("Standard Deviation of Fund: {}".format(std_daily_ret))
    print("Standard Deviation of SPY : {}\n".format(std_daily_ret_SPY))
    print("Average Daily Return of Fund: {}".format(avg_daily_ret))
    print("Average Daily Return of SPY : {}\n".format(avg_daily_ret_SPY))
    print("Final Portfolio Value: {}".format(end_val))

def test_zero():
    of = "orders/orders_IBM.csv"
    sv = 1000
    portvals = compute_portvals(orders_file = of, start_val = sv)

    # Actual
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = assess_portval(portvals)
    # Expected
    exp_cum_ret = 0.0787526
    exp_avg_daily = 0.000353426354584
    exp_std_daily = 0.00711102080156
    exp_sharpe = 0.788982285751

    # assert(isclose(exp_cum_ret, cum_ret))
    # assert(isclose(exp_avg_daily, avg_daily_ret))
    # assert(isclose(exp_std_daily, std_daily_ret))
    # assert(isclose(exp_sharpe, sharpe_ratio))

def test_one():
    of = "orders/orders_IBM.csv"
    sv = 1000
    portvals = compute_portvals(orders_file = of, start_val = sv)

    # Actual
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = assess_portval(portvals)
    # Expected
    exp_cum_ret = 0.05016
    exp_avg_daily = 0.000365289198877
    exp_std_daily = 0.00560508094997
    exp_sharpe = 1.03455887842

    # assert(isclose(exp_cum_ret, cum_ret))
    # assert(isclose(exp_avg_daily, avg_daily_ret))
    # assert(isclose(exp_std_daily, std_daily_ret))
    # assert(isclose(exp_sharpe, sharpe_ratio))

if __name__ == "__main__":
    test_code()
    test_zero()
    test_one()
