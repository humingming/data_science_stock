import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
import csv

sys.path.insert(0, '../mc3_p1')
import LinRegLearner as lrl
import KNNLearner as knn
import BagLearner as bl

sys.path.insert(0, '../mc2_p1')
sys.path.insert(0, '../data')
import util

ALLOWED_LOT = 100 # Only allow to buy/sell 100 shares.
TIME_FRAME = 5 # 5 days


###########################
### Data analysis tools ###
###########################

def normalise_data(df):
    """ Normalise stock prices using the first row of dataframe. """
    return df / df.ix[0, :]

def compute_rolling_mean(df):
    return df.rolling(window=TIME_FRAME).mean().dropna()

def compute_rolling_std(df):
    return df.rolling(window=TIME_FRAME).std().dropna()

def compute_bollinger(df, rm, rstd):
    """ Compute normalised bollinger value. """
    bb_feature = (df - rm) / (2 * rstd)
    return bb_feature.dropna()

def compute_momentum(df):
    """ Compute price momentum. 
        momentum[t] = (price[t] / price[t-N]) - 1
    """ 
    past_df = df.shift(TIME_FRAME)
    df = (df / past_df) - 1
    return df.dropna()

def compute_future_return(df):
    """ Future return is computed like so:
        price[t] = (price[t+N] / price[t]) - 1
    """
    gains_df = df.shift(-TIME_FRAME)
    return ((gains_df / df) - 1).dropna()

def analyse_data(symbols, dates):
    """ Read and analyse data 
    Computing:
    1) For X-values
        - Bollinger Band values
        - Momentum
        - Volatility (std)
    2) For Y-value
        - Future 5-day return
    Pack and return as nd-arrays to be used by learner.

    Note: Remove the first and last N rows of the results since rolling 
    values start after N days and future returns end N days earlier.
    """
    df = util.get_data(symbols, dates, addSPY=False, addCash=False).dropna()
    df = normalise_data(df)

    momentum = compute_momentum(df)
    rm = compute_rolling_mean(df) # used to compute bollinger value
    rstd = compute_rolling_std(df)
    # Remove first N days, since no rolling values are available.
    b_val = compute_bollinger(df[TIME_FRAME:], rm, rstd)
    fr = compute_future_return(df)[TIME_FRAME:]

    # Merge in a single df and remove last N rows - no future returns availabe. 
    result_x = pd.concat([momentum, rstd, b_val], axis=1, join='inner')[:-TIME_FRAME]
    data_x = result_x.as_matrix()
    data_y = fr.as_matrix()

    return data_x, data_y

#########################
### Learner functions ###
#########################

def train_learner(learner, data_x, data_y, query_x):
    """ Train a learner by feeding analysed data.
        Then, make an estimate for data from query_x and return it.
    """
    learner.add_evidence(data_x, data_y)
    estimate = learner.query(query_x)

    return estimate

def test_learner(dates_learn, dates_test, symbols):
    """ Test a learner.
        1) Train it with analysed data in range dates_learn
        2) Analyse the data in range dates_test.
        3) Return the estimated Y and the actual Y.
    """
    data_x, data_y = analyse_data(symbols, dates_learn)
    query_x, actual_y = analyse_data(symbols, dates_test)

    learner = bl.BagLearner(learner = knn.KNNLearner, kwargs={"k":3}, bags=12) 
    estimate_y = train_learner(learner, data_x, data_y, query_x)

    return estimate_y.flatten(), actual_y.flatten()

################
### Strategy ###
################

def trade_strategy(exp_change, df_prices):
    """ Generate trade orders based on expected price change. 
        An order consists of type (BUY/SELL) and a date.
        An order is triggered on day i and is then stopped on day i+N (N days later).
    """
    buy_threshold = 0.01
    sell_threshold = -0.01
    # Remove first N days, no expected change for those.
    df_prices = df_prices[TIME_FRAME:]
    dates = df_prices.index.map(lambda x: x.strftime('%Y-%m-%d'))

    orders = []

    for i in range(len(exp_change)):
        if exp_change[i] > buy_threshold:
            orders.append(['BUY', dates[i]])
            orders.append(['SELL', dates[i + TIME_FRAME]])
        
        elif exp_change[i] < sell_threshold:
            orders.append(['SELL', dates[i]])
            orders.append(['BUY', dates[i + TIME_FRAME]])

    return orders

def orders_to_csv(symbol, orders):
    """ Write orders to csv file. 
        Used for testing with the market simulator from mc2_p1. 

        Note: orders are not chronologically sorted. This is handled by the
        market simulator.
    """
    with open('orders/orders_{0}.csv'.format(symbol), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Date', 'Symbol', 'Order', 'Shares'])
        for order in orders:
            writer.writerow([order[1], symbol, order[0], ALLOWED_LOT])
        

#############
### Plots ###
#############

def compare_plot(est, actual): 
    """ Compare estimated and actual change in price. """
    # est and actual are ndarrays, convert them to dataframes. 
    df_est = pd.DataFrame(est) 
    df_actual = pd.DataFrame(actual)
    df_temp = pd.concat([df_est, df_actual], keys=['Estimated Change', 'Actual Change'], axis=1)
    axis = util.plot_data(df_temp)
    plt.show()


############
### Main ###
############

if __name__ == "__main__":
    # Get date ranges and symbols.
    dates_learn = pd.date_range('2002-12-31', '2009-12-31')
    print("this is dates_learn")
    print(dates_learn)
    dates_test = pd.date_range('2009-12-31', '2011-01-31')
    symbols = ['ML4T-220']

    # Get estimated and actual price changes.
    est, act = test_learner(dates_learn, dates_test, symbols)
    print("est is")
    print(est)
    print("act is")
    print(act)



    df_prices = util.get_data(symbols, dates_test, False, False).dropna() 
    trade_strategy(est, df_prices)

    # Print statistics and plot the chart.
    rmse = math.sqrt(((act - est) ** 2).sum()/act.shape[0])
    print('\nKNN in sample results') 
    print('RMSE: ', rmse)
    c = np.corrcoef(est, y=act)
    print('corr:', c[0,1])
    compare_plot(est, act)



    # # Generate orders and write to csv.
    # orders = trade_strategy(est, df_prices)
    # orders_to_csv(symbols[0], orders)

