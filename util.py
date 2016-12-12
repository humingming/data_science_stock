"""MLT: Utility code."""

import os
import pandas as pd
import matplotlib.pyplot as plt


def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def read_orders(orders_file):
    """ Read data from orders csv file. """
    df = pd.read_csv(orders_file, index_col='Date',
            parse_dates=True, usecols=['Date', 'Symbol', 'Order', 'Shares'], na_values=['nan'])

    df = df.sort_index(axis=0) # Sort by date.
    symbols = set(df.ix[:, 0]) # Symbols are at index 0, get them as set.
    dates = df.index # Get the dates

    return df, symbols, dates


def get_data(symbols, dates, addSPY=False, addCash=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
            
    # Add a 'CASH' column to keep track of cash positions.
    if addCash:
        df_cash = pd.DataFrame(1.0, index=dates, columns=['CASH'])
        df = df.join(df_cash)

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
