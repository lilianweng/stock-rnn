"""
Fetch the daily stock prices from Google Finance for stocks in S & P 500.
@author: lilianweng
"""
import click
import os
import pandas as pd
import pandas_datareader.data as web
import random
import time
import urllib.request, urllib.error, urllib.parse

from bs4 import BeautifulSoup
from datetime import datetime

DATA_DIR = "data"
RANDOM_SLEEP_TIMES = (1, 5)

# This repo "github.com/datasets/s-and-p-500-companies" has some other information about
# S & P 500 companies.
SP500_LIST_URL = "https://raw.githubusercontent.com/datasets/s-and-p-companies-financials/master/data/constituents-financials.csv"
SP500_LIST_PATH = os.path.join(DATA_DIR, "constituents-financials.csv")


def _download_sp500_list():
    if os.path.exists(SP500_LIST_PATH):
        return

    print("Downloading ...", SP500_LIST_URL)
    f = urllib.request.urlretrieve(SP500_LIST_URL, SP500_LIST_PATH)
    return


def _load_symbols():
    _download_sp500_list()
    df_sp500 = pd.read_csv(SP500_LIST_PATH)
    df_sp500.sort_values(by='Market Cap', ascending=False, inplace=True)
    stock_symbols = df_sp500['Symbol'].unique().tolist()
    print("Loaded %d stock symbols" % len(stock_symbols))
    return stock_symbols


def fetch_prices(symbol, out_name):
    """
    Fetch daily stock prices for stock `symbol`, since 1980-01-01.

    Args:
        symbol (str): a stock abbr. symbol, like "GOOG" or "AAPL".

    Returns: a bool, whether the fetch is succeeded.
    """
    # Format today's date to match Google's finance history api.
    start = datetime(1980, 1, 1)
    end = datetime.now().strftime("%Y-%m-%d")
    try:
        print("Fetching {} ...".format(symbol))
        web.DataReader(symbol, 'quandl', start, end).to_csv(out_name)
    except:
        print("Failed when fetching {}".format(symbol))
        return False

    data = pd.read_csv(out_name)
    if data.empty:
        print("Remove {} because the data set is empty.".format(out_name))
        os.remove(out_name)
    else:
        dates = data.iloc[:, 0].tolist()
        print("# Fetched rows: %d [%s to %s]" % (data.shape[0], dates[-1], dates[0]))

    # Take a rest
    # sleep_time = random.randint(*RANDOM_SLEEP_TIMES)
    # print("Sleeping ... %ds" % sleep_time)
    # time.sleep(sleep_time)
    return True


@click.command(help="Fetch stock prices data")
@click.option('--continued', is_flag=True)
def main(continued):
    random.seed(time.time())
    num_failure = 0

    # This is S&P 500 index
    # fetch_prices('INDEXSP%3A.INX')

    symbols = _load_symbols()
    for idx, sym in enumerate(symbols):
        out_name = os.path.join(DATA_DIR, sym + ".csv")
        if continued and os.path.exists(out_name):
            print("Fetched", sym)
            continue

        succeeded = fetch_prices(sym, out_name)
        num_failure += int(not succeeded)

        if idx % 10 == 0:
            print("# Failures so far [%d/%d]: %d" % (idx + 1, len(symbols), num_failure))


if __name__ == "__main__":
    main()
