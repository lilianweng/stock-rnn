"""
Fetch the daily stock prices from Google Finance for stocks in S & P 500.
@author: lilianweng
"""
import click
import os
import pandas as pd
import random
import time
import urllib2

from BeautifulSoup import BeautifulSoup
from datetime import datetime

DATA_DIR = "data"
RANDOM_SLEEP_TIMES = (1, 5)

# This repo "github.com/datasets/s-and-p-500-companies" has some other information about
# S & P 500 companies.
SP500_LIST_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents-financials.csv"
SP500_LIST_PATH = os.path.join(DATA_DIR, "constituents-financials.csv")


def _download_sp500_list():
    if os.path.exists(SP500_LIST_PATH):
        return

    f = urllib2.urlopen(SP500_LIST_URL)
    print "Downloading ...", SP500_LIST_URL
    with open(SP500_LIST_PATH, 'w') as fin:
        print >> fin, f.read()
    return


def _load_symbols():
    _download_sp500_list()
    df_sp500 = pd.read_csv(SP500_LIST_PATH)
    df_sp500.sort('Market Cap', ascending=False, inplace=True)
    stock_symbols = df_sp500['Symbol'].unique().tolist()
    print "Loaded %d stock symbols" % len(stock_symbols)
    return stock_symbols


def fetch_prices(symbol, out_name):
    """
    Fetch daily stock prices for stock `symbol`, since 1980-01-01.

    Args:
        symbol (str): a stock abbr. symbol, like "GOOG" or "AAPL".

    Returns: a bool, whether the fetch is succeeded.
    """
    # Format today's date to match Google's finance history api.
    now_datetime = datetime.now().strftime("%b+%d,+%Y")

    BASE_URL = "https://finance.google.com/finance/historical?output=csv&q={0}&startdate=Jan+1%2C+1980&enddate={1}"
    symbol_url = BASE_URL.format(
        urllib2.quote(symbol),
        urllib2.quote(now_datetime, '+')
    )
    print "Fetching {} ...".format(symbol)
    print symbol_url

    try:
        f = urllib2.urlopen(symbol_url)
        with open(out_name, 'w') as fin:
            print >> fin, f.read()
    except urllib2.HTTPError:
        print "Failed when fetching {}".format(symbol)
        return False

    data = pd.read_csv(out_name)
    if data.empty:
        print "Remove {} because the data set is empty.".format(out_name)
        os.remove(out_name)
    else:
        dates = data.iloc[:,0].tolist()
        print "# Fetched rows: %d [%s to %s]" % (data.shape[0], dates[-1], dates[0])

    # Take a rest
    sleep_time = random.randint(*RANDOM_SLEEP_TIMES)
    print "Sleeping ... %ds" % sleep_time
    time.sleep(sleep_time)
    return True


@click.command(help="Fetch stock prices data")
@click.option('--continued', is_flag=True)
def main(continued):
    random.seed(time.time())
    num_failure = 0

    # This is S&P 500 index
    #fetch_prices('INDEXSP%3A.INX')

    symbols = _load_symbols()
    for idx, sym in enumerate(symbols):
        out_name = os.path.join(DATA_DIR, sym + ".csv")
        if continued and os.path.exists(out_name):
            print "Fetched", sym
            continue

        succeeded = fetch_prices(sym, out_name)
        num_failure += int(not succeeded)

        if idx % 10 == 0:
            print "# Failures so far [%d/%d]: %d" % (idx + 1, len(symbols), num_failure)


if __name__ == "__main__":
    main()
