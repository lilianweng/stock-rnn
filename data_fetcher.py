"""
Fetch the daily stock prices from Google Finance for stocks in S & P 500.
@author: lilianweng
"""
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


def fetch_prices(symbol):
    """
    Fetch daily stock prices for stock `symbol`, since 1980-01-01.

    Args:
        symbol (str): a stock abbr. symbol, like "GOOG" or "AAPL".

    Returns: a bool, whether the fetch is succeeded.
    """
    out_name = os.path.join(DATA_DIR, symbol + ".csv")
    if os.path.exists(out_name):
        return True

    # Format today's date to match Google's finance history api.
    now_datetime = datetime.now().strftime("%b+%d,+%Y")

    BASE_URL = "https://www.google.com/finance/historical?output=csv&q={0}&startdate=Jan+1%2C+1980&enddate={1}"
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

    lines = open(out_name).readlines()
    last_date = lines[2].split(',')[0]
    first_date = lines[-2].split(',')[0]
    print "%d lines [%s to %s]" % (len(lines) - 1, first_date, last_date)

    # Take a rest
    sleep_time = random.randint(*RANDOM_SLEEP_TIMES)
    print "Sleeping ... %ds" % sleep_time
    time.sleep(sleep_time)
    return True


def main():
    random.seed(time.time())
    num_failure = 0

    # This is S&P 500 index
    #fetch_prices('INDEXSP%3A.INX')
    symbols = _load_symbols()
    for idx, sym in enumerate(symbols):
        succeeded = fetch_prices(sym)
        num_failure += int(not succeeded)

        if idx % 10 == 0:
            print "# Failures so far [%d/%d]: %d" % (idx + 1, len(symbols), num_failure)


if __name__ == "__main__":
    main()
