from __future__ import print_function

import os
import time

import numpy as np
import pandas as pd
from stock.utils import get_path


class StockData(object):
    def __init__(self,
                 stock_sym,
                 input_size=1,
                 num_steps=300,
                 test_ratio=0.1,
                 normalized=True,
                 price_column='Close'):

        np.random.seed(int(time.time()))

        self.symbol = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.normalized = normalized

        # Read csv file
        raw_df = pd.read_csv(get_path(f"data/{stock_sym}.csv"))
        self.raw_seq = np.array(raw_df[price_column].tolist())
        self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)

        print("Loaded:", self.info())

    def info(self):
        return f"StockData [{self.symbol}] train:{self.train_X.shape} test:{self.train_y.shape}"

    def _prepare_data(self, seq):
        # split into items of input_size
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
               for i in range(len(seq) // self.input_size)]

        if self.normalized:
            # Normalized by dividing the last element of the last input window.
            seq = [seq[0] / seq[0][0] - 1.0] + [
                curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]
            seq = np.array(seq) * 100.  # percentage of price changes!
            assert not any(map(pd.isnull, seq.flatten()))

        # split into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = list(range(num_batches))
        np.random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y


class StockDataSet(object):
    def __init__(self,
                 stock_syms,
                 input_size=1,
                 num_steps=300,
                 test_ratio=0.1,
                 normalized=True,
                 price_column='Close'):
        assert len(stock_syms) > 0

        self.stock_syms = stock_syms
        self.datasets = [
            StockData(
                sym,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio,
                normalized=normalized,
                price_column=price_column,
            ) for sym in stock_syms
        ]

        self._prepare_merged_test_data()

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]

    def generate_one_epoch(self, batch_size):
        for d_idx, d in enumerate(self.datasets):
            for batch_X, batch_y in d.generate_one_epoch(batch_size):
                yield d_idx, batch_X, batch_y

    def _prepare_merged_test_data(self):
        """Merged test data of different stocks.
        """
        self.test_X = []
        self.test_y = []
        self.test_symbols = []  # str
        self.test_labels = []  # int

        for d_idx, d in enumerate(self.datasets):
            self.test_X += list(d.test_X)
            self.test_y += list(d.test_y)
            self.test_symbols += [[d.symbol]] * len(d.test_X)
            self.test_labels += [[d_idx]] * len(d.test_X)

        self.test_X = np.array(self.test_X)
        self.test_y = np.array(self.test_y)
        self.test_symbols = np.array(self.test_symbols)
        self.test_labels = np.array(self.test_labels)

        print("stock symbols =", self.stock_syms)
        print("len(test_X) =", len(self.test_X))
        print("len(test_y) =", len(self.test_y))
        print("len(test_symbols) =", len(self.test_symbols))
        print("len(test_labels) =", len(self.test_labels))


def load_dataset(input_size, num_steps, k=None, target_symbol=None, test_ratio=0.05):
    if target_symbol is not None:
        return StockDataSet(
            [target_symbol],
            input_size=input_size,
            num_steps=num_steps,
            test_ratio=test_ratio,
        )

    # Load metadata of s & p 500 stocks
    info = pd.read_csv("data/constituents-financials.csv")
    info = info.rename(columns={col: col.lower().replace(' ', '_') for col in info.columns})
    info['file_exists'] = info['symbol'].map(lambda x: os.path.exists("data/{}.csv".format(x)))
    print(info['file_exists'].value_counts().to_dict())

    info = info[info['file_exists'] == True].reset_index(drop=True)
    info = info.sort('market_cap', ascending=False).reset_index(drop=True)

    if k is not None:
        info = info.head(k)

    print("Head of S&P 500 info:\n", info.head())

    # Generate embedding meta file
    info[['symbol', 'sector']].to_csv(os.path.join("logs/metadata.tsv"), sep='\t', index=False)
    stock_symbols = info['symbol'].tolist()
    return StockDataSet(
        stock_symbols,
        input_size=input_size,
        num_steps=num_steps,
        test_ratio=test_ratio,
    )
