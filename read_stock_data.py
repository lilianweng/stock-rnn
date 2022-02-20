# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 21:59:55 2018

@author: lankuohsing
"""
# In[]
import numpy as np
import os
import pandas as pd
import random
import time
# In[]
# Read csv file
stock_sym='_SP500'
input_size=1
num_steps=30
test_ratio=0.1
normalized=True
close_price_only=True

# In[]
def _prepare_data(seq):
    # split into items of input_size
    #注：input_size代表取出连续的值的个数，可看成将它们作为一组特征输入
    seq = [np.array(seq[i * input_size: (i + 1) * input_size]) for i in range(len(seq) // input_size)]
    if normalized:#all the values are divided by the last unknown price
        seq = [seq[0] / seq[0][0] - 1.0] + [
              curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

    # split into groups of num_steps
    X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])
    y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])

    train_size = int(len(X) * (1.0 - test_ratio))
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    return train_X, train_y, test_X, test_y
# In[]
raw_df = pd.read_csv(os.path.join("data", "%s.csv" % stock_sym))
# In[]
# Merge into one sequence
if close_price_only:
    raw_seq = raw_df['Close'].tolist()
else:
    raw_seq = [price for tup in raw_df[['Open', 'Close']].values for price in tup]
# In[]
raw_seq = np.array(raw_seq)
# In[]
train_X, train_y, test_X, test_y = _prepare_data(raw_seq)
